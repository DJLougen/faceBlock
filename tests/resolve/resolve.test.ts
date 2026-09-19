import { describe, expect, test } from "bun:test";
import { resolveCandidates } from "../../src/resolve/resolve.ts";

/** Minimal Response stand-in; resolve.ts only reads ok/status/json. */
function jsonResponse(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    json: async () => body,
  } as unknown as Response;
}

function errorResponse(status = 500): Response {
  return { ok: false, status, json: async () => ({}) } as unknown as Response;
}

/** Routes a request URL to a canned Wikimedia payload. */
function router(routes: (url: string) => Response | null): typeof fetch {
  return (async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    const res = routes(url);
    if (!res) throw new Error(`unexpected request: ${url}`);
    return res;
  }) as unknown as typeof fetch;
}

const WIKI_LEAD = {
  query: {
    pages: {
      "1": {
        title: "Donald Trump",
        original: { source: "https://upload.wikimedia.org/a/original_portrait.jpg" },
        thumbnail: { source: "https://upload.wikimedia.org/a/thumb_portrait.jpg" },
        images: [{ title: "File:Donald Trump official portrait.jpg" }],
      },
    },
  },
};

const COMMONS_SEARCH = {
  query: {
    pages: {
      "10": {
        title: "File:Donald Trump official portrait.jpg",
        imageinfo: [
          {
            url: "https://upload.wikimedia.org/b/Donald_Trump_official_portrait.jpg",
            thumburl: "https://upload.wikimedia.org/b/thumb_640.jpg",
            extmetadata: {
              ImageDescription: { value: "President <b>Donald Trump</b> official portrait" },
              Categories: { value: "Presidents|Portraits" },
            },
          },
        ],
      },
      "11": {
        title: "File:Coat of Arms of Trump International Golf Club.svg",
        imageinfo: [
          {
            url: "https://upload.wikimedia.org/b/Coat_of_Arms.svg",
            extmetadata: { Categories: { value: "Coats of arms" } },
          },
        ],
      },
      "12": {
        title: "File:Donald Trump signature.png",
        imageinfo: [
          {
            url: "https://upload.wikimedia.org/b/Donald_Trump_signature.png",
            extmetadata: { ImageDescription: { value: "Signature of Donald Trump" } },
          },
        ],
      },
    },
  },
};

const WIKIDATA_SEARCH = { search: [{ id: "Q22686", label: "Donald Trump" }] };
const WIKIDATA_ENTITY = {
  entities: {
    Q22686: {
      claims: {
        P18: [{ mainsnak: { datavalue: { value: "Official_Presidential_Portrait.jpg" } } }],
        P373: [{ mainsnak: { datavalue: { value: "Donald Trump" } } }],
      },
    },
  },
};

const COMMONS_CATEGORY = {
  query: {
    pages: {
      "20": {
        title: "File:Donald Trump RNC July 2016.jpg",
        imageinfo: [
          {
            url: "https://upload.wikimedia.org/c/Donald_Trump_RNC_July_2016.jpg",
            thumburl: "https://upload.wikimedia.org/c/thumb_rnc.jpg",
            extmetadata: { ImageDescription: { value: "Donald Trump at the RNC" } },
          },
        ],
      },
    },
  },
};

function fullRoutes(url: string): Response | null {
  if (url.includes("en.wikipedia.org")) return jsonResponse(WIKI_LEAD);
  if (url.includes("www.wikidata.org") && url.includes("wbsearchentities")) {
    return jsonResponse(WIKIDATA_SEARCH);
  }
  if (url.includes("www.wikidata.org") && url.includes("wbgetentities")) {
    return jsonResponse(WIKIDATA_ENTITY);
  }
  if (url.includes("commons.wikimedia.org")) {
    if (url.includes("incategory")) return jsonResponse(COMMONS_CATEGORY);
    return jsonResponse(COMMONS_SEARCH);
  }
  return null;
}

describe("resolveCandidates", () => {
  test("returns scored, sorted, deduped candidates from all sources", async () => {
    const result = await resolveCandidates("Donald Trump", { fetchImpl: router(fullRoutes) });

    expect(result.name).toBe("Donald Trump");
    expect(result.candidates.length).toBeGreaterThan(0);

    const scores = result.candidates.map((c) => c.score);
    const sorted = [...scores].sort((a, b) => b - a);
    expect(scores).toEqual(sorted);

    const urls = result.candidates.map((c) => c.url);
    expect(new Set(urls).size).toBe(urls.length);
  });

  test("excludes non-raster and non-https URLs", async () => {
    const result = await resolveCandidates("Donald Trump", { fetchImpl: router(fullRoutes) });
    for (const candidate of result.candidates) {
      expect(candidate.url.startsWith("https://")).toBe(true);
      expect(/\.(jpg|jpeg|png|webp)$/i.test(candidate.url)).toBe(true);
    }
  });

  test("ranks the real portrait above the signature", async () => {
    const result = await resolveCandidates("Donald Trump", { fetchImpl: router(fullRoutes) });
    const portrait = result.candidates.find((c) => c.filename.includes("official portrait"));
    const signature = result.candidates.find((c) => c.filename.includes("signature"));
    expect(portrait).toBeDefined();
    if (signature) expect(portrait!.score).toBeGreaterThan(signature.score);
  });

  test("resolves Wikidata P18 through Commons imageinfo", async () => {
    const result = await resolveCandidates("Donald Trump", { fetchImpl: router(fullRoutes) });
    expect(result.candidates.some((c) => c.source === "wikidata")).toBe(true);
  });

  test("one failing endpoint still yields the others' candidates", async () => {
    const partial = router((url) => {
      if (url.includes("en.wikipedia.org")) return errorResponse();
      return fullRoutes(url);
    });
    const result = await resolveCandidates("Donald Trump", {
      fetchImpl: partial,
      retryAttempts: 1,
    });
    expect(result.candidates.length).toBeGreaterThan(0);
  });

  test("all endpoints failing resolves to an empty list, not a rejection", async () => {
    const result = await resolveCandidates("Nobody Atall", {
      fetchImpl: router(() => errorResponse()),
      retryAttempts: 1,
    });
    expect(result.candidates).toEqual([]);
    expect(result.name).toBe("Nobody Atall");
  });

  test("a thrown fetch is tolerated like an HTTP error", async () => {
    const throwing = router(() => {
      throw new Error("network down");
    });
    const result = await resolveCandidates("Donald Trump", {
      fetchImpl: throwing,
      retryAttempts: 1,
    });
    expect(result.candidates).toEqual([]);
  });

  test("honours the limit", async () => {
    const result = await resolveCandidates("Donald Trump", {
      fetchImpl: router(fullRoutes),
      limit: 1,
    });
    expect(result.candidates.length).toBe(1);
  });

  test("dedupes URLs that differ only by tracking query params", async () => {
    const duplicated = {
      query: {
        pages: {
          "1": {
            title: "File:Donald Trump official portrait.jpg",
            imageinfo: [
              {
                url: "https://upload.wikimedia.org/x/portrait.jpg?utm_source=a",
                extmetadata: {},
              },
            ],
          },
          "2": {
            title: "File:Donald Trump official portrait.jpg",
            imageinfo: [
              {
                url: "https://upload.wikimedia.org/x/portrait.jpg?utm_source=b",
                extmetadata: {},
              },
            ],
          },
        },
      },
    };
    const result = await resolveCandidates("Donald Trump", {
      fetchImpl: router((url) =>
        url.includes("commons.wikimedia.org") ? jsonResponse(duplicated) : jsonResponse({}),
      ),
    });
    expect(result.candidates.length).toBe(1);
  });

  test("returns nothing rather than unrelated faces when no candidate is relevant", async () => {
    // Real failure mode: a Commons search for a name with no public photos
    // returns unrelated files whose captions happen to contain those words.
    const noise = {
      query: {
        pages: {
          "30": {
            title: "File:In the riding school BHL16741348.jpg",
            imageinfo: [
              {
                url: "https://upload.wikimedia.org/n/riding_school.jpg",
                extmetadata: {
                  ImageDescription: { value: "Theo and Browne discuss the man behind the gun" },
                },
              },
            ],
          },
        },
      },
    };
    const result = await resolveCandidates("Theo Browne", {
      fetchImpl: router((url) =>
        url.includes("commons.wikimedia.org") ? jsonResponse(noise) : jsonResponse({}),
      ),
      retryAttempts: 1,
    });
    expect(result.candidates).toEqual([]);
  });

  test("keeps candidates that clear the floor", async () => {
    const result = await resolveCandidates("Donald Trump", { fetchImpl: router(fullRoutes) });
    expect(result.candidates.length).toBeGreaterThan(0);
    for (const c of result.candidates) expect(c.score).toBeGreaterThanOrEqual(0.4);
  });

  test("records the queries it issued", async () => {
    const result = await resolveCandidates("Donald Trump", { fetchImpl: router(fullRoutes) });
    expect(result.queries.length).toBeGreaterThan(0);
    expect(result.queries.every((q) => typeof q === "string")).toBe(true);
  });
});
