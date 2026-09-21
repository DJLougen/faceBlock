/**
 * Self-seed reference acquisition: turn a person's name into candidate photos.
 *
 * Plan §12 (browser-assisted discovery) and §14 (media privacy).
 *
 * EGRESS RULE: the person's NAME is the only thing that leaves the device —
 * the same string the user would type into a search box. Images, face crops,
 * embeddings, live page media, browsing history, and match results are never
 * transmitted. Downloads happen here; nothing is uploaded.
 *
 * Sources are keyless and CORS-enabled so no account or API key is needed:
 *   - Wikipedia  : the curated lead portrait plus in-article photos
 *   - Wikidata   : P18 (canonical image) and P373 (Commons category)
 *   - Commons    : bulk search, whose descriptive file names are the signal
 *
 * Every endpoint is best-effort. One failure contributes nothing; only an
 * all-endpoints failure yields an empty candidate list, never a rejection.
 */

import { scoreCandidate } from "./score.ts";
import type { CandidateImage, ResolveResult } from "./types.ts";

const WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php";
const WIKIDATA_API = "https://www.wikidata.org/w/api.php";
const COMMONS_API = "https://commons.wikimedia.org/w/api.php";

/** Preview/download width requested from the APIs. */
const THUMB_WIDTH = 640;

/** Candidates we keep when nothing scored above zero, so obscure names still resolve. */
const FALLBACK_KEEP = 5;

/**
 * Minimum context score for a candidate to be worth downloading.
 *
 * Below this the hit is incidental — a scattered token match inside a long
 * caption, a plaque, a signature, a look-alike. Returning those wastes
 * downloads and shows the user unrelated faces, so it is better to report
 * "no public reference photos found" than to pad the list with noise.
 */
const MIN_RELEVANT_SCORE = 0.4;

/** Raster formats we can actually decode and embed. */
const PHOTO_EXT: Record<string, true> = {
  jpg: true,
  jpeg: true,
  png: true,
  webp: true,
};

export interface ResolveOptions {
  aliases?: readonly string[];
  /** Maximum candidates returned. Default 48. */
  limit?: number;
  /**
   * Fetch implementation. Injected so tests never touch the network; defaults
   * to the ambient fetch inside a browser extension page or Bun.
   */
  fetchImpl?: typeof fetch;
  /** Attempts per request including the first. Default 3. Tests use 1. */
  retryAttempts?: number;
  /** Per-request ceiling in ms covering head AND body. Default 15s. */
  requestTimeoutMs?: number;
  /**
   * Overall resolution deadline in ms. Default 90s. Expiry is terminal for
   * new work but NOT an error: whatever candidates were already gathered are
   * returned. A caller abort via `signal` IS an error and rejects.
   */
  overallDeadlineMs?: number;
  signal?: AbortSignal;
}

/**
 * Terminal abort for resolution. `kind` distinguishes a caller-initiated
 * abort (rejects resolveCandidates) from the internal overall deadline
 * (returns whatever was gathered). Named AbortError so generic abort
 * handling recognises it.
 */
export class ResolveAbortError extends Error {
  readonly kind: "caller" | "deadline";
  constructor(kind: "caller" | "deadline") {
    super(
      kind === "caller"
        ? "faceBlock: resolve aborted by caller"
        : "faceBlock: resolve exceeded its overall deadline",
    );
    this.name = "AbortError";
    this.kind = kind;
  }
}

/** The abort reason a signal carries, normalised to ResolveAbortError. */
function abortReason(signal: AbortSignal): ResolveAbortError {
  return signal.reason instanceof ResolveAbortError
    ? signal.reason
    : new ResolveAbortError("caller");
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) throw abortReason(signal);
}

/** Rejects with the signal's reason the moment it fires, even if `work` hangs. */
function raceSignal<T>(work: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) {
    return Promise.reject(
      signal.reason instanceof Error ? signal.reason : new ResolveAbortError("caller"),
    );
  }
  return new Promise<T>((resolve, reject) => {
    // Propagate the raw reason: the overall signal carries ResolveAbortError
    // (terminal), a per-request signal carries a plain timeout Error
    // (retriable) — the caller's catch classifies it.
    const onAbort = () =>
      reject(signal.reason instanceof Error ? signal.reason : new ResolveAbortError("caller"));
    signal.addEventListener("abort", onAbort, { once: true });
    work.then(resolve, reject).finally(() => signal.removeEventListener("abort", onAbort));
  });
}


/* ---------- small helpers ---------- */

function stripHtml(text: string): string {
  return text
    .replace(/<[^>]*>/g, " ")
    .replace(/&[a-z]+;/gi, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** Path extension of a URL, lowercased, ignoring the query string. */
function extensionOf(url: string): string {
  let path: string;
  try {
    path = new URL(url).pathname;
  } catch {
    return "";
  }
  const dot = path.lastIndexOf(".");
  return dot === -1 ? "" : path.slice(dot + 1).toLowerCase();
}

/** Dedupe key: the URL without its query string, so API tracking params collapse. */
function dedupeKey(url: string): string {
  try {
    const u = new URL(url);
    return `${u.origin}${u.pathname}`;
  } catch {
    return url;
  }
}

function isUsablePhotoUrl(url: string): boolean {
  if (!url.startsWith("https://")) return false;
  return PHOTO_EXT[extensionOf(url)] === true;
}

/** `File:Donald_Trump.jpg` -> `Donald_Trump.jpg` */
function fileTitleToName(title: string): string {
  return title.replace(/^File:/i, "").trim();
}

interface RawCandidate {
  url: string;
  thumbUrl?: string;
  filename: string;
  pageTitle: string;
  caption: string;
  source: CandidateImage["source"];
}

/* ---------- endpoint queries ---------- */

function delay(ms: number): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  setTimeout(resolve, ms);
  return promise;
}

/**
 * GET JSON with bounded retry.
 *
 * Wikimedia rate-limits bursts (HTTP 429) and the resolver issues several
 * requests per enrollment. Without this retry a busy moment looks to the user
 * like "no photos exist for this person", which is a silent lie — so a 429 or
 * 5xx is retried, honouring `Retry-After` when the server sends it.
 */
async function getJson(
  fetchImpl: typeof fetch,
  url: string,
  signal: AbortSignal,
  attempts: number,
  requestTimeoutMs: number,
): Promise<unknown | null> {
  for (let attempt = 0; attempt < attempts; attempt++) {
    throwIfAborted(signal);
    // Per-request deadline covering head AND body. The request signal is
    // separate from the overall signal so a slow request is a retryable
    // failure, while an overall-deadline or caller abort is terminal.
    const req = new AbortController();
    const onAbort = () => req.abort(signal.reason);
    signal.addEventListener("abort", onAbort, { once: true });
    const timer = setTimeout(() => req.abort(new Error("request-timeout")), requestTimeoutMs);
    try {
      const res = await raceSignal(fetchImpl(url, {
        credentials: "omit",
        signal: req.signal,
      }), req.signal);
      if (res.ok) return (await raceSignal(res.json(), req.signal)) as unknown;

      const retryable = res.status === 429 || res.status >= 500;
      if (!retryable) return null;

      const retryAfter = Number(res.headers?.get?.("retry-after") ?? NaN);
      const backoffMs = Number.isFinite(retryAfter)
        ? Math.min(retryAfter * 1000, 2000)
        : 250 * 2 ** attempt;
      await raceSignal(delay(backoffMs), signal);
    } catch (e) {
      // Overall-deadline or caller abort is terminal — never retried, never
      // continued into the next source.
      if (signal.aborted) throw abortReason(signal);
      if (e instanceof ResolveAbortError) throw e;
      // A thrown fetch (offline, per-request timeout, DNS) is retried like a 5xx.
      await raceSignal(delay(250 * 2 ** attempt), signal);
    } finally {
      clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
    }
  }
  return null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : null;
}

function buildUrl(endpoint: string, params: Record<string, string>): string {
  // origin=* is what makes MediaWiki serve anonymous cross-origin requests.
  const search = new URLSearchParams({ format: "json", origin: "*", ...params });
  return `${endpoint}?${search.toString()}`;
}

/** Wikipedia: the curated lead image (`pageimages`) plus in-article file titles. */
async function fromWikipedia(
  fetchImpl: typeof fetch,
  name: string,
  signal: AbortSignal,
  retry: number,
  reqMs: number,
): Promise<RawCandidate[]> {
  const url = buildUrl(WIKIPEDIA_API, {
    action: "query",
    redirects: "1",
    titles: name,
    prop: "pageimages|images",
    piprop: "original|thumbnail",
    pithumbsize: String(THUMB_WIDTH),
    imlimit: "100",
  });

  const data = asRecord(await getJson(fetchImpl, url, signal, retry, reqMs));
  const query = asRecord(data?.["query"]);
  const pages = asRecord(query?.["pages"]);
  if (!pages) return [];

  const out: RawCandidate[] = [];

  for (const page of Object.values(pages)) {
    const p = asRecord(page);
    if (!p) continue;
    const title = typeof p["title"] === "string" ? p["title"] : name;

    const original = asRecord(p["original"]);
    const thumbnail = asRecord(p["thumbnail"]);
    const originalUrl = typeof original?.["source"] === "string" ? original["source"] : "";
    const thumbUrl = typeof thumbnail?.["source"] === "string" ? thumbnail["source"] : undefined;

    if (originalUrl) {
      out.push({
        url: originalUrl,
        thumbUrl,
        filename: fileTitleToName(title),
        pageTitle: title,
        caption: "Wikipedia lead image",
        source: "wikipedia",
      });
    }

    const images = p["images"];
    if (Array.isArray(images)) {
      for (const entry of images) {
        const t = asRecord(entry)?.["title"];
        if (typeof t === "string" && /^File:/i.test(t)) {
          // Resolved to a real URL through Commons below.
          out.push({
            url: "",
            filename: fileTitleToName(t),
            pageTitle: title,
            caption: "Wikipedia article image",
            source: "wikipedia",
          });
        }
      }
    }
  }

  return out;
}

/** Commons `imageinfo` for a batch of File: titles (max 50 per request). */
async function fromCommonsTitles(
  fetchImpl: typeof fetch,
  titles: string[],
  pageTitle: string,
  source: CandidateImage["source"],
  signal: AbortSignal,
  retry: number,
  reqMs: number,
): Promise<RawCandidate[]> {
  if (titles.length === 0) return [];
  const out: RawCandidate[] = [];

  for (let i = 0; i < titles.length; i += 50) {
    const batch = titles.slice(i, i + 50);
    const url = buildUrl(COMMONS_API, {
      action: "query",
      titles: batch.join("|"),
      prop: "imageinfo",
      iiprop: "url|extmetadata",
      iiurlwidth: String(THUMB_WIDTH),
    });

    const data = asRecord(await getJson(fetchImpl, url, signal, retry, reqMs));
    const pages = asRecord(asRecord(data?.["query"])?.["pages"]);
    if (!pages) continue;

    for (const page of Object.values(pages)) {
      const p = asRecord(page);
      if (!p) continue;
      const info = Array.isArray(p["imageinfo"]) ? asRecord(p["imageinfo"][0]) : null;
      if (!info) continue;

      const originalUrl = typeof info["url"] === "string" ? info["url"] : "";
      const thumbUrl = typeof info["thumburl"] === "string" ? info["thumburl"] : undefined;
      const extmetadata = asRecord(info["extmetadata"]);
      const description = extmetadata?.["ImageDescription"];
      const descText =
        typeof asRecord(description)?.["value"] === "string"
          ? stripHtml(String(asRecord(description)?.["value"]))
          : "";
      const categories = extmetadata?.["Categories"];
      const catText =
        typeof asRecord(categories)?.["value"] === "string"
          ? String(asRecord(categories)?.["value"]).replace(/\|/g, " ")
          : "";

      out.push({
        url: originalUrl,
        thumbUrl,
        filename: fileTitleToName(typeof p["title"] === "string" ? p["title"] : ""),
        pageTitle,
        caption: [descText, catText].filter(Boolean).join(" ").slice(0, 500),
        source,
      });
    }
  }

  return out;
}

/** Wikidata: P18 image + P373 Commons category for the best matching entity. */
async function fromWikidata(
  fetchImpl: typeof fetch,
  name: string,
  signal: AbortSignal,
  retry: number,
  reqMs: number,
): Promise<{ candidates: RawCandidate[]; category: string | null }> {
  const searchUrl = buildUrl(WIKIDATA_API, {
    action: "wbsearchentities",
    language: "en",
    type: "item",
    limit: "5",
    search: name,
  });

  const searchData = asRecord(await getJson(fetchImpl, searchUrl, signal, retry, reqMs));
  const hits = searchData?.["search"];
  if (!Array.isArray(hits) || hits.length === 0) return { candidates: [], category: null };

  const qid = asRecord(hits[0])?.["id"];
  if (typeof qid !== "string") return { candidates: [], category: null };

  const entityUrl = buildUrl(WIKIDATA_API, {
    action: "wbgetentities",
    props: "claims",
    ids: qid,
  });

  const entityData = asRecord(await getJson(fetchImpl, entityUrl, signal, retry, reqMs));
  const entities = asRecord(entityData?.["entities"]);
  const entity = asRecord(entities?.[qid]);
  const claims = asRecord(entity?.["claims"]);
  if (!claims) return { candidates: [], category: null };

  const valuesOf = (pid: string): string[] => {
    const list = claims[pid];
    if (!Array.isArray(list)) return [];
    const out: string[] = [];
    for (const stmt of list) {
      const value = asRecord(asRecord(asRecord(stmt)?.["mainsnak"])?.["datavalue"])?.["value"];
      if (typeof value === "string") out.push(value);
    }
    return out;
  };

  const titles = valuesOf("P18").map((f) => `File:${f}`);
  const category = valuesOf("P373")[0] ?? null;
  const candidates = await fromCommonsTitles(fetchImpl, titles, name, "wikidata", signal, retry, reqMs);

  return { candidates, category };
}

/** Commons search: the bulk source, with the descriptive file names. */
async function fromCommonsSearch(
  fetchImpl: typeof fetch,
  search: string,
  signal: AbortSignal,
  retry: number,
  reqMs: number,
): Promise<RawCandidate[]> {
  const url = buildUrl(COMMONS_API, {
    action: "query",
    generator: "search",
    gsrsearch: search,
    gsrnamespace: "6",
    gsrlimit: "50",
    prop: "imageinfo",
    iiprop: "url|extmetadata",
    iiurlwidth: String(THUMB_WIDTH),
  });

  const data = asRecord(await getJson(fetchImpl, url, signal, retry, reqMs));
  const pages = asRecord(asRecord(data?.["query"])?.["pages"]);
  if (!pages) return [];

  const out: RawCandidate[] = [];
  for (const page of Object.values(pages)) {
    const p = asRecord(page);
    if (!p) continue;
    const info = Array.isArray(p["imageinfo"]) ? asRecord(p["imageinfo"][0]) : null;
    if (!info) continue;

    const originalUrl = typeof info["url"] === "string" ? info["url"] : "";
    const thumbUrl = typeof info["thumburl"] === "string" ? info["thumburl"] : undefined;
    const extmetadata = asRecord(info["extmetadata"]);
    const descText =
      typeof asRecord(extmetadata?.["ImageDescription"])?.["value"] === "string"
        ? stripHtml(String(asRecord(extmetadata?.["ImageDescription"])?.["value"]))
        : "";
    const catText =
      typeof asRecord(extmetadata?.["Categories"])?.["value"] === "string"
        ? String(asRecord(extmetadata?.["Categories"])?.["value"]).replace(/\|/g, " ")
        : "";

    out.push({
      url: originalUrl,
      thumbUrl,
      filename: fileTitleToName(typeof p["title"] === "string" ? p["title"] : ""),
      pageTitle: "Wikimedia Commons",
      caption: [descText, catText].filter(Boolean).join(" ").slice(0, 500),
      source: "commons",
    });
  }

  return out;
}

/* ---------- public entrypoint ---------- */

/**
 * Find candidate photos of `name`.
 *
 * Candidates are scored by their surrounding text (see scoreCandidate), sorted
 * best-first, deduped, and capped at `opts.limit`.
 */
export async function resolveCandidates(
  name: string,
  opts: ResolveOptions = {},
): Promise<ResolveResult> {
  const fetchImpl = opts.fetchImpl ?? globalThis.fetch;
  const aliases = opts.aliases ?? [];
  const limit = opts.limit ?? 48;
  const retry = opts.retryAttempts ?? 3;
  const reqMs = opts.requestTimeoutMs ?? 15_000;

  // One overall signal per resolution: the caller's abort and the internal
  // deadline funnel into it. Abort is terminal — no retries, no further
  // sources — but a deadline expiry returns what was already gathered while
  // a caller abort rejects.
  const overall = new AbortController();
  const onCallerAbort = () => overall.abort(new ResolveAbortError("caller"));
  if (opts.signal) {
    if (opts.signal.aborted) throw new ResolveAbortError("caller");
    opts.signal.addEventListener("abort", onCallerAbort, { once: true });
  }
  let timedOut = false;
  const deadlineTimer = setTimeout(() => {
    timedOut = true;
    overall.abort(new ResolveAbortError("deadline"));
  }, opts.overallDeadlineMs ?? 90_000);
  const signal = overall.signal;

  const queries: string[] = [];

  queries.push(buildUrl(WIKIPEDIA_API, { action: "query", titles: name }));
  queries.push(buildUrl(WIKIDATA_API, { action: "wbsearchentities", search: name }));
  queries.push(buildUrl(COMMONS_API, { action: "query", gsrsearch: name }));

  const raws: RawCandidate[] = [];
  let wikidataCategory: string | null = null;
  try {
    // Deliberately SEQUENTIAL, not Promise.all: firing these concurrently gets
    // the client rate-limited (HTTP 429) and the swallowed failures then look
    // like "this person has no photos". Enrollment is a one-off user action, so
    // paying a few hundred milliseconds of latency for reliability is correct.
    const wiki = await fromWikipedia(fetchImpl, name, signal, retry, reqMs);
    raws.push(...wiki);
    const wikidata = await fromWikidata(fetchImpl, name, signal, retry, reqMs);
    raws.push(...wikidata.candidates);
    wikidataCategory = wikidata.category;
    const commons = await fromCommonsSearch(fetchImpl, name, signal, retry, reqMs);
    raws.push(...commons);

    // The Wikipedia `images` list carries titles but no URLs; resolve them here,
    // and use the Wikidata Commons category for a second, tighter bulk search.
    const unresolved = raws.filter((r) => r.url === "").map((r) => `File:${r.filename}`);
    if (unresolved.length > 0) {
      const resolved = await fromCommonsTitles(
        fetchImpl,
        unresolved,
        name,
        "wikipedia",
        signal,
        retry,
        reqMs,
      );
      raws.push(...resolved);
    }

    if (wikidataCategory) {
      queries.push(buildUrl(COMMONS_API, { action: "query", gsrsearch: `incategory:${wikidataCategory}` }));
      const inCategory = await fromCommonsSearch(
        fetchImpl,
        `incategory:"${wikidataCategory}"`,
        signal,
        retry,
        reqMs,
      );
      raws.push(...inCategory);
    }
  } catch (e) {
    if (e instanceof ResolveAbortError) {
      // Caller abort is terminal and propagates; the overall deadline keeps
      // whatever was gathered before it fired.
      if (e.kind === "caller") throw e;
    } else {
      throw e;
    }
  } finally {
    clearTimeout(deadlineTimer);
    if (opts.signal) opts.signal.removeEventListener("abort", onCallerAbort);
  }

  const seen = new Set<string>();
  const kept: CandidateImage[] = [];

  for (const raw of raws) {
    if (!isUsablePhotoUrl(raw.url)) continue;
    const key = dedupeKey(raw.url);
    if (seen.has(key)) continue;
    seen.add(key);
    kept.push({
      ...raw,
      score: scoreCandidate(name, aliases, {
        filename: raw.filename,
        pageTitle: raw.pageTitle,
        caption: raw.caption,
      }),
    });
  }

  kept.sort((a, b) => b.score - a.score);

  // Honest emptiness: when nothing clears the relevance floor, report nothing
  // found rather than handing the user unrelated faces to sort through.
  const candidates = kept.filter((c) => c.score >= MIN_RELEVANT_SCORE).slice(0, limit);

  return { name, candidates, queries, timedOut };
}
