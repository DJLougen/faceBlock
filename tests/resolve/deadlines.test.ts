import { describe, expect, test } from "bun:test";
import { ResolveAbortError, resolveCandidates } from "../../src/resolve/resolve.ts";

function jsonResponse(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    json: async () => body,
  } as unknown as Response;
}

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
        images: [],
      },
    },
  },
};

// Real-timer exceptions throughout: the request/overall deadlines are the
// units under test, so they are exercised with small genuine timeouts.

describe("resolveCandidates deadlines", () => {
  test("a stalled request head is a retriable failure, not a hang", async () => {
    const hanging = (() => new Promise<Response>(() => {})) as unknown as typeof fetch;
    const result = await resolveCandidates("Donald Trump", {
      fetchImpl: hanging,
      retryAttempts: 1,
      requestTimeoutMs: 30,
      overallDeadlineMs: 5_000,
    });
    expect(result.candidates).toEqual([]);
  });

  test("a stalled response body is bounded by the request deadline", async () => {
    const stalledBody = router(() => {
      return {
        ok: true,
        status: 200,
        json: () => new Promise(() => {}), // body never arrives
      } as unknown as Response;
    });
    const result = await resolveCandidates("Donald Trump", {
      fetchImpl: stalledBody,
      retryAttempts: 1,
      requestTimeoutMs: 30,
      overallDeadlineMs: 5_000,
    });
    expect(result.candidates).toEqual([]);
  });

  test("the overall deadline is terminal: later sources are never requested", async () => {
    const seen: string[] = [];
    const mixed = (async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      seen.push(url);
      if (url.includes("en.wikipedia.org")) {
        return new Promise<Response>(() => {}); // stalls forever
      }
      return jsonResponse({});
    }) as unknown as typeof fetch;
    const result = await resolveCandidates("Donald Trump", {
      fetchImpl: mixed,
      retryAttempts: 3,
      requestTimeoutMs: 40,
      overallDeadlineMs: 100,
    });
    // Wikipedia stalled; the deadline fired before the other sources ran.
    expect(seen.every((u) => !u.includes("commons.wikimedia.org"))).toBe(true);
    expect(result.candidates).toEqual([]);
    // Deadline honesty: the result must say it timed out, not claim a
    // definitive "no photos exist".
    expect(result.timedOut).toBe(true);
  });

  test("caller abort is terminal and rejects — no retry, no next source", async () => {
    const ctrl = new AbortController();
    const seen: string[] = [];
    const counting = (async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      seen.push(url);
      if (url.includes("en.wikipedia.org")) {
        return new Promise<Response>(() => {});
      }
      return jsonResponse({});
    }) as unknown as typeof fetch;
    const pending = resolveCandidates("Donald Trump", {
      fetchImpl: counting,
      signal: ctrl.signal,
      retryAttempts: 3,
      requestTimeoutMs: 5_000,
      overallDeadlineMs: 60_000,
    });
    ctrl.abort();
    await expect(pending).rejects.toBeInstanceOf(ResolveAbortError);
    expect(seen.length).toBe(1);
  });

  test("an already-aborted signal rejects before any request", async () => {
    const ctrl = new AbortController();
    ctrl.abort();
    let calls = 0;
    const counting = (async () => {
      calls += 1;
      return jsonResponse({});
    }) as unknown as typeof fetch;
    await expect(
      resolveCandidates("Donald Trump", { fetchImpl: counting, signal: ctrl.signal }),
    ).rejects.toBeInstanceOf(ResolveAbortError);
    expect(calls).toBe(0);
  });

  test("normal failures still resolve to an empty list, not a rejection", async () => {
    const failing = router(() => ({ ok: false, status: 500, json: async () => ({}) }) as Response);
    const result = await resolveCandidates("Nobody Atall", {
      fetchImpl: failing,
      retryAttempts: 1,
      requestTimeoutMs: 1_000,
      overallDeadlineMs: 10_000,
    });
    expect(result.candidates).toEqual([]);
    // A clean (if empty) resolve did not hit the deadline.
    expect(result.timedOut).toBe(false);
  });

  test("a healthy resolve still returns candidates under the deadlines", async () => {
    const result = await resolveCandidates("Donald Trump", {
      fetchImpl: router((url) =>
        url.includes("en.wikipedia.org") ? jsonResponse(WIKI_LEAD) : jsonResponse({}),
      ),
      retryAttempts: 1,
      requestTimeoutMs: 2_000,
      overallDeadlineMs: 10_000,
    });
    expect(result.candidates.length).toBeGreaterThan(0);
    expect(result.timedOut).toBe(false);
  });
});
