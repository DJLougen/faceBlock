import { describe, expect, test } from "bun:test";
import {
  MAX_BASE64_CHARS,
  MAX_IMAGE_BYTES,
  base64ToBytes,
  checkImageType,
  fetchImageBytes,
} from "../../extension/net.ts";

function bytesResponse(
  chunks: Uint8Array[],
  headers: Record<string, string> = {},
): Response {
  return {
    ok: true,
    status: 200,
    headers: { get: (k: string) => headers[k.toLowerCase()] ?? null },
    body: {
      getReader() {
        let i = 0;
        return {
          read: async () =>
            i < chunks.length
              ? { done: false, value: chunks[i++]! }
              : { done: true, value: undefined },
          cancel: async () => undefined,
        };
      },
    },
  } as unknown as Response;
}

function fetchOf(res: Response | (() => Promise<Response>)): typeof fetch {
  return (async () => (typeof res === "function" ? res() : res)) as unknown as typeof fetch;
}

describe("fetchImageBytes", () => {
  test("returns the streamed bytes within the cap", async () => {
    const res = bytesResponse([new Uint8Array([1, 2, 3]), new Uint8Array([4])], {
      "content-type": "image/jpeg",
    });
    const blob = await fetchImageBytes("https://x/y.jpg", "test image", {
      fetchImpl: fetchOf(res),
    });
    expect(blob.size).toBe(4);
  });

  test("rejects a declared content-length over the cap before reading the body", async () => {
    const res = bytesResponse([], {
      "content-type": "image/jpeg",
      "content-length": String(MAX_IMAGE_BYTES + 1),
    });
    await expect(
      fetchImageBytes("https://x/big.jpg", "big image", { fetchImpl: fetchOf(res) }),
    ).rejects.toThrow("12 MB");
  });

  test("rejects a streamed body that grows past the cap", async () => {
    const half = new Uint8Array(MAX_IMAGE_BYTES / 2 + 1);
    const res = bytesResponse([half, half], { "content-type": "image/jpeg" });
    await expect(
      fetchImageBytes("https://x/fat.jpg", "fat image", { fetchImpl: fetchOf(res) }),
    ).rejects.toThrow("12 MB limit (streamed)");
  });

  // Real-timer exception: the abort deadline is the unit under test, so the
  // stalled stream is released by the genuine timeout.
  test("a stalled body settles at the deadline instead of hanging", async () => {
    const res = {
      ok: true,
      status: 200,
      headers: { get: () => "image/jpeg" },
      body: {
        getReader() {
          return {
            read: () => new Promise(() => {}), // never resolves
            cancel: async () => undefined,
          };
        },
      },
    } as unknown as Response;
    await expect(
      fetchImageBytes("https://x/stalled.jpg", "stalled image", {
        fetchImpl: fetchOf(res),
        timeoutMs: 30,
      }),
    ).rejects.toThrow();
  });

  test("a stalled fetch head settles at the deadline", async () => {
    const hanging = (() => new Promise<Response>(() => {})) as unknown as typeof fetch;
    await expect(
      fetchImageBytes("https://x/hang.jpg", "hung image", {
        fetchImpl: hanging,
        timeoutMs: 30,
      }),
    ).rejects.toThrow();
  });

  test("HTTP errors and SVG are rejected", async () => {
    const notFound = { ok: false, status: 404, headers: { get: () => null } } as unknown as Response;
    await expect(
      fetchImageBytes("https://x/404.jpg", "missing", { fetchImpl: fetchOf(notFound) }),
    ).rejects.toThrow("404");
    const svg = bytesResponse([new Uint8Array([60])], { "content-type": "image/svg+xml" });
    await expect(
      fetchImageBytes("https://x/icon.svg", "svg", { fetchImpl: fetchOf(svg) }),
    ).rejects.toThrow("SVG");
  });
});

describe("checkImageType", () => {
  test("refuses SVG and non-image types, allows raster types", () => {
    expect(() => checkImageType("image/svg+xml", "x")).toThrow("SVG");
    expect(() => checkImageType("text/html", "x")).toThrow("unsupported media type");
    expect(() => checkImageType("image/png", "x")).not.toThrow();
    expect(() => checkImageType("", "x")).not.toThrow();
  });
});

describe("base64ToBytes", () => {
  test("decodes a valid payload", () => {
    const out = base64ToBytes(btoa("hello"));
    expect(out).not.toBeNull();
    expect(new TextDecoder().decode(out!)).toBe("hello");
  });

  test("caps the string BEFORE decoding", () => {
    const oversized = "A".repeat(MAX_BASE64_CHARS + 4);
    expect(base64ToBytes(oversized)).toBeNull();
  });

  test("rejects empty, non-string, and malformed input", () => {
    expect(base64ToBytes("")).toBeNull();
    expect(base64ToBytes(undefined)).toBeNull();
    expect(base64ToBytes(42)).toBeNull();
    expect(base64ToBytes("!!!not-base64!!!")).toBeNull();
  });
});
