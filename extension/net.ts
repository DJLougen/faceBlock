/**
 * Bounded network + byte helpers for the offscreen document. Pure of DOM —
 * fetch is injectable so tests never touch the network.
 *
 * Every download is credential-less, time-bounded, and byte-capped. The cap
 * is enforced three ways: the declared content-length, the streamed body
 * total, and (for base64 payloads) the string length BEFORE decoding.
 */

/** Streamed download cap shared by every image path. */
export const MAX_IMAGE_BYTES = 12 * 1024 * 1024; // 12 MiB

/**
 * Maximum base64 characters accepted before decoding. base64 inflates by
 * 4/3, so the character cap is derived from the byte cap — a longer string
 * would decode to more than MAX_IMAGE_BYTES and is rejected up front.
 */
export const MAX_BASE64_CHARS = Math.ceil(MAX_IMAGE_BYTES / 3) * 4;

export function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export function checkImageType(type: string, what: string): void {
  const t = type.split(";")[0]!.trim().toLowerCase();
  if (t === "image/svg+xml") {
    throw new Error(
      `faceBlock: ${what} is SVG — refused because SVG can pull external resources`,
    );
  }
  if (t && t !== "application/octet-stream" && !t.startsWith("image/")) {
    throw new Error(`faceBlock: ${what} has unsupported media type "${t}"`);
  }
}

/** Rejects when `signal` fires; used to bound body reads that ignore it. */
function abortable<T>(work: Promise<T>, signal: AbortSignal, what: string): Promise<T> {
  if (signal.aborted) {
    return Promise.reject(new Error(`faceBlock: ${what} aborted`));
  }
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(new Error(`faceBlock: ${what} aborted`));
    signal.addEventListener("abort", onAbort, { once: true });
    work.then(resolve, reject).finally(() => signal.removeEventListener("abort", onAbort));
  });
}

/**
 * Fetch image bytes with no credentials, an abort timeout covering BOTH the
 * response head and the streamed body, and a streamed byte cap. The body
 * read races the signal explicitly so a stalled stream settles even when
 * the underlying fetch does not propagate the abort promptly.
 */
export async function fetchImageBytes(
  url: string,
  what: string,
  opts: { fetchImpl?: typeof fetch; timeoutMs?: number } = {},
): Promise<Blob> {
  const fetchImpl = opts.fetchImpl ?? globalThis.fetch;
  const timeoutMs = opts.timeoutMs ?? 15_000;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    let res: Response;
    try {
      res = await abortable(
        fetchImpl(url, {
          credentials: "omit",
          redirect: "follow",
          signal: ctrl.signal,
        }),
        ctrl.signal,
        `fetch of ${what}`,
      );
    } catch (e) {
      throw new Error(
        `faceBlock: fetch of ${what} failed — ${ctrl.signal.aborted ? "timed out" : errText(e)}`,
      );
    }
    if (!res.ok) {
      throw new Error(`faceBlock: fetch of ${what} returned HTTP ${res.status}`);
    }
    checkImageType(res.headers.get("content-type") ?? "", what);
    const declared = Number(res.headers.get("content-length") ?? 0);
    if (declared > MAX_IMAGE_BYTES) {
      throw new Error(
        `faceBlock: ${what} is ${(declared / 1048576).toFixed(1)} MB — over the 12 MB limit`,
      );
    }
    const type = (res.headers.get("content-type") ?? "").split(";")[0]!.trim();
    if (!res.body) {
      const blob = await abortable(res.blob(), ctrl.signal, `fetch of ${what}`);
      if (blob.size > MAX_IMAGE_BYTES) {
        throw new Error(`faceBlock: ${what} exceeds the 12 MB limit`);
      }
      return blob;
    }
    const reader = res.body.getReader();
    const chunks: Uint8Array<ArrayBuffer>[] = [];
    let total = 0;
    try {
      for (;;) {
        const { done, value } = await abortable(reader.read(), ctrl.signal, `fetch of ${what}`);
        if (done) break;
        total += value.byteLength;
        if (total > MAX_IMAGE_BYTES) {
          throw new Error(`faceBlock: ${what} exceeds the 12 MB limit (streamed)`);
        }
        chunks.push(new Uint8Array(value));
      }
    } catch (e) {
      // Release the underlying stream on abort/timeout/overflow so the
      // download does not keep draining in the background.
      await reader.cancel().catch(() => undefined);
      throw e;
    }
    return new Blob(chunks, { type: type || "application/octet-stream" });
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Decode a base64 payload into bytes, capped BEFORE decoding. Returns null
 * for empty, oversized, or malformed input — callers turn that into their
 * own error message.
 */
export function base64ToBytes(b64: unknown): Uint8Array<ArrayBuffer> | null {
  if (typeof b64 !== "string" || b64.length === 0 || b64.length > MAX_BASE64_CHARS) {
    return null;
  }
  try {
    const binary = atob(b64);
    const out = new Uint8Array(new ArrayBuffer(binary.length));
    for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
    return out;
  } catch {
    return null;
  }
}
