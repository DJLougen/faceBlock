/**
 * FaceBlock service worker.
 *
 * Owns the chrome.storage.local blocklist (`faceblockState`), serializes every
 * storage mutation through one write chain, brokers ANALYZE/RESOLVE_PREVIEW/
 * CONFIRM_ENROLL work to a singleton offscreen inference document, and
 * broadcasts revision changes to content scripts. Images, names, and
 * embeddings never leave the extension.
 *
 * Message contract (see protocol.ts):
 *   UI      -> {target:'background', type:'GET_STATE'|'BLOCK_NAME'|'RESOLVE_PREVIEW'|'CONFIRM_ENROLL'|'REMOVE'|'SET_ENABLED'}
 *   Content -> {target:'background', type:'PROCESS_IMAGE', url}
 *   Content -> {target:'background', type:'ANALYZE_FRAME', jpegBase64}
 *   Worker  -> {target:'offscreen',  type:'ANALYZE'|'ANALYZE_FRAME'|'RESOLVE_PREVIEW'|'CONFIRM_ENROLL', name?, url?, jpegBase64?, identities?, faces?, identityId?}
 *   Worker  -> {target:'content',   type:'STATE_CHANGED', revision, enabled}
 *
 * Enrollment never persists on preview: RESOLVE_PREVIEW (and its legacy
 * alias BLOCK_NAME) only gathers faces; CONFIRM_ENROLL is the sole write
 * path, and an explicit identityId must name an existing saved identity.
 */

import type { BlockList, EnrollPreview, FrameResult, ImageResult, SavedIdentity } from "./protocol.ts";
import { mergeConfirmedIdentity, selectPreviewIdentityId } from "./enroll.ts";



/* ---- minimal chrome typings (extension/ is outside the tsconfig project) ---- */

interface ChromeSender {
  tab?: { id?: number; url?: string };
  url?: string;
  id?: string;
}

declare const chrome: {
  storage: {
    local: {
      get(keys: string): Promise<Record<string, unknown>>;
      set(items: Record<string, unknown>): Promise<void>;
    };
  };
  offscreen: {
    hasDocument(): Promise<boolean>;
    createDocument(options: {
      url: string;
      reasons: string[];
      justification: string;
    }): Promise<void>;
    closeDocument(): Promise<void>;
  };
  runtime: {
    id: string;
    onMessage: {
      addListener(
        callback: (
          message: unknown,
          sender: ChromeSender,
          sendResponse: (response: unknown) => void,
        ) => boolean | undefined,
      ): void;
    };
    sendMessage(message: unknown): Promise<unknown>;
    lastError?: { message?: string };
  };
  tabs: {
    query(queryInfo: Record<string, never>): Promise<Array<{ id?: number }>>;
    sendMessage(tabId: number, message: unknown): Promise<unknown>;
  };
};

/* ---- constants ---- */

const STORAGE_KEY = "faceblockState";
const OFFSCREEN_URL = "offscreen.html";

/** Concurrent ANALYZE jobs forwarded to the offscreen document. */
const MAX_INFLIGHT_ANALYZE = 2;
/** Pending ANALYZE waiters; beyond this the content script is told we're busy. */
const MAX_QUEUED_ANALYZE = 150;
/** Video frames in flight; a second ANALYZE_FRAME is dropped, never queued. */
const MAX_INFLIGHT_FRAME = 1;
/** Bounded per-revision image-result cache (LRU eviction). */
const RESULT_CACHE_LIMIT = 500;

const EMPTY_RESULT: ImageResult = { width: 0, height: 0, faceCount: 0, regions: [] };

/* ---- state: memory cache is authoritative; storage is the durable copy ---- */

let cached: BlockList | null = null;
let writeChain: Promise<unknown> = Promise.resolve();

async function loadState(): Promise<BlockList> {
  if (cached) return cached;
  const raw = await chrome.storage.local.get(STORAGE_KEY);
  const existing = raw[STORAGE_KEY] as Partial<BlockList> | undefined;
  if (existing && Array.isArray(existing.identities) && typeof existing.enabled === "boolean") {
    cached = {
      identities: existing.identities as SavedIdentity[],
      enabled: existing.enabled,
      revision: typeof existing.revision === "number" ? existing.revision : 0,
    };
    return cached;
  }
  // First run (or corrupt payload): default to enabled with an empty blocklist.
  cached = { identities: [], enabled: true, revision: 0 };
  await chrome.storage.local.set({ [STORAGE_KEY]: cached });
  return cached;
}

/**
 * Serialize all storage modifications. `fn` receives the latest committed
 * state and may return a new state via commit(); concurrent mutations queue
 * instead of clobbering each other.
 */
function mutate<T>(fn: (state: BlockList) => T | Promise<T>): Promise<T> {
  const run = writeChain.then(() => loadState()).then(fn);
  writeChain = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

/** Bump revision, persist, then broadcast. Only called inside mutate(). */
async function commit(state: BlockList): Promise<BlockList> {
  const next: BlockList = { ...state, revision: state.revision + 1 };
  await chrome.storage.local.set({ [STORAGE_KEY]: next });
  cached = next;
  resultCache.clear();
  broadcast(next);
  return next;
}

function broadcast(state: BlockList): void {
  const message = {
    target: "content",
    type: "STATE_CHANGED",
    revision: state.revision,
    enabled: state.enabled,
  };
  void chrome.tabs
    .query({})
    .then((tabs) => {
      for (const tab of tabs) {
        if (tab.id == null) continue;
        // Tabs without the content script (chrome://, extension pages, new
        // tabs) reject — expected, not an error worth surfacing.
        void chrome.tabs.sendMessage(tab.id, message).catch(() => {});
      }
    })
    .catch(() => {});
}

/* ---- offscreen document: single-flight creation, one recreate-retry ---- */

let offscreenFlight: Promise<void> | null = null;

function ensureOffscreen(): Promise<void> {
  if (!offscreenFlight) {
    offscreenFlight = (async () => {
      if (await chrome.offscreen.hasDocument()) return;
      await chrome.offscreen.createDocument({
        url: OFFSCREEN_URL,
        reasons: ["WORKERS"],
        justification: "Local face detection and embedding inference for blocked identities.",
      });
    })();
    // A failed creation must not poison future attempts.
    offscreenFlight.catch(() => {
      offscreenFlight = null;
    });
  }
  return offscreenFlight;
}

interface OffscreenResponse {
  ok?: boolean;
  error?: string;
  identity?: SavedIdentity;
  result?: ImageResult;
  preview?: EnrollPreview;
}

async function sendToOffscreen<R extends { ok?: boolean; error?: string } = OffscreenResponse>(
  message: Record<string, unknown>,
): Promise<R> {
  let lastError: unknown;
  for (let attempt = 0; attempt < 2; attempt++) {
    await ensureOffscreen();
    try {
      const response = (await chrome.runtime.sendMessage({
        ...message,
        target: "offscreen",
      })) as R | undefined;
      if (response == null) {
        throw new Error("Offscreen inference document did not respond.");
      }
      return response;
    } catch (error) {
      lastError = error;
      // The document may have died between hasDocument() and sendMessage.
      // Drop it so the retry recreates cleanly; surface the error if it persists.
      offscreenFlight = null;
      try {
        await chrome.offscreen.closeDocument();
      } catch {
        /* already gone */
      }
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

/* ---- ANALYZE job queue + bounded result cache ---- */

let inflight = 0;
const waiters: Array<() => void> = [];

async function queued<T>(job: () => Promise<T>): Promise<T> {
  if (inflight >= MAX_INFLIGHT_ANALYZE) {
    if (waiters.length >= MAX_QUEUED_ANALYZE) {
      throw new Error("FaceBlock is busy — too many images queued.");
    }
    await new Promise<void>((resolve) => waiters.push(resolve));
  }
  inflight += 1;
  try {
    return await job();
  } finally {
    inflight -= 1;
    const next = waiters.shift();
    if (next) next();
  }
}

/**
 * Video frames get their own counter, separate from `inflight`: a stale frame
 * is worthless, so a second ANALYZE_FRAME while one is being analysed is
 * rejected as "busy" immediately instead of queued behind it.
 */
let inflightFrames = 0;

const resultCache = new Map<string, ImageResult>();

function cacheResult(key: string, result: ImageResult): void {
  resultCache.delete(key);
  resultCache.set(key, result);
  while (resultCache.size > RESULT_CACHE_LIMIT) {
    const oldest = resultCache.keys().next();
    if (oldest.done) break;
    resultCache.delete(oldest.value);
  }
}

/* ---- handlers ---- */

/**
 * Shared preview path for RESOLVE_PREVIEW and its legacy alias BLOCK_NAME.
 * Never persists: the offscreen document gathers and embeds candidate faces
 * and returns a preview; only CONFIRM_ENROLL writes storage.
 *
 * `identityId` handling: an explicit id must name an existing saved identity
 * (the refresh path) — anything else fails closed. Otherwise the preview
 * targets an existing identity when the canonical or typed name resolves to
 * one, so confirming replaces rather than duplicates.
 */
async function resolvePreview(
  message: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const name = typeof message.name === "string" ? message.name.trim() : "";
  if (!name) return { ok: false, error: "Enter a name to block." };
  const explicitId =
    typeof message.identityId === "string" && message.identityId !== ""
      ? message.identityId
      : null;
  const state = await loadState();
  if (explicitId && !state.identities.some((i) => i.id === explicitId)) {
    return { ok: false, error: `No saved identity has id "${explicitId}".` };
  }
  // Long-running: gathers candidates, downloads them, and runs inference.
  const response = await sendToOffscreen({ type: "RESOLVE_PREVIEW", name });
  if (!response.ok || !response.preview) {
    return { ok: false, error: response.error ?? "Could not gather reference photos." };
  }
  const preview = response.preview;
  const target = selectPreviewIdentityId(state, {
    explicitId,
    canonicalId: preview.identityId ?? null,
    name: preview.name || name,
  });
  if (target.error) return { ok: false, error: target.error };
  if (target.identityId) {
    preview.identityId = target.identityId;
  } else {
    delete preview.identityId;
  }
  return { ok: true, preview };
}
async function processImage(
  message: Record<string, unknown>,
  sender: ChromeSender,
): Promise<Record<string, unknown>> {
  if (!sender.tab || sender.tab.id == null) {
    return { ok: false, error: "PROCESS_IMAGE is only accepted from content scripts." };
  }
  const url = typeof message.url === "string" ? message.url : "";
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return { ok: false, error: "Invalid image URL." };
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    return { ok: false, error: "Only http(s) images are processed." };
  }

  const state = await loadState();
  if (!state.enabled || state.identities.length === 0) {
    // Nothing to match against — skip inference entirely.
    return { ok: true, result: EMPTY_RESULT };
  }

  const key = `${state.revision}|${url}`;
  const hit = resultCache.get(key);
  if (hit) return { ok: true, result: hit };

  try {
    const response = await queued(() =>
      sendToOffscreen({ type: "ANALYZE", url, identities: state.identities }),
    );
    if (!response.ok || !response.result) {
      return { ok: false, error: response.error ?? "Analysis failed." };
    }
    cacheResult(key, response.result);
    return { ok: true, result: response.result };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) };
  }
}

interface OffscreenFrameResponse {
  ok?: boolean;
  error?: string;
  result?: FrameResult;
}

async function processFrame(
  message: Record<string, unknown>,
  sender: ChromeSender,
): Promise<Record<string, unknown>> {
  if (!sender.tab || sender.tab.id == null) {
    return { ok: false, error: "ANALYZE_FRAME is only accepted from content scripts." };
  }
  // The frame travels as base64: runtime.sendMessage JSON-serializes, so a
  // binary payload would arrive as `{}` with the bytes lost.
  const jpegBase64 = typeof message.jpegBase64 === "string" ? message.jpegBase64 : "";
  if (jpegBase64.length === 0) {
    return { ok: false, error: "ANALYZE_FRAME requires a non-empty jpegBase64 string." };
  }

  const state = await loadState();
  if (!state.enabled || state.identities.length === 0) {
    // Nothing to match against — skip inference entirely.
    return { ok: true, result: EMPTY_RESULT };
  }

  // Drop, don't queue: by the time a queued frame ran it would be stale.
  if (inflightFrames >= MAX_INFLIGHT_FRAME) {
    return { ok: false, error: "busy" };
  }
  inflightFrames += 1;
  try {
    // Frames are never put in resultCache — unlike an image URL a frame is
    // not revisitable, so there is no cache key worth keeping.
    const response = await sendToOffscreen<OffscreenFrameResponse>({
      type: "ANALYZE_FRAME",
      jpegBase64,
      identities: state.identities,
    });
    if (!response.ok || !response.result) {
      return { ok: false, error: response.error ?? "Frame analysis failed." };
    }
    return { ok: true, result: response.result };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) };
  } finally {
    inflightFrames -= 1;
  }
}

async function handleMessage(
  message: Record<string, unknown>,
  sender: ChromeSender,
): Promise<Record<string, unknown>> {
  // Only our extension-owned pages may manage identities or read embeddings.
  // Content scripts receive a redacted state for boot-time visibility gating.
  const fromContent = !sender.url?.startsWith(`chrome-extension://${chrome.runtime.id}/`);
  switch (message.type) {
    case "GET_STATE":
      if (fromContent) {
        const state = await loadState();
        return { ok: true, state: { enabled: state.enabled, revision: state.revision, identities: [] } };
      }
      // Cheap path: never touches the offscreen document or the job queue,
      // so the UI stays responsive during long inference.
      return { ok: true, state: await loadState() };

    case "BLOCK_NAME":
    // Self-seed enrollment. Both routes carry embeddings in the response, so
    // they are extension-pages-only. BLOCK_NAME is the legacy alias of
    // RESOLVE_PREVIEW: it returns a preview and never persists.
    case "RESOLVE_PREVIEW": {
      if (fromContent) return { ok: false, error: "Extension pages only." };
      return resolvePreview(message);
    }

    case "REMOVE": {
      if (fromContent) return { ok: false, error: "Extension pages only." };
      const id = typeof message.id === "string" ? message.id : "";
      if (!id) return { ok: false, error: "Missing identity id." };
      const state = await mutate((s) => {
        if (!s.identities.some((i) => i.id === id)) return s;
        return commit({ ...s, identities: s.identities.filter((i) => i.id !== id) });
      });
      return { ok: true, state };
    }

    case "SET_ENABLED": {
      if (fromContent) return { ok: false, error: "Extension pages only." };
      const enabled = message.enabled === true;
      const state = await mutate((s) => (s.enabled === enabled ? s : commit({ ...s, enabled })));
      return { ok: true, state };
    }

    case "PROCESS_IMAGE":
      return processImage(message, sender);

    case "ANALYZE_FRAME":
      return processFrame(message, sender);



    case "CONFIRM_ENROLL": {
      if (fromContent) return { ok: false, error: "Extension pages only." };
      const name = typeof message.name === "string" ? message.name.trim() : "";
      if (!name) return { ok: false, error: "Missing name." };
      const faces = Array.isArray(message.faces) ? message.faces : [];
      if (faces.length === 0) {
        return { ok: false, error: "Keep at least one reference face." };
      }
      const explicitId =
        typeof message.identityId === "string" && message.identityId !== ""
          ? message.identityId
          : null;
      // Fail closed before the offscreen round-trip: an explicit identityId
      // is the refresh path and must name a saved identity.
      if (explicitId) {
        const state = await loadState();
        if (!state.identities.some((i) => i.id === explicitId)) {
          return { ok: false, error: `No saved identity has id "${explicitId}".` };
        }
      }
      const response = await sendToOffscreen({
        type: "CONFIRM_ENROLL",
        name,
        faces,
        ...(explicitId ? { identityId: explicitId } : {}),
      });
      if (!response.ok || !response.identity) {
        return { ok: false, error: response.error ?? "Enrollment failed." };
      }
      const identity = response.identity;
      const next = await mutate((s) => {
        // Revalidate inside the serialized write: a second page may have
        // REMOVED the identity while the confirmation was in flight, and a
        // late refresh must not resurrect it.
        if (explicitId && !s.identities.some((i) => i.id === explicitId)) {
          throw new Error(`No saved identity has id "${explicitId}".`);
        }
        // Re-running for the same person replaces their references rather
        // than silently keeping stale ones; merge preserves the existing id,
        // stored threshold, and createdAt.
        return commit(mergeConfirmedIdentity(s, identity));
      });
      return { ok: true, state: next, identity };
    }

    default:
      return { ok: false, error: `Unknown message type: ${String(message.type)}` };
  }
}

/* ---- wiring ---- */

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Routing filter: only messages addressed to the background worker.
  if (
    sender.id !== chrome.runtime.id ||
    typeof message !== "object" ||
    message === null ||
    (message as Record<string, unknown>).target !== "background"
  ) {
    return undefined;
  }
  handleMessage(message as Record<string, unknown>, sender)
    .then(sendResponse)
    .catch((error: unknown) => {
      sendResponse({ ok: false, error: error instanceof Error ? error.message : String(error) });
    });
  return true; // async sendResponse
});

// Initialize persisted defaults on every worker start (install, restart, wake).
void loadState();
