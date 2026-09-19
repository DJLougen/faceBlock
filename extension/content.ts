/**
 * FaceBlock content script (MV3, http/https, all_frames).
 *
 * Watches visible <img> elements, asks the background worker to analyze them
 * on-device, and draws opaque fixed-position masks over matched faces. Masks
 * live in a closed shadow root so page CSS/JS cannot restyle or inspect them;
 * no identity data, embeddings, or names ever enter the page DOM.
 *
 * Video: visible <video> elements are sampled at a LOW rate — never per
 * frame; the plan forbids 30/60 FPS recognition. Between samples, masks are
 * coasted on requestVideoFrameCallback so they follow faces at full video
 * rate; a fast-moving face can briefly show a stale box. Sampled frames are
 * analysed on device and discarded; frames, embeddings, and match results
 * are never transmitted.
 *
 * Fast mode: images are never pre-hidden. A face is concealed only after a
 * positive local match, so a matched face may be briefly visible on first
 * sight (disclosed once via console.info).
 */

import {
  BOX_MARGIN_X,
  BOX_MARGIN_Y,
  BOX_SCALE_X,
  BOX_SCALE_Y,
  MIN_MEDIA_PX,
} from "../src/shared/config.ts";
import { expandBox } from "../src/overlay/coordinates.ts";
import { sampleVideoFrame } from "../src/cv/raster.ts";
import { applyDetections, coastTrack, type Track } from "../src/tracking/iou.ts";
import type { Box, ObjectFit, Size } from "../src/shared/types.ts";
import type { FrameResult, ImageResult } from "./protocol.ts";

/* ------------------------------------------------------------------ */
/* Chrome runtime shim (kept local so this file has no @types/chrome   */
/* dependency and cannot collide with a global `chrome` declaration).  */
/* ------------------------------------------------------------------ */

interface RuntimeShim {
  sendMessage(message: unknown, callback: (response: unknown) => void): void;
  onMessage: { addListener(listener: (message: unknown) => void): void };
  lastError?: { message?: string };
}

const runtime: RuntimeShim | undefined = (
  globalThis as { chrome?: { runtime?: RuntimeShim } }
).chrome?.runtime;

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

/** Hard bound on tracked elements so a pathological page cannot grow the map. */
const MAX_TRACKED = 4096;
/** Hard bound on videos under active sampling; the largest visible ones win. */
const MAX_TRACKED_VIDEOS = 8;
/** At most this many ANALYZE_FRAME requests in flight across all videos. */
const MAX_INFLIGHT_FRAMES = 2;
/** Frames are downscaled to this width before analysis — small and cheap. */
const SAMPLE_MAX_WIDTH = 480;
/**
 * Adaptive recognition cadence: a video with a matched face on screen is
 * re-recognized more often because a moving match needs tighter correction,
 * while an empty frame only needs occasional re-checks.
 */
const SAMPLE_IDLE_MS = 800;
const SAMPLE_ACTIVE_MS = 250;
/** IoU threshold shared by applyDetections and our track->detection identity handoff. */
const TRACK_IOU = 0.3;
/** Consecutive missed detection rounds before a track is dropped. */
const TRACK_MAX_MISSES = 4;
/** Detached-element sweep + SPA rescan backstop. */
const SWEEP_MS = 4000;
/** Periodic realign covers CSS transitions/animations that fire no events. */
const REALIGN_MS = 1000;
/** Delay before re-evaluating after a src/srcset mutation so currentSrc settles. */
const SRC_REEVAL_MS = 40;
/** Marks our overlay host so the MutationObserver skips our own mutations. */
const LAYER_ATTR = "data-fb-layer";
const MASK_CSS = "position:absolute;background:#000;pointer-events:none;display:block;";
const FITS: Record<string, true> = { fill: true, contain: true, cover: true, none: true, "scale-down": true };
const stats = {
  seen: 0,
  queued: 0,
  sent: 0,
  masked: 0,
  skippedScheme: 0,
  skippedSmall: 0,
  skippedEmpty: 0,
  droppedStale: 0,
  overflow: 0,
  errors: 0,
  videoSeen: 0,
  videoFramesSent: 0,
  videoBusy: 0,
  videoMasked: 0,
  videoUnanalyzable: 0,
  get tracked(): number {
    return tracked.size;
  },
  get trackedVideos(): number {
    return vtracked.size;
  },
};
(globalThis as Record<string, unknown>).__faceblockStats = stats;

/* ------------------------------------------------------------------ */
/* State                                                               */
/* ------------------------------------------------------------------ */

interface Tracked {
  img: HTMLImageElement;
  visible: boolean;
  /** Last URL we evaluated; dedupes queueing across IO/RO/load/mutation. */
  evaluated: string | null;
  /** True when the last evaluation skipped the image for being < MIN_MEDIA_PX. */
  skipSmall: boolean;
  /** Stale-response guard: bumped on src change, untrack, revision change. */
  token: number;
  regions: ImageResult["regions"] | null;
  source: Size | null;
  masks: HTMLElement[];
  evalTimer: number | undefined;
}

/**
 * Results for URLs already analysed, so a re-render can be masked in the SAME
 * task that inserts the node -- before the browser paints it.
 *
 * Without this, every image needed a round trip to the background and the
 * offscreen document (tens to hundreds of milliseconds) before a mask could be
 * drawn, and the code header admitted a matched face was briefly visible on
 * first sight. On a timeline the same URL re-renders constantly while scrolling
 * and navigating, so most sightings are repeats and can be blocked with no
 * visible frame at all.
 *
 * Bounded like the background's cache: unbounded growth on a long session is a
 * memory leak.
 */
const RESULT_CACHE_LIMIT = 500;
const resultByUrl = new Map<string, ImageResult>();

function rememberResult(url: string, result: ImageResult): void {
  resultByUrl.delete(url);
  resultByUrl.set(url, result);
  while (resultByUrl.size > RESULT_CACHE_LIMIT) {
    const oldest = resultByUrl.keys().next();
    if (oldest.done) break;
    resultByUrl.delete(oldest.value);
  }
}

const tracked = new Map<HTMLImageElement, Tracked>();
const queue: Tracked[] = [];
let pumping = false;
let enabled = false;
let revision = 0;

let overlayHost: HTMLElement | null = null;
let overlayRoot: ShadowRoot | null = null;

/* ------------------------------------------------------------------ */
/* Messaging                                                           */
/* ------------------------------------------------------------------ */

function send(message: unknown): Promise<Record<string, unknown> | null> {
  const { promise, resolve } = Promise.withResolvers<Record<string, unknown> | null>();
  try {
    runtime!.sendMessage(message, (response: unknown) => {
      // Reading lastError consumes it; a missing response is not fatal.
      void runtime!.lastError;
      resolve(response && typeof response === "object" ? (response as Record<string, unknown>) : null);
    });
  } catch {
    resolve(null);
  }
  return promise;
}

/* ------------------------------------------------------------------ */
/* Overlay layer                                                       */
/* ------------------------------------------------------------------ */

function ensureOverlayRoot(): ShadowRoot | null {
  if (overlayRoot && overlayHost && overlayHost.isConnected) return overlayRoot;
  const docEl = document.documentElement;
  if (!docEl) return null;
  if (!overlayHost || !overlayRoot) {
    overlayHost = document.createElement("div");
    overlayHost.setAttribute(LAYER_ATTR, "");
    overlayHost.style.cssText =
      "position:fixed;left:0;top:0;width:0;height:0;z-index:2147483647;pointer-events:none;";
    overlayRoot = overlayHost.attachShadow({ mode: "closed" });
  }
  if (!overlayHost.isConnected) docEl.appendChild(overlayHost);
  return overlayRoot;
}

function hideMasks(t: { masks: HTMLElement[] }): void {
  for (const m of t.masks) m.style.display = "none";
}

function clearMasks(t: Tracked): void {
  for (const m of t.masks) m.remove();
  t.masks = [];
  t.regions = null;
}

/* ------------------------------------------------------------------ */
/* Geometry: source-image box -> viewport box                          */
/* ------------------------------------------------------------------ */

interface PosComponent {
  pct: number;
  px: number;
  fromEnd: boolean;
}

function centerPos(): PosComponent {
  return { pct: 50, px: 0, fromEnd: false };
}

function axisOffset(c: PosComponent, free: number): number {
  const o = c.px + (c.pct / 100) * free;
  return c.fromEnd ? free - o : o;
}

/**
 * Parse the common object-position forms: keywords, lengths, percentages,
 * and keyword+offset pairs ("right 10px"). Unsupported tokens (calc(), em,
 * 3/4-value edge offsets) fall back to center for that axis.
 */
function parseObjectPosition(value: string): { x: PosComponent; y: PosComponent } {
  const res = { x: centerPos(), y: centerPos() };
  const tokens = (value || "center").trim().toLowerCase().split(/\s+/).filter(Boolean);
  let pending: "x" | "y" | null = null;
  let xDone = false;
  let yDone = false;
  const lengthOf = (tok: string): { pct: number; px: number } | null => {
    if (tok.endsWith("%")) {
      const n = parseFloat(tok);
      return Number.isFinite(n) ? { pct: n, px: 0 } : null;
    }
    if (tok.endsWith("px")) {
      const n = parseFloat(tok);
      return Number.isFinite(n) ? { pct: 0, px: n } : null;
    }
    if (tok === "0") return { pct: 0, px: 0 };
    return null;
  };
  for (const tok of tokens) {
    if (tok === "left" || tok === "right") {
      res.x = { pct: 0, px: 0, fromEnd: tok === "right" };
      xDone = true;
      pending = "x";
      continue;
    }
    if (tok === "top" || tok === "bottom") {
      res.y = { pct: 0, px: 0, fromEnd: tok === "bottom" };
      yDone = true;
      pending = "y";
      continue;
    }
    if (tok === "center") {
      if (!xDone) {
        res.x = centerPos();
        xDone = true;
      } else if (!yDone) {
        res.y = centerPos();
        yDone = true;
      }
      pending = null;
      continue;
    }
    const len = lengthOf(tok);
    if (!len) continue; // unsupported token: keep defaults
    if (pending) {
      res[pending].px = len.px;
      res[pending].pct = len.pct;
      pending = null;
    } else if (!xDone) {
      res.x = { ...len, fromEnd: false };
      xDone = true;
    } else if (!yDone) {
      res.y = { ...len, fromEnd: false };
      yDone = true;
    }
  }
  return res;
}

/** Map a source-pixel box into the img's content box, honoring object-fit/position. */
function mapRegion(
  region: Box,
  source: Size,
  content: Size,
  fit: ObjectFit,
  pos: { x: PosComponent; y: PosComponent },
): Box {
  const sw = source.width;
  const sh = source.height;
  if (sw <= 0 || sh <= 0 || content.width <= 0 || content.height <= 0) {
    return { x: 0, y: 0, width: 0, height: 0 };
  }
  let sx: number;
  let sy: number;
  switch (fit) {
    case "contain": {
      sx = sy = Math.min(content.width / sw, content.height / sh);
      break;
    }
    case "cover": {
      sx = sy = Math.max(content.width / sw, content.height / sh);
      break;
    }
    case "none": {
      sx = sy = 1;
      break;
    }
    case "scale-down": {
      sx = sy =
        sw > content.width || sh > content.height
          ? Math.min(content.width / sw, content.height / sh)
          : 1;
      break;
    }
    default: {
      // "fill" and any unknown value: stretch to the content box.
      sx = content.width / sw;
      sy = content.height / sh;
      break;
    }
  }
  const drawnW = sw * sx;
  const drawnH = sh * sy;
  const offX = axisOffset(pos.x, content.width - drawnW);
  const offY = axisOffset(pos.y, content.height - drawnH);
  return {
    x: region.x * sx + offX,
    y: region.y * sy + offY,
    width: region.width * sx,
    height: region.height * sy,
  };
}

interface ContentRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** Content box of the element in viewport coordinates (border box minus border/padding). */
function contentRect(el: HTMLElement, cs: CSSStyleDeclaration): ContentRect | null {
  const r = el.getBoundingClientRect();
  if (r.width <= 0 || r.height <= 0) return null;
  const num = (v: string): number => {
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : 0;
  };
  const width =
    r.width - num(cs.borderLeftWidth) - num(cs.borderRightWidth) - num(cs.paddingLeft) - num(cs.paddingRight);
  const height =
    r.height - num(cs.borderTopWidth) - num(cs.borderBottomWidth) - num(cs.paddingTop) - num(cs.paddingBottom);
  if (width <= 0 || height <= 0) return null;
  return {
    left: r.left + num(cs.borderLeftWidth) + num(cs.paddingLeft),
    top: r.top + num(cs.borderTopWidth) + num(cs.paddingTop),
    width,
    height,
  };
}

/* ------------------------------------------------------------------ */
/* Mask layout                                                         */
/* ------------------------------------------------------------------ */

function layoutMasks(t: Tracked): void {
  if (!t.regions || !t.source || !t.img.isConnected) {
    hideMasks(t);
    return;
  }
  const cs = getComputedStyle(t.img);
  const rect = contentRect(t.img, cs);
  if (!rect) {
    hideMasks(t);
    return;
  }
  const root = ensureOverlayRoot();
  if (!root) {
    hideMasks(t);
    return;
  }
  const fit = FITS[cs.objectFit] ? (cs.objectFit as ObjectFit) : "fill";
  const pos = parseObjectPosition(cs.objectPosition);
  const content: Size = { width: rect.width, height: rect.height };
  for (let i = 0; i < t.regions.length; i++) {
    const region = t.regions[i]!;
    let mask = t.masks[i];
    if (!mask) {
      mask = document.createElement("div");
      mask.style.cssText = MASK_CSS;
      t.masks[i] = mask;
      root.appendChild(mask);
    }
    const mapped = mapRegion(region, t.source, content, fit, pos);
    const box = expandBox(
      mapped,
      { marginX: BOX_MARGIN_X, marginY: BOX_MARGIN_Y, scaleX: BOX_SCALE_X, scaleY: BOX_SCALE_Y },
      content,
    );
    if (box.width <= 0 || box.height <= 0) {
      mask.style.display = "none";
      continue;
    }
    mask.style.display = "block";
    mask.style.left = `${rect.left + box.x}px`;
    mask.style.top = `${rect.top + box.y}px`;
    mask.style.width = `${box.width}px`;
    mask.style.height = `${box.height}px`;
  }
}

let realignPending = false;
function scheduleRealign(): void {
  if (realignPending) return;
  realignPending = true;
  requestAnimationFrame(() => {
    realignPending = false;
    realign();
  });
}

function realign(): void {
  try {
    for (const t of tracked.values()) {
      if (t.regions && t.regions.length) layoutMasks(t);
    }
    for (const v of vtracked.values()) layoutVideoMasks(v);
  } catch {
    stats.errors++;
  }
}

/* ------------------------------------------------------------------ */
/* Processing queue (concurrency = 1; background owns the URL cache)   */
/* ------------------------------------------------------------------ */

function maybeQueue(t: Tracked): void {
  if (!enabled || !t.visible || !t.img.isConnected) return;
  const url = t.img.currentSrc || t.img.src || "";
  if (!url) {
    t.evaluated = "";
    return;
  }
  if (url === t.evaluated) return;
  t.evaluated = url;
  if (!/^https?:\/\//i.test(url)) {
    // blob:, data:, file:, about: — cannot be fetched by the extension.
    stats.skippedScheme++;
    t.skipSmall = false;
    return;
  }
  if (t.img.complete && t.img.naturalWidth === 0) {
    stats.skippedEmpty++;
    t.skipSmall = false;
    return;
  }
  if (t.img.clientWidth < MIN_MEDIA_PX || t.img.clientHeight < MIN_MEDIA_PX) {
    stats.skippedSmall++;
    t.skipSmall = true;
    return;
  }
  t.skipSmall = false;
  queue.push(t);
  stats.queued++;
  void pump();
}

async function pump(): Promise<void> {
  if (pumping) return;
  pumping = true;
  try {
    while (queue.length) {
      const t = queue.shift()!;
      if (tracked.get(t.img) !== t || !t.img.isConnected || !enabled) continue;
      const url = t.img.currentSrc || t.img.src || "";
      // evaluated is reset on src mutation, so a mismatch means this entry is stale.
      if (!url || url !== t.evaluated || !/^https?:\/\//i.test(url)) continue;
      const token = ++t.token;
      const rev = revision;
      stats.sent++;
      const res = await send({ target: "background", type: "PROCESS_IMAGE", url });
      if (tracked.get(t.img) !== t || t.token !== token || rev !== revision) {
        stats.droppedStale++;
        continue;
      }
      if (!res || res.ok !== true) {
        stats.errors++;
        continue;
      }
      const result = res.result as ImageResult | undefined;
      if (!result || !Array.isArray(result.regions)) continue;
      rememberResult(url, result);
      t.source = { width: result.width, height: result.height };
      t.regions = result.regions;
      if (result.regions.length) {
        stats.masked++;
        try {
          layoutMasks(t);
        } catch {
          stats.errors++;
        }
      } else {
        clearMasks(t);
      }
    }
  } catch {
    stats.errors++;
  } finally {
    pumping = false;
  }
}

/* ------------------------------------------------------------------ */
/* Tracking                                                            */
/* ------------------------------------------------------------------ */

function track(img: HTMLImageElement): void {
  // Fast path: an image we have already judged gets its mask now, in this task,
  // so it is never shown unblocked. This runs before any queueing.
  if (tracked.has(img)) return;
  if (tracked.size >= MAX_TRACKED) {
    stats.overflow++;
    return;
  }
  stats.seen++;
  const t: Tracked = {
    img,
    visible: false,
    evaluated: null,
    skipSmall: false,
    token: 0,
    regions: null,
    source: null,
    masks: [],
    evalTimer: undefined,
  };
  tracked.set(img, t);
  io.observe(img);
  ro.observe(img);

  // Synchronous fast path. `track` runs from the insertion observer, which is
  // the same task that added the node -- the browser has not painted it yet. If
  // this URL has already been judged, the mask is drawn now and the image is
  // never shown unblocked. Marking it evaluated also stops the queue from
  // spending a round trip to learn what we already know.
  const known = img.currentSrc || img.src || "";
  if (known) {
    const cached = resultByUrl.get(known);
    if (cached) {
      t.evaluated = known;
      t.source = { width: cached.width, height: cached.height };
      t.regions = cached.regions;
      if (cached.regions.length) {
        stats.masked++;
        try {
          layoutMasks(t);
        } catch {
          stats.errors++;
        }
      }
      return;
    }
  }
  maybeQueue(t);
}

function untrack(img: HTMLImageElement): void {
  const t = tracked.get(img);
  if (!t) return;
  t.token++;
  if (t.evalTimer !== undefined) clearTimeout(t.evalTimer);
  clearMasks(t);
  io.unobserve(img);
  ro.unobserve(img);
  tracked.delete(img);
}

function scan(root: Node): void {
  if (root instanceof HTMLImageElement) {
    track(root);
    return;
  }
  if (root instanceof HTMLVideoElement) {
    trackVideo(root);
    return;
  }
  if (root === overlayHost) return;
  if (root instanceof Element || root instanceof Document || root instanceof DocumentFragment) {
    for (const img of root.querySelectorAll<HTMLImageElement>("img")) track(img);
    for (const video of root.querySelectorAll<HTMLVideoElement>("video")) trackVideo(video);
    // querySelectorAll does not cross shadow boundaries, and modern sites
    // render their players inside open shadow roots. Walk into them, or those
    // videos are simply never discovered.
    for (const host of root.querySelectorAll<HTMLElement>("*")) {
      const shadow = host.shadowRoot;
      if (shadow) scan(shadow);
    }
  }
}

function scheduleEval(t: Tracked): void {
  if (t.evalTimer !== undefined) clearTimeout(t.evalTimer);
  t.evalTimer = setTimeout(() => {
    t.evalTimer = undefined;
    try {
      maybeQueue(t);
    } catch {
      stats.errors++;
    }
  }, SRC_REEVAL_MS) as unknown as number;
}

/** src/srcset/sizes changed: drop masks immediately, invalidate in-flight, re-evaluate. */
function onImgAttr(img: HTMLImageElement): void {
  const t = tracked.get(img);
  if (!t) {
    track(img);
    return;
  }
  t.token++;
  clearMasks(t);
  t.evaluated = null;
  t.skipSmall = false;
  scheduleEval(t);
}

/* ------------------------------------------------------------------ */
/* Video tracking                                                      */
/* ------------------------------------------------------------------ */

/**
 * One video under watch.
 *
 * `tracks` are in FRAME pixel coordinates — the downscaled sample the detector
 * actually saw. Rendering scales frame -> intrinsic video -> viewport, so the
 * masks stay correct whatever size the sample happened to be.
 */
interface Vtracked {
  video: HTMLVideoElement;
  visible: boolean;
  tracks: Track[];
  masks: HTMLElement[];
  /** Dimensions of the frame the current tracks were detected in. */
  frameSize: Size | null;
  /** Timestamp of the last recognition request. */
  lastSampleMs: number;
  /** At most one frame in flight per video; more would only queue stale work. */
  inflight: boolean;
  /** Canvas taint or decode failure — never sample this video again. */
  unanalyzable: boolean;
  /** Stale-response guard, the same role Tracked.token plays for images. */
  token: number;
  frameHandle: number | undefined;
  rafHandle: number | undefined;
}

const vtracked = new Map<HTMLVideoElement, Vtracked>();
/** ANALYZE_FRAME requests in flight across all videos. */
let framesInflight = 0;

type FrameCallbackVideo = HTMLVideoElement & {
  requestVideoFrameCallback(cb: () => void): number;
  cancelVideoFrameCallback(handle: number): void;
};

function hasFrameCallback(video: HTMLVideoElement): video is FrameCallbackVideo {
  return (
    "requestVideoFrameCallback" in video && typeof video.requestVideoFrameCallback === "function"
  );
}

/** Intrinsic video dimensions, or null before the first frame is decoded. */
function videoSource(video: HTMLVideoElement): Size | null {
  if (video.videoWidth <= 0 || video.videoHeight <= 0) return null;
  return { width: video.videoWidth, height: video.videoHeight };
}

function videoArea(video: HTMLVideoElement): number {
  const r = video.getBoundingClientRect();
  return r.width * r.height;
}

/* ---- mask rendering ---- */

/**
 * Place one mask from a frame-space box.
 *
 * Frame -> intrinsic -> viewport. The intrinsic step matters because the
 * sample is downscaled; the viewport step reuses the image pipeline's
 * object-fit/object-position mapping so letterboxed video lands correctly.
 */
function positionVideoMask(v: Vtracked, mask: HTMLElement, frameBox: Box): void {
  const video = v.video;
  const source = videoSource(video);
  const frame = v.frameSize;
  if (!source || !frame || !video.isConnected || frame.width <= 0 || frame.height <= 0) {
    mask.style.display = "none";
    return;
  }
  const cs = getComputedStyle(video);
  const rect = contentRect(video, cs);
  if (!rect) {
    mask.style.display = "none";
    return;
  }
  const intrinsic: Box = {
    x: (frameBox.x * source.width) / frame.width,
    y: (frameBox.y * source.height) / frame.height,
    width: (frameBox.width * source.width) / frame.width,
    height: (frameBox.height * source.height) / frame.height,
  };
  const fit = FITS[cs.objectFit] ? (cs.objectFit as ObjectFit) : "fill";
  const pos = parseObjectPosition(cs.objectPosition);
  const content: Size = { width: rect.width, height: rect.height };
  const mapped = mapRegion(intrinsic, source, content, fit, pos);
  const box = expandBox(
    mapped,
    { marginX: BOX_MARGIN_X, marginY: BOX_MARGIN_Y, scaleX: BOX_SCALE_X, scaleY: BOX_SCALE_Y },
    content,
  );
  if (box.width <= 0 || box.height <= 0) {
    mask.style.display = "none";
    return;
  }
  mask.style.display = "block";
  mask.style.left = `${rect.left + box.x}px`;
  mask.style.top = `${rect.top + box.y}px`;
  mask.style.width = `${box.width}px`;
  mask.style.height = `${box.height}px`;
}

/**
 * Move every mask to its coasted position.
 *
 * Called once per PRESENTED frame. Recognition is deliberately slow, so this
 * is what keeps the mask glued to a moving face; while the video is paused the
 * velocity is zero and the boxes stay exactly where they were detected.
 */
function renderVideoMasks(v: Vtracked, nowMs: number): void {
  for (let i = 0; i < v.tracks.length; i++) {
    const mask = v.masks[i];
    if (mask) positionVideoMask(v, mask, coastTrack(v.tracks[i]!, nowMs));
  }
}

/** Realign hook: re-place masks from the current tracks without re-detecting. */
function layoutVideoMasks(v: Vtracked): void {
  renderVideoMasks(v, performance.now());
}

/** Keep one mask element per track, in track order. */
function syncVideoMasks(v: Vtracked): void {
  const root = ensureOverlayRoot();
  if (!root) return;
  while (v.masks.length < v.tracks.length) {
    const mask = document.createElement("div");
    mask.style.cssText = MASK_CSS;
    v.masks.push(mask);
    root.appendChild(mask);
  }
  while (v.masks.length > v.tracks.length) {
    v.masks.pop()?.remove();
  }
}

/* ---- sampling ---- */

function shouldSample(v: Vtracked, nowMs: number): boolean {
  if (!enabled || v.unanalyzable || v.inflight) return false;
  if (!v.visible || !v.video.isConnected) return false;
  if (document.visibilityState !== "visible") return false;
  if (framesInflight >= MAX_INFLIGHT_FRAMES) return false;
  if (v.video.readyState < 2) return false;
  const r = v.video.getBoundingClientRect();
  if (r.width < MIN_MEDIA_PX || r.height < MIN_MEDIA_PX) return false;
  // A matched face needs tighter correction than an empty frame.
  const interval = v.tracks.length > 0 ? SAMPLE_ACTIVE_MS : SAMPLE_IDLE_MS;
  return nowMs - v.lastSampleMs >= interval;
}

/**
 * Base64-encode frame bytes.
 *
 * `chrome.runtime.sendMessage` JSON-serializes its payload rather than using
 * structured clone, so an ArrayBuffer arrives as `{}` — the bytes are lost.
 * Binary therefore has to cross the boundary as text.
 */
function toBase64(bytes: ArrayBuffer): string {
  const view = new Uint8Array(bytes);
  let binary = "";
  const CHUNK = 8192;
  for (let i = 0; i < view.length; i += CHUNK) {
    binary += String.fromCharCode(...view.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

async function sampleVideo(v: Vtracked): Promise<void> {
  const video = v.video;
  v.inflight = true;
  framesInflight++;
  v.lastSampleMs = performance.now();
  const token = v.token;
  try {
    const sample = await sampleVideoFrame(video, SAMPLE_MAX_WIDTH);
    if (!sample.ok) {
      if (sample.reason === "tainted") {
        // The canvas is tainted, so this video's pixels can never be read.
        // Stop sampling it — retrying every tick would burn CPU forever — and
        // say so out loud, because a silent stop looks like a broken feature.
        v.unanalyzable = true;
        stats.videoUnanalyzable++;
        console.warn(
          "[faceblock] video not readable: this video is cross-origin and does not " +
            "allow canvas access, so its frames cannot be analysed. Images on this " +
            "page are unaffected.",
          video.currentSrc || video.src,
        );
      }
      // "not-ready" and "unsupported" are recoverable: leave the video tracked
      // so a later tick (or a play/seek event) can sample it successfully.
      return;
    }
    stats.videoFramesSent++;
    const res = await send({
      target: "background",
      type: "ANALYZE_FRAME",
      jpegBase64: toBase64(sample.jpeg),
    });
    if (token !== v.token) return; // untracked or reset while in flight
    if (!res || res.ok !== true) {
      if (res && res.error === "busy") stats.videoBusy++;
      return;
    }
    const result = res.result as FrameResult | undefined;
    if (!result) return;
    v.frameSize = { width: result.width, height: result.height };
    // The offscreen document returns only MATCHED faces, so every detection is
    // a face worth masking and every resulting track inherits that identity.
    const detections: Box[] = result.regions.map((r) => ({
      x: r.x,
      y: r.y,
      width: r.width,
      height: r.height,
    }));
    v.tracks = applyDetections(v.tracks, detections, performance.now(), {
      iouThreshold: TRACK_IOU,
      maxMisses: TRACK_MAX_MISSES,
    });
    if (detections.length > 0) stats.videoMasked += detections.length;
    syncVideoMasks(v);
    renderVideoMasks(v, performance.now());
  } catch {
    stats.errors++;
  } finally {
    framesInflight--;
    v.inflight = false;
  }
}

/* ---- lifecycle ---- */

function cancelVideoLoop(v: Vtracked): void {
  if (v.frameHandle !== undefined && hasFrameCallback(v.video)) {
    v.video.cancelVideoFrameCallback(v.frameHandle);
    v.frameHandle = undefined;
  }
  if (v.rafHandle !== undefined) {
    cancelAnimationFrame(v.rafHandle);
    v.rafHandle = undefined;
  }
}

/**
 * Drive one video: render masks every presented frame, recognise on a slow
 * adaptive cadence.
 *
 * requestVideoFrameCallback fires once per presented video frame, so masks
 * move at video rate; it does not fire while paused, so a paused video falls
 * back to requestAnimationFrame (velocity is zero, so nothing needs moving —
 * the rAF loop exists only to pick up a fresh detection).
 */
function armVideoLoop(v: Vtracked): void {
  const tick = (): void => {
    if (!vtracked.has(v.video)) return;
    const now = performance.now();
    if (v.tracks.length > 0) renderVideoMasks(v, now);
    if (shouldSample(v, now)) void sampleVideo(v);
    armVideoLoop(v);
  };
  if (!v.video.paused && hasFrameCallback(v.video)) {
    v.frameHandle = v.video.requestVideoFrameCallback(tick);
  } else {
    v.rafHandle = requestAnimationFrame(tick);
  }
}

/**
 * Playback jumped or restarted: previous tracks are meaningless, so drop them
 * and recognise immediately rather than waiting out the idle interval.
 */
function onVideoSignal(this: HTMLVideoElement): void {
  const v = vtracked.get(this);
  if (!v) return;
  cancelVideoLoop(v);
  v.tracks = [];
  syncVideoMasks(v);
  v.lastSampleMs = 0;
  armVideoLoop(v);
}

function trackVideo(video: HTMLVideoElement): void {
  if (vtracked.has(video)) return;
  if (vtracked.size >= MAX_TRACKED_VIDEOS) {
    // Bound the work: keep watching the largest videos, which are the ones a
    // blocked face is most likely to be visible in.
    let smallestEl: HTMLVideoElement | null = null;
    let smallestArea = Number.POSITIVE_INFINITY;
    for (const el of vtracked.keys()) {
      const area = videoArea(el);
      if (area < smallestArea) {
        smallestArea = area;
        smallestEl = el;
      }
    }
    if (!smallestEl || smallestArea >= videoArea(video)) return;
    untrackVideo(smallestEl);
  }
  const entry: Vtracked = {
    video,
    visible: false,
    tracks: [],
    masks: [],
    frameSize: null,
    lastSampleMs: 0,
    inflight: false,
    unanalyzable: false,
    token: 0,
    frameHandle: undefined,
    rafHandle: undefined,
  };
  vtracked.set(video, entry);
  io.observe(video);
  ro.observe(video);
  video.addEventListener("seeked", onVideoSignal);
  video.addEventListener("play", onVideoSignal);
  video.addEventListener("pause", onVideoSignal);
  video.addEventListener("loadeddata", onVideoSignal);
  armVideoLoop(entry);
}

function untrackVideo(video: HTMLVideoElement): void {
  const v = vtracked.get(video);
  if (!v) return;
  v.token++;
  cancelVideoLoop(v);
  video.removeEventListener("seeked", onVideoSignal);
  video.removeEventListener("play", onVideoSignal);
  video.removeEventListener("pause", onVideoSignal);
  video.removeEventListener("loadeddata", onVideoSignal);
  io.unobserve(video);
  ro.unobserve(video);
  for (const mask of v.masks) mask.remove();
  v.masks.length = 0;
  vtracked.delete(video);
}

/** Drop every video track and mask, then let visible ones re-detect. */
function resetVideos(): void {
  for (const v of vtracked.values()) {
    v.token++;
    v.tracks = [];
    v.frameSize = null;
    v.lastSampleMs = 0;
    syncVideoMasks(v);
  }
}

/** A tab switch back should not wait out the sampling interval. */
function onVisibilityChange(): void {
  if (document.visibilityState !== "visible") return;
  for (const v of vtracked.values()) v.lastSampleMs = 0;
}

/* ------------------------------------------------------------------ */
/* Observers                                                           */
/* ------------------------------------------------------------------ */

const io = new IntersectionObserver(
  (entries) => {
    try {
      for (const e of entries) {
        const target = e.target;
        const t = tracked.get(target as HTMLImageElement);
        if (!t) {
          const v = vtracked.get(target as HTMLVideoElement);
          if (v) {
            v.visible = e.isIntersecting;
            // Sample promptly on entry rather than waiting out the interval.
            if (v.visible) v.lastSampleMs = 0;
          }
          continue;
        }
        t.visible = e.isIntersecting;
        if (t.visible) maybeQueue(t);
      }
    } catch {
      stats.errors++;
    }
  },
  { root: null, threshold: 0 },
);

const ro = new ResizeObserver((entries) => {
  try {
    for (const e of entries) {
      const t = tracked.get(e.target as HTMLImageElement);
      if (!t) continue;
      // An image skipped for being tiny may have grown into range.
      if (t.skipSmall) {
        t.skipSmall = false;
        t.evaluated = null;
        maybeQueue(t);
      }
    }
    scheduleRealign();
  } catch {
    stats.errors++;
  }
});

const mo = new MutationObserver((records) => {
  try {
    let removed = false;
    for (const rec of records) {
      if (rec.type === "attributes") {
        const el = rec.target;
        let img: HTMLImageElement | null = null;
        if (el instanceof HTMLImageElement) img = el;
        else if (el instanceof HTMLSourceElement) {
          img = el.parentElement?.querySelector("img") ?? null;
        }
        if (img) onImgAttr(img);
        continue;
      }
      for (const node of rec.addedNodes) scan(node);
      if (rec.removedNodes.length) removed = true;
    }
    if (removed) scheduleSweep();
  } catch {
    stats.errors++;
  }
});

let sweepTimer: number | undefined;
function scheduleSweep(): void {
  if (sweepTimer !== undefined) return;
  sweepTimer = setTimeout(() => {
    sweepTimer = undefined;
    sweep();
  }, 2000) as unknown as number;
}

/** Drop detached elements; rescan the document as an SPA-navigation backstop. */
function sweep(): void {
  try {
    for (const img of tracked.keys()) {
      if (!img.isConnected) untrack(img);
    }
    for (const video of vtracked.keys()) {
      if (!video.isConnected) untrackVideo(video);
    }
    scan(document);
  } catch {
    stats.errors++;
  }
}

/** currentSrc resolves asynchronously for srcset/lazy images; re-evaluate on load. */
function onLoad(e: Event): void {
  try {
    const img = e.target;
    if (!(img instanceof HTMLImageElement)) return;
    let t = tracked.get(img);
    if (!t) {
      track(img);
      t = tracked.get(img);
    }
    if (!t) return;
    const url = img.currentSrc || img.src || "";
    if (url && url !== t.evaluated) {
      t.evaluated = null;
      maybeQueue(t);
    }
  } catch {
    stats.errors++;
  }
}

/* ------------------------------------------------------------------ */
/* State sync                                                          */
/* ------------------------------------------------------------------ */

function resetAll(): void {
  queue.length = 0;
  // Cached judgements belong to the previous blocklist; drop them all.
  resultByUrl.clear();
  for (const t of tracked.values()) {
    t.token++;
    if (t.evalTimer !== undefined) clearTimeout(t.evalTimer);
    t.evalTimer = undefined;
    clearMasks(t);
    t.evaluated = null;
    t.skipSmall = false;
  }
  if (enabled) {
    for (const t of tracked.values()) maybeQueue(t);
  }
  resetVideos();
}

function onMessage(message: unknown): void {
  try {
    const m = message as {
      target?: unknown;
      type?: unknown;
      enabled?: unknown;
      revision?: unknown;
    };
    if (!m || m.target !== "content" || m.type !== "STATE_CHANGED") return;
    enabled = m.enabled === true;
    revision = typeof m.revision === "number" ? m.revision : revision + 1;
    resetAll();
  } catch {
    stats.errors++;
  }
}

async function init(): Promise<void> {
  const res = await send({ target: "background", type: "GET_STATE" });
  const state =
    res && res.ok === true
      ? (res.state as { enabled?: unknown; revision?: unknown } | undefined)
      : undefined;
  enabled = state?.enabled === true;
  revision = typeof state?.revision === "number" ? state.revision : 0;
  if (enabled) {
    for (const t of tracked.values()) maybeQueue(t);
  }
}

/* ------------------------------------------------------------------ */
/* Boot                                                                */
/* ------------------------------------------------------------------ */

function boot(): void {
  console.info(
    "[faceblock] active — faces are masked only after on-device analysis; " +
      "a matched face may be briefly visible on first sight. Counters: globalThis.__faceblockStats",
  );
  mo.observe(document, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["src", "srcset", "sizes"],
  });
  document.addEventListener("load", onLoad, true);
  window.addEventListener("scroll", scheduleRealign, { capture: true, passive: true });
  window.addEventListener("resize", scheduleRealign, { passive: true });
  runtime!.onMessage.addListener(onMessage);
  document.addEventListener("visibilitychange", onVisibilityChange);
  setInterval(sweep, SWEEP_MS);
  setInterval(realign, REALIGN_MS);
  scan(document);
  void init();
}

if (runtime) {
  try {
    boot();
  } catch {
    stats.errors++;
  }
}
