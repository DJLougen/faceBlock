import type { Raster } from "../shared/types.ts";

export function imageToRaster(img: HTMLImageElement): Raster {
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("2d canvas unavailable");
  ctx.drawImage(img, 0, 0);
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return { width: canvas.width, height: canvas.height, data };
}

/**
 * Result of sampling: a failure is either RECOVERABLE (the video simply was
 * not ready yet — try again later) or PERMANENT (the canvas is tainted, so the
 * pixels can never be read). Conflating the two permanently disables video
 * whose very first sample happened to land too early.
 */
export type VideoFrameSample =
  | { ok: true; jpeg: ArrayBuffer; width: number; height: number }
  | { ok: false; reason: "not-ready" | "tainted" | "unsupported" };

/**
 * Grab the current video frame as a downscaled JPEG for on-device analysis.
 *
 * Never throws. Sampling never seeks, pauses, or otherwise disturbs playback.
 *
 * The most likely permanent failure in the wild is a cross-origin <video>
 * without CORS: drawImage() taints the canvas and encoding then throws
 * SecurityError. That is reported as "tainted" so the caller can stop trying.
 * A video that merely has no drawable frame yet is "not-ready" — recoverable.
 *
 * The frame is drawn, encoded, and discarded — the canvas and pixels are
 * never retained or transmitted.
 */
export async function sampleVideoFrame(
  video: HTMLVideoElement,
  maxWidth: number,
): Promise<VideoFrameSample> {
  if (video.readyState < 2) return { ok: false, reason: "not-ready" }; // HAVE_CURRENT_DATA
  const srcW = video.videoWidth;
  const srcH = video.videoHeight;
  if (!Number.isFinite(srcW) || !Number.isFinite(srcH) || srcW <= 0 || srcH <= 0) {
    return { ok: false, reason: "not-ready" };
  }
  // Downscale only — a small source is analysed at native size.
  const scale = Math.min(1, maxWidth / srcW);
  const width = Math.max(1, Math.round(srcW * scale));
  const height = Math.max(1, Math.round(srcH * scale));
  try {
    if (typeof OffscreenCanvas !== "undefined") {
      const canvas = new OffscreenCanvas(width, height);
      const ctx = canvas.getContext("2d");
      if (!ctx) return { ok: false, reason: "unsupported" };
      ctx.drawImage(video, 0, 0, width, height);
      const blob = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.7 });
      return { ok: true, jpeg: await blob.arrayBuffer(), width, height };
    }
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return { ok: false, reason: "unsupported" };
    ctx.drawImage(video, 0, 0, width, height);
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.7),
    );
    if (!blob) return { ok: false, reason: "unsupported" };
    return { ok: true, jpeg: await blob.arrayBuffer(), width, height };
  } catch (error) {
    // SecurityError from a tainted (cross-origin) canvas lands here and is
    // permanent; anything else is treated as recoverable.
    if (error instanceof DOMException && error.name === "SecurityError") {
      return { ok: false, reason: "tainted" };
    }
    return { ok: false, reason: "not-ready" };
  }
}

/**
 * Rasterise only the neighbourhood of a box, not the whole image.
 *
 * Alignment reads a 112x112 chip derived from the five landmarks, so it needs
 * nothing outside the face. Rasterising a 2400x2400 photo pulled ~23 MB of
 * pixels through getImageData to feed a 112px crop — measurably ~40 ms of the
 * per-image cost, and it scales with the photo rather than the face.
 *
 * Returns the raster plus the offset to subtract from any full-image
 * coordinates before using them against it.
 */
export function imageToRasterRegion(
  img: HTMLImageElement,
  box: { x: number; y: number; width: number; height: number },
  pad = 0.6,
): { raster: Raster; offsetX: number; offsetY: number } {
  const iw = img.naturalWidth;
  const ih = img.naturalHeight;
  const padX = box.width * pad;
  const padY = box.height * pad;
  const x0 = Math.max(0, Math.floor(box.x - padX));
  const y0 = Math.max(0, Math.floor(box.y - padY));
  const x1 = Math.min(iw, Math.ceil(box.x + box.width + padX));
  const y1 = Math.min(ih, Math.ceil(box.y + box.height + padY));
  const w = Math.max(1, x1 - x0);
  const h = Math.max(1, y1 - y0);
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("2d canvas unavailable");
  ctx.drawImage(img, x0, y0, w, h, 0, 0, w, h);
  const { data } = ctx.getImageData(0, 0, w, h);
  return { raster: { width: w, height: h, data }, offsetX: x0, offsetY: y0 };
}
