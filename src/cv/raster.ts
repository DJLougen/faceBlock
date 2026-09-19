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
 * Grab the current video frame as a downscaled JPEG for on-device analysis.
 *
 * Returns null — never throws — when the frame cannot be sampled: the video
 * has no drawable data yet (readyState < HAVE_CURRENT_DATA), has zero or
 * non-finite dimensions, or encoding fails. The most likely failure in the
 * wild is a cross-origin <video> without CORS: drawImage() then taints the
 * canvas and convertToBlob/toBlob throws SecurityError. The caller treats
 * that video as unanalysable. Sampling never seeks, pauses, or otherwise
 * disturbs playback.
 *
 * The frame is drawn, encoded, and discarded — the canvas and pixels are
 * never retained or transmitted (plan §14).
 */
export async function sampleVideoFrame(
  video: HTMLVideoElement,
  maxWidth: number,
): Promise<{ jpeg: ArrayBuffer; width: number; height: number } | null> {
  if (video.readyState < 2) return null; // HAVE_CURRENT_DATA
  const srcW = video.videoWidth;
  const srcH = video.videoHeight;
  if (!Number.isFinite(srcW) || !Number.isFinite(srcH) || srcW <= 0 || srcH <= 0) {
    return null;
  }
  // Downscale only — a small source is analysed at native size.
  const scale = Math.min(1, maxWidth / srcW);
  const width = Math.max(1, Math.round(srcW * scale));
  const height = Math.max(1, Math.round(srcH * scale));
  try {
    if (typeof OffscreenCanvas !== "undefined") {
      const canvas = new OffscreenCanvas(width, height);
      const ctx = canvas.getContext("2d");
      if (!ctx) return null;
      ctx.drawImage(video, 0, 0, width, height);
      const blob = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.7 });
      return { jpeg: await blob.arrayBuffer(), width, height };
    }
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, width, height);
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.7),
    );
    if (!blob) return null;
    return { jpeg: await blob.arrayBuffer(), width, height };
  } catch {
    // SecurityError from a tainted (cross-origin) canvas lands here.
    return null;
  }
}
