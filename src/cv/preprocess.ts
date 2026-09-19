/** Raster/embedding preprocessing: RGB extraction, resize, normalize, layout. */

import type { Raster } from "../shared/types.ts";

/** RGBA raster -> packed HWC RGB floats in 0..255. Alpha is dropped. */
export function rasterToRgb(raster: Raster): Float32Array {
  const { width, height, data } = raster;
  const out = new Float32Array(width * height * 3);
  const px = width * height;
  for (let i = 0; i < px; i++) {
    const s = i * 4;
    const d = i * 3;
    out[d] = data[s]!;
    out[d + 1] = data[s + 1]!;
    out[d + 2] = data[s + 2]!;
  }
  return out;
}

/** Bilinear resize of HWC RGB floats. Pixel-center aligned, edge-clamped. */
export function resizeBilinearRgb(
  rgb: Float32Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
): Float32Array {
  const out = new Float32Array(dstW * dstH * 3);
  const scaleX = srcW / dstW;
  const scaleY = srcH / dstH;
  for (let dy = 0; dy < dstH; dy++) {
    let sy = (dy + 0.5) * scaleY - 0.5;
    if (sy < 0) sy = 0;
    if (sy > srcH - 1) sy = srcH - 1;
    const y0 = Math.floor(sy);
    const y1 = Math.min(y0 + 1, srcH - 1);
    const fy = sy - y0;
    for (let dx = 0; dx < dstW; dx++) {
      let sx = (dx + 0.5) * scaleX - 0.5;
      if (sx < 0) sx = 0;
      if (sx > srcW - 1) sx = srcW - 1;
      const x0 = Math.floor(sx);
      const x1 = Math.min(x0 + 1, srcW - 1);
      const fx = sx - x0;
      const w00 = (1 - fx) * (1 - fy);
      const w01 = fx * (1 - fy);
      const w10 = (1 - fx) * fy;
      const w11 = fx * fy;
      const i00 = (y0 * srcW + x0) * 3;
      const i01 = (y0 * srcW + x1) * 3;
      const i10 = (y1 * srcW + x0) * 3;
      const i11 = (y1 * srcW + x1) * 3;
      const d = (dy * dstW + dx) * 3;
      for (let c = 0; c < 3; c++) {
        out[d + c] =
          rgb[i00 + c]! * w00 +
          rgb[i01 + c]! * w01 +
          rgb[i10 + c]! * w10 +
          rgb[i11 + c]! * w11;
      }
    }
  }
  return out;
}

/** (x - mean) / std into a new array; input is not mutated. */
export function normalizeRgb(rgb: Float32Array, mean: number, std: number): Float32Array {
  const out = new Float32Array(rgb.length);
  const inv = std === 0 ? 1 : 1 / std;
  for (let i = 0; i < rgb.length; i++) out[i] = (rgb[i]! - mean) * inv;
  return out;
}

/** HWC RGB -> planar NCHW (R plane, then G, then B). */
export function hwcToNchw(hwc: Float32Array, h: number, w: number): Float32Array {
  const out = new Float32Array(h * w * 3);
  const plane = h * w;
  for (let i = 0; i < plane; i++) {
    const s = i * 3;
    out[i] = hwc[s]!;
    out[plane + i] = hwc[s + 1]!;
    out[2 * plane + i] = hwc[s + 2]!;
  }
  return out;
}
