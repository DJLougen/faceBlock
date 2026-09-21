/** Face alignment: canonical 5-point template + 2D similarity warp to 112x112. */

import { ALIGN_SIZE } from "../shared/config.ts";
import type { FaceDetection, Point, Raster, SimilarityTransform } from "../shared/types.ts";
import { rasterToRgb, resizeBilinearRgb } from "./preprocess.ts";

/** ArcFace-style canonical 5-point template on a 112x112 crop, in pixels. */
export const CANONICAL_5: readonly Point[] = [
  { x: 38.2946, y: 51.6963 }, // left eye
  { x: 73.5318, y: 51.5014 }, // right eye
  { x: 56.0252, y: 71.7366 }, // nose
  { x: 41.5493, y: 92.3655 }, // left mouth
  { x: 70.7299, y: 92.2041 }, // right mouth
];

const IDENTITY: SimilarityTransform = { a: 1, b: 0, tx: 0, ty: 0 };

/**
 * Least-squares 2D similarity (Umeyama) mapping src -> dst:
 * [a, -b, tx; b, a, ty]. Needs >=2 points; fewer returns identity.
 */
export function estimateSimilarity(
  src: readonly Point[],
  dst: readonly Point[],
): SimilarityTransform {
  const n = Math.min(src.length, dst.length);
  if (n < 2) return { ...IDENTITY };

  let msx = 0;
  let msy = 0;
  let mdx = 0;
  let mdy = 0;
  for (let i = 0; i < n; i++) {
    msx += src[i]!.x;
    msy += src[i]!.y;
    mdx += dst[i]!.x;
    mdy += dst[i]!.y;
  }
  msx /= n;
  msy /= n;
  mdx /= n;
  mdy /= n;

  let denom = 0;
  let numA = 0;
  let numB = 0;
  for (let i = 0; i < n; i++) {
    const ux = src[i]!.x - msx;
    const uy = src[i]!.y - msy;
    const vx = dst[i]!.x - mdx;
    const vy = dst[i]!.y - mdy;
    denom += ux * ux + uy * uy;
    numA += ux * vx + uy * vy;
    numB += ux * vy - uy * vx;
  }
  if (denom === 0) {
    // Degenerate source: recover translation only.
    return { a: 1, b: 0, tx: mdx - msx, ty: mdy - msy };
  }
  const a = numA / denom;
  const b = numB / denom;
  return {
    a,
    b,
    tx: mdx - (a * msx - b * msy),
    ty: mdy - (b * msx + a * msy),
  };
}

/** x' = a*x - b*y + tx; y' = b*x + a*y + ty. */
export function applySimilarity(p: Point, t: SimilarityTransform): Point {
  return {
    x: t.a * p.x - t.b * p.y + t.tx,
    y: t.b * p.x + t.a * p.y + t.ty,
  };
}

/**
 * Warp HWC RGB by a similarity transform. Destination pixels are inverse-mapped
 * to source coordinates and bilinearly sampled; out-of-bounds texels are 0.
 */
export function warpSimilarityRgb(
  rgb: Float32Array,
  srcW: number,
  srcH: number,
  t: SimilarityTransform,
  dstW: number,
  dstH: number,
): Float32Array {
  const out = new Float32Array(dstW * dstH * 3);
  const det = t.a * t.a + t.b * t.b;
  if (det === 0) return out;
  const ia = t.a / det;
  const ib = -t.b / det;
  // Inverse of [a,-b,tx; b,a,ty] is [ia,-ib,itx; ib,ia,ity].
  const itx = -(ia * t.tx - ib * t.ty);
  const ity = -(ib * t.tx + ia * t.ty);

  for (let dy = 0; dy < dstH; dy++) {
    for (let dx = 0; dx < dstW; dx++) {
      const sx = ia * dx - ib * dy + itx;
      const sy = ib * dx + ia * dy + ity;
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const fx = sx - x0;
      const fy = sy - y0;
      const d = (dy * dstW + dx) * 3;
      for (let c = 0; c < 3; c++) {
        let acc = 0;
        // Zero-padding: each tap contributes only if in bounds.
        if (y0 >= 0 && y0 < srcH) {
          if (x0 >= 0 && x0 < srcW) acc += rgb[(y0 * srcW + x0) * 3 + c]! * (1 - fx) * (1 - fy);
          if (x0 + 1 >= 0 && x0 + 1 < srcW)
            acc += rgb[(y0 * srcW + x0 + 1) * 3 + c]! * fx * (1 - fy);
        }
        if (y0 + 1 >= 0 && y0 + 1 < srcH) {
          if (x0 >= 0 && x0 < srcW) acc += rgb[((y0 + 1) * srcW + x0) * 3 + c]! * (1 - fx) * fy;
          if (x0 + 1 >= 0 && x0 + 1 < srcW)
            acc += rgb[((y0 + 1) * srcW + x0 + 1) * 3 + c]! * fx * fy;
        }
        out[d + c] = acc;
      }
    }
  }
  return out;
}
/**
 * The source-image rectangle alignFace actually reads for a detection.
 *
 * With >=5 landmarks the warp inverse-maps every destination texel of the
 * outSize square back into the source; each sampled source point then reads
 * texels floor(s)..floor(s)+1. The footprint is therefore the bounding box
 * of the inverse-mapped destination corners expanded by ONE SOURCE PIXEL —
 * a margin in destination space would be too small when the transform
 * upscales a small face (measured: a 0.05-scale face diverged by >100 per
 * channel with a dest-space margin).
 *
 * Without landmarks the fallback reads only the detection box expanded 20%
 * about its center.
 *
 * Rasterising exactly this rect (clamped to the image) and offsetting the
 * detection by the rect's origin produces the same aligned chip as running
 * alignFace on the full raster — every texel the warp can sample is inside
 * the region, and out-of-image texels are zero in both cases.
 */
export function alignmentSourceRect(
  detection: FaceDetection,
  outSize = ALIGN_SIZE,
): { x: number; y: number; width: number; height: number } {
  const landmarks = detection.landmarks;
  if (landmarks !== undefined && landmarks.length >= 5) {
    const scale = outSize / 112;
    const dst = CANONICAL_5.map((p) => ({ x: p.x * scale, y: p.y * scale }));
    const t = estimateSimilarity(landmarks.slice(0, 5), dst);
    const det = t.a * t.a + t.b * t.b;
    if (det === 0) {
      // Degenerate transform: the warp emits zeros, so the footprint is empty.
      return { x: 0, y: 0, width: 0, height: 0 };
    }
    const ia = t.a / det;
    const ib = -t.b / det;
    const itx = -(ia * t.tx - ib * t.ty);
    const ity = -(ib * t.tx + ia * t.ty);
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    // Map the actual destination texel range [0, outSize-1], then widen by
    // one SOURCE texel on each side: the bilinear taps read floor(s) and
    // floor(s)+1, and the extra pixel also absorbs float32 rounding.
    for (const dy of [0, outSize - 1]) {
      for (const dx of [0, outSize - 1]) {
        const sx = ia * dx - ib * dy + itx;
        const sy = ib * dx + ia * dy + ity;
        if (sx < minX) minX = sx;
        if (sx > maxX) maxX = sx;
        if (sy < minY) minY = sy;
        if (sy > maxY) maxY = sy;
      }
    }
    const x0 = Math.floor(minX) - 1;
    const y0 = Math.floor(minY) - 1;
    return {
      x: x0,
      y: y0,
      width: Math.floor(maxX) + 3 - x0,
      height: Math.floor(maxY) + 3 - y0,
    };
  }

  const { box } = detection;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  const ew = box.width * 1.2;
  const eh = box.height * 1.2;
  return {
    x: Math.floor(cx - ew / 2),
    y: Math.floor(cy - eh / 2),
    width: Math.ceil(cx + ew / 2) - Math.floor(cx - ew / 2),
    height: Math.ceil(cy + eh / 2) - Math.floor(cy - eh / 2),
  };
}


/**
 * Align a detected face to outSize x outSize HWC RGB floats.
 * With >=5 landmarks: similarity-warp the first 5 onto CANONICAL_5 scaled to
 * outSize/112. Without landmarks: crop the box expanded 20% and resize.
 */
export function alignFace(
  raster: Raster,
  detection: FaceDetection,
  outSize = ALIGN_SIZE,
): Float32Array {
  const rgb = rasterToRgb(raster);
  const landmarks = detection.landmarks;
  if (landmarks !== undefined && landmarks.length >= 5) {
    const scale = outSize / 112;
    const dst = CANONICAL_5.map((p) => ({ x: p.x * scale, y: p.y * scale }));
    const t = estimateSimilarity(landmarks.slice(0, 5), dst);
    return warpSimilarityRgb(rgb, raster.width, raster.height, t, outSize, outSize);
  }

  // Fallback: crop the detection box expanded 20% about its center, then resize.
  const { box } = detection;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  const ew = box.width * 1.2;
  const eh = box.height * 1.2;
  const x0 = Math.max(0, Math.floor(cx - ew / 2));
  const y0 = Math.max(0, Math.floor(cy - eh / 2));
  const x1 = Math.min(raster.width, Math.ceil(cx + ew / 2));
  const y1 = Math.min(raster.height, Math.ceil(cy + eh / 2));
  const cw = x1 - x0;
  const ch = y1 - y0;
  if (cw <= 0 || ch <= 0) return new Float32Array(outSize * outSize * 3);

  const crop = new Float32Array(cw * ch * 3);
  for (let y = 0; y < ch; y++) {
    const srcRow = ((y0 + y) * raster.width + x0) * 3;
    const dstRow = y * cw * 3;
    crop.set(rgb.subarray(srcRow, srcRow + cw * 3), dstRow);
  }
  return resizeBilinearRgb(crop, cw, ch, outSize, outSize);
}
