import { describe, expect, test } from "bun:test";
import { alignFace, alignmentSourceRect, CANONICAL_5 } from "../../src/cv/align.ts";
import { regionRect } from "../../src/cv/raster.ts";
import type { FaceDetection, Raster } from "../../src/shared/types.ts";
/**
 * Equivalence proof for regional alignment: alignFace on a raster cropped to
 * alignmentSourceRect (clamped to the image via regionRect) must produce the
 * same chip as alignFace on the full raster, because the rect provably
 * covers every source texel the warp can sample.
 */

/** Deterministic synthetic photo — gradients plus structure, no RNG. */
function makeRaster(width: number, height: number): Raster {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      data[i] = (x * 7 + y * 3) % 256;
      data[i + 1] = (x * 2 + y * 11) % 256;
      data[i + 2] = (x * x + y) % 256;
      data[i + 3] = 255;
    }
  }
  return { width, height, data };
}

/** Slice a rect out of a raster — mirrors what imageToRasterRegion draws. */
function sliceRaster(
  raster: Raster,
  rect: { x: number; y: number; width: number; height: number },
): Raster {
  const data = new Uint8ClampedArray(rect.width * rect.height * 4);
  for (let y = 0; y < rect.height; y++) {
    const src = ((rect.y + y) * raster.width + rect.x) * 4;
    data.set(raster.data.subarray(src, src + rect.width * 4), y * rect.width * 4);
  }
  return { width: rect.width, height: rect.height, data };
}

function offsetDetection(det: FaceDetection, ox: number, oy: number): FaceDetection {
  return {
    ...det,
    box: { ...det.box, x: det.box.x - ox, y: det.box.y - oy },
    landmarks: det.landmarks?.map((p) => ({ x: p.x - ox, y: p.y - oy })),
  };
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i]! - b[i]!));
  return max;
}

const FACE: FaceDetection = {
  box: { x: 120, y: 60, width: 80, height: 100 },
  confidence: 0.9,
  landmarks: [
    { x: 140, y: 95 },
    { x: 180, y: 95 },
    { x: 160, y: 115 },
    { x: 145, y: 140 },
    { x: 175, y: 140 },
  ],
};

describe("alignmentSourceRect", () => {
  test("covers the detection box and landmarks for a landmarked face", () => {
    const rect = alignmentSourceRect(FACE);
    for (const p of FACE.landmarks!) {
      expect(p.x).toBeGreaterThanOrEqual(rect.x);
      expect(p.x).toBeLessThanOrEqual(rect.x + rect.width);
      expect(p.y).toBeGreaterThanOrEqual(rect.y);
      expect(p.y).toBeLessThanOrEqual(rect.y + rect.height);
    }
    expect(rect.width).toBeGreaterThan(0);
    expect(rect.height).toBeGreaterThan(0);
  });

  test("falls back to the expanded box without landmarks", () => {
    const bare: FaceDetection = { box: { x: 100, y: 50, width: 40, height: 40 }, confidence: 0.8 };
    const rect = alignmentSourceRect(bare);
    // The fallback reads box*1.2 about its center.
    expect(rect.x).toBeLessThanOrEqual(100 - 4);
    expect(rect.x + rect.width).toBeGreaterThanOrEqual(140 + 4);
  });
});

describe("regional alignment equivalence", () => {
  test("alignFace on the region raster matches the full raster (landmarks)", () => {
    const img = { naturalWidth: 320, naturalHeight: 240 };
    const raster = makeRaster(320, 240);
    const rect = regionRect(img, alignmentSourceRect(FACE), 0);
    const region = sliceRaster(raster, rect);
    const full = alignFace(raster, FACE);
    const regional = alignFace(region, offsetDetection(FACE, rect.x, rect.y));
    expect(regional.length).toBe(full.length);
    // Float32 association noise only — the sampled texels are identical.
    expect(maxAbsDiff(full, regional)).toBeLessThan(1e-2);
  });

  test("alignFace on the region raster matches the full raster (box fallback)", () => {
    const bare: FaceDetection = { box: { x: 40, y: 30, width: 50, height: 60 }, confidence: 0.8 };
    const img = { naturalWidth: 320, naturalHeight: 240 };
    const raster = makeRaster(320, 240);
    const rect = regionRect(img, alignmentSourceRect(bare), 0);
    const region = sliceRaster(raster, rect);
    const full = alignFace(raster, bare);
    const regional = alignFace(region, offsetDetection(bare, rect.x, rect.y));
    expect(maxAbsDiff(full, regional)).toBeLessThan(1e-2);
  });

  test("a face at the image edge still matches (clamped region)", () => {
    const edge: FaceDetection = {
      box: { x: 0, y: 0, width: 60, height: 70 },
      confidence: 0.9,
      landmarks: [
        { x: 15, y: 20 },
        { x: 45, y: 20 },
        { x: 30, y: 35 },
        { x: 18, y: 55 },
        { x: 42, y: 55 },
      ],
    };
    const img = { naturalWidth: 320, naturalHeight: 240 };
    const raster = makeRaster(320, 240);
    const rect = regionRect(img, alignmentSourceRect(edge), 0);
    const region = sliceRaster(raster, rect);
    const full = alignFace(raster, edge);
    const regional = alignFace(region, offsetDetection(edge, rect.x, rect.y));
    expect(maxAbsDiff(full, regional)).toBeLessThan(1e-2);
  });
});

describe("regional alignment equivalence — measured regression fixtures", () => {
  // Reproduces the probe that caught the dest-space margin bug: landmarks are
  // CANONICAL_5 scaled by `scale` and offset to (100.1, 100.1) in a 200x200
  // gradient. Small scales upscale hard, so the footprint margin must be in
  // source space.
  for (const scale of [0.05, 0.1, 0.2, 0.5, 1, 2]) {
    test(`landmark scale ${scale} matches full-raster alignment`, () => {
      const img = { naturalWidth: 200, naturalHeight: 200 };
      const raster = makeRaster(200, 200);
      const landmarks = CANONICAL_5.map((p) => ({
        x: p.x * scale + 100.1,
        y: p.y * scale + 100.1,
      }));
      const xs = landmarks.map((p) => p.x);
      const ys = landmarks.map((p) => p.y);
      const det: FaceDetection = {
        box: {
          x: Math.min(...xs) - 4,
          y: Math.min(...ys) - 4,
          width: Math.max(...xs) - Math.min(...xs) + 8,
          height: Math.max(...ys) - Math.min(...ys) + 8,
        },
        confidence: 0.9,
        landmarks,
      };
      const rect = regionRect(img, alignmentSourceRect(det), 0);
      const region = sliceRaster(raster, rect);
      const full = alignFace(raster, det);
      const regional = alignFace(region, offsetDetection(det, rect.x, rect.y));
      expect(regional.length).toBe(full.length);
      expect(maxAbsDiff(full, regional)).toBeLessThan(1e-2);
    });
  }

  test("rotated landmarks at the image border still match", () => {
    const img = { naturalWidth: 200, naturalHeight: 200 };
    const raster = makeRaster(200, 200);
    // Rotate CANONICAL_5 ~30° about its centroid and push it to the corner so
    // the footprint is clamped by the image edge.
    const cx = CANONICAL_5.reduce((s, p) => s + p.x, 0) / 5;
    const cy = CANONICAL_5.reduce((s, p) => s + p.y, 0) / 5;
    const cos = Math.cos(Math.PI / 6);
    const sin = Math.sin(Math.PI / 6);
    const landmarks = CANONICAL_5.map((p) => ({
      x: (p.x - cx) * cos - (p.y - cy) * sin + 8,
      y: (p.x - cx) * sin + (p.y - cy) * cos + 8,
    }));
    const xs = landmarks.map((p) => p.x);
    const ys = landmarks.map((p) => p.y);
    const det: FaceDetection = {
      box: {
        x: Math.min(...xs) - 2,
        y: Math.min(...ys) - 2,
        width: Math.max(...xs) - Math.min(...xs) + 4,
        height: Math.max(...ys) - Math.min(...ys) + 4,
      },
      confidence: 0.9,
      landmarks,
    };
    const rect = regionRect(img, alignmentSourceRect(det), 0);
    const region = sliceRaster(raster, rect);
    const full = alignFace(raster, det);
    const regional = alignFace(region, offsetDetection(det, rect.x, rect.y));
    expect(maxAbsDiff(full, regional)).toBeLessThan(1e-2);
  });
});
