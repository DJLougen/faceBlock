import { describe, expect, test } from "bun:test";
import {
  alignFace,
  applySimilarity,
  CANONICAL_5,
  estimateSimilarity,
  warpSimilarityRgb,
} from "../../src/cv/align.ts";
import type { Point, Raster } from "../../src/shared/types.ts";

function solidRaster(w: number, h: number, r: number, g: number, b: number): Raster {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    data[i * 4] = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
    data[i * 4 + 3] = 255;
  }
  return { width: w, height: h, data };
}

describe("estimateSimilarity", () => {
  test("recovers pure translation", () => {
    const src: Point[] = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 0, y: 10 },
    ];
    const dst = src.map((p) => ({ x: p.x + 5, y: p.y - 3 }));
    const t = estimateSimilarity(src, dst);
    expect(t.a).toBeCloseTo(1, 6);
    expect(t.b).toBeCloseTo(0, 6);
    expect(t.tx).toBeCloseTo(5, 6);
    expect(t.ty).toBeCloseTo(-3, 6);
  });

  test("recovers scale + rotation + translation", () => {
    // True transform: scale 2, rotate 90deg CCW, translate (7, -2).
    const a = 0;
    const b = 2;
    const src: Point[] = [
      { x: 1, y: 2 },
      { x: 4, y: 0 },
      { x: -1, y: 3 },
      { x: 2, y: -2 },
    ];
    const dst = src.map((p) => ({ x: a * p.x - b * p.y + 7, y: b * p.x + a * p.y - 2 }));
    const t = estimateSimilarity(src, dst);
    expect(t.a).toBeCloseTo(a, 6);
    expect(t.b).toBeCloseTo(b, 6);
    expect(t.tx).toBeCloseTo(7, 6);
    expect(t.ty).toBeCloseTo(-2, 6);
  });

  test("fewer than 2 points returns identity", () => {
    const t = estimateSimilarity([{ x: 1, y: 1 }], [{ x: 9, y: 9 }]);
    expect(t).toEqual({ a: 1, b: 0, tx: 0, ty: 0 });
  });
});

describe("applySimilarity", () => {
  test("identity leaves the point unchanged", () => {
    const p = { x: 3.5, y: -7.25 };
    expect(applySimilarity(p, { a: 1, b: 0, tx: 0, ty: 0 })).toEqual(p);
  });

  test("round-trips through the estimated transform", () => {
    const src: Point[] = [
      { x: 0, y: 0 },
      { x: 8, y: 1 },
      { x: 2, y: 9 },
    ];
    const dst = src.map((p) => ({ x: 2 * p.x - p.y + 4, y: p.x + 2 * p.y - 6 }));
    const t = estimateSimilarity(src, dst);
    for (let i = 0; i < src.length; i++) {
      const q = applySimilarity(src[i]!, t);
      expect(q.x).toBeCloseTo(dst[i]!.x, 5);
      expect(q.y).toBeCloseTo(dst[i]!.y, 5);
    }
  });
});

describe("warpSimilarityRgb", () => {
  test("identity warp copies the image", () => {
    const rgb = new Float32Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);
    const out = warpSimilarityRgb(rgb, 2, 2, { a: 1, b: 0, tx: 0, ty: 0 }, 2, 2);
    expect(Array.from(out)).toEqual(Array.from(rgb));
  });

  test("out-of-bounds samples are 0", () => {
    const rgb = new Float32Array([255, 255, 255]);
    const out = warpSimilarityRgb(rgb, 1, 1, { a: 1, b: 0, tx: 5, ty: 0 }, 2, 1);
    // dst x=0 maps to src x=-5 -> 0; dst x=1 maps to src x=-4 -> 0.
    for (const v of out) expect(v).toBe(0);
  });
});

describe("alignFace", () => {
  test("without landmarks returns outSize*outSize*3 via box crop", () => {
    const raster = solidRaster(64, 64, 200, 100, 50);
    const out = alignFace(raster, {
      box: { x: 10, y: 10, width: 20, height: 20 },
      confidence: 0.9,
    });
    expect(out.length).toBe(112 * 112 * 3);
    // Solid-color crop stays solid after resize.
    expect(out[0]!).toBeCloseTo(200, 4);
    expect(out[1]!).toBeCloseTo(100, 4);
    expect(out[2]!).toBeCloseTo(50, 4);
  });

  test("with 5 landmarks warps onto the scaled canonical template", () => {
    const raster = solidRaster(112, 112, 128, 128, 128);
    const landmarks = CANONICAL_5.map((p) => ({ ...p }));
    const out = alignFace(raster, {
      box: { x: 0, y: 0, width: 112, height: 112 },
      confidence: 0.9,
      landmarks,
    });
    expect(out.length).toBe(112 * 112 * 3);
    // Identity alignment of a solid image stays solid in the interior.
    const center = (56 * 112 + 56) * 3;
    expect(out[center]!).toBeCloseTo(128, 4);
  });
});
