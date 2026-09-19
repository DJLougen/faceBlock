import { describe, expect, test } from "bun:test";
import {
  hwcToNchw,
  normalizeRgb,
  rasterToRgb,
  resizeBilinearRgb,
} from "../../src/cv/preprocess.ts";
import type { Raster } from "../../src/shared/types.ts";

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

describe("rasterToRgb", () => {
  test("2x2 red raster drops alpha and yields RGB floats", () => {
    const rgb = rasterToRgb(solidRaster(2, 2, 255, 0, 0));
    expect(rgb.length).toBe(2 * 2 * 3);
    for (let i = 0; i < 4; i++) {
      expect(rgb[i * 3]).toBe(255);
      expect(rgb[i * 3 + 1]).toBe(0);
      expect(rgb[i * 3 + 2]).toBe(0);
    }
  });
});

describe("resizeBilinearRgb", () => {
  test("1x1 white upscaled to 2x2 stays ~255", () => {
    const rgb = rasterToRgb(solidRaster(1, 1, 255, 255, 255));
    const out = resizeBilinearRgb(rgb, 1, 1, 2, 2);
    expect(out.length).toBe(2 * 2 * 3);
    for (let i = 0; i < out.length; i++) expect(out[i]!).toBeCloseTo(255, 4);
  });

  test("2x2 downscale to 1x1 averages the four pixels", () => {
    // Corners: 0, 100, 200, 255 gray.
    const rgb = new Float32Array([0, 0, 0, 100, 100, 100, 200, 200, 200, 255, 255, 255]);
    const out = resizeBilinearRgb(rgb, 2, 2, 1, 1);
    // Center-aligned sample at (0.5, 0.5): equal weights.
    expect(out[0]!).toBeCloseTo((0 + 100 + 200 + 255) / 4, 4);
  });
});

describe("normalizeRgb", () => {
  test("mean 127.5 std 127.5 maps 255 near 1 and does not mutate input", () => {
    const rgb = new Float32Array([0, 127.5, 255]);
    const out = normalizeRgb(rgb, 127.5, 127.5);
    expect(out[2]!).toBeCloseTo(1, 6);
    expect(out[0]!).toBeCloseTo(-1, 6);
    expect(out[1]!).toBeCloseTo(0, 6);
    expect(rgb[2]).toBe(255);
  });
});

describe("hwcToNchw", () => {
  test("planar layout: R plane, then G, then B", () => {
    // 1x2 image: pixel0 = (1,2,3), pixel1 = (4,5,6).
    const hwc = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = hwcToNchw(hwc, 1, 2);
    expect(Array.from(out)).toEqual([1, 4, 2, 5, 3, 6]);
  });
});
