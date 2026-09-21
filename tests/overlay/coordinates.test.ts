import { describe, expect, test } from "bun:test";
import {
  expandBox,
  expandMaskOverlay,
  mapSourceBoxToRendered,
} from "../../src/overlay/coordinates.ts";
import {
  MASK_OVERLAY_PADDING_BOTTOM,
  MASK_OVERLAY_PADDING_TOP,
  MASK_OVERLAY_PADDING_X,
} from "../../src/overlay/coordinates.ts";
import type { Box } from "../../src/shared/types.ts";

function expectBox(actual: Box, expected: Box, digits = 6): void {
  expect(actual.x).toBeCloseTo(expected.x, digits);
  expect(actual.y).toBeCloseTo(expected.y, digits);
  expect(actual.width).toBeCloseTo(expected.width, digits);
  expect(actual.height).toBeCloseTo(expected.height, digits);
}

describe("mapSourceBoxToRendered", () => {
  test("fill scales x and y independently", () => {
    const out = mapSourceBoxToRendered(
      { x: 10, y: 20, width: 30, height: 40 },
      { width: 100, height: 100 },
      { width: 200, height: 50 },
      "fill",
    );
    expectBox(out, { x: 20, y: 10, width: 60, height: 20 });
  });

  test("contain letterboxes with centered offset", () => {
    // source 100x50 into 100x100: scale = min(1, 2) = 1, y offset = 25
    const out = mapSourceBoxToRendered(
      { x: 10, y: 10, width: 20, height: 20 },
      { width: 100, height: 50 },
      { width: 100, height: 100 },
      "contain",
    );
    expectBox(out, { x: 10, y: 35, width: 20, height: 20 });
  });

  test("cover crops with centered offset", () => {
    // source 100x50 into 100x100: scale = max(1, 2) = 2, y offset = 0
    const out = mapSourceBoxToRendered(
      { x: 10, y: 10, width: 20, height: 20 },
      { width: 100, height: 50 },
      { width: 100, height: 100 },
      "cover",
    );
    expectBox(out, { x: -30, y: 20, width: 40, height: 40 });
  });

  test("contain and cover produce different y offsets on mismatched aspect", () => {
    const source = { width: 160, height: 90 };
    const rendered = { width: 320, height: 240 };
    const region = { x: 10, y: 10, width: 40, height: 40 };
    const contain = mapSourceBoxToRendered(region, source, rendered, "contain");
    const cover = mapSourceBoxToRendered(region, source, rendered, "cover");
    expect(contain.y).not.toBeCloseTo(cover.y, 3);
  });

  test("none centers the unscaled image", () => {
    const out = mapSourceBoxToRendered(
      { x: 5, y: 5, width: 20, height: 10 },
      { width: 20, height: 10 },
      { width: 100, height: 100 },
      "none",
    );
    expectBox(out, { x: 45, y: 50, width: 20, height: 10 });
  });

  test("scale-down acts as contain when source is larger", () => {
    const out = mapSourceBoxToRendered(
      { x: 0, y: 0, width: 200, height: 100 },
      { width: 200, height: 100 },
      { width: 100, height: 100 },
      "scale-down",
    );
    expect(out.width).toBe(100);
    expect(out.height).toBe(50);
  });

  test("scale-down acts as none when source fits", () => {
    const out = mapSourceBoxToRendered(
      { x: 2, y: 3, width: 20, height: 10 },
      { width: 20, height: 10 },
      { width: 100, height: 100 },
      "scale-down",
    );
    expectBox(out, { x: 42, y: 48, width: 20, height: 10 });
  });

  test("zero source size returns a zero box without NaN", () => {
    const out = mapSourceBoxToRendered(
      { x: 10, y: 10, width: 20, height: 20 },
      { width: 0, height: 100 },
      { width: 200, height: 200 },
      "contain",
    );
    expect(out.width).toBe(0);
    expect(out.height).toBe(0);
    expect(Number.isNaN(out.width)).toBe(false);
    expect(Number.isNaN(out.height)).toBe(false);
  });
});

describe("expandMaskOverlay", () => {
  test("adds more padding below the face than the old margin+scale defaults", () => {
    const face = { x: 10, y: 10, width: 20, height: 20 };
    const out = expandMaskOverlay(face);
    const oldBottom = face.y + face.height * (1.25 - 0.2);
    const newBottom = out.y + out.height;
    expect(newBottom - (face.y + face.height)).toBeGreaterThan(oldBottom - (face.y + face.height));
  });

  test("overlay padding expands asymmetrically for hair and beard for hair and beard", () => {
    const out = expandMaskOverlay({ x: 10, y: 10, width: 20, height: 20 });
    expectBox(out, {
      x: 10 - 20 * MASK_OVERLAY_PADDING_X,
      y: 10 - 20 * MASK_OVERLAY_PADDING_TOP,
      width: 20 * (1 + 2 * MASK_OVERLAY_PADDING_X),
      height: 20 * (1 + MASK_OVERLAY_PADDING_TOP + MASK_OVERLAY_PADDING_BOTTOM),
    });
  });

  test("clamp keeps the box inside the canvas", () => {
    const out = expandMaskOverlay(
      { x: 2, y: 2, width: 20, height: 20 },
      { width: 30, height: 30 },
    );
    expectBox(out, { x: 0, y: 0, width: 25.6, height: 28.4 });
    expect(out.x).toBeGreaterThanOrEqual(0);
    expect(out.y).toBeGreaterThanOrEqual(0);
    expect(out.x + out.width).toBeLessThanOrEqual(30);
    expect(out.y + out.height).toBeLessThanOrEqual(30);
  });
});

describe("expandBox", () => {
  test("margin and scale grow the box", () => {
    const out = expandBox(
      { x: 10, y: 10, width: 20, height: 20 },
      { marginX: 0.1, marginY: 0.1, scaleX: 1.2, scaleY: 1.2 },
    );
    expectBox(out, { x: 8, y: 8, width: 24, height: 24 });
    expect(out.width).toBeGreaterThan(20);
    expect(out.height).toBeGreaterThan(20);
  });

  test("clamp never produces negative width/height", () => {
    const out = expandBox(
      { x: -50, y: -50, width: 10, height: 10 },
      { marginX: 0, marginY: 0, scaleX: 1, scaleY: 1 },
      { width: 30, height: 30 },
    );
    expect(out.width).toBe(0);
    expect(out.height).toBe(0);
  });
});
