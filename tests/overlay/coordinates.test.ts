import { describe, expect, test } from "bun:test";
import { expandBox, mapSourceBoxToRendered } from "../../src/overlay/coordinates.ts";
import { BOX_MARGIN_X, BOX_MARGIN_Y, BOX_SCALE_X, BOX_SCALE_Y } from "../../src/shared/config.ts";
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
      { x: 10, y: 10, width: 20, height: 20 },
      { width: 100, height: 100 },
      { width: 200, height: 100 },
      "fill"
    );
    expectBox(out, { x: 20, y: 10, width: 40, height: 20 });
  });

  test("contain letterboxes with centered offset", () => {
    // source 100x50 into 100x100: scale = min(1, 2) = 1, y offset = 25
    const out = mapSourceBoxToRendered(
      { x: 10, y: 10, width: 20, height: 20 },
      { width: 100, height: 50 },
      { width: 100, height: 100 },
      "contain"
    );
    expectBox(out, { x: 10, y: 35, width: 20, height: 20 });
  });

  test("cover crops with centered offset", () => {
    // source 100x50 into 100x100: scale = max(1, 2) = 2, y offset = 0
    const out = mapSourceBoxToRendered(
      { x: 10, y: 10, width: 20, height: 20 },
      { width: 100, height: 50 },
      { width: 100, height: 100 },
      "cover"
    );
    expectBox(out, { x: -30, y: 20, width: 40, height: 40 });
  });

  test("contain and cover produce different y offsets on mismatched aspect", () => {
    const box: Box = { x: 10, y: 10, width: 20, height: 20 };
    const source = { width: 100, height: 50 };
    const rendered = { width: 100, height: 100 };
    const contain = mapSourceBoxToRendered(box, source, rendered, "contain");
    const cover = mapSourceBoxToRendered(box, source, rendered, "cover");
    expect(contain.y).not.toBeCloseTo(cover.y, 6);
    expect(contain.y).toBeGreaterThan(cover.y);
  });

  test("none centers the unscaled image", () => {
    const out = mapSourceBoxToRendered(
      { x: 10, y: 10, width: 20, height: 20 },
      { width: 100, height: 50 },
      { width: 200, height: 150 },
      "none"
    );
    // offset = ((200-100)/2, (150-50)/2) = (50, 50)
    expectBox(out, { x: 60, y: 60, width: 20, height: 20 });
  });

  test("scale-down acts as contain when source is larger", () => {
    const box: Box = { x: 10, y: 10, width: 20, height: 20 };
    const source = { width: 200, height: 100 };
    const rendered = { width: 100, height: 100 };
    const scaled = mapSourceBoxToRendered(box, source, rendered, "scale-down");
    const contain = mapSourceBoxToRendered(box, source, rendered, "contain");
    expectBox(scaled, contain);
  });

  test("scale-down acts as none when source fits", () => {
    const box: Box = { x: 10, y: 10, width: 20, height: 20 };
    const source = { width: 50, height: 50 };
    const rendered = { width: 100, height: 100 };
    const scaled = mapSourceBoxToRendered(box, source, rendered, "scale-down");
    const none = mapSourceBoxToRendered(box, source, rendered, "none");
    expectBox(scaled, none);
  });

  test("zero source size returns a zero box without NaN", () => {
    const out = mapSourceBoxToRendered(
      { x: 10, y: 10, width: 20, height: 20 },
      { width: 0, height: 0 },
      { width: 200, height: 100 },
      "cover"
    );
    expectBox(out, { x: 0, y: 0, width: 0, height: 0 });
    expect(Number.isNaN(out.x)).toBe(false);
    expect(Number.isNaN(out.y)).toBe(false);
    expect(Number.isNaN(out.width)).toBe(false);
    expect(Number.isNaN(out.height)).toBe(false);
  });
});

describe("expandBox", () => {
  test("margin and scale grow the box", () => {
    const out = expandBox(
      { x: 10, y: 10, width: 20, height: 20 },
      { marginX: 0.1, marginY: 0.1, scaleX: 1.2, scaleY: 1.2 }
    );
    // x = 10 - 20*0.1 = 8, w = 20*1.2 = 24
    expectBox(out, { x: 8, y: 8, width: 24, height: 24 });
    expect(out.width).toBeGreaterThan(20);
    expect(out.height).toBeGreaterThan(20);
  });

  test("config defaults expand asymmetrically", () => {
    const out = expandBox(
      { x: 10, y: 10, width: 20, height: 20 },
      { marginX: BOX_MARGIN_X, marginY: BOX_MARGIN_Y, scaleX: BOX_SCALE_X, scaleY: BOX_SCALE_Y }
    );
    // x = 10 - 20*0.15 = 7, y = 10 - 20*0.2 = 6, w = 24, h = 25
    expectBox(out, { x: 7, y: 6, width: 24, height: 25 });
  });

  test("clamp keeps the box inside the canvas", () => {
    const out = expandBox(
      { x: 2, y: 2, width: 20, height: 20 },
      { marginX: 0.5, marginY: 0.5, scaleX: 2, scaleY: 2 },
      { width: 30, height: 30 }
    );
    // unclamped: x=-8, y=-8, w=40, h=40 -> clamped to [0,0,30,30]
    expectBox(out, { x: 0, y: 0, width: 30, height: 30 });
    expect(out.x).toBeGreaterThanOrEqual(0);
    expect(out.y).toBeGreaterThanOrEqual(0);
    expect(out.x + out.width).toBeLessThanOrEqual(30);
    expect(out.y + out.height).toBeLessThanOrEqual(30);
  });

  test("clamp never produces negative width/height", () => {
    const out = expandBox(
      { x: -50, y: -50, width: 10, height: 10 },
      { marginX: 0, marginY: 0, scaleX: 1, scaleY: 1 },
      { width: 30, height: 30 }
    );
    expect(out.width).toBe(0);
    expect(out.height).toBe(0);
  });
});
