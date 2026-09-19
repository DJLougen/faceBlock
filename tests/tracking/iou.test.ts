import { describe, expect, test } from "bun:test";
import { applyDetections, coastTrack, createTrack, iou } from "../../src/tracking/iou.ts";
import type { Track } from "../../src/tracking/iou.ts";
import type { Box } from "../../src/shared/types.ts";

function box(x: number, y: number, width: number, height: number): Box {
  return { x, y, width, height };
}

function expectBox(actual: Box, expected: Box, digits = 6): void {
  expect(actual.x).toBeCloseTo(expected.x, digits);
  expect(actual.y).toBeCloseTo(expected.y, digits);
  expect(actual.width).toBeCloseTo(expected.width, digits);
  expect(actual.height).toBeCloseTo(expected.height, digits);
}

describe("iou", () => {
  test("identical boxes score 1", () => {
    expect(iou(box(10, 20, 30, 40), box(10, 20, 30, 40))).toBe(1);
  });

  test("disjoint boxes score 0", () => {
    expect(iou(box(0, 0, 10, 10), box(100, 100, 10, 10))).toBe(0);
    // Touching edges still have zero intersection area.
    expect(iou(box(0, 0, 10, 10), box(10, 0, 10, 10))).toBe(0);
  });

  test("partial overlap returns intersection over union", () => {
    // Intersection 5x10 = 50, union 100 + 100 - 50 = 150.
    expect(iou(box(0, 0, 10, 10), box(5, 0, 10, 10))).toBeCloseTo(50 / 150, 9);
  });

  test("contained box scores area ratio", () => {
    // Inner 10x10 inside outer 20x20: 100 / 400 = 0.25.
    expect(iou(box(0, 0, 20, 20), box(5, 5, 10, 10))).toBeCloseTo(0.25, 9);
    expect(iou(box(5, 5, 10, 10), box(0, 0, 20, 20))).toBeCloseTo(0.25, 9);
  });

  test("zero-area and negative-size boxes score 0, never NaN", () => {
    const good = box(0, 0, 10, 10);
    for (const bad of [box(0, 0, 0, 10), box(0, 0, 10, 0), box(0, 0, 0, 0), box(0, 0, -5, 10)]) {
      expect(iou(bad, good)).toBe(0);
      expect(iou(good, bad)).toBe(0);
      expect(iou(bad, bad)).toBe(0);
    }
  });

  test("result is always finite and within [0, 1]", () => {
    const cases: [Box, Box][] = [
      [box(0, 0, 10, 10), box(0, 0, 10, 10)],
      [box(0, 0, 10, 10), box(20, 20, 10, 10)],
      [box(0, 0, 10, 10), box(9.999, 9.999, 10, 10)],
      [box(-50, -50, 100, 100), box(0, 0, 1, 1)],
      [box(0, 0, 0.001, 0.001), box(0, 0, 1000, 1000)],
    ];
    for (const [a, b] of cases) {
      const v = iou(a, b);
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});

describe("createTrack", () => {
  test("copies the box and initialises motion state", () => {
    const src = box(1, 2, 3, 4);
    const track = createTrack(src, 1000);
    expectBox(track.box, src);
    expect(track.box).not.toBe(src);
    expect(track.vx).toBe(0);
    expect(track.vy).toBe(0);
    expect(track.lastSeenMs).toBe(1000);
    expect(track.misses).toBe(0);
    // Mutating the caller's box must not leak into the track.
    src.x = 999;
    expect(track.box.x).toBe(1);
  });
});

describe("coastTrack", () => {
  test("extrapolates along velocity", () => {
    const track: Track = { box: box(100, 50, 20, 20), vx: 0.5, vy: -0.25, lastSeenMs: 1000, misses: 0 };
    expectBox(coastTrack(track, 1400), box(300, -50, 20, 20));
  });

  test("nowMs earlier than lastSeenMs returns the box unchanged", () => {
    const track: Track = { box: box(100, 50, 20, 20), vx: 0.5, vy: 0.5, lastSeenMs: 1000, misses: 0 };
    expectBox(coastTrack(track, 500), box(100, 50, 20, 20));
  });

  test("does not mutate the track", () => {
    const track: Track = { box: box(0, 0, 10, 10), vx: 1, vy: 1, lastSeenMs: 0, misses: 0 };
    coastTrack(track, 500);
    expectBox(track.box, box(0, 0, 10, 10));
    expect(track.lastSeenMs).toBe(0);
  });
});

describe("applyDetections", () => {
  test("empty inputs return an empty array", () => {
    expect(applyDetections([], [], 0)).toEqual([]);
  });

  test("re-detection at a shifted position updates box and derives velocity", () => {
    const tracks = [createTrack(box(0, 0, 10, 10), 0)];
    // Overlaps the track by enough to clear the IoU threshold: recognition is
    // low rate, so the run must be long enough to associate.
    const next = applyDetections(tracks, [box(5, 0, 10, 10)], 400);
    expect(next).toHaveLength(1);
    const t = next[0]!;
    expectBox(t.box, box(5, 0, 10, 10));
    // Moved +5 x over 400 ms.
    expect(t.vx).toBeCloseTo(0.0125, 9);
    expect(t.vy).toBeCloseTo(0, 9);
    expect(t.lastSeenMs).toBe(400);
    expect(t.misses).toBe(0);
    // The derived velocity carries the mask forward between recognitions.
    expectBox(coastTrack(t, 800), box(10, 0, 10, 10));
  });

  test("a long jump between recognitions associates by centre distance, not IoU", () => {
    // Pure IoU would score this pair 0 and spawn a duplicate track, which is
    // what makes masks flicker during real video motion.
    const tracks = [createTrack(box(0, 0, 100, 100), 0)];
    const next = applyDetections(tracks, [box(120, 0, 100, 100)], 400);
    expect(next).toHaveLength(1);
    expect(iou(box(0, 0, 100, 100), box(120, 0, 100, 100))).toBe(0);
    expectBox(next[0]!.box, box(120, 0, 100, 100));
    expect(next[0]!.vx).toBeCloseTo(0.3, 9);
    expect(next[0]!.misses).toBe(0);
  });

  test("a jump beyond the association radius still starts a new track", () => {
    const tracks = [createTrack(box(0, 0, 100, 100), 0)];
    const next = applyDetections(tracks, [box(500, 500, 100, 100)], 400);
    expect(next).toHaveLength(2);
  });

  test("zero elapsed time keeps the previous velocity", () => {
    const track: Track = { box: box(0, 0, 10, 10), vx: 0.5, vy: -0.5, lastSeenMs: 100, misses: 0 };
    const next = applyDetections([track], [box(5, 5, 10, 10)], 100);
    expect(next[0]!.vx).toBe(0.5);
    expect(next[0]!.vy).toBe(-0.5);
  });

  test("a missed track keeps its slot, counts misses, and coasts via coastTrack", () => {
    const moving: Track = { box: box(0, 0, 10, 10), vx: 0.1, vy: 0, lastSeenMs: 0, misses: 0 };
    const still = createTrack(box(100, 100, 10, 10), 0);
    const det = [box(100, 100, 10, 10)];
    // Miss 1: order preserved, the missed track stays at index 0. Its box is
    // still the last OBSERVED box — position comes from coastTrack, so the
    // consumer's per-frame render must not have motion counted twice.
    const r1 = applyDetections([moving, still], det, 400);
    expect(r1).toHaveLength(2);
    expect(r1[0]!.misses).toBe(1);
    expect(r1[0]!.lastSeenMs).toBe(0);
    expectBox(r1[0]!.box, box(0, 0, 10, 10));
    expectBox(coastTrack(r1[0]!, 400), box(40, 0, 10, 10));
    expect(r1[1]!.misses).toBe(0);
    // Miss 2: still under the default maxMisses = 2, coasting continues.
    const r2 = applyDetections(r1, det, 800);
    expect(r2).toHaveLength(2);
    expect(r2[0]!.misses).toBe(2);
    expectBox(r2[0]!.box, box(0, 0, 10, 10));
    expectBox(coastTrack(r2[0]!, 800), box(80, 0, 10, 10));
    // Miss 3 exceeds maxMisses: the track is dropped.
    const r3 = applyDetections(r2, det, 1200);
    expect(r3).toHaveLength(1);
    expectBox(r3[0]!.box, box(100, 100, 10, 10));
  });

  test("closer pairing wins and neither track steals the other's detection", () => {
    // Detections are passed in REVERSED track order: an implementation that
    // pairs by position instead of IoU would swap them.
    const a = createTrack(box(0, 0, 10, 10), 0);
    const b = createTrack(box(100, 0, 10, 10), 0);
    const dA = box(1, 0, 10, 10);
    const dB = box(99, 0, 10, 10);
    const next = applyDetections([a, b], [dB, dA], 100);
    expect(next).toHaveLength(2);
    expectBox(next[0]!.box, dA);
    expectBox(next[1]!.box, dB);
  });

  test("a contested detection goes to the higher-IoU track", () => {
    // d overlaps both tracks: IoU 6/14 with A, 7/13 with B. Greedy
    // association must give it to B, leaving A with a miss.
    const a = createTrack(box(0, 0, 10, 10), 0);
    const b = createTrack(box(7, 0, 10, 10), 0);
    const d = box(4, 0, 10, 10);
    const next = applyDetections([a, b], [d], 100);
    expect(next).toHaveLength(2);
    expect(next[0]!.misses).toBe(1);
    expectBox(next[0]!.box, box(0, 0, 10, 10));
    expectBox(next[1]!.box, d);
    expect(next[1]!.misses).toBe(0);
  });

  test("a detection far from every track becomes a new track", () => {
    const tracks = [createTrack(box(0, 0, 10, 10), 0)];
    const next = applyDetections(tracks, [box(0, 0, 10, 10), box(500, 500, 20, 20)], 100);
    expect(next).toHaveLength(2);
    expectBox(next[1]!.box, box(500, 500, 20, 20));
    expect(next[1]!.misses).toBe(0);
    expect(next[1]!.lastSeenMs).toBe(100);
  });

  test("maxMisses: 0 removes an unmatched track immediately", () => {
    const tracks = [createTrack(box(0, 0, 10, 10), 0)];
    expect(applyDetections(tracks, [], 100, { maxMisses: 0 })).toEqual([]);
    // The matched track still survives.
    const kept = applyDetections(tracks, [box(0, 0, 10, 10)], 100, { maxMisses: 0 });
    expect(kept).toHaveLength(1);
  });

  test("iouThreshold controls the IoU pass", () => {
    const tracks = [createTrack(box(0, 0, 10, 10), 0)];
    // box(4,4,10,10) against box(0,0,10,10): intersection 6x6 = 36,
    // union = 100 + 100 - 36 = 164, so IoU = 36/164 = 0.2195 — below the 0.3
    // default but above a 0.2 override. The centre fallback is disabled here
    // because it would associate these two regardless of the threshold.
    const det = box(4, 4, 10, 10);
    expect(iou(box(0, 0, 10, 10), det)).toBeCloseTo(0.2195, 3);
    const strict = applyDetections(tracks, [det], 100, { centreFallback: false });
    expect(strict).toHaveLength(2); // unmatched track + new track
    const loose = applyDetections(tracks, [det], 100, {
      iouThreshold: 0.2,
      centreFallback: false,
    });
    expect(loose).toHaveLength(1);
    expectBox(loose[0]!.box, det);
  });

  test("the IoU pass claims the overlapping detection before the fallback runs", () => {
    const tracks = [createTrack(box(0, 0, 10, 10), 0)];
    const overlapping = box(2, 0, 10, 10); // high IoU
    const nearbyButDisjoint = box(20, 0, 10, 10); // IoU 0, inside the radius
    const next = applyDetections(tracks, [nearbyButDisjoint, overlapping], 100);
    expect(next).toHaveLength(2); // the track takes one, the other is new
    expectBox(next[0]!.box, overlapping);
  });

  test("inputs are never mutated", () => {
    const tracks: Track[] = [
      { box: box(0, 0, 10, 10), vx: 0.1, vy: 0.2, lastSeenMs: 0, misses: 1 },
      createTrack(box(100, 100, 10, 10), 0),
    ];
    const detections = [box(100, 100, 10, 10), box(500, 500, 5, 5)];
    const tracksSnapshot = JSON.parse(JSON.stringify(tracks)) as unknown;
    const detectionsSnapshot = JSON.parse(JSON.stringify(detections)) as unknown;
    const next = applyDetections(tracks, detections, 400);
    expect(JSON.parse(JSON.stringify(tracks))).toEqual(tracksSnapshot);
    expect(JSON.parse(JSON.stringify(detections))).toEqual(detectionsSnapshot);
    // Returned boxes are fresh objects, not aliases of the inputs.
    expect(next[1]!.box).not.toBe(detections[0]);
    expect(next[0]!.box).not.toBe(tracks[0]!.box);
  });
});
