import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";

// Detector pass-order and merge policy, exercised against the REAL functions
// from src/cv/yunet.ts — not a copied implementation. The inference seams
// (detectAtSize / detectRegion) are stubbed at the sandbox boundary so the
// orchestration itself is what is under test:
//
//   pass 0: fast 320px full-frame
//   pass 1: full 640px, only when pass 0 finds nothing
//   pass 2: 3x3 overlapping tiled pass, only when >= CROWD_FACE_COUNT faces
//   merge:  tiled additions dropped on IoU > 0.4 OR centre containment;
//           partially-visible edge boxes are deliberately kept.
//
// These tests document current policy — including the known asymmetry that
// detectTiled dedups internally by IoU alone while merge() adds containment.

const source = readFileSync(new URL("../../src/cv/yunet.ts", import.meta.url), "utf8");


function constSource(name: string): string {
  const m = source.match(new RegExp(`^const ${name} = [^;]+;`, "m"));
  if (!m) throw new Error(`Missing yunet constant: ${name}`);
  return m[0];
}

interface FaceBox {
  x: number;
  y: number;
  width: number;
  height: number;
}
interface Face {
  box: FaceBox;
  confidence: number;
  landmarks?: { x: number; y: number }[];
}

/** Deterministic fixture face used across pass-policy cases. */
const face = (x: number, y: number, w: number, h: number, confidence = 0.9): Face => ({
  box: { x, y, width: w, height: h },
  confidence,
});

function functionSource(name: string): string {
  const start = source.search(new RegExp(`^(?:export\\s+)?(?:async\\s+)?function ${name}[<(]`, "m"));
  if (start < 0) throw new Error(`Missing yunet function: ${name}`);
  const end = source.indexOf("\n}", start);
  if (end < 0) throw new Error(`Missing function boundary: ${name}`);
  return source.slice(start, end + 2).replace(/^export\s+/, "");
}

/** Build the sandboxed module with scripted detectAtSize/detectRegion. */
function harness(opts: {
  atSize?: (inputSize: number) => Face[];
  region?: (ox: number, oy: number, w: number, h: number) => Face[];
}) {
  const calls = { atSize: [] as number[], region: [] as { ox: number; oy: number; w: number; h: number }[] };
  const code =
    [
      "FAST_INPUT_SIZE",
      "FULL_INPUT_SIZE",
      "MERGE_IOU",
      "TILE_GRID",
      "TILE_OVERLAP",
      "CROWD_FACE_COUNT",
    ]
      .map(constSource)
      .join("\n") +
    "\n" +
    ["overlapIoU", "centerOf", "contains", "merge", "detectFacesYuNet", "detectTiled"]
      .map(functionSource)
      .join("\n") +
    "\nglobalThis.API = { detectFacesYuNet, merge, detectTiled, overlapIoU };";

  const sandbox: Record<string, unknown> = {
    document: { createElement: () => ({}) },
    detectAtSize: async (_d: unknown, _i: unknown, inputSize: number) => {
      calls.atSize.push(inputSize);
      return opts.atSize?.(inputSize) ?? [];
    },
    detectRegion: async (
      _d: unknown,
      _i: unknown,
      ox: number,
      oy: number,
      w: number,
      h: number,
    ) => {
      calls.region.push({ ox, oy, w, h });
      return opts.region?.(ox, oy, w, h) ?? [];
    },
  };
  runInNewContext(new Bun.Transpiler({ loader: "ts", target: "bun" }).transformSync(code), sandbox);
  const api = sandbox.API as {
    detectFacesYuNet: (detector: unknown, image: unknown, opts?: unknown) => Promise<Face[]>;
    merge: (existing: Face[], add: Face[]) => Face[];
    detectTiled: (detector: unknown, image: unknown, inputSize: number, s: number, n: number) => Promise<Face[]>;
    overlapIoU: (a: Face, b: Face) => number;
  };
  return { api, calls };
}

const image = (w: number, h: number) => ({ naturalWidth: w, naturalHeight: h });

describe("detectFacesYuNet pass policy", () => {
  test("runs the fast 320 pass first and stops when it finds a face", async () => {
    const { api, calls } = harness({ atSize: () => [face(10, 10, 50, 50)] });
    const out = await api.detectFacesYuNet({}, image(1000, 800));
    expect(calls.atSize).toEqual([320]);
    expect(calls.region).toHaveLength(0);
    expect(out).toHaveLength(1);
  });

  test("falls back to the full 640 pass only when the fast pass is empty", async () => {
    const { api, calls } = harness({
      atSize: (size) => (size === 640 ? [face(10, 10, 50, 50)] : []),
    });
    const out = await api.detectFacesYuNet({}, image(1000, 800));
    expect(calls.atSize).toEqual([320, 640]);
    expect(out).toHaveLength(1);
  });

  test("empty at both sizes returns empty and never tiles", async () => {
    const { api, calls } = harness({ atSize: () => [] });
    const out = await api.detectFacesYuNet({}, image(1000, 800));
    expect(calls.atSize).toEqual([320, 640]);
    expect(calls.region).toHaveLength(0);
    expect(out).toHaveLength(0);
  });

  test("a single face does not trigger the tiled crowd pass", async () => {
    const { api, calls } = harness({ atSize: () => [face(10, 10, 50, 50)] });
    await api.detectFacesYuNet({}, image(1000, 800));
    expect(calls.region).toHaveLength(0);
  });

  test("two or more faces trigger the tiled pass and merge its additions", async () => {
    const { api, calls } = harness({
      atSize: () => [face(10, 10, 50, 50), face(400, 300, 60, 60)],
      region: (ox, oy) => (ox === 0 && oy === 0 ? [face(700, 600, 40, 40)] : []),
    });
    const out = await api.detectFacesYuNet({}, image(1000, 800));
    expect(calls.atSize).toEqual([320]);
    expect(calls.region.length).toBeGreaterThan(0);
    expect(out).toHaveLength(3);
    expect(out[2]!.box).toEqual({ x: 700, y: 600, width: 40, height: 40 });
  });
});

describe("merge dedup policy", () => {
  test("drops a tiled box overlapping an existing one above the IoU bar", () => {
    const { api } = harness({});
    const out = api.merge([face(100, 100, 80, 80)], [face(105, 100, 80, 80)]);
    expect(out).toHaveLength(1);
  });

  test("drops a sub-region duplicate by centre containment even under the IoU bar", () => {
    const { api } = harness({});
    // Small box sits inside the big one off-centre: IoU ~0.09, but its centre
    // is inside the parent — the same face boxed twice by the tiled pass.
    const out = api.merge([face(100, 100, 200, 200)], [face(240, 240, 60, 60)]);
    expect(out).toHaveLength(1);
  });

  test("drops a parent duplicate when the existing centre is inside the new box", () => {
    const { api } = harness({});
    const out = api.merge([face(240, 240, 60, 60)], [face(100, 100, 200, 200)]);
    expect(out).toHaveLength(1);
    expect(out[0]!.box.x).toBe(240);
  });

  test("keeps partially overlapping boxes when neither centre is inside the other", () => {
    const { api } = harness({});
    // Edge case the policy deliberately keeps: a partially visible face at the
    // frame border overlaps a neighbour but is a different face.
    const out = api.merge([face(0, 100, 80, 80)], [face(60, 100, 80, 80)]);
    expect(out).toHaveLength(2);
  });

  test("preserves existing order and appends survivors", () => {
    const { api } = harness({});
    const a = face(0, 0, 50, 50);
    const b = face(200, 0, 50, 50);
    const c = face(400, 0, 50, 50);
    const out = api.merge([a, b], [face(2, 2, 50, 50), c]);
    expect(out.map((f) => f.box.x)).toEqual([0, 200, 400]);
  });
});

describe("detectTiled", () => {
  test("maps tile-local boxes back to image coordinates", async () => {
    const { api, calls } = harness({
      region: (ox, oy) => (ox === 101 && oy === 75 ? [face(10, 20, 50, 60)] : []),
    });
    const out = await api.detectTiled({}, image(400, 300), 320, 0.5, 0.3);
    expect(calls.region.length).toBeGreaterThan(0);
    const found = out.find((f) => f.box.width === 50);
    expect(found?.box.x).toBe(111);
    expect(found?.box.y).toBe(95);
  });

  test("dedups the same face found in overlapping tiles by IoU", async () => {
    const { api } = harness({
      // Two adjacent tiles both see the same face near their shared seam.
      region: (ox, oy) =>
        oy === 0 && ox === 0
          ? [face(90, 10, 60, 60)]
          : oy === 0 && ox === 101
            ? [face(-11, 10, 60, 60)]
            : [],
    });
    const out = await api.detectTiled({}, image(400, 300), 320, 0.5, 0.3);
    // Both map to ~(90,10,60,60): heavy overlap -> one kept.
    expect(out).toHaveLength(1);
  });

  test("skips tiles smaller than 64px on either axis", async () => {
    const { api, calls } = harness({});
    await api.detectTiled({}, image(200, 200), 320, 0.5, 0.3);
    for (const c of calls.region) {
      expect(c.w).toBeGreaterThanOrEqual(64);
      expect(c.h).toBeGreaterThanOrEqual(64);
    }
    // 200px image: 3 usable columns/rows, the 4th strip is 50px and skipped.
    expect(calls.region).toHaveLength(9);
  });

  test("documents the IoU-only internal dedup: a contained duplicate survives the tiled pass", async () => {
    // LIMITATION DOCUMENTATION: inside detectTiled, dedup is overlapIoU-only.
    // A tile-local sub-box whose centre lands inside another tile's box but
    // with IoU < 0.4 is KEPT here — only merge() removes it afterwards.
    const { api } = harness({
      region: (ox, oy) =>
        oy === 0 && ox === 0
          ? [face(0, 0, 140, 140)]
          : oy === 0 && ox === 101
            ? [face(0, 0, 40, 40)]
            : [],
    });
    const tiled = await api.detectTiled({}, image(400, 300), 320, 0.5, 0.3);
    // (0,0,140,140) and (101,0,40,40): IoU ~0.08, containment only -> both kept.
    expect(tiled).toHaveLength(2);
    // The final merge is what actually drops it.
    const merged = api.merge([], tiled);
    expect(merged).toHaveLength(1);
  });
});
