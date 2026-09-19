import { describe, expect, test } from "bun:test";
import { decodeYuNet, nms, yunetPriors, type YuNetPrior } from "../../src/cv/yunet.ts";

/** Build per-stride tensors where every cell is background, then fill one cell. */
function blank(priors: YuNetPrior[]) {
  return priors.map((p) => new Float32Array(p.cols * p.rows));
}

function bboxTensors(priors: YuNetPrior[]) {
  return priors.map((p) => new Float32Array(p.cols * p.rows * 4));
}

function kpsTensors(priors: YuNetPrior[]) {
  return priors.map((p) => new Float32Array(p.cols * p.rows * 10));
}

const PRIORS = yunetPriors(64, [8]); // 8x8 grid at stride 8 for readable maths

describe("yunetPriors", () => {
  test("grids are floor(input/stride) per axis", () => {
    expect(yunetPriors(640, [8, 16, 32])).toEqual([
      { stride: 8, cols: 80, rows: 80 },
      { stride: 16, cols: 40, rows: 40 },
      { stride: 32, cols: 20, rows: 20 },
    ]);
  });
});

describe("decodeYuNet", () => {
  test("centre is (grid + offset) * stride and size is exp(w/h) * stride", () => {
    const { stride, cols } = PRIORS[0]!;
    const cls = blank(PRIORS);
    const obj = blank(PRIORS);
    const bbox = bboxTensors(PRIORS);
    const kps = kpsTensors(PRIORS);

    const r = 2;
    const c = 3;
    const cell = r * cols + c;
    cls[0]![cell] = 0.81; // sqrt(0.81 * 1.0) = 0.9
    obj[0]![cell] = 1;
    // offset 0.5 => centre exactly at the cell centre
    bbox[0]![cell * 4] = 0.5;
    bbox[0]![cell * 4 + 1] = 0.5;
    bbox[0]![cell * 4 + 2] = Math.log(2); // exp -> 2 * stride wide
    bbox[0]![cell * 4 + 3] = Math.log(1); // exp -> 1 * stride tall

    const faces = decodeYuNet({ cls, obj, bbox, kps }, PRIORS, 0.5);
    expect(faces).toHaveLength(1);
    const f = faces[0]!;
    expect(f.score).toBeCloseTo(0.9, 5);
    expect(f.box.width).toBeCloseTo(2 * stride, 5);
    expect(f.box.height).toBeCloseTo(stride, 5);
    // Centre = (c + 0.5) * stride, and the box is centred on it.
    expect(f.box.x + f.box.width / 2).toBeCloseTo((c + 0.5) * stride, 5);
    expect(f.box.y + f.box.height / 2).toBeCloseTo((r + 0.5) * stride, 5);
  });

  test("score below threshold is dropped", () => {
    const cls = blank(PRIORS);
    const obj = blank(PRIORS);
    const bbox = bboxTensors(PRIORS);
    const kps = kpsTensors(PRIORS);
    cls[0]![0] = 0.09; // sqrt(0.09) = 0.3, below 0.5
    obj[0]![0] = 1;
    expect(decodeYuNet({ cls, obj, bbox, kps }, PRIORS, 0.5)).toHaveLength(0);
  });

  test("a confident class with no objectness can never pass", () => {
    // sqrt(cls * obj) is why this matters: obj=0 must veto cls=1.
    const cls = blank(PRIORS);
    const obj = blank(PRIORS);
    const bbox = bboxTensors(PRIORS);
    const kps = kpsTensors(PRIORS);
    cls[0]![5] = 1;
    obj[0]![5] = 0;
    expect(decodeYuNet({ cls, obj, bbox, kps }, PRIORS, 0.1)).toHaveLength(0);
  });

  test("keypoints map through the same grid arithmetic", () => {
    const { stride, cols } = PRIORS[0]!;
    const cls = blank(PRIORS);
    const obj = blank(PRIORS);
    const bbox = bboxTensors(PRIORS);
    const kps = kpsTensors(PRIORS);
    const r = 1;
    const c = 1;
    const cell = r * cols + c;
    cls[0]![cell] = 1;
    obj[0]![cell] = 1;
    // five keypoints at increasing x within the cell
    for (let k = 0; k < 5; k++) {
      kps[0]![cell * 10 + k * 2] = 0.1 * (k + 1);
      kps[0]![cell * 10 + k * 2 + 1] = 0.5;
    }
    const faces = decodeYuNet({ cls, obj, bbox, kps }, PRIORS, 0.5);
    expect(faces).toHaveLength(1);
    const pts = faces[0]!.kps;
    expect(pts).toHaveLength(5);
    for (let k = 0; k < 5; k++) {
      expect(pts[k]!.x).toBeCloseTo((c + 0.1 * (k + 1)) * stride, 5);
      expect(pts[k]!.y).toBeCloseTo((r + 0.5) * stride, 5);
    }
  });

  test("missing tensors for a stride are skipped, not fatal", () => {
    const cls = blank(PRIORS);
    const obj = blank(PRIORS);
    const bbox = bboxTensors(PRIORS);
    const kps = kpsTensors(PRIORS);
    cls[0]![0] = 1;
    obj[0]![0] = 1;
    const faces = decodeYuNet(
      { cls: [cls[0], undefined], obj: [obj[0], undefined], bbox: [bbox[0], undefined], kps: [kps[0], undefined] },
      PRIORS,
      0.5,
    );
    expect(faces).toHaveLength(1);
  });
});

describe("nms", () => {
  const face = (x: number, score: number) => ({
    box: { x, y: 0, width: 10, height: 10 },
    score,
    kps: [],
  });

  test("keeps the higher-scoring box and drops its heavy overlap", () => {
    const kept = nms([face(0, 0.6), face(2, 0.9)], 0.3);
    expect(kept).toHaveLength(1);
    expect(kept[0]!.score).toBe(0.9);
  });

  test("keeps boxes that do not overlap", () => {
    expect(nms([face(0, 0.6), face(100, 0.9)], 0.3)).toHaveLength(2);
  });

  test("is order-independent for the surviving set", () => {
    const a = nms([face(0, 0.6), face(100, 0.9)], 0.3).map((f) => f.score).sort();
    const b = nms([face(100, 0.9), face(0, 0.6)], 0.3).map((f) => f.score).sort();
    expect(a).toEqual(b);
  });
});
