/**
 * YuNet face detector (MIT, 232 KB) via ONNX Runtime Web.
 *
 * WHY THIS REPLACES THE LANDMARKER AS THE DETECTOR: MediaPipe's FaceLandmarker
 * is a landmark model tuned for near-frontal faces. It returned ZERO faces for
 * profile views and for faces that occupy a small fraction of a large photo —
 * two failure modes that leave nothing downstream to match, so no amount of
 * similarity filtering can recover them. YuNet is a purpose-built face detector
 * trained for faces from roughly 10x10 to 300x300 px, and it reports the five
 * landmarks the ArcFace alignment needs, so it slots into the existing
 * pipeline in place of the landmarker's detection role.
 *
 * Licence: MIT (Shiqi Yu). Commercially usable, unlike the ArcFace weights it
 * is paired with.
 *
 * Decoding follows libfacedetection's published scheme: per stride, box centre
 * is (grid + offset) * stride, size is exp(w/h) * stride, and the five
 * keypoints are (grid + k) * stride. Score is sqrt(cls * obj). Boxes are
 * non-max-suppressed, then scaled back to source-image pixels.
 */

import * as ort from "onnxruntime-web/wasm";
import type { FaceDetection, Point } from "../shared/types.ts";

/**
 * Input side used for the first, cheap pass. The 2026may export takes a
 * dynamic input shape, so a small tensor costs a quarter of the pixels of 640:
 * less canvas read-back, less pixel packing, and a smaller matmul.
 */
const FAST_INPUT_SIZE = 320;
/** Full-resolution input side, used for images big enough to hide a small face. */
const FULL_INPUT_SIZE = 640;
/** Feature strides YuNet predicts at. */
const STRIDES = [8, 16, 32] as const;
/** Keypoints per face (10 floats: 5 x/y pairs). */
const KPS_PER_FACE = 10;
const KPS_PER_STRIDE = 5;

export interface YuNetPrior {
  stride: number;
  cols: number;
  rows: number;
}

/** Cell grid each stride predicts over, for a given square input. */
export function yunetPriors(inputSize = FULL_INPUT_SIZE, strides: readonly number[] = STRIDES): YuNetPrior[] {
  return strides.map((stride) => {
    const side = Math.floor(inputSize / stride);
    return { stride, cols: side, rows: side };
  });
}

export interface YuNetDetector {
  session: ort.InferenceSession;
  inputName: string;
  /** Prior grids keyed by the input side they were generated for. */
  priorsBySize: Map<number, YuNetPrior[]>;
  outputNames: Record<string, string>;
}

/**
 * Load the detector. `wasmPaths` must serve the onnxruntime-web binaries the
 * extension already bundles.
 */
export async function createYuNetDetector(
  modelPath = "/models/face_detection_yunet_2023mar.onnx",
  wasmPaths = "/ort/",
): Promise<YuNetDetector> {
  ort.env.wasm.wasmPaths = wasmPaths;
  // MV3 forbids blob: workers, so ORT must not spawn its proxy worker.
  ort.env.wasm.proxy = false;
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ["wasm"],
  });
  const inputName = session.inputNames[0];
  if (!inputName) throw new Error("faceBlock: YuNet session exposes no input");
  // Resolve stride-suffixed output names from whatever the graph actually has,
  // rather than assuming an export layout.
  const outputNames: Record<string, string> = {};
  for (const kind of ["cls", "obj", "bbox", "kps"]) {
    for (const stride of STRIDES) {
      const wanted = `${kind}_${stride}`;
      const actual = session.outputNames.find((n) => n === wanted || n.endsWith(`/${wanted}`) || n.includes(wanted));
      if (actual) outputNames[wanted] = actual;
    }
  }
  const priorsBySize = new Map<number, YuNetPrior[]>();
  for (const size of [FAST_INPUT_SIZE, FULL_INPUT_SIZE]) priorsBySize.set(size, yunetPriors(size));
  return { session, inputName, priorsBySize, outputNames };
}

function iou(
  a: { x: number; y: number; width: number; height: number },
  b: { x: number; y: number; width: number; height: number },
): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  if (x2 <= x1 || y2 <= y1) return 0;
  const inter = (x2 - x1) * (y2 - y1);
  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}

interface RawFace {
  box: { x: number; y: number; width: number; height: number };
  score: number;
  kps: Point[];
}

/**
 * Greedy NMS over raw detections, highest score first.
 */
export function nms(faces: RawFace[], threshold: number): RawFace[] {
  const sorted = [...faces].sort((a, b) => b.score - a.score);
  const keep: RawFace[] = [];
  for (const face of sorted) {
    if (keep.some((k) => iou(k.box, face.box) > threshold)) continue;
    keep.push(face);
  }
  return keep;
}

/**
 * Decode raw YuNet outputs into faces in INPUT_SIZE coordinates.
 * Exported for testing: the arithmetic is easy to get subtly wrong and hard to
 * see once it is buried in a session call.
 */
export function decodeYuNet(
  tensors: {
    cls: (Float32Array | undefined)[];
    obj: (Float32Array | undefined)[];
    bbox: (Float32Array | undefined)[];
    kps: (Float32Array | undefined)[];
  },
  priors: YuNetPrior[],
  scoreThreshold: number,
): RawFace[] {
  const faces: RawFace[] = [];
  for (let p = 0; p < priors.length; p++) {
    const { stride, cols, rows } = priors[p]!;
    const cls = tensors.cls[p];
    const obj = tensors.obj[p];
    const bbox = tensors.bbox[p];
    const kps = tensors.kps[p];
    if (!cls || !obj || !bbox || !kps) continue;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cell = r * cols + c;
        // libfacedetection scores a face as sqrt(cls * obj); either alone can
        // be confidently wrong on background.
        const score = Math.sqrt(Math.max(0, cls[cell]!) * Math.max(0, obj[cell]!));
        if (score < scoreThreshold) continue;
        const b = cell * 4;
        const cx = (c + bbox[b]!) * stride;
        const cy = (r + bbox[b + 1]!) * stride;
        const w = Math.exp(bbox[b + 2]!) * stride;
        const h = Math.exp(bbox[b + 3]!) * stride;
        const points: Point[] = [];
        for (let k = 0; k < KPS_PER_STRIDE; k++) {
          const kb = cell * KPS_PER_FACE + k * 2;
          points.push({ x: (c + kps[kb]!) * stride, y: (r + kps[kb + 1]!) * stride });
        }
        faces.push({
          box: { x: cx - w / 2, y: cy - h / 2, width: w, height: h },
          score,
          kps: points,
        });
      }
    }
  }
  return faces;
}

/**
 * Detect faces in an image, returning boxes and ArcFace-ordered landmarks in
 * SOURCE-image pixel coordinates.
 */
/** Run one detection pass at a given input side, in source-image coordinates. */
async function detectAtSize(
  detector: YuNetDetector,
  image: HTMLImageElement,
  inputSize: number,
  scoreThreshold: number,
  nmsThreshold: number,
): Promise<FaceDetection[]> {
  const srcW = image.naturalWidth;
  const srcH = image.naturalHeight;
  const canvas = document.createElement("canvas");
  canvas.width = inputSize;
  canvas.height = inputSize;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("faceBlock: 2d canvas unavailable for YuNet input");
  // The exported model expects a BGR tensor scaled 0..255; the image is
  // stretched to the input, then boxes are scaled back by the inverse.
  ctx.drawImage(image, 0, 0, srcW, srcH, 0, 0, inputSize, inputSize);
  const { data } = ctx.getImageData(0, 0, inputSize, inputSize);

  const plane = inputSize * inputSize;
  const chw = new Float32Array(3 * plane);
  // BGR, not RGB. The canvas hands back RGB, but this detector was trained on
  // OpenCV's native BGR channel order, and feeding it RGB measurably weakens
  // it: on the profile view the top score fell from 0.898 to 0.798, and a
  // crowd photo yielded 16 faces instead of 18.
  for (let i = 0; i < plane; i++) {
    const o = i * 4;
    chw[i] = data[o + 2]!;
    chw[plane + i] = data[o + 1]!;
    chw[2 * plane + i] = data[o]!;
  }

  const feeds: Record<string, ort.Tensor> = {
    [detector.inputName]: new ort.Tensor("float32", chw, [1, 3, inputSize, inputSize]),
  };
  const results = await detector.session.run(feeds);

  const pick = (kind: string, stride: number): Float32Array | undefined => {
    const name = detector.outputNames[`${kind}_${stride}`];
    if (!name) return undefined;
    const t = results[name];
    return t ? (t.data as Float32Array) : undefined;
  };

  const raw = decodeYuNet(
    {
      cls: STRIDES.map((s) => pick("cls", s)),
      obj: STRIDES.map((s) => pick("obj", s)),
      bbox: STRIDES.map((s) => pick("bbox", s)),
      kps: STRIDES.map((s) => pick("kps", s)),
    },
    detector.priorsBySize.get(inputSize) ?? yunetPriors(inputSize),
    scoreThreshold,
  );

  const sx = srcW / inputSize;
  const sy = srcH / inputSize;
  return nms(raw, nmsThreshold).map((face) => toFaceDetection(face, sx, sy, 0, 0));
}

/**
 * Map a raw detection from input-tensor space into image pixels.
 *
 * YuNet reports its five keypoints as (right eye, left eye, nose, right mouth,
 * left mouth). The eyes and mouth corners are sorted by x so the order matches
 * CANONICAL_5, which is what the ArcFace alignment expects. YuNet's keypoints
 * are learned, so unlike the landmarker this covers profile and occluded faces.
 */
function toFaceDetection(
  face: { box: { x: number; y: number; width: number; height: number }; score: number; kps: Point[] },
  sx: number,
  sy: number,
  offX: number,
  offY: number,
): FaceDetection {
  const scaled = face.kps.map((p) => ({ x: p.x * sx + offX, y: p.y * sy + offY }));
  const eyeA = scaled[0]!;
  const eyeB = scaled[1]!;
  const nose = scaled[2]!;
  const mouthA = scaled[3]!;
  const mouthB = scaled[4]!;
  const [leftEye, rightEye] = eyeA.x <= eyeB.x ? [eyeA, eyeB] : [eyeB, eyeA];
  const [leftMouth, rightMouth] = mouthA.x <= mouthB.x ? [mouthA, mouthB] : [mouthB, mouthA];
  return {
    box: {
      x: face.box.x * sx + offX,
      y: face.box.y * sy + offY,
      width: face.box.width * sx,
      height: face.box.height * sy,
    },
    confidence: face.score,
    landmarks: [leftEye, rightEye, nose, leftMouth, rightMouth],
  };
}

/** Boxes overlapping more than this are the same face seen twice. */
const MERGE_IOU = 0.4;
/** Grid divisions per axis for the small-face tiling pass. */
const TILE_GRID = 3;
/** Fraction of each tile that overlaps its neighbour, so faces on seams survive. */
const TILE_OVERLAP = 0.25;
/** Faces found before the image is treated as a group shot worth tiling. */
const CROWD_FACE_COUNT = 2;

function overlapIoU(a: FaceDetection, b: FaceDetection): number {
  const x1 = Math.max(a.box.x, b.box.x);
  const y1 = Math.max(a.box.y, b.box.y);
  const x2 = Math.min(a.box.x + a.box.width, b.box.x + b.box.width);
  const y2 = Math.min(a.box.y + a.box.height, b.box.y + b.box.height);
  if (x2 <= x1 || y2 <= y1) return 0;
  const inter = (x2 - x1) * (y2 - y1);
  const union = a.box.width * a.box.height + b.box.width * b.box.height - inter;
  return union > 0 ? inter / union : 0;
}

function merge(existing: FaceDetection[], add: FaceDetection[]): FaceDetection[] {
  const out = [...existing];
  for (const face of add) {
    if (!out.some((m) => overlapIoU(m, face) > MERGE_IOU)) out.push(face);
  }
  return out;
}

/**
 * Run the detector over the image, spending resolution only where it pays.
 *
 * The trigger is FACE COUNT, not image size. An earlier version gated tiling on
 * the image's smaller side being over 800 px, which meant the 897x648 crowd
 * photo that the tiling was measured on never tiled at all -- the gate
 * contradicted the data it came from. Counting faces is both simpler and
 * directly tied to what tiling is for: a group shot has small faces; a portrait
 * does not.
 *
 *  - 320 pass first: 7 ms against 24 ms at 640.
 *  - Nothing found -> the 640 pass runs. Measured: sunglasses plus a face
 *    covering on a 460 px photo gives 0 faces at 320 and 1 at 0.747 at 640.
 *  - Two or more faces -> a group shot, so a 3x3 tiled pass is merged in.
 *    Measured on the crowd photo: 18 faces plain, 23 with tiles, reaching faces
 *    30 px across. Tiling at 640 found one more for three times the cost, so the
 *    tiles stay at the cheap input.
 *
 * Tiling is what recovers small faces: a crop makes a small face occupy more of
 * the detector's fixed input, which is the same reason whole-image downscaling
 * loses it.
 */
export async function detectFacesYuNet(
  detector: YuNetDetector,
  image: HTMLImageElement,
  opts: { scoreThreshold?: number; nmsThreshold?: number } = {},
): Promise<FaceDetection[]> {
  const scoreThreshold = opts.scoreThreshold ?? 0.5;
  const nmsThreshold = opts.nmsThreshold ?? 0.3;
  if (image.naturalWidth === 0 || image.naturalHeight === 0) {
    throw new Error("faceBlock: detectFacesYuNet received an image with no pixels");
  }

  let faces = await detectAtSize(detector, image, FAST_INPUT_SIZE, scoreThreshold, nmsThreshold);
  if (faces.length === 0) {
    faces = await detectAtSize(detector, image, FULL_INPUT_SIZE, scoreThreshold, nmsThreshold);
  }
  if (faces.length >= CROWD_FACE_COUNT) {
    const tiled = await detectTiled(detector, image, FAST_INPUT_SIZE, scoreThreshold, nmsThreshold);
    faces = merge(faces, tiled);
  }
  return faces;
}

/**
 * Draw one region of the image at the detector's input size and detect in it.
 * Returns boxes in REGION-LOCAL pixels; the caller adds the region's origin.
 */
async function detectRegion(
  detector: YuNetDetector,
  image: HTMLImageElement,
  originX: number,
  originY: number,
  regionW: number,
  regionH: number,
  inputSize: number,
  scoreThreshold: number,
  nmsThreshold: number,
  canvas: HTMLCanvasElement,
): Promise<FaceDetection[]> {
  if (canvas.width !== inputSize || canvas.height !== inputSize) {
    canvas.width = inputSize;
    canvas.height = inputSize;
  }
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return [];
  ctx.clearRect(0, 0, inputSize, inputSize);
  ctx.drawImage(image, originX, originY, regionW, regionH, 0, 0, inputSize, inputSize);
  const { data } = ctx.getImageData(0, 0, inputSize, inputSize);

  const plane = inputSize * inputSize;
  const chw = new Float32Array(3 * plane);
  // BGR: this detector was trained on OpenCV's native channel order.
  for (let i = 0; i < plane; i++) {
    const o = i * 4;
    chw[i] = data[o + 2]!;
    chw[plane + i] = data[o + 1]!;
    chw[2 * plane + i] = data[o]!;
  }
  const results = await detector.session.run({
    [detector.inputName]: new ort.Tensor("float32", chw, [1, 3, inputSize, inputSize]),
  });
  const pick = (kind: string, stride: number): Float32Array | undefined => {
    const name = detector.outputNames[`${kind}_${stride}`];
    if (!name) return undefined;
    const t = results[name];
    return t ? (t.data as Float32Array) : undefined;
  };
  const raw = decodeYuNet(
    {
      cls: STRIDES.map((s) => pick("cls", s)),
      obj: STRIDES.map((s) => pick("obj", s)),
      bbox: STRIDES.map((s) => pick("bbox", s)),
      kps: STRIDES.map((s) => pick("kps", s)),
    },
    detector.priorsBySize.get(inputSize) ?? yunetPriors(inputSize),
    scoreThreshold,
  );
  const sx = regionW / inputSize;
  const sy = regionH / inputSize;
  const out: FaceDetection[] = [];
  for (const face of nms(raw, nmsThreshold)) {
    out.push(toFaceDetection(face, sx, sy, 0, 0));
  }
  return out;
}

/**
 * Overlapping tiles, mapped back to full-image coordinates.
 */
async function detectTiled(
  detector: YuNetDetector,
  image: HTMLImageElement,
  inputSize: number,
  scoreThreshold: number,
  nmsThreshold: number,
): Promise<FaceDetection[]> {
  const width = image.naturalWidth;
  const height = image.naturalHeight;
  const tileW = Math.ceil(width / TILE_GRID);
  const tileH = Math.ceil(height / TILE_GRID);
  const stepX = Math.max(1, Math.round(tileW * (1 - TILE_OVERLAP)));
  const stepY = Math.max(1, Math.round(tileH * (1 - TILE_OVERLAP)));

  const canvas = document.createElement("canvas");
  const found: FaceDetection[] = [];
  for (let oy = 0; oy < height; oy += stepY) {
    for (let ox = 0; ox < width; ox += stepX) {
      const cw = Math.min(tileW, width - ox);
      const ch = Math.min(tileH, height - oy);
      if (cw < 64 || ch < 64) continue;
      const tile = await detectRegion(
        detector, image, ox, oy, cw, ch, inputSize, scoreThreshold, nmsThreshold, canvas,
      );
      for (const face of tile) {
        const mapped: FaceDetection = {
          box: { ...face.box, x: face.box.x + ox, y: face.box.y + oy },
          confidence: face.confidence,
          landmarks: face.landmarks?.map((p) => ({ x: p.x + ox, y: p.y + oy })),
        };
        if (!found.some((f) => overlapIoU(f, mapped) > MERGE_IOU)) found.push(mapped);
      }
    }
  }
  return found;
}

