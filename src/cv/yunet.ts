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
/** Images with a smaller side at or below this use the cheap input. */
const SMALL_IMAGE_SIDE = 800;
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
  const out: FaceDetection[] = [];
  for (const face of nms(raw, nmsThreshold)) {
    const scaled = face.kps.map((p) => ({ x: p.x * sx, y: p.y * sy }));
    // YuNet reports (right eye, left eye, nose, right mouth, left mouth).
    // Sort the eyes and mouth corners by x so the order matches CANONICAL_5,
    // exactly as the landmarker path does.
    const eyeA = scaled[0]!;
    const eyeB = scaled[1]!;
    const nose = scaled[2]!;
    const mouthA = scaled[3]!;
    const mouthB = scaled[4]!;
    const [leftEye, rightEye] = eyeA.x <= eyeB.x ? [eyeA, eyeB] : [eyeB, eyeA];
    const [leftMouth, rightMouth] = mouthA.x <= mouthB.x ? [mouthA, mouthB] : [mouthB, mouthA];
    out.push({
      box: {
        x: face.box.x * sx,
        y: face.box.y * sy,
        width: face.box.width * sx,
        height: face.box.height * sy,
      },
      // A real detector score here, unlike the landmarker path's placeholder.
      confidence: face.score,
      landmarks: [leftEye, rightEye, nose, leftMouth, rightMouth],
    });
  }
  return out;
}

/**
 * Input side for an image.
 *
 * Detection cost scales with the square of the input, so 320 costs 7 ms against
 * 24 ms at 640. But a smaller input also raises the smallest face the detector
 * can see (~10 px in the tensor): at 320 a face must occupy twice the fraction
 * of the frame it would at 640.
 *
 * So the choice is made by IMAGE size, in ONE pass. A small image cannot hide a
 * face tiny enough to need 640 — avatars and video frames are exactly this case
 * and get the cheap pass. A large photo may hide small or distant faces, so it
 * always gets full resolution. An earlier version ran 320 first and returned
 * early when it found anything, which silently skipped the small faces in any
 * large photo that also contained one close-up.
 */
function inputSizeFor(srcW: number, srcH: number): number {
  return Math.min(srcW, srcH) <= SMALL_IMAGE_SIDE ? FAST_INPUT_SIZE : FULL_INPUT_SIZE;
}

export async function detectFacesYuNet(
  detector: YuNetDetector,
  image: HTMLImageElement,
  opts: { scoreThreshold?: number; nmsThreshold?: number } = {},
): Promise<FaceDetection[]> {
  const scoreThreshold = opts.scoreThreshold ?? 0.5;
  const nmsThreshold = opts.nmsThreshold ?? 0.3;
  const srcW = image.naturalWidth;
  const srcH = image.naturalHeight;
  if (srcW === 0 || srcH === 0) {
    throw new Error("faceBlock: detectFacesYuNet received an image with no pixels");
  }
  return detectAtSize(detector, image, inputSizeFor(srcW, srcH), scoreThreshold, nmsThreshold);
}
