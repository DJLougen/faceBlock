import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import type { FaceDetection, Point } from "../shared/types.ts";

/**
 * MediaPipe Face Mesh indices used to derive the ArcFace 5-point alignment.
 * Eye centers are midpoints of the inner/outer corner pair, not single
 * corners: left eye = midpoint(33, 133), right eye = midpoint(362, 263).
 * Nose tip = 1, mouth corners = 61 and 291.
 */
const IDX_LEFT_EYE_A = 33;
const IDX_LEFT_EYE_B = 133;
const IDX_RIGHT_EYE_A = 362;
const IDX_RIGHT_EYE_B = 263;
const IDX_NOSE_TIP = 1;
const IDX_MOUTH_A = 61;
const IDX_MOUTH_B = 291;

/** Highest Face Mesh index we read; a face with fewer landmarks is unusable. */
const MIN_LANDMARKS = IDX_MOUTH_B + 1;

/**
 * MediaPipe's default detection/presence floors (0.5) are tuned for frontal
 * faces. A side-on face scores below them and produced NO detection at all, so
 * nothing downstream could ever match it — turning "the person looked away"
 * into "the person was not blocked". Measured on held-out photos: two outright
 * profile shots yielded zero faces at the default, while every face that was
 * detected scored 0.56-0.64 and matched comfortably.
 *
 * Lowering the floors recovers those poses. The cost is more spurious boxes on
 * non-faces, which the identity match then rejects — a cheap trade, since a
 * missed face is a silent failure while a spurious box costs one embedding.
 *
 * 0.2 is the measured floor. Dropping to 0.1 was tried and rejected: it did NOT
 * recover the remaining near-90-degree side view, it grew a false face on a
 * text document, and overall masking got worse. The extreme profile is a Face
 * Landmarker limit, not something a threshold can be ground past.
 */
const MIN_FACE_DETECTION_CONFIDENCE = 0.2;
const MIN_FACE_PRESENCE_CONFIDENCE = 0.2;

/**
 * Create a FaceLandmarker, preferring the GPU delegate and falling back to
 * CPU when WebGL is unavailable or the GPU graph fails to build.
 * wasmDir must serve the MediaPipe vision WASM files locally
 * (vision_wasm_internal.js/.wasm, vision_wasm_nosimd_internal.js/.wasm).
 */
export async function createFaceLandmarker(
  wasmDir = "/mediapipe-wasm",
  modelPath = "/models/face_landmarker.task",
): Promise<FaceLandmarker> {
  let fileset;
  try {
    fileset = await FilesetResolver.forVisionTasks(wasmDir);
  } catch (e) {
    throw new Error(
      `faceBlock: cannot load MediaPipe WASM runtime from "${wasmDir}" — ` +
        `serve vision_wasm_internal.{js,wasm} and vision_wasm_nosimd_internal.{js,wasm} there. ` +
        `Cause: ${errText(e)}`,
    );
  }
  const options = (delegate: "GPU" | "CPU") => ({
    baseOptions: { modelAssetPath: modelPath, delegate },
    runningMode: "IMAGE" as const,
    numFaces: 8,
    minFaceDetectionConfidence: MIN_FACE_DETECTION_CONFIDENCE,
    minFacePresenceConfidence: MIN_FACE_PRESENCE_CONFIDENCE,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });
  try {
    return await FaceLandmarker.createFromOptions(fileset, options("GPU"));
  } catch (gpuErr) {
    try {
      return await FaceLandmarker.createFromOptions(fileset, options("CPU"));
    } catch (cpuErr) {
      throw new Error(
        `faceBlock: FaceLandmarker init failed on GPU (${errText(gpuErr)}) ` +
          `and CPU (${errText(cpuErr)}). Check that "${modelPath}" is served ` +
          `and is a valid face_landmarker.task bundle.`,
      );
    }
  }
}

/**
 * Detect faces and emit ArcFace-order landmarks:
 * [leftEye, rightEye, nose, leftMouth, rightMouth] in image pixels.
 * Left/right are assigned by sorted image x so the order matches CANONICAL_5
 * even when the input image is mirrored.
 */
export function detectFaces(
  landmarker: FaceLandmarker,
  image: HTMLImageElement,
): FaceDetection[] {
  const w = image.naturalWidth;
  const h = image.naturalHeight;
  if (w === 0 || h === 0) {
    throw new Error(
      "faceBlock: detectFaces received an image with no pixels — " +
        "wait for load/decode before calling",
    );
  }
  const result = landmarker.detect(image);
  const out: FaceDetection[] = [];
  for (const lm of result.faceLandmarks) {
    if (lm.length < MIN_LANDMARKS) continue;
    let minX = 1;
    let minY = 1;
    let maxX = 0;
    let maxY = 0;
    for (const p of lm) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }
    const eyeA = midpoint(lm[IDX_LEFT_EYE_A]!, lm[IDX_LEFT_EYE_B]!, w, h);
    const eyeB = midpoint(lm[IDX_RIGHT_EYE_A]!, lm[IDX_RIGHT_EYE_B]!, w, h);
    const mouthA = pixel(lm[IDX_MOUTH_A]!, w, h);
    const mouthB = pixel(lm[IDX_MOUTH_B]!, w, h);
    const [leftEye, rightEye] = eyeA.x <= eyeB.x ? [eyeA, eyeB] : [eyeB, eyeA];
    const [leftMouth, rightMouth] =
      mouthA.x <= mouthB.x ? [mouthA, mouthB] : [mouthB, mouthA];
    out.push({
      box: {
        x: minX * w,
        y: minY * h,
        width: (maxX - minX) * w,
        height: (maxY - minY) * h,
      },
      // FaceLandmarker exposes no per-face detection score. This 1 is a
      // documented placeholder so FaceDetection.confidence stays a number;
      // it is NOT a measured confidence and must not drive thresholds.
      confidence: 1,
      landmarks: [
        leftEye,
        rightEye,
        pixel(lm[IDX_NOSE_TIP]!, w, h),
        leftMouth,
        rightMouth,
      ],
    });
  }
  return out;
}

function pixel(p: { x: number; y: number }, w: number, h: number): Point {
  return { x: p.x * w, y: p.y * h };
}

function midpoint(
  a: { x: number; y: number },
  b: { x: number; y: number },
  w: number,
  h: number,
): Point {
  return { x: ((a.x + b.x) / 2) * w, y: ((a.y + b.y) / 2) * h };
}

function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
