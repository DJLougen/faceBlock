import * as ort from "onnxruntime-web/wasm";
import { ALIGN_SIZE } from "../shared/config.ts";
import { hwcToNchw, normalizeRgb } from "./preprocess.ts";
import { l2Normalize } from "../matching/cosine.ts";

export type Embedder = {
  session: ort.InferenceSession;
  inputName: string;
  outputName: string;
};

/**
 * Load the ArcFace recognition model on the WASM backend.
 *
 * wasmPaths must serve the onnxruntime-web WASM binaries locally
 * (ort-wasm-simd-threaded.wasm is the default build used here). numThreads=1
 * keeps the runtime on a single thread so no cross-origin-isolation
 * (COOP/COEP) headers are required. Only the "wasm" execution provider is
 * requested — it is the reliable default; WebGPU is not enabled because it
 * has not been tested with this model.
 */
export async function createEmbedder(
  modelPath = "/models/w600k_mbf.onnx",
  wasmPaths = "/ort/",
): Promise<Embedder> {
  ort.env.wasm.wasmPaths = wasmPaths;
  ort.env.wasm.numThreads = 1;
  let session: ort.InferenceSession;
  try {
    session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ["wasm"],
    });
  } catch (e) {
    throw new Error(
      `faceBlock: cannot create ONNX session for "${modelPath}" — ` +
        `serve the model and the ort-wasm-simd-threaded.* binaries under ` +
        `"${wasmPaths}". Cause: ${e instanceof Error ? e.message : String(e)}`,
    );
  }
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  if (!inputName || !outputName) {
    throw new Error(
      `faceBlock: "${modelPath}" exposes no input/output names — ` +
        `expected an ArcFace ONNX model (input [N,3,112,112], output [N,512])`,
    );
  }
  return { session, inputName, outputName };
}

/**
 * aligned HWC RGB 0..255 at ALIGN_SIZE → L2-normalized embedding.
 *
 * Normalization follows the official InsightFace arcface_onnx.py recipe for
 * w600k_mbf: the graph has no leading Sub/Mul normalization nodes, so
 * input_mean = input_std = 127.5 and the blob is (pixel - 127.5) / 127.5 in
 * RGB order (blobFromImages swapRB=True on BGR input ≡ RGB here).
 */
export async function embedAligned(
  embedder: Embedder,
  alignedHwc: Float32Array,
): Promise<Float32Array> {
  const expected = ALIGN_SIZE * ALIGN_SIZE * 3;
  if (alignedHwc.length !== expected) {
    throw new Error(
      `faceBlock: embedAligned expected ${expected} floats ` +
        `(${ALIGN_SIZE}x${ALIGN_SIZE}x3 HWC), got ${alignedHwc.length} — ` +
        `pass the output of alignFace()`,
    );
  }
  const normed = normalizeRgb(alignedHwc, 127.5, 127.5);
  const nchw = hwcToNchw(normed, ALIGN_SIZE, ALIGN_SIZE);
  const input = new ort.Tensor("float32", nchw, [1, 3, ALIGN_SIZE, ALIGN_SIZE]);
  let outputs: ort.InferenceSession.OnnxValueMapType;
  try {
    outputs = await embedder.session.run({ [embedder.inputName]: input });
  } finally {
    input.dispose();
  }
  try {
    const raw = outputs[embedder.outputName];
    if (!raw) {
      throw new Error(
        `faceBlock: ONNX output "${embedder.outputName}" missing from session result`,
      );
    }
    const data = raw.data as Float32Array;
    let sumSq = 0;
    for (let i = 0; i < data.length; i++) {
      const v = data[i]!;
      if (!Number.isFinite(v)) {
        throw new Error(
          `faceBlock: embedder produced a non-finite value at index ${i} — ` +
            `the model or input raster is corrupt`,
        );
      }
      sumSq += v * v;
    }
    if (sumSq === 0) {
      throw new Error(
        "faceBlock: embedder produced an all-zero descriptor — " +
          "the aligned face crop is likely blank",
      );
    }
    return l2Normalize(new Float32Array(data));
  } finally {
    for (const name of Object.keys(outputs)) outputs[name]?.dispose();
  }
}
