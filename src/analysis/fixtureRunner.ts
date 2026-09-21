/**
 * ONNX-backed appearance-change diagnostics for local fixtures.
 * Uses raster decode (jpeg-js) so Bun tests avoid happy-dom drawImage hangs.
 */

import { resolve } from "node:path";
import { alignFace, alignmentSourceRect } from "../cv/align.ts";
import { createEmbedder, embedAligned, type Embedder } from "../cv/embedder.ts";
import { rasterRegion } from "../cv/raster.ts";
import { createYuNetDetector, detectFacesYuNetRaster, type YuNetDetector } from "../cv/yunet.ts";
import { cosineNormalized } from "../matching/cosine.ts";
import { matchFace } from "../matching/matcher.ts";
import { expandDetectionBox } from "../overlay/coordinates.ts";
import { MATCH_THRESHOLD } from "../../extension/enroll.ts";
import type { BlockedIdentity, Box, FaceDetection, Raster } from "../shared/types.ts";
import { loadFixtureRaster } from "./fixtureDecode.ts";
import {
  legacyAnalyzerMaskBox,
  legacyFullPipelineMaskBox,
} from "./legacyMask.ts";

export const DEFAULT_MODEL_ROOT = resolve(import.meta.dir, "../../demo/public");

export interface FixtureAssetPaths {
  modelRoot?: string;
  yunetModel?: string;
  embedderModel?: string;
  wasmDir?: string;
}

export interface FaceAppearanceReport {
  detectionBox: Box;
  confidence: number;
  bestCosine: number | null;
  matched: boolean;
  matchedIdentityId: string | null;
  maskCurrent: Box;
  maskLegacyAnalyzer: Box;
  maskLegacyFullPipeline: Box;
}

export interface AppearanceQueryReport {
  imagePath: string;
  width: number;
  height: number;
  faceCount: number;
  threshold: number;
  minAgreements: number;
  faces: FaceAppearanceReport[];
}

export interface AppearanceFixtureReport {
  fixtureId: string;
  enrollPaths: string[];
  queryPath: string;
  identityId: string;
  enrollFaceCounts: number[];
  query: AppearanceQueryReport;
}

function abs(root: string, rel: string): string {
  return resolve(root, rel.replace(/^\/+/, ""));
}

async function embedDetectedFaceRaster(
  embedder: Embedder,
  raster: Raster,
  det: FaceDetection,
): Promise<Float32Array> {
  const { raster: region, offsetX, offsetY } = rasterRegion(raster, alignmentSourceRect(det), 0);
  const local: FaceDetection = {
    ...det,
    box: { ...det.box, x: det.box.x - offsetX, y: det.box.y - offsetY },
    landmarks: det.landmarks?.map((pt) => ({ x: pt.x - offsetX, y: pt.y - offsetY })),
  };
  return embedAligned(embedder, alignFace(region, local));
}

export async function createFixtureModels(paths: FixtureAssetPaths = {}): Promise<{
  detector: YuNetDetector;
  embedder: Embedder;
}> {
  const root = paths.modelRoot ?? DEFAULT_MODEL_ROOT;
  const yunetPath = abs(root, paths.yunetModel ?? "models/face_detection_yunet_2026may.onnx");
  const embedPath = abs(root, paths.embedderModel ?? "models/w600k_mbf.onnx");
  const wasmPath = abs(root, paths.wasmDir ?? "ort/");
  const fileUrl = (p: string) => `file://${p}`;
  const wasmUrl = fileUrl(wasmPath.endsWith("/") ? wasmPath : `${wasmPath}/`);
  const [detector, embedder] = await Promise.all([
    createYuNetDetector(fileUrl(yunetPath), wasmUrl),
    createEmbedder(fileUrl(embedPath), wasmUrl),
  ]);
  return { detector, embedder };
}

function bestCosine(
  embedding: Float32Array,
  identities: readonly BlockedIdentity[],
): { bestCosine: number | null; bestIdentityId: string | null } {
  let best: number | null = null;
  let bestId: string | null = null;
  for (const identity of identities) {
    for (const g of identity.embeddings) {
      const s = cosineNormalized(embedding, g);
      if (best === null || s > best) {
        best = s;
        bestId = identity.id;
      }
    }
  }
  return { bestCosine: best, bestIdentityId: bestId };
}

export async function runAppearanceFixture(
  opts: {
    fixtureId: string;
    enrollPaths: string[];
    queryPath: string;
    identityId: string;
    minAgreements?: number;
    threshold?: number;
  } & FixtureAssetPaths,
): Promise<AppearanceFixtureReport> {
  const { detector, embedder } = await createFixtureModels(opts);
  const minAgreements = opts.minAgreements ?? 1;
  const threshold = opts.threshold ?? MATCH_THRESHOLD;

  const enrollEmbeddings: Float32Array[] = [];
  const enrollFaceCounts: number[] = [];
  for (const p of opts.enrollPaths) {
    const raster = await loadFixtureRaster(p);
    const dets = await detectFacesYuNetRaster(detector, raster);
    enrollFaceCounts.push(dets.length);
    if (dets.length === 1) {
      enrollEmbeddings.push(await embedDetectedFaceRaster(embedder, raster, dets[0]!));
    }
  }
  if (enrollEmbeddings.length === 0) {
    throw new Error(
      `fixture ${opts.fixtureId}: no enroll embeddings — face counts: ${enrollFaceCounts.join(", ")}`,
    );
  }

  const identity: BlockedIdentity = {
    id: opts.identityId,
    displayName: opts.identityId,
    embeddings: enrollEmbeddings,
    threshold,
    createdAt: 0,
  };

  const queryRaster = await loadFixtureRaster(opts.queryPath);
  const w = queryRaster.width;
  const h = queryRaster.height;
  const imageSize = { width: w, height: h };
  const dets = await detectFacesYuNetRaster(detector, queryRaster);

  const faces: FaceAppearanceReport[] = [];
  for (const det of dets) {
    const embedding = await embedDetectedFaceRaster(embedder, queryRaster, det);
    const match = matchFace(embedding, [identity], { minAgreements, defaultThreshold: threshold });
    const best = bestCosine(embedding, [identity]);
    faces.push({
      detectionBox: { ...det.box },
      confidence: det.confidence,
      bestCosine: best.bestCosine,
      matched: match !== null,
      matchedIdentityId: match?.identityId ?? null,
      maskCurrent: expandDetectionBox(det.box, imageSize),
      maskLegacyAnalyzer: legacyAnalyzerMaskBox(det.box, imageSize),
      maskLegacyFullPipeline: legacyFullPipelineMaskBox(det.box, imageSize),
    });
  }

  return {
    fixtureId: opts.fixtureId,
    enrollPaths: opts.enrollPaths,
    queryPath: opts.queryPath,
    identityId: opts.identityId,
    enrollFaceCounts,
    query: {
      imagePath: opts.queryPath,
      width: w,
      height: h,
      faceCount: dets.length,
      threshold,
      minAgreements,
      faces,
    },
  };
}

export function primarySubjectFace(report: AppearanceQueryReport): FaceAppearanceReport | null {
  if (report.faces.length === 0) return null;
  let best = report.faces[0]!;
  for (const face of report.faces.slice(1)) {
    const a = face.bestCosine ?? -1;
    const b = best.bestCosine ?? -1;
    if (a > b) best = face;
  }
  return best;
}
