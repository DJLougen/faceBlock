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
import { clusterEmbeddings } from "../resolve/cluster.ts";
import { MIN_REFERENCE_IMAGES } from "../shared/config.ts";
import { MATCH_THRESHOLD } from "../../extension/enroll.ts";
import type { BlockedIdentity, Box, FaceDetection, Raster } from "../shared/types.ts";
import { loadFixtureRaster } from "./fixtureDecode.ts";
import {
  legacyAnalyzerMaskBox,
  legacyFullPipelineMaskBox,
} from "./legacyMask.ts";

export const DEFAULT_MODEL_ROOT = resolve(import.meta.dir, "../../demo/public");

/** Same near-duplicate cutoff as extension/offscreen.ts RESOLVE_PREVIEW. */
export const ENROLL_DUPLICATE_COSINE = 0.98;

/** Same agreement count as extension/offscreen.ts ANALYZE paths. */
export const ANALYZE_MIN_AGREEMENTS = 1;

export type EnrollmentMode = "curated" | "resolve-cluster";

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
  enrollment: EnrollmentBuildResult;
  query: AppearanceQueryReport;
}

export interface EnrollmentBuildResult {
  mode: EnrollmentMode;
  galleryPaths: string[];
  gallerySize: number;
  meetsMinReferenceImages: boolean;
  rejected: { path: string; reason: string }[];
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
    const gallery =
      identity.prototypes && identity.prototypes.length > 0
        ? identity.prototypes
        : identity.embeddings;
    for (const g of gallery) {
      const s = cosineNormalized(embedding, g);
      if (best === null || s > best) {
        best = s;
        bestId = identity.id;
      }
    }
  }
  return { bestCosine: best, bestIdentityId: bestId };
}

interface EmbeddedEnrollPath {
  path: string;
  embedding: Float32Array;
}

/**
 * Mirror production enrollment: one face per photo, near-duplicate collapse, then
 * either curated (references.json path) or resolve-cluster (Wikimedia path).
 */
export async function buildProductionEnrollment(
  detector: YuNetDetector,
  embedder: Embedder,
  enrollPaths: readonly string[],
  mode: EnrollmentMode,
): Promise<{ enrollment: EnrollmentBuildResult; enrollFaceCounts: number[]; identity: BlockedIdentity; threshold: number }> {
  const threshold = MATCH_THRESHOLD;
  const embedded: EmbeddedEnrollPath[] = [];
  const enrollFaceCounts: number[] = [];
  const rejected: { path: string; reason: string }[] = [];

  for (const p of enrollPaths) {
    const raster = await loadFixtureRaster(p);
    const dets = await detectFacesYuNetRaster(detector, raster);
    enrollFaceCounts.push(dets.length);
    if (dets.length !== 1) {
      rejected.push({ path: p, reason: `${dets.length} faces` });
      continue;
    }
    embedded.push({
      path: p,
      embedding: await embedDetectedFaceRaster(embedder, raster, dets[0]!),
    });
  }

  const unique: EmbeddedEnrollPath[] = [];
  for (const item of embedded) {
    if (
      unique.some((u) => cosineNormalized(u.embedding, item.embedding) >= ENROLL_DUPLICATE_COSINE)
    ) {
      rejected.push({ path: item.path, reason: "duplicate" });
      continue;
    }
    unique.push(item);
  }

  let galleryPaths: string[];
  let galleryEmbeddings: Float32Array[];
  if (mode === "curated") {
    galleryPaths = unique.map((u) => u.path);
    galleryEmbeddings = unique.map((u) => u.embedding);
  } else {
    const cluster = clusterEmbeddings(
      unique.map((u) => u.embedding),
      { maxPrototypes: 8 },
    );
    for (const rej of cluster.rejected) {
      rejected.push({ path: unique[rej.index]?.path ?? "", reason: rej.reason });
    }
    galleryPaths = cluster.prototypes.map((i) => unique[i]!.path);
    galleryEmbeddings = cluster.prototypes.map((i) => unique[i]!.embedding);
  }

  if (galleryEmbeddings.length === 0) {
    throw new Error(
      `enrollment produced no gallery vectors — face counts: ${enrollFaceCounts.join(", ")}`,
    );
  }

  const enrollment: EnrollmentBuildResult = {
    mode,
    galleryPaths,
    gallerySize: galleryEmbeddings.length,
    meetsMinReferenceImages: galleryEmbeddings.length >= MIN_REFERENCE_IMAGES,
    rejected,
  };

  const identity: BlockedIdentity = {
    id: "fixture",
    displayName: "fixture",
    embeddings: galleryEmbeddings,
    threshold,
    createdAt: 0,
  };

  return { enrollment, enrollFaceCounts, identity, threshold };
}

export async function queryAppearanceImage(
  detector: YuNetDetector,
  embedder: Embedder,
  queryPath: string,
  identity: BlockedIdentity,
  opts?: { minAgreements?: number; threshold?: number },
): Promise<AppearanceQueryReport> {
  const minAgreements = opts?.minAgreements ?? ANALYZE_MIN_AGREEMENTS;
  const threshold = opts?.threshold ?? identity.threshold;
  const queryRaster = await loadFixtureRaster(queryPath);
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
    imagePath: queryPath,
    width: w,
    height: h,
    faceCount: dets.length,
    threshold,
    minAgreements,
    faces,
  };
}

export async function runAppearanceFixture(
  opts: {
    fixtureId: string;
    enrollPaths: string[];
    queryPath: string;
    identityId: string;
    enrollmentMode?: EnrollmentMode;
    minAgreements?: number;
    threshold?: number;
  } & FixtureAssetPaths,
): Promise<AppearanceFixtureReport> {
  const { detector, embedder } = await createFixtureModels(opts);
  const enrollmentMode = opts.enrollmentMode ?? "curated";
  const minAgreements = opts.minAgreements ?? ANALYZE_MIN_AGREEMENTS;
  const { enrollment, enrollFaceCounts, identity, threshold } = await buildProductionEnrollment(
    detector,
    embedder,
    opts.enrollPaths,
    enrollmentMode,
  );
  identity.id = opts.identityId;
  identity.displayName = opts.identityId;
  if (opts.threshold !== undefined) {
    identity.threshold = opts.threshold;
  }

  const query = await queryAppearanceImage(detector, embedder, opts.queryPath, identity, {
    minAgreements,
    threshold: opts.threshold ?? threshold,
  });

  return {
    fixtureId: opts.fixtureId,
    enrollPaths: opts.enrollPaths,
    queryPath: opts.queryPath,
    identityId: opts.identityId,
    enrollFaceCounts,
    enrollment,
    query,
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
