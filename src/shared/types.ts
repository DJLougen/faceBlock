/** Canonical types for FaceBlock v0. Do not invent parallel shapes. */

export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Size {
  width: number;
  height: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface FaceDetection {
  box: Box;
  confidence: number;
  landmarks?: Point[];
}

export interface MediaTarget {
  id: string;
  element?: unknown;
  sourceUrl?: string;
  width: number;
  height: number;
}

export interface BlockedIdentity {
  id: string;
  displayName?: string;
  embeddings: Float32Array[];
  prototypes?: Float32Array[];
  hardNegatives?: Float32Array[];
  threshold: number;
  createdAt: number;
}

export interface CensorRegion {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  identityId: string;
}

export interface MatchResult {
  identityId: string;
  score: number;
}

export type ObjectFit = "fill" | "contain" | "cover" | "none" | "scale-down";

/** RGBA raster, row-major, 4 bytes per pixel. */
export interface Raster {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export interface SimilarityTransform {
  a: number;
  b: number;
  tx: number;
  ty: number;
}
