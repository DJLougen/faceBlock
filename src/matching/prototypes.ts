/** Farthest-point prototype selection over L2-normalized embeddings. */

import { cosineNormalized, dot, l2Norm, l2Normalize } from "./cosine.ts";

/** Cosine distance d = 1 - cosine similarity, computed on normalized copies. */
export function cosineDistance(a: Float32Array, b: Float32Array): number {
  return 1 - cosineNormalized(l2Normalize(a), l2Normalize(b));
}

/**
 * Farthest-point sampling on L2-normalized embeddings under cosine distance.
 * Returns up to `maxPrototypes` new L2-normalized Float32Array copies.
 * Inputs are never mutated.
 */
export function selectPrototypes(
  embeddings: Float32Array[],
  maxPrototypes: number,
): Float32Array[] {
  if (embeddings.length === 0 || maxPrototypes <= 0) return [];

  const k = Math.min(maxPrototypes, embeddings.length);
  const normed = embeddings.map((e) => l2Normalize(e));

  // Seed with the embedding of largest pre-normalization L2 norm (index 0 on ties).
  let seed = 0;
  let bestNorm = l2Norm(embeddings[0]!);
  for (let i = 1; i < embeddings.length; i++) {
    const n = l2Norm(embeddings[i]!);
    if (n > bestNorm) {
      bestNorm = n;
      seed = i;
    }
  }

  const selected: number[] = [seed];
  // minDist[i] = min cosine distance from point i to the selected set.
  const minDist = new Float64Array(embeddings.length);
  for (let i = 0; i < embeddings.length; i++) {
    minDist[i] = i === seed ? -Infinity : 1 - dot(normed[i]!, normed[seed]!);
  }

  while (selected.length < k) {
    let best = -1;
    let bestD = -Infinity;
    for (let i = 0; i < embeddings.length; i++) {
      if (minDist[i]! > bestD) {
        bestD = minDist[i]!;
        best = i;
      }
    }
    selected.push(best);
    for (let i = 0; i < embeddings.length; i++) {
      const d = 1 - dot(normed[i]!, normed[best]!);
      if (d < minDist[i]!) minDist[i] = d;
    }
    minDist[best] = -Infinity;
  }

  return selected.map((i) => new Float32Array(normed[i]!));
}
