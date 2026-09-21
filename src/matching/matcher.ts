/** Gallery matching: query embedding vs blocked identities. */

import {
  DEFAULT_THRESHOLD,
  HARD_NEGATIVE_MARGIN,
  MIN_GALLERY_AGREEMENTS,
} from "../shared/config.ts";
import type { BlockedIdentity, MatchResult } from "../shared/types.ts";
import { cosineNormalized, l2Normalize } from "./cosine.ts";

function topKCosines(
  q: Float32Array,
  gallery: readonly Float32Array[],
  k: number,
): Float64Array {
  const top = new Float64Array(k);
  top.fill(Number.NEGATIVE_INFINITY);
  for (const g of gallery) {
    const s = cosineNormalized(q, g);
    if (k === 0 || s <= top[k - 1]!) continue;
    let i = k - 1;
    while (i > 0 && s > top[i - 1]!) {
      top[i] = top[i - 1]!;
      i--;
    }
    top[i] = s;
  }
  return top;
}

/**
 * Best-scoring identity whose gallery clears its threshold, or null.
 * Hard negatives veto a match when (score - maxNegScore) < HARD_NEGATIVE_MARGIN.
 */
export function matchFace(
  query: Float32Array,
  identities: readonly BlockedIdentity[],
  opts?: { defaultThreshold?: number; minAgreements?: number },
): MatchResult | null {
  if (identities.length === 0) return null;
  const q = l2Normalize(query);
  let best: MatchResult | null = null;
  for (const identity of identities) {
    const gallery =
      identity.prototypes && identity.prototypes.length > 0
        ? identity.prototypes
        : identity.embeddings;
    if (gallery.length === 0) continue;
    const kNeed = Math.min(opts?.minAgreements ?? MIN_GALLERY_AGREEMENTS, gallery.length);
    const top = topKCosines(q, gallery, kNeed);
    const score = top[0]!;
    const negs = identity.hardNegatives;
    if (negs && negs.length > 0) {
      const neg = topKCosines(q, negs, 1)[0]!;
      if (score - neg < HARD_NEGATIVE_MARGIN) continue;
    }
    const thresh = Number.isFinite(identity.threshold)
      ? identity.threshold
      : (opts?.defaultThreshold ?? DEFAULT_THRESHOLD);
    if (score >= thresh && top[kNeed - 1]! >= thresh && (best === null || score > best.score)) {
      best = { identityId: identity.id, score };
    }
  }
  return best;
}
