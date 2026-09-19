/** Gallery matching: query embedding vs blocked identities. */

import { DEFAULT_THRESHOLD, HARD_NEGATIVE_MARGIN } from "../shared/config.ts";
import type { BlockedIdentity, MatchResult } from "../shared/types.ts";
import { cosineNormalized, l2Normalize } from "./cosine.ts";

function maxCosine(q: Float32Array, gallery: readonly Float32Array[]): number {
  let best = -Infinity;
  for (const g of gallery) {
    const s = cosineNormalized(q, l2Normalize(g));
    if (s > best) best = s;
  }
  return best;
}

/**
 * Best-scoring identity whose gallery clears its threshold, or null.
 * Hard negatives veto a match when (score - maxNegScore) < HARD_NEGATIVE_MARGIN.
 */
export function matchFace(
  query: Float32Array,
  identities: readonly BlockedIdentity[],
  opts?: { defaultThreshold?: number },
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
    const score = maxCosine(q, gallery);
    const negs = identity.hardNegatives;
    if (negs && negs.length > 0) {
      const neg = maxCosine(q, negs);
      if (score - neg < HARD_NEGATIVE_MARGIN) continue;
    }
    const thresh = Number.isFinite(identity.threshold)
      ? identity.threshold
      : (opts?.defaultThreshold ?? DEFAULT_THRESHOLD);
    if (score >= thresh && (best === null || score > best.score)) {
      best = { identityId: identity.id, score };
    }
  }
  return best;
}
