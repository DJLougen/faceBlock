import { describe, expect, test } from "bun:test";
import { cosineNormalized, l2Normalize } from "../../src/matching/cosine.ts";
import { matchFace } from "../../src/matching/matcher.ts";
import {
  DEFAULT_THRESHOLD,
  HARD_NEGATIVE_MARGIN,
  shouldCensor,
} from "../../src/matching/thresholds.ts";
import { EMBED_DIM } from "../../src/shared/config.ts";
import { mulberry32, randomUnitVector } from "../../src/shared/rng.ts";
import type { BlockedIdentity } from "../../src/shared/types.ts";

function identity(partial: Partial<BlockedIdentity> & { id: string }): BlockedIdentity {
  return { embeddings: [], threshold: DEFAULT_THRESHOLD, createdAt: 0, ...partial };
}

describe("matchFace", () => {
  test("self-embedding matches enrolled identity", () => {
    const emb = randomUnitVector(EMBED_DIM, mulberry32(1));
    const result = matchFace(emb, [identity({ id: "alice", embeddings: [emb] })]);
    expect(result).not.toBeNull();
    expect(result!.identityId).toBe("alice");
    expect(result!.score).toBeCloseTo(1, 5);
  });

  test("single-ref gallery still matches (k falls back to 1)", () => {
    const emb = randomUnitVector(EMBED_DIM, mulberry32(21));
    const result = matchFace(emb, [identity({ id: "alice", embeddings: [emb] })]);
    expect(result).not.toBeNull();
  });

  test("one spike among two refs does not censor", () => {
    const emb = randomUnitVector(EMBED_DIM, mulberry32(22));
    const decoy = randomUnitVector(EMBED_DIM, mulberry32(23));
    expect(cosineNormalized(emb, decoy)).toBeLessThan(DEFAULT_THRESHOLD);
    const result = matchFace(emb, [
      identity({ id: "alice", embeddings: [emb, decoy] }),
    ]);
    expect(result).toBeNull();
  });

  test("two agreeing refs still censor", () => {
    const emb = randomUnitVector(EMBED_DIM, mulberry32(24));
    const twin = new Float32Array(emb);
    const result = matchFace(emb, [
      identity({ id: "alice", embeddings: [emb, twin] }),
    ]);
    expect(result).not.toBeNull();
    expect(result!.identityId).toBe("alice");
  });

  test("unrelated unit vector below 0.7 does not match", () => {
    const emb = randomUnitVector(EMBED_DIM, mulberry32(2));
    const query = randomUnitVector(EMBED_DIM, mulberry32(3));
    // Guard the premise: these seeds really are unrelated.
    expect(cosineNormalized(query, emb)).toBeLessThan(DEFAULT_THRESHOLD);
    expect(matchFace(query, [identity({ id: "alice", embeddings: [emb] })])).toBeNull();
  });

  test("hard negative near the query blocks censor", () => {
    const emb = randomUnitVector(EMBED_DIM, mulberry32(4));
    // Query is the enrolled embedding itself: score ~1.
    // A hard negative at cosine ~0.999 leaves score - neg < HARD_NEGATIVE_MARGIN.
    const jitter = randomUnitVector(EMBED_DIM, mulberry32(5));
    const neg = new Float32Array(EMBED_DIM);
    for (let i = 0; i < EMBED_DIM; i++) neg[i] = emb[i]! + 0.05 * jitter[i]!;
    const negN = l2Normalize(neg);
    expect(1 - cosineNormalized(emb, negN)).toBeLessThan(HARD_NEGATIVE_MARGIN);
    const result = matchFace(emb, [
      identity({ id: "alice", embeddings: [emb], hardNegatives: [negN] }),
    ]);
    expect(result).toBeNull();
  });

  test("distant hard negative does not block", () => {
    const emb = randomUnitVector(EMBED_DIM, mulberry32(6));
    const neg = randomUnitVector(EMBED_DIM, mulberry32(7));
    const result = matchFace(emb, [
      identity({ id: "alice", embeddings: [emb], hardNegatives: [neg] }),
    ]);
    expect(result).not.toBeNull();
    expect(result!.identityId).toBe("alice");
  });

  test("empty identities returns null", () => {
    const query = randomUnitVector(EMBED_DIM, mulberry32(8));
    expect(matchFace(query, [])).toBeNull();
  });

  test("prototypes are used instead of raw embeddings when present", () => {
    const proto = randomUnitVector(EMBED_DIM, mulberry32(9));
    const decoy = randomUnitVector(EMBED_DIM, mulberry32(10));
    expect(cosineNormalized(proto, decoy)).toBeLessThan(DEFAULT_THRESHOLD);
    // Raw embeddings would not match; only the prototype does.
    const result = matchFace(proto, [
      identity({ id: "alice", embeddings: [decoy], prototypes: [proto] }),
    ]);
    expect(result).not.toBeNull();
    expect(result!.identityId).toBe("alice");
  });

  test("identity with empty gallery is skipped", () => {
    const query = randomUnitVector(EMBED_DIM, mulberry32(11));
    expect(matchFace(query, [identity({ id: "alice" })])).toBeNull();
  });

  test("best-scoring passing identity wins", () => {
    const emb = randomUnitVector(EMBED_DIM, mulberry32(12));
    const far = randomUnitVector(EMBED_DIM, mulberry32(13));
    const result = matchFace(emb, [
      identity({ id: "low", embeddings: [far], threshold: -1 }),
      identity({ id: "high", embeddings: [emb] }),
    ]);
    expect(result!.identityId).toBe("high");
  });

  test("non-finite identity threshold falls back to opts.defaultThreshold", () => {
    const emb = randomUnitVector(EMBED_DIM, mulberry32(14));
    const strict = identity({ id: "alice", embeddings: [emb], threshold: NaN });
    expect(matchFace(emb, [strict], { defaultThreshold: 1.1 })).toBeNull();
    expect(matchFace(emb, [strict], { defaultThreshold: 0.5 })).not.toBeNull();
  });
});

describe("shouldCensor", () => {
  test("score >= threshold", () => {
    expect(shouldCensor(0.7, 0.7)).toBe(true);
    expect(shouldCensor(0.69, 0.7)).toBe(false);
  });
});
