import { describe, expect, test } from "bun:test";
import { EMBED_DIM } from "../../src/shared/config.ts";
import { cosine, l2Norm } from "../../src/matching/cosine.ts";
import { cosineDistance, selectPrototypes } from "../../src/matching/prototypes.ts";

function vec(fill: (i: number) => number): Float32Array {
  const v = new Float32Array(EMBED_DIM);
  for (let i = 0; i < EMBED_DIM; i++) v[i] = fill(i);
  return v;
}

/** Deterministic pseudo-random-ish vector from a seed. */
function seededVec(seed: number, scale = 1): Float32Array {
  return vec((i) => scale * Math.sin(seed * 12.9898 + i * 78.233));
}

describe("selectPrototypes", () => {
  test("empty input returns []", () => {
    expect(selectPrototypes([], 3)).toEqual([]);
  });

  test("maxPrototypes <= 0 returns []", () => {
    const e = [seededVec(1), seededVec(2)];
    expect(selectPrototypes(e, 0)).toEqual([]);
    expect(selectPrototypes(e, -2)).toEqual([]);
  });

  test("k >= n returns n normalized unique copies", () => {
    const e = [seededVec(1), seededVec(2), seededVec(3)];
    const protos = selectPrototypes(e, 10);
    expect(protos.length).toBe(3);
    for (const p of protos) {
      expect(l2Norm(p)).toBeCloseTo(1, 5);
      expect(p).not.toBe(e[0]);
    }
    // unique: no two prototypes identical
    for (let i = 0; i < protos.length; i++) {
      for (let j = i + 1; j < protos.length; j++) {
        expect(cosine(protos[i]!, protos[j]!)).toBeLessThan(0.9999);
      }
    }
    // each prototype matches a distinct source embedding
    const used = new Set<number>();
    for (const p of protos) {
      let best = -1;
      let bestC = -Infinity;
      for (let i = 0; i < e.length; i++) {
        const c = cosine(p, e[i]!);
        if (c > bestC) {
          bestC = c;
          best = i;
        }
      }
      expect(bestC).toBeCloseTo(1, 4);
      used.add(best);
    }
    expect(used.size).toBe(3);
  });

  test("two well-separated clusters with k=2 picks one from each", () => {
    const a1 = vec((i) => (i === 0 ? 1 : 0.001 * i));
    const a2 = vec((i) => (i === 0 ? 1 : -0.001 * i));
    const b1 = vec((i) => (i === 1 ? 1 : 0.001 * i));
    const b2 = vec((i) => (i === 1 ? 1 : -0.001 * i));
    const protos = selectPrototypes([a1, a2, b1, b2], 2);
    expect(protos.length).toBe(2);
    expect(cosine(protos[0]!, protos[1]!)).toBeLessThan(0.5);
  });

  test("does not mutate input embeddings", () => {
    const e = [seededVec(5), seededVec(6), seededVec(7)];
    const snapshot = e.map((v) => new Float32Array(v));
    selectPrototypes(e, 2);
    for (let i = 0; i < e.length; i++) {
      expect(e[i]).toEqual(snapshot[i]!);
    }
  });

  test("returned prototypes are copies, not aliases", () => {
    const e = [seededVec(8), seededVec(9)];
    const protos = selectPrototypes(e, 2);
    for (const p of protos) {
      expect(e.includes(p)).toBe(false);
      p[0] = 999;
    }
    // mutating a prototype must not affect inputs
    for (const v of e) expect(v[0]).not.toBe(999);
  });
});

describe("cosineDistance", () => {
  test("identical direction -> 0", () => {
    const a = seededVec(1);
    expect(cosineDistance(a, a)).toBeCloseTo(0, 5);
  });

  test("orthogonal -> 1", () => {
    const a = vec((i) => (i === 0 ? 1 : 0));
    const b = vec((i) => (i === 1 ? 1 : 0));
    expect(cosineDistance(a, b)).toBeCloseTo(1, 5);
  });

  test("opposite -> 2", () => {
    const a = vec((i) => (i === 0 ? 1 : 0));
    const b = vec((i) => (i === 0 ? -1 : 0));
    expect(cosineDistance(a, b)).toBeCloseTo(2, 5);
  });
});
