import { describe, expect, test } from "bun:test";
import { clusterEmbeddings } from "../../src/resolve/cluster.ts";
import { l2Normalize } from "../../src/matching/cosine.ts";
import { EMBED_DIM } from "../../src/shared/config.ts";
import { mulberry32, randomNormal, randomUnitVector } from "../../src/shared/rng.ts";

/** Unit vector near `direction`: the same face photographed under mild variation. */
function nearVec(direction: Float32Array, noise: number, rng: () => number): Float32Array {
  const v = new Float32Array(direction.length);
  for (let i = 0; i < v.length; i++) v[i] = direction[i]! + noise * randomNormal(rng);
  return l2Normalize(v);
}

function group(direction: Float32Array, count: number, noise: number, rng: () => number): Float32Array[] {
  const out: Float32Array[] = [];
  for (let i = 0; i < count; i++) out.push(nearVec(direction, noise, rng));
  return out;
}

function expectSortedUnique(indices: number[], n: number): void {
  for (let i = 1; i < indices.length; i++) {
    expect(indices[i]!).toBeGreaterThan(indices[i - 1]!);
  }
  expect(new Set(indices).size).toBe(indices.length);
  for (const i of indices) {
    expect(i).toBeGreaterThanOrEqual(0);
    expect(i).toBeLessThan(n);
  }
}

describe("clusterEmbeddings", () => {
  test("keeps the dominant group and rejects a smaller separated group", () => {
    const rng = mulberry32(42);
    // Two unrelated directions: near-orthogonal in 128 dims, so the groups split.
    const dirA = randomUnitVector(EMBED_DIM, mulberry32(1));
    const dirB = randomUnitVector(EMBED_DIM, mulberry32(2));
    const embeddings = [...group(dirA, 6, 0.05, rng), ...group(dirB, 3, 0.05, rng)];

    const r = clusterEmbeddings(embeddings);
    expect(r.kept).toEqual([0, 1, 2, 3, 4, 5]);
    expect(r.rejected.map((x) => x.index)).toEqual([6, 7, 8]);
    for (const rej of r.rejected) expect(rej.reason).toBe("outlier");
    expectSortedUnique(r.kept, embeddings.length);
    expectSortedUnique(r.prototypes, embeddings.length);
    for (const p of r.prototypes) expect(r.kept).toContain(p);
  });

  test("tight group of 8 keeps all, 2 far outliers rejected", () => {
    const rng = mulberry32(7);
    const dir = randomUnitVector(EMBED_DIM, mulberry32(10));
    const embeddings = [
      ...group(dir, 8, 0.03, rng),
      randomUnitVector(EMBED_DIM, mulberry32(20)),
      randomUnitVector(EMBED_DIM, mulberry32(21)),
    ];

    const r = clusterEmbeddings(embeddings);
    expect(r.kept.length).toBe(8);
    expect(r.rejected.length).toBe(2);
    expect(r.rejected.map((x) => x.index)).toEqual([8, 9]);
    for (const rej of r.rejected) expect(rej.reason).toBe("outlier");
    expectSortedUnique(r.kept, embeddings.length);
    expectSortedUnique(r.prototypes, embeddings.length);
  });

  test("prototype count respects maxPrototypes and stays duplicate-free", () => {
    const rng = mulberry32(99);
    const dir = randomUnitVector(EMBED_DIM, mulberry32(30));
    // Mild noise: one connected component, but enough spread for farthest-point
    // sampling to pick meaningfully different poses. In 128 dims the noise energy
    // scales with dim, so even 0.05 drops pairwise cosine to ~0.75.
    const embeddings = group(dir, 10, 0.05, rng);

    const capped = clusterEmbeddings(embeddings, { maxPrototypes: 3 });
    expect(capped.kept.length).toBe(10);
    expect(capped.prototypes.length).toBe(3);
    expectSortedUnique(capped.prototypes, embeddings.length);

    const uncapped = clusterEmbeddings(embeddings);
    expect(uncapped.prototypes.length).toBe(8);
    expectSortedUnique(uncapped.prototypes, embeddings.length);
  });

  test("empty input returns empty result", () => {
    expect(clusterEmbeddings([])).toEqual({ kept: [], rejected: [], prototypes: [] });
  });

  test("single embedding is kept and prototyped", () => {
    const r = clusterEmbeddings([randomUnitVector(EMBED_DIM, mulberry32(40))]);
    expect(r).toEqual({ kept: [0], rejected: [], prototypes: [0] });
  });

  test("identical embeddings cluster together with a single prototype", () => {
    const v = randomUnitVector(EMBED_DIM, mulberry32(50));
    const embeddings = [v, new Float32Array(v), new Float32Array(v), new Float32Array(v)];

    const r = clusterEmbeddings(embeddings);
    expect(r.kept).toEqual([0, 1, 2, 3]);
    expect(r.rejected).toEqual([]);
    // Duplicates collapse onto one input index instead of repeating the same vector.
    expect(r.prototypes).toEqual([0]);
  });

  test("no component reaching minCluster keeps the best-connected component", () => {
    const rng = mulberry32(5);
    const dirA = randomUnitVector(EMBED_DIM, mulberry32(60));
    const dirB = randomUnitVector(EMBED_DIM, mulberry32(61));
    // Two isolated pairs: largest component (2) is below the default minCluster (3).
    const embeddings = [...group(dirA, 2, 0.01, rng), ...group(dirB, 2, 0.01, rng)];

    const r = clusterEmbeddings(embeddings);
    expect(r.kept).toEqual([0, 1]);
    expect(r.rejected.map((x) => x.index)).toEqual([2, 3]);
    for (const rej of r.rejected) expect(rej.reason).toBe("small-cluster");
    expectSortedUnique(r.kept, embeddings.length);
    expectSortedUnique(r.prototypes, embeddings.length);
  });

  test("does not mutate input embeddings", () => {
    const rng = mulberry32(11);
    const dir = randomUnitVector(EMBED_DIM, mulberry32(70));
    const embeddings = group(dir, 5, 0.05, rng);
    const snapshot = embeddings.map((v) => new Float32Array(v));

    clusterEmbeddings(embeddings);
    for (let i = 0; i < embeddings.length; i++) {
      expect(embeddings[i]).toEqual(snapshot[i]!);
    }
  });
});
