/** Turn candidate face embeddings into one consistent cluster plus diverse prototypes. */

import { cosineNormalized, l2Normalize } from "../matching/cosine.ts";
import { selectPrototypes } from "../matching/prototypes.ts";

export interface ClusterRejection {
  index: number;
  reason: string;
}

export interface ClusterResult {
  kept: number[]; // indices into the input array
  rejected: ClusterRejection[];
  prototypes: number[]; // indices chosen for pose/lighting diversity
}

/**
 * Neighbor radius in cosine distance: embeddings closer than this are linked.
 * Operating point, not a model claim — loose enough that the same person across
 * angles/lighting stays connected, tight enough that a different person splits off.
 */
const DEFAULT_RADIUS = 0.35;

/** Below this many members the dominant component is too weak to trust. */
const DEFAULT_MIN_CLUSTER = 3;

/** Union-find over embedding indices; roots are arbitrary, only membership matters. */
function makeUnionFind(n: number): { find: (i: number) => number; union: (a: number, b: number) => void } {
  const parent = new Int32Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;
  const find = (i: number): number => {
    let r = i;
    while (parent[r] !== r) r = parent[r]!;
    while (parent[i] !== r) {
      const next = parent[i]!;
      parent[i] = r;
      i = next;
    }
    return r;
  };
  const union = (a: number, b: number): void => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[rb] = ra;
  };
  return { find, union };
}

/**
 * Cluster candidate embeddings and pick diverse prototypes from the dominant cluster.
 * Works on L2-normalized copies; inputs are never mutated.
 *
 * The similarity graph links embeddings whose cosine similarity >= 1 - radius.
 * The largest connected component is assumed to be the real person repeated across
 * photos; everything outside it is rejected as an outlier. When even the largest
 * component is smaller than `minCluster`, the component of the best-connected node
 * is kept anyway and the rest are rejected as "small-cluster" — a low-confidence
 * result the caller can surface rather than silently trusting.
 */
export function clusterEmbeddings(
  embeddings: readonly Float32Array[],
  opts?: { radius?: number; maxPrototypes?: number; minCluster?: number },
): ClusterResult {
  const n = embeddings.length;
  if (n === 0) return { kept: [], rejected: [], prototypes: [] };
  if (n === 1) return { kept: [0], rejected: [], prototypes: [0] };

  const radius = opts?.radius ?? DEFAULT_RADIUS;
  const minCluster = opts?.minCluster ?? DEFAULT_MIN_CLUSTER;
  const maxPrototypes = opts?.maxPrototypes ?? 8;
  const minSimilarity = 1 - radius;

  const normed = embeddings.map((e) => l2Normalize(e));

  // O(n^2) pairwise linking is fine: candidate sets are capped at a few hundred.
  const uf = makeUnionFind(n);
  const degree = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (cosineNormalized(normed[i]!, normed[j]!) >= minSimilarity) {
        uf.union(i, j);
        degree[i]!++;
        degree[j]!++;
      }
    }
  }

  // Group members by component; first-seen order keeps ties deterministic.
  const components = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const root = uf.find(i);
    const members = components.get(root);
    if (members) members.push(i);
    else components.set(root, [i]);
  }

  let dominant: number[] = [];
  for (const members of components.values()) {
    if (members.length > dominant.length) dominant = members;
  }

  let reason = "outlier";
  if (dominant.length < minCluster) {
    // Low-confidence path: keep the component of the best-connected node instead
    // of the largest component, since a dense tiny clique is a better guess than a
    // sparse chain that merely has one more member.
    let best = 0;
    for (let i = 1; i < n; i++) {
      if (degree[i]! > degree[best]!) best = i;
    }
    dominant = components.get(uf.find(best))!;
    reason = "small-cluster";
  }

  const keptSet = new Set(dominant);
  const kept = [...keptSet].sort((a, b) => a - b);
  const rejected: ClusterRejection[] = [];
  for (let i = 0; i < n; i++) {
    if (!keptSet.has(i)) rejected.push({ index: i, reason });
  }

  // Farthest-point sampling over the kept embeddings deliberately spans different
  // angles and lighting, so the gallery generalizes instead of duplicating one pose.
  const keptEmbeddings = kept.map((i) => embeddings[i]!);
  const protos = selectPrototypes(keptEmbeddings, maxPrototypes);

  // Map each returned prototype back to its input index by nearest cosine. Exact
  // duplicates inside `kept` collapse onto the first matching index, so identical
  // embeddings yield one prototype instead of several copies of the same vector.
  const protoSet = new Set<number>();
  for (const p of protos) {
    let bestIdx = -1;
    let bestCos = -Infinity;
    for (const i of kept) {
      const c = cosineNormalized(p, normed[i]!);
      if (c > bestCos) {
        bestCos = c;
        bestIdx = i;
      }
    }
    if (bestIdx >= 0) protoSet.add(bestIdx);
  }
  const prototypes = [...protoSet].sort((a, b) => a - b);

  return { kept, rejected, prototypes };
}
