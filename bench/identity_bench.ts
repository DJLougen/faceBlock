/**
 * Synthetic ArcFace-like identity matching benchmark.
 *
 * Measures matcher quality on L2-normalized embeddings with a known generative
 * process (same-id gaussian jitter, independent distractors). This is NOT a
 * claim about a face detector or MobileFaceNet — only about matching.
 *
 * Primary product metric (plan §28): false censors per 1,000 unrelated faces.
 */
import { matchFace } from "../src/matching/matcher.ts";
import { cosineNormalized, l2Normalize } from "../src/matching/cosine.ts";
import { DEFAULT_THRESHOLD, EMBED_DIM } from "../src/shared/config.ts";
import { mulberry32, randomNormal, randomUnitVector } from "../src/shared/rng.ts";
import type { BlockedIdentity } from "../src/shared/types.ts";

const SEED = 42;
const N_IDENTITIES = 20;
const REFS_PER = 8;
const POS_PER = 25;
const N_NEGATIVES = 2000;
/** Per-dim gaussian jitter. In 128-d, 0.044 → same-id cosine ~0.80. */
const SAME_NOISE = 0.044;
const HARD_POS_FRACTION = 0.2;
const HARD_POS_NOISE = 0.085;
const HARD_NEG_FRACTION = 0.2;
const HARD_NEG_BLEND = 0.52;
const THRESHOLD = DEFAULT_THRESHOLD;
/** Fail closed if the operating point censors nobody — FPR=0 would be a fake win. */
const TPR_FLOOR = 0.5;
const LATENCY_QUERIES = 1000;
const LATENCY_GALLERY_IDS = 100;
const LATENCY_EMB_PER_ID = 10;

function jitter(mean: Float32Array, noise: number, rng: () => number): Float32Array {
  const v = new Float32Array(mean.length);
  for (let i = 0; i < mean.length; i++) {
    v[i] = mean[i]! + noise * randomNormal(rng);
  }
  return l2Normalize(v);
}

function blend(a: Float32Array, b: Float32Array, t: number): Float32Array {
  const v = new Float32Array(a.length);
  const u = 1 - t;
  for (let i = 0; i < a.length; i++) v[i] = u * a[i]! + t * b[i]!;
  return l2Normalize(v);
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return NaN;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx]!;
}

function auc(pos: number[], neg: number[]): number {
  const nPos = pos.length;
  const nNeg = neg.length;
  if (nPos === 0 || nNeg === 0) return NaN;
  let wins = 0;
  let ties = 0;
  for (let i = 0; i < nPos; i++) {
    const p = pos[i]!;
    for (let j = 0; j < nNeg; j++) {
      const n = neg[j]!;
      if (p > n) wins++;
      else if (p === n) ties++;
    }
  }
  return (wins + 0.5 * ties) / (nPos * nNeg);
}

function galleryScore(query: Float32Array, embeddings: Float32Array[]): number {
  let best = -Infinity;
  for (let i = 0; i < embeddings.length; i++) {
    const s = cosineNormalized(query, embeddings[i]!);
    if (s > best) best = s;
  }
  return best;
}

function fail(msg: string): never {
  console.error(`GATE_FAIL ${msg}`);
  process.exit(1);
}

function main(): void {
  const rng = mulberry32(SEED);

  const means: Float32Array[] = [];
  const identities: BlockedIdentity[] = [];
  for (let i = 0; i < N_IDENTITIES; i++) {
    const mean = randomUnitVector(EMBED_DIM, rng);
    means.push(mean);
    const embeddings: Float32Array[] = [];
    for (let r = 0; r < REFS_PER; r++) embeddings.push(jitter(mean, SAME_NOISE, rng));
    identities.push({
      id: `id_${i}`,
      embeddings,
      threshold: THRESHOLD,
      createdAt: 0,
    });
  }

  const posQueries: { q: Float32Array; id: string }[] = [];
  const nHardPos = Math.round(POS_PER * HARD_POS_FRACTION);
  for (let i = 0; i < N_IDENTITIES; i++) {
    for (let p = 0; p < POS_PER; p++) {
      const noise = p < nHardPos ? HARD_POS_NOISE : SAME_NOISE;
      posQueries.push({ q: jitter(means[i]!, noise, rng), id: `id_${i}` });
    }
  }

  const negQueries: Float32Array[] = [];
  const nHard = Math.round(N_NEGATIVES * HARD_NEG_FRACTION);
  for (let i = 0; i < N_NEGATIVES; i++) {
    if (i < nHard) {
      const target = means[i % N_IDENTITIES]!;
      const other = randomUnitVector(EMBED_DIM, rng);
      negQueries.push(blend(other, target, HARD_NEG_BLEND));
    } else {
      negQueries.push(randomUnitVector(EMBED_DIM, rng));
    }
  }

  const enrolled = identities[0]!.embeddings[0]!;
  const selfHit = matchFace(enrolled, identities);
  if (selfHit === null || selfHit.identityId !== "id_0") {
    fail("enrolled self-embedding did not match id_0");
  }
  if (Math.abs(cosineNormalized(enrolled, enrolled) - 1) > 1e-5) {
    fail("self cosine is not 1");
  }

  let tp = 0;
  let fn = 0;
  const posScores: number[] = [];
  for (let i = 0; i < posQueries.length; i++) {
    const item = posQueries[i]!;
    const hit = matchFace(item.q, identities);
    const score = galleryScore(item.q, identities.find((x) => x.id === item.id)!.embeddings);
    posScores.push(score);
    if (hit !== null && hit.identityId === item.id) tp++;
    else fn++;
  }

  let fp = 0;
  const negScores: number[] = [];
  for (let i = 0; i < negQueries.length; i++) {
    const q = negQueries[i]!;
    const hit = matchFace(q, identities);
    let best = -Infinity;
    for (let k = 0; k < identities.length; k++) {
      const s = galleryScore(q, identities[k]!.embeddings);
      if (s > best) best = s;
    }
    negScores.push(best);
    if (hit !== null) fp++;
  }

  const nPos = posQueries.length;
  const nNeg = negQueries.length;
  if (nPos === 0 || nNeg === 0) fail("empty evaluation split");

  const tpr = tp / nPos;
  const falseCensorsPer1k = (1000 * fp) / nNeg;
  const fnPer100 = (100 * fn) / nPos;
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const rocAuc = auc(posScores, negScores);
  if (tpr < TPR_FLOOR) {
    fail(`tpr ${tpr} below floor ${TPR_FLOOR} (operating point is not measuring FPR)`);
  }

  for (const [name, value] of [
    ["tpr", tpr],
    ["false_censors_per_1k", falseCensorsPer1k],
    ["fn_per_100", fnPer100],
    ["precision", precision],
    ["roc_auc", rocAuc],
  ] as const) {
    if (!Number.isFinite(value)) fail(`non-finite ${name}`);
  }

  const latIdentities: BlockedIdentity[] = [];
  for (let i = 0; i < LATENCY_GALLERY_IDS; i++) {
    const embeddings: Float32Array[] = [];
    const mean = randomUnitVector(EMBED_DIM, rng);
    for (let r = 0; r < LATENCY_EMB_PER_ID; r++) embeddings.push(jitter(mean, SAME_NOISE, rng));
    latIdentities.push({
      id: `lat_${i}`,
      embeddings,
      threshold: THRESHOLD,
      createdAt: 0,
    });
  }
  const latQueries: Float32Array[] = [];
  for (let i = 0; i < LATENCY_QUERIES; i++) latQueries.push(randomUnitVector(EMBED_DIM, rng));

  for (let i = 0; i < 32; i++) matchFace(latQueries[i % latQueries.length]!, latIdentities);

  const timesUs: number[] = [];
  const tBatch0 = performance.now();
  for (let i = 0; i < LATENCY_QUERIES; i++) {
    const t0 = performance.now();
    matchFace(latQueries[i]!, latIdentities);
    timesUs.push((performance.now() - t0) * 1000);
  }
  const batchMs = performance.now() - tBatch0;
  timesUs.sort((a, b) => a - b);
  const matchP50Us = percentile(timesUs, 50);
  const matchP95Us = percentile(timesUs, 95);
  const matchMeanUs = (batchMs * 1000) / LATENCY_QUERIES;

  if (!Number.isFinite(matchP50Us) || !Number.isFinite(matchP95Us)) fail("non-finite latency");

  console.log(`METRIC false_censors_per_1k=${falseCensorsPer1k}`);
  console.log(`METRIC tpr=${tpr}`);
  console.log(`METRIC fn_per_100=${fnPer100}`);
  console.log(`METRIC precision=${precision}`);
  console.log(`METRIC roc_auc=${rocAuc}`);
  console.log(`METRIC match_p50_us=${matchP50Us}`);
  console.log(`METRIC match_p95_us=${matchP95Us}`);
  console.log(`METRIC match_mean_us=${matchMeanUs}`);
  console.log(`METRIC tp=${tp}`);
  console.log(`METRIC fp=${fp}`);
  console.log(`METRIC fn=${fn}`);

  console.log(`ASI seed=${SEED}`);
  console.log(`ASI dim=${EMBED_DIM}`);
  console.log(`ASI n_identities=${N_IDENTITIES}`);
  console.log(`ASI n_negatives=${N_NEGATIVES}`);
  console.log(`ASI threshold=${THRESHOLD}`);
  console.log(`ASI same_noise=${SAME_NOISE}`);
  console.log(`ASI hard_neg_fraction=${HARD_NEG_FRACTION}`);
  console.log(`ASI hard_neg_blend=${HARD_NEG_BLEND}`);
  console.log(`ASI tpr_floor=${TPR_FLOOR}`);

  const posSorted = posScores.slice().sort((a, b) => a - b);
  const negSorted = negScores.slice().sort((a, b) => a - b);
  console.error(
    `bench n_pos=${nPos} n_neg=${nNeg} tp=${tp} fp=${fp} fn=${fn} tpr=${tpr.toFixed(4)} ` +
      `false_censors_per_1k=${falseCensorsPer1k.toFixed(4)} roc_auc=${rocAuc.toFixed(4)} ` +
      `pos_p50=${percentile(posSorted, 50).toFixed(4)} pos_p05=${percentile(posSorted, 5).toFixed(4)} ` +
      `neg_p95=${percentile(negSorted, 95).toFixed(4)} neg_p50=${percentile(negSorted, 50).toFixed(4)} ` +
      `match_p50_us=${matchP50Us.toFixed(3)} match_p95_us=${matchP95Us.toFixed(3)}`,
  );
}

main();
