/** L2 / cosine primitives. Matching assumes embeddings are L2-normalized. */

export function l2Norm(v: Float32Array): number {
  let ss = 0;
  for (let i = 0; i < v.length; i++) ss += v[i]! * v[i]!;
  return Math.sqrt(ss);
}

export function l2Normalize(v: Float32Array): Float32Array {
  const out = new Float32Array(v.length);
  const n = l2Norm(v);
  const inv = n === 0 ? 1 : 1 / n;
  for (let i = 0; i < v.length; i++) out[i] = v[i]! * inv;
  return out;
}

export function l2NormalizeInPlace(v: Float32Array): Float32Array {
  const n = l2Norm(v);
  const inv = n === 0 ? 1 : 1 / n;
  for (let i = 0; i < v.length; i++) v[i] = v[i]! * inv;
  return v;
}

export function dot(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(`dot length mismatch: ${a.length} vs ${b.length}`);
  }
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

/** Cosine for already-L2-normalized vectors: similarity = dot. */
export function cosineNormalized(a: Float32Array, b: Float32Array): number {
  return dot(a, b);
}

/** Cosine after L2-normalizing copies. */
export function cosine(a: Float32Array, b: Float32Array): number {
  return dot(l2Normalize(a), l2Normalize(b));
}
