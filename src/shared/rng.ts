/** Deterministic PRNG. Never use Math.random in the harness path. */

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomNormal(rng: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export function randomUnitVector(dim: number, rng: () => number): Float32Array {
  const v = new Float32Array(dim);
  let ss = 0;
  for (let i = 0; i < dim; i++) {
    const x = randomNormal(rng);
    v[i] = x;
    ss += x * x;
  }
  const inv = 1 / Math.sqrt(ss || 1);
  for (let i = 0; i < dim; i++) v[i]! *= inv;
  return v;
}
