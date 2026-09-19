import { describe, expect, test } from "bun:test";
import { cosine, cosineNormalized, dot, l2Norm, l2Normalize } from "../../src/matching/cosine.ts";

describe("cosine", () => {
  test("self-similarity is 1", () => {
    const v = l2Normalize(new Float32Array([3, 4, 0]));
    expect(l2Norm(v)).toBeCloseTo(1, 6);
    expect(cosineNormalized(v, v)).toBeCloseTo(1, 6);
    expect(cosine(v, v)).toBeCloseTo(1, 6);
  });

  test("orthogonal vectors are ~0", () => {
    const a = l2Normalize(new Float32Array([1, 0, 0]));
    const b = l2Normalize(new Float32Array([0, 1, 0]));
    expect(cosineNormalized(a, b)).toBeCloseTo(0, 6);
  });

  test("opposite vectors are -1", () => {
    const a = l2Normalize(new Float32Array([1, 2, 3]));
    const b = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) b[i] = -a[i]!;
    expect(cosineNormalized(a, b)).toBeCloseTo(-1, 6);
  });

  test("dot throws on length mismatch", () => {
    expect(() => dot(new Float32Array(2), new Float32Array(3))).toThrow();
  });
});
