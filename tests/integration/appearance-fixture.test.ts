/**
 * ONNX appearance-change fixture — NOT geometry-only coverage.
 *
 * eef58d1 padding does not fix embedding drift; this test records detection,
 * bestCosine, matched, and mask bounds on a local same-person pair.
 */
import { describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import {
  primarySubjectFace,
  runAppearanceFixture,
  DEFAULT_MODEL_ROOT,
} from "../../src/analysis/fixtureRunner.ts";
import { MATCH_THRESHOLD } from "../../extension/enroll.ts";

const ROOT = resolve(import.meta.dir, "../..");
const MODEL = resolve(DEFAULT_MODEL_ROOT, "models/w600k_mbf.onnx");
const ENROLL = resolve(ROOT, "demo/public/samples/theo.jpg");
const QUERY = resolve(ROOT, "demo/public/samples/theo-user-02.jpg");
const modelsPresent = existsSync(MODEL) && existsSync(ENROLL) && existsSync(QUERY);

describe.skipIf(!modelsPresent)("appearance fixture (ONNX)", () => {
  test("theo.jpg enroll → theo-user-02 query: detect, cosine, mask bounds", async () => {
    const report = await runAppearanceFixture({
      fixtureId: "theo-clean-to-user-labeled",
      enrollPaths: [ENROLL],
      queryPath: QUERY,
      identityId: "theo-browne",
    });

    expect(report.enrollFaceCounts).toEqual([1]);
    expect(report.query.faceCount).toBeGreaterThanOrEqual(1);
    expect(report.query.threshold).toBe(MATCH_THRESHOLD);

    const primary = primarySubjectFace(report.query);
    expect(primary).not.toBeNull();

    // Measured on w600k_mbf + YuNet 2026may (2026-09-21).
    expect(primary!.confidence).toBeGreaterThan(0.8);
    expect(primary!.bestCosine).not.toBeNull();
    expect(primary!.bestCosine!).toBeGreaterThan(0.35);
    expect(primary!.bestCosine!).toBeLessThan(0.45);

    // Without appearance slack this pair false-negatives at 0.397 vs threshold 0.4.
    expect(primary!.matched).toBe(true);

    // Padding change grows the mask when a match exists — geometry alone is insufficient.
    expect(primary!.maskCurrent.height).toBeGreaterThan(primary!.maskLegacyAnalyzer.height);
    expect(primary!.maskCurrent.y).toBeLessThan(primary!.maskLegacyAnalyzer.y);
  }, 60_000);
});
