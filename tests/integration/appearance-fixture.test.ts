/**
 * ONNX appearance-change fixtures — diagnostics and calibrated match expectations.
 *
 * Padding does not fix embedding drift. These tests record detection, cosine,
 * matched, and mask bounds. Threshold slack is intentionally NOT applied;
 * tune only from measured FAR/FRR on this corpus (or improve enroll diversity).
 */
import { describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import {
  ANALYZE_MIN_AGREEMENTS,
  buildProductionEnrollment,
  createFixtureModels,
  primarySubjectFace,
  queryAppearanceImage,
  runAppearanceFixture,
  DEFAULT_MODEL_ROOT,
} from "../../src/analysis/fixtureRunner.ts";
import { MATCH_THRESHOLD } from "../../extension/enroll.ts";

const ROOT = resolve(import.meta.dir, "../..");
const MODEL = resolve(DEFAULT_MODEL_ROOT, "models/w600k_mbf.onnx");
const THEO = resolve(ROOT, "demo/public/samples/theo.jpg");
const THEO_USER02 = resolve(ROOT, "demo/public/samples/theo-user-02.jpg");
const DWARKESH = resolve(ROOT, "demo/public/samples/dwarkesh.jpg");
const UNRELATED = resolve(ROOT, "demo/public/samples/unrelated.jpg");

const samplesPresent =
  existsSync(MODEL) &&
  existsSync(THEO) &&
  existsSync(THEO_USER02) &&
  existsSync(DWARKESH) &&
  existsSync(UNRELATED);

describe.skipIf(!samplesPresent)("appearance fixture (ONNX)", () => {
  test("appearance-drift probe: single ref false-negatives at threshold", async () => {
    const report = await runAppearanceFixture({
      fixtureId: "theo-appearance-drift-probe",
      enrollPaths: [THEO],
      queryPath: THEO_USER02,
      identityId: "theo-browne",
      enrollmentMode: "curated",
    });

    expect(report.enrollment.gallerySize).toBe(1);
    expect(report.enrollment.meetsMinReferenceImages).toBe(false);
    expect(report.enrollFaceCounts).toEqual([1]);
    expect(report.query.faceCount).toBeGreaterThanOrEqual(1);
    expect(report.query.threshold).toBe(MATCH_THRESHOLD);
    expect(report.query.minAgreements).toBe(ANALYZE_MIN_AGREEMENTS);

    const primary = primarySubjectFace(report.query);
    expect(primary).not.toBeNull();

    // Measured on w600k_mbf + YuNet 2026may (2026-09-21).
    expect(primary!.confidence).toBeGreaterThan(0.8);
    expect(primary!.bestCosine).not.toBeNull();
    expect(primary!.bestCosine!).toBeGreaterThan(0.35);
    expect(primary!.bestCosine!).toBeLessThan(0.45);

    // Same-person appearance drift sits just under MATCH_THRESHOLD — do not lower threshold to pass.
    expect(primary!.matched).toBe(false);

    // Padding grows the mask when a match exists — geometry alone is insufficient.
    expect(primary!.maskCurrent.height).toBeGreaterThan(primary!.maskLegacyAnalyzer.height);
    expect(primary!.maskCurrent.y).toBeLessThan(primary!.maskLegacyAnalyzer.y);
  }, 60_000);

  test("curated references: in-gallery positive and unrelated negatives", async () => {
    const { detector, embedder } = await createFixtureModels();
    const { enrollment, identity } = await buildProductionEnrollment(
      detector,
      embedder,
      [THEO, THEO_USER02],
      "curated",
    );
    identity.id = "theo-browne";
    identity.displayName = "theo-browne";

    expect(enrollment.gallerySize).toBe(2);
    expect(enrollment.meetsMinReferenceImages).toBe(false);

    const cases = [
      { path: THEO_USER02, expectMatch: true, label: "positive-in-gallery" },
      { path: DWARKESH, expectMatch: false, label: "negative-unrelated" },
      { path: UNRELATED, expectMatch: false, label: "negative-control" },
    ] as const;

    for (const c of cases) {
      const query = await queryAppearanceImage(detector, embedder, c.path, identity);
      const primary = primarySubjectFace(query);
      expect(primary, c.label).not.toBeNull();
      expect(primary!.matched, c.label).toBe(c.expectMatch);
    }
  }, 120_000);
});
