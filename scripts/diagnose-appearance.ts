/**
 * Print ONNX diagnostics for appearance-change fixtures (detection, cosine, mask bounds).
 *
 *   bun scripts/diagnose-appearance.ts
 *   bun scripts/diagnose-appearance.ts --fixture theo-clean-to-user-labeled
 */
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  primarySubjectFace,
  runAppearanceFixture,
  type AppearanceFixtureReport,
} from "../src/analysis/fixtureRunner.ts";

interface FixtureDef {
  id: string;
  enroll: string[];
  query: string;
  identityId: string;
}

const ROOT = resolve(import.meta.dir, "..");

async function loadFixtures(): Promise<FixtureDef[]> {
  const raw = await readFile(resolve(ROOT, "demo/public/samples/appearance-fixtures.json"), "utf8");
  const parsed = JSON.parse(raw) as { fixtures: FixtureDef[] };
  return parsed.fixtures;
}

function summarize(report: AppearanceFixtureReport): unknown {
  const primary = primarySubjectFace(report.query);
  return {
    fixtureId: report.fixtureId,
    enrollFaceCounts: report.enrollFaceCounts,
    query: {
      path: report.queryPath,
      faceCount: report.query.faceCount,
      threshold: report.query.threshold,
      minAgreements: report.query.minAgreements,
    },
    primarySubject: primary
      ? {
          confidence: round(primary.confidence),
          bestCosine: primary.bestCosine === null ? null : round(primary.bestCosine),
          matched: primary.matched,
          detectionBox: roundBox(primary.detectionBox),
          maskCurrent: roundBox(primary.maskCurrent),
          maskLegacyAnalyzer: roundBox(primary.maskLegacyAnalyzer),
          maskLegacyFullPipeline: roundBox(primary.maskLegacyFullPipeline),
          maskDelta: primary.maskCurrent.height - primary.maskLegacyAnalyzer.height,
        }
      : null,
    allFaces: report.query.faces.map((f) => ({
      confidence: round(f.confidence),
      bestCosine: f.bestCosine === null ? null : round(f.bestCosine),
      matched: f.matched,
    })),
  };
}

function round(n: number): number {
  return Math.round(n * 1000) / 1000;
}

function roundBox(b: { x: number; y: number; width: number; height: number }) {
  return { x: round(b.x), y: round(b.y), width: round(b.width), height: round(b.height) };
}

const fixtureArg = process.argv.indexOf("--fixture");
const fixtureId = fixtureArg >= 0 ? process.argv[fixtureArg + 1] : null;
const fixtures = await loadFixtures();
const selected = fixtureId ? fixtures.filter((f) => f.id === fixtureId) : fixtures;
if (selected.length === 0) {
  console.error(`No fixture named ${fixtureId ?? "(none)"}`);
  process.exit(2);
}

for (const def of selected) {
  const report = await runAppearanceFixture({
    fixtureId: def.id,
    enrollPaths: def.enroll.map((p) => resolve(ROOT, p)),
    queryPath: resolve(ROOT, def.query),
    identityId: def.identityId,
  });
  console.log(JSON.stringify(summarize(report), null, 2));
}
