/**
 * Run the appearance fixture corpus and print FAR/FRR-style counts.
 *
 *   bun scripts/eval-appearance-corpus.ts
 */
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  buildProductionEnrollment,
  createFixtureModels,
  primarySubjectFace,
  queryAppearanceImage,
  type EnrollmentMode,
} from "../src/analysis/fixtureRunner.ts";

interface QueryCase {
  path: string;
  label: string;
  expectMatch: boolean;
  samePerson?: boolean;
}

interface FixtureDef {
  id: string;
  enrollmentMode?: EnrollmentMode;
  enroll: string[];
  query?: string;
  expectMatch?: boolean;
  samePerson?: boolean;
  queries?: QueryCase[];
  identityId: string;
}

const ROOT = resolve(import.meta.dir, "..");

async function loadFixtures(): Promise<FixtureDef[]> {
  const raw = await readFile(resolve(ROOT, "demo/public/samples/appearance-fixtures.json"), "utf8");
  const parsed = JSON.parse(raw) as { fixtures: FixtureDef[] };
  return parsed.fixtures;
}

function casesFor(def: FixtureDef): QueryCase[] {
  if (def.queries?.length) return def.queries;
  if (def.query) {
    return [
      {
        path: def.query,
        label: def.id,
        expectMatch: def.expectMatch ?? false,
        samePerson: (def as { samePerson?: boolean }).samePerson,
      },
    ];
  }
  return [];
}

const fixtures = await loadFixtures();
const { detector, embedder } = await createFixtureModels();

let tp = 0;
let fn = 0;
let tn = 0;
let fp = 0;
let appearanceFn = 0;
let appearanceTp = 0;

for (const def of fixtures) {
  const mode = def.enrollmentMode ?? "curated";
  const enrollPaths = def.enroll.map((p) => resolve(ROOT, p));
  const { enrollment, identity } = await buildProductionEnrollment(
    detector,
    embedder,
    enrollPaths,
    mode,
  );
  identity.id = def.identityId;
  identity.displayName = def.identityId;

  console.log(
    JSON.stringify(
      {
        fixtureId: def.id,
        enrollmentMode: mode,
        gallerySize: enrollment.gallerySize,
        meetsMinReferenceImages: enrollment.meetsMinReferenceImages,
        rejected: enrollment.rejected,
      },
      null,
      2,
    ),
  );

  for (const c of casesFor(def)) {
    const queryPath = resolve(ROOT, c.path);
    const query = await queryAppearanceImage(detector, embedder, queryPath, identity);
    const primary = primarySubjectFace(query);
    const matched = primary?.matched ?? false;
    const bestCosine = primary?.bestCosine ?? null;

    if (c.expectMatch && matched) tp++;
    else if (c.expectMatch && !matched) fn++;
    else if (!c.expectMatch && !matched) tn++;
    else fp++;

    if (c.samePerson) {
      if (matched) appearanceTp++;
      else appearanceFn++;
    }

    console.log(
      JSON.stringify({
        fixtureId: def.id,
        label: c.label,
        query: c.path,
        expectMatch: c.expectMatch,
        matched,
        bestCosine: bestCosine === null ? null : Math.round(bestCosine * 1000) / 1000,
      }),
    );
  }
}

const positive = tp + fn;
const negative = tn + fp;
const appearancePositive = appearanceTp + appearanceFn;
console.log(
  JSON.stringify(
    {
      summary: {
        truePositive: tp,
        falseNegative: fn,
        trueNegative: tn,
        falsePositive: fp,
        frr: positive > 0 ? fn / positive : null,
        far: negative > 0 ? fp / negative : null,
        appearanceSamePerson: {
          truePositive: appearanceTp,
          falseNegative: appearanceFn,
          frr: appearancePositive > 0 ? appearanceFn / appearancePositive : null,
        },
      },
      note:
        "Corpus is tiny and curated Theo refs still lack MIN_REFERENCE_IMAGES=3. Do not tune MATCH_THRESHOLD from these counts alone.",
    },
    null,
    2,
  ),
);
