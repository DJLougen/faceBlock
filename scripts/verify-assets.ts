/**
 * Fail-closed asset verification for the FaceBlock extension.
 *
 * Two gates:
 *   verifySourceAssets() — the repo's demo/public payload is exactly what
 *     provenance.json claims (bytes + sha256 for EVERY recorded model,
 *     including legacy ones), the bundled ONNX Runtime files are byte-identical
 *     to the installed onnxruntime-web package at the pinned version, every
 *     asset the extension code references actually exists, and the license
 *     metadata is consistent.
 *   verifyBuiltTree(dir, root, profile) — a built tree contains ONLY the
 *     allowlisted payload. The "demo" profile (default build output) keeps the
 *     declared sample fixtures so the local demo works. The "package" profile
 *     (the staging tree the zip is made from) forbids every samples/** file —
 *     the demo corpus asserts no redistribution license — and requires
 *     references.json to be an empty array.
 *
 * Everything here is a check, never a fix: a mismatch is reported, not
 * repaired. Importable (tests exercise these functions directly); running the
 * file as a script exits non-zero on any issue.
 */
import { existsSync } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { join, resolve } from "node:path";
import { createHash } from "node:crypto";

export interface VerifyIssue {
  check: string;
  detail: string;
}
export interface VerifyResult {
  ok: boolean;
  issues: VerifyIssue[];
}

/** Models the extension actually loads at runtime (extension/offscreen.ts). */
export const SHIPPED_MODELS = [
  "face_detection_yunet_2026may.onnx",
  "w600k_mbf.onnx",
] as const;

/** ORT WASM runtime pair served from ort/ for the offscreen document. */
export const ORT_RUNTIME_FILES = [
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
] as const;

/** Third-party license/terms texts that must ship next to the payload. */
export const THIRD_PARTY_LICENSES = [
  "yunet-MIT.txt",
  "onnxruntime-web-MIT.txt",
  "onnxruntime-web-ThirdPartyNotices.txt",
  "insightface-w600k-TERMS.txt",
] as const;

/** Files allowed at the root of the built extension / archive. */
export const DIST_TOP_LEVEL_FILES = [
  "manifest.json",
  "background.js",
  "content.js",
  "offscreen.js",
  "offscreen.html",
  "options.js",
  "options.html",
  "options.css",
  "references.json",
  "build-info.json",
  "LICENSE",
  "NOTICE",
] as const;

/**
 * Payload that exists in the repo but must never ship:
 *  - the MediaPipe landmarker (retired detector, no longer loaded anywhere);
 *  - every demo sample photo (sources.json asserts no redistribution license,
 *    so package builds carry zero sample bytes).
 * Originals stay on disk; the allowlist excludes them from packaged output.
 */
export const NEVER_SHIPPED = ["models/face_landmarker.task"] as const;

const MODELS_DIR = "demo/public/models";
const ORT_DIR = "demo/public/ort";
const SAMPLES_DIR = "demo/public/samples";
const ORT_PKG_DIST = "node_modules/onnxruntime-web/dist";

function issue(list: VerifyIssue[], check: string, detail: string): void {
  list.push({ check, detail });
}

async function sha256File(path: string): Promise<string> {
  const bytes = await readFile(path);
  return createHash("sha256").update(bytes).digest("hex");
}

async function fileBytes(path: string): Promise<number | null> {
  try {
    return (await stat(path)).size;
  } catch {
    return null;
  }
}

interface ProvenanceModel {
  file?: string;
  bytes?: number;
  sha256?: string;
  shipped?: boolean;
}
interface Provenance {
  models?: ProvenanceModel[];
  runtime?: { "onnxruntime-web"?: { version?: string } };
}

interface ReferenceEntry {
  path?: string;
  source?: string;
}
interface ReferencePerson {
  id?: string;
  name?: string;
  references?: ReferenceEntry[];
}
interface SampleSources {
  images?: Record<string, { local?: string; image?: string }>;
}

async function readJson(path: string): Promise<unknown> {
  return JSON.parse(await readFile(path, "utf8"));
}

/**
 * A declared sample path must be exactly /samples/<filename> — no traversal,
 * no backslashes, no nesting — so a malformed ledger can never point outside
 * demo/public/samples/.
 */
const SAMPLE_LOCAL_RE = /^\/samples\/[A-Za-z0-9._-]+$/;

/**
 * Sample filenames the demo corpus declares in sources.json, as
 * demo/public-relative paths. Fail-closed: a missing, malformed, or empty
 * ledger is an error, not an empty set.
 */
export async function declaredSamples(
  root: string,
): Promise<{ samples: Set<string>; error: string | null }> {
  const samples = new Set<string>(["samples/sources.json"]);
  let sources: SampleSources;
  try {
    sources = (await readJson(join(root, SAMPLES_DIR, "sources.json"))) as SampleSources;
  } catch (e) {
    return { samples, error: `cannot read ${SAMPLES_DIR}/sources.json: ${String(e)}` };
  }
  const images = sources.images;
  if (!images || typeof images !== "object" || Object.keys(images).length === 0) {
    return { samples, error: `${SAMPLES_DIR}/sources.json declares no images` };
  }
  for (const [key, entry] of Object.entries(images)) {
    if (typeof entry.local !== "string" || !SAMPLE_LOCAL_RE.test(entry.local)) {
      return {
        samples,
        error: `sources.json entry ${key} has invalid local path ${JSON.stringify(entry.local)} (must be /samples/<filename>)`,
      };
    }
    samples.add(entry.local.slice(1));
  }
  return { samples, error: null };
}

/**
 * Classify one archive/dist-relative path against the ship allowlist.
 * `allowedSamples` is the set produced by declaredSamples() for the demo
 * profile; pass an empty set (package profile) to forbid every samples file.
 */
export function classifyDistPath(
  rel: string,
  allowedSamples: ReadonlySet<string> = new Set(),
): "allowed" | "forbidden" {
  const norm = rel.replace(/\\/g, "/").replace(/^\.\//, "").replace(/\/$/, "");
  if (norm.length === 0) return "forbidden";
  if (norm.endsWith(".DS_Store") || norm.endsWith(".map")) return "forbidden";
  if ((NEVER_SHIPPED as readonly string[]).includes(norm)) return "forbidden";
  if ((DIST_TOP_LEVEL_FILES as readonly string[]).includes(norm)) return "allowed";
  const slash = norm.indexOf("/");
  if (slash < 0) return "forbidden";
  const dir = norm.slice(0, slash);
  const rest = norm.slice(slash + 1);
  switch (dir) {
    case "models":
      return (SHIPPED_MODELS as readonly string[]).includes(rest) ||
        rest === "provenance.json"
        ? "allowed"
        : "forbidden";
    case "ort":
      return (ORT_RUNTIME_FILES as readonly string[]).includes(rest)
        ? "allowed"
        : "forbidden";
    case "licenses":
      return (THIRD_PARTY_LICENSES as readonly string[]).some(
        (name) => rest === `third-party/${name}`,
      )
        ? "allowed"
        : "forbidden";
    case "samples":
      return allowedSamples.has(norm) ? "allowed" : "forbidden";
    default:
      return "forbidden";
  }
}

/** The complete file set a built tree / archive must contain. */
export function requiredDistFiles(allowedSamples: ReadonlySet<string>): string[] {
  return [
    ...DIST_TOP_LEVEL_FILES,
    ...SHIPPED_MODELS.map((f) => `models/${f}`),
    "models/provenance.json",
    ...ORT_RUNTIME_FILES.map((f) => `ort/${f}`),
    ...THIRD_PARTY_LICENSES.map((f) => `licenses/third-party/${f}`),
    ...allowedSamples,
  ];
}

async function checkProvenanceModels(
  root: string,
  issues: VerifyIssue[],
): Promise<void> {
  const provPath = join(root, MODELS_DIR, "provenance.json");
  let prov: Provenance;
  try {
    prov = (await readJson(provPath)) as Provenance;
  } catch (e) {
    issue(issues, "provenance", `cannot read ${provPath}: ${String(e)}`);
    return;
  }
  if (!Array.isArray(prov.models) || prov.models.length === 0) {
    issue(issues, "provenance", "provenance.json has no models[] entries");
    return;
  }
  const seen = new Set<string>();
  for (const m of prov.models) {
    if (typeof m.file !== "string" || m.file.length === 0) {
      issue(issues, "provenance", "model entry missing file name");
      continue;
    }
    seen.add(m.file);
    const path = join(root, MODELS_DIR, m.file);
    const size = await fileBytes(path);
    if (size === null) {
      if (m.shipped !== false) {
        issue(issues, "provenance", `shipped model missing on disk: ${m.file}`);
      }
      continue; // legacy entries may be absent; presence is verified when found
    }
    // Every recorded model that exists on disk is hash-checked — including
    // shipped:false legacy entries, so a corrupted ledger can't hide drift.
    if (typeof m.bytes === "number" && m.bytes !== size) {
      issue(
        issues,
        "provenance",
        `${m.file}: byte count ${size} != provenance ${m.bytes}`,
      );
    }
    if (typeof m.sha256 === "string") {
      const actual = await sha256File(path);
      if (actual !== m.sha256) {
        issue(
          issues,
          "provenance",
          `${m.file}: sha256 ${actual} != provenance ${m.sha256}`,
        );
      }
    } else {
      issue(issues, "provenance", `${m.file}: no sha256 recorded`);
    }
  }
  for (const required of SHIPPED_MODELS) {
    if (!seen.has(required)) {
      issue(issues, "provenance", `active model ${required} absent from provenance`);
    }
  }
}

async function checkOrtRuntime(root: string, issues: VerifyIssue[]): Promise<void> {
  // The bundled runtime must match the installed package byte-for-byte AND be
  // the version pinned in provenance.json — a silent dependency bump that
  // recopies files must not pass unnoticed.
  let pinned: string | null = null;
  try {
    const prov = (await readJson(join(root, MODELS_DIR, "provenance.json"))) as Provenance;
    const v = prov.runtime?.["onnxruntime-web"]?.version;
    if (typeof v === "string" && v.length > 0) pinned = v;
    else issue(issues, "ort", "provenance.json does not pin runtime.onnxruntime-web.version");
  } catch (e) {
    issue(issues, "ort", `cannot read provenance.json for runtime pin: ${String(e)}`);
  }
  try {
    const pkg = (await readJson(join(root, "node_modules/onnxruntime-web/package.json"))) as {
      version?: string;
    };
    if (pinned !== null && pkg.version !== pinned) {
      issue(
        issues,
        "ort",
        `installed onnxruntime-web ${JSON.stringify(pkg.version)} != pinned ${JSON.stringify(pinned)}`,
      );
    }
  } catch (e) {
    issue(issues, "ort", `cannot read installed onnxruntime-web version: ${String(e)}`);
  }
  try {
    const pkg = (await readJson(join(root, "package.json"))) as {
      dependencies?: Record<string, string>;
    };
    if (!pkg.dependencies || typeof pkg.dependencies["onnxruntime-web"] !== "string") {
      issue(issues, "ort", "package.json does not declare an onnxruntime-web dependency");
    }
  } catch (e) {
    issue(issues, "ort", `cannot read package.json dependencies: ${String(e)}`);
  }
  for (const name of ORT_RUNTIME_FILES) {
    const bundled = join(root, ORT_DIR, name);
    const installed = join(root, ORT_PKG_DIST, name);
    if (!existsSync(bundled)) {
      issue(issues, "ort", `missing bundled runtime file ${ORT_DIR}/${name}`);
      continue;
    }
    if (!existsSync(installed)) {
      issue(
        issues,
        "ort",
        `cannot verify ${name}: ${ORT_PKG_DIST}/${name} not installed (run bun install)`,
      );
      continue;
    }
    const [a, b] = await Promise.all([sha256File(bundled), sha256File(installed)]);
    if (a !== b) {
      issue(
        issues,
        "ort",
        `${name}: bundled copy diverges from installed onnxruntime-web (${a} != ${b})`,
      );
    }
  }
}

/**
 * Every asset the extension code loads through extUrl() must exist on disk —
 * this ties the runtime inventory to the source, not just to the ledger.
 */
async function checkReferencedAssets(root: string, issues: VerifyIssue[]): Promise<void> {
  const offscreenPath = join(root, "extension/offscreen.ts");
  let source: string;
  try {
    source = await readFile(offscreenPath, "utf8");
  } catch (e) {
    issue(issues, "references", `cannot read ${offscreenPath}: ${String(e)}`);
    return;
  }
  const refs = new Set<string>();
  for (const match of source.matchAll(/extUrl\(\s*"([^"]+)"/g)) {
    refs.add(match[1]!);
  }
  if (refs.size === 0) {
    issue(issues, "references", "no extUrl() asset references found in offscreen.ts");
    return;
  }
  for (const ref of refs) {
    const base =
      ref.startsWith("models/") || ref.startsWith("ort/") || ref.startsWith("samples/")
        ? join(root, "demo/public", ref)
        : join(root, "extension", ref);
    if (!existsSync(base)) {
      issue(issues, "references", `offscreen.ts references ${ref} but ${base} is missing`);
    }
  }
  // references.json local sample paths resolve against the demo payload root.
  try {
    const refs2 = (await readJson(join(root, "extension/references.json"))) as ReferencePerson[];
    for (const person of refs2) {
      for (const r of person.references ?? []) {
        if (typeof r.path === "string" && !/^https?:\/\//i.test(r.path)) {
          const p = join(root, "demo/public", r.path);
          if (!existsSync(p)) {
            issue(issues, "references", `references.json path ${r.path} missing at ${p}`);
          }
        }
      }
    }
  } catch (e) {
    issue(issues, "references", `cannot read extension/references.json: ${String(e)}`);
  }
}

async function checkLicenseMetadata(root: string, issues: VerifyIssue[]): Promise<void> {
  const licensePath = join(root, "LICENSE");
  const noticePath = join(root, "NOTICE");
  if (!existsSync(licensePath)) {
    issue(issues, "license", "LICENSE file missing");
  } else {
    const text = await readFile(licensePath, "utf8");
    if (!/GNU AFFERO GENERAL PUBLIC LICENSE/.test(text)) {
      issue(issues, "license", "LICENSE does not contain the AGPL text");
    }
  }
  if (!existsSync(noticePath)) {
    issue(issues, "license", "NOTICE file missing");
  } else {
    const text = await readFile(noticePath, "utf8");
    if (!/Affero General Public License/.test(text)) {
      issue(issues, "license", "NOTICE does not state the AGPL grant");
    }
  }
  try {
    const pkg = (await readJson(join(root, "package.json"))) as { license?: string };
    if (pkg.license !== "AGPL-3.0-or-later") {
      issue(
        issues,
        "license",
        `package.json license is ${JSON.stringify(pkg.license)}; expected "AGPL-3.0-or-later" per NOTICE`,
      );
    }
  } catch (e) {
    issue(issues, "license", `cannot read package.json: ${String(e)}`);
  }
  for (const name of THIRD_PARTY_LICENSES) {
    const p = join(root, "licenses/third-party", name);
    if (!existsSync(p)) {
      issue(issues, "license", `missing third-party terms file licenses/third-party/${name}`);
    }
  }
}

/** Verify the source payload in the repo. Fail-closed: any issue fails. */
export async function verifySourceAssets(root: string = "."): Promise<VerifyResult> {
  const issues: VerifyIssue[] = [];
  await checkProvenanceModels(root, issues);
  await checkOrtRuntime(root, issues);
  await checkReferencedAssets(root, issues);
  await checkLicenseMetadata(root, issues);
  // The sample ledger must parse and declare at least one image; a malformed
  // or empty ledger is a failure, not an empty allowlist.
  const declared = await declaredSamples(root);
  if (declared.error !== null) {
    issue(issues, "samples", declared.error);
  } else {
    // Every declared sample must exist on disk.
    for (const rel of declared.samples) {
      const p = join(root, "demo/public", rel);
      if (!existsSync(p)) {
        issue(issues, "samples", `sources.json declares ${rel} but ${p} is missing`);
      }
    }
    // Every local path in references.json must be a declared sample — the
    // enrolment fixtures can only use what the ledger accounts for.
    try {
      const refs = (await readJson(join(root, "extension/references.json"))) as ReferencePerson[];
      for (const person of refs) {
        for (const r of person.references ?? []) {
          if (typeof r.path === "string" && !/^https?:\/\//i.test(r.path)) {
            if (!declared.samples.has(r.path)) {
              issue(
                issues,
                "samples",
                `references.json uses ${r.path} which sources.json does not declare`,
              );
            }
          }
        }
      }
    } catch (e) {
      issue(issues, "samples", `cannot read extension/references.json: ${String(e)}`);
    }
  }
  return { ok: issues.length === 0, issues };
}

export type DistProfile = "demo" | "package";

/**
 * Verify a built tree. Fail-closed: any issue fails.
 *  - "demo" (default): the local-demo build; declared sample fixtures are
 *    required and allowed, references.json may point at them.
 *  - "package": the staging tree the zip is made from; zero samples/** files
 *    are allowed and references.json must be an empty array, because the demo
 *    corpus asserts no redistribution license.
 */
export async function verifyBuiltTree(
  distDir: string = "dist-extension",
  root: string = ".",
  profile: DistProfile = "demo",
): Promise<VerifyResult> {
  const issues: VerifyIssue[] = [];
  const dist = resolve(root, distDir);
  if (!existsSync(dist)) {
    return {
      ok: false,
      issues: [{ check: "dist", detail: `${distDir} does not exist — run the build first` }],
    };
  }

  let allowedSamples: ReadonlySet<string> = new Set();
  if (profile === "demo") {
    const declared = await declaredSamples(root);
    if (declared.error !== null) {
      issue(issues, "samples", declared.error);
    }
    allowedSamples = declared.samples;
  }

  // 1. Required payload present.
  for (const rel of requiredDistFiles(allowedSamples)) {
    if (!existsSync(join(dist, rel))) {
      issue(issues, "dist", `required file missing from build: ${rel}`);
    }
  }

  // 2. Nothing outside the allowlist — in the package profile this rejects
  //    every samples/** file, the retired landmarker, and anything unapproved.
  for (const rel of new Bun.Glob("**/*").scanSync({ cwd: dist, onlyFiles: true })) {
    const norm = rel.replace(/\\/g, "/");
    if (classifyDistPath(norm, allowedSamples) !== "allowed") {
      issue(issues, "dist", `unapproved file in build output: ${norm}`);
    }
  }

  // 3. Built provenance must be a non-empty ledger that records every active
  //    model with a sha256 and byte count — an empty or sparse ledger is a
  //    failure, not a pass.
  let builtProv: Provenance | null = null;
  try {
    builtProv = (await readJson(join(dist, "models/provenance.json"))) as Provenance;
  } catch (e) {
    issue(issues, "dist", `cannot read built models/provenance.json: ${String(e)}`);
  }
  const builtModels = builtProv?.models;
  if (!Array.isArray(builtModels) || builtModels.length === 0) {
    issue(issues, "dist", "built provenance.json has no models[] entries");
  } else {
    const recorded = new Set<string>();
    for (const m of builtModels) {
      if (typeof m.file !== "string" || m.file.length === 0) {
        issue(issues, "dist", "built provenance has a model entry with no file name");
        continue;
      }
      recorded.add(m.file);
      if (m.shipped === false) {
        if (existsSync(join(dist, "models", m.file))) {
          issue(issues, "dist", `legacy model ${m.file} must not be packaged`);
        }
        continue;
      }
      if (typeof m.sha256 !== "string" || m.sha256.length === 0) {
        issue(issues, "dist", `built provenance records no sha256 for ${m.file}`);
        continue;
      }
      if (typeof m.bytes !== "number") {
        issue(issues, "dist", `built provenance records no byte count for ${m.file}`);
      }
      const p = join(dist, "models", m.file);
      if (!existsSync(p)) continue; // already reported as missing
      const actual = await sha256File(p);
      if (actual !== m.sha256) {
        issue(issues, "dist", `built ${m.file}: sha256 ${actual} != provenance ${m.sha256}`);
      }
      const size = await fileBytes(p);
      if (size !== null && typeof m.bytes === "number" && size !== m.bytes) {
        issue(issues, "dist", `built ${m.file}: ${size} bytes != provenance ${m.bytes}`);
      }
    }
    for (const required of SHIPPED_MODELS) {
      if (!recorded.has(required)) {
        issue(issues, "dist", `built provenance does not record active model ${required}`);
      }
    }
  }

  // 4. Built ORT runtime still matches the installed package.
  for (const name of ORT_RUNTIME_FILES) {
    const bundled = join(dist, "ort", name);
    const installed = join(root, ORT_PKG_DIST, name);
    if (!existsSync(bundled)) continue; // reported above
    if (!existsSync(installed)) {
      issue(issues, "dist", `cannot verify built ort/${name}: onnxruntime-web not installed`);
      continue;
    }
    const [a, b] = await Promise.all([sha256File(bundled), sha256File(installed)]);
    if (a !== b) {
      issue(issues, "dist", `built ort/${name} diverges from installed onnxruntime-web`);
    }
  }

  // 5. Notices in the build are byte-identical to the repo's — LICENSE,
  //    NOTICE, and every third-party terms file.
  const noticePairs: Array<[string, string]> = [
    ["LICENSE", "LICENSE"],
    ["NOTICE", "NOTICE"],
    ...THIRD_PARTY_LICENSES.map(
      (name): [string, string] => [
        `licenses/third-party/${name}`,
        `licenses/third-party/${name}`,
      ],
    ),
  ];
  for (const [shippedRel, sourceRel] of noticePairs) {
    const shippedPath = join(dist, shippedRel);
    const sourcePath = join(root, sourceRel);
    if (!existsSync(shippedPath) || !existsSync(sourcePath)) continue;
    const [a, b] = await Promise.all([sha256File(shippedPath), sha256File(sourcePath)]);
    if (a !== b) {
      issue(issues, "dist", `built ${shippedRel} differs from the repository ${sourceRel}`);
    }
  }

  // 6. Built manifest is a real MV3 manifest whose version matches package.json.
  try {
    const manifest = (await readJson(join(dist, "manifest.json"))) as {
      manifest_version?: number;
      version?: string;
    };
    if (manifest.manifest_version !== 3) {
      issue(issues, "dist", "built manifest.json is not manifest_version 3");
    }
    const pkg = (await readJson(join(root, "package.json"))) as { version?: string };
    if (manifest.version !== pkg.version) {
      issue(
        issues,
        "dist",
        `built manifest version ${JSON.stringify(manifest.version)} != package.json ${JSON.stringify(pkg.version)}`,
      );
    }
  } catch (e) {
    issue(issues, "dist", `cannot read built manifest.json: ${String(e)}`);
  }

  // 7. References policy per profile.
  try {
    const refs = (await readJson(join(dist, "references.json"))) as unknown;
    if (profile === "package") {
      // The packaged archive ships no sample bytes, so it must carry no
      // curated references that would point at missing files.
      if (!Array.isArray(refs) || refs.length !== 0) {
        issue(
          issues,
          "dist",
          "package profile requires references.json to be an empty array (no curated refs without shipped samples)",
        );
      }
    } else if (Array.isArray(refs)) {
      for (const person of refs as ReferencePerson[]) {
        for (const r of person.references ?? []) {
          if (typeof r.path === "string" && !/^https?:\/\//i.test(r.path)) {
            if (!allowedSamples.has(r.path)) {
              issue(
                issues,
                "dist",
                `built references.json uses ${r.path} which sources.json does not declare`,
              );
            } else if (!existsSync(join(dist, r.path))) {
              issue(issues, "dist", `built references.json path ${r.path} missing from build`);
            }
          }
        }
      }
    } else {
      issue(issues, "dist", "built references.json is not an array");
    }
  } catch (e) {
    issue(issues, "dist", `cannot read built references.json: ${String(e)}`);
  }

  return { ok: issues.length === 0, issues };
}

/**
 * Verify a zip archive's file listing (paths relative to archive root) against
 * the same allowlist used for the built tree. Directory entries (trailing
 * slash) are ignored; the archive must contain the complete required file
 * set and nothing unapproved. Pass an empty `allowedSamples` for the package
 * profile so any samples/** entry fails.
 */
export function verifyArchiveListing(
  entries: Iterable<string>,
  allowedSamples: ReadonlySet<string>,
): VerifyResult {
  const issues: VerifyIssue[] = [];
  const seen = new Set<string>();
  for (const raw of entries) {
    if (raw.endsWith("/")) continue; // directory entry, not a file
    const rel = raw.replace(/\\/g, "/").replace(/^\.\//, "");
    if (rel.length === 0) continue;
    seen.add(rel);
    if (classifyDistPath(rel, allowedSamples) !== "allowed") {
      issue(issues, "archive", `unapproved file in archive: ${rel}`);
    }
  }
  for (const required of requiredDistFiles(allowedSamples)) {
    if (!seen.has(required)) {
      issue(issues, "archive", `archive is missing required file ${required}`);
    }
  }
  for (const rel of NEVER_SHIPPED) {
    if (seen.has(rel)) {
      issue(issues, "archive", `archive contains excluded legacy asset ${rel}`);
    }
  }
  return { ok: issues.length === 0, issues };
}

function report(label: string, result: VerifyResult): boolean {
  if (result.ok) {
    console.log(`${label}: OK`);
    return true;
  }
  console.error(`${label}: FAILED with ${result.issues.length} issue(s)`);
  for (const i of result.issues) console.error(`  [${i.check}] ${i.detail}`);
  return false;
}

if (import.meta.main) {
  const args = process.argv.slice(2);
  const checkDist = args.includes("--dist") || args.includes("--all");
  const checkSrc = !args.includes("--dist") || args.includes("--all");
  let ok = true;
  if (checkSrc) ok = report("source assets", await verifySourceAssets(".")) && ok;
  if (checkDist) ok = report("built tree", await verifyBuiltTree("dist-extension", ".")) && ok;
  if (!ok) process.exit(1);
}
