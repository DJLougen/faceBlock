import { cp, rm, stat, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { verifyArchiveListing, verifyBuiltTree } from "./verify-assets";

// The extension manifest is the single source of truth for the version; the
// archive name reads it instead of hard-coding one that can drift.
const manifest = JSON.parse(await Bun.file("extension/manifest.json").text()) as { version?: string };
if (typeof manifest.version !== "string" || manifest.version.length === 0) {
  throw new Error('extension/manifest.json has no usable "version" field');
}
const version = manifest.version;

// Shell out to the same `bun run build:extension` entry point a developer runs
// by hand, so a packaged build can never drift from a manual one. The build
// already verifies the source payload fail-closed.
await Bun.$`bun run build:extension`;

const out = resolve("dist-extension");
// Verify the demo build first — the staging tree is derived from it.
const built = await verifyBuiltTree("dist-extension", ".", "demo");
if (!built.ok) {
  for (const i of built.issues) console.error(`  [${i.check}] ${i.detail}`);
  throw new Error(`built-tree verification failed (${built.issues.length} issue(s)); refusing to package`);
}
const builtManifest = JSON.parse(await Bun.file(`${out}/manifest.json`).text()) as { manifest_version?: number };
if (builtManifest.manifest_version !== 3) {
  throw new Error(`dist-extension/manifest.json is missing manifest_version: 3 (got ${JSON.stringify(builtManifest.manifest_version)})`);
}

// Package from a staging tree, never the demo dist: the demo sample photos
// assert no redistribution license (demo/public/samples/sources.json), so the
// archive carries zero sample bytes and an empty references.json — curated
// enrolment fixtures are a local-demo feature only.
const stage = resolve("dist-package");
await rm(stage, { recursive: true, force: true });
await cp(out, stage, { recursive: true });
await rm(`${stage}/samples`, { recursive: true, force: true });
await writeFile(`${stage}/references.json`, "[]\n");
await writeFile(`${stage}/build-info.json`, JSON.stringify({
  builtAt: new Date().toISOString(),
  scope: "Local research preview covering webpage images and video. Experimental operating point; not a calibrated identity benchmark.",
  samplePolicy: "excluded — demo photos carry no documented redistribution license; curated references removed",
}, null, 2));

const staged = await verifyBuiltTree("dist-package", ".", "package");
if (!staged.ok) {
  for (const i of staged.issues) console.error(`  [${i.check}] ${i.detail}`);
  throw new Error(`staging-tree verification failed (${staged.issues.length} issue(s)); refusing to package`);
}

const zipBin = Bun.which("zip");
if (!zipBin) throw new Error("system `zip` not found on PATH; cannot create the archive");
const unzipBin = Bun.which("unzip");
if (!unzipBin) throw new Error("system `unzip` not found on PATH; cannot verify the archive");

const archive = resolve(`faceBlock-${version}.zip`);
await rm(archive, { force: true });
// Zip the CONTENTS of the staging tree so manifest.json sits at the archive
// root — what both "Load unpacked" on an extracted folder and a Web Store
// upload expect. `.DS_Store` is the only exclusion: macOS junk, not payload.
await Bun.$`${zipBin} -r ${archive} . -x "*.DS_Store"`.cwd(stage).quiet();

// Verify what actually landed in the archive, not just what was staged:
// the listing is checked against the package allowlist (zero samples), and
// `unzip -t` CRC-tests every member so a corrupt archive fails closed.
const listing = await Bun.$`${unzipBin} -Z1 ${archive}`.quiet();
const archiveCheck = verifyArchiveListing(listing.text().split("\n"), new Set());
if (!archiveCheck.ok) {
  for (const i of archiveCheck.issues) console.error(`  [${i.check}] ${i.detail}`);
  throw new Error(`archive verification failed (${archiveCheck.issues.length} issue(s))`);
}
await Bun.$`${unzipBin} -tqq ${archive}`.quiet();

const info = await stat(archive);
const fileCount = [...new Bun.Glob("**/*").scanSync({ cwd: stage, onlyFiles: true })]
  .filter((path) => !path.endsWith(".DS_Store")).length;
console.log(`Packaged faceBlock v${version}: ${archive}`);
console.log(`${fileCount} files, ${info.size} bytes`);
console.log("NOTE: local research/verification bundle — demo photos excluded (no redistribution license documented); recognition weights remain non-commercial research only. Not a release-cleared distributable.");
