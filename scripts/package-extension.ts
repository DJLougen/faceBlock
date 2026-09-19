import { rm, stat } from "node:fs/promises";
import { resolve } from "node:path";

// The extension manifest is the single source of truth for the version; the
// archive name reads it instead of hard-coding one that can drift.
const manifest = JSON.parse(await Bun.file("extension/manifest.json").text()) as { version?: string };
if (typeof manifest.version !== "string" || manifest.version.length === 0) {
  throw new Error('extension/manifest.json has no usable "version" field');
}
const version = manifest.version;

// Shell out to the same `bun run build:extension` entry point a developer runs
// by hand, so a packaged build can never drift from a manual one. Importing
// build-extension.ts would work too (it is top-level await), but going through
// the package.json script also exercises the script wiring itself.
await Bun.$`bun run build:extension`;

const out = resolve("dist-extension");
const builtManifest = JSON.parse(await Bun.file(`${out}/manifest.json`).text()) as { manifest_version?: number };
if (builtManifest.manifest_version !== 3) {
  throw new Error(`dist-extension/manifest.json is missing manifest_version: 3 (got ${JSON.stringify(builtManifest.manifest_version)})`);
}

const zipBin = Bun.which("zip");
if (!zipBin) throw new Error("system `zip` not found on PATH; cannot create the archive");

const archive = resolve(`faceBlock-${version}.zip`);
await rm(archive, { force: true });
// Zip the CONTENTS of dist-extension/ so manifest.json sits at the archive
// root — what both "Load unpacked" on an extracted folder and a Web Store
// upload expect. `.DS_Store` is the only exclusion: macOS junk, not payload.
await Bun.$`${zipBin} -r ${archive} . -x "*.DS_Store"`.cwd(out).quiet();

const info = await stat(archive);
const fileCount = [...new Bun.Glob("**/*").scanSync({ cwd: out, onlyFiles: true })]
  .filter((path) => !path.endsWith(".DS_Store")).length;
console.log(`Packaged faceBlock v${version}: ${archive}`);
console.log(`${fileCount} files, ${info.size} bytes`);
