import { cp, mkdir, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { SHIPPED_MODELS, verifySourceAssets } from "./verify-assets";

// Fail closed before touching the output: a missing or tampered model must
// stop the build, not ship silently inside it.
const source = await verifySourceAssets(".");
if (!source.ok) {
  for (const i of source.issues) console.error(`  [${i.check}] ${i.detail}`);
  throw new Error(`source asset verification failed (${source.issues.length} issue(s))`);
}

const out = resolve("dist-extension");
// Clear the output first: a stale model or bundle from a previous build would
// otherwise ship silently alongside the current one.
await rm(out, { recursive: true, force: true });
await mkdir(out, { recursive: true });
for (const [entry, format] of [["background", "esm"], ["offscreen", "esm"], ["options", "esm"], ["content", "iife"]] as const) {
  const result = await Bun.build({
    entrypoints: [`extension/${entry}.ts`],
    outdir: out,
    target: "browser",
    format,
    naming: `${entry}.js`,
    minify: false,
  });
  if (!result.success) throw new Error(result.logs.join("\n"));
}
for (const name of ["manifest.json", "references.json", "options.html", "options.css", "offscreen.html"]) {
  await cp(`extension/${name}`, `${out}/${name}`);
}
// Payload is an explicit allowlist, not a directory copy: only the models the
// runtime actually loads ship. The retired MediaPipe landmarker stays in
// demo/public/models/ but is never packaged (see verify-assets NEVER_SHIPPED).
await mkdir(`${out}/models`, { recursive: true });
for (const model of SHIPPED_MODELS) {
  await cp(`demo/public/models/${model}`, `${out}/models/${model}`);
}
await cp("demo/public/models/provenance.json", `${out}/models/provenance.json`);
// The local-demo build keeps the declared sample fixtures so the fixture
// pages and curated references work offline. `bun run package` strips them —
// the demo corpus asserts no redistribution license.
for (const folder of ["ort", "samples"]) {
  await cp(`demo/public/${folder}`, `${out}/${folder}`, { recursive: true });
}
// License notices travel with the payload: the project's own grant plus the
// verbatim third-party terms for every bundled model and runtime.
await cp("LICENSE", `${out}/LICENSE`);
await cp("NOTICE", `${out}/NOTICE`);
await cp("licenses", `${out}/licenses`, { recursive: true });
await writeFile(`${out}/build-info.json`, JSON.stringify({
  builtAt: new Date().toISOString(),
  scope: "Local research preview covering webpage images and video. Experimental operating point; not a calibrated identity benchmark.",
  samplePolicy: "demo fixtures bundled for local use only — no redistribution license asserted; the packaged zip excludes them",
}, null, 2));
console.log(`Unpacked Chromium extension ready: ${out}`);
