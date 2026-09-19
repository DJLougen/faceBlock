import { cp, mkdir, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

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
for (const folder of ["models", "ort", "samples"]) {
  await cp(`demo/public/${folder}`, `${out}/${folder}`, { recursive: true });
}
await writeFile(`${out}/build-info.json`, JSON.stringify({ builtAt: new Date().toISOString(), scope: "Local research preview. Still images only; not a calibrated identity benchmark." }, null, 2));
console.log(`Unpacked Chromium extension ready: ${out}`);
