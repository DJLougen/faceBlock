/**
 * Delivery-gate tests for scripts/verify-assets.ts. These exercise the real
 * verifier functions — a fake but complete repo root lets the fail-closed
 * paths be tested without copying the real 14 MB models.
 */
import { describe, expect, test } from "bun:test";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import {
  classifyDistPath,
  declaredSamples,
  DIST_TOP_LEVEL_FILES,
  NEVER_SHIPPED,
  ORT_RUNTIME_FILES,
  requiredDistFiles,
  SHIPPED_MODELS,
  SPONSOR_FILES,
  THIRD_PARTY_LICENSES,
  verifyArchiveListing,
  verifyBuiltTree,
  verifySourceAssets,
} from "../../scripts/verify-assets";

const sha = (s: string) => createHash("sha256").update(s).digest("hex");

const DEMO_SAMPLES = new Set(["samples/sources.json", "samples/img.jpg"]);
const NO_SAMPLES = new Set<string>();

/** A minimal but complete fake repo root + built demo tree that should pass. */
async function makeFakeRoot(): Promise<{ root: string; dist: string }> {
  const root = await mkdtemp(join(tmpdir(), "fb-verify-"));
  const dist = join(root, "dist-extension");
  const w = async (rel: string, content: string) => {
    const p = join(root, rel);
    await mkdir(join(p, ".."), { recursive: true });
    await writeFile(p, content);
  };
  const wd = async (rel: string, content: string) => {
    const p = join(dist, rel);
    await mkdir(join(p, ".."), { recursive: true });
    await writeFile(p, content);
  };

  // Source payload.
  const yunet = "fake yunet weights";
  const w600k = "fake w600k weights";
  const provenance = JSON.stringify({
    models: [
      { file: "face_detection_yunet_2026may.onnx", bytes: yunet.length, sha256: sha(yunet), shipped: true },
      { file: "w600k_mbf.onnx", bytes: w600k.length, sha256: sha(w600k), shipped: true },
      { file: "face_landmarker.task", bytes: 4, sha256: sha("old!"), shipped: false },
    ],
    runtime: { "onnxruntime-web": { version: "9.9.9-test" } },
  });
  await w("demo/public/models/face_detection_yunet_2026may.onnx", yunet);
  await w("demo/public/models/w600k_mbf.onnx", w600k);
  await w("demo/public/models/face_landmarker.task", "old!");
  await w("demo/public/models/provenance.json", provenance);
  await w("demo/public/ort/ort-wasm-simd-threaded.mjs", "ort js");
  await w("demo/public/ort/ort-wasm-simd-threaded.wasm", "ort wasm");
  await w("node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs", "ort js");
  await w("node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm", "ort wasm");
  await w("node_modules/onnxruntime-web/package.json", JSON.stringify({ version: "9.9.9-test" }));
  await w("demo/public/samples/sources.json", JSON.stringify({
    images: { "img.jpg": { local: "/samples/img.jpg" } },
  }));
  await w("demo/public/samples/img.jpg", "jpeg bytes");
  await w("extension/offscreen.ts", [
    'const a = extUrl("models/face_detection_yunet_2026may.onnx");',
    'const b = extUrl("models/w600k_mbf.onnx");',
    'const c = extUrl("ort/");',
    'const d = extUrl("references.json");',
  ].join("\n"));
  await w("extension/references.json", JSON.stringify([
    { id: "x", references: [{ path: "samples/img.jpg" }] },
  ]));
  await w("package.json", JSON.stringify({
    license: "AGPL-3.0-or-later",
    version: "0.0.0-test",
    dependencies: { "onnxruntime-web": "^9.9.9-test" },
  }));
  await w("LICENSE", "GNU AFFERO GENERAL PUBLIC LICENSE\nVersion 3");
  await w("NOTICE", "GNU Affero General Public License grant text");
  for (const name of THIRD_PARTY_LICENSES) {
    await w(`licenses/third-party/${name}`, `terms for ${name}`);
  }

  // Built demo tree mirroring the source payload.
  for (const f of DIST_TOP_LEVEL_FILES) await wd(f, f === "manifest.json" ? JSON.stringify({ manifest_version: 3, version: "0.0.0-test" }) : `built ${f}`);
  await wd("references.json", JSON.stringify([
    { id: "x", references: [{ path: "samples/img.jpg" }] },
  ]));
  await wd("models/face_detection_yunet_2026may.onnx", yunet);
  await wd("models/w600k_mbf.onnx", w600k);
  await wd("models/provenance.json", provenance);
  await wd("ort/ort-wasm-simd-threaded.mjs", "ort js");
  await wd("ort/ort-wasm-simd-threaded.wasm", "ort wasm");
  await wd("samples/sources.json", JSON.stringify({ images: { "img.jpg": { local: "/samples/img.jpg" } } }));
  await wd("samples/img.jpg", "jpeg bytes");
  await wd("LICENSE", "GNU AFFERO GENERAL PUBLIC LICENSE\nVersion 3");
  await wd("NOTICE", "GNU Affero General Public License grant text");
  for (const name of THIRD_PARTY_LICENSES) {
    await wd(`licenses/third-party/${name}`, `terms for ${name}`);
  }
  for (const file of SPONSOR_FILES) {
    await wd(`sponsors/${file}`, `<svg>${file}</svg>`);
  }
  return { root, dist };
}

/** Copy the fake demo dist into a package-profile staging tree. */
async function makePackageStage(root: string, dist: string): Promise<string> {
  const stage = join(root, "dist-package");
  await mkdir(stage, { recursive: true });
  for (const rel of new Bun.Glob("**/*").scanSync({ cwd: dist, onlyFiles: true })) {
    if (rel.startsWith("samples/")) continue;
    const dest = join(stage, rel);
    await mkdir(join(dest, ".."), { recursive: true });
    await writeFile(dest, await Bun.file(join(dist, rel)).text());
  }
  await writeFile(join(stage, "references.json"), "[]\n");
  return stage;
}

describe("classifyDistPath", () => {
  test("allows every shipped model, ort file, license, and top-level file", () => {
    for (const m of SHIPPED_MODELS) {
      expect(classifyDistPath(`models/${m}`, DEMO_SAMPLES)).toBe("allowed");
    }
    expect(classifyDistPath("models/provenance.json", DEMO_SAMPLES)).toBe("allowed");
    for (const f of ORT_RUNTIME_FILES) {
      expect(classifyDistPath(`ort/${f}`, DEMO_SAMPLES)).toBe("allowed");
    }
    for (const f of THIRD_PARTY_LICENSES) {
      expect(classifyDistPath(`licenses/third-party/${f}`, DEMO_SAMPLES)).toBe("allowed");
    }
    for (const f of DIST_TOP_LEVEL_FILES) {
      expect(classifyDistPath(f, DEMO_SAMPLES)).toBe("allowed");
    }
    expect(classifyDistPath("samples/img.jpg", DEMO_SAMPLES)).toBe("allowed");
    for (const f of SPONSOR_FILES) {
      expect(classifyDistPath(`sponsors/${f}`, DEMO_SAMPLES)).toBe("allowed");
      expect(classifyDistPath(`sponsors/${f}`, NO_SAMPLES)).toBe("allowed");
    }
  });
  test("forbids the retired landmarker and anything outside the allowlist", () => {
    for (const rel of NEVER_SHIPPED) {
      expect(classifyDistPath(rel, DEMO_SAMPLES)).toBe("forbidden");
    }
    for (const bad of [
      "models/face_landmarker.task",
      "mediapipe-wasm/vision_wasm_internal.wasm",
      "models/evil.onnx",
      "ort/ort-wasm-simd-threaded.jsep.wasm",
      "licenses/third-party/evil.txt",
      "samples/undeclared.jpg",
      "sponsors/evil.svg",
      "samples/appearance-fixtures.json",
      "samples/../secret",
      "background.js.map",
      ".DS_Store",
      "src/cv/yunet.ts",
      "node_modules/x/index.js",
    ]) {
      expect(classifyDistPath(bad, DEMO_SAMPLES)).toBe("forbidden");
    }
  });
  test("package profile (empty sample set) forbids every samples file", () => {
    expect(classifyDistPath("samples/img.jpg", NO_SAMPLES)).toBe("forbidden");
    expect(classifyDistPath("samples/sources.json", NO_SAMPLES)).toBe("forbidden");
  });
});

describe("verifyArchiveListing", () => {
  test("accepts a complete demo listing including directory entries", () => {
    const entries = [
      "models/", "ort/", "samples/", "licenses/", "licenses/third-party/",
      ...requiredDistFiles(DEMO_SAMPLES),
    ];
    expect(verifyArchiveListing(entries, DEMO_SAMPLES).ok).toBe(true);
  });
  test("accepts a complete package listing with zero samples", () => {
    const entries = [
      "models/", "ort/", "licenses/", "licenses/third-party/",
      ...requiredDistFiles(NO_SAMPLES),
    ];
    expect(verifyArchiveListing(entries, NO_SAMPLES).ok).toBe(true);
  });
  test("fails closed on a missing required file", () => {
    const entries = requiredDistFiles(NO_SAMPLES).filter((f) => f !== "NOTICE");
    const res = verifyArchiveListing(entries, NO_SAMPLES);
    expect(res.ok).toBe(false);
    expect(res.issues.some((i) => i.detail.includes("NOTICE"))).toBe(true);
  });
  test("fails closed on the excluded legacy model and on sample bytes", () => {
    const res = verifyArchiveListing(
      [...requiredDistFiles(NO_SAMPLES), "models/face_landmarker.task"],
      NO_SAMPLES,
    );
    expect(res.ok).toBe(false);
    const withSample = verifyArchiveListing(
      [...requiredDistFiles(NO_SAMPLES), "samples/img.jpg"],
      NO_SAMPLES,
    );
    expect(withSample.ok).toBe(false);
  });
});

describe("declaredSamples", () => {
  test("rejects a malformed ledger instead of returning an empty allowlist", async () => {
    const root = await mkdtemp(join(tmpdir(), "fb-samples-"));
    await mkdir(join(root, "demo/public/samples"), { recursive: true });
    await writeFile(join(root, "demo/public/samples/sources.json"), "{not json");
    const res = await declaredSamples(root);
    expect(res.error).not.toBeNull();
    await rm(root, { recursive: true });
  });
  test("rejects an empty images ledger", async () => {
    const root = await mkdtemp(join(tmpdir(), "fb-samples-"));
    await mkdir(join(root, "demo/public/samples"), { recursive: true });
    await writeFile(join(root, "demo/public/samples/sources.json"), JSON.stringify({ images: {} }));
    const res = await declaredSamples(root);
    expect(res.error).not.toBeNull();
    await rm(root, { recursive: true });
  });
  test("rejects traversal and nested local paths", async () => {
    const root = await mkdtemp(join(tmpdir(), "fb-samples-"));
    await mkdir(join(root, "demo/public/samples"), { recursive: true });
    for (const local of ["/samples/../secret", "/samples/a/b.jpg", "/samples\\x.jpg", "samples/x.jpg"]) {
      await writeFile(join(root, "demo/public/samples/sources.json"), JSON.stringify({
        images: { "x.jpg": { local } },
      }));
      const res = await declaredSamples(root);
      expect(res.error).not.toBeNull();
    }
    await rm(root, { recursive: true });
  });
});

describe("verifySourceAssets / verifyBuiltTree on a synthetic root", () => {
  test("a complete consistent tree passes the source gate and demo profile", async () => {
    const { root } = await makeFakeRoot();
    const src = await verifySourceAssets(root);
    expect(src.issues).toEqual([]);
    expect(src.ok).toBe(true);
    const built = await verifyBuiltTree("dist-extension", root, "demo");
    expect(built.issues).toEqual([]);
    expect(built.ok).toBe(true);
    await rm(root, { recursive: true });
  });
  test("the package profile passes on a sample-free staging tree with empty refs", async () => {
    const { root, dist } = await makeFakeRoot();
    await makePackageStage(root, dist);
    const res = await verifyBuiltTree("dist-package", root, "package");
    expect(res.issues).toEqual([]);
    expect(res.ok).toBe(true);
    await rm(root, { recursive: true });
  });
  test("the package profile rejects a tree that still contains samples", async () => {
    const { root, dist } = await makeFakeRoot();
    const stage = await makePackageStage(root, dist);
    await mkdir(join(stage, "samples"), { recursive: true });
    await writeFile(join(stage, "samples/img.jpg"), "jpeg bytes");
    const res = await verifyBuiltTree("dist-package", root, "package");
    expect(res.ok).toBe(false);
    expect(res.issues.some((i) => i.detail.includes("samples/img.jpg"))).toBe(true);
    await rm(root, { recursive: true });
  });
  test("the package profile rejects non-empty references.json", async () => {
    const { root, dist } = await makeFakeRoot();
    const stage = await makePackageStage(root, dist);
    await writeFile(join(stage, "references.json"), JSON.stringify([
      { id: "x", references: [{ path: "https://example.com/a.jpg" }] },
    ]));
    const res = await verifyBuiltTree("dist-package", root, "package");
    expect(res.ok).toBe(false);
    expect(res.issues.some((i) => i.detail.includes("empty array"))).toBe(true);
    await rm(root, { recursive: true });
  });
  test("missing dist fails closed", async () => {
    const { root } = await makeFakeRoot();
    const res = await verifyBuiltTree("no-such-dir", root);
    expect(res.ok).toBe(false);
    await rm(root, { recursive: true });
  });
  test("a tampered built model fails on sha256", async () => {
    const { root, dist } = await makeFakeRoot();
    await writeFile(join(dist, "models/w600k_mbf.onnx"), "tampered weights");
    const res = await verifyBuiltTree("dist-extension", root);
    expect(res.ok).toBe(false);
    expect(res.issues.some((i) => i.detail.includes("sha256"))).toBe(true);
    await rm(root, { recursive: true });
  });
  test("an empty built provenance ledger fails closed", async () => {
    const { root, dist } = await makeFakeRoot();
    await writeFile(join(dist, "models/provenance.json"), JSON.stringify({ models: [] }));
    const res = await verifyBuiltTree("dist-extension", root);
    expect(res.ok).toBe(false);
    expect(res.issues.some((i) => i.detail.includes("no models[]"))).toBe(true);
    await rm(root, { recursive: true });
  });
  test("a provenance entry without sha256 fails closed", async () => {
    const { root, dist } = await makeFakeRoot();
    const prov = JSON.parse(
      await Bun.file(join(dist, "models/provenance.json")).text(),
    ) as { models: Array<Record<string, unknown>> };
    delete prov.models[0]!.sha256;
    await writeFile(join(dist, "models/provenance.json"), JSON.stringify(prov));
    const res = await verifyBuiltTree("dist-extension", root);
    expect(res.ok).toBe(false);
    expect(res.issues.some((i) => i.detail.includes("no sha256"))).toBe(true);
    await rm(root, { recursive: true });
  });
  test("the excluded legacy model in dist fails closed", async () => {
    const { root, dist } = await makeFakeRoot();
    await writeFile(join(dist, "models/face_landmarker.task"), "old!");
    const res = await verifyBuiltTree("dist-extension", root);
    expect(res.ok).toBe(false);
    await rm(root, { recursive: true });
  });
  test("a built LICENSE that differs from the repo's fails", async () => {
    const { root, dist } = await makeFakeRoot();
    await writeFile(join(dist, "LICENSE"), "some other license");
    const res = await verifyBuiltTree("dist-extension", root);
    expect(res.ok).toBe(false);
    expect(res.issues.some((i) => i.detail.includes("LICENSE"))).toBe(true);
    await rm(root, { recursive: true });
  });
  test("a malformed sources.json fails the built-tree gate too", async () => {
    const { root } = await makeFakeRoot();
    await writeFile(join(root, "demo/public/samples/sources.json"), "{broken");
    const res = await verifyBuiltTree("dist-extension", root);
    expect(res.ok).toBe(false);
    await rm(root, { recursive: true });
  });
  test("an ORT version bump without a provenance pin update fails", async () => {
    const { root } = await makeFakeRoot();
    await writeFile(
      join(root, "node_modules/onnxruntime-web/package.json"),
      JSON.stringify({ version: "10.0.0" }),
    );
    const res = await verifySourceAssets(root);
    expect(res.ok).toBe(false);
    expect(res.issues.some((i) => i.detail.includes("pinned"))).toBe(true);
    await rm(root, { recursive: true });
  });
});

describe("verifySourceAssets on the real repo", () => {
  test("the checked-in payload passes the source gate", async () => {
    const res = await verifySourceAssets(".");
    expect(res.issues).toEqual([]);
    expect(res.ok).toBe(true);
  });
});
