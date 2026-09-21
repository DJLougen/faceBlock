/**
 * Delivery metadata consistency: license identifiers, manifest/archive naming
 * inputs, and the model provenance ledger must agree with each other and with
 * what actually ships.
 */
import { describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import {
  NEVER_SHIPPED,
  SHIPPED_MODELS,
  THIRD_PARTY_LICENSES,
} from "../../scripts/verify-assets";

const readJson = async (p: string) => JSON.parse(await readFile(p, "utf8")) as Record<string, unknown>;

describe("license metadata", () => {
  test("package.json license matches the NOTICE grant", async () => {
    const pkg = await readJson("package.json");
    expect(pkg.license).toBe("AGPL-3.0-or-later");
    const notice = await readFile("NOTICE", "utf8");
    expect(notice).toMatch(/GNU Affero General Public License/);
    expect(notice).toMatch(/version 3 of the License, or \(at your option\) any\s+later version/);
  });
  test("LICENSE is the AGPL text and is unchanged in scope", async () => {
    const text = await readFile("LICENSE", "utf8");
    expect(text).toMatch(/GNU AFFERO GENERAL PUBLIC LICENSE/);
    expect(text).toMatch(/Version 3, 19 November 2007/);
  });
  test("every bundled third-party component has a terms file", () => {
    for (const name of THIRD_PARTY_LICENSES) {
      expect(existsSync(`licenses/third-party/${name}`)).toBe(true);
    }
  });
  test("the recognition weights terms file states a restriction, not a grant", async () => {
    const terms = await readFile("licenses/third-party/insightface-w600k-TERMS.txt", "utf8");
    expect(terms).toMatch(/non-commercial research purposes only/i);
    expect(terms).toMatch(/NO LICENSE GRANT/i);
  });
});

describe("manifest and packaging inputs", () => {
  test("extension manifest is MV3 and its version matches package.json", async () => {
    const manifest = await readJson("extension/manifest.json");
    const pkg = await readJson("package.json");
    expect(manifest.manifest_version).toBe(3);
    expect(manifest.version).toBe(pkg.version);
  });
  test("sponsor placeholder is web-accessible on http(s) pages", async () => {
    const manifest = await readJson("extension/manifest.json");
    const war = manifest.web_accessible_resources as Array<Record<string, unknown>>;
    expect(Array.isArray(war)).toBe(true);
    const sponsor = war.find((entry) =>
      (entry.resources as string[] | undefined)?.includes("sponsors/placeholder.svg"),
    );
    expect(sponsor).toBeDefined();
    expect(sponsor!.matches).toEqual(["http://*/*", "https://*/*"]);
    expect(sponsor!.use_dynamic_url).toBe(true);
    expect(existsSync("extension/sponsors/placeholder.svg")).toBe(true);
  });
  test("package.json exposes the verify and packaging gates", async () => {
    const pkg = await readJson("package.json");
    const scripts = pkg.scripts as Record<string, string>;
    expect(scripts["verify:assets"]).toBe("bun scripts/verify-assets.ts");
    expect(scripts["verify:browser"]).toBe("bun scripts/verify-browser.ts");
    expect(scripts["package"]).toContain("package-extension.ts");
    expect(scripts["build:extension"]).toContain("build-extension.ts");
  });
});

describe("model provenance ledger", () => {
  test("every shipped model entry has a file, byte count, and sha256 that match disk", async () => {
    const prov = await readJson("demo/public/models/provenance.json");
    const models = prov.models as Array<Record<string, unknown>>;
    expect(Array.isArray(models)).toBe(true);
    const byFile = new Map(models.map((m) => [m.file as string, m]));
    for (const name of SHIPPED_MODELS) {
      const entry = byFile.get(name);
      expect(entry, `provenance missing ${name}`).toBeDefined();
      expect(entry!.shipped).not.toBe(false);
      const path = `demo/public/models/${name}`;
      expect(existsSync(path)).toBe(true);
      const bytes = await readFile(path);
      expect(entry!.bytes).toBe(bytes.length);
      const digest = createHash("sha256").update(bytes).digest("hex");
      expect(entry!.sha256).toBe(digest);
    }
  });
  test("the retired landmarker is recorded but marked not shipped", async () => {
    const prov = await readJson("demo/public/models/provenance.json");
    const models = prov.models as Array<Record<string, unknown>>;
    const landmarker = models.find((m) => m.file === "face_landmarker.task");
    expect(landmarker).toBeDefined();
    expect(landmarker!.shipped).toBe(false);
    // The original file stays on disk; it is only excluded from packaging.
    expect(existsSync("demo/public/models/face_landmarker.task")).toBe(true);
    expect(NEVER_SHIPPED).toContain("models/face_landmarker.task");
  });
  test("the active detector is the 2026may YuNet build", async () => {
    const prov = await readJson("demo/public/models/provenance.json");
    const models = prov.models as Array<Record<string, unknown>>;
    const yunet = models.find((m) => m.file === "face_detection_yunet_2026may.onnx");
    expect(yunet).toBeDefined();
    expect(yunet!.shipped).not.toBe(false);
  });
});
