/**
 * Static server for the fixture pages.
 *
 * Deliberately tiny: the fixtures are plain HTML that load the bundled models
 * and runtimes, so all they need is a file server. This replaces a full
 * bundler dependency for what amounts to `open a page`.
 *
 *   bun run dev   ->  http://127.0.0.1:5173/extension-test.html
 */
import { join, normalize } from "node:path";

const ROOT = join(import.meta.dir, "..", "demo");
const PORT = Number(Bun.env.PORT ?? 5173);
const TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
  ".task": "application/octet-stream",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".mp4": "video/mp4",
};

const server = Bun.serve({
  port: PORT,
  hostname: "127.0.0.1",
  async fetch(req) {
    const path = decodeURIComponent(new URL(req.url).pathname);
    // Contain every request inside demo/ — never serve outside the fixture tree.
    const rel = normalize(path).replace(/^(\.\.[/\\])+/, "").replace(/^[/\\]+/, "");
    const name = rel === "" ? "extension-test.html" : rel;
    // Fixture pages live in demo/; their assets (models, ort, samples) live in
    // demo/public/ and are addressed from the root, exactly as before.
    let file = Bun.file(join(ROOT, name));
    if (!(await file.exists())) file = Bun.file(join(ROOT, "public", name));
    if (!(await file.exists())) file = Bun.file(join(ROOT, "public", "samples", name));
    if (!(await file.exists())) return new Response("not found", { status: 404 });
    const ext = name.slice(name.lastIndexOf("."));
    return new Response(file, {
      headers: { "content-type": TYPES[ext] ?? "application/octet-stream" },
    });
  },
});

console.log(`Fixture pages on http://127.0.0.1:${server.port}/`);
console.log(`  extension-test.html   images      (screenshot 03)`);
console.log(`  video-test.html       video       (screenshot 04)`);
console.log(`  pose-test.html        profile views`);
console.log(`  video-edge-test.html  shadow-dom / late / cross-origin video`);
