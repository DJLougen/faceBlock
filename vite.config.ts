import { createReadStream, existsSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [{
    name: "local-ort-runtime",
    configureServer(server) {
      // ORT dynamically imports its glue. Serve public glue before Vite's
      // module transform rejects a public .mjs requested with ?import.
      server.middlewares.use((req, res, next) => {
        const pathname = (req.url ?? "").split("?")[0]!;
        if (!/^\/ort\/ort-wasm[\w.-]+\.mjs$/.test(pathname)) return next();
        const file = resolve("demo/public", pathname.slice(1));
        if (!existsSync(file)) return next();
        res.setHeader("Content-Type", "text/javascript");
        createReadStream(file).pipe(res);
      });
    },
  }],
  root: "demo",
  publicDir: "public",
  server: { host: "127.0.0.1", port: 5173, strictPort: true },
  preview: { host: "127.0.0.1", port: 5173, strictPort: true },
});
