/**
 * verify-browser.ts — finite, reproducible end-to-end verification of the
 * FaceBlock extension inside a parent-launched ISOLATED Chromium.
 *
 * The script never launches a browser and never touches a real profile: it
 * attaches to an existing CDP endpoint and refuses to run unless the browser
 * process provably runs with its own --user-data-dir and a matching
 * --remote-debugging-port. All fixture traffic is loopback; nothing is
 * uploaded and no embeddings leave extension pages.
 *
 * Usage:
 *   bun scripts/verify-browser.ts \
 *     --cdp http://127.0.0.1:9337 \
 *     --base-url http://127.0.0.1:5175 \
 *     --extension-dir /abs/path/to/dist-extension \
 *     [--cross-origin-url http://127.0.0.1:8899] \
 *     [--output provenance/browser-smoke.json] \
 *     [--evidence-dir <dir for screenshots>] \
 *     [--timeout-ms 300000]
 *
 * Exit codes: 0 all checks passed · 1 one or more checks failed ·
 *             2 usage/preflight/isolation error (no checks ran).
 *
 * Output JSON marks every fixture's evidence class (self-match vs
 * same-identity-different-artifact vs control) — demo results are a wiring
 * regression, never held-out accuracy.
 */
import { createHash } from "node:crypto";
import { mkdtempSync, mkdirSync, realpathSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";

/* ============================== args ============================== */

interface Args {
  cdp: string;
  baseUrl: string;
  extensionDir: string;
  crossOriginUrl: string | null;
  output: string | null;
  evidenceDir: string;
  timeoutMs: number;
}

function parseArgs(argv: string[]): Args {
  const get = (name: string): string | null => {
    const i = argv.indexOf(name);
    return i >= 0 && i + 1 < argv.length ? argv[i + 1]! : null;
  };
  const cdp = get("--cdp");
  const baseUrl = get("--base-url");
  const extensionDir = get("--extension-dir");
  if (!cdp || !baseUrl || !extensionDir) {
    console.error(
      "usage: bun scripts/verify-browser.ts --cdp http://127.0.0.1:PORT " +
        "--base-url http://127.0.0.1:PORT --extension-dir /abs/path " +
        "[--cross-origin-url URL] [--output FILE] [--evidence-dir DIR] [--timeout-ms N]",
    );
    process.exit(2);
  }
  if (!isAbsolute(extensionDir)) {
    console.error("--extension-dir must be an absolute path");
    process.exit(2);
  }
  return {
    cdp,
    baseUrl: baseUrl.replace(/\/+$/, ""),
    extensionDir,
    crossOriginUrl: get("--cross-origin-url")?.replace(/\/+$/, "") ?? null,
    output: get("--output"),
    evidenceDir: get("--evidence-dir") ?? mkdtempSync(join(tmpdir(), "faceblock-verify-")),
    timeoutMs: Number(get("--timeout-ms") ?? 300_000),
  };
}

const args = parseArgs(process.argv.slice(2));

/* ============================== report ============================== */

type Status = "pass" | "fail" | "skip";

interface Check {
  id: string;
  status: Status;
  detail: string;
  data?: unknown;
}

const checks: Check[] = [];
const notes: string[] = [];
const models: Record<string, unknown> = {};
const fixtureAssets: Record<string, unknown> = {};
const extensionBuild: Record<string, unknown> = {};
let timedOut = false;

function record(id: string, status: Status, detail: string, data?: unknown): void {
  checks.push(data === undefined ? { id, status, detail } : { id, status, detail, data });
  const mark = status === "pass" ? "PASS" : status === "fail" ? "FAIL" : "SKIP";
  console.log(`[${mark}] ${id} — ${detail}`);
}

/** Run one gated check; thrown errors become failures with context. */
async function check(id: string, fn: () => Promise<string | void>): Promise<boolean> {
  try {
    const detail = (await fn()) ?? "ok";
    record(id, "pass", detail);
    return true;
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    record(id, "fail", msg);
    // A global-timeout failure is fatal: the CDP socket is already closed and
    // every later check would fail identically. Bail to the report writer.
    if (timedOut) throw e;
    return false;
  }
}

function assert(cond: unknown, message: string): asserts cond {
  if (!cond) throw new Error(message);
}

function sleep(ms: number): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  setTimeout(resolve, ms);
  return promise;
}

/* ============================== CDP client ============================== */

interface CdpEvent {
  method: string;
  params?: Record<string, unknown>;
  sessionId?: string;
}

class Cdp {
  private ws: WebSocket;
  private nextId = 1;
  private pending = new Map<
    number,
    { resolve: (v: unknown) => void; reject: (e: Error) => void; timer: Timer }
  >();
  readonly events: CdpEvent[] = [];
  /** Set once the socket is closed; pending and future sends reject fast. */
  closed = false;
  private openPromise: Promise<void>;

  constructor(wsUrl: string) {
    this.ws = new WebSocket(wsUrl);
    const { promise, resolve, reject } = Promise.withResolvers<void>();
    this.openPromise = promise;
    const timer = setTimeout(() => reject(new Error("CDP websocket open timed out")), 10_000);
    this.ws.addEventListener("open", () => {
      clearTimeout(timer);
      resolve();
    });
    this.ws.addEventListener("error", () => {
      clearTimeout(timer);
      reject(new Error("CDP websocket failed to connect"));
    });
    this.ws.addEventListener("message", (ev) => {
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(typeof ev.data === "string" ? ev.data : String(ev.data));
      } catch {
        return;
      }
      if (typeof msg.id === "number") {
        const p = this.pending.get(msg.id);
        if (!p) return;
        this.pending.delete(msg.id);
        clearTimeout(p.timer);
        if (msg.error) {
          const err = msg.error as { message?: string };
          p.reject(new Error(err.message ?? "CDP error"));
        } else {
          p.resolve(msg.result);
        }
      } else if (typeof msg.method === "string") {
        this.events.push(msg as unknown as CdpEvent);
      }
    });
  }

  async ready(): Promise<void> {
    await this.openPromise;
  }

  send<T = Record<string, unknown>>(
    method: string,
    params: Record<string, unknown> = {},
    sessionId?: string,
    timeoutMs = 30_000,
  ): Promise<T> {
    if (this.closed) return Promise.reject(new Error("CDP connection closed"));
    const id = this.nextId++;
    const payload: Record<string, unknown> = { id, method, params };
    if (sessionId) payload.sessionId = sessionId;
    const { promise, resolve, reject } = Promise.withResolvers<T>();
    const timer = setTimeout(() => {
      this.pending.delete(id);
      reject(new Error(`${method} timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, timer });
    this.ws.send(JSON.stringify(payload));
    return promise;
  }

  close(): void {
    this.closed = true;
    for (const p of this.pending.values()) {
      clearTimeout(p.timer);
      p.reject(new Error("CDP connection closed"));
    }
    this.pending.clear();
    try {
      this.ws.close();
    } catch {
      /* already closed */
    }
  }
}

/* ---------- per-target helpers ---------- */

interface Target {
  id: string;
  sessionId: string;
}

async function openTarget(cdp: Cdp, url: string): Promise<Target> {
  const { targetId } = await cdp.send<{ targetId: string }>("Target.createTarget", {
    url,
    newWindow: false,
    background: false,
  });
  const { sessionId } = await cdp.send<{ sessionId: string }>("Target.attachToTarget", {
    targetId,
    flatten: true,
  });
  // Deterministic viewport: the fixture grid is ~11 cards over several rows,
  // and the content script only samples media the IntersectionObserver can
  // see. 1440x1200 keeps every fixture element reachable by scrollIntoView.
  await cdp.send(
    "Emulation.setDeviceMetricsOverride",
    { width: 1440, height: 1200, deviceScaleFactor: 1, mobile: false },
    sessionId,
  );
  return { id: targetId, sessionId };
}

async function closeTarget(cdp: Cdp, t: Target): Promise<void> {
  try {
    await cdp.send("Target.detachFromTarget", { sessionId: t.sessionId }, undefined, 5_000);
  } catch {
    /* best effort */
  }
  try {
    await cdp.send("Target.closeTarget", { targetId: t.id }, undefined, 5_000);
  } catch {
    /* best effort */
  }
}

async function evaluate<T = unknown>(
  cdp: Cdp,
  sessionId: string,
  expression: string,
  timeoutMs = 30_000,
  contextId?: number,
): Promise<T> {
  const res = await cdp.send<{
    result?: { value?: T };
    exceptionDetails?: { exception?: { description?: string }; text?: string };
  }>(
    "Runtime.evaluate",
    { expression, returnByValue: true, awaitPromise: true, ...(contextId ? { contextId } : {}) },
    sessionId,
    timeoutMs,
  );
  if (res.exceptionDetails) {
    throw new Error(
      res.exceptionDetails.exception?.description ?? res.exceptionDetails.text ?? "evaluation failed",
    );
  }
  return res.result?.value as T;
}

async function waitFor(
  cdp: Cdp,
  sessionId: string,
  expression: string,
  timeoutMs: number,
  what: string,
  contextId?: number,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let lastErr = "";
  while (Date.now() < deadline) {
    if (cdp.closed) throw new Error(`CDP connection closed while waiting for ${what}`);
    try {
      if (await evaluate<boolean>(cdp, sessionId, expression, 10_000, contextId)) return;
    } catch (e) {
      lastErr = e instanceof Error ? e.message : String(e);
    }
    await sleep(250);
  }
  throw new Error(
    `timed out (${Math.round(timeoutMs / 1000)}s) waiting for ${what}${lastErr ? ` — last error: ${lastErr}` : ""}`,
  );
}

async function screenshot(cdp: Cdp, sessionId: string, name: string): Promise<string> {
  const res = await cdp.send<{ data: string }>(
    "Page.captureScreenshot",
    { format: "png" },
    sessionId,
    30_000,
  );
  const path = join(args.evidenceDir, `${name}.png`);
  writeFileSync(path, Buffer.from(res.data, "base64"));
  return path;
}

/* ---------- mask collection via pierced DOM ---------- */

interface MaskBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

interface DomNode {
  nodeId: number;
  nodeName?: string;
  nodeValue?: string;
  attributes?: string[];
  children?: DomNode[];
  shadowRoots?: DomNode[];
  contentDocument?: DomNode;
}

/**
 * Visible in-page warning badges (e.g. "FaceBlock: video not analyzable"):
 * #text nodes inside the closed overlay layer. Same pierce walk as masks.
 */
async function collectWarnings(cdp: Cdp, sessionId: string): Promise<string[]> {
  const { root } = await cdp.send<{ root: DomNode }>(
    "DOM.getDocument",
    { depth: -1, pierce: true },
    sessionId,
  );
  const warnings: string[] = [];
  const walk = (node: DomNode, inLayer: boolean): void => {
    const attrs = attrsOf(node);
    const isLayer = "data-fb-layer" in attrs;
    if (inLayer && node.nodeName === "#text" && typeof node.nodeValue === "string") {
      const t = node.nodeValue.trim();
      if (t.startsWith("FaceBlock:")) warnings.push(t);
    }
    for (const sr of node.shadowRoots ?? []) walk(sr, inLayer || isLayer);
    for (const c of node.children ?? []) walk(c, inLayer || isLayer);
    if (node.contentDocument) walk(node.contentDocument, inLayer || isLayer);
  };
  walk(root, false);
  return warnings;
}

function attrsOf(node: DomNode): Record<string, string> {
  const out: Record<string, string> = {};
  const a = node.attributes ?? [];
  for (let i = 0; i + 1 < a.length; i += 2) out[a[i]!] = a[i + 1]!;
  return out;
}

/**
 * Visible mask boxes: the overlay host's shadow children with display:block.
 * The overlay root is a CLOSED shadow root — page JS cannot see it, but
 * DOM.getDocument with pierce:true exposes it to the debugger, which is the
 * only honest way to count what the user actually sees.
 */
async function collectMasks(cdp: Cdp, sessionId: string): Promise<MaskBox[]> {
  const { root } = await cdp.send<{ root: DomNode }>(
    "DOM.getDocument",
    { depth: -1, pierce: true },
    sessionId,
  );
  const masks: MaskBox[] = [];
  const walk = (node: DomNode, inLayer: boolean): void => {
    const attrs = attrsOf(node);
    const isLayer = "data-fb-layer" in attrs;
    if (inLayer && node.nodeName === "DIV") {
      const style = attrs.style ?? "";
      // Only the exact opaque-mask signature counts — the layer may also host
      // non-mask nodes (warnings, debug UI) which must never inflate the count.
      // Accept both "#000" and the serialized "rgb(0, 0, 0)" form.
      const isMask =
        /position:\s*absolute/.test(style) &&
        /background:\s*(#000|rgb\(0,\s*0,\s*0\))/.test(style) &&
        /pointer-events:\s*none/.test(style);
      const num = (prop: string): number => {
        const m = style.match(new RegExp(`${prop}:\\s*([-\\d.]+)px`));
        return m ? parseFloat(m[1]!) : 0;
      };
      const width = num("width");
      const height = num("height");
      if (isMask && !/display:\s*none/.test(style) && width > 0 && height > 0) {
        masks.push({ left: num("left"), top: num("top"), width, height });
      }
    }
    for (const sr of node.shadowRoots ?? []) walk(sr, inLayer || isLayer);
    for (const c of node.children ?? []) walk(c, inLayer || isLayer);
    if (node.contentDocument) walk(node.contentDocument, inLayer || isLayer);
  };
  walk(root, false);
  return masks;
}

interface Rect {
  left: number;
  top: number;
  right: number;
  bottom: number;
  width: number;
  height: number;
}

function intersects(a: MaskBox, r: Rect): boolean {
  return (
    a.width > 0 &&
    a.height > 0 &&
    a.left < r.right &&
    a.left + a.width > r.left &&
    a.top < r.bottom &&
    a.top + a.height > r.top
  );
}

/** getBoundingClientRect for a selector, or null when absent/zero-size. */
async function elementRect(cdp: Cdp, sessionId: string, selector: string): Promise<Rect | null> {
  return evaluate<Rect | null>(
    cdp,
    sessionId,
    `(() => { const el = document.querySelector(${JSON.stringify(selector)});
      if (!el) return null; const r = el.getBoundingClientRect();
      return { left: r.left, top: r.top, right: r.right, bottom: r.bottom, width: r.width, height: r.height }; })()`,
  );
}

/**
 * __faceblockStats lives in the content script's ISOLATED world — invisible
 * to Runtime.evaluate's default (main) context. Callers must pass the
 * isolated contextId resolved by resolveStatsContext().
 */
async function pageStats(
  cdp: Cdp,
  sessionId: string,
  contextId: number,
): Promise<Record<string, number>> {
  return evaluate<Record<string, number>>(
    cdp,
    sessionId,
    `(() => { const s = globalThis.__faceblockStats; if (!s) return null;
      const o = {}; for (const k of Object.keys(s)) o[k] = s[k];
      o.tracked = s.tracked; o.trackedVideos = s.trackedVideos; return o; })()`,
    30_000,
    contextId,
  );
}

/**
 * Find the content script's isolated execution context for a page session:
 * Runtime.enable emits executionContextCreated for every world; ours is the
 * non-default context where chrome.runtime.id is the extension's id and
 * __faceblockStats exists. Never injects anything into the page's main world.
 */
async function resolveStatsContext(
  cdp: Cdp,
  sessionId: string,
  extensionId: string,
  timeoutMs = 30_000,
): Promise<number> {
  await cdp.send("Runtime.enable", {}, sessionId);
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const contexts = cdp.events.filter(
      (e) => e.sessionId === sessionId && e.method === "Runtime.executionContextCreated",
    );
    for (const e of contexts) {
      const ctx = e.params?.context as { id?: number; auxData?: { isDefault?: boolean } } | undefined;
      const id = ctx?.id;
      if (id == null || ctx?.auxData?.isDefault !== false) continue;
      try {
        const probe = await evaluate<{ id: string | null; hasStats: boolean }>(
          cdp,
          sessionId,
          `({ id: globalThis.chrome?.runtime?.id ?? null, hasStats: !!globalThis.__faceblockStats })`,
          5_000,
          id,
        );
        if (probe.id === extensionId && probe.hasStats) return id;
      } catch {
        /* context may be gone; keep scanning */
      }
    }
    await sleep(250);
  }
  throw new Error("content-script isolated world (chrome.runtime.id + __faceblockStats) not found");
}

/** Scroll an element into view and give IntersectionObserver a beat to fire. */
async function scrollIntoView(cdp: Cdp, sessionId: string, selector: string): Promise<void> {
  await evaluate(
    cdp,
    sessionId,
    `(() => { const el = document.querySelector(${JSON.stringify(selector)});
      if (el) el.scrollIntoView({ block: "center", inline: "center" }); })()`,
  );
  await sleep(400);
}

/* ============================== preflight ============================== */

function isLoopback(url: string): boolean {
  try {
    const u = new URL(url);
    return u.hostname === "127.0.0.1" || u.hostname === "localhost" || u.hostname === "[::1]";
  } catch {
    return false;
  }
}

async function probe(
  url: string,
  timeoutMs = 5_000,
): Promise<{ ok: boolean; status: number; bytes: number }> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { signal: ctrl.signal });
    const body = await res.arrayBuffer();
    return { ok: res.ok, status: res.status, bytes: body.byteLength };
  } catch {
    return { ok: false, status: 0, bytes: 0 };
  } finally {
    clearTimeout(timer);
  }
}

async function hashFile(path: string): Promise<{ sha256: string; bytes: number }> {
  const buf = await Bun.file(path).arrayBuffer();
  return { sha256: createHash("sha256").update(new Uint8Array(buf)).digest("hex"), bytes: buf.byteLength };
}

/** Chrome derives an unpacked extension's id from sha256(abs path), nibbles mapped 0-f -> a-p. */
function extensionIdFromPath(dir: string): string {
  const hash = createHash("sha256").update(dir, "utf8").digest("hex").slice(0, 32);
  return hash.replace(/[0-9a-f]/g, (c) => String.fromCharCode(97 + parseInt(c, 16)));
}

/* ============================== isolation ============================== */

interface Isolation {
  product: string;
  userDataDir: string;
  extensionDir: string;
  evidence: string;
}

/**
 * Prove the attached browser is a throwaway instance, using only CDP:
 * Browser.getBrowserCommandLine returns the browser process's own argv.
 * Fail closed — if we cannot prove an isolated profile we never drive it.
 */
async function verifyIsolation(cdp: Cdp, extensionDir: string): Promise<Isolation> {
  const version = await cdp.send<{ product?: string; userAgent?: string }>("Browser.getVersion");
  const product = version.product ?? "unknown";

  const cl = await cdp.send<{ arguments?: string[]; executablePath?: string }>(
    "Browser.getBrowserCommandLine",
  );
  const argv = cl.arguments ?? [];
  assert(argv.length > 0, "Browser.getBrowserCommandLine returned no argv — cannot prove isolation");

  const flagValue = (name: string): string | null => {
    for (let i = 0; i < argv.length; i++) {
      const a = argv[i]!;
      if (a === name) return argv[i + 1] ?? null;
      if (a.startsWith(`${name}=`)) return a.slice(name.length + 1);
    }
    return null;
  };

  const userDataDir = flagValue("--user-data-dir");
  assert(
    userDataDir != null && userDataDir.length > 0,
    "browser was NOT launched with --user-data-dir — refusing to touch what may be a real profile",
  );
  const home = process.env.HOME ?? "";
  const forbiddenRoots = [
    `${home}/Library/Application Support/Google/Chrome`,
    `${home}/Library/Application Support/Chromium`,
    `${home}/Library/Application Support/Microsoft Edge`,
    `${home}/Library/Application Support/BraveSoftware`,
  ];
  assert(
    !forbiddenRoots.some((root) => userDataDir === root || userDataDir.startsWith(root + "/")),
    `browser user-data-dir ${userDataDir} is a real browser profile — refusing`,
  );
  // The profile must live in the OS temp dir AND carry the marker file the
  // parent writes before launch — together they prove a purpose-created
  // throwaway profile rather than an arbitrary pre-existing directory.
  const uddReal = realpathSync(userDataDir);
  const tmpReal = realpathSync(tmpdir());
  assert(
    uddReal === tmpReal || uddReal.startsWith(tmpReal + "/"),
    `user-data-dir ${uddReal} is not inside ${tmpReal} — refusing a non-temporary profile`,
  );
  const marker = Bun.file(join(uddReal, ".faceblock-test-profile"));
  assert(
    (await marker.exists()) && (await marker.text()) === "faceblock-isolated-verification\n",
    `profile ${uddReal} lacks the .faceblock-test-profile marker — the parent must create it before launch`,
  );

  const cdpPort = new URL(args.cdp).port;
  const debugPort = flagValue("--remote-debugging-port");
  assert(
    debugPort === cdpPort,
    `browser argv lacks --remote-debugging-port=${cdpPort} — cannot prove this endpoint belongs to the isolated instance`,
  );

  // The extension under test must be the one the parent loaded into this
  // isolated browser — not some other build left over in a reused profile.
  const loaded = argv
    .map((a, i) =>
      a === "--load-extension" ? argv[i + 1] : a.startsWith("--load-extension=") ? a.slice(17) : null,
    )
    .filter((v): v is string => typeof v === "string" && v.length > 0);
  const want = realpathSync(extensionDir);
  assert(
    loaded.some((p) => realpathSync(p) === want),
    `browser was not launched with --load-extension=${want} (found: ${loaded.join(", ") || "none"})`,
  );

  return {
    product,
    userDataDir,
    extensionDir: want,
    evidence: `product=${product} user-data-dir=${userDataDir} load-extension=${want}`,
  };
}

/** Pull one numeric constant (or `?? N` default) out of a source file. */
async function sourceConstant(file: string, name: string): Promise<number | null> {
  const text = await Bun.file(file).text().catch(() => "");
  const m =
    text.match(new RegExp(`(?:const|let|var)\\s+${name}\\s*=\\s*([\\d.]+)`)) ??
    text.match(new RegExp(`${name}\\s*\\?\\?\\s*([\\d.]+)`));
  return m ? Number(m[1]) : null;
}

/* ============================== extension plumbing ============================== */

interface TargetInfo {
  targetId: string;
  type: string;
  url: string;
}

async function listTargets(cdp: Cdp): Promise<TargetInfo[]> {
  const { targetInfos } = await cdp.send<{ targetInfos: TargetInfo[] }>("Target.getTargets");
  return targetInfos;
}

async function findExtensionId(cdp: Cdp, candidates: string[]): Promise<string> {
  const targets = await listTargets(cdp);
  for (const t of targets) {
    const m = t.url.match(/^chrome-extension:\/\/([a-p]{32})\//);
    if (m) return m[1]!;
  }
  return candidates[0] ?? "";
}

/* ---------- message shapes (extension/protocol.ts) ---------- */

interface PreviewFace {
  url: string;
  filename: string;
  embedding: number[];
}

interface Preview {
  name: string;
  identityId?: string;
  candidatesTried: number;
  facesFound: number;
  kept: PreviewFace[];
  rejected: { url: string; reason: string }[];
}

interface StoredIdentity {
  id: string;
  name: string;
}

interface BlockState {
  identities: StoredIdentity[];
  enabled: boolean;
  revision: number;
}

interface FaceDiag {
  box: { x: number; y: number; width: number; height: number };
  bestCosine: number | null;
  bestIdentityId: string | null;
  matched: boolean;
  matchedIdentityId?: string | null;
}

interface Diagnostics {
  faceCount: number;
  threshold: number;
  minAgreements: number;
  droppedIdentities: number;
  faces: FaceDiag[];
}

interface BgResponse {
  ok?: boolean;
  error?: string;
  state?: BlockState;
  preview?: Preview;
  identity?: StoredIdentity;
  result?: { diagnostics?: Diagnostics };
}

/** Send a message from an extension page; resolves with the response or an {ok:false} shell. */
function pageSendExpr(message: Record<string, unknown>, timeoutMs: number): string {
  return `new Promise((resolve) => {
    const t = setTimeout(() => resolve({ ok: false, error: "sendMessage timed out" }), ${timeoutMs});
    try {
      chrome.runtime.sendMessage(${JSON.stringify(message)}, (r) => {
        clearTimeout(t);
        resolve(r === undefined ? { ok: false, error: chrome.runtime.lastError?.message ?? "no response" } : r);
      });
    } catch (e) { clearTimeout(t); resolve({ ok: false, error: String(e) }); }
  })`;
}

/* ============================== main ============================== */

interface FixtureResult {
  url: string;
  evidenceClass: "self-match" | "same-identity-different-artifact" | "control" | "mixed";
  stats?: Record<string, number>;
  masks?: MaskBox[];
  diagnostics?: unknown;
  screenshot?: string;
}

const fixtures: Record<string, FixtureResult> = {};
const createdTargets: Target[] = [];
let cdp: Cdp | null = null;
let exitCode = 0;

async function main(): Promise<void> {
  /* ---- preflight ---- */
  assert(isLoopback(args.cdp), "--cdp must be a loopback URL");
  assert(isLoopback(args.baseUrl), "--base-url must be a loopback URL");
  if (args.crossOriginUrl) {
    assert(isLoopback(args.crossOriginUrl), "--cross-origin-url must be loopback");
  }
  mkdirSync(args.evidenceDir, { recursive: true });

  const manifestFile = Bun.file(join(args.extensionDir, "manifest.json"));
  assert(await manifestFile.exists(), `no manifest.json in --extension-dir ${args.extensionDir}`);
  const manifest = JSON.parse(await manifestFile.text()) as {
    name?: string;
    version?: string;
    background?: { service_worker?: string };
  };
  for (const rel of [
    manifest.background?.service_worker ?? "background.js",
    "offscreen.html",
    "offscreen.js",
    "options.html",
    "options.js",
    "content.js",
    "references.json",
    "models/face_detection_yunet_2026may.onnx",
    "models/w600k_mbf.onnx",
    "samples/theo.jpg",
    "samples/dwarkesh.jpg",
    "samples/unrelated.jpg",
  ]) {
    assert(await Bun.file(join(args.extensionDir, rel)).exists(), `extension dir missing ${rel}`);
  }

  const fixtureProbe = await probe(`${args.baseUrl}/verify-extension.html`);
  assert(fixtureProbe.ok, `fixture server ${args.baseUrl} not serving verify-extension.html`);
  const videoProbe = await probe(`${args.baseUrl}/video-theo.mp4`);
  assert(videoProbe.ok && videoProbe.bytes > 0, `fixture server missing video-theo.mp4`);

  // Cross-origin VIDEO is a required fixture: the parent provides a second
  // loopback origin (default 127.0.0.1:8899). If it is not serving the real
  // video, fail preflight rather than pass on empty evidence. The
  // cross-origin IMAGE subcase is optional and may skip.
  let crossImg: string | null = null;
  let crossVid: string | null = null;
  const crossBase = args.crossOriginUrl ?? "http://127.0.0.1:8899";
  const vidProbe = await probe(`${crossBase}/video-theo.mp4`);
  assert(
    vidProbe.ok && vidProbe.bytes > 0,
    `cross-origin fixture ${crossBase}/video-theo.mp4 unavailable — required for the tainted-canvas check`,
  );
  crossVid = `${crossBase}/video-theo.mp4`;
  const imgProbe = await probe(`${crossBase}/samples/theo.jpg`);
  if (imgProbe.ok) {
    crossImg = `${crossBase}/samples/theo.jpg`;
  } else {
    notes.push(`cross-origin image asset not served at ${crossBase} — image CORS check will skip`);
  }

  /* ---- model + fixture hashes (repo-relative identities) ---- */
  const repoRoot = resolve(import.meta.dir, "..");
  for (const rel of [
    "models/face_detection_yunet_2026may.onnx",
    "models/w600k_mbf.onnx",
    "models/face_landmarker.task",
    "ort/ort-wasm-simd-threaded.wasm",
  ]) {
    const p = join(args.extensionDir, rel);
    if (await Bun.file(p).exists()) models[rel] = await hashFile(p);
  }
  for (const rel of [
    "demo/public/samples/theo.jpg",
    "demo/public/samples/theo-user-02.jpg",
    "demo/public/samples/dwarkesh.jpg",
    "demo/public/samples/unrelated.jpg",
    "demo/public/video-theo.mp4",
    "demo/verify-extension.html",
    "demo/video-edge-test.html",
    "demo/extension-test.html",
  ]) {
    const p = join(repoRoot, rel);
    if (await Bun.file(p).exists()) fixtureAssets[rel] = await hashFile(p);
  }

  /* ---- built bundle hashes: tie results to the exact runtime code ---- */
  for (const rel of [
    "background.js",
    "offscreen.js",
    "content.js",
    "options.js",
    "manifest.json",
    "build-info.json",
  ]) {
    const p = join(args.extensionDir, rel);
    if (await Bun.file(p).exists()) extensionBuild[rel] = await hashFile(p);
  }

  /* ---- connect + isolation ---- */
  const versionRes = await probe(`${args.cdp}/json/version`);
  assert(versionRes.ok, `CDP endpoint ${args.cdp} not reachable`);
  const versionJson = (await (await fetch(`${args.cdp}/json/version`)).json()) as {
    webSocketDebuggerUrl?: string;
  };
  assert(versionJson.webSocketDebuggerUrl, "CDP endpoint did not report webSocketDebuggerUrl");
  const conn = new Cdp(versionJson.webSocketDebuggerUrl);
  cdp = conn;
  await conn.ready();
  await conn.send("Target.setDiscoverTargets", { discover: true });

  const isolation = await verifyIsolation(conn, args.extensionDir);
  record("browser.isolated-profile", "pass", isolation.evidence);

  /* ---- extension discovery ---- */
  const extDirReal = realpathSync(args.extensionDir);
  const idCandidates = [extensionIdFromPath(args.extensionDir), extensionIdFromPath(extDirReal)];
  const extensionId = await findExtensionId(conn, idCandidates);
  assert(/^[a-p]{32}$/.test(extensionId), "could not determine extension id");

  /* ---- options page driver ---- */
  const options = await openTarget(conn, `chrome-extension://${extensionId}/options.html`);
  createdTargets.push(options);
  await waitFor(conn, options.sessionId, `document.readyState === "complete"`, 15_000, "options page load");
  const optionsTitle = await evaluate<string>(conn, options.sessionId, "document.title");
  assert(/faceblock/i.test(optionsTitle ?? ""), `options page title unexpected: ${optionsTitle}`);
  record("extension.options-loads", "pass", `options.html loaded (id ${extensionId})`);

  const sendBg = (msg: Record<string, unknown>, timeoutMs = 180_000) =>
    evaluate<BgResponse>(conn, options.sessionId, pageSendExpr(msg, timeoutMs), timeoutMs + 15_000);

  // Wake the service worker and confirm it answers.
  const state0 = await sendBg({ target: "background", type: "GET_STATE" });
  assert(state0.ok === true && state0.state, `GET_STATE failed: ${JSON.stringify(state0)}`);
  const swTargets = (await listTargets(conn)).filter(
    (t) => t.type === "service_worker" && t.url.includes(extensionId),
  );
  assert(swTargets.length > 0, "background service worker not found among CDP targets");
  record("extension.worker-loads", "pass", `service worker live: ${swTargets[0]!.url}`);

  const getState = async (): Promise<BlockState> => {
    const res = await sendBg({ target: "background", type: "GET_STATE" });
    assert(res.ok === true && res.state, `GET_STATE failed: ${JSON.stringify(res)}`);
    return res.state;
  };

  /* ---- enrollment contract ---- */
  const before = await getState();
  // A fresh isolated profile must start empty — a rerun against a dirty
  // profile would silently mutate state the run did not create.
  assert(
    before.identities.length === 0,
    `isolated profile already has ${before.identities.length} identities — launch a fresh profile`,
  );
  const previewRes = await sendBg({ target: "background", type: "RESOLVE_PREVIEW", name: "Theo Browne" });
  assert(previewRes.ok === true && previewRes.preview, `RESOLVE_PREVIEW failed: ${JSON.stringify(previewRes)}`);
  const preview = previewRes.preview;
  assert(preview.kept.length > 0, "curated RESOLVE_PREVIEW returned zero kept faces");
  for (const face of preview.kept) {
    assert(
      face.url.startsWith("chrome-extension://") || face.url.startsWith(args.baseUrl),
      `preview face url is not local: ${face.url}`,
    );
  }
  const afterPreview = await getState();
  assert(
    afterPreview.identities.length === before.identities.length &&
      afterPreview.revision === before.revision,
    "RESOLVE_PREVIEW mutated stored state — preview must not persist",
  );
  record(
    "enroll.preview-no-persist",
    "pass",
    `preview kept=${preview.kept.length} facesFound=${preview.facesFound} candidates=${preview.candidatesTried}; storage unchanged`,
    { kept: preview.kept.map((f) => f.filename), rejected: preview.rejected },
  );

  // BLOCK_NAME must no longer persist without confirmation.
  const blockRes = await sendBg({ target: "background", type: "BLOCK_NAME", name: "Theo Browne" });
  const afterBlock = await getState();
  assert(
    afterBlock.identities.length === before.identities.length &&
      afterBlock.revision === before.revision,
    "BLOCK_NAME persisted an identity without CONFIRM_ENROLL",
  );
  const blockPreview = blockRes.preview ?? null;
  record(
    "enroll.block-name-no-bypass",
    "pass",
    `BLOCK_NAME returned ${blockRes.ok ? (blockPreview ? "preview" : "ok") : `error: ${blockRes.error}`}; storage unchanged`,
  );

  // Confirm persists exactly one identity.
  const facesToKeep = (blockPreview ?? preview).kept;
  const identityId = (blockPreview ?? preview).identityId;
  const confirmRes = await sendBg({
    target: "background",
    type: "CONFIRM_ENROLL",
    name: "Theo Browne",
    faces: facesToKeep,
    ...(identityId ? { identityId } : {}),
  });
  assert(confirmRes.ok === true && confirmRes.identity, `CONFIRM_ENROLL failed: ${JSON.stringify(confirmRes)}`);
  const afterConfirm = await getState();
  const theo = afterConfirm.identities.find((i) => i.id === confirmRes.identity!.id);
  assert(theo, "confirmed identity missing from stored state");
  assert(
    afterConfirm.identities.length === before.identities.length + 1,
    "CONFIRM_ENROLL did not add exactly one identity",
  );
  record("enroll.confirm-persists", "pass", `identity "${theo.name}" (${theo.id}) persisted at revision ${afterConfirm.revision}`);

  // Dwarkesh via the same contract so extension-test.html masks both.
  const dwarkeshPreview = await sendBg({ target: "background", type: "RESOLVE_PREVIEW", name: "Dwarkesh Patel" });
  assert(
    dwarkeshPreview.ok === true && dwarkeshPreview.preview && dwarkeshPreview.preview.kept.length > 0,
    `Dwarkesh preview failed: ${JSON.stringify(dwarkeshPreview)}`,
  );
  const dwarkeshConfirm = await sendBg({
    target: "background",
    type: "CONFIRM_ENROLL",
    name: "Dwarkesh Patel",
    faces: dwarkeshPreview.preview!.kept,
  });
  assert(dwarkeshConfirm.ok === true, `Dwarkesh CONFIRM_ENROLL failed: ${JSON.stringify(dwarkeshConfirm)}`);
  record("enroll.second-identity", "pass", "Dwarkesh Patel enrolled via preview+confirm");

  // Refresh preview then cancel — through the real options UI, not a bare
  // message: click the identity's "Refresh reference photos" action, wait for
  // the preview, click Cancel, and confirm storage is byte-for-byte retained.
  const preRefresh = await getState();
  const uiFlow = await evaluate<{ clickedRefresh: boolean; previewShown: boolean; cancelled: boolean }>(
    conn,
    options.sessionId,
    `(async () => {
      const out = { clickedRefresh: false, previewShown: false, cancelled: false };
      const buttons = [...document.querySelectorAll("button")];
      const refresh = buttons.find((b) => /refresh reference photos/i.test(b.textContent ?? ""));
      if (!refresh) return out;
      out.clickedRefresh = true;
      refresh.click();
      const deadline = Date.now() + 120000;
      while (Date.now() < deadline) {
        const p = document.querySelector("#preview");
        if (p && !p.hidden) { out.previewShown = true; break; }
        await new Promise((r) => setTimeout(r, 250));
      }
      if (!out.previewShown) return out;
      const cancel = [...document.querySelectorAll(".preview-actions button")]
        .find((b) => /cancel/i.test(b.textContent ?? ""));
      if (!cancel) return out;
      cancel.click();
      await new Promise((r) => setTimeout(r, 300));
      const p = document.querySelector("#preview");
      out.cancelled = !p || p.hidden === true;
      return out;
    })()`,
    135_000,
  );
  assert(uiFlow.clickedRefresh, 'no "Refresh reference photos" button on the identity row');
  assert(uiFlow.previewShown, "refresh preview never appeared in the options UI");
  assert(uiFlow.cancelled, "Cancel did not dismiss the preview");
  const postRefresh = await getState();
  assert(
    postRefresh.revision === preRefresh.revision &&
      JSON.stringify(postRefresh.identities) === JSON.stringify(preRefresh.identities),
    "cancelled refresh mutated stored state",
  );
  record("enroll.refresh-cancel-retains", "pass", "UI refresh preview cancelled; stored identity unchanged");
  // Bogus explicit identityId must fail closed.
  const bogus = await sendBg({
    target: "background",
    type: "CONFIRM_ENROLL",
    name: "Theo Browne",
    faces: facesToKeep,
    identityId: "nonexistent-identity-id",
  });
  assert(bogus.ok === false, "CONFIRM_ENROLL accepted a nonexistent identityId");
  record("enroll.confirm-bad-identity-id", "pass", `rejected: ${String(bogus.error).slice(0, 120)}`);

  /* ---- diagnostics seam (extension-page ANALYZE with diagnostics:true) ---- */
  const identities = (await getState()).identities;
  const diagSend = (url: string) =>
    evaluate<BgResponse>(
      conn,
      options.sessionId,
      pageSendExpr(
        { target: "offscreen", type: "ANALYZE", url, identities, diagnostics: true },
        120_000,
      ),
      135_000,
    );
  const diagAvailable = await check("diag.extension-only-analyze", async () => {
    const res = await diagSend(`${args.baseUrl}/samples/theo.jpg`);
    assert(res.ok === true && res.result, `diagnostic ANALYZE failed: ${JSON.stringify(res)}`);
    assert(res.result.diagnostics, "ANALYZE response carried no diagnostics block");
    const d = res.result.diagnostics;
    return `faceCount=${d.faceCount} threshold=${d.threshold} faces=${d.faces.length}`;
  });
  const diagAnalyze = async (url: string): Promise<Diagnostics | null> => {
    if (!diagAvailable) return null;
    const res = await diagSend(url);
    return res.result?.diagnostics ?? null;
  };

  /**
   * Positive diagnostic: the image must actually contain detected faces AND at
   * least one matched face with a finite unthresholded score — a detection
   * miss must not masquerade as a tested negative.
   */
  const diagPositive = async (url: string, label: string): Promise<Diagnostics | null> => {
    const d = await diagAnalyze(url);
    if (d == null) return null;
    assert(d.faceCount > 0, `${label}: diagnostic detected zero faces — cannot prove matching`);
    const matched = d.faces.filter((f) => f.matched);
    assert(matched.length > 0, `${label}: faces detected but none matched a blocked identity`);
    for (const f of matched) {
      assert(
        f.bestCosine != null && Number.isFinite(f.bestCosine),
        `${label}: matched face has non-finite bestCosine`,
      );
    }
    return d;
  };

  /**
   * Control diagnostic: faces must be detected (so the negative is real) and
   * none may match. faceCount=0 would be a detection miss, not a clean control.
   */
  const diagControl = async (url: string, label: string): Promise<Diagnostics | null> => {
    const d = await diagAnalyze(url);
    if (d == null) return null;
    assert(d.faceCount > 0, `${label}: control image has no detected face — negative is untested`);
    const matched = d.faces.filter((f) => f.matched);
    assert(matched.length === 0, `${label}: control face matched a blocked identity`);
    return d;
  };

  /**
   * Frame diagnostics: capture a real video frame in the fixture page as JPEG,
   * then run the offscreen ANALYZE_FRAME path from the extension page so the
   * response carries per-face unthresholded diagnostics.
   */
  const diagFrame = async (videoSelector: string): Promise<Diagnostics | null> => {
    if (!diagAvailable) return null;
    const jpegBase64 = await evaluate<string | null>(
      conn,
      page.sessionId,
      `(() => {
        const v = document.querySelector(${JSON.stringify(videoSelector)});
        if (!v || v.readyState < 2 || v.videoWidth === 0) return null;
        const c = document.createElement("canvas");
        const scale = Math.min(1, 480 / v.videoWidth);
        c.width = Math.round(v.videoWidth * scale);
        c.height = Math.round(v.videoHeight * scale);
        const ctx = c.getContext("2d");
        if (!ctx) return null;
        try { ctx.drawImage(v, 0, 0, c.width, c.height); } catch { return null; }
        return c.toDataURL("image/jpeg", 0.7).split(",")[1] ?? null;
      })()`,
    );
    if (!jpegBase64) return null;
    const res = await evaluate<BgResponse>(
      conn,
      options.sessionId,
      pageSendExpr(
        { target: "offscreen", type: "ANALYZE_FRAME", jpegBase64, identities, diagnostics: true },
        120_000,
      ),
      135_000,
    );
    return res.result?.diagnostics ?? null;
  };

  // Image diagnostics captured once; the video check merges its frame diag in.
  let imageDiags: Record<string, unknown> = {};

  /* ---- fixture: verify-extension.html ---- */
  const fixtureParams = [
    crossImg ? `crossImg=${encodeURIComponent(crossImg)}` : "",
    crossVid ? `crossVid=${encodeURIComponent(crossVid)}` : "",
  ].filter(Boolean);
  const fixtureUrl =
    `${args.baseUrl}/verify-extension.html` + (fixtureParams.length ? `?${fixtureParams.join("&")}` : "");
  const page = await openTarget(conn, fixtureUrl);
  createdTargets.push(page);
  await waitFor(conn, page.sessionId, `document.readyState === "complete"`, 15_000, "fixture load");
  // Stats live in the content script's isolated world — resolve its context.
  const statsCtx = await resolveStatsContext(conn, page.sessionId, extensionId);

  const expectedMaskedImgs = ["#img-theo", "#img-dwarkesh", "#img-theo-dup", "#img-picture", "#img-late"];
  if (crossImg) expectedMaskedImgs.push("#img-cross");
  // img-swap starts on the control image and is flipped later by the harness.
  const expectedCleanImgs = ["#img-control", "#img-small", "#img-swap"];
  // img-small is never sent (sub-MIN_MEDIA_PX); control + swap are analysed.
  const expectedSent = expectedMaskedImgs.length + 2;

  // Scroll each media element into view so IntersectionObserver admits it —
  // the grid is taller than the viewport and offscreen media is never queued.
  for (const sel of [...expectedMaskedImgs, ...expectedCleanImgs, "#vid-main", "#vid-late"]) {
    await scrollIntoView(conn, page.sessionId, sel);
  }
  if (crossImg) await scrollIntoView(conn, page.sessionId, "#img-cross");
  if (crossVid) {
    await scrollIntoView(conn, page.sessionId, "#vid-cross");
    await scrollIntoView(conn, page.sessionId, "#vid-cors-recover");
  }

  await check("fixture.images-masked", async () => {
    await waitFor(
      conn,
      page.sessionId,
      `(() => { const s = globalThis.__faceblockStats; return s && s.sent >= ${expectedSent}; })()`,
      120_000,
      `content script to analyse ${expectedSent} images`,
      statsCtx,
    );
    await waitFor(
      conn,
      page.sessionId,
      `(() => { const s = globalThis.__faceblockStats; return s.masked >= ${expectedMaskedImgs.length}; })()`,
      60_000,
      `${expectedMaskedImgs.length} masked images`,
      statsCtx,
    );
    const stats = await pageStats(conn, page.sessionId, statsCtx);
    const masks = await collectMasks(conn, page.sessionId);
    const missing: string[] = [];
    for (const sel of expectedMaskedImgs) {
      const rect = await elementRect(conn, page.sessionId, sel);
      if (!rect) {
        missing.push(`${sel} (element missing)`);
        continue;
      }
      if (!masks.some((m) => intersects(m, rect))) missing.push(sel);
    }
    assert(missing.length === 0, `no mask over: ${missing.join(", ")}`);
    for (const sel of expectedCleanImgs) {
      const rect = await elementRect(conn, page.sessionId, sel);
      if (rect && masks.some((m) => intersects(m, rect))) {
        throw new Error(`unexpected mask over ${sel}`);
      }
    }
    assert(stats.errors === 0, `content script reported ${stats.errors} errors`);
    const shot = await screenshot(conn, page.sessionId, "fixture-images");
    fixtures["verify-extension"] = {
      url: fixtureUrl,
      evidenceClass: "mixed",
      stats,
      masks,
      diagnostics: (imageDiags = {
        theo: await diagPositive(`${args.baseUrl}/samples/theo.jpg`, "theo"),
        dwarkesh: await diagPositive(`${args.baseUrl}/samples/dwarkesh.jpg`, "dwarkesh"),
        control: await diagControl(`${args.baseUrl}/samples/unrelated.jpg`, "control"),
      }),
      screenshot: shot,
    };
    return `${expectedMaskedImgs.length} images masked, controls clean, stats=${JSON.stringify(stats)}`;
  });

  /* ---- src mutation: same element re-evaluates to a blocked image ---- */
  await check("fixture.image-src-swap", async () => {
    await evaluate(
      conn,
      page.sessionId,
      `document.querySelector("#img-swap").src = ${JSON.stringify(`${args.baseUrl}/samples/theo.jpg`)}`,
    );
    await scrollIntoView(conn, page.sessionId, "#img-swap");
    const deadline = Date.now() + 60_000;
    let covered = false;
    while (Date.now() < deadline) {
      const rect = await elementRect(conn, page.sessionId, "#img-swap");
      const masks = await collectMasks(conn, page.sessionId);
      if (rect && masks.some((m) => intersects(m, rect))) {
        covered = true;
        break;
      }
      await sleep(400);
    }
    assert(covered, "img-swap was never masked after src changed to a blocked face");
    return "src mutation re-evaluated and masked";
  });

  /* ---- fixture: video on the same page ---- */
  await check("fixture.video-masked", async () => {
    await scrollIntoView(conn, page.sessionId, "#vid-main");
    await waitFor(
      conn,
      page.sessionId,
      `(() => { const s = globalThis.__faceblockStats; return s && s.videoFramesSent >= 1 && s.videoMasked >= 1; })()`,
      90_000,
      "video frame analysis + mask",
      statsCtx,
    );
    // videoMasked is cumulative across every video on the page — prove the
    // mask over THIS element with a DOM-level poll, not a single snapshot.
    const deadline = Date.now() + 60_000;
    let covered = false;
    while (Date.now() < deadline) {
      const rect = await elementRect(conn, page.sessionId, "#vid-main");
      const masks = await collectMasks(conn, page.sessionId);
      if (rect && masks.some((m) => intersects(m, rect))) {
        covered = true;
        break;
      }
      await sleep(400);
    }
    assert(covered, "no mask over #vid-main");
    // Per-fixture frame evidence: run the real offscreen ANALYZE_FRAME path on
    // a captured frame and record detector counts + unthresholded similarities.
    const frameDiag = await diagFrame("#vid-main");
    assert(frameDiag != null, "frame diagnostics unavailable for #vid-main");
    assert(frameDiag.faceCount > 0, "frame diagnostics reported zero faces on the video");
    const fx = fixtures["verify-extension"];
    if (fx) fx.diagnostics = { ...imageDiags, videoFrame: frameDiag };
    return "video masked while playing";
  });

  await check("fixture.video-pause-retains", async () => {
    await evaluate(conn, page.sessionId, `document.querySelector("#vid-main").pause()`);
    await sleep(700);
    const masks = await collectMasks(conn, page.sessionId);
    const rect = await elementRect(conn, page.sessionId, "#vid-main");
    assert(rect && masks.some((m) => intersects(m, rect)), "mask lost while paused");
    await evaluate(conn, page.sessionId, `document.querySelector("#vid-main").play()`);
    return "mask retained across pause";
  });

  await check("fixture.video-seek-redetects", async () => {
    const statsBefore = await pageStats(conn, page.sessionId, statsCtx);
    await evaluate(
      conn,
      page.sessionId,
      `(() => { const v = document.querySelector("#vid-main"); v.currentTime = Math.min(1.0, (v.duration || 2) / 2); })()`,
    );
    await waitFor(
      conn,
      page.sessionId,
      `(() => { const s = globalThis.__faceblockStats; return s.videoFramesSent > ${statsBefore.videoFramesSent ?? 0}; })()`,
      30_000,
      "re-detection after seek",
      statsCtx,
    );
    return "seek triggered a fresh frame analysis";
  });

  await check("fixture.video-late-src", async () => {
    await scrollIntoView(conn, page.sessionId, "#vid-late");
    // DOM-level proof for THIS element, not just a cumulative counter.
    const deadline = Date.now() + 60_000;
    let covered = false;
    while (Date.now() < deadline) {
      const rect = await elementRect(conn, page.sessionId, "#vid-late");
      const masks = await collectMasks(conn, page.sessionId);
      if (rect && masks.some((m) => intersects(m, rect))) {
        covered = true;
        break;
      }
      await sleep(400);
    }
    assert(covered, "no mask over #vid-late after its src attached at 4s");
    return "late-src video recovered and masked";
  });

  await check("fixture.video-cross-origin", async () => {
    await scrollIntoView(conn, page.sessionId, "#vid-cross");
    await waitFor(
      conn,
      page.sessionId,
      `(() => { const s = globalThis.__faceblockStats; return s.videoUnanalyzable >= 1; })()`,
      45_000,
      "cross-origin video to be marked unanalyzable",
      statsCtx,
    );
    const stats = await pageStats(conn, page.sessionId, statsCtx);
    assert((stats.videoUnanalyzable ?? 0) >= 1, "videoUnanalyzable counter did not increment");
    // Visible marker: the user must see the "not analyzable" badge, not just a counter.
    const warnings = await collectWarnings(conn, page.sessionId);
    assert(
      warnings.some((w) => w.includes("not analyzable")),
      `no visible unanalyzable badge (warnings: ${JSON.stringify(warnings)})`,
    );
    return "tainted video reported once, stopped, and visibly badged";
  });

  /* ---- same-element CORS recovery: cross-origin src -> same-origin src ---- */
  await check("fixture.video-cors-recover", async () => {
    await scrollIntoView(conn, page.sessionId, "#vid-cors-recover");
    // Let it hit the tainted path first (both cross videos increment the counter).
    await waitFor(
      conn,
      page.sessionId,
      `(() => { const s = globalThis.__faceblockStats; return s.videoUnanalyzable >= 2; })()`,
      45_000,
      "recover video to be marked unanalyzable",
      statsCtx,
    );
    await evaluate(
      conn,
      page.sessionId,
      `(() => { const v = document.querySelector("#vid-cors-recover"); v.src = ${JSON.stringify(`${args.baseUrl}/video-theo.mp4`)}; v.load(); })()`,
    );
    const deadline = Date.now() + 60_000;
    let covered = false;
    while (Date.now() < deadline) {
      const rect = await elementRect(conn, page.sessionId, "#vid-cors-recover");
      const masks = await collectMasks(conn, page.sessionId);
      if (rect && masks.some((m) => intersects(m, rect))) {
        covered = true;
        break;
      }
      await sleep(400);
    }
    assert(
      covered,
      "video stayed unanalyzable after src moved to a same-origin URL — no recovery path",
    );
    return "same element recovered after src moved same-origin";
  });

  if (crossImg) {
    await check("fixture.image-cross-origin", async () => {
      await scrollIntoView(conn, page.sessionId, "#img-cross");
      const deadline = Date.now() + 60_000;
      let covered = false;
      while (Date.now() < deadline) {
        const rect = await elementRect(conn, page.sessionId, "#img-cross");
        const masks = await collectMasks(conn, page.sessionId);
        if (rect && masks.some((m) => intersects(m, rect))) {
          covered = true;
          break;
        }
        await sleep(400);
      }
      assert(covered, "cross-origin image not masked — extension fetch path failed");
      return "cross-origin image masked via extension-side fetch";
    });
  } else {
    record("fixture.image-cross-origin", "skip", "cross-origin image asset not served");
  }

  /* ---- fixture: video-edge-test.html cases ---- */
  // "late" exercises the 6s delayed-src recovery path on the same page shape.
  const edgeCases = ["shadow", "late", "cross", "plain"] as const;
  for (const name of edgeCases) {
    const id = `fixture.video-edge-${name}`;
    await check(id, async () => {
      const t = await openTarget(conn, `${args.baseUrl}/video-edge-test.html?case=${name}`);
      createdTargets.push(t);
      await waitFor(conn, t.sessionId, `document.readyState === "complete"`, 15_000, `${name} page load`);
      const edgeCtx = await resolveStatsContext(conn, t.sessionId, extensionId);
      // The mount point must be in view before IntersectionObserver admits the video.
      await evaluate(
        conn,
        t.sessionId,
        `(() => { const el = document.querySelector("#mount") ?? document.body;
          el.scrollIntoView({ block: "center" }); })()`,
      );
      await sleep(400);
      if (name === "cross") {
        await waitFor(
          conn,
          t.sessionId,
          `(() => { const s = globalThis.__faceblockStats; return s && s.videoUnanalyzable >= 1; })()`,
          45_000,
          "cross-origin video flagged unanalyzable",
          edgeCtx,
        );
        const warnings = await collectWarnings(conn, t.sessionId);
        assert(
          warnings.some((w) => w.includes("not analyzable")),
          `edge/cross: no visible unanalyzable badge (warnings: ${JSON.stringify(warnings)})`,
        );
        return "cross-origin video reported unanalyzable, control image unaffected";
      }
      // DOM-level mask proof for this page's video specifically.
      const deadline = Date.now() + 90_000;
      let covered = false;
      while (Date.now() < deadline) {
        const masks = await collectMasks(conn, t.sessionId);
        if (masks.length >= 1) {
          covered = true;
          break;
        }
        await sleep(400);
      }
      assert(covered, `no mask rendered on ${name} video`);
      const shot = await screenshot(conn, t.sessionId, `edge-${name}`);
      fixtures[`video-edge-${name}`] = {
        url: `${args.baseUrl}/video-edge-test.html?case=${name}`,
        evidenceClass: "same-identity-different-artifact",
        stats: await pageStats(conn, t.sessionId, edgeCtx),
        screenshot: shot,
      };
      return `${name} video masked`;
    });
  }

  /* ---- fixture: extension-test.html (original demo) ---- */
  await check("fixture.extension-test", async () => {
    const t = await openTarget(conn, `${args.baseUrl}/extension-test.html`);
    createdTargets.push(t);
    await waitFor(conn, t.sessionId, `document.readyState === "complete"`, 15_000, "extension-test load");
    const extCtx = await resolveStatsContext(conn, t.sessionId, extensionId);
    await waitFor(
      conn,
      t.sessionId,
      `(() => { const s = globalThis.__faceblockStats; return s && s.masked >= 2; })()`,
      90_000,
      "theo + dwarkesh masked",
      extCtx,
    );
    const masks = await collectMasks(conn, t.sessionId);
    for (const sel of ["#theo", "#dwarkesh"]) {
      const rect = await elementRect(conn, t.sessionId, sel);
      assert(rect && masks.some((m) => intersects(m, rect)), `no mask over ${sel}`);
    }
    const control = await elementRect(conn, t.sessionId, "#unrelated");
    assert(control && !masks.some((m) => intersects(m, control)), "control image was masked");
    const shot = await screenshot(conn, t.sessionId, "extension-test");
    fixtures["extension-test"] = {
      url: `${args.baseUrl}/extension-test.html`,
      evidenceClass: "self-match",
      stats: await pageStats(conn, t.sessionId, extCtx),
      masks,
      diagnostics: {
        theo: await diagPositive(`${args.baseUrl}/samples/theo.jpg`, "theo"),
        dwarkesh: await diagPositive(`${args.baseUrl}/samples/dwarkesh.jpg`, "dwarkesh"),
        control: await diagControl(`${args.baseUrl}/samples/unrelated.jpg`, "control"),
      },
      screenshot: shot,
    };
    return "both enrolled faces masked, control clean";
  });

  /* ---- privacy: page JS sees no chrome.runtime, no embeddings ---- */
  await check("privacy.page-isolation", async () => {
    const leaked = await evaluate<{ runtimeId: string | null; embeddingsInDom: boolean }>(
      conn,
      page.sessionId,
      `(() => {
        const out = { runtimeId: null, embeddingsInDom: false };
        try { out.runtimeId = globalThis.chrome?.runtime?.id ?? null; } catch {}
        out.embeddingsInDom = document.documentElement.innerHTML.includes("embedding");
        return out;
      })()`,
    );
    assert(leaked.runtimeId === null, `page JS can see chrome.runtime.id=${leaked.runtimeId}`);
    assert(!leaked.embeddingsInDom, "embedding data found in page DOM");
    return "no chrome.runtime in page world; no embeddings in DOM";
  });
}

/* ============================== run ============================== */

const watchdog = setTimeout(() => {
  // Route through the normal report path: close the socket so pending sends
  // reject, main() unwinds, and the JSON receipt is still written.
  timedOut = true;
  cdp?.close();
}, args.timeoutMs);

let fatal: string | null = null;
try {
  await main();
} catch (e) {
  fatal = e instanceof Error ? e.message : String(e);
  exitCode = 2;
}
clearTimeout(watchdog);
if (timedOut) {
  fatal = `global timeout ${args.timeoutMs}ms exceeded${fatal ? `: ${fatal}` : ""}`;
  exitCode = 2;
}

if (checks.some((c) => c.status === "fail")) exitCode = exitCode === 2 ? 2 : 1;

/* ---- cleanup: detach + close every target we created ---- */
// main() assigns cdp asynchronously; TS narrows the module-level binding to
// null at this point, so read it through a widening barrier instead.
const readCdp = (): Cdp | null => cdp;
const live = readCdp();
if (live) {
  for (const t of createdTargets) await closeTarget(live, t);
  live.close();
}

const report = {
  tool: "faceblock verify-browser",
  generatedAt: new Date().toISOString(),
  ok: exitCode === 0,
  fatal,
  inputs: {
    cdp: args.cdp,
    baseUrl: args.baseUrl,
    crossOriginUrl: args.crossOriginUrl,
    extensionDirEntries: "hashed below; absolute local paths intentionally omitted",
  },
  testedArtifact: {
    kind: "local-demo unpacked extension (dist-extension build)",
    directory: "--extension-dir (absolute path intentionally omitted)",
    packagedZipTested: false,
    note: "This run exercises the local-demo build with bundled sample media. It does NOT verify the packaged ZIP, which omits uncleared sample photos and uses an empty public reference resolver.",
  },
  interpretation:
    "Demo fixtures reuse the bundled enrollment references and a bundled video of the same " +
    "person. Masked results are SELF-MATCH / same-identity wiring evidence — NOT held-out " +
    "recognition accuracy or a false-positive rate. All inference ran on-device inside the " +
    "isolated browser profile; no photos, frames, or embeddings were uploaded — screenshots " +
    "and this JSON are the only artifacts, written to the local evidence directory.",
  policy: {
    detector: "YuNet ONNX via onnxruntime-web WASM (ort-wasm-simd-threaded)",
    embedder: "w600k_mbf ONNX (MobileFaceNet-style), L2-normalized cosine",
    matching: "score = max cosine over identity embeddings; block when score >= threshold",
    multiPass: "full-frame pass, then tiled pass gated on face count; merge drops duplicates by containment/IoU",
    video: "sampled frames (not every frame); masks track between analyses; tainted (cross-origin) video reported once and skipped",
    evidenceClass: "self-match — fixtures reuse enrollment references; not a held-out benchmark",
  },
  models,
  extensionBuild,
  fixtureAssets,
  checks,
  fixtures,
  notes,
  evidenceDir: args.evidenceDir,
};

const outPath = args.output ?? join(args.evidenceDir, "verify-results.json");
mkdirSync(join(outPath, ".."), { recursive: true });
writeFileSync(outPath, JSON.stringify(report, null, 2));
console.log(`\nresults: ${outPath}`);
console.log(`evidence: ${args.evidenceDir}`);
console.log(
  `status: ${exitCode === 0 ? "PASS" : "FAIL"} (${checks.filter((c) => c.status === "pass").length} pass, ${checks.filter((c) => c.status === "fail").length} fail, ${checks.filter((c) => c.status === "skip").length} skip)`,
);
process.exit(exitCode);
