import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";

// Exercise the actual offscreen functions without loading ORT or a Chrome DOM.
// Deterministic time: window.setTimeout is captured and fired manually — no
// real wall-clock waits.
//
// Realm note: runInNewContext's realm cannot be registered with Bun's test
// matcher (a VM-realm promise handed to expect().rejects hangs the runner).
// All awaited promises are wrapped into host-realm promises via .then, and
// admission release (several .finally hops deep in the VM realm) is drained
// with condition-driven setImmediate ticks — never fixed sleeps.
const source = readFileSync(new URL("../../extension/offscreen.ts", import.meta.url), "utf8");
function functionSource(name: string): string {
  const start = source.search(new RegExp(`^(?:async )?function ${name}[<(]`, "m"));
  if (start < 0) throw new Error(`Missing offscreen function: ${name}`);
  const end = source.indexOf("\n}", start);
  if (end < 0) throw new Error(`Missing function boundary: ${name}`);
  return source.slice(start, end + 2);
}
const executable = new Bun.Transpiler({ loader: "ts", target: "bun" }).transformSync(
  `let previewInFlight = 0;\n${functionSource("withTimeout")}\n${functionSource("resolvePreview")}\n` +
    `globalThis.API = { resolvePreview, inFlight: () => previewInFlight };`,
);

function vmRejects(p: Promise<unknown>, match: string): Promise<void> {
  return p.then(
    () => {
      throw new Error(`expected rejection "${match}", got fulfillment`);
    },
    (e: unknown) => {
      const message = e instanceof Error ? e.message : String(e);
      if (!message.includes(match)) throw new Error(`expected "${match}", got "${message}"`);
    },
  );
}

function vmResolves<T>(p: Promise<T>): Promise<T> {
  return p.then((value) => value, (e: unknown) => {
    throw e instanceof Error ? e : new Error(String(e));
  });
}

test("preview response times out while admission stays held until real settlement", async () => {
  const timers: (() => void)[] = [];
  const work = Promise.withResolvers<{ preview: { name: string } }>();
  const sandbox: Record<string, unknown> = {
    Promise,
    clearTimeout: (callback: () => void) => {
      const i = timers.indexOf(callback);
      if (i !== -1) timers.splice(i, 1);
    },
    MAX_CONCURRENT_PREVIEWS: 2,
    JOB_TIMEOUT_MS: 120_000,
    buildPreview: () => work.promise,
    window: {
      setTimeout(callback: () => void) {
        timers.push(callback);
        return callback;
      },
      clearTimeout: (callback: () => void) => {
        const i = timers.indexOf(callback);
        if (i !== -1) timers.splice(i, 1);
      },
    },
  };
  runInNewContext(executable, sandbox);
  const api = sandbox.API as {
    resolvePreview(name: string): Promise<{ preview: { name: string } }>;
    inFlight(): number;
  };

  const first = vmRejects(api.resolvePreview("fixture"), "enrollment preview timed out");
  const second = vmRejects(api.resolvePreview("fixture"), "enrollment preview timed out");
  expect(api.inFlight()).toBe(2);
  expect(timers.length).toBe(2);
  await vmRejects(api.resolvePreview("fixture"), "already in progress");

  for (const timeout of [...timers]) timeout();
  await Promise.all([first, second]);
  expect(api.inFlight()).toBe(2);
  expect(timers.length).toBe(0);
  await vmRejects(api.resolvePreview("fixture"), "already in progress");

  work.resolve({ preview: { name: "fixture" } });
  // Admission is released from a .finally chain several promise hops deep in
  // the VM realm; drain macrotasks until the condition is actually met
  // (setImmediate introduces no wall-clock delay).
  while (api.inFlight() > 0) {
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
  expect(api.inFlight()).toBe(0);
  expect(timers.length).toBe(0);
  const ok = await vmResolves(api.resolvePreview("fixture"));
  expect(ok).toEqual({ preview: { name: "fixture" } });
  expect(api.inFlight()).toBe(0);
  expect(timers.length).toBe(0);
});
