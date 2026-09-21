/**
 * Minimal DOM/clock/chrome harness for driving the REAL extension/content.ts
 * under bun:test. No DOM library is installed, so this fakes exactly the
 * surface content.ts touches: elements, observers, timers, sendMessage.
 *
 * Install BEFORE importing content.ts — it captures globals at module eval:
 *
 *   const h = installHarness();
 *   await import("../../extension/content.ts");
 *
 * Time is manual: h.advance(ms) fires due timers/rAF; h.flush() drains
 * microtasks (resolved sendMessage promises).
 */

export class Clock {
  now = 0;
  private nextId = 1;
  private timers = new Map<number, { at: number; fn: () => void; interval: number | null }>();
  private rafHandles = new Map<number, { cancelled: boolean; timerId: number }>();

  setTimeout = (fn: () => void, ms = 0): number => {
    const id = this.nextId++;
    this.timers.set(id, { at: this.now + ms, fn, interval: null });
    return id;
  };
  setInterval = (fn: () => void, ms = 0): number => {
    const id = this.nextId++;
    this.timers.set(id, { at: this.now + ms, fn, interval: ms });
    return id;
  };
  clearTimer = (id: number | undefined): void => {
    if (id !== undefined) this.timers.delete(id);
  };

  /** rAF is deferred while the document is hidden, matching real browsers. */
  requestAnimationFrame = (fn: () => void): number => {
    const handle = this.nextId++;
    const state = { cancelled: false, timerId: 0 };
    const schedule = (): void => {
      state.timerId = this.setTimeout(() => {
        if (state.cancelled) return;
        if (doc.visibilityState !== "visible") {
          schedule();
          return;
        }
        fn();
      }, 16);
    };
    schedule();
    this.rafHandles.set(handle, state);
    return handle;
  };
  cancelAnimationFrame = (handle: number): void => {
    const state = this.rafHandles.get(handle);
    if (state) {
      state.cancelled = true;
      this.timers.delete(state.timerId);
    }
  };

  advance(ms: number): void {
    const target = this.now + ms;
    for (;;) {
      let dueId = -1;
      let dueAt = Infinity;
      for (const [id, t] of this.timers) {
        if (t.at <= target && t.at < dueAt) {
          dueAt = t.at;
          dueId = id;
        }
      }
      if (dueId < 0) break;
      const t = this.timers.get(dueId)!;
      this.now = t.at;
      if (t.interval === null) this.timers.delete(dueId);
      else t.at = this.now + t.interval;
      t.fn();
    }
    this.now = target;
  }
}

/* ---- fake DOM ---- */

let doc: FakeDocument;

class FakeNode {
  parentNode: FakeNode | null = null;
  children: FakeNode[] = [];
  get isConnected(): boolean {
    let n: FakeNode | null = this;
    while (n) {
      if (n === doc) return true;
      n = n.parentNode;
    }
    return false;
  }
  get parentElement(): FakeElement | null {
    return this.parentNode instanceof FakeElement ? this.parentNode : null;
  }
  appendChild<C extends FakeNode>(c: C): C {
    c.parentNode = this;
    this.children.push(c);
    return c;
  }
  remove(): void {
    if (this.parentNode) {
      const i = this.parentNode.children.indexOf(this);
      if (i >= 0) this.parentNode.children.splice(i, 1);
      this.parentNode = null;
    }
  }
  private *descendants(): Generator<FakeNode> {
    for (const c of this.children) {
      yield c;
      yield* c.descendants();
    }
  }
  private matches(sel: string): boolean {
    if (sel === "*") return this instanceof FakeElement;
    if (sel === "img") return this instanceof FakeImg;
    if (sel === "video") return this instanceof FakeVideo;
    return false;
  }
  querySelectorAll(sel: string): FakeNode[] {
    const out: FakeNode[] = [];
    for (const d of this.descendants()) if (d.matches(sel)) out.push(d);
    return out;
  }
  querySelector(sel: string): FakeNode | null {
    for (const d of this.descendants()) if (d.matches(sel)) return d;
    return null;
  }
}

export class FakeElement extends FakeNode {
  tagName = "DIV";
  style: Record<string, string> = {};
  textContent = "";
  rect = { left: 0, top: 0, width: 360, height: 360, right: 360, bottom: 360 };
  clientWidth = 360;
  clientHeight = 360;
  shadowRoot: FakeFragment | null = null;
  _shadow: FakeFragment | null = null;
  _cs: Record<string, string> | null = null;
  private listeners = new Map<string, Set<(e: unknown) => void>>();
  private attrs = new Map<string, string>();
  dataset: Record<string, string> = {};

  setAttribute(k: string, v: string): void {
    this.attrs.set(k, v);
  }
  getAttribute(k: string): string | null {
    return this.attrs.get(k) ?? null;
  }
  addEventListener(type: string, fn: (e: unknown) => void): void {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(fn);
  }
  removeEventListener(type: string, fn: (e: unknown) => void): void {
    this.listeners.get(type)?.delete(fn);
  }
  fire(type: string): void {
    // Handlers like onVideoSignal rely on `this` being the element.
    for (const fn of [...(this.listeners.get(type) ?? [])]) {
      fn.call(this, { target: this });
    }
  }
  listenerCount(type: string): number {
    return this.listeners.get(type)?.size ?? 0;
  }
  getBoundingClientRect(): typeof this.rect {
    return this.rect;
  }
  attachShadow(_opts: { mode: string }): FakeFragment {
    this._shadow = new FakeFragment();
    this._shadow.parentNode = this;
    harness.shadowRoots.push(this._shadow);
    return this._shadow;
  }
  replaceChildren(...nodes: FakeNode[]): void {
    for (const c of [...this.children]) c.remove();
    for (const n of nodes) this.appendChild(n);
  }
  append(...nodes: FakeNode[]): void {
    for (const n of nodes) this.appendChild(n);
  }
}

export class FakeImg extends FakeElement {
  tagName = "IMG";
  src = "";
  currentSrc = "";
  complete = true;
  naturalWidth = 400;
  naturalHeight = 300;
}

export class FakeVideo extends FakeElement {
  tagName = "VIDEO";
  src = "";
  currentSrc = "";
  paused = false;
  readyState = 4;
  videoWidth = 640;
  videoHeight = 360;
  /** When true, drawImage throws SecurityError (tainted canvas). */
  tainted = false;
  private rvfc = new Map<number, () => void>();
  private rvfcId = 1;
  requestVideoFrameCallback(cb: () => void): number {
    const id = this.rvfcId++;
    this.rvfc.set(id, cb);
    return id;
  }
  cancelVideoFrameCallback(id: number): void {
    this.rvfc.delete(id);
  }
  /** Deliver one presented frame: fires the pending rVFC callback once. */
  fireFrame(): void {
    const cbs = [...this.rvfc.values()];
    this.rvfc.clear();
    for (const cb of cbs) cb();
  }
  get pendingFrames(): number {
    return this.rvfc.size;
  }
}

export class FakeSource extends FakeElement {
  tagName = "SOURCE";
}

export class FakeFragment extends FakeNode {}

class FakeCanvas extends FakeElement {
  width = 0;
  height = 0;
  getContext(): null {
    return null;
  }
}

class FakeDocument extends FakeNode {
  documentElement = new FakeElement();
  body = new FakeElement();
  visibilityState = "visible";
  private listeners = new Map<string, Set<(e: unknown) => void>>();
  constructor() {
    super();
    this.documentElement.parentNode = this;
    this.appendChild(this.documentElement);
    this.documentElement.appendChild(this.body);
  }
  createElement(tag: string): FakeElement {
    if (tag === "img") return new FakeImg();
    if (tag === "video") return new FakeVideo();
    if (tag === "source") return new FakeSource();
    if (tag === "canvas") return new FakeCanvas();
    return new FakeElement();
  }
  addEventListener(type: string, fn: (e: unknown) => void): void {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(fn);
  }
  fire(type: string, target?: FakeNode): void {
    for (const fn of [...(this.listeners.get(type) ?? [])]) fn({ target });
  }
  setVisibility(state: string): void {
    this.visibilityState = state;
    this.fire("visibilitychange");
  }
}

/* ---- fake observers ---- */

class FakeIO {
  cb: (entries: { target: FakeNode; isIntersecting: boolean }[]) => void;
  observed = new Set<FakeNode>();
  constructor(cb: FakeIO["cb"], _opts?: unknown) {
    this.cb = cb;
    harness.io = this;
  }
  observe(el: FakeNode): void {
    this.observed.add(el);
  }
  unobserve(el: FakeNode): void {
    this.observed.delete(el);
  }
  fire(el: FakeNode, isIntersecting: boolean): void {
    this.cb([{ target: el, isIntersecting }]);
  }
}

class FakeRO {
  cb: (entries: { target: FakeNode }[]) => void;
  observed = new Set<FakeNode>();
  constructor(cb: FakeRO["cb"]) {
    this.cb = cb;
    harness.ro = this;
  }
  observe(el: FakeNode): void {
    this.observed.add(el);
  }
  unobserve(el: FakeNode): void {
    this.observed.delete(el);
  }
  fire(el: FakeNode): void {
    this.cb([{ target: el }]);
  }
}

interface FakeMORecord {
  type: string;
  target?: FakeNode;
  addedNodes?: FakeNode[];
  removedNodes?: FakeNode[];
}

class FakeMO {
  cb: (records: FakeMORecord[]) => void;
  constructor(cb: FakeMO["cb"]) {
    this.cb = cb;
    harness.mo = this;
  }
  observe(): void {}
  fire(records: FakeMORecord[]): void {
    this.cb(records);
  }
}

/* ---- fake OffscreenCanvas (drives the real sampleVideoFrame) ---- */

class FakeOffscreenCanvas {
  width: number;
  height: number;
  constructor(w: number, h: number) {
    this.width = w;
    this.height = h;
  }
  getContext(): { drawImage(src: FakeVideo): void } | null {
    return {
      drawImage(src: FakeVideo) {
        if (src.tainted) throw new DOMException("tainted", "SecurityError");
      },
    };
  }
  async convertToBlob(): Promise<{ arrayBuffer(): Promise<ArrayBuffer> }> {
    return { arrayBuffer: async () => new ArrayBuffer(8) };
  }
}

/* ---- harness ---- */

export interface SentMessage {
  msg: { type?: string; [k: string]: unknown };
  cb: (response: unknown) => void;
}

export interface Harness {
  clock: Clock;
  doc: FakeDocument;
  io: FakeIO;
  ro: FakeRO;
  mo: FakeMO;
  sent: SentMessage[];
  warns: unknown[][];
  shadowRoots: FakeFragment[];
  readonly stats: Record<string, number>;
  windowListeners: Map<string, Set<() => void>>;
  messageListener: ((m: unknown) => void) | null;
  advance(ms: number): void;
  flush(): Promise<void>;
  reply(type: string, response: unknown): boolean;
  pendingOf(type: string): SentMessage[];
  fireMessage(m: unknown): void;
  fireWindow(type: string): void;
  overlay(): FakeFragment | null;
  overlayMasks(): FakeElement[];
  overlayMaskCount(): number;
  overlayBadges(): FakeElement[];
  addImage(src: string): FakeImg;
  addVideo(src: string, opts?: { paused?: boolean }): FakeVideo;
  reset(): Promise<void>;
  uninstall(): void;
}

let harness: Harness;

const DEFAULT_CS: Record<string, string> = {
  objectFit: "fill",
  objectPosition: "center",
  borderLeftWidth: "0px",
  borderRightWidth: "0px",
  borderTopWidth: "0px",
  borderBottomWidth: "0px",
  paddingLeft: "0px",
  paddingRight: "0px",
  paddingTop: "0px",
  paddingBottom: "0px",
};

export function installHarness(): Harness {
  const clock = new Clock();
  doc = new FakeDocument();
  const sent: SentMessage[] = [];
  const warns: unknown[][] = [];
  const shadowRoots: FakeFragment[] = [];
  const windowListeners = new Map<string, Set<() => void>>();

  const fakeWindow = {
    addEventListener(type: string, fn: () => void): void {
      if (!windowListeners.has(type)) windowListeners.set(type, new Set());
      windowListeners.get(type)!.add(fn);
    },
  };

  const fakeChrome = {
    runtime: {
      sendMessage(msg: SentMessage["msg"], cb: (r: unknown) => void): void {
        sent.push({ msg, cb });
      },
      onMessage: {
        addListener(fn: (m: unknown) => void): void {
          harness.messageListener = fn;
        },
      },
      getURL: (path: string) => `chrome-extension://test/${path}`,
    },
  };

  const g = globalThis as Record<string, unknown>;
  const saved = new Map<string, unknown>();
  const set = (k: string, v: unknown): void => {
    if (!saved.has(k)) saved.set(k, g[k]);
    try {
      Object.defineProperty(g, k, { value: v, configurable: true, writable: true });
    } catch {
      // Non-configurable global: assignment throws in strict mode, so patch
      // the property on the object itself (performance.now) as a fallback.
      try {
        (g as Record<string, unknown>)[k] = v;
      } catch {
        if (k === "performance") {
          Object.defineProperty(performance, "now", {
            value: () => clock.now,
            configurable: true,
          });
        }
      }
    }
  };

  set("performance", { now: () => clock.now });
  set("setTimeout", clock.setTimeout);
  set("clearTimeout", clock.clearTimer);
  set("setInterval", clock.setInterval);
  set("clearInterval", clock.clearTimer);
  set("requestAnimationFrame", clock.requestAnimationFrame);
  set("cancelAnimationFrame", clock.cancelAnimationFrame);
  set("document", doc);
  set("window", fakeWindow);
  set("chrome", fakeChrome);
  set("Node", FakeNode);
  set("Element", FakeElement);
  set("HTMLElement", FakeElement);
  set("HTMLImageElement", FakeImg);
  set("HTMLVideoElement", FakeVideo);
  set("HTMLSourceElement", FakeSource);
  set("Document", FakeDocument);
  set("DocumentFragment", FakeFragment);
  set("IntersectionObserver", FakeIO);
  set("ResizeObserver", FakeRO);
  set("MutationObserver", FakeMO);
  set("OffscreenCanvas", FakeOffscreenCanvas);
  set("getComputedStyle", (el: FakeElement) => el._cs ?? DEFAULT_CS);

  const origWarn = console.warn;
  const origInfo = console.info;
  console.warn = (...a: unknown[]) => {
    warns.push(a);
  };
  console.info = () => {};

  harness = {
    clock,
    doc,
    io: undefined as unknown as FakeIO,
    ro: undefined as unknown as FakeRO,
    mo: undefined as unknown as FakeMO,
    sent,
    warns,
    shadowRoots,
    get stats() {
      return (globalThis as Record<string, unknown>).__faceblockStats as Record<string, number>;
    },
    windowListeners,
    messageListener: null,

    advance(ms: number): void {
      clock.advance(ms);
    },
    async flush(): Promise<void> {
      // Drain every currently-resolvable microtask chain.
      for (let i = 0; i < 3; i++) {
        const { promise, resolve } = Promise.withResolvers<void>();
        setImmediate(resolve);
        await promise;
      }
    },
    reply(type: string, response: unknown): boolean {
      const i = sent.findIndex((s) => s.msg.type === type);
      if (i < 0) return false;
      const s = sent.splice(i, 1)[0]!;
      s.cb(response);
      return true;
    },
    pendingOf(type: string): SentMessage[] {
      return sent.filter((s) => s.msg.type === type);
    },
    fireMessage(m: unknown): void {
      harness.messageListener?.(m);
    },
    fireWindow(type: string): void {
      for (const fn of [...(windowListeners.get(type) ?? [])]) fn();
    },
    overlay(): FakeFragment | null {
      return shadowRoots[0] ?? null;
    },
    overlayMasks(): FakeElement[] {
      const root = shadowRoots[0];
      if (!root) return [];
      return root.children.filter(
        (c): c is FakeElement => c instanceof FakeElement && c.textContent === "",
      );
    },
    overlayMaskCount(): number {
      const root = shadowRoots[0];
      if (!root) return 0;
      return root.children.filter(
        (c): c is FakeElement =>
          c instanceof FakeElement &&
          c.dataset.fbMask === "1" &&
          c.style.display !== "none",
      ).length;
    },
    overlayBadges(): FakeElement[] {
      const root = shadowRoots[0];
      if (!root) return [];
      return root.children.filter(
        (c): c is FakeElement => c instanceof FakeElement && c.textContent !== "",
      );
    },
    addImage(src: string): FakeImg {
      const img = new FakeImg();
      img.src = src;
      img.currentSrc = src;
      doc.body.appendChild(img);
      harness.mo.fire([{ type: "childList", addedNodes: [img], removedNodes: [] }]);
      harness.io.fire(img, true);
      return img;
    },
    addVideo(src: string, opts?: { paused?: boolean }): FakeVideo {
      const v = new FakeVideo();
      v.src = src;
      v.currentSrc = src;
      if (opts?.paused) v.paused = true;
      doc.body.appendChild(v);
      harness.mo.fire([{ type: "childList", addedNodes: [v], removedNodes: [] }]);
      harness.io.fire(v, true);
      return v;
    },
    async reset(): Promise<void> {
      // Disable first so nothing re-queues mid-teardown, then resolve every
      // outstanding reply so in-flight counters unwind before the sweep.
      harness.fireMessage({ target: "content", type: "STATE_CHANGED", enabled: false, earnEnabled: false, revision: 1 });
      while (sent.length) sent.splice(0, 1)[0]!.cb({ ok: false, error: "reset" });
      await harness.flush();
      // Detach everything, let the sweep untrack it.
      for (const c of [...doc.body.children]) c.remove();
      for (const c of [...doc.documentElement.children]) {
        if (c !== doc.body) c.remove();
      }
      clock.advance(5000);
      await harness.flush();
      // The overlay shadow root persists across tests (module-level host is
      // reused); only its children are removed by untrack/clearMasks.
      warns.length = 0;
      doc.visibilityState = "visible";
      const s = harness.stats;
      if (s) {
        for (const k of Object.keys(s)) {
          try {
            (s as Record<string, number>)[k] = 0;
          } catch {
            /* getter-only counters */
          }
        }
      }
      harness.fireMessage({ target: "content", type: "STATE_CHANGED", enabled: true, earnEnabled: false, revision: 1 });
    },
    uninstall(): void {
      for (const [k, v] of saved) {
        Object.defineProperty(g, k, { value: v, configurable: true, writable: true });
      }
      console.warn = origWarn;
      console.info = origInfo;
    },
  };
  return harness;
}
