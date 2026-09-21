/* Options-page regression tests.
 *
 * options.ts is a DOM script with no exports, so these tests drive the real
 * module against a minimal DOM stub: elements keep children/listeners, clicks
 * bubble, and a <label> forwards a click to its wrapped control exactly like a
 * browser (skipped when the click already landed on the control, so a tile
 * click can never double-toggle). The chrome stub records every message sent
 * to the background and answers through a per-test handler.
 */

import { afterAll, describe, expect, test } from "bun:test";
import type { BlockList, EnrollPreview, EnrollPreviewFace, SavedIdentity } from "../../extension/protocol.ts";

/* ---- DOM stub ---- */

interface StubEvent {
  type: string;
  target: StubEl;
  key?: string;
  preventDefault(): void;
}

class StubEl {
  tagName: string;
  children: StubEl[] = [];
  parent: StubEl | null = null;
  listeners = new Map<string, Array<(event: StubEvent) => void>>();
  textContent = "";
  hidden = false;
  disabled = false;
  checked = false;
  value = "";
  type = "";
  className = "";
  tabIndex = -1;
  src = "";
  alt = "";
  title = "";
  referrerPolicy = "";
  loading = "";
  classList = {
    add: (...names: string[]) => {
      for (const name of names) this.classList._set.add(name);
      this.classList._sync();
    },
    remove: (...names: string[]) => {
      for (const name of names) this.classList._set.delete(name);
      this.classList._sync();
    },
    contains: (name: string) => this.classList._set.has(name),
    _set: new Set<string>(),
    _sync: () => {
      this.className = [...this.classList._set].join(" ");
    },
  };

  constructor(tag: string) {
    this.tagName = tag.toUpperCase();
  }

  append(...nodes: StubEl[]): void {
    for (const node of nodes) {
      node.parent = this;
      this.children.push(node);
    }
  }

  replaceChildren(...nodes: StubEl[]): void {
    for (const child of this.children) child.parent = null;
    this.children = [];
    this.append(...nodes);
  }

  remove(): void {
    if (!this.parent) return;
    const siblings = this.parent.children;
    const index = siblings.indexOf(this);
    if (index >= 0) siblings.splice(index, 1);
    this.parent = null;
  }

  addEventListener(type: string, listener: (event: StubEvent) => void): void {
    const list = this.listeners.get(type) ?? [];
    list.push(listener);
    this.listeners.set(type, list);
  }

  dispatch(type: string, init: { key?: string } = {}): void {
    // Disabled controls do not fire click handlers in a real browser.
    if (type === "click" && this.disabled) return;
    // Activating a checkbox flips its state before listeners run.
    if (type === "click" && this.tagName === "INPUT" && this.type === "checkbox") {
      this.checked = !this.checked;
    }
    const event: StubEvent = { type, target: this, key: init.key, preventDefault() {} };
    let node: StubEl | null = this;
    while (node) {
      for (const listener of node.listeners.get(type) ?? []) listener(event);
      node = node.parent;
    }
    if (type === "click") {
      // Label activation: a click anywhere inside a <label> that did not land
      // on the control itself is forwarded to the control — exactly once.
      let label: StubEl | null = this.tagName === "LABEL" ? this : this.parent;
      while (label && label.tagName !== "LABEL") label = label.parent;
      if (label) {
        const control = firstDescendant(
          label,
          (el) => el.tagName === "INPUT" && el.type === "checkbox",
        );
        if (control && control !== this) control.dispatch("click");
      }
    }
  }

  click(): void {
    this.dispatch("click");
  }
}

function firstDescendant(root: StubEl, pred: (el: StubEl) => boolean): StubEl | null {
  for (const child of root.children) {
    if (pred(child)) return child;
    const found = firstDescendant(child, pred);
    if (found) return found;
  }
  return null;
}

function walk(root: StubEl, pred: (el: StubEl) => boolean, out: StubEl[] = []): StubEl[] {
  if (pred(root)) out.push(root);
  for (const child of root.children) walk(child, pred, out);
  return out;
}



type Handler = (message: Record<string, unknown>) => Record<string, unknown>;

const sent: Record<string, unknown>[] = [];
let handler: Handler = () => ({ ok: true });
let storageListener:
  | ((changes: Record<string, { newValue?: unknown }>, area: string) => void)
  | null = null;

const byId = new Map<string, StubEl>();
function el(id: string): StubEl {
  const found = byId.get(id);
  if (!found) throw new Error(`stub missing #${id}`);
  return found;
}

for (const [id, tag] of [
  ["enabled-toggle", "input"],
  ["enabled-label", "span"],
  ["block-form", "form"],
  ["name-input", "input"],
  ["block-btn", "button"],
  ["progress", "p"],
  ["error", "p"],
  ["identity-list", "ul"],
  ["empty-note", "p"],
  ["preview", "section"],
  ["getting-started", "section"],
] as const) {
  byId.set(id, new StubEl(tag));
}
el("enabled-toggle").type = "checkbox";
el("name-input").type = "text";
el("block-btn").type = "submit";

(globalThis as Record<string, unknown>).document = {
  body: new StubEl("body"),
  getElementById: (id: string) => byId.get(id) ?? null,
  createElement: (tag: string) => new StubEl(tag),
};
(globalThis as Record<string, unknown>).location = { search: "" };
(globalThis as Record<string, unknown>).chrome = {
  runtime: {
    lastError: undefined,
    sendMessage: (message: unknown) => {
      const record = message as Record<string, unknown>;
      sent.push(record);
      return Promise.resolve(handler(record));
    },
  },
  storage: {
    onChanged: {
      addListener: (
        callback: (changes: Record<string, { newValue?: unknown }>, area: string) => void,
      ) => {
        storageListener = callback;
      },
    },
  },
};

/* ---- fixtures ---- */

function face(filename: string, score = 0.9): EnrollPreviewFace {
  return {
    url: `https://upload.wikimedia.org/${filename}`,
    filename,
    source: "wikimedia",
    score,
    embedding: [0.1, 0.2],
  };
}

// A face from the extension's bundled reference directory — the curated path
// that used to enroll silently via BLOCK_NAME.
function curatedFace(filename: string, score = 0.95): EnrollPreviewFace {
  return {
    url: `chrome-extension://faceblock/references/${filename}`,
    filename,
    source: "bundled",
    score,
    embedding: [0.1, 0.2],
  };
}

function preview(name: string, kept: EnrollPreviewFace[], identityId?: string): EnrollPreview {
  return {
    name,
    candidatesTried: 12,
    facesFound: kept.length,
    kept,
    rejected: [{ url: "https://example.com/other.jpg", reason: "no face found" }],
    ...(identityId ? { identityId } : {}),
  };
}

function identity(id: string, name: string): SavedIdentity {
  return {
    id,
    name,
    embeddings: [[0.1, 0.2]],
    threshold: 0.4,
    sources: ["https://commons.wikimedia.org/wiki/File:Ada.jpg?x=1"],
    createdAt: 1_700_000_000_000,
  };
}

function blockList(identities: SavedIdentity[], revision: number): BlockList {
  return { identities, enabled: true, revision };
}

function injectState(next: BlockList): void {
  storageListener?.({ faceblockState: { newValue: next } }, "local");
}

function sentOf(type: string): Record<string, unknown>[] {
  return sent.filter((message) => message.type === type);
}

function previewButtons(): StubEl[] {
  return walk(el("preview"), (node) => node.tagName === "BUTTON");
}

function buttonByText(text: string): StubEl | undefined {
  return previewButtons().find((node) => node.textContent === text);
}

async function flush(): Promise<void> {
  // The stubs resolve every message synchronously, so all pending work is
  // microtasks. Yielding repeatedly drains the chain deterministically — no
  // wall-clock timer.
  for (let i = 0; i < 20; i += 1) await Promise.resolve();
}

/* ---- drive the real module ---- */

// Dynamic import is required: options.ts touches document/chrome at module
// load, so it can only be evaluated after the stubs above are installed.
await import("../../extension/options.ts");
await flush(); // settle the module's top-level GET_STATE before tests run

/* ---- tests ---- */

describe("options enrollment flow", () => {
  test("curated/bundled references still go through RESOLVE_PREVIEW and nothing persists before Confirm", async () => {
    sent.length = 0;
    handler = (message) => {
      if (message.type === "RESOLVE_PREVIEW") {
        return { ok: true, preview: preview("Ada Lovelace", [curatedFace("ada-1.jpg"), curatedFace("ada-2.jpg")]) };
      }
      return { ok: true };
    };

    el("name-input").value = "Ada Lovelace";
    el("block-form").dispatch("submit");
    await flush();

    // Universal preview path: no BLOCK_NAME shortcut, no silent persistence.
    const resolves = sentOf("RESOLVE_PREVIEW");
    expect(resolves.length).toBe(1);
    expect(resolves[0]!.name).toBe("Ada Lovelace");
    expect("identityId" in resolves[0]!).toBe(false);
    expect(sentOf("BLOCK_NAME").length).toBe(0);
    expect(sentOf("CONFIRM_ENROLL").length).toBe(0);

    // Preview is shown with one tile per kept face, each a label wrapping its
    // checkbox so the caption is the checkbox's associated label.
    const previewEl = el("preview");
    expect(previewEl.hidden).toBe(false);
    const tiles = walk(previewEl, (node) => node.className === "face-tile");
    expect(tiles.length).toBe(2);
    for (const tile of tiles) {
      expect(tile.tagName).toBe("LABEL");
      const box = firstDescendant(
        tile,
        (node) => node.tagName === "INPUT" && node.type === "checkbox",
      );
      expect(box).not.toBeNull();
      expect(box!.checked).toBe(true);
    }
    expect(buttonByText("Yes, block Ada")).toBeDefined();
    expect(buttonByText("Cancel")).toBeDefined();

    // Confirm persists exactly the checked faces.
    buttonByText("Yes, block Ada")!.click();
    await flush();
    const confirms = sentOf("CONFIRM_ENROLL");
    expect(confirms.length).toBe(1);
    expect(confirms[0]!.name).toBe("Ada Lovelace");
    expect((confirms[0]!.faces as EnrollPreviewFace[]).length).toBe(2);
    expect("identityId" in confirms[0]!).toBe(false);
    expect(el("preview").hidden).toBe(true);
  });

  test("confirming with every face unticked keeps the preview open and sends nothing", async () => {
    sent.length = 0;
    handler = (message) => {
      if (message.type === "RESOLVE_PREVIEW") {
        return { ok: true, preview: preview("Grace Hopper", [face("grace-1.jpg")]) };
      }
      return { ok: true };
    };

    el("name-input").value = "Grace Hopper";
    el("block-form").dispatch("submit");
    await flush();

    const tile = walk(el("preview"), (node) => node.className === "face-tile")[0]!;
    const box = firstDescendant(
      tile,
      (node) => node.tagName === "INPUT" && node.type === "checkbox",
    )!;
    box.click(); // untick the only face
    expect(box.checked).toBe(false);

    buttonByText("Yes, block Grace")!.click();
    await flush();

    expect(sentOf("CONFIRM_ENROLL").length).toBe(0);
    expect(el("preview").hidden).toBe(false); // still open for correction
    expect(el("error").hidden).toBe(false);
    expect(el("error").textContent).toContain("at least one photo");

    // Clean up: cancel the still-open preview.
    buttonByText("Cancel")!.click();
    await flush();
    expect(el("preview").hidden).toBe(true);
  });

  test("cancel preserves the saved blocklist and sends no mutation", async () => {
    const ada = identity("ada", "Ada Lovelace");
    injectState(blockList([ada], 1));
    sent.length = 0;
    handler = (message) => {
      if (message.type === "RESOLVE_PREVIEW") {
        return { ok: true, preview: preview("Alan Turing", [face("alan-1.jpg")]) };
      }
      return { ok: true };
    };

    el("name-input").value = "Alan Turing";
    el("block-form").dispatch("submit");
    await flush();
    expect(el("preview").hidden).toBe(false);

    buttonByText("Cancel")!.click();
    await flush();

    expect(el("preview").hidden).toBe(true);
    expect(sentOf("CONFIRM_ENROLL").length).toBe(0);
    expect(sentOf("REMOVE").length).toBe(0);
    // The existing identity is still rendered — cancel must not drop state.
    const names = walk(el("identity-list"), (node) => node.tagName === "STRONG").map(
      (node) => node.textContent,
    );
    expect(names).toEqual(["Ada Lovelace"]);
  });

  test("re-typing a saved name routes to a refresh preview carrying identityId", async () => {
    const ada = identity("ada", "Ada Lovelace");
    injectState(blockList([ada], 2));
    sent.length = 0;
    handler = (message) => {
      if (message.type === "RESOLVE_PREVIEW") {
        // Backend echoes the identity this preview would update.
        return {
          ok: true,
          preview: preview("Ada Lovelace", [face("ada-new.jpg")], "ada"),
        };
      }
      if (message.type === "CONFIRM_ENROLL") {
        return { ok: true, state: blockList([ada], 3), identity: ada };
      }
      return { ok: true };
    };

    el("name-input").value = "ada lovelace"; // different case, same person
    el("block-form").dispatch("submit");
    await flush();

    const resolves = sentOf("RESOLVE_PREVIEW");
    expect(resolves.length).toBe(1);
    expect(resolves[0]!.identityId).toBe("ada");
    expect(sentOf("BLOCK_NAME").length).toBe(0);
    expect(sentOf("CONFIRM_ENROLL").length).toBe(0); // preview only, no silent write

    expect(buttonByText("Yes, update Ada")).toBeDefined();
    buttonByText("Yes, update Ada")!.click();
    await flush();

    const confirms = sentOf("CONFIRM_ENROLL");
    expect(confirms.length).toBe(1);
    expect(confirms[0]!.identityId).toBe("ada");
  });

  test("Refresh reference photos on a saved identity previews and persists only on confirm", async () => {
    const ada = identity("ada", "Ada Lovelace");
    injectState(blockList([ada], 4));
    sent.length = 0;
    handler = (message) => {
      if (message.type === "RESOLVE_PREVIEW") {
        // No identityId echoed: the requested id must still be carried through.
        return { ok: true, preview: preview("Ada Lovelace", [face("ada-r1.jpg"), face("ada-r2.jpg")]) };
      }
      if (message.type === "CONFIRM_ENROLL") {
        return { ok: true, state: blockList([ada], 5), identity: ada };
      }
      return { ok: true };
    };

    const refresh = walk(
      el("identity-list"),
      (node) => node.tagName === "BUTTON" && node.textContent === "Refresh reference photos",
    )[0];
    expect(refresh).toBeDefined();
    refresh!.click();
    await flush();

    const resolves = sentOf("RESOLVE_PREVIEW");
    expect(resolves.length).toBe(1);
    expect(resolves[0]!.name).toBe("Ada Lovelace");
    expect(resolves[0]!.identityId).toBe("ada");
    expect(sentOf("CONFIRM_ENROLL").length).toBe(0); // nothing persisted yet

    const confirm = buttonByText("Yes, update Ada")!;
    confirm.click();
    confirm.click(); // second click while in flight must not duplicate the write
    await flush();

    const confirms = sentOf("CONFIRM_ENROLL");
    expect(confirms.length).toBe(1);
    expect(confirms[0]!.identityId).toBe("ada");
    expect((confirms[0]!.faces as EnrollPreviewFace[]).length).toBe(2);
  });

  test("face tiles toggle once via tile click, caption click, and direct checkbox click", async () => {
    sent.length = 0;
    handler = (message) => {
      if (message.type === "RESOLVE_PREVIEW") {
        return { ok: true, preview: preview("Katherine Johnson", [face("kj-1.jpg")]) };
      }
      return { ok: true };
    };

    el("name-input").value = "Katherine Johnson";
    el("block-form").dispatch("submit");
    await flush();

    const tile = walk(el("preview"), (node) => node.className === "face-tile")[0]!;
    const box = firstDescendant(
      tile,
      (node) => node.tagName === "INPUT" && node.type === "checkbox",
    )!;

    tile.click(); // whole-tile activation through the label
    expect(box.checked).toBe(false);
    tile.click();
    expect(box.checked).toBe(true);

    const caption = firstDescendant(tile, (node) => node.className === "face-caption")!;
    caption.click(); // clicking the label text toggles once, not twice
    expect(box.checked).toBe(false);

    box.click(); // clicking the control itself toggles once — no double toggle
    expect(box.checked).toBe(true);

    buttonByText("Cancel")!.click();
    await flush();
  });
});

// Restore the globals this file stubbed so other test files in the same
// `bun test` process never see UI mocks.
afterAll(() => {
  const g = globalThis as Record<string, unknown>;
  delete g.document;
  delete g.location;
  delete g.chrome;
});
