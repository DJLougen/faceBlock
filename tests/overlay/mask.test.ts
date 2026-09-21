import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import {
  MIN_SPONSOR_MASK_EDGE_PX,
  applyMaskStyle,
  createMaskShell,
  maskAppearanceMatches,
  qualifiesForSponsorMask,
  resolveMaskStyle,
  styleMaskForEarn,
} from "../../src/overlay/mask.ts";

class StubEl {
  style = new Proxy({} as CSSStyleDeclaration, {
    get(_target, prop) {
      if (typeof prop !== "string") return undefined;
      if (prop === "cssText") {
        return Object.entries(this.fields).map(([key, value]) => `${key}:${value}`).join(";");
      }
      return this.fields[prop] ?? "";
    },
    set(_target, prop, value) {
      if (typeof prop !== "string") return false;
      if (prop === "cssText") {
        for (const key of Object.keys(this.fields)) delete this.fields[key];
        for (const part of String(value).split(";")) {
          const splitAt = part.indexOf(":");
          if (splitAt < 0) continue;
          const raw = part.slice(0, splitAt).trim();
          const parsed = part.slice(splitAt + 1).trim();
          if (raw.length === 0) continue;
          this.fields[raw.replace(/-([a-z])/g, (_, c: string) => c.toUpperCase())] = parsed;
        }
        return true;
      }
      this.fields[prop] = String(value);
      return true;
    },
    fields: {},
  } as ProxyHandler<CSSStyleDeclaration> & { fields: Record<string, string> });
  dataset: Record<string, string | undefined> = {};
  children: StubEl[] = [];
  childElementCount = 0;
  textContent = "";
  setAttribute(): void {}
  replaceChildren(...nodes: StubEl[]): void {
    this.children = nodes;
    this.childElementCount = nodes.length;
  }
  append(...nodes: StubEl[]): void {
    this.children.push(...nodes);
    this.childElementCount = this.children.length;
  }
}

describe("mask earn gating", () => {
  test("sponsor requires opt-in and minimum edge length", () => {
    expect(resolveMaskStyle(false, 200, 200)).toBe("black");
    expect(resolveMaskStyle(true, MIN_SPONSOR_MASK_EDGE_PX - 1, 200)).toBe("black");
    expect(resolveMaskStyle(true, 200, MIN_SPONSOR_MASK_EDGE_PX - 1)).toBe("black");
    expect(resolveMaskStyle(true, MIN_SPONSOR_MASK_EDGE_PX, MIN_SPONSOR_MASK_EDGE_PX)).toBe(
      "sponsor",
    );
  });

  test("qualifiesForSponsorMask is symmetric on width and height", () => {
    expect(qualifiesForSponsorMask(119, 500)).toBe(false);
    expect(qualifiesForSponsorMask(500, 119)).toBe(false);
    expect(qualifiesForSponsorMask(120, 120)).toBe(true);
  });
});

describe("styleMaskForEarn appearance cache", () => {
  const previousDocument = globalThis.document;
  beforeAll(() => {
    // Frozen CI has no DOM library. These tests only need createElement, cssText,
    // dataset, and stable child identity.
    globalThis.document = {
      createElement: () => new StubEl(),
    } as unknown as Document;
  });
  afterAll(() => {
    globalThis.document = previousDocument;
  });

  test("createMaskShell applies non-interactive shell styles", () => {
    const mask = createMaskShell();
    expect(mask.style.pointerEvents).toBe("none");
    expect(mask.style.position).toBe("absolute");
    expect(mask.dataset.fbMask).toBe("1");
    expect(mask.dataset.fbMaskStyle).toBeUndefined();
  });

  test("repeated same-style calls preserve child node identity", () => {
    const mask = createMaskShell();
    const box = { width: 200, height: 200 };
    const url = "chrome-extension://test/sponsors/placeholder.svg";

    styleMaskForEarn(mask, true, box, url);
    expect(mask.dataset.fbMaskStyle).toBe("sponsor");
    expect(mask.childElementCount).toBe(2);
    const creative = mask.children[0]!;
    const label = mask.children[1]!;

    for (let i = 0; i < 24; i++) {
      mask.style.left = `${10 + i}px`;
      mask.style.top = `${20 + i}px`;
      mask.style.width = `${box.width + i}px`;
      mask.style.height = `${box.height}px`;
      styleMaskForEarn(mask, true, box, url);
      expect(mask.children[0]).toBe(creative);
      expect(mask.children[1]).toBe(label);
      expect(mask.childElementCount).toBe(2);
    }
  });

  test("applyMaskStyle does not replaceChildren on cache hit", () => {
    const mask = createMaskShell();
    const url = "chrome-extension://test/sponsors/placeholder.svg";
    let replaceCalls = 0;
    const orig = mask.replaceChildren.bind(mask);
    mask.replaceChildren = (...nodes: Parameters<typeof orig>) => {
      replaceCalls++;
      return orig(...nodes);
    };

    applyMaskStyle(mask, "sponsor", url);
    expect(replaceCalls).toBe(1);
    expect(mask.childElementCount).toBe(2);
    const creative = mask.children[0]!;

    for (let i = 0; i < 24; i++) {
      mask.style.left = `${i}px`;
      applyMaskStyle(mask, "sponsor", url);
      expect(mask.children[0]).toBe(creative);
    }
    expect(replaceCalls).toBe(1);
    expect(maskAppearanceMatches(mask, "sponsor", url)).toBe(true);
  });

  test("repeated black-style calls do not rebuild children", () => {
    const mask = createMaskShell();
    const box = { width: 80, height: 80 };

    styleMaskForEarn(mask, false, box, "unused");
    expect(mask.dataset.fbMaskStyle).toBe("black");
    expect(mask.childElementCount).toBe(0);

    for (let i = 0; i < 24; i++) {
      mask.style.left = `${i}px`;
      styleMaskForEarn(mask, false, box, "unused");
      expect(mask.childElementCount).toBe(0);
      expect(mask.dataset.fbMaskStyle).toBe("black");
    }
  });

  test("black↔sponsor transition rebuilds appearance", () => {
    const mask = createMaskShell();
    const large = { width: 200, height: 200 };
    const small = { width: 80, height: 80 };
    const url = "chrome-extension://test/sponsors/placeholder.svg";

    styleMaskForEarn(mask, true, large, url);
    expect(mask.dataset.fbMaskStyle).toBe("sponsor");
    expect(mask.childElementCount).toBe(2);

    styleMaskForEarn(mask, true, small, url);
    expect(mask.dataset.fbMaskStyle).toBe("black");
    expect(mask.childElementCount).toBe(0);

    styleMaskForEarn(mask, true, large, url);
    expect(mask.dataset.fbMaskStyle).toBe("sponsor");
    expect(mask.childElementCount).toBe(2);
  });
});
