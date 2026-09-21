import { describe, expect, test, beforeAll } from "bun:test";
import {
  MIN_SPONSOR_MASK_EDGE_PX,
  applyMaskStyle,
  createMaskShell,
  maskAppearanceMatches,
  qualifiesForSponsorMask,
  resolveMaskStyle,
  styleMaskForEarn,
} from "../../src/overlay/mask.ts";

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
  beforeAll(async () => {
    if (typeof document !== "undefined") return;
    const { Window } = await import("happy-dom");
    const window = new Window();
    globalThis.document = window.document as unknown as Document;
    globalThis.HTMLElement = window.HTMLElement as unknown as typeof HTMLElement;
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
