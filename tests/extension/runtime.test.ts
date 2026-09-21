import { afterAll, beforeEach, describe, expect, test } from "bun:test";
import { installHarness } from "./harness.ts";

// The harness must be installed before content.ts evaluates: the module
// captures chrome.runtime, observers, timers and DOM globals at load time,
// so a static import cannot work — this dynamic import is the test seam.
const h = installHarness();
await import("../../extension/content.ts");
h.reply("GET_STATE", { ok: true, state: { enabled: true, revision: 1 } });
await h.flush();

afterAll(() => h.uninstall());
beforeEach(async () => {
  await h.reset();
});

const REGION = { x: 10, y: 10, width: 40, height: 40, confidence: 0.9, identityId: "id1" };
const REGION2 = { x: 100, y: 100, width: 40, height: 40, confidence: 0.9, identityId: "id1" };
const FRAME = (regions: object[]) => ({
  ok: true,
  result: { width: 480, height: 270, regions },
});
const IMAGE = (regions: object[]) => ({
  ok: true,
  result: { width: 400, height: 300, faceCount: regions.length, regions },
});

describe("video generation invalidation", () => {
  test("a reply captured before a seek is dropped, never applied", async () => {
    const v = h.addVideo("https://x/v.mp4");
    h.advance(1000);
    v.fireFrame();
    await h.flush();
    expect(h.pendingOf("ANALYZE_FRAME")).toHaveLength(1);

    // The seek lands while the frame is in flight: the reply describes pixels
    // that are no longer on screen.
    v.fire("seeking");
    h.reply("ANALYZE_FRAME", FRAME([REGION]));
    await h.flush();
    expect(h.overlayMasks()).toHaveLength(0);

    // The next presented frame re-samples and the fresh reply applies.
    v.fireFrame();
    await h.flush();
    h.reply("ANALYZE_FRAME", FRAME([REGION]));
    await h.flush();
    expect(h.overlayMasks()).toHaveLength(1);
  });

  test("a same-generation reply older than the age bound is dropped", async () => {
    const v = h.addVideo("https://x/v.mp4");
    h.advance(1000);
    v.fireFrame();
    await h.flush();
    expect(h.pendingOf("ANALYZE_FRAME")).toHaveLength(1);

    // The reply sat behind other work past the capture-age bound: ancient
    // pixels must not seed fresh-looking tracks.
    h.advance(3000);
    h.reply("ANALYZE_FRAME", FRAME([REGION]));
    await h.flush();
    expect(h.stats.videoStale).toBe(1);
    expect(h.overlayMasks()).toHaveLength(0);
  });

  test("in-flight accounting survives invalidation: sampling resumes", async () => {
    const v = h.addVideo("https://x/v.mp4");
    h.advance(1000);
    v.fireFrame();
    await h.flush();
    v.fire("seeking"); // invalidate while in flight
    h.reply("ANALYZE_FRAME", FRAME([REGION]));
    await h.flush();
    // inflight must have unwound: the next frame sends a new request.
    v.fireFrame();
    await h.flush();
    expect(h.pendingOf("ANALYZE_FRAME")).toHaveLength(1);
  });
});

describe("video resource latch", () => {
  test("a tainted sample cannot latch a new source", async () => {
    const v = h.addVideo("https://x/a.mp4");
    v.tainted = true;
    h.advance(1000);
    v.fireFrame(); // sample starts; drawImage will throw SecurityError
    // The source changes before the failed sample resolves.
    v.currentSrc = "https://x/b.mp4";
    v.src = "https://x/b.mp4";
    v.fire("loadstart");
    await h.flush();
    expect(h.stats.videoUnanalyzable).toBe(0);
    expect(h.overlayBadges()).toHaveLength(0);
  });

  test("tainted latch holds for the same resource, clears on a new one", async () => {
    const v = h.addVideo("https://x/a.mp4");
    v.tainted = true;
    h.advance(1000);
    v.fireFrame();
    await h.flush();
    expect(h.stats.videoUnanalyzable).toBe(1);
    expect(h.overlayBadges()).toHaveLength(1);
    expect(h.overlayBadges()[0]!.textContent).toContain("not analyzable");

    // A seek on the same resource must NOT clear the latch.
    v.fire("seeking");
    h.advance(5000);
    v.fireFrame();
    await h.flush();
    expect(h.pendingOf("ANALYZE_FRAME")).toHaveLength(0);

    // emptied -> new src -> loadstart (currentSrc still old) -> loadeddata
    // (new URL resolved): the exact sequence that must release the latch.
    v.fire("emptied");
    v.currentSrc = "https://x/b.mp4";
    v.src = "https://x/b.mp4";
    v.tainted = false;
    v.fire("loadstart");
    v.fire("loadeddata");
    h.advance(1000);
    v.fireFrame();
    await h.flush();
    expect(h.pendingOf("ANALYZE_FRAME")).toHaveLength(1);
    h.reply("ANALYZE_FRAME", FRAME([REGION]));
    await h.flush();
    expect(h.overlayMasks()).toHaveLength(1);
    expect(h.overlayBadges()).toHaveLength(0);
  });
});

describe("hidden tab", () => {
  test("a playing video does not coast across hidden time", async () => {
    const v = h.addVideo("https://x/v.mp4");
    h.advance(1000);
    v.fireFrame();
    await h.flush();
    h.reply("ANALYZE_FRAME", FRAME([REGION]));
    await h.flush();
    expect(h.overlayMasks()).toHaveLength(1);

    h.doc.setVisibility("hidden");
    h.advance(60000); // a minute hidden: no rAF, no sampling
    h.doc.setVisibility("visible");
    await h.flush();
    expect(h.overlayMasks()).toHaveLength(0);
  });

  test("a paused video keeps its frozen mask across hidden time", async () => {
    const v = h.addVideo("https://x/v.mp4", { paused: true });
    h.advance(1000);
    h.advance(100); // rAF-driven loop for paused video
    await h.flush();
    h.reply("ANALYZE_FRAME", FRAME([REGION]));
    await h.flush();
    expect(h.overlayMasks()).toHaveLength(1);

    h.doc.setVisibility("hidden");
    h.advance(60000);
    h.doc.setVisibility("visible");
    await h.flush();
    // Same frozen frame: the mask stays, rebased so hidden time is not coasted.
    expect(h.overlayMasks()).toHaveLength(1);
    expect(h.overlayMasks()[0]!.style.display).toBe("block");
  });
});

describe("video degraded coverage", () => {
  test("busy replies count and a streak surfaces a degraded badge", async () => {
    const v = h.addVideo("https://x/v.mp4");
    h.advance(1000);
    for (let i = 0; i < 3; i++) {
      v.fireFrame();
      await h.flush();
      h.reply("ANALYZE_FRAME", { ok: false, error: "busy" });
      await h.flush();
      h.advance(1000);
    }
    expect(h.stats.videoBusy).toBe(3);
    expect(h.overlayBadges()).toHaveLength(1);
    expect(h.overlayBadges()[0]!.textContent).toContain("degraded");

    // A fresh valid result clears the badge.
    v.fireFrame();
    await h.flush();
    h.reply("ANALYZE_FRAME", FRAME([REGION]));
    await h.flush();
    expect(h.overlayBadges()).toHaveLength(0);
    expect(h.overlayMasks()).toHaveLength(1);
  });

  test("tracks expired by the age bound flag degraded coverage", async () => {
    const v = h.addVideo("https://x/v.mp4");
    h.advance(1000);
    v.fireFrame();
    await h.flush();
    h.reply("ANALYZE_FRAME", FRAME([REGION]));
    await h.flush();
    expect(h.overlayMasks()).toHaveLength(1);

    // Stall the analyser: the pending reply never resolves, so no detection
    // rounds run and misses never accrue — only the age bound can fire.
    // Advance past the 250ms active cadence (still well under the 2s age
    // bound) so the next presented frame actually samples.
    h.advance(300);
    v.fireFrame();
    await h.flush();
    expect(h.pendingOf("ANALYZE_FRAME")).toHaveLength(1);
    h.advance(3100); // past MAX_TRACK_AGE_MS; realign prunes the expired track
    expect(h.overlayMasks()).toHaveLength(0);
    expect(h.overlayBadges()).toHaveLength(1);
  });
});

describe("video mask reconciliation", () => {
  test("two detections -> one -> zero masks, then drop after maxMisses", async () => {
    const v = h.addVideo("https://x/v.mp4");
    h.advance(1000);
    v.fireFrame();
    await h.flush();
    h.reply("ANALYZE_FRAME", FRAME([REGION, REGION2]));
    await h.flush();
    expect(h.overlayMasks()).toHaveLength(2);

    // One detection: the unmatched track accrues misses but keeps coasting —
    // masks follow tracks, so it takes maxMisses+1 single-region rounds to
    // shrink to one mask.
    for (let i = 0; i < 5; i++) {
      h.advance(1000);
      v.fireFrame();
      await h.flush();
      h.reply("ANALYZE_FRAME", FRAME([REGION]));
      await h.flush();
    }
    expect(h.overlayMasks()).toHaveLength(1);

    // Empty rounds accrue misses on the survivor; it drops after maxMisses.
    for (let i = 0; i < 5; i++) {
      h.advance(1000);
      v.fireFrame();
      await h.flush();
      h.reply("ANALYZE_FRAME", FRAME([]));
      await h.flush();
    }
    expect(h.overlayMasks()).toHaveLength(0);
  });
});

describe("image transient retry", () => {
  test("a failed analysis retries the unchanged URL with bounded backoff", async () => {
    h.addImage("https://x/a.jpg");
    await h.flush();
    expect(h.pendingOf("PROCESS_IMAGE")).toHaveLength(1);
    h.reply("PROCESS_IMAGE", { ok: false, error: "offscreen not ready" });
    await h.flush();
    expect(h.stats.errors).toBe(1);

    // Backoff: first retry lands at ~400ms, not before.
    h.advance(399);
    await h.flush();
    expect(h.pendingOf("PROCESS_IMAGE")).toHaveLength(0);
    h.advance(2);
    await h.flush();
    expect(h.pendingOf("PROCESS_IMAGE")).toHaveLength(1);
    h.reply("PROCESS_IMAGE", IMAGE([REGION]));
    await h.flush();
    expect(h.overlayMasks()).toHaveLength(1);
    expect(h.stats.imgRetried).toBe(1);
  });

  test("retries are bounded: a permanently failing URL stops", async () => {
    h.addImage("https://x/a.jpg");
    await h.flush();
    for (let i = 0; i < 10; i++) {
      const p = h.pendingOf("PROCESS_IMAGE");
      if (!p.length) break;
      h.reply("PROCESS_IMAGE", { ok: false, error: "down" });
      await h.flush();
      h.advance(5000);
      await h.flush();
    }
    // 1 initial + IMG_RETRY_MAX retries, then silence.
    expect(h.stats.errors).toBe(5);
    expect(h.stats.imgRetried).toBe(4);
    h.advance(60000);
    await h.flush();
    expect(h.pendingOf("PROCESS_IMAGE")).toHaveLength(0);
  });

  test("a malformed ok:true reply is retried like a transient failure", async () => {
    h.addImage("https://x/a.jpg");
    await h.flush();
    h.reply("PROCESS_IMAGE", { ok: true }); // no result payload
    await h.flush();
    expect(h.stats.errors).toBe(1);
    h.advance(500);
    await h.flush();
    expect(h.pendingOf("PROCESS_IMAGE")).toHaveLength(1);
  });

  test("a stale image reply after src change is dropped", async () => {
    const img = h.addImage("https://x/a.jpg");
    await h.flush();
    expect(h.pendingOf("PROCESS_IMAGE")).toHaveLength(1);
    // src mutation invalidates the in-flight request.
    img.src = "https://x/b.jpg";
    img.currentSrc = "https://x/b.jpg";
    h.mo.fire([{ type: "attributes", target: img }]);
    h.reply("PROCESS_IMAGE", IMAGE([REGION]));
    await h.flush();
    expect(h.stats.droppedStale).toBe(1);
    expect(h.overlayMasks()).toHaveLength(0);
    // The re-eval timer fires and the new URL is analysed.
    h.advance(100);
    await h.flush();
    expect(h.pendingOf("PROCESS_IMAGE")).toHaveLength(1);
    h.reply("PROCESS_IMAGE", IMAGE([REGION]));
    await h.flush();
    expect(h.overlayMasks()).toHaveLength(1);
  });
});

describe("image mask reconciliation", () => {
  test("two regions -> one -> zero removes stale surplus masks", async () => {
    const img = h.addImage("https://x/a.jpg");
    await h.flush();
    h.reply("PROCESS_IMAGE", IMAGE([REGION, REGION2]));
    await h.flush();
    expect(h.overlayMasks()).toHaveLength(2);

    img.src = "https://x/b.jpg";
    img.currentSrc = "https://x/b.jpg";
    h.mo.fire([{ type: "attributes", target: img }]);
    h.advance(100);
    await h.flush();
    h.reply("PROCESS_IMAGE", IMAGE([REGION]));
    await h.flush();
    expect(h.overlayMasks()).toHaveLength(1);

    img.src = "https://x/c.jpg";
    img.currentSrc = "https://x/c.jpg";
    h.mo.fire([{ type: "attributes", target: img }]);
    h.advance(100);
    await h.flush();
    h.reply("PROCESS_IMAGE", IMAGE([]));
    await h.flush();
    expect(h.overlayMasks()).toHaveLength(0);
  });

  test("responsive currentSrc change via resize invalidates through the same path", async () => {
    const img = h.addImage("https://x/small.jpg");
    await h.flush();
    h.reply("PROCESS_IMAGE", IMAGE([REGION]));
    await h.flush();
    expect(h.overlayMasks()).toHaveLength(1);

    // srcset picks a different candidate on resize: no attribute mutation,
    // no load event — only the ResizeObserver sees the element again.
    img.currentSrc = "https://x/large.jpg";
    h.ro.fire(img);
    await h.flush();
    expect(h.overlayMasks()).toHaveLength(0); // stale masks dropped immediately
    expect(h.pendingOf("PROCESS_IMAGE")).toHaveLength(1);
    h.reply("PROCESS_IMAGE", IMAGE([REGION2]));
    await h.flush();
    expect(h.overlayMasks()).toHaveLength(1);
  });
});
