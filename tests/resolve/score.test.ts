import { describe, expect, test } from "bun:test";
import { scoreCandidate } from "../../src/resolve/score.ts";
import type { CandidateContext } from "../../src/resolve/types.ts";

const ALIASES = ["theo", "t3", "t3dotgg"];

/** Real contexts observed from Wikimedia, not invented examples. */
const REAL: Record<string, CandidateContext> = {
  officialPortrait: {
    filename: "Donald Trump official portrait.jpg",
    pageTitle: "Donald Trump",
    caption: "President Donald Trump poses for his official portrait at The White House",
  },
  jan2025Portrait: {
    filename: "January 2025 Official Presidential Portrait of Donald J. Trump.jpg",
    pageTitle: "Donald Trump",
    caption: "Official 2025 inaugural portrait of Donald Trump",
  },
  rnc2016: {
    filename: "Donald Trump RNC July 2016.jpg",
    pageTitle: "Wikimedia Commons",
    caption: "Trump promised to bring sweeping political change",
  },
  swearingIn: {
    filename: "Donald Trump swearing in ceremony.jpg",
    pageTitle: "Donald Trump",
    caption: "President Donald Trump",
  },
  sonIii: {
    filename: "Donald Trump III 170119-A-DR853-615.jpg",
    pageTitle: "Wikimedia Commons",
    caption: "Families and guests of President-elect Donald J. Trump",
  },
  impersonators: {
    filename: "Kim Jong-un and Donald Trump impersonators.jpg",
    pageTitle: "Wikimedia Commons",
    caption: "Howard X and Dennis Alan, impersonators",
  },
  signature: {
    filename: "Donald Trump signature.png",
    pageTitle: "Wikimedia Commons",
    caption: "Signature of Donald Trump",
  },
  coatOfArms: {
    filename: "Coat of Arms of Trump International Golf Club.svg",
    pageTitle: "Wikimedia Commons",
    caption: "",
  },
};

describe("scoreCandidate", () => {
  test("a curated portrait scores high", () => {
    expect(scoreCandidate("Donald Trump", [], REAL["officialPortrait"]!)).toBeGreaterThan(0.6);
  });

  test("a middle initial does not break the full-name match", () => {
    // "Donald J. Trump" must still register the contiguous name.
    expect(scoreCandidate("Donald Trump", [], REAL["jan2025Portrait"]!)).toBeGreaterThan(0.6);
  });

  test("varied real portraits all score high", () => {
    for (const key of ["rnc2016", "swearingIn"]) {
      expect(scoreCandidate("Donald Trump", [], REAL[key]!)).toBeGreaterThan(0.6);
    }
  });

  test("non-photo files are suppressed", () => {
    expect(scoreCandidate("Donald Trump", [], REAL["signature"]!)).toBeLessThan(0.15);
    expect(scoreCandidate("Donald Trump", [], REAL["coatOfArms"]!)).toBeLessThan(0.15);
  });

  test("look-alikes are suppressed even when the name matches", () => {
    expect(scoreCandidate("Donald Trump", [], REAL["impersonators"]!)).toBeLessThan(0.15);
  });

  test("a same-name relative is suppressed", () => {
    expect(scoreCandidate("Donald Trump", [], REAL["sonIii"]!)).toBeLessThan(0.15);
  });

  test("a better candidate outranks a suppressed one", () => {
    const good = scoreCandidate("Donald Trump", [], REAL["officialPortrait"]!);
    const bad = scoreCandidate("Donald Trump", [], REAL["impersonators"]!);
    expect(good).toBeGreaterThan(bad);
  });

  test("a solo portrait outranks a co-subject photo", () => {
    // Both name-match fully, so only the composition evidence separates them.
    const solo = scoreCandidate("Donald Trump", [], REAL["officialPortrait"]!);
    const shared = scoreCandidate("Donald Trump", [], {
      filename: "Donald Trump and Bill Clinton.jpg",
      pageTitle: "Wikimedia Commons",
      caption: "",
    });
    expect(shared).toBeLessThan(solo);
    expect(shared).toBeLessThan(0.6);
  });

  test("co-subject penalises either ordering of the names", () => {
    const afterName = scoreCandidate("Donald Trump", [], {
      filename: "Donald Trump with Melania Trump.jpg",
      pageTitle: "Wikimedia Commons",
      caption: "",
    });
    const beforeName = scoreCandidate("Donald Trump", [], {
      filename: "Bill Clinton and Donald Trump at the U.S. Open in 2000.jpg",
      pageTitle: "Wikimedia Commons",
      caption: "",
    });
    expect(afterName).toBeLessThan(0.6);
    expect(beforeName).toBeLessThan(0.6);
  });

  test("engravings are not treated as face photos", () => {
    expect(
      scoreCandidate("Ada Lovelace", [], {
        filename: "Engraved portrait of Ada Lovelace.png",
        pageTitle: "Wikimedia Commons",
        caption: "Engraving of Ada Lovelace",
      }),
    ).toBeLessThan(0.15);
  });

  test("a composition word breaks ties between equally named files", () => {
    const withPortrait = scoreCandidate("Donald Trump", [], REAL["officialPortrait"]!);
    const actionShot = scoreCandidate("Donald Trump", [], REAL["rnc2016"]!);
    expect(withPortrait).toBeGreaterThan(actionShot);
  });

  test("the composition bonus cannot rescue a suppressed file", () => {
    // "portrait" would otherwise add the bonus, but the look-alike factor dominates.
    expect(
      scoreCandidate("Donald Trump", [], {
        filename: "Donald Trump impersonator portrait.jpg",
        pageTitle: "Wikimedia Commons",
        caption: "",
      }),
    ).toBeLessThan(0.15);
  });

  test("aliases count when the full name does not appear", () => {
    const ctx: CandidateContext = {
      filename: "t3dotgg stream screenshot.jpg",
      pageTitle: "Twitch",
      caption: "theo talking about typescript",
    };
    expect(scoreCandidate("Theo Browne", ALIASES, ctx)).toBeGreaterThan(0.1);
  });

  test("empty context scores zero", () => {
    expect(scoreCandidate("Donald Trump", [], { filename: "", pageTitle: "", caption: "" })).toBe(0);
  });

  test("empty name scores zero", () => {
    expect(scoreCandidate("", [], REAL["officialPortrait"]!)).toBe(0);
  });

  test("score is always a finite value in 0..1", () => {
    const contexts: CandidateContext[] = [
      ...Object.values(REAL),
      { filename: "!!!!", pageTitle: "---", caption: "?????" },
      { filename: "trump", pageTitle: "trump", caption: "trump trump trump" },
    ];
    for (const ctx of contexts) {
      const score = scoreCandidate("Donald Trump", [], ctx);
      expect(Number.isFinite(score)).toBe(true);
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    }
  });

  test("an unrelated person scores low", () => {
    expect(
      scoreCandidate("Donald Trump", [], {
        filename: "Ada Lovelace portrait.jpg",
        pageTitle: "Ada Lovelace",
        caption: "Mathematician Ada Lovelace",
      }),
    ).toBeLessThan(0.3);
  });
});
