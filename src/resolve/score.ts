/**
 * Context scoring for discovered candidate photos.
 *
 * This is the plan's "contextual validation" layer (§12) and the reason a
 * filename is useful evidence: Wikimedia file names are descriptive
 * ("Donald_Trump_official_portrait.jpg"), so the text around an image tells us
 * whether it is even plausibly the person BEFORE we spend inference on it.
 *
 * Scores are operating points, not calibrated probabilities — they only order
 * candidates. The clustering step and the user's confirmation decide what is
 * actually enrolled.
 */

import type { CandidateContext } from "./types.ts";

/**
 * Files that are essentially never a usable face photo of the subject.
 * Multiplicative (near-total) suppression: a match here means the entity name
 * appears in a caption or a category instead of depicting a face.
 */
const NON_PHOTO_TERMS = [
  "signature",
  "autograph",
  "logo",
  "seal",
  "coat of arms",
  "flag",
  "map",
  "chart",
  "diagram",
  "schematic",
  "poster",
  "book cover",
  "album",
  "soundtrack",
  "manuscript",
  "banknote",
  "stamp",
  "coin",
  "grave",
  "tomb",
  "headstone",
  "memorial",
  "plaque",
  "fleuron",
  "ornament",
  "blue plaque",
  "caricature",
  "cartoon",
  "drawing",
  "sketch",
  "illustration",
  // Pre-photographic likenesses: an engraving is not a face photo.
  "engraving",
  "engraved",
  "lithograph",
  "etching",
  "woodcut",
  "aquatint",
] as const;

/**
 * Look-alikes: the name matches but the pixels are not the person. Observed in
 * the wild — a Commons search for a public figure returns impersonator photos.
 */
const LOOKALIKE_TERMS = [
  "impersonator",
  "impersonators",
  "impersonation",
  "lookalike",
  "look-alike",
  "look alike",
  "wax figure",
  "waxwork",
  "wax museum",
  "tussauds",
  "statue",
  "mannequin",
  "dummy",
  "puppet",
  "cosplay",
  "mask",
] as const;

/**
 * Fabricated likenesses. An AI-generated image is not a photograph of the
 * person, so enrolling it would teach the matcher a face that was never theirs.
 * Observed in the wild: a Commons search for a public figure returned an
 * AI-generated meme whose file name contained the correct full name.
 */
const FABRICATED_TERMS = [
  "ai generated",
  "midjourney",
  "stable diffusion",
  "dall e",
  "deepfake",
  "cgi render",
  "generated image",
] as const;

/** Group shots: the person is present but the face is small or off-centre. */
const CROWD_TERMS = [
  "family",
  "crowd",
  "audience",
  "delegation",
  "group photo",
  "cabinet meeting",
  "with wife",
  "with husband",
  " and wife",
  " and husband",
] as const;

/**
 * A generational suffix directly after the matched name means a DIFFERENT
 * person who shares the name (e.g. "Donald Trump III" is the son).
 */
const GENERATIONAL_SUFFIXES: Record<string, true> = {
  jr: true,
  sr: true,
  ii: true,
  iii: true,
  iv: true,
};

const NON_PHOTO_FACTOR = 0.1;
const LOOKALIKE_FACTOR = 0.05;
const FABRICATED_FACTOR = 0.05;
const CROWD_FACTOR = 0.6;
const SUFFIX_FACTOR = 0.1;
/** A photo shared with another named person is a poor single-face reference. */
const CO_SUBJECT_FACTOR = 0.5;
/** Tie-breaker: a solo portrait is a better reference than an action shot. */
const PREFERENCE_BONUS = 0.05;

const CO_SUBJECT_JOINERS: Record<string, true> = { and: true, with: true };

/** Composition words that indicate a usable, face-forward reference. */
const PREFERRED_TERMS = ["portrait", "headshot", "closeup", "close up", "official", "photo"] as const;

/** Lowercase, split file-name punctuation into spaces, collapse whitespace. */
function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/[_\-+()[\],./]/g, " ")
    .replace(/[^a-z0-9 ]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Drop standalone single letters so middle initials stop breaking a contiguous
 * full-name match ("Donald J. Trump" must still match "Donald Trump").
 */
function stripInitials(normalized: string): string {
  return normalized.replace(/\b[a-z]\b/g, " ").replace(/\s+/g, " ").trim();
}

function tokensOf(normalized: string): string[] {
  return normalized.split(" ").filter((t) => t.length >= 2);
}

/** Word-boundary-ish search; the haystack is already normalized. */
function hasTerm(haystack: string, term: string): boolean {
  const at = haystack.indexOf(term);
  if (at === -1) return false;
  const before = at === 0 ? " " : haystack[at - 1]!;
  const after = haystack[at + term.length] ?? " ";
  return !/[a-z0-9]/.test(before) && !/[a-z0-9]/.test(after);
}

function clamp01(n: number): number {
  if (!Number.isFinite(n)) return 0;
  return n < 0 ? 0 : n > 1 ? 1 : n;
}

/**
 * Score how likely an image with this surrounding text depicts `name`.
 * Returns 0..1; never NaN.
 *
 * Positive evidence builds up: name-token coverage, a contiguous full-name
 * match, the name appearing in the FILE NAME specifically, and alias hits.
 * Negative evidence then multiplies the score down, because a single
 * "signature" or "impersonators" marker should outweigh a name match.
 */
export function scoreCandidate(
  name: string,
  aliases: readonly string[],
  ctx: CandidateContext,
): number {
  // Evidence is the FILE NAME and the caption. The page title is deliberately
  // NOT scored: an article titled "Ada Lovelace" links many images that are not
  // her (a book icon, her husband, a sonnet), and scoring the title promoted
  // all of them. pageTitle stays available for display only.
  const fileHaystack = stripInitials(normalize(ctx.filename));
  const haystack = normalize(`${ctx.filename} ${ctx.caption}`);
  if (haystack === "") return 0;

  // Contiguous matching runs on an initials-stripped copy of the text.
  const contigHaystack = stripInitials(haystack);
  const hayTokens = new Set(tokensOf(haystack));

  const nameTokens = tokensOf(normalize(name));
  if (nameTokens.length === 0) return 0;

  const fullName = nameTokens.join(" ");

  let score = 0;

  const covered = nameTokens.filter((t) => hayTokens.has(t)).length;
  const fullNameHit = contigHaystack.includes(fullName);

  // Only a contiguous full-name hit is strong evidence. Scattered tokens are
  // common noise: a search for "Theo Browne" matches "Theo Fleuron" and a book
  // caption mentioning both words, none of which show the person.
  if (fullNameHit) {
    score += 0.75;
  } else if (covered === nameTokens.length) {
    score += 0.15;
  } else {
    score += 0.1 * (covered / nameTokens.length);
  }

  if (fileHaystack.includes(fullName)) score += 0.15;

  const aliasHit = aliases.some((alias) => {
    const a = stripInitials(normalize(alias));
    return a.length >= 3 && contigHaystack.includes(a);
  });
  if (aliasHit && !fullNameHit) score += 0.1;

  // Composition tie-breaker. Deliberately reads the FILE NAME only: captions
  // nearly always contain words like "photo" or "portrait", so scoring the
  // whole haystack makes every candidate tie and the ranking useless.
  if (PREFERRED_TERMS.some((t) => hasTerm(fileHaystack, t))) score += PREFERENCE_BONUS;

  // Negative evidence.
  if (NON_PHOTO_TERMS.some((t) => hasTerm(haystack, t))) score *= NON_PHOTO_FACTOR;
  if (LOOKALIKE_TERMS.some((t) => hasTerm(haystack, t))) score *= LOOKALIKE_FACTOR;
  if (FABRICATED_TERMS.some((t) => hasTerm(haystack, t))) score *= FABRICATED_FACTOR;
  if (CROWD_TERMS.some((t) => hasTerm(haystack, t))) score *= CROWD_FACTOR;

  if (fullNameHit) {
    const at = contigHaystack.indexOf(fullName);
    const rest = contigHaystack.slice(at + fullName.length).trim();
    const next = rest.split(" ")[0] ?? "";
    const previous = contigHaystack.slice(0, at).trim().split(" ").pop() ?? "";

    if (GENERATIONAL_SUFFIXES[next] === true) score *= SUFFIX_FACTOR;

    // "Trump and Clinton" / "Clinton with Trump": the face we want shares the
    // frame, so it is small, angled, or ambiguous.
    if (CO_SUBJECT_JOINERS[next] === true || CO_SUBJECT_JOINERS[previous] === true) {
      score *= CO_SUBJECT_FACTOR;
    }
  }

  return clamp01(score);
}
