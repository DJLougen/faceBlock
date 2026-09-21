/** Shapes for self-seed reference acquisition. */

/**
 * Text that travels WITH a discovered image. Ranking reads only this text, never
 * the pixels, so ordering candidates costs no inference and stays offline-safe.
 *
 * Only the person's name is ever sent to the endpoints that produce this; see
 * resolve.ts for the egress rule.
 */
export interface CandidateContext {
  /** e.g. "Donald_Trump_official_portrait.jpg" */
  filename: string;
  /** e.g. "Donald Trump" */
  pageTitle: string;
  /** Description / alt / category text. May be "". */
  caption: string;
}

export interface CandidateImage extends CandidateContext {
  /** Direct http(s) URL of the original file — the provenance record, not what we download. */
  url: string;
  /** Smaller preview URL when the source offers one; also the cheaper download. */
  thumbUrl?: string;
  /** Which endpoint produced this candidate. */
  source: "wikipedia" | "wikidata" | "commons";
  /** Context score in 0..1. See scoreCandidate. */
  score: number;
}

export interface ResolveResult {
  name: string;
  /** Deduped, sorted by score descending, capped at the requested limit. */
  candidates: CandidateImage[];
  /** Human-readable record of the endpoints queried, for the UI to disclose. */
  queries: string[];
  /**
   * True when the overall deadline fired before all sources finished —
   * `candidates` is then partial, and an empty list means "search timed out",
   * not "no photos exist".
   */
  timedOut?: boolean;
}
