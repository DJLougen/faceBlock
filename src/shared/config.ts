/** Conservative v0 defaults. Thresholds are operating points, not model claims. */

export const EMBED_DIM = 128;
export const ALIGN_SIZE = 112;

/** Conservative cosine threshold for L2-normalized embeddings. */
export const DEFAULT_THRESHOLD = 0.73;

/** Require (posScore - maxHardNegScore) >= this, else refuse to censor. */
export const HARD_NEGATIVE_MARGIN = 0.02;

/** Censor only if this many gallery vectors clear the threshold (k=1 when gallery is smaller). */
export const MIN_GALLERY_AGREEMENTS = 5;

export const MAX_PROTOTYPES = 5;
export const MIN_REFERENCE_IMAGES = 3;

/**
 * Fractional padding applied around YuNet's tight face box before masking.
 * Top is larger for hair; bottom is larger for beards/jaw — the old uniform
 * margin+scale pair grew mostly upward and left beards exposed.
 */
export const BOX_PADDING_X = 0.18;
export const BOX_PADDING_TOP = 0.42;
export const BOX_PADDING_BOTTOM = 0.32;

/** @deprecated Use BOX_PADDING_* — kept for callers still on margin+scale expandBox. */
export const BOX_MARGIN_X = BOX_PADDING_X;
/** @deprecated Use BOX_PADDING_TOP */
export const BOX_MARGIN_Y = BOX_PADDING_TOP;
export const BOX_MARGIN_TOP = BOX_PADDING_TOP;
export const BOX_MARGIN_BOTTOM = BOX_PADDING_BOTTOM;
export const BOX_SCALE_X = 1 + 2 * BOX_PADDING_X;
export const BOX_SCALE_Y = 1 + BOX_PADDING_TOP + BOX_PADDING_BOTTOM;

export const MIN_FACE_PX = 24;
export const MIN_MEDIA_PX = 32;

export const DETECTOR_CONCURRENCY = 1;
export const EMBED_CONCURRENCY = 4;

/**
 * When a blocked identity has only one gallery vector, w600k_mbf often scores
 * same-person appearance drift just under MATCH_THRESHOLD (measured 0.397 on the
 * local theo.jpg → theo-user-02.jpg fixture). Apply this slack only for
 * single-vector galleries; multi-vector identities keep the full threshold.
 */
export const SINGLE_GALLERY_APPEARANCE_SLACK = 0.025;
