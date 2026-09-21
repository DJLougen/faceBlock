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

export const BOX_MARGIN_X = 0.15;
export const BOX_MARGIN_Y = 0.2;
export const BOX_SCALE_X = 1.2;
export const BOX_SCALE_Y = 1.25;

export const MIN_FACE_PX = 24;
export const MIN_MEDIA_PX = 32;

export const DETECTOR_CONCURRENCY = 1;
export const EMBED_CONCURRENCY = 4;
