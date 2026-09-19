export {
  cosine,
  cosineNormalized,
  dot,
  l2Norm,
  l2Normalize,
  l2NormalizeInPlace,
} from "./cosine.ts";
export { matchFace } from "./matcher.ts";
export {
  DEFAULT_THRESHOLD,
  HARD_NEGATIVE_MARGIN,
  shouldCensor,
} from "./thresholds.ts";
export type { BlockedIdentity, MatchResult } from "../shared/types.ts";
