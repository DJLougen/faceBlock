/** Threshold policy for match scores. */

export { DEFAULT_THRESHOLD, HARD_NEGATIVE_MARGIN } from "../shared/config.ts";

/** A match censors when its score reaches the operating threshold. */
export function shouldCensor(score: number, threshold: number): boolean {
  return score >= threshold;
}
