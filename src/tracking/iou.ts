import type { Box } from "../shared/types.ts";

/** Centre-distance fallback radius, as a multiple of the larger box dimension. */
const ASSOCIATION_RADIUS_FACTOR = 1.5;
/** Floor so a small face box can still associate across short motion. */
const MIN_ASSOCIATION_RADIUS_PX = 24;

/**
 * Hard wall-clock bound on how long a track may coast on prediction alone.
 *
 * Misses only accrue when a detection round actually runs, so a track can
 * outlive its usefulness without ever accumulating misses: a hidden tab runs
 * no rAF and no sampling, a paused-then-scrubbed video may not sample for
 * seconds, and a busy analyser drops rounds entirely. In every case the
 * velocity extrapolation keeps marching the mask across the screen long
 * after the pixels it described are gone. Two seconds is far beyond the
 * worst-case inter-detection gap (~4 active misses at 250 ms) yet far below
 * any realistic hidden-tab duration, so it only ever fires when prediction
 * has genuinely lost contact with the video.
 */
export const MAX_TRACK_AGE_MS = 2000;

/**
 * A face being followed across video frames between low-rate recognition
 * rounds. Recognition is expensive, so it runs every few hundred ms; this
 * record is what lets the mask stay glued to the face at full video rate.
 */
export interface Track {
  /** Last OBSERVED box, in frame pixels. */
  box: Box;
  /** Velocity, pixels per millisecond, from the last two observations. */
  vx: number;
  vy: number;
  /** Timestamp of the last matched detection. */
  lastSeenMs: number;
  /** Consecutive detection rounds without a match. */
  misses: number;
}

/**
 * Intersection-over-union of two boxes. Returns 0 for disjoint boxes and for
 * either box having non-positive area, 1 for identical boxes. Never NaN;
 * the result is clamped to [0, 1].
 */
export function iou(a: Box, b: Box): number {
  if (a.width <= 0 || a.height <= 0 || b.width <= 0 || b.height <= 0) return 0;
  const iw = Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x);
  const ih = Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y);
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  const union = a.width * a.height + b.width * b.height - inter;
  if (union <= 0) return 0;
  const v = inter / union;
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/**
 * Start tracking a freshly detected face. The box is copied so later
 * mutation of the caller's object cannot corrupt the track.
 */
export function createTrack(box: Box, nowMs: number): Track {
  return {
    box: { x: box.x, y: box.y, width: box.width, height: box.height },
    vx: 0,
    vy: 0,
    lastSeenMs: nowMs,
    misses: 0,
  };
}

/**
 * Predicted box at `nowMs`: last observed position plus velocity times the
 * elapsed time. This prediction is what moves the mask between recognition
 * rounds — detection runs at a low rate, but the face keeps moving every
 * frame. Elapsed time is clamped to >= 0 so a backwards clock cannot
 * extrapolate into the past. Pure: the track is never mutated.
 */
export function coastTrack(track: Track, nowMs: number): Box {
  const dt = Math.max(0, nowMs - track.lastSeenMs);
  return {
    x: track.box.x + track.vx * dt,
    y: track.box.y + track.vy * dt,
    width: track.box.width,
    height: track.box.height,
  };
}

/**
 * Fold one recognition round into the track set. Called once per sampled
 * frame with that frame's detected face boxes.
 *
 * Association is greedy by IoU against each track's last observed box:
 * the best-scoring pair above `iouThreshold` is consumed first, and ties
 * break deterministically on lowest track index then lowest detection
 * index. Matched tracks trust the detection over the prediction; unmatched
 * tracks coast forward and accrue a miss; leftover detections spawn new
 * tracks; tracks missing more than `maxMisses` rounds are dropped so a
 * lost face falls back to fresh detection.
 *
 * `maxAgeMs` drops tracks whose last observation is older than the bound
 * BEFORE association, so an ancient track cannot steal a fresh detection
 * (which would hand it a huge, meaningless velocity). Omit it to keep pure
 * miss-count lifetime.
 *
 * Returns a new array; neither `tracks` nor its entries are mutated.
 */
export function applyDetections(
  tracks: readonly Track[],
  detections: readonly Box[],
  nowMs: number,
  opts?: { iouThreshold?: number; maxMisses?: number; centreFallback?: boolean; maxAgeMs?: number }
): Track[] {
  const iouThreshold = opts?.iouThreshold ?? 0.3;
  const maxMisses = opts?.maxMisses ?? 2;
  const centreFallback = opts?.centreFallback ?? true;
  const maxAgeMs = opts?.maxAgeMs;

  // Age gate: a track whose last OBSERVED box is older than the bound is
  // dropped outright, regardless of its miss count. This is the only lifetime
  // limit that still works when detection rounds stop happening at all.
  if (maxAgeMs !== undefined) {
    tracks = tracks.filter((t) => nowMs - t.lastSeenMs <= maxAgeMs);
  }
  // Greedy maximum-IoU assignment. Scanning tracks outer / detections inner
  // in ascending index order and keeping the first strictly-best score makes
  // ties resolve to the lowest track index, then the lowest detection index.
  const detTaken = new Array<boolean>(detections.length).fill(false);
  const detForTrack = new Array<number>(tracks.length).fill(-1);
  for (;;) {
    let bestIou = iouThreshold;
    let bestT = -1;
    let bestD = -1;
    for (let t = 0; t < tracks.length; t++) {
      if (detForTrack[t] !== -1) continue;
      const track = tracks[t]!;
      for (let d = 0; d < detections.length; d++) {
        if (detTaken[d]) continue;
        const score = iou(track.box, detections[d]!);
        if (score > bestIou) {
          bestIou = score;
          bestT = t;
          bestD = d;
        }
      }
    }
    if (bestT < 0) break;
    detForTrack[bestT] = bestD;
    detTaken[bestD] = true;
  }

  // Second pass: nearest-centre fallback for pairs IoU could not pair.
  //
  // Recognition runs every 250-800 ms. In that window a head routinely travels
  // further than any overlap, so IoU alone would spawn a fresh track each round
  // while the old one lingered, and the mask would flicker between duplicates.
  // Associate the nearest remaining pair whose centre distance falls inside a
  // radius scaled to the box, which grows with the face so big faces may move
  // further than small ones.
  for (;;) {
    if (!centreFallback) break;
    let bestDist = Number.POSITIVE_INFINITY;
    let bestT = -1;
    let bestD = -1;
    for (let t = 0; t < tracks.length; t++) {
      if (detForTrack[t] !== -1) continue;
      const track = tracks[t]!;
      const radius = Math.max(
        ASSOCIATION_RADIUS_FACTOR * Math.max(track.box.width, track.box.height),
        MIN_ASSOCIATION_RADIUS_PX,
      );
      const cx = track.box.x + track.box.width / 2;
      const cy = track.box.y + track.box.height / 2;
      for (let d = 0; d < detections.length; d++) {
        if (detTaken[d]) continue;
        const det = detections[d]!;
        const dx = det.x + det.width / 2 - cx;
        const dy = det.y + det.height / 2 - cy;
        const dist = Math.hypot(dx, dy);
        if (dist > radius) continue;
        if (dist < bestDist) {
          bestDist = dist;
          bestT = t;
          bestD = d;
        }
      }
    }
    if (bestT < 0) break;
    detForTrack[bestT] = bestD;
    detTaken[bestD] = true;
  }

  const next: Track[] = [];
  for (let t = 0; t < tracks.length; t++) {
    const track = tracks[t]!;
    const d = detForTrack[t]!;
    if (d >= 0) {
      const det = detections[d]!;
      const dt = nowMs - track.lastSeenMs;
      // Velocity comes from the last two OBSERVED boxes. A zero or negative
      // elapsed time would divide by nothing useful, so keep the old velocity.
      next.push({
        box: { x: det.x, y: det.y, width: det.width, height: det.height },
        vx: dt > 0 ? (det.x - track.box.x) / dt : track.vx,
        vy: dt > 0 ? (det.y - track.box.y) / dt : track.vy,
        lastSeenMs: nowMs,
        misses: 0,
      });
    } else {
      const misses = track.misses + 1;
      if (misses > maxMisses) continue;
      // Not seen this round. `box` stays the last OBSERVED box and
      // `lastSeenMs` is unchanged: the consumer renders coastTrack(track, now)
      // on every presented frame, so advancing the box here would count the
      // same motion twice and the mask would drift ahead of the face.
      next.push({
        box: { x: track.box.x, y: track.box.y, width: track.box.width, height: track.box.height },
        vx: track.vx,
        vy: track.vy,
        lastSeenMs: track.lastSeenMs,
        misses,
      });
    }
  }
  for (let d = 0; d < detections.length; d++) {
    if (!detTaken[d]) next.push(createTrack(detections[d]!, nowMs));
  }
  return next;
}
