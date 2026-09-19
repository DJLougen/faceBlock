# FaceBlock matching — measurement record (segment 1)

Local autoresearch record of the **synthetic identity matcher**, not a submission, not a face-detector result, not an on-X live measurement. The keep rule is independent of headline FPR: `bun test tests` must pass and TPR must stay ≥ 0.5 (harness fail-closed).

## Task

Lower `false_censors_per_1k = 1000 * fp / 2000` on a frozen seeded embedding protocol, without dropping TPR through the floor, and cut match latency when FPR does not rise.

Scored artifact: `matchFace` in `src/matching/matcher.ts` plus `DEFAULT_THRESHOLD` / `MIN_GALLERY_AGREEMENTS` in `src/shared/config.ts`.

## Box / Harness

- Hardware: Apple M3 Max, CPU, bun 1.4.2
- Entrypoint: `bash autoresearch.sh` (unit tests then `bench/identity_bench.ts`)
- Protocol deviations: none vs the frozen bench file. Cool-gates / GPU N/A.
- Session ran on `master` (not `autoresearch/*`); discard would not full-reset.

## Scoring surface

| Number | Value | Authority |
| --- | --- | --- |
| Official sealed face-ID score | n/a | none exists for this repo |
| Local `false_censors_per_1k` | 1.5 (run 8) | this harness, seed 42 |

These two columns are **not comparable**. The local analog is synthetic L2 embeddings, not MobileFaceNet / LFW / X.com.

## Results

| run | change | FPR/1k | tpr | fp | match_p50_us | verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | max-cosine, thr=0.7, k=1 | 100 | 0.870 | 200 | 248.58 | baseline |
| 2 | k=2 2nd-best must clear thr | 58 | 0.840 | 116 | 257.71 | keep |
| 3 | k=3 | 32 | 0.820 | 64 | 269.46 | keep |
| 4 | k=4 | 19 | 0.812 | 38 | 270.96 | keep |
| 5 | k=5 | 10 | 0.804 | 20 | 270.33 | keep |
| 6 | skip gallery L2 copy | 10 | 0.804 | 20 | 100.88 | keep (speed) |
| 7 | thr 0.70→0.72 | 2.5 | 0.802 | 5 | 101.12 | keep |
| 8 | thr 0.72→0.73 | 1.5 | 0.800 | 3 | 100.17 | keep (best) |

Headline vs baseline (run 1 → 8): FPR/1k **100 → 1.5 (−98.5%)**, TPR **0.87 → 0.80 (−8.0% relative)**, match_p50_us **248.58 → 100.17 (−59.7%)**. roc_auc stayed 0.9751 on every run (decision rule changed, scores did not).

## Retracted results

None. No run was flagged. Run 6 did not improve FPR; it is kept as a Pareto latency win with bit-identical tp/fp/fn vs run 5.

## Closed levers

- **k-gt-5-on-8ref** — k=2..5 is the anti-spike ladder; higher k overfits the blend-0.52 hard-neg generator. Product cap is `MAX_PROTOTYPES=5`.
- **threshold-gt-073** — 0.72→0.73 bought 2 FPs and cost 1 TP. `pos_p05=0.664`; 0.74+ eats hard positives for a 3-FP remainder.
- **skip-l2-does-not-move-fpr** — run 5 vs 6 identical counts; mechanism is alloc+scale.
- **second-identity-margin** — hard-negs point at the *target* id; other-id scores stay ~0.22. Jev value_E confidence 0.17.
- **match-time-prototypes** — bench scores 8 raw embeddings; FPS is dead on the metric path. Jev value_D=0.89.

## Protocol

- One coherent change per run.
- Fail-closed: empty splits, non-finite metrics, self-match, TPR < 0.5, unit tests.
- Noise: FPR is a 2000-trial count (integer fp). 1 FP = 0.5 / 1k. Effects of several FPs are above that grain.
- Same-binary: run 6 is the latency A/B against run 5 with identical decisions.
- Jev (jev-1.13.0) calibrated first (heads noul=0.49, envelope 0.85/0.15). Ranking used per-item scores (A=2.76 > B=2.24). Round-2 `how` confidence was 0.39 — plurality, not authority. 3 requests, 3076 input / 424 output tokens.

## Files / Sources

- `provenance/PROVENANCE.json` — canonical numbers
- `bench/identity_bench.ts` — frozen protocol
- `src/matching/matcher.ts`, `src/shared/config.ts` — kept matcher
- Baseline commit `a38b632b2c00`
- Plan: `plan(20260919-140217).md` §28 (primary = false censors per 1,000 unrelated faces)
