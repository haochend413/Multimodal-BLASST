# Future directions (logged Nov 3, 2026; nothing here has been run yet)

Facts and pointers for the "future work / next steps" part of the report: source material, not report text.
Details and full citations: `../notes/one_threshold_designs.md`, `../notes/one_threshold_literature.md` (both desk
research, AI-assisted, no experiments run).

## 1. One threshold, smarter bar (teammate's idea)

**Idea.** Keep ONE lambda, but change the bar (the running max) each tile is compared against, so the same
lambda treats text and video fairly. Alternative to a separate lambda per region.

**Why (from our data).** The problem is the bar, not lambda: for text rows after the video, the running max is
already set by the attention sink and transcript before any video key is visited. Median T->V margin is -4.85
(best video key = 0.8% of the bar) vs -2.04 for V->V; only 1.7% of T->V tiles ever hold a new running max.

**Two candidates:**
- **Own-modality bar:** judge a video tile against the best *video* key seen so far. Free in a kernel: the
  dense result does not depend on the order key tiles are visited, so for text rows visit video tiles first.
  Keeps BLASST's per-key bound (tighter). Same as BLASST on text-only input.
  Prediction: T->V % skipped falls toward V->V's; the region gap stops depending on layout
  (today: ~40 / ~33 / ~16 points for layouts A / B / C).
  Risks: later frames keep raising the bar (try ascending and descending frame order); a sink inside the video.
- **Sink-excluded bar:** always keep the sink tile, but leave it out of the running max. Sinks get large scores
  but carry near-zero values (VATP, arXiv 2406.12335). Combine with the own-modality bar.

**What is NOT new:** a fixed per-region shift of scores (per-modality temperature or offset) is exactly per-region
lambda (skip if margin + c_r < ln lambda  <=>  margin < ln lambda_r). Per-head z-scored margins = per-head lambda.
Test for "per-region lambda in disguise": compare its skipped tiles with a per-region-lambda rule that skips the
same % in each region.

**Ruled out (predicted to make T->V worse or not help):** mass-based / top-p rules (text rows really do put
little mass on video: 4.7-6.5% of their attention); published bar variants that put the sink INTO the bar;
final-row-max bar (needs a second QK^T pass; yardstick only).

**Novelty.** No paper found reports % skipped per region for any skip rule. Closest relatives: MoDES (arXiv
2511.15690, CVPR 2026: separate text/vision thresholds for MoE expert skipping in VLMs); MadaKV (arXiv
2506.15724: modality-aware KV-cache eviction). One-threshold methods such as XAttention stay near dense on
Video-MME but do not look inside regions. (Paper-specific numbers in the literature note are not all verified.)

**Test plan and cost.** One dense recording pass on 30 prompts, then offline comparison at matched 50% skipped:
per-region % skipped, weight lost for question rows and the answer position, vs one lambda and vs per-region
lambda (~0.5 GPU-h). Then Video-MME accuracy for the winner only on the 90 pilot questions (~1.5 GPU-h).

## 2. Two facts useful in the report

- **BLASST's guarantee is empty at our lambdas.** Each skipped key's weight < lambda x the best key's, so a
  128-key tile holds < 128 lambda x the best key. That is > 1 (says nothing) once lambda > 1/128 = 0.0078:
  every Qwen lambda (0.08-0.23) and Llama at 8K (0.013). Only Llama 16K (0.0052) gives a real per-tile bound.
  So weight lost, not the guarantee, measures damage.
- **Exact error of skipping** for one row: (output with skipping) - (dense output)
  = weight lost x ((kept output) - (average value of the skipped keys)). Losing weight hurts only if the
  skipped keys' values differ from what is kept.

## 3. Other directions already decided (Nov 1)

- **Per-region lambda at scale (main study, steps 2-3):** the pilot (dense 65/90, one lambda 61/90, T->V never
  skipped 64/90) needs several hundred questions or a higher skip level to be conclusive; also a calibrated
  rule for choosing lambda (each prompt's exact 50% lambda was used, which a deployed system cannot do).
- **Max vs mass** (HiLS, arXiv 2607.02980; note `../notes/paper_2607.02980.md`): BLASST judges a tile by its max; the
  softmax cares about its mass (up to ln 128 = 4.85 apart on a flat tile). About 5 GPU-h.
- **Mask stability / prefetch** (ECHO, OSDI 2026; note `../notes/paper_osdi26_liu.md`): how much the kept-tile set
  changes between decode steps, layers and neighboring query tiles. About 2-3 GPU-h.
