# Improved Blocked Attention Sparsity (CS 498 ML Systems, UIUC, Fall 2026)

Snapshot of **Nov 3, 2026**, the day before the mid-semester report. Team: Haochen Ding, Lucas Cai, Yifei Li,
Kevin Yang. All experiments ran on NCSA DeltaAI (NVIDIA GH200 GPUs): **4.41 charged GPU-hours** in total.

> **AI agent?** Read `AGENTS.md` first (short), then this file, then `reports/` in order.

## The project in one paragraph

BLASST (arXiv 2512.12087, NVIDIA) makes long-prompt attention cheaper without any training. Inside a
FlashAttention-style loop it skips any 128 x 128 **tile** of the attention matrix whose scores sit far below
the best score seen so far, using **one threshold lambda** everywhere. We (1) reproduced BLASST on
Llama-3.1-8B-Instruct (text), then (2) asked whether one lambda is fair on a video-language model
(Qwen2.5-VL-7B-Instruct), where the attention matrix mixes text and video tokens. It is not: **one lambda cuts
the question's view of the video hardest.** Protecting those few tiles looks cheap and may keep accuracy, but
the accuracy evidence so far is only a hint.

## Status in six lines

1. **Baseline reproduced.** At about 50% of tiles skipped, the RULER score drops 1.20 points at 8K and 0.44 at 16K
   (paper: 0.37 and 0.19). Our code matches NVIDIA's reference (ModelOpt) within 0.32 points of % skipped
   (using NVIDIA's convention for the last, partial query tile).
2. **Main finding.** On Qwen2.5-VL video prompts, one lambda that skips 50% of all tiles skips **84% of T->V tiles**
   (text after the video looking at the video) but only **47% of V->V tiles**, on all 10 prompts.
3. **Cause.** Text rows' running max (the "bar") is set high by the attention sink and earlier text before any
   video key is visited, so video tiles look small next to it.
4. **Weight lost.** Text rows give the video only about 5% of their attention, but the 50% lambda removes over half
   of it (about 70% for the answer token).
5. **Accuracy pilot** (90 Video-MME questions, about 46% skipped live): dense **65**, one lambda **61**, never
   skipping question->video tiles **64**. Same direction as the attention data, but only 3-4 answers differ
   (p = 0.125): a hint, not a result.
6. **Next:** per-region lambda tested on about 300 questions; a "smarter bar" variant that keeps one lambda
   (`reports/4_future_directions.md`). Neither has been run yet.

## Map

```
CS498_BLASST_project/
├── README.md              you are here: overview, glossary, key numbers
├── AGENTS.md              instructions for an AI agent reading this folder
├── Project_Proposal.pdf   the project proposal (original plan, 3 ideas)
├── reports/               the write-ups; the source of truth for every number
│   ├── 1_baseline_llama.md        BLASST reproduced on Llama-3.1-8B, RULER (Nov 1)
│   ├── 2_vlm_regions_step1.md     per-region % skipped on Qwen2.5-VL-7B video (Nov 1)
│   ├── 3_vlm_regions_batch2.md    weight lost, layout check, accuracy pilot (Nov 1-2)
│   ├── 4_future_directions.md     next steps and ideas, not yet run (Nov 3)
│   └── figures/                   all 6 figures (PNG)
├── results/               raw CSV / JSON behind the reports        (see results/README.md)
├── code/                  every script that produced them, as run  (see code/README.md)
└── notes/                 background desk research: papers, related work  (see notes/README.md)
```

Reading order: this file -> `reports/1` (skim section 1) -> `reports/2` -> `reports/3` -> `reports/4`.
Use `results/`, `code/` and `notes/` only to check a number, re-run something, or look up a paper.

## Glossary (every file uses these words)

- **Tile**: 128 query rows x 128 keys of the attention score matrix, counted per (layer, query head).
- **Running max / the bar**: for each query row, the largest score among the keys visited so far. Keys are
  visited tile by tile, left to right (prompt order).
- **Skip rule (BLASST)**: skip a tile (no exp, no P.V; QK^T is still computed) when **every** one of its 128 rows
  has (tile row max - running row max) < ln(lambda). One row with a strong score keeps the tile for all rows.
- **lambda**: the threshold. In words: skip a tile if its best key's weight is below lambda x the bar's weight.
- **Margin**: max over the tile's rows of (tile row max - running row max); skipped iff margin < ln(lambda).
  Margins do not depend on lambda, so **one dense pass gives the skip set for every lambda** (offline).
- **% skipped**: skipped tiles / causal tiles (tiles on or below the diagonal).
- **Dense**: nothing skipped (standard attention).
- **Calibrated lambda**: lambda = a / L (L = prompt tokens), a fitted for a target % skipped. Llama:
  a = 106.5 (50%), 846 (75%), 2,934 (90%).
- **"The 50% lambda"** (VLM study): each prompt's own exact lambda that skips 50% of its tiles on the dense
  margins. A measuring tool, not a deployable rule (it needs a dense pass first).
- **Offline vs live**: offline = computed from dense-pass margins; live = skipping really on during prefill.
  Live skipping changes what later layers see (50% offline became about 46% live in the pilot).
- **Regions**: every 128-token block is text, video, or mixed. A tile's region is (query block kind -> key block
  kind): **T->T**, **T->V** (text after the video looking at video keys), **V->T** (video looking back at
  earlier text), **V->V**, **mixed**. Video tokens are `input_ids == 151656`.
- **question->video**: tiles whose query block holds any question token and whose key block holds any video token.
- **Weight lost**: for one (layer, head, query row), the share of its dense softmax weight (100% over all its keys,
  across all tiles) that sits on skipped tiles.
- **Attention sink**: the first token(s) of a prompt get very large scores from most rows but carry little value
  information.
- **Layouts**: **A** `[transcript part 1][video][transcript part 2][question]` (ours); **B**
  `[video][subtitles][question]` (Video-MME with subtitles); **C** `[video][question]` (Video-MME default).

## Key numbers

**Baseline, Llama-3.1-8B-Instruct** (`reports/1_baseline_llama.md`):

| Quantity | Ours | Paper |
|---|---|---|
| % skipped, calibrated 50% target, 4K / 8K / 16K / 32K | 46.73 / 50.82 / 53.38 / 55.90 | 54.20 / 49.70 / 52.20 / 46.96 |
| % skipped, fixed lambda = 1e-3, 4K / 8K / 16K / 32K | 8.35 / 16.31 / 26.99 / 39.32 | 23.09 / 37.92 / 52.38 / 65.72 |
| RULER (3 tasks x 50), 8K: dense -> about 50% | 95.07 -> 93.87 (48.25% skipped) | 95.07 -> 94.70 |
| RULER, 16K | 95.33 -> 94.89 (51.40% skipped) | 94.80 -> 94.61 |

The fixed-lambda gap is probably the tile shape: with 16 x 64 tiles we get 38.18% at 8K vs the paper's 37.92
(a fit after the fact, not proven). So we compare at matched % skipped, not matched lambda.

**Regions, Qwen2.5-VL-7B, layout A** (5 long Video-MME videos x 8K and 16K = 10 prompts; `reports/2`):

| Region | Share of tiles | % skipped at 50% overall | at 75% overall | Median margin |
|---|---|---|---|---|
| V->V | 71.57% | 46.87 | 75.17 | -2.04 |
| T->V | 11.68% | **83.95** | 96.62 | **-4.85** |
| T->T | 2.55% | 31.20 | 48.34 | -0.30 |
| V->T | 12.34% | 38.62 | 57.33 | -1.16 |
| mixed | 1.87% | 57.80 | 82.16 | -2.60 |

Checks: the gap holds at matched distance from the diagonal (28-55 points), for the question itself (86.02% vs
83.48% for transcript rows), and in 26 of 28 layers. Qwen needs a much larger lambda for 50% (median 0.151 at 8K,
0.097 at 16K) than Llama (0.0130 / 0.0052).

**Weight lost at the 50% lambda, layout A, 8K / 16K** (`reports/3`, item 1):

| Rows | Attention on video keys (dense) | Video attention removed | Share of their video attention removed | Total weight lost |
|---|---|---|---|---|
| question | 4.24% / 5.63% | 2.41 / 3.14 points | 56.9% / 55.7% | 4.51% / 4.57% |
| answer position | 2.36% / 3.29% | 1.69 / 2.26 points | 71.6% / 68.7% | 3.56% / 3.63% |
| all rows | 33.29% / 42.64% | 4.40 / 5.98 points | 13.2% / 14.0% | 5.83% / 6.83% |
| Llama, all rows | | | | 2.74% / 2.29% |

**Layout check**, question->video minus V->V at the 50% lambda, 8K / 16K (`reports/3`, item 2): A 40.8 / 38.0
points; B 34.6 / 31.8; C 16.1 / 17.0. The effect survives every layout but its size depends on where text sits.

**Accuracy pilot** (`reports/3`, item 3): layout C, about 16K tokens, 128 frames, 30 short and medium Video-MME
videos x 3 questions = 90, skipping on during prefill.

| Setting | Correct / 90 | Live % skipped | right -> wrong vs dense | wrong -> right |
|---|---|---|---|---|
| dense | 65 | 0 | | |
| one lambda (50% on dense margins) | 61 | 45.8 | 4 | 0 |
| question->video never skipped, rest at one shared lambda | 64 | 45.7 | 1 | 0 |

## How we got here

- The proposal had three ideas on top of the reproduction: cheaper tile scoring from downsampled Q/K, reusing skip
  masks across layers, and per-modality thresholds. Desk research found close published work for the first two
  and for per-layer / per-head thresholds (`notes/related_work.md`, `notes/directions_idea1.md`,
  `notes/directions_idea2.md`), so **per-modality (per-region) thresholds on video became the main study**.
- Considered and dropped: time-series models (prompts of 128-512 tokens have too few tiles to matter); a
  "certified skip" from per-tile score bounds (a teammate's offline measurements showed the bounds are too loose
  to certify any skippable tile).

## Main caveats

- Small samples: 10 prompts (5 videos) for the region study; 90 questions from 30 videos for accuracy.
- Per-prompt exact lambda (oracle); Llama's calibration does not transfer to Qwen, and no Qwen calibration rule exists yet.
- The region gap depends on the prompt layout; the pilot used layout C, where it is smallest.
- **No speed numbers.** Our attention function is a PyTorch reference that computes every tile and then masks
  the skipped ones; per-query-head decisions overstate what a GQA kernel would save.
- Weight lost is not output damage; only accuracy measures damage.

## Compute

Baseline 3.02, VLM step 1 0.13, VLM batch 2 1.26: **4.41 charged GPU-hours** (DeltaAI, allocation `beqg-dtai-gh`).

## Not included (too big; still on DeltaAI, ask Kevin)

Saved margins and Q/K for Llama (`results/margins/` 3 GB, `results/qk/` 22 GB), Qwen margins and per-row weight
arrays (`vlm_margins/` 153 MB, `vlm2_tensors/` 5.4 GB), decoded frames and prompts, model and dataset caches.

## Deadlines

Mid-semester report: 4 pages, by email, **6:00 PM CST Wed Nov 4, 2026**. Presentations Dec 2 / 7 / 9. Final
report (8 pages) and code zip: Dec 18.
