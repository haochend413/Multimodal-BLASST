> **In this package:** the data files named below are in `../results/vlm_regions/`, figures in `figures/`, code in `../code/`. Not included (too big, still on DeltaAI): `vlm_margins/`, `vlm_inputs/`. Report copied unchanged from DeltaAI except this line.

# Per-region % skipped on Qwen2.5-VL-7B: the kill test (step 1 of the per-region lambda study)

Run Nov 1, 2026 on DeltaAI (GH200). Every number below comes from the CSV/JSON files in this folder
(`/work/nvme/beqg/zyang21/blasst/results/vlm_regions/`); this file is meant to stand alone.

## Takeaway (3 sentences)

Under one lambda that skips 50% of all tiles, the four attention regions of a long-video prompt skip very
differently: text rows looking at video (T->V) skip 84% of their tiles, video looking at video (V->V) 47%,
video looking back at text (V->T) 39%, and text looking at text (T->T) 31%; T->V is skipped more than V->V on
all 10 prompts (by 34.5 to 44.3 points), in 26 of 28 layers, at matched distance from the diagonal, and for the
question itself. The direction is the opposite of our hunch: one lambda does not spare the question's view of the video,
it cuts it hardest (the question rows skip 86% of video tiles), so if that view carries the answer, a single
lambda is aggressive exactly where it is riskiest, and a per-region lambda can protect it cheaply because T->V
is only 12% of the tiles. This is a measurement on a dense pass, not an accuracy result: whether skipping T->V
actually costs Video-MME accuracy is step 3.

## Words

- **Tile**: 128 query rows x 128 keys of the attention score matrix. **Skip rule**: skip a tile when every one
  of its 128 rows has (tile row max - running row max) < ln(lambda). **Margin**: the largest of those 128
  differences; a tile is skipped exactly when margin < ln(lambda). **% skipped** = skipped / causal tiles.
- **Region**: every 128-token block of the prompt is *text* (no video tokens), *video* (all video tokens) or
  *mixed* (straddles the end of the video). A tile's region is (query block kind -> key block kind):
  **V->V** video rows looking at video keys; **T->V** text rows after the video (transcript part 2 and the
  question) looking at video keys; **T->T** text looking at text; **V->T** video rows looking back at the
  transcript before the video; **mixed** either block is mixed. Video tokens are exactly `input_ids == 151656`;
  `<|vision_start|>`, `<|vision_end|>` and chat-control tokens count as text.
- Tiles are counted per (layer, query head): 28 layers x 28 query heads, so one 16K prompt has about 6.5 million
  causal tiles.

## Setup

- Model `Qwen/Qwen2.5-VL-7B-Instruct`, bf16, transformers 4.57.3. Our attention function (`blasst_attn.py`,
  unchanged from the Llama baseline) runs in the 28-layer language model only:
  `attn_implementation={"text_config": "blasst", "vision_config": "sdpa"}`; the vision encoder always uses
  standard attention. GQA: 28 query heads share 4 key/value heads.
- One **dense** prefill per prompt (skipping off) with margins recorded. The margins give % skipped at any
  lambda afterwards, same trick as Fig. 1 of the baseline.
- For each prompt we pick the **one lambda that skips exactly 50%** (and separately 75%) of all its causal tiles,
  over all layers and heads, the way BLASST uses one lambda everywhere, and ask where those skips land.
- Prompt layout (Qwen chat template): `[system] [user: transcript part 1, ~1,024 tokens] [video]
  [transcript part 2, 512-1,024 tokens] [question + 4 options + "Answer with the option's letter"] [assistant]`.
  Transcript = the Video-MME subtitle file of the same video. Part 1 is trimmed so the video starts at token
  1,024, exactly on a tile boundary, so only the block at the end of the video is mixed.
- Video: frames evenly spaced over the whole video, each resized to 336 x 588 pixels (252 tokens per pair of
  frames). The number of frames sets the length.
- Videos: Video-MME (`lmms-lab/Video-MME`), chunk 1 of 20 (30 videos). Rule: has subtitles with enough text,
  not "short", one per domain, first in ID order; all 5 picked are "long" (34-53 min). First question per video.

| Video | Domain | Minutes | Length | Frames | Video tokens | Text tokens | Total L |
|---|---|---|---|---|---|---|---|
| 068rdc75mHM | Film & Television | 53.3 | 8K / 16K | 46 / 112 | 5,796 / 14,112 | 2,162 | 7,958 / 16,274 |
| 0Jbc3Ah4EIc | Knowledge | 39.8 | 8K / 16K | 48 / 112 | 6,048 / 14,112 | 2,144 | 8,192 / 16,256 |
| 0bwyCF0HteE | Artistic Performance | 34.0 | 8K / 16K | 48 / 112 | 6,048 / 14,112 | 2,111 | 8,159 / 16,223 |
| 0iED5BojTMI | Sports Competition | 38.2 | 8K / 16K | 50 / 116 | 6,300 / 14,616 | 1,665 | 7,965 / 16,281 |
| 0kRsiSdDFYg | Life Record | 42.2 | 8K / 16K | 48 / 112 | 6,048 / 14,112 | 2,132 | 8,180 / 16,244 |

## Correctness check (`check.json`)

Our function with skipping off vs standard attention (sdpa) in the language model, same inputs:
- Logits of the answer letters A-D at the last position differ by at most 0.25 on all 10 prompts (bf16
  resolution at these values is 0.125). The largest difference over the full vocabulary there is 0.125-0.25.
- Top token at each of the last 512 positions of one 8K prompt: same at 503 of 512. All 9 disagreements are
  near-ties: standard attention's top two tokens differ by 0.125 or less there (3 exact ties).
- On that prompt the chosen answer flipped (ours A, sdpa C) because A and C were exactly tied (26.875) in ours.
- Dense answers (sdpa): 4 of 10 correct (chance is 2.5 of 10). Not a goal of this step; recorded in `prompts.csv`.

## Main result (`region_table.csv`)

Pooled over all 10 prompts, each prompt at its own lambda that skips 50% (or 75%) of all its tiles.
"Tiles" counts (layer, head, query tile, key tile) causal tiles.

| Region | Tiles | Share of all tiles | % skipped at 50% overall | Range over the 10 prompts | % skipped at 75% overall |
|---|---|---|---|---|---|
| V->V | 28,709,296 | 71.57% | **46.87** | 42.6 to 48.3 | 75.17 |
| T->V | 4,686,752 | 11.68% | **83.95** | 81.0 to 87.8 | 96.62 |
| T->T | 1,022,336 | 2.55% | 31.20 | 28.0 to 36.0 | 48.34 |
| V->T | 4,948,608 | 12.34% | 38.62 | 30.8 to 45.0 | 57.33 |
| mixed | 748,720 | 1.87% | 57.80 | 50.4 to 61.2 | 82.16 |
| all | 40,115,712 | 100% | 50.0 | | 74.9 (one 8K prompt reaches only 73.1) |

By length (pooled over 5 prompts each):

| Region | 8K share | 8K % skipped at 50% | 8K at 75% | 16K share | 16K % skipped at 50% | 16K at 75% |
|---|---|---|---|---|---|---|
| V->V | 54.95% | 44.27 | 74.67 | 75.74% | 47.34 | 75.26 |
| T->V | 17.31% | 84.92 | 97.38 | 10.27% | 83.54 | 96.30 |
| T->T | 6.35% | 32.64 | 51.22 | 1.59% | 29.75 | 45.45 |
| V->T | 18.30% | 39.48 | 60.32 | 10.84% | 38.25 | 56.06 |
| mixed | 3.10% | 53.84 | 79.26 | 1.56% | 59.78 | 83.60 |

**Read-out (plan rule: compare V->V with T->V at 50% overall; under ~10 points = weak, over ~20 = alive):**
V->V 46.87% vs T->V 83.95%, a gap of **37.1 points** pooled and 34.5 to 44.3 points on each of the 10 prompts.
The idea is alive. The sign is reversed from the hunch: T->V is the *most* skipped region, not the least.

## Is it really text vs video? Three checks

**1. Same distance from the diagonal** (`distance_table.csv`). Text rows sit after the whole video, so every
T->V tile is off the diagonal, while V->V includes the diagonal tiles, where attention is always dense. Comparing
tiles the same number of blocks behind the query (% skipped at 50% overall):

| Distance (query block - key block) | 8K T->V | 8K V->V | 16K T->V | 16K V->V |
|---|---|---|---|---|
| 0 (diagonal) | none exist | 15.0 | none exist | 12.9 |
| 1-3 blocks (128-511 tokens back) | 85.5 | 31.4 | 83.8 | 29.1 |
| 4-15 blocks (512-2,047 tokens) | 85.8 | 45.2 | 81.6 | 41.0 |
| 16-47 blocks (2,048-6,143 tokens) | 85.0 | 49.7 | 83.9 | 47.8 |
| 48+ blocks (6,144+ tokens) | 82.1 | (1 tile pair only) | 83.7 | 55.6 |

At every matched distance T->V skips 28 to 55 points more than V->V. Distance alone does not explain the gap.

**2. The question itself** (`t2v_question_split.csv`). Most text after the video is transcript; the question and
options are the last 50-103 tokens. Splitting T->V by query block: blocks that hold any question tokens skip
**86.02%** of their video tiles (746,544 of 867,888), transcript-only blocks after the video skip 83.48%
(3,187,901 of 3,818,864). The question looks at the video no more than the transcript does.

**3. Per layer** (`per_layer.csv`, `per_layer.png`). T->V is the most skipped region in 26 of 28 layers (all
but layers 0 and 1, where V->V is higher: 77.0% vs 59.5% at 8K, layer 0). Over layers 2-27 the T->V minus V->V
gap has a median of 43.5 points at 8K and 40.2 at 16K, but varies a lot by layer (smallest 10.2 and 4.0). In layers 0-3, T->T and V->T are
almost never skipped (0.0-2.7%). In layers 4-27: T->V median 92% (lowest layer 70-71%), V->V median 46-51%
(range 12-78%), V->T median 44-46%, T->T median 35-38%.

## Why: the margins (`region_table.csv`)

Margin = how far a tile's strongest score sits below the row's running max, for the row closest to it
(natural log units; exp(margin) is that weight ratio). Pooled over the 10 prompts (tile-weighted average of
per-prompt percentiles):

| Region | 10th percentile | Median | 90th percentile | Tiles with margin exactly 0 (never skippable) |
|---|---|---|---|---|
| V->V | -5.46 | -2.04 | 0.00 | 16.6% |
| T->V | -8.56 | **-4.85** | -1.58 | **1.7%** |
| T->T | -5.03 | -0.30 | 0.00 | 47.0% |
| V->T | -5.31 | -1.16 | 0.00 | 37.0% |
| mixed | -5.78 | -2.60 | -0.02 | 13.7% |

A typical T->V tile's strongest key gets at most exp(-4.85) = 0.8% of the weight of the strongest key the row
has already seen; a typical V->V tile gets 13%. A margin of exactly 0 means the tile holds a new running max for
at least one of its rows, so no lambda <= 1 can skip it: that is true of 47% of T->T tiles and 37% of V->T tiles
(text keys, including the first block of the prompt), but only 1.7% of T->V tiles. In plain words: the text after
the video almost never finds a video key that beats what it already saw in the text before the video.

## Side findings

- **This VLM needs a much larger lambda than Llama text.** Lambda that skips 50% overall: 8K median 0.151 (range
  0.123-0.225), 16K median 0.097 (0.080-0.145). In BLASST's calibrated form a = lambda x L: 979-1,795 at 8K and
  1,307-2,362 at 16K. The Llama-3.1-8B baseline needed a = 106.5 for 50% (lambda = 0.0138 at 8K). So Llama's
  calibration cannot be reused, and a = lambda x L is not constant here (median 1,236 at 8K vs 1,577 at 16K).
- **75% overall is close to the limit.** Lambda for 75%: 0.42-0.64 at 16K, 0.76-1.0 at 8K; one 8K prompt reaches
  only 73.1% even at lambda = 1. BLASST's per-key guarantee (a skipped key's weight is below lambda) is weak at
  these values.

## What it means for step 2 (arithmetic, not a result)

T->V is 11.7% of tiles (17.3% at 8K, 10.3% at 16K). Even if T->V were never skipped, the other regions would
need to skip 56.6% of their tiles to keep 50% overall (60.5% at 8K, 55.7% at 16K), vs 46.9% for V->V today.
So protecting T->V is cheap in % skipped. Step 3 must test both directions at the same overall % skipped:
(a) protect T->V (smaller lambda there) and skip more V->V; (b) one lambda as today.

## Caveats

- 5 videos from one Video-MME chunk, all long, one question each, 2 lengths: 10 prompts. Every prompt shows the
  same pattern, but they share one layout and one model.
- Measured on a dense pass. With skipping on, later layers see slightly different inputs; Fig. 1 of the baseline
  used the same simplification, as does NVIDIA's calibration.
- The region shares depend on the prompt layout we chose (about 1,024 transcript tokens before the video,
  512-1,024 after). Without a transcript, T->T and V->T would almost vanish.
- Low frame count for 34-53 minute videos (46-116 frames, one every 18-70 s) at 336 x 588; dense accuracy is
  4 of 10. Fine for measuring attention, not for judging the model.
- Skip decisions are per query head, as in the baseline. A real kernel shares K/V loads across the 7 query heads of
  a GQA group, so byte savings would be lower.
- "Mixed" tiles (1.9%) are the block that holds the end of the video; reported but not interpreted.

## Cost

GPU: 2 jobs on `ghx4-interactive` (double charge): margins for all 10 prompts 1 min 55 s, correctness check
1 min 51 s. **0.13 charged GPU-hours.** Prefill time with margin recording: 2.8-3.6 s per 8K prompt, 8.5-11.9 s per
16K prompt; peak GPU memory 18.4 GB. Data: model 16 GB, one Video-MME chunk 5 GB, subtitles 8 MB.

## Files and how to re-run

In `/work/nvme/beqg/zyang21/blasst/`:

| File | What |
|---|---|
| `vlm_prepare.py` | CPU: picks the 5 videos, decodes frames with PyAV, builds the 10 prompts -> `vlm_inputs/` |
| `vlm_regions.py` | GPU: one dense prefill per prompt, margins -> `vlm_margins/<video>_<8k|16k>.pt` (fp16 `[28, 28, q tiles, k tiles]`, plus `is_vision`, `input_ids`, prompt facts; 153 MB total) |
| `vlm_check.py` | GPU: our function vs standard attention -> `check.json` |
| `vlm_analyze.py` | CPU: everything in this folder: `region_table.csv`, `per_layer.csv`, `distance_table.csv`, `t2v_question_split.csv`, `curves.csv`, `prompts.csv`, `region_curves.png`, `per_layer.png` |

```bash
source /work/nvme/beqg/zyang21/blasst/env.sh        # also cd's into blasst/
pip install "qwen-vl-utils==0.0.11" av              # once (already done)
python vlm_prepare.py                               # CPU, about 45 s
srun -A beqg-dtai-gh -p ghx4-interactive --gpus=1 --cpus-per-task=8 --mem=64g -t 00:30:00 \
     bash -c 'source /work/nvme/beqg/zyang21/blasst/env.sh; python vlm_regions.py'   # about 2 min
python vlm_analyze.py                               # CPU, about 15 s
```

Video decoding note: `decord` has no ARM build and this torchvision has no `read_video`, so `vlm_prepare.py`
decodes with PyAV (`pip install av`) and passes the frames to the Qwen processor directly.
