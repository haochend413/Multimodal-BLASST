<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Code

Historical experiment scripts from DeltaAI (NCSA, NVIDIA GH200 96 GB, ARM / aarch64 CPU).
The Nov 21 extension adds decode BLASST, ModelOpt parity checks and an attention-data collector.
See [Attention Lab](../attention-lab/README.md) for the new workflow; the historical reports and results
remain the record of the original runs.

## Before you run anything

- **Keep all files in one folder.** The scripts import each other by name (`import blasst_attn`,
  `import blasst_regions`, `import common`), and they expect to run from the folder that also holds `results/`.
  On DeltaAI that folder was `/work/nvme/beqg/zyang21/blasst/`.
- **Hard-coded paths.** These files set that path near the top; change it to your own folder: `common.py` (`W`),
  `fig1_analyze.py` and `fig2_analyze.py` (`R`), `tile_sensitivity.py`, `vlm_prepare.py`, `vlm_regions.py`,
  `vlm_check.py`, `vlm_analyze.py`, `vlm2_weights.py`, `vlm2_pilot.py`, `vlm2_analyze.py` (`W` / `OUT`),
  every `.sbatch` file, and `env.sh`.
- **Slurm account.** The `.sbatch` files and the `srun` lines use Kevin's account `-A beqg-dtai-gh`, a research
  group's allocation; use your own.
- **Outputs.** The scripts write to `results/` (baseline) and `results/vlm_regions/` (VLM). In this package those
  outputs are in `../results/`, with the write-ups and PNG figures moved to `../reports/`.

## The two core files

**`blasst_attn.py`**: the BLASST attention function, registered with Hugging Face transformers as
`attn_implementation="blasst"`.
- Prefill: takes 128 query rows at a time, computes their scores against all keys, applies the causal mask,
  splits the keys into 128-wide tiles, walks the tiles left to right keeping each row's running max, and skips a
  tile only if every row is below its running max by more than ln(lambda). Skipped tiles are set to -inf before a
  normal softmax and P.V.
- It is a **simulation**: QK^T is computed for every tile (as in the real BLASST kernel), and the softmax / P.V
  work is not actually saved either. It reports what would be skipped and returns what a skipping kernel would.
- Decode (one query token): BLASST on 1 × 128 cached-key tiles, inheriting the prefill threshold.
  `set_decode_threshold` selects an independent threshold or restores the original dense SDPA path.
- Threshold: `set_threshold(lam=...)` (fixed), `set_threshold(a=...)` (calibrated lambda = a / L), or
  `set_threshold(None)` (no skipping).
- Recording: with `STATE["record"] = True` it saves each tile's **margin** (max over the 128 rows of
  tile max - running max). A tile is skipped at lambda iff margin < ln(lambda), so one dense pass gives the skip
  set for every lambda (`masks_from_margins`). `STATE["dump_qk_layers"]` also saves Q and K.

The new `attention_study.py`, `prepare_videomme_study.py`, `prepare_vqav2_study.py`,
`compare_decode_modelopt.py`,
and their tests are documented
in [Attention Lab](../attention-lab/README.md).

### Causal score distributions around the skip cutoff

Collect compact histograms for the 30 VQA cases without exporting full score matrices:

```bash
python attention_study.py \
  --manifest "$HOME/blasst-results/vqav2-30.json" --sample-count 30 \
  --layers 0,14,27 --heads 0,4 --decode-tokens 2 \
  --score-distributions --study-lambda 0.1 \
  --output "$HOME/blasst-results/vqav2-30-score-distributions.json"
```

The score excess is each finite score minus its own row's cutoff, which is the
causal running max through the current tile plus `ln(lambda)`. Zero or greater
reaches the cutoff. The report records raw logits, score and row-max excess,
tile-max excess, and per-tile fractions of valid rows or scores reaching the
cutoff. Pure physical tiles and token-pair scores inside mixed tiles have
separate summaries. Existing calibrated thresholds can be used with
`--calibration FILE --target-keep 50` in place of `--study-lambda`.

To combine score-level analysis from a saved causal score companion with a
larger set of saved tile margins and create a figure, install Matplotlib locally:

```bash
python score_distribution_study.py \
  --score-export ../results/vqav2-image0-question0.json \
  --tile-report ../results/vqav2-30-row-keys.json \
  --output /tmp/vqav2-distributions.json --plot /tmp/vqav2-prefill.png
```

Use `python plot_score_distributions.py REPORT.json --phase decode --output DECODE.png`
to plot decode, or omit `--phase decode` to plot prefill from a newly collected report.

### Previous-layer T→V votes for V→V queries

Run a paired comparison against standard BLASST on the same VQA cases:

```bash
python attention_study.py \
  --manifest "$HOME/blasst-results/vqav2-30.json" --sample-count 30 \
  --lambda 0.1 --decode-tokens 16 \
  --tv-vv-sweep 0.25,0.5,0.75 \
  --output "$HOME/blasst-results/vqav2-tv-guided-vv.json"
```

Layer 0 uses standard causal BLASST. Each eligible text query tile and head can cast one
keep vote for a pure visual key tile. Mixed query tiles contribute votes from their text
rows only, using each row's original causal running max. At layer L, the keep fraction
from layer L-1 gates the pure V→V tiles in each visual query tile. Accepted queries can
read all causal pure visual keys. Rejected queries omit those key columns from the QK
matmul. Visual tokens remain available as keys to other queries. The policy keeps queries
with no eligible text votes and protects a first key tile when an entirely visual
prefix would otherwise leave the rows without any key. The mask is recomputed at every
layer. Decode keeps its regular attention policy.

Load the sweep JSON in **T→V-guided V→V** to inspect votes, reactivation, bypasses, and
generated answers. Answer comparison uses normalized full-response exact match against
the available references, and the report also counts changes relative to the baseline.
These are tile workload counts in the PyTorch reference. For a single-run tile and
score export, use `--tv-vv-threshold 0.5 --score-data` in place of the sweep flag.

For a paired VQA v2 accuracy comparison, run the baseline and one selected threshold
on the same validation questions:

```bash
python attention_study.py \
  --manifest "$HOME/blasst-results/vqav2-30.json" --sample-count 30 \
  --lambda 0.1 --decode-tokens 16 --tv-vv-sweep 0.5 --vqa-accuracy \
  --output "$HOME/blasst-results/vqav2-tv-guided-vv-accuracy.json"
```

Both policies receive the original VQA question followed by “Answer with a single word
or short phrase.” Greedy generation stops at EOS or the token cap. The official
[VQA evaluator](https://visualqa.org/evaluation.html) scores the decoded answer
against the ten references in the manifest. The JSON includes per-question scores,
aggregate accuracy, and the paired change from standard BLASST. Each policy also
writes a sibling `*-predictions.json` in the official question-ID/answer format.
The score covers the selected validation subset.

#### Full VQA v2 validation

The full validation split contains 214,354 questions. The same paired comparison
runs 428,708 generations. From `CS498_BLASST_project 2`, prepare the manifest
with a shared data root accessible to the GPU workers:

```bash
VQA_ROOT="$HOME/vqav2"
RESULTS="$HOME/blasst-results"
mkdir -p "$VQA_ROOT"
# Skip these two commands if "$VQA_ROOT/val2014" already contains the COCO images.
curl -fL --retry 3 -o "$VQA_ROOT/val2014.zip" \
  https://s3.amazonaws.com/images.cocodataset.org/zips/val2014.zip
unzip -n "$VQA_ROOT/val2014.zip" -d "$VQA_ROOT"

python code/prepare_vqav2_study.py --data-root "$VQA_ROOT" --all-validation --download \
  --output "$RESULTS/vqav2-full-val.json"
```

For a measured throughput pilot use the VQA accuracy command above with the
full manifest and `--vqa-shards 215 --vqa-shard-index 0` in place of the sample count.
That runs about 1,000 paired questions. Multiply seconds per paired question
by 214,354 and divide by 3,600 to estimate GPU-hours. Submit one GPU per shard.
Set `DELTAAI_ACCOUNT` and `DELTAAI_PARTITION` for your allocation and choose
time and concurrency from the pilot. This example uses 430 shards and up to
eight concurrent workers, with the Python environment already activated:

```bash
mkdir -p "$RESULTS/vqa-full-shards"
sbatch --account="$DELTAAI_ACCOUNT" --partition="$DELTAAI_PARTITION" \
  --gpus=1 --cpus-per-task=8 --mem=32G --time=02:00:00 --array=0-429%8 \
  code/vqa_full_shard.sh "$RESULTS/vqav2-full-val.json" "$RESULTS/vqa-full-shards" 430
```

The wrapper runs standard BLASST and relevance 0.5 at lambda 0.1. Each shard
checkpoints every 25 complete pairs and automatically resumes when submitted again
with the same arguments and output path. Give concurrent workers distinct shard IDs.
Once all shards finish, merge the results:

```bash
python code/vqa_shards.py "$RESULTS"/vqa-full-shards/shard-*.json \
  --output "$RESULTS/vqav2-full-accuracy.json"
```

The merge checks for missing or duplicate shards, duplicate questions, and
consistent settings. It writes the aggregate accuracy and bypass report plus
one official-format predictions file per policy. Load the merged summary or a
compact individual shard report in **T→V-guided V→V**.

### Bidirectional masking for pure V→V tiles

To test bidirectional V→V attention on one manifest case:

```bash
python attention_study.py \
  --manifest "$HOME/blasst-results/vqav2-30.json" --sample-index 0 \
  --layers 0,14,27 --heads 0,4 --decode-tokens 2 \
  --vv-prefix-attention --score-data \
  --output "$HOME/blasst-results/vqav2-vv-prefix.json"
```

In prefill, every tile with entirely visual query rows and key columns gets full
attention scores, including above-diagonal tiles. A tile with mixed
query rows or mixed key columns stays causal, as do T→T and cross-modal tiles.
The physical tile grid, token positions, and joint row softmax are preserved.
The flag applies to all language-model layers and heads. The layer and head options
select which ones are recorded. Decode uses its usual policy with the resulting cache.
The default run has live skipping off. Use the Attention Lab lambda slider to
compare offline skip decisions, or add `--lambda 0.1` to apply BLASST to every
eligible tile during generation, including pure V→V tiles.

For a causal baseline, repeat the command without `--vv-prefix-attention` and use
a different output filename. Load the JSON and score companion for each run in
Attention Lab to compare the masks and inspect individual scores.
For aggregate decision comparisons across a manifest, use `--vv-prefix-attention`
with `--tile-layer-study` and the desired sample range in place of `--score-data`.
The saved calibration config can be used to hold lambda fixed across mask policies.

### Cross-layer tile decisions

Collect full-score BLASST margins across every language-model layer, using the original
physical tiles and a shared running max:

```bash
python attention_study.py \
  --manifest "$HOME/blasst-results/vqav2-30.json" --sample-count 30 \
  --layers all --heads 0,4 --decode-tokens 2 \
  --tile-layer-study --study-lambda 0.1 \
  --output "$HOME/blasst-results/vqav2-30-tile-layers.json"
```

The model follows a dense trajectory. The offline decision is skip when the recorded margin
is below `ln(study_lambda)`. Each trajectory holds the case, phase, decode step, query head,
and query/key token ranges fixed. A whole physical tile is labeled from its valid lanes.
The mixed group retains its own votes. Decode has text queries only.

Load the JSON into **Cross-layer tiles** in Attention Lab. The chart pools by actual layer
distance, while the table lists the specific layer pairs and all four keep/skip transitions.
An earlier skip that turns into a keep is a disagreement that would matter for mask reuse.
Compare skip persistence with the later layer's own skip rate, since a region that almost
always skips can have high persistence even without useful layer-to-layer prediction.
The case selector shows a selected tile's margin and decision at every recorded layer.

For saved dense Attention Lab files, recompute the study without running the model:

```bash
python tile_layer_study.py ../results/attention-[0-5].json \
  --study-lambda 0.1 --output /tmp/blasst-layer-videomme-pilot.json
```

These six Video-MME prompts cover two videos and seven sampled layers. Distance 1 pools
0→1, 1→2, 13→14, and 26→27. Other distances can involve different layer pairs, so use the
all-layer run for a fuller distance comparison. For a CPU smoke test of the collector,
replace the manifest flags with `--demo`.

To track the indices that determine each margin, add `--study-max-indices` and use a
new output name. It records the query with the largest tile margin, the best key for
that query inside the tile, and its best key across tiles visited so far, including
the current tile. Indices are zero-based token positions. Exact ties use the first
query and key within a tile and the latest tile for ties in the running max.

The report measures which indices match across layer pairs. It also replays the first
recorded layer's indices using the current layer's scores at those two positions.
This estimates the skip decision from two dot products. The false-skip rate divides
by fresh full-score BLASST keeps, and the missed-skip rate divides by fresh skips.
Actual attention on kept tiles still uses the usual computation.

The **per-row key replay** uses one key per valid query row per tile. It takes each
row's winning key from the preceding model layer, reads the target layer's score at
that same query/key position, and rebuilds the running max from those candidate
scores across key tiles. Both the key-position match rate and max-score hit rate are
reported. The latter includes a different key that ties the current maximum.
Each adjacent layer pair starts with freshly computed source maxima. These are
offline replay results measured against full-score BLASST on the same trajectory.

If a dense export already has its matching `.scores.bin` file, reconstruct the same
statistics locally without rerunning the model:

```bash
OMP_NUM_THREADS=1 python tile_layer_study.py ../results/vqav2-image0-question0.json \
  --study-max-indices --output /tmp/blasst-row-key-vqa-pilot.json
```

#### Saved keep-rate calibration

From a saved cross-layer report, fit the 20%, 50%, and 80% keep settings once.
This example uses the first 10 cases for calibration and the remaining 20 for evaluation:

```bash
python tile_layer_study.py ../results/vqav2-30-row-keys.json \
  --calibrate --sample-count 10 --output ../results/vqav2-keep-calibration.json

for target in 20 50 80; do
  python tile_layer_study.py ../results/vqav2-30-row-keys.json \
    --sample-index 10 --sample-count 20 \
    --calibration ../results/vqav2-keep-calibration.json --target-keep "$target" \
    --output "/tmp/vqav2-keep-${target}-eval.json"
done
```

Thresholds are fitted separately for prefill and decode, pooling tile margins across
the calibration cases, layers, and heads in each phase. Use `--calibration-scope global`
to fit one threshold per target shared by both phases. The saved config includes the
calibration case IDs, achieved keep counts, and minimum attainable keep rate. Zero-margin
tiles are kept even at lambda 1, so some target rates cannot be reached. Each evaluation
reports achieved keep rates overall, by phase and tile group, and per case.

Both commands above work offline with saved margins. To apply the config while collecting
new cases, pass `--calibration /path/to/vqav2-keep-calibration.json --target-keep 50` to
`attention_study.py --tile-layer-study` in place of `--study-lambda`. Evaluation
cases use the saved cutoffs for every layer. Existing row-key replay records, when present,
are rescored at those cutoffs as well. Load each evaluation JSON in **Cross-layer tiles**.

### JL precheck for BLASST

To compare modality packing and resetting the BLASST bar at key-region boundaries,
run the paired region audit:

```bash
python attention_study.py \
  --manifest "$HOME/blasst-results/vqav2-30.json" --sample-count 30 \
  --layers 0,14,27 --heads 0,4 --decode-tokens 2 \
  --jl-region-audit --jl-dim 32 --jl-lambda 0.1 --jl-seed 0 \
  --output "$HOME/blasst-results/jl-vqav2-30-regions.json"
```

For both prefill and decode, the report contains `original_shared`,
`original_reset`, `packed_shared`, and `packed_reset`. The packed layout starts each contiguous
text or visual run at a tile boundary on both axes and masks the remaining lanes
of each run's final tile. It preserves sequence order, positions, and causal masks.
Both reset policies restart the running max independently for each query row on each
contiguous key run. `original_reset` reduces the key runs inside boundary tiles and
emits one decision per original physical tile. With text→visual→text keys, the
last text run gets a fresh bar. The first valid tile in a run starts with a zero
gap; its actual score sets the bar even when negative.

Compare `original_shared` with `original_reset` to isolate the reset on the original
tiles. `false_skips` and `missed_skips` use full BLASST under the same policy as the
reference. `false_skips_vs_shared_full` and `missed_skips_vs_shared_full` keep the full-score
reference fixed to the shared max for that layout. The full and projected paths
use their own score maxima under each policy and share the same underlying model
inputs. Score weight is measured against global dense softmax over all valid keys.
This audit measures masks and attention weight on a dense trajectory.

For a paired dimension ablation and gap distributions, run a dense-trajectory
sweep on the original tiles:

```bash
python attention_study.py \
  --manifest "$HOME/blasst-results/vqav2-30.json" --sample-count 30 \
  --layers 0,14,27 --heads 0,4 --decode-tokens 2 \
  --jl-sweep 8,16,32,64,96 --jl-lambda 0.1 --jl-seed 0 \
  --output "$HOME/blasst-results/jl-vqav2-30-sweep.json"
```

Load the resulting JSON in Attention Lab. Use **Case** in the **JL ablation** view
to switch between all pairs and an individual manifest entry. The report includes
both the pooled counts and each case's histograms and skip counts. The view overlays full and
projected gap histograms for text→text, text→visual, visual→text and visual→visual,
with prefill and decode selectable. A gap is the row's running max through the current
tile minus its best key of the selected modality inside that tile. Mixed tiles contribute
observations to the appropriate regions using the same all-key running max. Counts use
one-logit bins and include an overflow bucket for gaps of 80 or greater. A whole-tile
vote is counted once, under a pure modality pair or under mixed.

The sweep uses nested prefixes of one seeded random orthogonal basis to compare
dimensions on the same Q/K inputs. This basis differs from the earlier single-width
audit's basis. The false-skip rate divides by tiles full BLASST keeps, and the
missed-skip rate divides by tiles full BLASST skips. To smoke-test without model
weights, use `--demo --jl-sweep 4,8,16,24` in place of the manifest, sample count,
layer and head flags.

Use `--jl-projected-only` with `--jl-dim` to test whether projected scores alone reproduce
full BLASST's skip mask. Both passes use the same inputs, tile order, masks, and
`--jl-lambda`. Each pass builds its own running max. The projected score uses the
original score scaling multiplied by `d/m`, where `d` is the head width and `m`
the sketch width. The full pass is the reference and supplies model generation.
There is no bound or full-score fallback in the projected skip decision.

Read `metadata.jl_projected.phases` for full and projected skip counts, matched
skips, false skips (only projected skips), and missed skips (only full skips). Counts
are broken down by phase and tile modality. Each exported tile includes
`projected_margin`, `projected_skipped`, and `full_skipped_at_jl_lambda`.

For multiple image/question pairs, prepare a VQA v2 manifest using
`prepare_vqav2_study.py --images 30 --questions-per-image 1`. Add `--sample-count 30`
to the projected-only audit command, keeping the same `--sample-index` as the start of
the requested manifest range. One process reuses the model for all pairs. The output JSON
contains pooled tile counts and the number of pairs with false skips, broken down by phase
and modality. Individual Attention Lab audits are in a sibling directory named
`<output-stem>-samples`. Use a different output filename for a different projection width
or seed.

Use `--jl-split-tiles both` to break each mixed prefill attention tile at text and
visual boundaries on either axis. Each resulting tile has homogeneous query rows
and homogeneous key columns. The split totals count each resulting tile with at least
one valid query–key pair once. Token positions, key order and causal masks are
preserved. Reducing real lanes gives the same decisions as padding to 128 with
masked score entries.

The `query` and `key` modes split only the specified side. The
`query-on-visual-keys` mode splits mixed query rows only against entirely visual
key columns. This isolates the effect of splitting the query vote in that region.

The same run reports standard tiles under `phases.prefill` and the split layout
under `split_prefill` in the batch report. For a single sample, both are under
`metadata.jl_projected` and prefill views include `split_tiles`. Splitting changes
the number of tile decisions, so compare the share of dense weight wrongly dropped
(`false_skip_dense_weight / dense_weight`) and tile counts together. Full and
projected BLASST are replayed independently on each layout. The split path is an
audit of skip decisions and does not change model generation or measure kernel speed.

Start with a CPU smoke test from this directory. The demo has 32-dimensional heads.

```bash
python attention_study.py --demo --decode-tokens 1 --jl-dim 8 \
  --jl-lambda 0.1 --output /tmp/jl-demo.json
```

For an image on CUDA, use the same collector and select a few layers and query heads.
The command also accepts `--video` or an existing VQA v2/Video-MME `--manifest`.

```bash
python attention_study.py --image /absolute/path/to/image.jpg \
  --question "What is shown in the image?" --layers 0,14,27 --heads 0,4 \
  --decode-tokens 2 --jl-dim 32 --jl-lambda 0.1 --output /tmp/jl-image-audit.json

jq '.metadata.jl_precheck.phases' /tmp/jl-image-audit.json
```

Each row compares its maximum projected upper score against its exact running max plus
`ln(lambda)`. Every eligible row must pass before the tile skips full QK. Otherwise the
ordinary BLASST rule decides whether to use V.

- **Default residual bound.** For the random orthonormal basis `P`, use
  `(Pᵀq)·(Pᵀk) + ||q − PPᵀq|| ||k − PPᵀk||`, scaled as an attention logit.
  Orthogonal decomposition and Cauchy–Schwarz give this upper bound in exact arithmetic.
- **JL margin.** Add `--jl-epsilon 0.5` to try
  `(d/m)(Pᵀq)·(Pᵀk) + epsilon ||q|| ||k||`, where `m` is the sketch dimension and
  `d` the head dimension. This tunable margin can miss. Compare seeds with `--jl-seed`
  and inspect `upper_violations`, `false_early_skips`, and `skip_mask_disagreements`.

Audit mode uses exact scores to verify decisions on the same Q/K inputs and leaves
generation unchanged. Compare `baseline_skips_caught` and `fallback_rate` for pure text,
pure visual, and mixed key tiles. The token-pair section splits even mixed tiles into
T→T, T→V, V→T, and V→V. It reports mean absolute JL logit error and pair-weighted
Q/K norms separately for prefill and decode. Compare T→T with T→V to hold query
modality fixed. Groups with zero observations are omitted.

For live generation, repeat the CUDA command with `--jl-live --lambda 0.1` and a new
output filename. Live applies JL to every head in the selected layers and ordinary
BLASST to the other layers. Match both lambda flags when comparing against a run
of ordinary BLASST. Live output contains generation metadata and skip counts. Use audit
mode for the full scores and modality breakdowns.

The PyTorch prototype builds sketches from full keys on each call, including decode.
Live early skips bypass the tile's full QK matmul and value computation after that
setup. Tile counts are per query head and do not measure kernel speed or HBM traffic.

**`blasst_regions.py`**: the region-aware version for the VLM study (`attn_implementation="blasst_regions"`;
prefill only, batch size 1).
- Same tiles, skip rule and margins as `blasst_attn.py`.
- **A lambda per region**: each 128-token block is text (0), mixed (1) or video (2), set with `set_kinds`. A tile's
  threshold is `ln_lam_table[query kind, key kind]`, a 3 x 3 table set with `set_table`; -inf means never skip.
  The same value everywhere is plain BLASST.
- **Probe mode** (`set_probe`): attention stays dense, and for each lambda in a list it measures how much of each
  query row's softmax weight sits on the tiles that lambda would skip (split into video keys and other keys).
  This is the "weight lost" of report 3.

Both are tested: `test_blasst.py` (tiny random Llama, no download; with skipping off it equals PyTorch's
standard attention, and online skips equal offline skips from margins) and `test_blasst_regions.py` (CPU; equal to
`blasst_attn.py` with one lambda, per-region decisions equal offline margins, probe equals a brute-force sum).

## Environment

```bash
conda deactivate 2>/dev/null
module reset                                   # not 'module purge': that breaks the PyTorch module
module load python/miniforge3_pytorch/2.11.0   # torch 2.11.0+cu130, Python 3.12
export W=/your/work/dir
export HF_HOME=$W/hf PIP_CACHE_DIR=$W/pip-cache TOKENIZERS_PARALLELISM=false
python -m venv --system-site-packages $W/blasst/venv     # torch comes from the module
source $W/blasst/venv/bin/activate
pip install "transformers==4.57.3" hf_transfer
pip install "qwen-vl-utils==0.0.11" av                    # VLM only; av = PyAV, decodes video on ARM (no decord build)
pip install "nvidia-modelopt[hf]==0.47.0"                 # only for crosscheck_modelopt.py
```

`env.sh` holds these lines minus the venv creation and installs. Every job runs `source env.sh` first; it also
`cd`s into `blasst/`. For Qwen2.5-VL, load with
`attn_implementation={"text_config": "blasst", "vision_config": "sdpa"}` so only the language model uses our
function (a plain string would also route the vision encoder through it and break it).

**Models (Hugging Face):** `meta-llama/Llama-3.1-8B-Instruct` (gated: request access, then `hf auth login`),
`Qwen/Qwen2.5-VL-7B-Instruct`.

**Data:**
- RULER: `self-long/RULER-llama3-1M`, folders `niah_multikey_1/2/3`, `vt`, `fwe` x `4k/8k/16k/32k`
  (`snapshot_download(..., repo_type="dataset", allow_patterns=[...], local_dir=$W/blasst/data/ruler)`).
  Prompts already have the Llama-3 chat template applied.
- Video-MME: `lmms-lab/Video-MME` into `$W/data/videomme`: `videomme/test-00000-of-00001.parquet`, `subtitle.zip`
  (unzip it there) and `videos_chunked_01.zip`; batch 2 also needs `videos_chunked_02.zip` and `_03.zip`.

## Run order

GPU steps without an `.sbatch` file were run as a short interactive job:
`srun -A <account> -p ghx4-interactive --gpus=1 --cpus-per-task=8 --mem=64g -t 00:30:00 bash -c 'source $W/blasst/env.sh; python <script>'`
(`ghx4-interactive` is charged at 2x; real runs used partition `ghx4`.)

**Baseline (Llama-3.1-8B, RULER)** -> `reports/1_baseline_llama.md`
1. `test_blasst.py` (GPU, about 25 s; `--cpu` also works): correctness tests.
2. `smoke_test.py` (GPU): one 4K RULER sample, dense vs ours.
3. `sbatch fig1.sbatch` (about 21 min): margins for 4K-32K samples, plus saved Q/K.
4. `python fig1_analyze.py` (CPU): Fig. 1 and `calibration.json` (the lambda = a / L fit). Step 7 needs it.
5. Optional: `python crosscheck_modelopt.py niah_multikey_2:4k:0 fwe:4k:0 vt:8k:0` (GPU): vs NVIDIA ModelOpt.
6. Optional: `python tile_sensitivity.py results/qk/8k/*.pt` (GPU): % skipped vs tile shape.
7. RULER accuracy, `sbatch --job-name=<name> fig2.sbatch <length> <config...>`: `8k dense`; `8k cal50 cal75 cal90`;
   `16k dense`; `16k cal50`, `16k cal75`, `16k cal90` (one job each).
8. `python fig2_analyze.py` (CPU): Fig. 2 and its tables.

**VLM step 1 (Qwen2.5-VL-7B, 5 long Video-MME videos, 8K and 16K)** -> `reports/2_vlm_regions_step1.md`
1. `python vlm_prepare.py` (CPU, about 45 s): builds the 10 prompts into `vlm_inputs/`.
2. `vlm_regions.py` (GPU, about 2 min): dense prefills recording margins into `vlm_margins/`.
3. `vlm_check.py` (GPU): our function with skipping off vs standard attention.
4. `python vlm_analyze.py` (CPU, about 15 s): every step-1 table and figure.

**VLM batch 2** -> `reports/3_vlm_regions_batch2.md`
1. `python test_blasst_regions.py` (CPU, seconds).
2. `python vlm2_prepare.py` (CPU, about 2 min): layouts B and C plus the 30-video pilot prompts, into `vlm2_inputs/`.
3. `sbatch vlm2_weights.sbatch` (about 13 min): items 1-2, per-row weight arrays into `vlm2_tensors/`.
4. `sbatch vlm2_pilot.sbatch` (about 55 min): item 3, the accuracy pilot; resumes from `v2_pilot.jsonl`.
5. `python vlm2_analyze.py` (CPU, about 2 min): every `v2_*` file.

## Files

| File | What it does |
|---|---|
| `env.sh` | Environment setup; `source` it in every shell and Slurm job. |
| `blasst_attn.py` | The BLASST attention function (see above). |
| `test_blasst.py` | Tests for `blasst_attn.py` on a tiny random Llama. |
| `common.py` | Shared helpers for the Llama runs: load model, load RULER rows, encode, greedy decoding, RULER scoring, margins to % skipped. |
| `smoke_test.py` | One RULER sample: dense vs ours with skipping off (logits, answer, time), plus % skipped at a few lambdas. |
| `fig1_sweep.py` | Dense prefills that record margins: % skipped vs lambda per sample at 4K-32K; also saves Q/K for a few samples. |
| `fig1.sbatch` | Slurm job for `fig1_sweep.py`. |
| `fig1_analyze.py` | CPU: Fig. 1 figures and tables, and the calibration fit (`calibration.json`). |
| `crosscheck_modelopt.py` | Our % skipped vs NVIDIA ModelOpt's on the same samples. |
| `tile_sensitivity.py` | % skipped for other tile shapes, from saved Q/K. |
| `fig2_ruler.py` | RULER accuracy with skipping on during prefill: dense vs calibrated 50 / 75 / 90%. |
| `fig2.sbatch` | Slurm job for `fig2_ruler.py`; arguments are the length and the configs. |
| `fig2_analyze.py` | CPU: Fig. 2 table, answer flips vs dense, figure. |
| `vlm_prepare.py` | Step 1, CPU: picks 5 long Video-MME videos, decodes frames with PyAV, builds 10 prompts (layout A). |
| `vlm_regions.py` | Step 1, GPU: one dense Qwen2.5-VL prefill per prompt, recording margins (language model only). |
| `vlm_check.py` | Step 1, GPU: our function with skipping off vs standard attention on Qwen2.5-VL. |
| `vlm_analyze.py` | Step 1, CPU: % skipped per region at each prompt's 50% / 75% lambda; distance, question and per-layer checks. |
| `blasst_regions.py` | Region-aware attention: a lambda per region, plus the probe for weight lost (see above). |
| `test_blasst_regions.py` | CPU tests for `blasst_regions.py`. |
| `vlm2_prepare.py` | Batch 2, CPU: layouts B and C for the 5 step-1 videos, plus the pilot prompts (30 short and medium videos, 3 questions each). |
| `vlm2_weights.py` | Batch 2 items 1-2, GPU: dense passes with the probe; Qwen layouts A / B / C and Llama on RULER. |
| `vlm2_weights.sbatch` | Slurm job for `vlm2_weights.py`. |
| `vlm2_pilot.py` | Batch 2 item 3, GPU: Video-MME accuracy, dense vs one lambda vs question->video never skipped. |
| `vlm2_pilot.sbatch` | Slurm job for `vlm2_pilot.py`. |
| `vlm2_analyze.py` | Batch 2, CPU: every `v2_*` table and figure. |
