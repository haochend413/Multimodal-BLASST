<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Attention Lab

Interactive attention study for Qwen2.5-VL video prompts, plus BLASST decode parity checks.
The historical reports describe the original prefill-only runs. This extension does not
replace those measurements or claim a GPU speedup.

## Website

Hosted privately at https://blasst-attention-lab.resilientbat1.chatgpt.site after deployment.
For local use, serve this folder's `dist` directory with `python -m http.server 8765 --directory dist`.
Open http://localhost:8765. No frontend build or Node packages are needed.

Use **Load experiment** to open JSON files from the collector; files stay in browser memory.
Multiple datasets can be loaded and selected. Choose phase, layer, query head, decode step,
and heatmap metric. Region buttons highlight cells containing the selected token-pair region.
The distribution and region statistics always describe the complete selected view.
Click a block to expand its matrix of scaled QK logits. Each score shows the query-to-key
modality pair, and masked positions are labeled in the matrix. The bundled demo includes
token scores. For recorded data, use **Load experiment** to select both the JSON summary and
its matching `.scores.bin` file. Files can also be selected separately, in either order.

Open **JL ablation** to see the synthetic dimension sweep or use **Load experiment**
to open a report collected with `attention_study.py --jl-sweep`. The phase selector
updates the skip-rate chart and the four modality gap distributions. The dimension
selector changes the projected distribution overlaid on the full-score distribution.
Use **Case** to view the pooled results or one sample. Older sweep exports contain the
pooled view and need a fresh collection to include cases.
See the [JL run guide](../code/README.md#jl-precheck-for-blasst).

Load a JSON from `attention_study.py --jl-region-audit` to open **Region audit**.
Its charts compare tile decisions, JL skip rate, dropped dense attention weight,
and dense weight on false skips for original and packed tiles with a shared or
reset max. Use the phase, case, and tile-group filters
to inspect each part of the audit.

Load a JSON from `attention_study.py --tile-layer-study` or `tile_layer_study.py` to open
**Cross-layer tiles**. Compare agreement and skip persistence against layer distance,
inspect the keep/skip transitions for specific layer pairs, and choose a case, head,
and tile to display its layer-by-layer decisions and margins.
See the [cross-layer run guide](../code/README.md#cross-layer-tile-decisions).

The bundled demonstration uses seeded random Q/K tensors and a planted attention sink.
It is explicitly synthetic, not measured Qwen, Video-MME, or accuracy data.

## Install and run

From the project `code` folder, in a CUDA environment with model access:

```bash
pip install 'torch>=2.8' 'transformers==4.57.3' 'nvidia-modelopt[hf]==0.47.0' av numpy
```

ModelOpt is needed only for the parity script. The collector uses the custom Hugging Face
attention implementation. This is a reference simulation and intentionally computes all tiles.
No Slurm account, dataset root or model-cache path is hard-coded in the new scripts.

Collect a **dense trajectory** first for offline analysis:

```bash
python attention_study.py \
  --video /absolute/path/to/video.mp4 \
  --text-before 'Context or transcript before the video.' \
  --question 'What happens in the video?' \
  --frames 16 --layers 0,7,14,21,27 --heads 0,4 \
  --decode-tokens 16 --max-tokens 8192 \
  --output /absolute/path/to/dense-attention.json
```

Use text before the video to ensure V→T has meaningful text keys. Generated queries are text,
so decode contains T→T and T→V; V→T and V→V are correctly empty during text decode.
The vision encoder retains standard SDPA; the language model uses the custom attention.
The collector forces the requested decode calls even if the model emits EOS, so later steps
may include EOS continuations. It generates one additional output token because the first
output is produced by prefill; N decode steps require N subsequent forwards.

For a **live sparse trajectory**, add `--lambda 0.01 --decode-lambda 0.001`.
The decode threshold inherits prefill unless explicitly supplied. Zero means dense.
Each run uses the same prompt and video but sparse trajectories can generate different text.
Do not interpret cross-trajectory differences as same-input kernel error.

For browser inspection of individual scores, add `--score-data` to collection. The collector
writes an FP32 `.scores.bin` companion next to the JSON. The browser reads the selected
query chunk on demand. Older summary JSONs still show the tile map and distributions.
For offline PyTorch analysis, `--raw-dir /absolute/path/to/score-shards` exports a separate
`.pt` shard per selected layer, head and query chunk. Future positions are `-inf`.

JSON stores, for every selected layer/head/phase/step:

- Tile signed mean, mean absolute score, maximum, dense softmax weight, margin, recorded skip
  decision, exact query/key spans, and visual fractions.
- Exact token-pair counts, signed/absolute score sums, dense weight sums, and 80-bin score
  histograms for T→T, V→T, T→V and V→V. Mixed tiles are split at token level for these summaries.
- Histograms cover [-40,40]; overflow and underflow counts are retained separately. Means use
  all finite scores, including histogram outliers. Future/masked positions are excluded.

Softmax weights are computed on unskipped scores for the recorded trajectory. A region's
weight percentage is its summed dense weight divided by all regions' summed dense weight
in that view. It is not conditional on that region or a measure of output accuracy.
The threshold slider compares saved margins with ln(lambda). It neither reruns later layers
nor recalculates a live sparse trajectory. The map's color range is recomputed per view/metric;
use numerical values and histograms when comparing heads or datasets.

### Resource bounds

Default export selects five layers and two query heads, rather than every head. JSON is
limited by `--max-tiles` (default 200,000), the prompt by `--max-tokens`, and the website rejects
files over 100 MB. To record all Qwen layers/heads, explicitly enumerate them and raise the
limits as needed. A full uncompressed FP32 score dump is approximately
`4 * layers * heads * prompt_tokens²` bytes for prefill: 28 layers × 28 heads × 8192² is
about 210 GB, before decode and file overhead. The chunked implementation and summaries
avoid retaining that whole tensor, but collection performs extra softmax and CPU reductions.
Benchmark a small pilot and estimate GPU-hours before a larger study; no real GPU run has
been performed as part of this implementation.

For CPU smoke testing without weights:

```bash
python attention_study.py --demo --decode-tokens 4 --score-data --output demo.json
```

## Image + text: VQA v2 validation

The same collector and visualization also support image/question inputs from
[VQA v2](https://visualqa.org/download.html). `V` always means **visual**: image tokens for
these runs, video tokens for Video-MME runs. No visualization redeployment is needed.
Existing video manifests, recordings, and token-score companion files remain supported.

From the project `code` directory, prepare one image and its first benchmark question:

```bash
python prepare_vqav2_study.py \
  --data-root "$HOME/blasst-data/vqav2" \
  --download --images 1 --questions-per-image 1 --seed 98 \
  --output "$HOME/blasst-results/vqav2.json"
```

This downloads the **official validation** question ZIP (about 3.5 MB) and annotation ZIP
(about 10.5 MB), plus only the selected individual COCO validation images. It does not
fetch the complete COCO image archive. JSON may also be supplied directly as
`v2_OpenEnded_mscoco_val2014_questions.json` and `v2_mscoco_val2014_annotations.json` inside
`--data-root`; local images belong in `val2014/`. Omit `--download` to use local data only.
Use `--image-id 42 --images 1` to select an exact COCO validation image instead of seeded
sampling. Increase `--images` or `--questions-per-image` to prepare more samples.

Input is `[image][benchmark question][Answer briefly.]`; the model's chat template adds
its normal control tokens. Unlike Video-MME, VQA v2 uses open-ended answers, not A–D
choices. Ten human reference answers and the annotation's consensus answer stay in metadata;
they are never included in the model prompt. Source ZIP/JSON and image SHA-256 hashes are
recorded in the manifest to identify the downloaded version. Sampling selects images first,
then orders each image's questions by question ID. Sample index 0 is the first selected pair.

Run inside an allocated CUDA GPU job, with the same layer/head selection as the expanded
video experiment:

```bash
python attention_study.py \
  --manifest "$HOME/blasst-results/vqav2.json" --sample-index 0 \
  --layers 0,1,2,10,11,12,13,14,15,16,17,26,27 \
  --heads 0,4,8,12,16,20,24,27 \
  --decode-tokens 16 --max-tokens 8192 --max-tiles 500000 \
  --score-data \
  --output "$HOME/blasst-results/vqav2-image0-question0.json"
```

Transfer **both** `vqav2-image0-question0.json` and
`vqav2-image0-question0.scores.bin` to the Mac. Select both with **Load experiment** in
Attention Lab. The JSON alone supports tile summaries and distributions; the binary
companion enables individual-token inspection. There is no need to also use `--raw-dir`
unless separate PyTorch score shards are desired.

`--image-max-pixels` (default 1,003,520) controls image resizing and therefore the visual
sequence length. The image still preserves its original aspect ratio. No `--frames` flag
is needed for images. For a manually supplied image, use `--image /absolute/path/image.jpg`
with `--question '...'` instead of a manifest.

The collector detects both `image_token_id` and `video_token_id` as visual placeholders;
vision-start/end and other control tokens stay text. The image encoder uses SDPA, while
the language-model attention uses the same causal BLASST implementation and recorder.
Do not interpret an image as one video frame: their processor/position conventions differ.
V→T is limited to the text preceding the image, including system text. Generated decode
queries are text, so decode contains T→T and T→V only.

These are attention diagnostics, not official VQA accuracy scores. Fixed-length decode can
continue after EOS. Preparation and image preprocessing are verifiable on CPU; actual model
attention collection requires CUDA and has not been run on this Mac.

## Public benchmark workflow: Video-MME

Use the original Video-MME benchmark to match the historical experiments:
[official project](https://github.com/MME-Benchmarks/Video-MME) and
[public data](https://huggingface.co/datasets/lmms-eval/Video-MME).
The original `lmms-lab/Video-MME` location now redirects to `lmms-eval/Video-MME`.
This is v1, not Video-MME-v2. Annotations contain 900 videos and 2,700 questions,
with short, medium and long duration categories. The adapter pins revision
`ead1408f75b618502df9a1d8e0950166bf0a2a0b` for downloads.

From the project `code` directory, prepare a one-video pilot:

```bash
python prepare_videomme_study.py \
  --data-root /scratch/videomme \
  --download --chunks 01 \
  --durations short --videos-per-duration 1 --questions-per-video 1 \
  --layout C --seed 98 \
  --output /scratch/blasst/videomme-C.json
```

Archive 01 is about 5.18 GB compressed; extraction requires additional disk space.
Only requested chunks are fetched. Model weights are separate. The prepare command
runs on CPU and does not run a model. If your existing Video-MME folder already has
`videomme/test-00000-of-00001.parquet` and extracted MP4s, omit `--download` and point
`--data-root` there. Existing local annotations are marked `local_unverified` rather
than falsely claiming their revision has been checked.

On a CUDA machine, collect the selected benchmark sample:

```bash
python attention_study.py \
  --manifest /scratch/blasst/videomme-C.json --sample-index 0 \
  --frames 8 --layers 0,14 --heads 0 --decode-tokens 4 \
  --output /scratch/blasst/videomme-C-0-dense.json
```

Load the resulting attention JSON into the website, **not** the preparation manifest.
The question, four choices, video ID, question ID, ground-truth answer, duration, source
revision, seed and prompt-layout tag are preserved in metadata. The ground-truth answer
is never appended to the model prompt. To record another sample, change `--sample-index`
and use a different output filename. Each invocation loads the model again; this simple
pilot workflow is not optimized for large batch throughput.

For a larger attention study, set `--durations short,medium,long`, increase
`--videos-per-duration`, and optionally request all three questions per video with
`--questions-per-video 3`. Add chunks such as `--chunks 01,02,03` when the available
local subset does not contain enough videos. Sampling is seeded and stratified by
requested duration among **available local videos**; it is not a globally representative
random sample of all 900 videos. The manifest fails rather than silently reducing a stratum.

Compare these prompt layouts on the same downloaded video pool and seed:

- **C**: `[video][benchmark question and choices]`, no subtitles.
- **B**: `[video][truncated subtitles][benchmark question and choices]`.
- **A**: `[first subtitle half][video][second subtitle half][benchmark question and choices]`.

Prepare A/B with `--download --layout A` or `--download --layout B`; that also fetches
`subtitle.zip` (about 7.5 MB). Subtitles are limited to `--subtitle-chars` (default 4,000)
and split by characters for A. These are explicitly **diagnostic layouts**: no frame-aligned
subtitle selection or token/tile alignment is claimed, and they do not replicate the
original report's A/B prompts exactly. A/B require available subtitles, so their candidate
pool may differ from C; match video IDs explicitly for paired comparisons.

Low frame counts, sampled questions and fixed-length decode are attention diagnostics,
not official Video-MME leaderboard scores. No new Video-MME inference results have been
collected on this machine. The website demonstration remains synthetic until real exports
are loaded. Do not upload benchmark media or question text to the hosted site; the static
explorer analyzes loaded files locally. Follow Video-MME's academic-use data terms.

## Decode parity

```bash
python compare_decode_modelopt.py --device cpu --output decode-modelopt.json
# Or --device cuda in the GPU environment.
python -m pytest test_decode_study.py -q
python test_blasst.py --cpu
python test_blasst_regions.py
```

The comparison uses identical FP32 Q/K/V tensors for both implementations, with MHA/GQA,
partial key tiles, several cache lengths and thresholds, and a planted sink to ensure
nonzero skipping. It compares exact element masks AND attention outputs against installed
ModelOpt, failing on a mismatch. The checked-in result has 32 cases, zero mask mismatches,
and maximum absolute output error 5.960464477539062e-7, using ModelOpt 0.47.0 on CPU.
This establishes controlled decode parity, not full-model Qwen accuracy or GPU kernel parity.

At exact threshold equality the existing custom rule keeps the tile while ModelOpt's
strict `>` keep rule discards it. This convention is retained to preserve existing prefill
experiments; the parity report includes equality counts. Padding of the final query tile
also differs in prefill, as documented in the original baseline report.

`blasst_attn.py` now defaults to applying the shared threshold in both phases:

```python
ba.set_threshold(lam=0.01)               # prefill; inherited by decode
ba.set_decode_threshold(lam=0.001)      # optional independent decode threshold
ba.set_decode_threshold(enabled=False) # original dense SDPA decode behavior
ba.set_decode_threshold(inherit=True)   # restore shared threshold
```

`get_margins()` returns prefill margins; `get_decode_margins()` returns a separate list per
layer, one tensor per decode step, because cached key lengths can cross tile boundaries.
`STATE['phase_stats']` separates prefill/decode counts. The historical `blasst_regions.py`
remains prefill-only; the new collector uses `blasst_attn.py` for both phases and classifies
its observations by modality.
