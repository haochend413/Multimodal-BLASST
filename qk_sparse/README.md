# qk_sparse — does vertical-vector sparsity exist outside video?

A self-contained framework for testing one claim across modalities:

> **Neighbouring queries share their important keys.** The useful entries of an
> attention map form short *vertical vectors* — one key column spanning a
> contiguous group of query rows — rather than isolated entries, full vertical
> lines, or square blocks.

This is the observation VecAttention (arXiv 2603.29494) builds on, measured in
video diffusion transformers. The question here is whether it is a property of
*video* models, of *long sequences* generally, or of transformers as such.

Nothing is imported from the VecAttention codebase. The metrics are defined
from `Q` and `K` alone.

## Install

```bash
pip install -r requirements.txt
```

Only `torch` and `numpy` are needed for the synthetic control. The rest of the
suite needs `transformers`, plus modality-specific extras (`av` for video,
`soundfile`/`librosa` for audio, `pillow` for images).

## Run

```bash
# 30 seconds, no network, no weights: does the harness measure what it claims?
python -m qk_sparse run --adapters synthetic-high synthetic-low --out results/

# the full cross-modality suite
python -m qk_sparse run --out results/ --samples 2 --device cuda

# tables
python -m qk_sparse report results/ --density 0.05 --out report.md
```

`python -m qk_sparse list` shows the adapters.

## What gets measured

For each (model, sample, layer, head), the sequence is cut into query blocks of
`P_q` rows. Each block is scored against its **allowed keys** — all keys for a
bidirectional model, or only keys *strictly before the block* for a causal one,
so every row in a block shares one allowed set and all patterns are compared on
the same ground. The near-diagonal region is excluded deliberately: real kernels
always keep it dense, so including it would flatter every pattern equally.

### 1. Pattern comparison at matched density

Every pattern family gets the **same entry budget** and differs only in the
shape it must spend it in. The metric is *recovered attention mass*: how much of
each row's softmax mass lands inside the kept entries.

| pattern | unit | what it stands for |
|---|---|---|
| `line` | one key, all queries in the head | MInference-style full vertical lines |
| `block` | query block × key block | XAttention / block-sparse kernels |
| `vertical` | query block × one key | **the claim under test** |
| `horizontal` | one query × key block | control: same coarse/fine split, other axis |
| `oracle` | per-row top-m entries | unstructured upper bound |

`vertical` here is chosen by an oracle that already knows the true attention, so
it measures the *pattern's ceiling*, not what a selector achieves.

### 2. Selector quality

What an implementable selector reaches, ranking keys without seeing the
attention map: VecAttention's mean-pooled query, a single centre row, a strided
sample of 4 rows (max or mean combined), and a random row. `pooled/ceiling` is
the fraction of the vertical ceiling the pooled estimator recovers.

The strided variants exist because they are the natural alternative if you would
rather *sample* queries in a block than average them.

### 3. Mechanism: agreement vs distance

Top-key Jaccard between a query and the query `Δ` positions away, against the
**chance level** for the same density (two independent top-m sets out of n
overlap by `m²/n` on average). Reported alongside the cosine between the query
vectors themselves, which separates the two candidate causes:

- **query similarity** — neighbouring queries are nearly the same vector, so
  they rank keys the same way (high `qcos`, high Jaccard);
- **key salience** — a few keys are important to everyone regardless (low
  `qcos`, still-high Jaccard).

### 4. Output error

Recovered mass is a proxy. What a kernel changes is `O = A V`, so each pattern's
keep-mask is applied, renormalised over the kept keys exactly as a masked
flash-attention kernel does, and scored as

    ||O_sparse - O_dense|| / ||O_dense||

averaged over rows. A pattern can keep 70% of the mass and emit a nearly correct
row, or a badly wrong one; this distinguishes them. Still not end-task accuracy,
but it is the quantity that propagates into the next layer.

`line` is not scored this way — its mask is only known after every block in the
head has been seen, by which point the attention matrix is gone.

### 5. Contiguous vs strided query grouping

A vertical vector requires the queries sharing a key set to be **adjacent in
sequence**. In a space-time grid that may be the wrong grouping: for VideoMAE
the same patch one tubelet later is 196 positions away, and measured key-set
agreement there (0.289) is *higher* than at the patch directly below it (0.279).

So for grid modalities the same retention is also measured over a strided group
(`anchor, anchor+s, anchor+2s, ...`) and reported beside the contiguous one. If
strided wins, contiguity — not thresholding — is what limits the pattern.

Validated on a control whose structure repeats every 196 positions: contiguous
grouping retains 0.276, strided 0.993.

### 6. Stability diagnostics

`cos_to_mean` and `dispersion` (within-block query spread), and `margin_pooled`
(normalised gap between the m-th and (m+1)-th ranked key). Pooling needs *both*
similar queries and a real margin: a block of near-identical queries still routes
unstably if many key scores are nearly tied.

## Sampling caveats that were fixed

- **Heads are sampled at random**, not taken as 0..N-1. Attention heads are
  heterogeneous, so the first four are not a sample of the rest; taking them was
  a silent bias in every per-model number.
- **Diffusion capture targets one mid-trajectory step.** Attention structure
  changes along the denoising trajectory, and step 0 is the least
  representative point. `video-dit` runs 20 steps and records step 10 only,
  gated so a pipeline's per-step attention calls are never mixed across steps.

## Reading the verdict

The decisive statistic is **`vertical / oracle`** — how much of per-row selection
survives when a whole block must share one key set. That is the claim itself.

`vertical − block` is reported but not used to decide, because part of it is
free: picking individual key columns beats picking key blocks even when queries
are entirely unrelated, purely from finer granularity along the key axis. The
independent-query control shows exactly that.

The synthetic controls pin the scale:

| control | vertical/oracle | Jaccard@1 | chance |
|---|---|---|---|
| `synthetic-high` (queries in a block identical) | 1.00 | 0.87 | 0.03 |
| `synthetic-low` (queries independent) | 0.27 | 0.02 | 0.03 |

Any real-model number only means something between those two poles. If a model
scores near 0.27, vertical vectors buy it nothing.

## Verification status

What has actually been run, so you know what to trust:

| adapter | status |
|---|---|
| `synthetic-high` / `synthetic-low` | verified — behaves exactly as the ground truth requires |
| `text-small` (SmolLM2-135M, 4096 tok) | verified end to end on CPU |
| `image` (DINOv2-base, 1370 tok) | verified end to end on CPU |
| `video` (VideoMAE, 1568 tok) | verified on GPU; needs the bias repair below |
| `audio` (Whisper-small, 1500 tok) | verified on GPU |
| `timeseries` (PatchTST, 43 tok) | runs, but see the scale caveat |
| `vlm` (Qwen2.5-VL-7B, ~20k tok) | **not yet executed** — closest to VecAttention's VLM setting |
| `video-dit` (Wan2.1-T2V-1.3B, 7800 tok) | **not yet executed** — closest to their generation setting |
| `text` (Qwen2.5-0.5B), `image-vit`, `video-timesformer` | same adapter classes as the verified ones, different weights |

Sample output from the verified runs (density 0.05, one sample each):

```
image  : vertical 0.419 vs block 0.255 vs oracle 0.448  -> retains 0.93, Jaccard@1 0.69 (27x chance)
text   : vertical 0.742 vs block 0.575 vs oracle 0.835  -> retains 0.89, Jaccard@1 0.49 (19x chance)
```

Both sit near the coherent pole of the synthetic scale, which is an early sign
the effect is not video-specific. One sample per model is not a result; it means
the instrument reads.

## Two traps this hit in practice

**Silently random attention weights.** VideoMAE stores BEiT-style `q_bias`/`v_bias`
with *no key bias at all*; recent transformers expects plain `query/key/value.bias`,
doesn't convert, and random-initialises all three. The model loads, runs, and
produces plausible-looking numbers computed from partly random Q/K.

Every load now goes through `output_loading_info=True`, and any randomly
initialised attention parameter marks the record `invalid_weights` and prints
`⚠️INVALID` in the tables. For VideoMAE the repair is exact and applied
automatically (`q_bias → query.bias`, `v_bias → value.bias`, `key.bias = 0`).
Check for this whenever you add a model.

**Budgets too small to measure.** At 43 tokens, density 0.05 buys *two* keys, and
two-element top-sets from neighbouring rows disagreeing says nothing about
geometry. Overlap is recorded at every density (`d<density>/jaccard@<delta>`,
alongside `budget_keys`), and the verdict refuses to rule when the budget drops
below 4 keys.

## Network caveat

Dataset streaming can **hang indefinitely** rather than fail — one run here sat
for 100 minutes at 0% CPU waiting on a stream. On a cluster, stage the assets
first and run against local files:

```bash
python scripts/fetch_assets.py --out assets/    # login node
HF_HUB_OFFLINE=1 DATA_DIR=assets ./scripts/run_suite.sh
```

## Adapters

| adapter | modality | model | tokens | causal |
|---|---|---|---|---|
| `text` | text | Qwen2.5-0.5B | 4096 | yes |
| `text-small` | text | SmolLM2-135M | 4096 | yes |
| `image` | image | DINOv2-base @518px | ~1370 | no |
| `image-vit` | image | ViT-base @224px | 197 | no |
| `video` | video | VideoMAE-base, 16 frames | ~1568 | no |
| `video-timesformer` | video | TimeSformer-base | ~1568 | no |
| `audio` | audio | Whisper-small encoder | 1500 | no |
| `timeseries` | time series | Granite PatchTST (ETTh1) | ~42 | no |
| `vlm` | video+text | Qwen2.5-VL-7B-Instruct | ~20k | yes |
| `vlm-3b` | video+text | Qwen2.5-VL-3B-Instruct | ~20k | yes |
| `video-dit` | video DiT | Wan2.1-T2V-1.3B (diffusers) | 7800 | no |
| `synthetic-high/low` | control | none | 2048 | no |

Adding one means ~30 lines in `models.py`; capture and metrics are shared.

### Caveats that affect interpretation

- **Time series sequences are tiny.** PatchTST turns a 512-step window into ~42
  patch tokens, so its block size drops to 8 and its numbers are not
  scale-comparable with the video case. It answers "do nearby queries agree" at a
  different scale, not the same question.
- **Image/video token order is raster.** Queries adjacent in sequence are
  horizontal neighbours in the image; the vertical neighbour is `W` positions
  away, and the same patch in the next frame is `H*W` away. The `Δ` sweep is what
  exposes this — expect structure at those strides, not only at `Δ=1`.
- **Scale is the main gap to VecAttention.** They measure HunyuanVideo-T2V-13B
  (~119k tokens), Wan2.1-T2V-14B, Qwen2.5-VL-7B (~26k) and InternVL-3.5-8B
  (~17k). The small adapters here run at 1.4k–4k, which is 10–80x smaller, and
  a stride-aligned query group cannot even be formed at that length. `vlm` and
  `video-dit` exist to close that gap on one GPU; `qk-dump` closes it exactly,
  by reading their own dumped tensors.
- **`video` here is understanding, not generation.** VecAttention's strongest
  claims come from video *diffusion* transformers with far longer sequences.
  VideoMAE at 1568 tokens is the small-scale stand-in.

## How capture works

Metrics need Q/K *post-RoPE*, as the kernel sees them, so hooking `q_proj`/
`k_proj` would give the wrong tensors. Instead `capture.py` patches
`torch.nn.functional.scaled_dot_product_attention`, which HuggingFace
(`attn_implementation="sdpa"`), diffusers and timm all funnel into, and also
rebinds aliases held by modules that imported the symbol directly.

Each sample runs **twice**: once to count attention calls and find the long
self-attention ones (so layer selection needs no per-architecture knowledge),
then once storing Q/K for a spread of depths.

A model that uses an eager `torch.matmul` attention path captures nothing, and
the runner reports that as a warning rather than an empty result.

## Output

`results/<adapter>.json` holds every per-block and per-probe measurement, with
the config and environment that produced it. `report` reduces these to medians
with interquartile ranges across (sample, layer, head) — head-level variance is
large, and a mean would hide it.

## Layout

```
qk_sparse/
  metrics.py    pattern families, overlap-vs-distance, coherence, margins
  capture.py    SDPA interception, layer selection
  models.py     adapters + registry
  data.py       inputs per modality, with --data-dir overrides
  runner.py     two-pass orchestration, JSON records
  aggregate.py  medians, tables, verdict
  cli.py        run / report / list
```
