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

The new `attention_study.py`, `compare_decode_modelopt.py`, and `test_decode_study.py` are documented
in [Attention Lab](../attention-lab/README.md).

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
