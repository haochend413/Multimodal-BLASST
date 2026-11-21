# Raw results

The CSV / JSON files behind the reports, laid out as the code writes them: Llama baseline at the top level,
Qwen2.5-VL study in `vlm_regions/`. The write-ups are in `../reports/` and the PNG figures in
`../reports/figures/`. Every number in the reports can be traced to a file here.

## Baseline, Llama-3.1-8B-Instruct (`../reports/1_baseline_llama.md`)

| File | What it holds |
|---|---|
| `smoke_niah_multikey_2_4k_row0.json` | Smoke test: one 4K RULER sample, dense vs ours. |
| `fig1_sparsity_vs_lambda.csv` | Raw: % skipped per sample at 121 lambdas (1e-6 to 1), 4K-32K. |
| `fig1_per_layer.csv` | % skipped per layer at a few lambdas. |
| `fig1_table.csv` | Per length: fixed lambda 1e-3 / 3e-3 and calibrated 50% / 75%, vs the paper. |
| `fig1_per_task.csv` | Per length and task: % skipped at 1e-3 and 3e-3. |
| `calibration.json` | Fitted calibration (lambda x L = alpha x exp(beta x s)): alpha, beta, and a for 50-90% targets, both fits. |
| `crosscheck_modelopt.json` | Our % skipped vs NVIDIA ModelOpt on 4 samples. |
| `tile_sensitivity_8k.csv`, `tile_sensitivity_16k.csv` | % skipped by tile shape and lambda. |
| `fig2_ruler_8k_dense.csv`, `fig2_ruler_8k_cal50_cal75_cal90.csv`, `fig2_ruler_16k_dense.csv`, `fig2_ruler_16k_cal50.csv`, `fig2_ruler_16k_cal75.csv`, `fig2_ruler_16k_cal90.csv` | Raw RULER results, one row per sample: prediction, answers, score, live % skipped, seconds. |
| `fig2_table.csv` | Score per task and config, achieved % skipped, change vs dense. |
| `fig2_flips.csv` | Per length, config and task: answers that got worse / better vs dense. |

## VLM study, Qwen2.5-VL-7B-Instruct (`vlm_regions/`)

Step 1 (`../reports/2_vlm_regions_step1.md`):

| File | What it holds |
|---|---|
| `prompts.csv` | The 10 step-1 prompts: video, frames, token counts, the 50% / 75% lambdas, dense answer. |
| `check.json` | Correctness check: our function with skipping off vs standard attention. |
| `region_table.csv` | % skipped per region at 50% / 75% overall, pooled and per prompt; margin percentiles. |
| `per_layer.csv` | % skipped per region and layer. |
| `distance_table.csv` | % skipped per region at matched distance from the diagonal. |
| `t2v_question_split.csv` | T->V split into the question's blocks vs the transcript's. |
| `curves.csv` | % skipped vs lambda per region. |

Batch 2 (`../reports/3_vlm_regions_batch2.md`):

| File | What it holds |
|---|---|
| `v2_weight_regions.csv` | Item 1, pooled: dense attention weight and weight lost at 50% / 75%, by row group and key type (video vs other), per layout and for Llama. |
| `v2_weight_regions_per_prompt.csv` | Same, per prompt. |
| `v2_weight_distribution.csv` | Spread of weight lost across rows: percentiles, share of rows losing over 1 / 5 / 10%. |
| `v2_weight_layers.csv` | Weight lost by layer. |
| `v2_layout_regions.csv` | Item 2: % skipped per region in layouts A, B and C. |
| `v2_pilot.jsonl` | Item 3 raw: one line per question and setting (90 x 3): answer, letter logits, lambda, live % skipped. |
| `v2_pilot_summary.csv` | Item 3: correct out of N per setting, plus live % skipped. |

## Not included (too big; still on DeltaAI)

`margins/` (3 GB) and `qk/` (22 GB) for Llama; `vlm_margins/` (153 MB) and `vlm2_tensors/` (5.4 GB) for Qwen;
decoded frames and prompts (`vlm_inputs/`, `vlm2_inputs/`).
