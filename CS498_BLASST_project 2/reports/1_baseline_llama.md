> **In this package:** the data files named below are in `../results/`, figures in `figures/`, code in `../code/`. Not included (too big, still on DeltaAI): `margins/`, `qk/`, `HANDOFF.md`. Report copied unchanged from DeltaAI except this line and the code path.

# BLASST baseline: results (Llama-3.1-8B-Instruct, bf16, DeltaAI GH200)

Run Nov 1, 2026. All numbers below come from the CSV and JSON files in this folder. Code is in
`/work/nvme/beqg/zyang21/blasst/` (copy in `../code/`).

Words used: a **tile** is a 128 query rows x 128 keys block of the attention matrix. The **skip rule**
skips a tile when every one of its 128 rows has (tile row max - running row max) < ln(lambda).
The **margin** of a tile is the largest of those 128 differences, so a tile is skipped exactly when
margin < ln(lambda). **% skipped** = skipped tiles / causal tiles (tiles on or below the diagonal), over all
32 layers and 32 query heads, prefill only. **Dense** = no skipping (PyTorch SDPA). **Calibrated** =
lambda = a / L, with L = number of prompt tokens and a fitted for a target % skipped.

## 1. Headline: what reproduced and what did not

| Quantity | Length | Ours | Paper | Reproduced? (within 3 pts sparsity / 1 pt RULER) |
|---|---|---|---|---|
| % skipped, calibrated 50% target | 4K | 46.73 | 54.20 (Table 6) | no (-7.5) |
| | 8K | 50.82 | 49.70 | **yes** (+1.1) |
| | 16K | 53.38 | 52.20 | **yes** (+1.2) |
| | 32K | 55.90 | 46.96 | no (+8.9) |
| % skipped, fixed lambda = 1e-3 | 4K / 8K / 16K / 32K | 8.35 / 16.31 / 26.99 / 39.32 | 23.09 / 37.92 / 52.38 / 65.72 | no (about half) |
| % skipped, fixed lambda = 3e-3 | 4K / 8K / 16K / 32K | 19.01 / 29.92 / 42.13 / 54.60 | 42.35 / 57.54 / 69.83 / 79.36 | no (about half) |
| RULER, dense vs calibrated 50% | 8K | 95.07 vs 93.87 (48.25% skipped) | 95.07 vs 94.70 (Table 3) | **yes** (our 50% score is 0.83 below the paper's; the paper's drop of 0.37 is inside our 95% range) |
| | 16K | 95.33 vs 94.89 (51.40% skipped) | 94.80 vs 94.61 | **yes** (drop 0.44 vs paper's 0.19) |

Short version:
- **The implementation is correct.** It passes all tests, gives the same output as standard attention
  when skipping is off, and matches NVIDIA's own reference code (ModelOpt) within 0.32 points of % skipped
  when we copy NVIDIA's treatment of the last, partial query tile (section 4).
- **Calibrated sparsity reproduces at 8K and 16K** (the two lengths the paper's RULER table uses).
- **Fixed-lambda sparsity is about half of the paper's Table 6, and NVIDIA's own code gives the same low
  numbers as ours.** So the gap is not a bug in our code; it most likely comes from how the paper's Table 6 was
  measured. Our best guess is the tile shape: with 16-row x 64-key tiles we land within 0.3 to 4.4 points of Table 6 (section 5).
  This is a guess that fits the numbers, not a proven cause.
- **RULER accuracy:** at about 50% skipped the 3-task average drops 1.20 points at 8K and 0.44 at 16K
  (paper: 0.37 and 0.19 on 13 tasks). Both are within sampling noise of the paper. At 75% skipped the
  average stays within 1.4 points of dense; at the 90% target (about 82% skipped) vt breaks (-13 points).

## 2. Correctness tests (`test_blasst.py`, tiny random Llama, about 25 s on a GPU)

All 5 tests pass:
1. Skipping off: our attention equals PyTorch SDPA (standard attention).
2. Skipping on, with a planted "attention sink" so that some tiles really are skipped: the tiles skipped
   during the run (online) equal the tiles predicted from saved margins (offline) and equal a simple
   unchunked reference; outputs match. Checked with full heads (MHA, 1024 tokens) and grouped heads
   (GQA, 700 tokens, a partial last tile).
3. A whole tiny Llama through transformers: same logits and same `generate()` text as SDPA; skipping
   actually happens with the planted sink.
4. Counting of causal tiles (the denominator of % skipped) matches a brute-force count.
5. Calibrated mode `a / L` equals fixed lambda = a / L; very large a is capped at lambda = 1; lambda > 1 is rejected.

Smoke test on the real model (`smoke_niah_multikey_2_4k_row0.json`, 3,704 tokens): skipping off gives the
same top answer token as SDPA (largest logit difference 0.31, normal for bf16). One prefill takes 0.64 s
with SDPA and 0.82 s with our function.

## 3. Fig. 1: % skipped vs lambda (`fig1_sparsity_vs_lambda.png`, `fig1_collapse_lambdaL.png`)

Samples: `self-long/RULER-llama3-1M`, tasks niah_multikey_1/2/3, vt, fwe, rows 100-105 of each
(30 samples per length at 4K, 8K, 16K; rows 100-102 at 32K, 15 samples). Rows 0-49 are kept for the RULER
test so that calibration and testing do not share samples. One dense pass per sample records the margins;
% skipped at any lambda is then exact (121 lambdas from 1e-6 to 1).

`fig1_table.csv` (mean over samples, standard deviation in the CSV):

| Length | Samples | Mean tokens | lambda 1e-3 | lambda 3e-3 | Calibrated 50% | Calibrated 75% |
|---|---|---|---|---|---|---|
| 4K | 30 | 3,706 | 8.35 | 19.01 | 46.73 | 72.20 |
| 8K | 30 | 7,706 | 16.31 | 29.92 | 50.82 | 76.06 |
| 16K | 30 | 15,919 | 26.99 | 42.13 | 53.38 | 78.17 |
| 32K | 15 | 32,294 | 39.32 | 54.60 | 55.90 | 79.60 |

Per task (`fig1_per_task.csv`): vt and fwe skip about 4.5 to 7.5 points more than the niah tasks at 4K-16K
(for example 8K, lambda 1e-3: fwe 20.43, vt 20.01, niah_multikey_1 12.95).

**Calibration** (`calibration.json`). Same rule as the paper's Algorithm 2 and NVIDIA's calibrator: take every
(lambda x L, % skipped) point with 10% to 90% skipped (6,454 points), fit lambda x L = alpha x exp(beta x s),
then a(target) = alpha x exp(beta x target). We fit it in log space: alpha = 1.686, beta = 8.29, giving
**a = 106.5 for 50%, 846 for 75%, 2,934 for 90%**. NVIDIA's default (ordinary least squares on lambda x L
itself) is pulled toward the biggest lambda x L values and undershoots: its 50% a = 52.2 gives only
37.6 / 41.1 / 43.4 / 46.1 % at 4K / 8K / 16K / 32K. Same model, different way of measuring the error;
we report the log-space fit and say so.

`lambda = a / L` holds roughly but not exactly: with a fixed a, % skipped still grows with length
(46.7 % at 4K to 55.9 % at 32K for the 50% target). The paper's own calibrated numbers swing the other
way (54.2 % at 4K, 46.96 % at 32K), so neither of us gets a flat line; ours is within 1.2 points at 8K and 16K.

## 4. Cross-check against NVIDIA ModelOpt (`crosscheck_modelopt.json`)

NVIDIA's reference package (`nvidia-modelopt` 0.47.0, `flash_skip_softmax`, installs fine on ARM) on the
same 4 samples (3 at 4K, 1 at 8K), mean % skipped:

| lambda | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 1e-1 |
|---|---|---|---|---|---|---|---|
| ours | 0.59 | 2.80 | 10.33 | 21.86 | 36.54 | 50.66 | 65.64 |
| ours, NVIDIA's last-tile padding | 0.52 | 2.49 | 9.39 | 20.13 | 33.95 | 47.29 | 61.46 |
| NVIDIA ModelOpt | 0.51 | 2.52 | 9.40 | 20.31 | 34.19 | 47.43 | 61.53 |

The only difference is the last query tile when the prompt length is not a multiple of 128. NVIDIA fills the
missing rows with a very negative number; those fake rows always "vote" to keep the tile, so the last query
tile is never skipped. Our code lets only the real rows vote. With NVIDIA's convention the two agree within
0.32 points on every sample and lambda (at most 0.26 at the 3 4K samples). All numbers in this report use our convention (it is the paper's rule applied to real rows).

## 5. Why fixed lambda gives half the paper's sparsity: tile shape (`tile_sensitivity_*.csv`)

`tile_sensitivity.py` recomputes the skip rule from the saved Q and K with other tile shapes
(rows that must all agree x keys per tile). 8K = 5 samples (one per task); 16K = 2 samples (niah_multikey_1, _2).

| Tile (rows x keys) | 8K, 1e-3 | 8K, 3e-3 | 16K, 1e-3 | 16K, 3e-3 |
|---|---|---|---|---|
| 128 x 128 (ours, NVIDIA reference) | 16.23 | 29.81 | 24.84 | 40.27 |
| 64 x 64 | 26.08 | 41.57 | 37.49 | 53.76 |
| 32 x 64 | 31.57 | 47.59 | 43.43 | 59.62 |
| 16 x 64 | **38.18** | **54.36** | **49.63** | **65.42** |
| 1 x 128 (every row decides alone) | 60.92 | 74.45 | 73.09 | 83.84 |
| **Paper, Table 6** | **37.92** | **57.54** | **52.38** | **69.83** |

Smaller tiles skip much more at the same lambda, because fewer rows have to agree. 16-row x 64-key tiles land
within 0.3 to 4.4 points of Table 6 at both lengths. Caveats: we picked the closest of 10 shapes after seeing the
numbers, and the 16K check has only 2 samples, both niah (which skip less than vt and fwe). So this is
"consistent with", not "proven". What is solid: **at a fixed lambda, tile shape changes % skipped by a factor of
2 or more**, which is why we compare with the paper at matched achieved sparsity (calibrated), not matched lambda.

## 6. Fig. 2: RULER accuracy with skipping on (`fig2_table.csv`, `fig2_accuracy_vs_sparsity.png`)

Rows 0-49 of niah_multikey_1, vt and fwe at 8K and 16K (150 samples per length and setting). Dense uses
PyTorch SDPA. Calibrated runs use our function with skipping really on during prefill (lambda = a / L with
the a values above); the answer is generated with standard attention. Score = RULER `string_match_all`
(share of reference answers found in the output, x 100).

| Length | Setting | % skipped (measured live) | niah_multikey_1 | vt | fwe | Average of 3 | Change vs dense (95% range) |
|---|---|---|---|---|---|---|---|
| 8K | dense | 0 | 98.0 | 99.2 | 88.00 | **95.07** | |
| 8K | calibrated 50% | 48.25 | 98.0 | 99.6 | 84.00 | **93.87** | -1.20 (-2.31 to -0.27) |
| 8K | calibrated 75% | 71.72 | 100.0 | 98.8 | 88.67 | 95.82 | +0.76 (-0.84 to +2.71) |
| 8K | calibrated 90% | 81.97 | 90.0 | 86.4 | 94.00 | 90.13 | -4.93 (-8.89 to -1.16) |
| 16K | dense | 0 | 98.0 | 100.0 | 88.00 | **95.33** | |
| 16K | calibrated 50% | 51.40 | 98.0 | 100.0 | 86.67 | **94.89** | -0.44 (-1.33 to +0.44) |
| 16K | calibrated 75% | 73.22 | 98.0 | 97.2 | 86.67 | 93.96 | -1.38 (-2.89 to 0.00) |
| 16K | calibrated 90% | 83.07 | 100.0 | 86.8 | 90.67 | 92.49 | -2.84 (-5.64 to -0.04) |
| **Paper Table 3, 8K** | dense vs about 50% | | | | | 95.07 vs 94.70 | -0.37 |
| **Paper Table 3, 16K** | dense vs about 50% | | | | | 94.80 vs 94.61 | -0.19 |

The 95% range comes from resampling: draw 50 of the 50 samples with replacement 10,000 times, keeping each
sample's dense and BLASST scores paired, and take the middle 95% of the resulting changes (`fig2_analyze.py`).
It shows how much the change could move just from which 50 samples we happened to use.

What it shows:
- **At about 50% skipped, accuracy matches the paper.** 16K: -0.44 points (paper -0.19). 8K: -1.20 points
  (paper -0.37); the paper's -0.37 is inside our 95% range, and all of the 8K drop is fwe, where 6 of 50
  samples each lost one of their 3 answers (`fig2_flips.csv`). niah_multikey_1 is unchanged at 50% and vt changes by one sample at 8K (99.2 -> 99.6) and not at all at 16K.
- **75% skipped is still close to dense** (+0.76 at 8K, -1.38 at 16K).
- **90% target breaks vt.** (It reaches only 82-83% skipped: the a = alpha x exp(beta x s) fit is less accurate
  near 90%.) vt scores 86.4 / 86.8 vs 99.2 / 100 dense, with 19 and 15 samples worse.
  niah_multikey_1 at 8K drops to 90.0. The average falls 4.93 (8K) and 2.84 (16K) points.
- fwe goes **up** with more skipping (88 -> 94 at 8K, 90% target). fwe asks for the 3 most frequent words.
  We do not know why skipping helps it here; treat fwe changes of a few points as noise in both directions.
- The dense 8K average happens to equal the paper's 13-task dense average (95.07); that is a coincidence of
  the 3 tasks, not a sign of a matched setup.

Seconds per sample (prefill + answer) are in the table CSV but are **not** a speed result: our function is a
PyTorch reference (Python loop over 128-row chunks). 8K: 0.99 s dense, 4.0 s ours; 16K: 1.55 s dense, 12.4 s ours.

## 7. GPU hours

Charged hours from `sacct` (charge = GPU time x billing factor; interactive partition costs 2x and more
memory costs more). Allocation balance before this work: 541 hours.

| Step | Jobs | Charged hours |
|---|---|---|
| Tests, smoke test, debugging (interactive) | 5 | 0.12 |
| Fig. 1 margins + Q/K dump, 4K-32K, 105 samples (ghx4) | 1 | 0.50 |
| ModelOpt cross-check (interactive) | 2 | 0.12 |
| Tile-shape check (interactive) | 2 | 0.07 |
| Fig. 2 dense, 8K + 16K (ghx4) | 2 | 0.13 |
| Fig. 2 calibrated 50/75/90%, 8K + 16K (ghx4) | 4 | 2.08 |
| **Total** | **16** | **3.02** |

The calibrated RULER runs dominate because our reference function is slower than SDPA.

## 8. Caveats to state in the report

- RULER: 3 tasks (niah_multikey_1, vt, fwe) x 50 samples, not the paper's 13-task average. One sample changes a
  task score by up to 2 points; niah_multikey_1 has 1 answer per sample (0 or 100), fwe 3, vt 5.
- Calibration used rows 100-105; RULER used rows 0-49, so the a values were never fitted on test samples.
- Log-space fit instead of NVIDIA's default linear fit (section 3).
- Last partial query tile convention differs from NVIDIA's: our numbers are 0.5 to 1.8 points higher at lambda 1e-3
  (up to 5 points at lambda 0.1) on the 4K samples, less at 8K (section 4).
- Our attention function is a PyTorch reference: a Python loop over 128-row chunks. It is slower than SDPA
  (per sample, prefill plus answer: 4.0 s vs 0.99 s at 8K, 12.4 s vs 1.55 s at 16K) and says nothing about speedup. Speed needs a
  real kernel; the paper's speedups come from custom CUDA kernels.
- Decode (generating the answer) always uses standard attention; sparsity numbers are prefill only, like
  the paper's Table 3.

## 9. Files

| File | What |
|---|---|
| `fig1_sparsity_vs_lambda.csv` | raw: % skipped per sample and lambda (121 lambdas) |
| `fig1_per_layer.csv` | % skipped per layer at a few lambdas |
| `fig1_table.csv`, `fig1_per_task.csv` | tables above |
| `calibration.json` | fitted alpha, beta, a for 50/60/70/75/80/90% (both fits) |
| `fig1_sparsity_vs_lambda.png`, `fig1_collapse_lambdaL.png` | Fig. 1 |
| `crosscheck_modelopt.json` | section 4 |
| `tile_sensitivity_8k.csv`, `tile_sensitivity_16k.csv` | section 5 |
| `fig2_ruler_*.csv` | raw: one row per sample (prediction, answers, score, % skipped, seconds) |
| `fig2_table.csv`, `fig2_flips.csv`, `fig2_accuracy_vs_sparsity.png` | Fig. 2 |
| `margins/`, `qk/`, `HANDOFF.md` | hand-off to teammates (3 GB margins, 22 GB Q/K) |
| `smoke_niah_multikey_2_4k_row0.json` | smoke test |

## 10. How to re-run

```bash
source /work/nvme/beqg/zyang21/blasst/env.sh          # also cd's into blasst/
srun -A beqg-dtai-gh -p ghx4-interactive --gpus=1 --cpus-per-task=8 --mem=64g -t 00:15:00 \
     bash -c 'source /work/nvme/beqg/zyang21/blasst/env.sh; python test_blasst.py'
sbatch fig1.sbatch                                    # margins + Fig. 1 raw CSV (about 21 min)
python fig1_analyze.py                                # CPU only: tables, fit, figures
sbatch --job-name=blasst-fig2-8k-dense fig2.sbatch 8k dense             # RULER: length, then configs
sbatch --job-name=blasst-fig2-8k-cal fig2.sbatch 8k cal50 cal75 cal90   # writes results/fig2_ruler_8k_cal50_cal75_cal90.csv
python fig2_analyze.py                                # CPU only
```
