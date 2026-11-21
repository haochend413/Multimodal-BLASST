# BLASST paper digest — arXiv 2512.12087

**Paper:** Yuan, Shinn, et al. (NVIDIA / Rice / UC Davis), "BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding." Proceedings of the 9th MLSys Conference (2026). UC Davis news reports it received a Best Paper Award at MLSys 2026.

**Versions read:** v1 (12 Dec 2025) and **v3 (28 Apr 2026, current)**, both fetched in full from arxiv.org/html and parsed locally (equations from LaTeXML `alttext`, tables cell-by-cell). All section/table numbers below are **v3** unless marked "v1". The v3 PDF (17 pages) was also downloaded. v1 → v3 changes that matter: the kernel-speedup table moved from Table 4 (v1, B200 only, different prefill config) to Table 5 (v3, B200 + H200), the headline prefill claim changed from "1.62x at 74.7% (H200)" (v1) to "1.52x at 71.9%" (v3), the accuracy tables are numerically identical, and v3 adds Appendices A (large models, MLA, per-dataset calibration, tile reordering), B (error bound) and C (artifact).

**Vocabulary (used throughout):** *tile* = one (query-block x key-block) of the attention matrix; *skip rule* = BLASST's per-tile test; *sparsity* = fraction of tiles skipped; *BMM1* = QK^T; *BMM2* = PV; *running max* = FlashAttention's online-softmax row maximum m_i.

Secondary sources used to fill what the paper omits (tile sizes, exact compared quantity, absolute kernel times, code): the MLSys artifact repo (`cameronshinn/blasst-ae-mlsys26`), NVIDIA's TensorRT-LLM tech blog 16 and sparse-attention docs, the FlashInfer API docs, and the NVIDIA Model-Optimizer (ModelOpt) source code. Each is cited where used.

---

## 1. The skip rule (Sec. 3.1, Algorithm 1, Fig. 1, Sec. 4)

### 1.1 Math

Standard attention (Eq. 1): `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`.

FlashAttention tiles the sequence into `T_r` query blocks and `T_c` key/value blocks of size `B_c` (Sec. 3.1.2; the paper never gives numeric sizes — see 1.4). For query block `i`, while sweeping key blocks `j = 1..T_c`:

```
S_ij        = Q_i K_j^T                      (scores, Alg. 1 line 6 — ALWAYS computed)
m~_i^(j)    = rowmax(S_ij)                   (tile-local row max, line 7)
m_i^(j)     = max(m_i^(j-1), m~_i^(j))       (running row max, line 8 — updated BEFORE the test)
if  m~_i^(j) - m_i^(j) < ln(lambda):  continue   (skip, lines 9-11)
P~_ij       = exp(S_ij - m_i^(j))            (line 12)
l_i^(j)     = e^{m_i^(j-1) - m_i^(j)} l_i^(j-1) + rowsum(P~_ij)
O_i^(j)     = e^{m_i^(j-1) - m_i^(j)} O_i^(j-1) + P~_ij V_j
O_i         = O_i^(T_c) / l_i^(T_c)
```

Eq. 2: the test guarantees `exp(m~_i^(j) - m_i^(j)) < lambda ~ 0`, i.e. every un-normalised softmax weight in the tile is below lambda, so the tile's contribution is "negligible". Sec. 3.1.1 describes it as a three-step approximation: (i) the ideal importance of a score is relative to the *global* row max; (ii) the global max is unknown mid-sweep, so the *running* max is used as a proxy; (iii) per-element tests are too expensive, so the tile-local max stands in for every element in the tile.

Note the ordering: the running max is updated with the current tile first (line 8), then compared. So the difference `m~ - m` is always <= 0, and a tile that raises the running max can never be skipped.

### 1.2 What is skipped vs. not skipped

Skipped for a pruned tile (Sec. 3.1.2 "Where We Save"):
1. CUDA-core work: the `exp` (MUFU.EX2 + FMUL + FADD per element) and the `rowsum` FADDs.
2. Tensor-core work: BMM2 = `P~_ij V_j`.
3. HBM traffic: the load of `V_j` — **decode kernel only** (see below).

**Not skipped, ever:** BMM1 = `Q_i K_j^T` and the K load (Sec. 4.1: "even as all QK^T (BMM1) operations are computed, the kernel dynamically skips ... softmax and ... BMM2"). The tile max comes from the full BMM1 output; there is no proxy score. NVIDIA's TRT-LLM blog 16 states explicitly that because BMM1 is still computed "the kernel-level gain is capped to 1.8x".

**Prefill kernel (Sec. 4.1, Fig. 3):** compute-bound, so it skips softmax + BMM2 but **still loads V from HBM** for every tile ("(1) memory bandwidth is not the bottleneck, (2) the prefetch pipeline benefits from predictable access, (3) conditional V loading latency would exceed the savings"). Fig. 3's toy schedule: 18 time units dense -> 14 with two of four loops skipped (50% sparsity).

**Decode kernel (Sec. 4.2, Fig. 4):** memory-bound, so it skips the **V load** (and BMM2). To avoid a scoreboard stall between BMM1 and the conditional V load, it batches `B` consecutive `K_j^T Q` products, then issues V loads only for surviving tiles (needs `B` small shared-memory buffers for the `S_j` tiles). Fig. 4 toy schedule: 38 time units -> 31. For MLA (compute-bound even in decode) the decode kernel also skips softmax.

**Skip-decision cost (Sec. 4):** per tile, one predicate per thread, one warp `VOTE`, one shared-memory `ATOMIC` per warp to combine across the softmax warpgroup; hidden behind existing ops. Measured overhead at 0% sparsity: 0.96-1.00x of baseline (Table 5, Sec. 5.3).

### 1.3 Prefill and decode

Applied to **both** phases with **separate thresholds** (abstract; Sec. 5.1; Table 2 has prefill-only, decode-only, and prefill+decode columns). For decode the "query block" is a single row, so the test is per key block for that one query.

### 1.4 Tile size, tile-level vote, causal handling, sparsity definition

The paper gives none of these. From NVIDIA's reference implementation (ModelOpt `modelopt/torch/sparsity/attention_sparsity/methods/flash_skip_softmax.py` and the Triton kernel `modelopt/torch/kernels/common/attention/triton_fa.py` + `kernels/sparsity/attention/skip_softmax_helpers.py`):

- **Tile = 128 query rows x 128 key columns** (`br = bc = 128` default; the Triton path "requires the fixed 128-wide KV calibration tile"). Decode: 1 x 128.
- **Tile-level vote = ALL rows must agree.** PyTorch backend: a tile is *kept* if `(block_max - running_max > ln lambda).any(dim=rows)`; Triton: `skip_tile = tl.min(can_skip) == 1`. So one query row with a non-negligible score vetoes the skip for all 128 rows. This matches the paper's warp-VOTE description.
- **Causal:** the causal mask is already folded into the scores (fully-masked tiles never enter the sweep). Sparsity is normalised over the lower-triangular tile count `num_block_rows*(2*num_block_cols - num_block_rows + 1)/2` when `is_causal`.
- **Sparsity** = 1 - (kept tiles)/(valid tiles), measured in-kernel by atomic counters (artifact: "exact attention sparsity percentages").
- **Compared quantity** = scaled logits (after `1/sqrt(d_k)`), natural-log threshold `ln lambda`; the Triton kernel works in log2 space (`SKIP_THRESHOLD_LOG2 = log2(lambda)`, scores pre-multiplied by `log2(e)`), which is the same test.

### 1.5 Error bound (Appendix B)

For a skipped tile, `exp(s_jk - M) <= exp(m~ - m) < lambda` because the running max `m <= M` (global max). Each skipped token's normalised weight `p_jk < lambda` (since `Z >= 1`), total mass per skipped tile `< B_c * lambda`, and

`|| y - y^ || <= delta * V_max < |S| * B_c * lambda * V_max`

where `S` = set of skipped tiles, `delta` = total skipped softmax mass, `V_max` = max value-vector norm. Renormalising over kept tiles only adds an `O(delta^2 V_max)` correction. (Note for idea 1: this bound relies on the *true* tile max; a proxy-based skip loses it.)

---

## 2. Threshold calibration (Sec. 3.2, Algorithm 2, Sec. 5.1, 5.4, Table 6, Table 12, App. A.3)

### 2.1 Observations that motivate it (Fig. 2, on Llama-3.1-8B, RULER NIAH_MULTI / VT / FWE, 8K-64K)

- Fig. 2 left: relative accuracy vs. *observed* sparsity collapses onto one curve across tasks and lengths; flat up to ~60-70% sparsity, then drops sharply. "Accuracy degradation is primarily determined by the sparsity ratio itself."
- Fig. 2 right: a fixed lambda gives very different sparsity at different lengths; text says 75% sparsity needs "lambda ~ 1e-4 for 8K but 1e-5 for 64K" (see the unit caveat in 2.4).

### 2.2 The rule

Eq. 3: **`lambda = a / L`**, `L` = context length, `a` = model- and phase-specific scale factor that depends on the target sparsity. Justification: rows are normalised to sum to 1, so longer rows have smaller average weights and need proportionally smaller thresholds.

**Algorithm 2 (calibration):** for each calibration sample `(x_i, L_i)` and each candidate `lambda_j` in a trial set `Lambda`, measure achieved sparsity `s_ij`; keep points with `s_min <= s_ij <= s_max` ("filter unreliable extremes"); record `(lambda_j * L_i, s_ij)`; fit

**`lambda * L = alpha * exp(beta * s)`**

by least squares. At inference with target sparsity `S`: **`lambda = alpha * exp(beta * S) / L`**. Everything is collected in **one forward pass** (all thresholds share the same scores). Sparsity target is adjustable at runtime with no recalibration. This is the "dynamic threshold": it is dynamic in `L` (and in `S`), not per-layer/head.

**Paper's setup (Sec. 5.1):** ~1000 RULER sequences at 4K/8K/16K/32K/64K to fit `alpha, beta`.

**ModelOpt implementation details (calibrator.py / config.py):** default trial set `[1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 3e-1, 5e-1, 7e-1]`; filter keeps `0.10 <= s <= 0.90`; `scipy.curve_fit` on all individual points (optionally in log space); `SKIP_SOFTMAX_CALIB` defaults: target 0.5/0.5 (prefill/decode), 64 RULER samples, `max_seqlen` 16384, chunk 4096. Fixed-threshold default `SKIP_SOFTMAX_DEFAULT`: prefill `lambda = 1e-3`, decode `1e-4`.

### 2.3 Calibrated values reported

**Table 12 (v3 App. A.3), Llama-3.1-8B, target 50%, calibrating on single RULER subsets:**

| Dataset | a (prefill) | prefill sparsity | a (decode) | decode sparsity |
|---|---|---|---|---|
| niah_single | 920 | 49.98% | 4.6 | 49.11% |
| niah_multikey | 1099 | 46.89% | 11.4 | 47.87% |
| niah_multivalue | 1012 | 47.82% | 11.7 | 48.17% |
| niah_multiquery | 1100 | 46.38% | 9.3 | 48.15% |
| cwe | 1020 | 46.69% | 10.8 | 51.04% |
| qa | 900 | 48.68% | 5.4 | 50.97% |

Text: prefill `a` is stable across tasks; decode `a` is ~2x smaller for retrieval-style tasks (niah_single, qa) because decoding attends narrowly; still, achieved sparsity stays near 50% in all cases, so one mixed-dataset calibration is "sufficient".

**TRT-LLM blog 16, Qwen3-30B-A3B-Instruct-2507, `threshold_scale_factor` (= a) vs. target sparsity:**

| target | prefill a | decode a |
|---|---|---|
| 0.1 | 18.76 | 0.32 |
| 0.2 | 44.37 | 0.86 |
| 0.3 | 104.97 | 2.30 |
| 0.4 | 248.40 | 6.17 |
| 0.5 | 587.18 | 16.52 |
| 0.6 | 1390.63 | 44.26 |
| 0.7 | 3293.04 | 118.62 |
| 0.8 | 7799.91 | 317.99 |
| 0.9 | 18471.56 | 852.20 |

(My derivation: these imply `beta ~ 8.6` for prefill and `~ 9.9` for decode, i.e. a ~e^0.86 = 2.4x increase of `a` per +10% sparsity. Prefill/decode `a` differ by ~35-100x, consistent with ModelOpt's 1e-3 / 1e-4 defaults and Table 12.)

### 2.4 WARNING: the absolute lambda values in the paper are not mutually consistent

Using the Algorithm-1 convention (ModelOpt: compare `block_max - running_max` of scaled logits against `ln lambda`, 128x128 tiles):

- Table 6: fixed `lambda = 1e-3` gives 52.4% at 16K on Llama-3.1-8B -> `a = lambda*L ~ 16`.
- Table 12 / TRT-LLM: `a ~ 900-1100` for the same model, phase and target -> `lambda ~ 0.06` at 16K (60x larger).
- Fig. 2 text: 75% at 8K needs `lambda ~ 1e-4` -> `a ~ 0.8`; but Table 6 says `lambda = 3e-3` at 8K only reaches 57.5%.
- Fig. 7: `lambda = 0.03` at 8K.
- Artifact kernel benchmarks: a "threshold" knob of 0.5-5 (unit undocumented) gives 0-97%.

These cannot all be the same unit. **Treat achieved sparsity as the invariant; never port a lambda between tables, figures, kernels, or your own code.** Calibrate your own `a` by measuring sparsity. (The paper itself only ever claims accuracy as a function of sparsity.)

---

## 3. Key numbers — teammate's claims verified

| Claim | Verdict | Source |
|---|---|---|
| fixed lambda = 1e-3: ~23% at 4K, 38% at 8K, 52% at 16K | **Confirmed**: 23.09 / 37.92 / 52.38 (and 65.72 at 32K, 74.63 at 64K), Llama-3.1-8B | Table 6 (v3) = Table 5 (v1) |
| RULER Llama-3.1-8B, ~50% sparsity: 8K 95.07 -> 94.70 | **Confirmed** | Table 3 (v3) = Table 2 (v1) |
| RULER Llama-3.1-8B, ~50% sparsity: 16K 94.80 -> 94.61 | **Confirmed** | Table 3 |

Caveat: Table 3's "~50%" is the *target* sparsity via calibration; the calibrated achieved sparsities at those lengths are 49.70% (8K) and 52.20% (16K) per Table 6's calibrated row, and Table 3 is prefill-only sparsity (the paper's "Prefill Phase Comparison").

### 3.1 Table 6 — sparsity vs. context length, fixed vs. calibrated (Llama-3.1-8B; parentheses = deviation from target)

| | 4K | 8K | 16K | 32K | 64K |
|---|---|---|---|---|---|
| **Target 50%** | | | | | |
| Fixed lambda = 1e-3 | 23.09 (-26.91) | 37.92 (-12.08) | 52.38 (+2.38) | 65.72 (+15.72) | 74.63 (+24.63) |
| Calibrated lambda = a/L | 54.20 (+4.20) | 49.70 (-0.30) | 52.20 (+2.20) | 46.96 (-3.04) | 48.75 (-1.25) |
| **Target 70%** | | | | | |
| Fixed lambda = 3e-3 | 42.35 (-27.65) | 57.54 (-12.46) | 69.83 (-0.17) | 79.36 (+9.36) | 84.63 (+14.63) |
| Calibrated lambda = a/L | 67.99 (-2.01) | 74.65 (+4.65) | 73.64 (+3.64) | 72.54 (+2.54) | 74.63 (+4.63) |

Text (Sec. 5.4): calibrated error averages 1.2% from target at 50%.

### 3.2 Table 3 — prefill comparison, Llama-3.1-8B-Instruct (RULER by length; LongBench v2 splits)

| Method | 4K | 8K | 16K | 32K | 64K | RULER avg | LB Easy | Hard | Short | Medium | Long | Overall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Dense | 96.16 | 95.07 | 94.80 | 92.33 | 87.69 | 93.21 | 29.7 | 32.5 | 38.3 | 28.8 | 25.0 | 31.4 |
| FlexPrefill | 95.99 | 93.67 | 92.73 | 88.14 | 81.14 | 87.72 | 28.8 | 23.8 | 24.4 | 26.5 | 26.2 | 25.7 |
| MInference | 96.54 | 94.06 | 91.37 | 85.79 | 83.03 | 84.15 | 28.6 | 32.8 | 36.7 | 30.2 | 24.1 | 31.2 |
| XAttention | 96.37 | 94.47 | 94.48 | 91.91 | 85.01 | 92.44 | 29.2 | 31.5 | 38.3 | 26.0 | 26.9 | 30.6 |
| BLASST (~50%) | 96.17 | 94.70 | 94.61 | 91.81 | 87.06 | 92.87 | 30.7 | 32.5 | 38.3 | 29.8 | 25.0 | 31.8 |

(The "RULER avg" column is copied as printed; 84.15 for MInference is not the arithmetic mean of its row — paper's number, not mine.)

### 3.3 Table 2 — main results at 50% / 75% target sparsity

| Model | Target | Prefill: RULER-32K | Prefill: LongBench | Decode: MATH500 | Decode: AIME24 | Decode: GPQA | P+D: RULER-32K | P+D: LongBench |
|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | Dense | 92.33 | 31.40 | 73.40 | 46.66 | 46.71 | 92.33 | 31.40 |
| | 50% | 91.81 | 31.80 | 73.71 | 46.15 | 46.31 | 91.79 | 32.40 |
| | 75% | 91.67 | 31.80 | 73.89 | 46.01 | 45.95 | 91.67 | 31.80 |
| Qwen3-8B | Dense | 91.90 | 33.60 | 95.87 | 75.00 | 61.21 | 91.90 | 33.60 |
| | 50% | 92.08 | 35.10 | 96.23 | 76.50 | 61.56 | 92.07 | 33.30 |
| | 75% | 92.11 | 34.40 | 96.07 | 75.33 | 61.51 | 91.74 | 33.10 |

Decode-phase reasoning numbers for Llama use the DeepSeek-R1-distilled Llama-3.1-8B (Sec. 5.1). Sampling: long-context tasks greedy, single run; reasoning tasks T=0.6, top-p 0.95, 10 samples (20 for AIME), best-of-N by majority vote.

### 3.4 Table 4 — decode comparison, Qwen3-8B (~50% target)

| Method | RULER-32K | LongBench | MATH500 | AIME24 | LiveCodeBench | GPQA | Avg |
|---|---|---|---|---|---|---|---|
| Dense | 91.90 | 33.60 | 95.87 | 75.00 | 53.83 | 61.21 | 68.57 |
| Quest | 56.23 | 30.30 | 94.18 | 71.50 | 52.17 | 60.12 | 60.75 |
| RocketKV | 87.89 | 30.60 | 95.88 | 73.54 | 53.10 | 60.50 | 66.91 |
| BLASST ~50% | 91.55 | 33.90 | 96.23 | 76.50 | 54.15 | 61.51 | 68.97 |

### 3.5 Other accuracy tables

**Table 7 — composition, Qwen3-8B, 16K:** Dense/Dense 93.22 RULER, 29.4 LongBench; XAttention/Dense 92.99 (-0.23), 29.1; XAttention/BLASST 92.89 (-0.33), 28.8; Dense/RocketKV 92.72 (-0.50), 30.0; BLASST/RocketKV 92.60 (-0.62), 29.4.

**Table 8 — RepoQA, Qwen3-Coder-30B-A3B-Instruct (achieved sparsity P / D):** 16K dense 0.897; BLASST prefill (64.1% / 0) 0.904; prefill+decode (64.1% / 48.4%) 0.882. 200K dense 0.850; prefill (57.5% / 0) 0.841; prefill+decode (57.5% / 40.8%) 0.838.

**Table 9 — Qwen3-30B-A3B-Instruct, LongBench V1 / V2 overall vs. target sparsity:** 0%: 47.77 / 36.28; 50%: 47.43 / 38.14; 60%: 47.47 / 39.53; 70%: 47.21 / 39.53; 80%: 46.50 / 37.21; 90%: 45.97 / 37.21.

**Table 10 — Llama-3.1-70B-Instruct, RULER-hard 8K / 16K:** 0%: 97.40 / 99.06; 20%: 97.38 / 98.98; 40%: 97.31 / 98.80; 60%: 97.20 / 98.59; 80%: 97.07 / 98.28.

**Table 11 — DeepSeek-R1 NVFP4 (MLA), GPQA Diamond / MMLU Pro / LiveCodeBench:** 0%: 0.7071 / 0.8302 / 0.5735; 50%: 0.7121 / 0.8283 / 0.5691; 60%: 0.7109 / 0.8266 / 0.5677.

**Fig. 8 (text only):** on RULER-16K Qwen3-8B at 70-90% sparsity BLASST degrades more gracefully than XAttention; no numbers in text.

**Fig. 6 (text only):** sparsity-aware fine-tuning (ProLong curriculum, BLASST active in forward; skipped tiles get no gradient) cuts accuracy loss by "up to 1.7x" in the 50-75% range; no table.

**TRT-LLM blog, Qwen3-30B-A3B-Instruct-2507, H200, LongBench V1 / V2 vs target sparsity:** 0.0: 47.79 / 35.81; 0.1: 47.92 / 40.00; 0.2: 47.80 / 38.14; 0.3: 47.84 / 38.60; 0.4: 47.84 / 39.53; 0.5: 46.87 / 37.67; 0.6: 46.45 / 36.28; 0.7: 45.56 / 34.88; 0.8: 43.38 / 36.74; 0.9: 39.84 / 39.07. (V2 has ~200 samples; blog says use V1 for trends.)

---

## 4. Kernel timing and speedup vs. sparsity (for the irreducible-fraction fit)

### 4.1 Paper Table 5 (v3) — speedup over "FlashAttention-3 BF16" baseline

Config (caption): prefill batch 1, 64K sequence; B200 decode batch 148, 32K; H200 decode batch 128, 16K. Heads / head dim are **not stated in v3**; v1's Table 4 states "1 Q head, 1 KV head, head dim 128, batch 148, 32K" for prefill and "32 Q heads, 4 KV heads, 128 head dim, batch 148, 32K" for decode — whether v3 kept those is unverified.

| B200 prefill sparsity | speedup | H200 prefill sparsity | speedup |
|---|---|---|---|
| 0.0% | 1.00x | 0.0% | 1.00x |
| 38.9% | 1.25x | 23.8% | 1.08x |
| 49.2% | 1.33x | 49.2% | 1.27x |
| 63.0% | 1.43x | 57.3% | 1.35x |
| 71.9% | 1.52x | 71.0% | 1.52x |
| 80.8% | 1.61x | 79.5% | 1.64x |
| 88.9% | 1.71x | 88.5% | 1.78x |
| 94.2% | 1.77x | 92.0% | 1.84x |

| B200 decode sparsity | speedup | H200 decode sparsity | speedup |
|---|---|---|---|
| 0.0% | 0.98x | 0.0% | 0.96x |
| 36.9% | 1.18x | 23.8% | 1.08x |
| 46.7% | 1.25x | 43.7% | 1.20x |
| 61.2% | 1.34x | 59.4% | 1.31x |
| 73.2% | 1.48x | 70.5% | 1.40x |
| 82.6% | 1.64x | 78.4% | 1.47x |
| 87.0% | 1.71x | 87.5% | 1.56x |
| 92.0% | 1.79x | — | — |

Sec. 5.3 summary: ~1.33x prefill / 1.25x decode at ~50% on Blackwell; 1.52x / 1.48x at ~70%; Hopper prefill up to 1.52x at 71.0%.

**v1 Table 4 (B200 only; prefill: batch 148, 1 Q head, 1 KV head, 32K, d=128; decode: batch 148, 32 Q / 4 KV heads, 32K, d=128):** prefill 0.00% 0.99x, 23.24% 1.07x, 32.78% 1.13x, 42.26% 1.18x, 50.67% 1.24x, 63.96% 1.34x, 73.47% 1.41x, 80.35% 1.49x; decode 0.00% 0.98x, 36.91% 1.18x, 46.68% 1.25x, 61.23% 1.34x, 73.15% 1.48x, 82.62% 1.64x, 87.01% 1.71x, 91.99% 1.79x. (v1 text: H200 prefill "up to 1.62x at 74.7%", Fig. 5 image only.) The decode column is identical to v3; the v3 prefill column is a new config and ~8% faster at equal sparsity.

### 4.2 Artifact repo expected outputs — absolute times (same kernel, skipping on vs. off)

Source: `github.com/cameronshinn/blasst-ae-mlsys26` READMEs (`hopper_prefill`, `hopper_decode`, `blackwell`). Config everywhere: **head dim 128, 64 Q heads, 4 KV heads; prefill batch 1; decode batch 64**. Hopper prefill = TRT-LLM `fmha_v2`, Hopper decode = `XQA`, Blackwell = `trtllm-gen` via FlashInfer. Baseline = the same dense kernel with skipping disabled ("NoSkip"). The "threshold" column is the repo's sweep knob (unit undocumented; not the paper's lambda). The 16K prefill FLOP count (14.03 ms at 627 TFLOP/s = 8.8 TFLOP = 4·L²·H·d) shows these are **non-causal** benchmarks; the TRT-LLM blog's 7403 us for the same shape is the causal number.

**H200 prefill BF16, 16K** (NoSkip 14.029 ms, 626.99 TFLOP/s):

| thr | sparsity | ms | TFLOP/s | speedup |
|---|---|---|---|---|
| 0.5 | 0.25% | 15.261 | 576.38 | 0.919 |
| 0.6 | 2.11% | 15.179 | 579.49 | 0.924 |
| 0.8 | 13.89% | 14.306 | 614.87 | 0.981 |
| 0.9 | 20.68% | 13.797 | 637.51 | 1.017 |
| 1.0 | 28.47% | 13.116 | 670.63 | 1.070 |
| 1.05 | 36.75% | 12.475 | 705.12 | 1.125 |
| 1.1 | 44.49% | 11.916 | 738.18 | 1.177 |
| 1.2 | 51.39% | 11.370 | 773.64 | 1.234 |
| 1.5 | 62.71% | 10.431 | 843.30 | 1.345 |
| 1.75 | 71.23% | 9.730 | 904.00 | 1.442 |
| 2.0 | 74.67% | 9.419 | 933.84 | 1.489 |
| 2.25 | 80.32% | 8.984 | 979.13 | 1.562 |
| 2.5 | 82.64% | 8.799 | 999.70 | 1.594 |
| 3.0 | 86.50% | 8.499 | 1034.97 | 1.651 |
| 4.0 | 90.78% | 8.285 | 1061.70 | 1.693 |
| 5.0 | 93.69% | 8.170 | 1076.63 | 1.717 |

**H200 prefill BF16, 64K** (NoSkip 226.371 ms, 621.71 TFLOP/s):

| thr | sparsity | ms | TFLOP/s | speedup |
|---|---|---|---|---|
| 0.5 | 10.23% | 234.891 | 599.16 | 0.964 |
| 0.6 | 23.78% | 217.560 | 646.89 | 1.040 |
| 0.8 | 49.24% | 185.137 | 760.18 | 1.223 |
| 0.9 | 57.27% | 174.456 | 806.72 | 1.298 |
| 1.0 | 64.56% | 164.684 | 854.59 | 1.375 |
| 1.05 | 70.97% | 156.194 | 901.04 | 1.449 |
| 1.1 | 75.80% | 149.838 | 939.27 | 1.511 |
| 1.2 | 79.52% | 144.912 | 971.20 | 1.562 |
| 1.5 | 84.85% | 138.043 | 1019.52 | 1.640 |
| 1.75 | 88.45% | 133.647 | 1053.05 | 1.694 |
| 2.0 | 89.84% | 131.919 | 1066.85 | 1.716 |
| 2.25 | 92.06% | 129.119 | 1089.98 | 1.753 |
| 2.5 | 92.96% | 128.499 | 1095.25 | 1.762 |
| 3.0 | 94.47% | 127.056 | 1107.68 | 1.782 |
| 4.0 | 96.19% | 125.730 | 1119.36 | 1.800 |
| 5.0 | 97.40% | 125.000 | 1125.90 | 1.811 |

Note: the sparsity points of paper Table 5 (H200 prefill) are exactly these 64K rows (23.8, 49.2, 57.3, 71.0, 79.5, 88.5, 92.0%), but Table 5's speedups are 4-5% higher (1.08 vs 1.040, 1.27 vs 1.223, 1.52 vs 1.449, 1.84 vs 1.753) — Table 5 normalises to FA3, the artifact to the same kernel with skipping off. Use the artifact normalisation for the irreducible-fraction fit.

**B200 prefill BF16, 16K** (baseline 3.647 ms; BLASST thr 0 = 3.863 ms, 0.77%):

| thr | sparsity | ms | TFLOP/s | speedup |
|---|---|---|---|---|
| 0.5 | 0.77% | 3.925 | 1120.51 | 0.929 |
| 0.6 | 0.79% | 3.909 | 1125.05 | 0.933 |
| 0.8 | 1.86% | 3.854 | 1141.06 | 0.946 |
| 0.9 | 4.36% | 3.788 | 1161.16 | 0.963 |
| 1.0 | 8.91% | 3.707 | 1186.45 | 0.984 |
| 1.05 | 11.74% | 3.696 | 1189.91 | 0.987 |
| 1.1 | 14.66% | 3.639 | 1208.49 | 1.002 |
| 1.2 | 20.20% | 3.544 | 1241.05 | 1.029 |
| 1.5 | 34.57% | 3.323 | 1323.34 | 1.097 |
| 1.75 | 43.76% | 3.195 | 1376.59 | 1.141 |
| 2.0 | 50.85% | 3.064 | 1435.22 | 1.190 |
| 2.25 | 56.57% | 2.951 | 1490.18 | 1.236 |
| 2.5 | 61.30% | 2.871 | 1531.90 | 1.270 |
| 3.0 | 68.76% | 2.737 | 1607.07 | 1.333 |
| 4.0 | 78.43% | 2.594 | 1695.26 | 1.406 |
| 5.0 | 84.45% | 2.506 | 1754.78 | 1.455 |
| 7.0 | 91.15% | 2.370 | 1855.97 | 1.539 |
| 10.0 | 95.75% | 2.280 | 1929.09 | 1.600 |
| 100.0 | 98.46% | 2.224 | 1977.88 | 1.640 |

**B200 prefill BF16, 64K** (baseline 55.502 ms; BLASST thr 0 = 59.160 ms, 0.19%):

| thr | sparsity | ms | TFLOP/s | speedup |
|---|---|---|---|---|
| 0.5 | 0.34% | 58.565 | 1201.55 | 0.948 |
| 0.6 | 2.01% | 57.610 | 1221.46 | 0.963 |
| 0.8 | 16.94% | 54.072 | 1301.39 | 1.026 |
| 0.9 | 27.98% | 51.574 | 1364.43 | 1.076 |
| 1.0 | 38.91% | 49.081 | 1433.73 | 1.131 |
| 1.05 | 43.76% | 47.890 | 1469.39 | 1.159 |
| 1.1 | 47.83% | 46.897 | 1500.49 | 1.183 |
| 1.2 | 54.23% | 45.257 | 1554.88 | 1.226 |
| 1.5 | 66.18% | 42.156 | 1669.24 | 1.317 |
| 1.75 | 72.01% | 40.537 | 1735.93 | 1.369 |
| 2.0 | 75.99% | 39.423 | 1784.94 | 1.408 |
| 2.25 | 79.00% | 38.588 | 1823.59 | 1.438 |
| 2.5 | 81.63% | 37.871 | 1858.11 | 1.466 |
| 3.0 | 85.24% | 36.845 | 1909.88 | 1.506 |
| 4.0 | 90.04% | 35.446 | 1985.22 | 1.566 |
| 5.0 | 92.86% | 34.562 | 2036.00 | 1.606 |
| 7.0 | 95.94% | 33.615 | 2093.37 | 1.651 |
| 10.0 | 97.88% | 32.975 | 2134.01 | 1.683 |
| 100.0 | 99.61% | 32.395 | 2172.23 | 1.713 |

(Paper Table 5 B200 prefill at 38.9% says 1.25x; the artifact at 38.91% says 1.131x; at ~72%: 1.52x vs 1.369x. Different baselines — Table 5's is FA3, which is a Hopper kernel.)

**H200 decode BF16 (batch 64, 64 Q / 4 KV heads, d=128).** 16K: NoSkip 0.486 ms, 4.319 TB/s; 64K: NoSkip 1.910 ms, 4.392 TB/s.

| thr | 16K sparsity | 16K ms | 16K speedup | 64K sparsity | 64K ms | 64K speedup |
|---|---|---|---|---|---|---|
| 0.4 | 0.25% | 0.486 | 1.000 | 7.58% | 1.818 | 1.050 |
| 0.5 | 2.67% | 0.481 | 1.011 | 22.90% | 1.690 | 1.130 |
| 0.6 | 9.50% | 0.467 | 1.041 | 40.02% | 1.554 | 1.228 |
| 0.7 | 19.65% | 0.446 | 1.089 | 54.59% | 1.448 | 1.318 |
| 0.8 | 31.24% | 0.422 | 1.150 | 65.64% | 1.375 | 1.389 |
| 0.9 | 42.25% | 0.401 | 1.212 | 73.81% | 1.322 | 1.444 |
| 1.0 | 51.63% | 0.385 | 1.262 | 79.79% | 1.285 | 1.486 |
| 1.2 | 66.17% | 0.361 | 1.347 | 87.39% | 1.240 | 1.540 |
| 1.5 | 79.26% | 0.340 | 1.431 | 93.20% | 1.207 | 1.583 |
| 2.0 | 89.39% | 0.325 | 1.495 | 96.90% | 1.186 | 1.610 |
| 5.0 | 98.00% | 0.314 | 1.549 | 99.50% | 1.173 | 1.627 |

(Thresholds <= 0.3 give 0% sparsity and ~1.00x at both lengths. FP8 columns also exist in the README: 16K NoSkip 0.262 ms; 64K NoSkip 1.016 ms; FP8 saturates lower, e.g. 16K 79.11% -> 1.328x, 64K 93.13% -> 1.466x.)

**B200 decode (batch 64, 16K / 64K):** times are reported (16K baseline 0.359 ms -> 0.231 ms at the largest threshold = 1.553x; 64K baseline 1.263 ms -> 0.763 ms = 1.655x) but the README notes the sparsity-collection pass segfaults and the sparsity column is hard-coded to 0% — unusable for a sparsity fit.

**TRT-LLM blog 16 dense baselines (causal, batch 1 prefill / batch 64 decode, 64 Q / 4 KV heads, d=128):** prefill H200 16K BF16 7403 us (594 TFLOP/s), FP8 5157 us; H200 64K BF16 115301 us, FP8 80550 us; B200 16K BF16 4273 us (1029 TFLOP/s), FP8 2886 us; B200 64K BF16 67775 us, FP8 43399 us. Decode H200 16K BF16 498 us (4.31 TB/s), FP8 266 us; H200 64K BF16 1962 us, FP8 1045 us; B200 16K BF16 303 us (7.08 TB/s), FP8 196 us; B200 64K BF16 1209 us, FP8 755 us. Speedup curves in that blog are images only.

### 4.3 My fit of the irreducible fraction (not the paper's; least squares `T(s)/T0 = c - m*s` over all points)

| Config (artifact, same-kernel baseline) | T0 | fit | T(s=1)/T0 | implied max speedup |
|---|---|---|---|---|
| H200 prefill BF16 16K | 14.029 ms | 1.076 - 0.533 s | 0.543 | 1.84x |
| H200 prefill BF16 64K | 226.371 ms | 1.060 - 0.524 s | 0.536 | 1.87x |
| B200 prefill BF16 16K | 3.647 ms | 1.067 - 0.458 s | 0.610 | 1.64x |
| B200 prefill BF16 64K | 55.502 ms | 1.061 - 0.468 s | 0.593 | 1.69x |
| H200 decode BF16 16K | 0.486 ms | 0.998 - 0.377 s | 0.622 | 1.61x |
| H200 decode BF16 64K | 1.910 ms | 0.984 - 0.386 s | 0.598 | 1.67x |
| H200 decode FP8 16K | 0.262 ms | 1.055 - 0.384 s | 0.671 | 1.49x |
| H200 decode FP8 64K | 1.016 ms | 1.026 - 0.386 s | 0.640 | 1.56x |
| Paper Table 5 B200 prefill (1/speedup) | — | 0.988 - 0.457 s | 0.531 | 1.88x |
| Paper Table 5 H200 prefill | — | 1.026 - 0.517 s | 0.510 | 1.96x |
| Paper Table 5 B200 decode | — | 1.031 - 0.502 s | 0.529 | 1.89x |
| Paper Table 5 H200 decode | — | 1.037 - 0.457 s | 0.580 | 1.72x |

Reading: **roughly 53-61% of dense prefill attention time (BMM1 + K load + skip-check overhead + un-hidden pipeline bubbles) is never removed by BLASST**, consistent with NVIDIA's "capped to 1.8x" statement (1/1.8 = 0.556). The intercepts above 1.0 (1.06-1.08) are the fixed overhead the paper reports as 0.93-1.00x at ~0% sparsity. This 53-61% is the ceiling idea (1) attacks.

### 4.4 End-to-end (Fig. 5 + TRT-LLM blog), Qwen3-30B-A3B-Instruct-2507, TRT-LLM in-flight batching

LongBench V1 (avg ISL 10K, OSL 6, concurrency 64), TTFT / TPOT in ms, H200 then B200: target 0.0: 9419.61 / 1731.80, 4854.55 / 928.45; 0.5: 8847.43 / 1633.08, 4548.07 / 873.80; 0.7: 8134.72 / 1508.60, 4385.12 / 846.64; 0.9: 8130.39 / 1516.16, 4150.44 / 798.93. Paper text: "1.1x speedup in TTFT and TPOT with only a marginal drop in LongBench V1". LongBench V2 (ISL 130K, OSL 200, concurrency 1) H200 TTFT 16486.70 -> 12507.95 ms at 0.9 (1.32x); B200 6990.59 -> 5276.67 (1.32x); TPOT essentially flat (9.34 -> 8.58 ms H200). NVIDIA developer blog: "estimated 1.4x" prefill at 128K and "projected 1.36x" decode for Llama 3.3 70B on GB200 NVL72 (no derivation given).

---

## 5. Models and benchmarks evaluated

- **LLMs:** Llama-3.1-8B-Instruct (main prefill model); Qwen3-8B(-Instruct) (main decode/reasoning model); DeepSeek-R1-distilled Llama-3.1-8B (long-generation reasoning); Qwen3-30B-A3B-Instruct(-2507) (Table 9, Fig. 5 end-to-end); Qwen3-Coder-30B-A3B-Instruct (Table 8, 200K); Llama-3.1-70B-Instruct (Table 10); DeepSeek-R1 NVFP4 (MLA, Table 11). Attention variants claimed: MHA, GQA, MQA, MLA.
- **Context lengths:** 4K-64K (RULER), 32K (Table 2), 16K/200K (RepoQA), ~130K average (LongBench V2 serving).
- **Benchmarks:** RULER (incl. hard subset, NIAH, VT, FWE, CWE, QA), LongBench v1 and v2, MATH500, AIME 2024, GPQA (Diamond), LiveCodeBench, MMLU Pro, RepoQA. NeMo-Skills for reasoning evals.
- **Decode results:** yes — Table 2 (decode-only and prefill+decode), Table 4, Table 8, Table 11, Table 5 kernels, Fig. 5 TPOT.
- **Multimodal / diffusion:** **none in the paper.** TRT-LLM's "VisualGen" docs (draft) list Skip Softmax as supported for diffusion transformers (Wan 2.2 example; per-layer `ignore`, `disabled_until_timestep` gating, each transformer component calibrated separately; TRTLLM and CUTEDSL backends) but give no quality or speed numbers; ModelOpt has `examples/diffusers/sparsity/wan22_skip_softmax.py`.
- **Baselines compared:** prefill — MInference, FlexPrefill, XAttention; decode — Quest, RocketKV (each at its paper's best config).

---

## 6. What is released

| Component | Where | Notes |
|---|---|---|
| Production kernels | TensorRT-LLM (`github.com/NVIDIA/TensorRT-LLM`), paper footnote 1 | Hopper prefill in `fmha_v2`, Hopper decode in `XQA`, Blackwell prefill+decode in `trtllm-gen`. Config `SkipSoftmaxAttentionConfig(threshold_scale_factor={"prefill": 1000.0, "decode": 500.0})` or `target_sparsity` (needs calibration metadata in checkpoint `config.json`: `a*exp(b*target_sparsity)` NumExpr formula). TRTLLM attention backend only; BF16 and FP8; works with KV reuse and chunked prefill. |
| FlashInfer | `flashinfer.prefill.trtllm_batch_context_with_kv_cache`, `flashinfer.decode.trtllm_batch_decode_with_kv_cache`, `trtllm_ragged_attention_deepseek` | Parameter `skip_softmax_threshold_scale_factor: Optional[float] = None`; "actual threshold = scale factor / context length"; cites arXiv 2512.12087; trtllm-gen backend (Blackwell). Incompatible with `enable_block_sparse_attention`. |
| MLSys artifact | `github.com/cameronshinn/blasst-ae-mlsys26` (Apache 2.0) | Kernel benchmarks only (no accuracy). Submodules: TensorRT-LLM (NVIDIA main + `pengbowang-nv` fork for Hopper decode), FlashInfer + a `cameronshinn/flashinfer` stats fork. Container `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6`. "Closed sm100 binaries" used to measure sparsity on Blackwell. Needs H200 / B200. |
| ModelOpt (TensorRT-Model-Optimizer) | `github.com/NVIDIA/Model-Optimizer`, example `examples/llm_sparsity/attention_sparsity/` (`hf_sa.py`, `README.md`, `download_ruler_data.sh`) | Method `flash_skip_softmax`. **pytorch backend**: patches `F.softmax` (needs `attn_implementation="eager"`), simulates 128x128 tile skipping on materialised scores — easiest way to measure sparsity / accuracy on an 8B HF model with no kernel. **triton backend**: fused FlashAttention kernel `modelopt/torch/kernels/common/attention/triton_fa.py` with in-kernel skip (`APPLY_SKIP_SOFTMAX`, `SKIP_THRESHOLD_LOG2`), skips softmax + V load + BMM2 per tile, paged-KV decode path, and a backward kernel with skipping (sparsity-aware training). RULER calibration (`SKIP_SOFTMAX_CALIB`, `SKIP_SOFTMAX_TRITON_CALIB`), export of `sparse_attention_config` to `config.json`, vLLM plugin. |
| Supported GPUs | NVIDIA kernels: Hopper and Blackwell only (paper, docs, artifact). | The ModelOpt Triton path has no documented architecture restriction (Triton generally needs Ampere+); **not verified on non-Hopper hardware.** |

Note: the ModelOpt README mis-expands BLASST as "Block-Level Adaptive Structured Sparse Training"; the paper's expansion is "BLocked Attention Sparsity via Softmax Thresholding".

---

## 7. Limitations, future work, and what bears on the three project ideas

**Stated by the paper**
- BMM1 is always computed; prefill still loads V (Sec. 4.1, with the remark that V skipping could be added "if future workloads/hardware shift to a memory-bound regime").
- Accuracy cliff beyond ~60-70% observed sparsity (Sec. 3.2, Fig. 2); NVIDIA blog calls >60% the "danger zone", especially multikey NIAH.
- Fixed thresholds are unusable across lengths (Table 6); calibration is required.
- Fig. 7 caption: "Substantial head-level and layer-level variance motivates adaptive thresholding strategies." Sec. 5.6 then argues the *uniform* threshold already "automatically adapts" (prunes more where attention is concentrated) — the paper leaves adaptive per-layer/head thresholds as motivation, never tests them.
- Table 12: decode `a` varies ~2.5x by task (4.6-11.7); prefill `a` is stable (900-1100).
- Conclusion: future = "hardware-aware sparse patterns, learned sparsity through training, and adaptive hybrid methods."
- TRT-LLM blog: decode gains are smaller at small batch / few heads (TP), "actively optimizing"; FP8 gains smaller than BF16.
- App. A.4: processing tiles in reverse order (local window + sinks first) to seed the running max earlier has negligible effect on VT, "noticeable" on FWE; BLASST is "robust to processing order".

**Idea 1 — score tiles with downsampled Q/K to skip BMM1.** Directly targets the 53-61% irreducible fraction (Sec. 4.3 here). But the paper's identity is "no proxy scores, no pre-computation" (contributions; Sec. 2.1 vs. SpargeAttention's "separate prediction step"; Sec. 5.6/Fig. 8 credits "actual softmax statistics rather than proxy scores" for graceful degradation vs. XAttention). Also: a skipped tile cannot update the running max, and the Appendix B guarantee (`p_jk < lambda`) no longer holds when the tile max is estimated. Design question to answer in the report: how often does the downsampled max under-estimate the true tile max by more than the margin to `ln lambda`?

**Idea 2 — reuse the skip mask across neighbouring layers.** Not discussed anywhere in the paper (no cross-layer similarity analysis; Fig. 7 only shows per-layer sparsity *levels*). Nothing pre-empts it. Caveat: the receiving layer then runs a static block mask without a threshold test, so its error is unbounded by Appendix B; and the sending layer's mask depends on its own running max order.

**Idea 3 — per-layer / per-head / per-modality thresholds.** Fig. 7 explicitly motivates it; Table 12 shows task-dependent `a`; per-*phase* thresholds (prefill vs decode, ~100x apart) already exist; TRT-LLM VisualGen already has per-layer `ignore` lists and timestep gating for diffusion, and ModelOpt calibrates each diffusion transformer component separately — so "per-component" exists in tooling but no paper result quantifies per-layer/head thresholds for LLMs.

**Observation (mine, from the reference code):** the tile vote is all-128-rows-must-agree, so a single outlier query row vetoes a whole 128x128 tile; the paper has no analysis of this, and it is a cheap axis to study (e.g. 64- or 32-row tiles, or per-row skipping in a Triton kernel).

---

## 8. Related work cited (training-free / block-sparse and neighbours), with arXiv IDs

Compute-side sparsity (Sec. 2.1): MInference 1.0 — 2407.02490; XAttention — 2503.16428; FlexPrefill — 2502.20766; SpargeAttn — 2502.18137 ("most similar design"; prefill-only, separate prediction step, no V-load skipping); SeerAttention-R — 2506.08889 (trained gates); FLASH-D — 2505.14201 (same online-softmax math, for hardware accelerators); DuoAttention — 2410.10819; Retrieval heads (Wu et al. 2025, no arXiv ID in refs); Sparse Transformer — 1904.10509; Longformer — 2004.05150; BigBird — 2007.14062.

Memory-side / KV sparsity (Sec. 2.2): H2O — 2306.14048; Quest — 2406.10774; RocketKV — 2502.14051; StreamingLLM — 2309.17453; InfLLM — 2402.04617; TOVA — 2401.06104; Rectified Sparse Attention — 2506.04108; TidalDecode — 2410.05076; RPC — 2505.13866; Loki (no arXiv ID in refs); ThinK — 2407.21018; KV compression for hyper-scaling (Lancucki et al. 2025, no ID in refs).

New attention variants (Sec. 2.3): NSA — 2502.11089; DSA / DeepSeek-V3.2-Exp (no ID in refs); Gated attention — 2505.06708; Mamba — 2312.00752; MLA / DeepSeek-V2 — 2405.04434.

Infrastructure: FlashAttention — 2205.14135; FlashAttention-3 — 2407.08608; FlashAttention-4 — 2603.05451; FlashInfer — 2501.01005; RULER — 2404.06654; RepoQA — 2406.06025; ProLong — Gao et al. 2025a (no ID in refs).

Table 1 feature matrix (accelerates prefill / decode / no training / no pre-computation): H2O x/v/v/v; SnapKV x/v/v/v; RocketKV x/v/v/x; Quest x/v/v/x; DuoAttention v/v/x/v; DSA v/v/x/v; MInference v/x/v/x; SpargeAttention v/x/v/x; XAttention v/x/v/x; BLASST v/v/v/v.

---

## 9. Numbers to reproduce

| # | Quantity | Paper value | Setup | Source |
|---|---|---|---|---|
| R1 | Sparsity at fixed lambda = 1e-3 | 23.09 / 37.92 / 52.38 / 65.72 / 74.63 % | Llama-3.1-8B, prefill, 4K/8K/16K/32K/64K RULER, 128x128 tiles, all-rows vote | Table 6 |
| R2 | Sparsity at fixed lambda = 3e-3 | 42.35 / 57.54 / 69.83 / 79.36 / 84.63 % | same | Table 6 |
| R3 | Calibrated lambda = a/L, target 50% | 54.20 / 49.70 / 52.20 / 46.96 / 48.75 % | same | Table 6 |
| R4 | Calibrated a for 50% prefill | ~900-1100 (unit caveat, Sec. 2.4) | Llama-3.1-8B, per RULER subset | Table 12 |
| R5 | RULER dense vs BLASST ~50% | 8K 95.07 -> 94.70; 16K 94.80 -> 94.61; 32K 92.33 -> 91.81; 64K 87.69 -> 87.06 | Llama-3.1-8B-Instruct, prefill-only, greedy | Table 3 |
| R6 | LongBench v2 overall | 31.4 -> 31.8 (50%), 31.8 (75%) | Llama-3.1-8B prefill | Tables 2, 3 |
| R7 | Accuracy cliff | flat to ~60-70% observed sparsity, then sharp drop | Llama-3.1-8B, NIAH_MULTI/VT/FWE | Fig. 2 |
| R8 | Per-layer/head sparsity spread | "substantial" (image only) | Llama-8B, 8K NIAH sample, lambda = 0.03 | Fig. 7 |
| R9 | Kernel speedup at ~50% / ~72% | prefill 1.33x / 1.52x (B200), 1.27x / 1.52x (H200); decode 1.25x / 1.48x (B200) | 64K prefill b=1; decode 32K b=148 (B200), 16K b=128 (H200); vs FA3 BF16 | Table 5 |
| R10 | Irreducible fraction T(s->1)/T0 | 0.54 (H200 prefill), 0.59-0.61 (B200 prefill), 0.60-0.62 (H200 decode BF16) — my fit | 64 Q / 4 KV heads, d=128, same-kernel baseline | artifact READMEs, Sec. 4.3 |
| R11 | Overhead at 0% sparsity | 0.96-1.00x (paper); 0.93-0.95x (artifact, B200) | — | Table 5; artifact |
| R12 | Prefill vs decode a ratio | ~100x (Llama: ~1000 vs ~5-12; Qwen3-30B: 587 vs 16.5 at 50%) | — | Table 12; TRT-LLM blog |

---

## 10. Unverified / not determinable

1. **Absolute lambda units** across Fig. 2, Table 6, Table 12, Fig. 7 and the artifact "threshold" knob are mutually inconsistent (Sec. 2.4); the paper's own simulation code is not released, so which convention produced Table 6 vs Table 12 cannot be checked. ModelOpt's `lambda = 1e-3` default is numerically consistent with Table 6.
2. **Heads / head dim / batch for v3 Table 5**: not stated in v3; v1 Table 4's config (1 Q + 1 KV head prefill; 32 Q / 4 KV decode) may or may not carry over. v3's baseline is stated as "FlashAttention-3 BF16" even on B200, where FA3 is not a native kernel; speedups differ from the artifact's same-kernel numbers by 4-10%.
3. **Fig. 2 (right) curve values, Fig. 5-9 data points, Fig. 7 per-layer/head numbers**: images only; nothing extractable beyond captions.
4. **Sparsity-aware training**: only "up to 1.7x less degradation" and "ProLong curriculum, fixed threshold" — no table, no threshold, no checkpoint.
5. **Exact tile size used by the production CUDA kernels** (fmha_v2 / XQA / trtllm-gen): not stated; 128x128 is from the ModelOpt reference and the artifact's "fixed 128-wide KV tile".
6. **Whether sparsity in Table 5/Table 6 is normalised over causal tiles only**: ModelOpt does so when `is_causal`; the artifact kernel benchmarks are non-causal (FLOP-count evidence); the paper is silent.
7. **B200 decode sparsity** in the artifact is hard-coded 0% (README-acknowledged bug).
8. **FlashInfer support for Hopper**: the API cites trtllm-gen (Blackwell); Hopper availability in FlashInfer was described as "will also be available" (issue #2306) in the blog and "available" in the later developer blog — not confirmed.
9. **Whether the ModelOpt Triton kernel runs on Ampere / consumer GPUs**: no documented restriction found, not tested.
10. **Calibration-set size for Table 6/12** ("~1000 sequences") vs ModelOpt default (64 samples): unknown which produced which table.

---

## Sources

- arXiv abs/html v1 and v3: https://arxiv.org/abs/2512.12087 , https://arxiv.org/html/2512.12087v3 , https://arxiv.org/html/2512.12087v1 (local copies in the session scratchpad: `blasst_v3.txt`, `blasst_v1.txt`, `blasst_v3.pdf`)
- MLSys 2026 proceedings page: https://proceedings.mlsys.org/paper_files/paper/2026/hash/c6ee784cbe46d854843e4c883a3321ef-Abstract-Conference.html
- Artifact: https://github.com/cameronshinn/blasst-ae-mlsys26 (READMEs under `hopper_prefill/`, `hopper_decode/`, `blackwell/`)
- TRT-LLM tech blog 16: https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.html
- TRT-LLM sparse-attention feature doc: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/sparse-attention.md ; VisualGen: https://nvidia.github.io/TensorRT-LLM/visual-gen/features/sparse-attention.html
- NVIDIA developer blog (16 Dec 2025): https://developer.nvidia.com/blog/accelerating-long-context-inference-with-skip-softmax-in-nvidia-tensorrt-llm/
- FlashInfer API: https://docs.flashinfer.ai/generated/flashinfer.prefill.trtllm_batch_context_with_kv_cache.html , https://docs.flashinfer.ai/generated/flashinfer.decode.trtllm_batch_decode_with_kv_cache.html
- ModelOpt: https://github.com/NVIDIA/Model-Optimizer — `examples/llm_sparsity/attention_sparsity/README.md`, `modelopt/torch/sparsity/attention_sparsity/{config.py, methods/flash_skip_softmax.py, methods/triton_skip_softmax.py, calibration/calibrator.py}`, `modelopt/torch/kernels/common/attention/triton_fa.py`, `modelopt/torch/kernels/sparsity/attention/skip_softmax_helpers.py`
- UC Davis news (Best Paper): https://ece.ucdavis.edu/news/new-method-significantly-boosts-ai-speed-energy-efficiency-existing-hardware
