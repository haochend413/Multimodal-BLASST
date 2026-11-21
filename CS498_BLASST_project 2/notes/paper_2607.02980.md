# arXiv 2607.02980: "Hierarchical Sparse Attention Done Right: Toward Infinite Context Modeling" (HiLS-Attention)

Read 2026-11-01 for the CS 498 project, because Prof. Lai recommended it. Source: the arXiv HTML of v1
(submitted 3 Jul 2026, the only version), downloaded and converted to text, then read in full: abstract,
Sections 1-8, references, and Appendices A-H. Figures are images and were not read; every number below comes
from the text or a table. Code: https://github.com/Tencent-Hunyuan/HiLS-Attention (153 stars at the time of
reading, training scripts, an SGLang fork, and a 7B checkpoint `tencent/HiLS-Attention-7B` on Hugging Face).
This paper was not in `related_work.md`.

**Words.** Ours: *tile* (128 query rows x 128 keys), *skip rule* (skip a tile when every row has tile row-max
minus running row-max < ln lambda), *margin*, *% skipped*, *dense*, *calibrated*. Theirs, translated:
- **Chunk**: a block of 64 consecutive *keys* (their S = 64). It is the key-axis half of our tile; they have no
  query-axis grouping in the math (each query token decides on its own), only in the kernel.
- **Chunk mass** Z_{i,c}: for one query row i, the sum of exp(score) over the 64 keys of chunk c. This is the
  un-normalized softmax weight the whole chunk would get. Its log, log Z_{i,c}, is a **LogSumExp (LSE)**.
- **Top-K**: keep the K = 32 chunks with the largest (estimated) mass. Fixed budget, not a threshold.
- **Native sparse training**: the model is trained with the sparse attention in place, so the loss is
  back-propagated through the chunk selection. The opposite of training-free.
- **Landmark token**: one extra special token appended to the end of every chunk; its query vector is a
  learned "representative query" for that chunk.
- **CPT**: continued pretraining of an existing checkpoint. **HoPE**: a positional encoding that keeps RoPE
  on the dimensions whose rotation period fits inside the training length and uses no position (NoPE) on
  the rest.

---

## 1. One-paragraph summary

Xiang Hu, Xinyu Wei, Hao Gu, Minshen Zhang, Tian Liang, Huayang Li, Lei Zhu, Yan Wang, Sirui Han, Yushi Bai,
Kewei Tu, Haitao Mi, Leo Liang (Tencent HY Team, ShanghaiTech, HKUST, UC San Diego). Preprint, no venue
listed. The problem: chunk-wise sparse attention (NSA, MoBA, InfLLM v2, DashAttention) never matches full
attention on retrieval, which the authors blame on inaccurate chunk selection caused by mean-pooled chunk
summaries and the fact that selection scores are discarded after the top-K and therefore never trained by
the language-modeling loss. The method, HiLS-Attention: (1) represent each 64-key chunk by a learned summary
key k'_c and a scalar bias b'_c computed from a landmark-token query, chosen so that q.k'_c + b'_c is the
first-order Taylor expansion of the chunk's LSE mass; (2) factorize attention into an inter-chunk softmax
over those surrogate masses times an intra-chunk softmax over tokens, so the surrogate is part of the forward
pass and gets gradients. Each query attends to the top-32 chunks (2,048 tokens) plus a 512-token sliding
window, so the cost per query is fixed. Key numbers: a 345M model trained at 8K gets perplexity 4.94 vs 4.96
for full attention at 8K, RULER single-needle / multi-key-multi-query / variable-tracking of 100 / 95 / 72 vs
100 / 97 / 34 for full attention at 8K, and still 96 / 89 / 43 at a 4M context (512x the training length).
Converting OLMo3-7B with 50B tokens of continued pretraining gives RULER 99.00 / 98.67 / 97.33 / 94.67 at
8K / 16K / 64K / 128K vs 99.67 / 33.00 / 1.33 / 0.00 for the same model continued-pretrained with full
attention, and LongBench-v1 overall 33.2 vs 29.0 (base) and 31.7 (base + YaRN 32K). On one H800 at the 345M
scale, prefill at 512K takes 5.0 s vs 67.0 s for full attention (13.5x) and one decode step 5.5 ms vs 85.9 ms
(15.7x); the two cross at about 16K, below which full attention is faster.

---

## 2. The method, precisely

### 2.1 Setup
- Keys are cut into chunks of S = 64 tokens. Query i always attends to a local sliding window of W = 512
  tokens (its left edge rounded down to a chunk boundary) plus K = 32 chunks chosen from the chunks that end
  before the window. So at most 2,048 + 512 = 2,560 keys per query: 31% of an 8K context, 3.9% of 64K,
  0.49% of 512K.
- Causal, decoder-only. The same rule is used in prefill and decode; the kernel is different (Sec. 2.6).

### 2.2 What is computed and what is skipped
- Computed for every query: 512 window scores; one score per chunk against the chunk summary (N/64 of them);
  then full scores against the 32 x 64 = 2,048 keys of the selected chunks.
- Skipped: everything else. **QK^T is never computed for an unselected chunk.** (This is the opposite of
  BLASST, which computes QK^T for every tile and skips only exp and PV.)
- Routing cost: O(N/S) per token, O(N^2/S) per sequence. The authors call this "the only quadratic term"
  (it is 1/64 of a full QK^T).

### 2.3 Decision rule
Top-K: for query i, rank candidate chunks by the surrogate score s^_{i,c} and keep the 32 largest (Eq. 9).
There is no threshold and no "skip if negligible" test; exactly 32 chunks are kept whenever 32 exist.

### 2.4 The surrogate score (Proposition 3.1, Appendix B): the part most relevant to us
Target: log Z_{i,c} = log sum_{j in chunk c} exp(q_i . k_j / sqrt(d)). Eq. (5) and Appendix A show two limits:
- if the 64 scores inside the chunk are nearly equal, log Z ~= mean(score) + ln 64;
- if one score dominates, log Z ~= max(score).

Mean-pooling the keys (NSA, MoBA; also MInference / SpargeAttention / CoSA in our survey) gives exactly
q . mean(k) = mean(score) (Eq. 6), so it is right only in the flat case; max-based summaries (they cite
MiniMax Sparse Attention) are right only in the spiky case. Neither tracks the true mass in general, and
"this mismatch may alter chunk rankings and cause important chunks to be missed."

Their fix: expand log Z around a representative query q'_c (first-order Taylor expansion). With
p_j = softmax over the chunk of (q'_c . k_j / sqrt(d)), the expansion is

    log Z_{i,c}  ~=  q_i . k'_c / sqrt(d) + b'_c,
    k'_c = sum_j p_j k_j      (keys weighted by how much q'_c attends to them),
    b'_c = -sum_j p_j log p_j (the entropy of that weighting).

The bias b'_c "adaptively interpolates" between the two limits: it approaches ln 64 when the chunk is flat
and 0 when one key dominates. Both k'_c and b'_c come from one attention call Attn(q'_c, K_c, K_c), cost
O(64) per chunk. q'_c is the query vector of the landmark token appended to chunk c (so it is produced by
the full transformer and learned). An ablation with one shared learnable query per layer instead of
landmarks keeps in-domain perplexity but loses extrapolation (perplexity at 512K: 7.94 vs 5.95).

A remark of mine (not in the paper): setting q'_c = 0 makes p uniform, k'_c = the mean key, and
b'_c = ln 64. So **mean-pooling is the zeroth-order version of their summary**, expanded at q = 0. Their
summary differs from mean-pooling in two ways: the keys are weighted by relevance to a representative query,
and the entropy term says how peaked the chunk is.

### 2.5 Why it is trainable: hierarchical softmax (Eq. 10)
The normal softmax weight of a selected token is factorized as
w_{i,j} = [exp(s_{i,j}) / Z_{i,c(j)}] x [Z_{i,c(j)} / Z_i], an intra-chunk softmax times the chunk's share of
the total mass. HiLS replaces the second factor by the surrogate Z^_{i,c} / Z^_i, where Z^_{i,c} =
exp(s^_{i,c}) and Z^_i sums the surrogates of the 32 kept chunks plus the exact window mass. Because the
surrogate now multiplies the output, the LM loss trains k'_c and b'_c directly. The authors argue this is
*better* than the exact naive block-sparse selection (compute full attention, keep the 32 chunks with the
largest exact mass): same 8K perplexity (4.94 vs 4.94) but much better beyond it (512K: 5.95 vs out of
memory; 128K: 4.71 vs 8.67). Their explanation: in full attention every irrelevant token keeps a small
non-zero weight, and over long contexts that noise accumulates; a compressed key averages the noise out.

### 2.6 Granularity, GQA, kernel, extra parameters
- Selection is per query token and per head (MHA). Under GQA, each head's normalized chunk weights are
  computed, the max over the heads in the group is taken, and the top-32 is chosen on those group scores,
  so "a chunk can be selected if it is important to any head in the group" (Appendix C).
- **Low-rank query calibration (Q-Cal)**: q^_i = q_i + W_up W_down h_i, rank r = 64 / 128 / 256 for the
  345M / 1.4B / 7B models, used only for chunk scoring. Removing it is the most damaging ablation
  (345M perplexity at 512K: 16.93 vs 5.95; RULER-512K single-needle 39 vs 99). Rank 128 hurts extrapolation
  relative to 64. The authors say the mechanism "is not yet fully understood."
- Extra parameters: 0.6% at 345M (landmark embedding + Q-Cal).
- **Kernel** (Sec. 4.2, 7.2): instead of one query per tile (NSA, which needs a GQA group of >= 16 heads to
  fill a 16-row tensor-core tile), HiLS packs M = 16 *adjacent query tokens*, takes the **union** of their
  selected chunks, and runs one (M x G, d) x (d, 64) matmul per chunk. The union over-fetches: at 64K, a
  single query's 32 chunks are 3.1% of the 1,032 visible chunks, but the union over 16 adjacent queries is
  102.1 chunks = 9.9% (at 4K: 43.6 of 57 chunks = 76.4%). 92.8% of the chunks used by one 16-query block
  were already used by the previous block. Note the parallel with BLASST's tile vote: BLASST skips a tile only
  if all 128 rows agree, which is the same "union of kept" logic with M = 128 rows instead of 16.
- Serving: SGLang with Triton kernels; KV paging granularity = chunk size.

### 2.7 Training-free? No.
Three recipes (Sec. 4.3, 6.2). (a) From scratch: 345M on about 30B tokens at 8K; 1.4B on 300B tokens.
(b) Full-parameter CPT of OLMo3-7B: about 50B tokens (13K steps x 512 sequences x 8K), LR 2e-4 to 2e-5
cosine, 5% of sequences converted on the fly into RULER-style retrieval examples. (c) "Landmark token
tuning", which they call "nearly training-free": base frozen, only landmark embeddings and Q-Cal trained
(<1% of parameters) on 5B tokens. **Recipe (c) does not deliver the long-context results**: RULER-8K 22.33,
16K 2.00, 64K 1.00 (Table 9), vs 99.00 / 98.67 / 97.33 for recipe (b). Only full CPT with HoPE works.

### 2.8 Phase, models, benchmarks
Prefill and decode (both benchmarked). Models: 345M GPT-2-Medium-shaped (16 layers, 16 query / 2 KV heads),
1.4B (22 layers, 32/4), OLMo3-7B (32 layers, 32/32 MHA). In the 7B, OLMo3's own 3:1 pattern of sliding-window
to full-attention layers is kept: HiLS replaces the 8 full-attention layers and the other 24 layers get a
512-token window (shrunk from 4K). Benchmarks: perplexity at 64 to 1M tokens; RULER (single needle,
multi-key multi-query, variable tracking) at 8K to 4M; LongBench-v1; 16 short-context tasks (MMLU, GSM8K,
HumanEval+, ...). Baselines: full attention (RoPE, HoPE), sliding window, NSA, DashAttention, InfLLM v2,
HSA-UltraLong, naive block-sparse, Landmark Attention, and YaRN-extended OLMo3.

### 2.9 Headline numbers
| Setting | HiLS | Full attention | Note |
|---|---|---|---|
| 345M, PPL at 8K / 32K / 128K / 512K (Table 1) | 4.94 / 4.34 / 4.71 / 5.95 | RoPE: 4.96 / >100; HoPE: 4.95 / 6.42 / >100 | NSA 5.01 / 7.62 / 11.75 / 19.14 |
| 345M, RULER 8K S-N / MK-MQ / VT (Table 2) | 100 / 95 / 72 | 100 / 97 / 34 (RoPE) | NSA 49 / 17 / 16; naive BSA 100 / 97 / 23 |
| 345M, RULER 512K and 4M | 99 / 91 / 66 and 96 / 89 / 43 | 0 / 0 / 0 | trained at 8K |
| 345M, mean-pool summary ablation, RULER 8K (Table 7) | 92 / 68 / 29 | | vs 100 / 95 / 72 with the learned summary |
| 1.4B, 300B tokens, 9-task short-context avg (Table 8) | 49.06 | 48.65 | |
| 7B CPT, RULER 8K / 16K / 64K / 128K (Table 9) | 99.00 / 98.67 / 97.33 / 94.67 | 512-window CPT: 99.67 / 33.00 / 1.33 / 0.00 | HiLS with RoPE instead of HoPE: 98.67 / 50.67 / 4.33 / 1.00 |
| 7B CPT, 11-task short-context avg (Table 9) | 43.35 | 43.24 (512-window CPT), 43.88 (base) | |
| 7B, PPL at 8K / 128K / 256K (Table 10) | 3.236 / 2.550 / 3.095 | 3.234 / 6.984 / 12.760 (512-window CPT) | |
| 7B, LongBench-v1 overall (Table 11) | 33.2 | 29.0 base; 31.7 base + YaRN 32K | > 8K subset: HiLS wins every category except Code |
| 345M, H800, 512K (Fig. 6 text) | prefill 5.0 s, decode 5.5 ms/token | 67.0 s, 85.9 ms | crossover at about 16K |

---

## 3. Relation to BLASST

### 3.1 Does it cite or compare against BLASST? No.
BLASST (2512.12087) is not in the 47 references. Neither is any training-free prefill method: no MInference,
XAttention, SpargeAttention, FlexPrefill, CoSA, Quest, or even FlashAttention. The word "threshold" does not
appear. The paper's world is *trained* sparse attention (NSA, MoBA, NOSA, DashAttention, InfLLM v2, HSA,
Landmark Attention, SeerAttention, DeepSeek's DSA, MiniMax SA). So it is not a competitor to BLASST in the
sense of "same setting, better numbers"; it is the other end of the design space.

### 3.2 Side by side
| | BLASST (our baseline) | HiLS |
|---|---|---|
| Needs training | No | Yes: 50B-token CPT for 7B; frozen-base variant fails on long context |
| Unit | tile: 128 queries x 128 keys | chunk: 64 keys; per query token in the math, 16 adjacent queries in the kernel |
| Decision | threshold: tile row-max minus running row-max < ln lambda, all 128 rows must agree | top-32 by a learned estimate of the chunk's LSE mass |
| Budget | input-dependent (calibrated lambda = a/L gives about 50%) | fixed 2,048 + 512 tokens regardless of length |
| What the decision looks at | the exact tile max (QK^T already computed) | an estimate of the chunk mass from one summary key + entropy bias |
| QK^T for a skipped unit | computed | never computed |
| Skipped work | exp, V load, PV | QK^T, exp, K and V loads, PV |
| Speed ceiling | 1.8x at kernel level (QK^T floor); 1.52x measured at 72% skipped | 13.5x prefill at 512K; slower than dense below about 16K |
| Accuracy claim | about 1 point RULER loss at 50% skipped | matches or beats dense; +38 VT points at 8K (345M) |
| Error guarantee | yes, per skipped element (p < lambda) | none; learned |
| Per-layer/head thresholds | one lambda | one K = 32 everywhere; HiLS in 1 of every 4 layers in the 7B |
| Multimodal | no | no |

### 3.3 Where it sits relative to our ideas

**Idea 1 (score tiles cheaply so QK^T is not computed for skipped tiles).** HiLS *is* a cheap scorer that
avoids QK^T for unselected chunks, at a cost of N/64 dot products per query. But it is learned, top-K, and
requires CPT, so it is not something we can drop into Llama-3.1-8B. What it adds to Idea 1 is an argument,
not a method: Section 2.2 is the clearest published statement of *what a tile score should estimate* (the
LSE mass) and *why mean-pooling, stride sampling and max-pooling get it wrong* (they estimate the mean or
the max, and the mass sits between mean + ln S and max depending on how peaked the tile is). Their Table 7
quantifies the damage for a trained 345M model: mean-pool summaries drop RULER-8K from 100 / 95 / 72 to
92 / 68 / 29. Nobody has measured this in the training-free setting on a dense-trained 8B model, which is
exactly what our saved Q/K allow (Experiment 1 below). Also note a mismatch that matters for us: BLASST's
skip rule looks at the tile *max*, while HiLS argues the right importance is the tile *mass*; for a flat
128-key tile the mass can be up to 128x (ln 128 = 4.85 nats) larger than the max would suggest.
- Certified skip (provable upper bound): not in HiLS. Their surrogate is an approximation with no bound.
- Seeding the running max with exact scores: not in HiLS; there is no running max at all.
- Smaller voting groups: partly. The kernel's M = 16 query packing with the union of chunks is the same
  trade-off as BLASST's 128-row vote, and Fig. 7 sweeps M in {1, 2, 4, 8, 16, 32, 64} (curve not readable,
  only the M = 16 numbers above are in the text). That is the only quantitative data point I know of on
  "how much extra work a query-group vote costs" for a top-K method; the BLASST analog is open.

**Idea 2 (reuse skip masks across layers).** Not in HiLS. Nothing is reused across layers; each HiLS layer
selects on its own. Two things are adjacent to Idea 2: (a) Sec. 7.2 measures reuse across *adjacent queries*
within a layer (92.8% of chunks carried over from one 16-query block to the next), which is a clean metric
we can copy for our masks (Experiment 4); (b) in the 7B, only 8 of 32 layers do long-range attention at all,
but that is OLMo3's architecture, not a reuse mechanism.
- Threshold-mask reuse with a guard band, runtime refresh check: not touched.

**Idea 3 (per-layer / head / modality thresholds).** Not in HiLS. K = 32 is fixed for every layer, head,
query and length. The GQA rule (max over heads, then one top-K per group) is a per-group *union*, not a
per-head budget. No multimodal experiments. One tangential point: the entropy bias b'_c is a per-chunk,
per-head, input-dependent correction to the score, so it is a form of "adaptive thresholding" that comes
for free from the summary rather than from calibration; that is an angle, not prior work on our idea.

---

## 4. Why Prof. Lai might have recommended it (my reading, ranked by likelihood)

1. **For Idea 1, as the principled version of "what a cheap tile score should compute."** Our survey found
   that proxies (mean-pool, stride, antidiagonal) are taken, and the honest contribution left was a
   controlled operator comparison. HiLS supplies the theory for that comparison: a tile score should
   approximate the tile's LSE mass; mean-pooling is its zeroth-order Taylor expansion; a first-order
   expansion around a representative query plus an entropy term is strictly more expressive at almost the
   same cost. That gives our Idea 1 a hypothesis to test instead of a sweep to report.
2. **To place BLASST on the map.** BLASST never skips QK^T and is capped at 1.8x; HiLS never computes it for
   95% of chunks and reaches 13.5x at 512K, but only after 50B tokens of CPT and a positional-encoding swap.
   The space between (training-free, skips QK^T, keeps a guarantee) is where our project lives, and the
   mid-semester report should say so. He may also want us to know that "sparse can beat dense" is now a
   claim with evidence (VT 72 vs 34 at 345M; LongBench 33.2 vs 29.0 at 7B), so "lossless" is no longer the
   only bar.
3. **Evaluation methodology.** The paper argues RULER on *small* models is the honest test of chunk
   selection ("larger models may compensate for retrieval inaccuracies"), uses perplexity at 6-8 context
   lengths as the low-noise metric, holds budgets equal across methods (window 512, chunk 64, top-32), and
   reports the overlap metric of Sec. 7.2. Our 8B RULER results have a 2-point noise floor on 150 samples;
   his pointer may be "use NLL / perplexity and matched budgets," which `directions_idea1.md` already
   proposed.
4. Less likely: as a baseline to run. Not feasible for us (needs training; the released 7B is OLMo3-based
   and our pipeline is Llama; nothing to compare at matched % skipped since HiLS has a fixed budget).

If we only get one thing from it, it should be (1).

---

## 5. What it leaves open that we can do in 5 weeks

Cost unit: 5 s per 8K prompt, 15 s per 16K prompt in our PyTorch simulation. One RULER configuration
(3 tasks x 50 samples x {8K, 16K} = 300 prompts) = 150 x 5 + 150 x 15 = 3,000 s, about 0.85 GPU-h; call it
1 GPU-h with generation. Everything below uses Llama-3.1-8B-Instruct and the saved Q/K (22 GB, 105 prompts,
4K-32K) or margins (3 GB) unless noted. **Do not** try to reproduce HiLS: it needs 50B tokens of CPT, and we
have 538 GPU-h and PyTorch only.

### Experiment 1 (recommended first): which quantity should a tile score estimate, and which training-free summary estimates it best?
- **What we build.** From the saved Q/K, for every (layer, head, query tile, key tile) compute, per query
  row, the exact tile row-max and the exact tile row-LSE (the mass). Then compute five cheap scores for the
  same tile: (a) mean-pooled key (the zeroth-order summary; MInference / SpargeAttention), (b) stride-sampled
  rows and keys (CoSA), (c) antidiagonal (XAttention), (d) max-pooled key per channel, and (e) **the HiLS
  first-order summary made training-free**: take the mean of the 128 queries in the query tile as the
  representative query q', compute p = softmax(q' . K_tile), k' = sum p_j k_j, b' = entropy(p), and score
  each row as q_i . k' + b'. Cost of (e): one 128-key row plus 128 d-dim dot products per tile = 2/128 of the
  tile's QK^T. (This is Proposition 3.1 with q' chosen without training; the paper's own shared-query
  ablation shows the expansion point matters, so we expect (e) to beat (a) by less than the learned version.)
- **What we measure.** For each score: rank correlation with the exact max and with the exact LSE; and the
  thing that matters for BLASST, *recall of kept tiles* (of the tiles BLASST keeps at the calibrated 50% and
  75% lambda, how many the proxy also keeps at the same % skipped). Split by layer and head and by how
  peaked the tile is (its entropy), to test the paper's claim that mean-pool fails on peaked tiles.
- **Comparison.** The five scores against each other and against BLASST's exact decision, all at matched
  % skipped.
- **Cost.** Offline on saved Q/K; about 1-2 GPU-h for all 105 prompts (the exact tile LSE needs the full
  score tiles, which is the expensive part). No new model runs.
- **Why it is a contribution.** The literature has one operator per paper and no sweep; the HiLS entropy
  summary has never been evaluated training-free; and no one has reported how often the *max* criterion and
  the *mass* criterion disagree on real 8B attention. It also tells Idea 1 which proxy to build on.

### Experiment 2: a mass-aware skip rule for BLASST
- **What we build.** Three rules, all one-pass from saved Q/K: (i) BLASST, tile max minus running max;
  (ii) tile LSE minus running max (HiLS's numerator, BLASST's denominator); (iii) tile LSE minus running LSE
  (DynSparse's denominator, which `directions_idea1.md` Direction D already proposed, now with the mass
  numerator). Rule (ii) keeps the one-pass margin trick, because a tile skipped under (ii) is also skipped
  under (i) and so never raises the running max. For each rule, choose lambda from the saved tiles to hit
  50% and 70% skipped.
- **What we measure.** Teacher-forced NLL change versus dense first (cheap), then RULER at 8K and 16K at
  matched % skipped. Also the fraction of tiles where (i) and (ii) disagree, split by layer and head.
- **Comparison.** Rule (ii) and (iii) against BLASST at the same % skipped.
- **Cost.** 2 rules x 2 sparsity levels = 4 RULER configurations, about 4 GPU-h, plus about 1 GPU-h of NLL.
- **Why it is a contribution.** It tests HiLS's central claim (mass, not max, is the right selection signal)
  inside a training-free kernel rule on an 8B LLM; DynSparse tested an LSE *denominator* only on video
  diffusion. Caveat to state up front: a real kernel cannot compute the tile LSE without the exp that BLASST
  skips, so the deployable version uses Experiment 1's proxy (e) or the bound max + ln 128; the exact-mass
  runs measure the headroom.

### Experiment 3: the cost of the vote, BLASST version of HiLS Fig. 7
- **What we build.** From saved margins, recompute the skip decision with G = 128, 64, 32, 16 and 1 query
  rows per vote (our tile-sensitivity table already has % skipped at fixed lambda: 16.23% at 128 x 128 vs
  38.18% at 16 x 64, lambda = 1e-3, 8K). Add the "loaded fraction" curve HiLS reports (tiles kept by the
  union of G rows divided by tiles kept by a single row), and the veto anatomy: how many rows block each
  kept tile.
- **What we measure.** RULER at matched % skipped for G = 128 vs G = 16 (2 configurations, about 2 GPU-h).
- **Why.** HiLS gives the only number for a top-K method (M = 16 costs 3.2x the single-query budget at 64K);
  the threshold-rule analog is unreported, and it decides whether a 64-row warpgroup vote is worth a kernel.

### Experiment 4: agreement of BLASST masks across adjacent query tiles and adjacent layers (feeds Idea 2)
- **What we build.** HiLS's overlap metric applied to our masks: for each layer and head, the fraction of
  tiles kept in query tile t+1 (key tiles 0..t) that were also kept in query tile t; and the same between
  layers l and l+1 at the same query tile, with the GQA-union head mapping from `directions_idea2.md`.
- **Cost.** CPU only, from the 3 GB margins. 0 GPU-h.
- **Why.** It is the premise check for Idea 2, and the query-axis number also opens a cheap variant nobody
  has published as far as our notes show: predict tile (t+1, j) from (t, j) with a guard band, as AMEND does
  across decode steps and LiteAttention across diffusion steps. (Openness not checked by search.)

Total for all four: roughly 8-10 GPU-h and about 8-10 engineering days.

---

## 6. Caveats

- Figures 1, 2, 5, 6 and 7 are images and were not read; the speed and overlap numbers come from the text
  that describes them, and the M-sweep curve of Fig. 7 is not available to me.
- No venue; v1 only; no third-party reproduction found. The GitHub repo was seen in one fetch: the `ops/`
  directory was not opened, and no license was visible.
- Sec. 6.2 calls HiLS "training-free" once; this contradicts the 50B-token CPT in the same section and
  probably means "without YaRN". The genuinely frozen variant (landmark token tuning) is the weak one.
- Table 9 confound: every CPT baseline was trained with 5% RULER-style data, which is why the full-attention
  512-window CPT also scores 99.67 at 8K. HiLS's advantage is beyond the 8K training length, and part of it
  is HoPE: HiLS with RoPE gets 50.67 at 16K and 4.33 at 64K, vs 98.67 and 97.33 with HoPE.
- Speed is measured at the 345M scale with Triton kernels on both sides, not against FlashAttention-3 or a
  vendor kernel, at batch size 1. The crossover at 16K means HiLS would be slower than dense at our 4K-16K.
- In the 7B, 24 of 32 layers see only a 512-token window; "sparse attention" there is partly architectural.
- "Mean-pooling is the zeroth-order expansion" is my derivation from Proposition 3.1, not a statement in
  the paper. The proposed training-free summary (Experiment 1e) is my construction; the paper only tests
  learned landmark queries and one shared learned query per layer.
- GPU-hour figures assume 5 s / 15 s per prompt; the saved-Q/K experiments were not timed.
- "Open" claims in Section 5 were checked against our existing notes only, not by new searches.
