# ECHO (OSDI 2026, Liu et al.): what it is and what it means for our BLASST project

Written Nov 1, 2026. Source: the full 22-page PDF at
`https://www.usenix.org/system/files/osdi26-liu-guangda.pdf` (downloaded and read end to end, including
the artifact appendix). Cross-checked against the USENIX abstract page (badge: "Artifact Evaluation
Available") and the code repository `https://github.com/sjtu-zhao-lab/ECHO`. Prof. Lai recommended this
paper to the team; this note says what it does, how it relates to BLASST and our three ideas, and what we
could do with it in the five weeks left.

Words used, same as `results/RESULTS.md`: a **tile** is a 128 query rows x 128 keys block of the attention
score matrix; the **skip rule** skips a tile when every one of its 128 rows has (tile row max - running
row max) < ln(lambda); the **margin** of a tile is the largest of those 128 differences, so a tile is skipped
exactly when margin < ln(lambda); **% skipped** = skipped tiles / causal tiles; **dense** = no skipping;
**calibrated** = lambda = a / L with a fitted for a target % skipped. New words from the paper are defined
where they first appear.

---

## 1. One-paragraph summary

**ECHO: Efficient KV Cache Offloading with Lossless Prefetching for Serving Native Sparse Attention LLMs.**
Guangda Liu, Wenhao Chen, Chengwei Li, Zhenyu Ning, Quan Chen, Shixuan Sun, Jieru Zhao (Shanghai Jiao
Tong University); Jing Lin, Yiwu Yao (Huawei); Minyi Guo (Guizhou University and SJTU). OSDI 2026, July
13-15, Seattle, pages 17-37. It is a **serving system**, not a new attention algorithm. The problem: models
with *native* (trained-in) sparse attention such as DeepSeek-V3.2 compute attention over only 2,048
selected tokens per query, but the KV cache still grows linearly with context length, so GPU memory
capacity, not compute, caps how many long requests can run at once; on 8 NVIDIA H20 GPUs (96 GB each)
one data-parallel worker holds at most 380K tokens of KV, i.e. 3 to 4 concurrent 100K-token requests. The
method: keep the full KV cache in host (CPU) memory, keep only the selected tokens' KV on the GPU, and
(a) manage the GPU-side cache with fixed-size integer tensors and parallel GPU operations so the whole
decode step fits in one CUDA graph, and (b) prefetch the KV of tokens *likely* to be selected while the
selection scores are still being computed, then fetch whatever was missed, so the output is bit-for-bit
what the model would produce without offloading ("lossless"). Headline numbers on DeepSeek-V3.2 (4-bit
AWQ, 8 x H20, 318 InfiniteBench requests of 80K-100K tokens): generation throughput up to 2.15x SGLang
and 4.1x vLLM when all requests arrive at once; up to 3.10x SGLang with a 200K-token GPU pool and 4.12x with
a 110K-token pool; the offloading bookkeeping costs 1.15 ms per decode step across all 61 layers (0.28% of
the step); time-to-first-token grows by at most 7.9%, inter-token latency by 2.7% to 27.8% depending on
load.

---

## 2. The method, precisely

### 2.1 The model it serves: DeepSeek Sparse Attention (DSA)

DeepSeek-V3.2 (671B-parameter mixture-of-experts) uses **DSA**: a small trained module called the
**indexer** scores every previous token for each query, and attention is then computed over the **top-k =
2,048** tokens with the highest scores. The indexer uses 64 heads of dimension 128 in FP8 with a ReLU, while
the main attention (multi-head latent attention, MLA, which stores one compressed 512-dimensional vector
per token instead of separate K and V) uses 128 heads. So the indexer is roughly the cheap scorer of our
Idea 1, except it is *trained* and *token-granular* (one score per token, not per tile). The indexer's own
K cache must stay on the GPU, which is why the paper says the KV footprint grows "steeper" than for a
dense model.

Key observation the paper makes about cost (Sec. 2.4, Fig. 4): the indexer is O(L^2) (it scores all L
previous tokens for every query), while recall (loading selected tokens' KV from the host) is O(L k) with
k fixed at 2,048. At about 100K context with a 90% GPU-cache hit rate, the indexer takes longer than the
recall. **This is the same structure as our "QK^T floor":** an irreducible quadratic scoring pass that is
never skipped. ECHO's response is not to remove it but to use it as a time window in which to hide data
movement.

### 2.2 What is stored where

- **Host pool** (CPU DRAM, 1.8M tokens, about 1,000 GB in the evaluation): the complete MLA KV cache of
  every active request. Every newly generated token's KV is written here immediately, so eviction from the
  GPU never needs a copy back.
- **GPU pool** (per layer, 200K tokens in the evaluation): a cache of the KV of recently selected tokens.
  Managed separately for each of the 61 layers because "each layer selects different tokens".
- **Indexer K cache**: always on the GPU (it is needed for every query).
- Metadata is five fixed-length integer tensors on the GPU (free bitmap, eviction priority, index buffer,
  GPU-slot-to-host-slot map, host-slot-to-GPU-slot map); 4 N_host + 13 N_gpu bytes, about 10 MB per layer,
  610 MB total.

### 2.3 Decision rules

- **What attention computes**: for each query, attention over exactly the top-2,048 tokens by indexer score
  (unchanged DSA). ECHO never changes which tokens are attended; it only changes where their KV lives and
  when it is moved.
- **Eviction** (what leaves the GPU pool): lowest-priority slots first, priority being an LRU-like counter
  bumped by the indexer score ranking; tokens selected in the current step are protected. Eviction is only
  a metadata update (the data is already in the host pool).
- **Guaranteed recall**: after the indexer finishes, every selected token not already in the GPU pool is
  loaded. This step is what makes the system lossless; prefetching only shrinks it.
- **Intra-query prefetching, decode** (Sec. 5.1): the k-th highest indexer score at step t+1 is predicted
  by an exponential moving average (EMA) of the k-th highest scores at earlier steps,
  s_hat(t+1) = alpha * s_hat(t) + (1 - alpha) * s(t), with alpha = 0.5 in Fig. 7. During the indexer
  kernel, any token whose score exceeds s_hat is fetched immediately, before the top-k is known. This turns
  a top-k rule into a threshold rule for the purpose of prefetching. Fig. 8 shows the number of tokens
  above the predicted threshold is close to 2,048 at most steps; a global counter caps how many tokens are
  prefetched (the cap grows with context length) so prefetching never slows the indexer.
- **Inter-query prefetching, prefill** (Sec. 5.2): prefill processes query blocks in order; while the
  indexer scores query block i+1, the tokens selected by block i are fetched. Exact top-k inside the
  kernel was too slow, so ECHO uses one round of radix select: a 256-bin histogram on the top 8 bits of
  the score, after subtracting the EMA-predicted k-th score so that the "threshold bin" is small (Fig. 9).
  Tokens in bins above the threshold bin are prefetched; the exact top-k is finished afterwards.
- **Kernel fusion** (Sec. 5.3): prefetching is fused into DeepGEMM's indexer kernels using warp
  specialization (different groups of GPU threads doing loading, matrix multiply, and prefetch) in a
  three-stage software pipeline, so the PCIe transfer overlaps the score computation. Data movement uses
  unified virtual memory (GPU kernels read host memory directly) so no CPU is in the loop.

### 2.4 Granularity, training, phase, deployment

- Granularity: **per token** (DSA selects tokens, not blocks). The Discussion says the cache manager could
  be changed to block granularity and extended "with the KV-head dimension used in GQA-based models", i.e.
  to models like Llama-3.1-8B, but that is not implemented.
- Training-free: **the system is; the sparsity is not.** ECHO adds no training, but it targets a model whose
  sparsity was trained in. The authors say ECHO "can also serve models with training-free sparse
  attention" but chose DSA because training-free methods "inherently incur accuracy degradation".
- Phase: both. Decode is the main target (intra-query prefetching, graph execution). Prefill gets
  inter-query prefetching. Recommended deployment is prefill/decode disaggregation (separate machines for
  the two phases) with offloading **disabled on prefill instances**, because during prefill "the total
  selection is the union set of all prefill tokens" and can touch nearly the whole KV cache.
- Built on SGLang v0.5.4 and DeepGEMM; metadata operations are Triton kernels; prefetch kernels are CUDA.

### 2.5 Evaluation setup and results, as exact numbers

Setup: one node, 8 x NVIDIA H20 (96 GB), PCIe Gen5 at 64 GB/s host-GPU, 224-core Xeon 8480+, 1.5 TB DRAM.
Baselines vLLM v0.11.1 and SGLang v0.5.4 (CUDA 12.9, PyTorch 2.8). Attention data-parallel over 8 GPUs and
MoE tensor-parallel over 8 for ECHO and SGLang; vLLM tensor-parallel for both. Chunked prefill 2,048.
Model: DeepSeek-V3.2-Exp 4-bit AWQ with the indexer and layers 0, 1, 2, 60 left unquantized. Long workload:
318 InfiniteBench requests of 80K-100K tokens (about 26M input tokens). Short workload: 100 ShareGPT
requests.

| Result | Number |
|---|---|
| Max generation throughput, all GPU memory, 256 output tokens, infinite request rate (Fig. 12a) | 2.15x SGLang, 4.1x vLLM |
| Effective batch size behind that (average active requests per decode step) | SGLang about 30 (325K-token GPU pool, 3-4 requests per GPU); vLLM about 1.5 (180K-token pool); ECHO larger via 1.8M-token host pool |
| GPU pool capped at 200K tokens (Fig. 13a) | 3.10x SGLang |
| GPU pool capped at 110K tokens (Fig. 13b) | up to 4.12x SGLang |
| Per-task throughput, no output cap (Fig. 14) | +27.07% Code.Debug, +2.83% En.MC, +7.11% En.QA, -1.74% Code.Run (75 / 160 / 31 / 42 requests) |
| Latency on ShareGPT, mixed prefill+decode (Fig. 15) | TTFT at most +7.9%; ITL +2.7% to +27.8% (worst at light load, 2.7-7.7% at rates >= 0.3 req/s); end-to-end +15.9-19.2% at 0.1-0.2 req/s, <= 7.2% at >= 0.3, < 4.6% from 0.5 to infinite |
| Prefill per-layer latency (Fig. 16a) | 6.82 ms -> 7.42 ms (+8.8%) |
| Decode all-layer latency (Fig. 16b) | +2.8% vs SGLang; MoE is about 70% of the step; +5.9 ms graph replay; +6.8 ms ReduceScatter jitter across workers |
| Offloading operations per decode step, all layers (Fig. 16c) | allocate 0.167 ms, free 0.572 ms, recall 0.145 ms, write-back 0.261 ms; total 1.15 ms = 0.28% of the step; 37 tokens recalled per step on average; 13 new tokens written back per worker per step |
| GPU pool hit rate per layer on InfiniteBench (Fig. 17) | 0.97-0.99 for most layers; layer 12 at 0.95; layer 17 at 0.88 |
| Intra-query prefetch microbenchmark, indexer + recall latency (Fig. 18) | up to 1.29x lower at hit rate 0.5, 1.51x at 0.9; "modest" at 0.97 |
| Intra-query prefetch end to end (Fig. 19) | up to 4% more throughput (hit rates are already high, MoE dominates) |
| Inter-query prefetch, prefill microbenchmark (Fig. 20) | at most 1.1x, assuming different query tokens select unrelated tokens |
| Allocator (Fig. 3b), 300K-token pool | SGLang 12 us per allocate+free; ECHO 6-8 us |
| CUDA-graph benefit (Fig. 3a) | +1.2x throughput for dense V3.1, +1.5x for sparse V3.2 |
| Sparse FlashMLA utilization at 8 requests of 64K on 8 x H20 (Fig. 2) | 53% |

Code: `https://github.com/sjtu-zhao-lab/ECHO`; artifact on Zenodo record 20376436 (Docker image, needs
8 x H20 and 1.5 TB RAM; not something we can run).

---

## 3. Relation to BLASST and to our ideas

### 3.1 Citations and comparisons

- **BLASST is not cited** (the word does not appear in the PDF).
- **No block-sparse prefill method is compared against.** MInference is cited once ([23]) only to explain
  the high hit rates ("vertical lines in the attention pattern"). FlexPrefill, XAttention, SpargeAttention
  are absent. Quest and InfiniGen are cited as examples of training-free score approximation; Twilight and
  Progressive Sparse Attention are cited as the "top-p" (threshold) selection that makes prefetching
  possible. Related offloading systems in Table 1: Strata, ArkVale, InfiniGen, FreeKV, CLO, SparseServe.
  InfiniGen, FreeKV and CLO prefetch *lossily* (they predict selection from the previous layer or step and
  attend to the prediction); ECHO's distinction is that its prefetch is only a hint and the exact selection
  always wins.
- The comparisons are with **serving systems** (SGLang, vLLM), measured in throughput and latency, not
  with attention algorithms measured in accuracy versus sparsity. There is no accuracy table at all,
  because by construction ECHO's outputs equal SGLang's.

### 3.2 The conceptual mapping to BLASST

| ECHO / DSA | BLASST (ours) |
|---|---|
| Indexer score per token, O(L^2), never skipped | QK^T per tile, never skipped (the QK^T floor, 54-61% of attention time) |
| Top-k = 2,048 tokens; must finish scoring before selecting | Threshold rule: a tile is decided as soon as its margin is known; no global sort |
| EMA-predicted k-th score turns top-k into a threshold so prefetch can start early | Not needed: lambda already is the threshold |
| "Lossless": exact selection always runs; prefetch only moves data earlier | BLASST's skip is already exact given the margin; what could be made "lossless" is any *reuse* of a mask (Idea 2) or any *cheap bound* (Idea 1's certified skip) |
| What it saves: GPU memory capacity and host-GPU transfer time | What BLASST saves: softmax + PV compute; it saves no KV bytes in prefill and, at decode, V reads only |
| Per-layer GPU pools because "each layer selects different tokens" | Indirect evidence that cross-layer reuse of a selection (Idea 2) is not free |
| Hit rate across decode steps 0.97-0.99 (token granularity) | Unknown for BLASST tile verdicts on Llama; AMEND reports 89.2% of verdicts unchanged step to step in its setting |

The single most relevant sentence in the paper, from the Discussion (Sec. 7): "Other sparse attention
mechanisms can benefit from the same prefetching framework when they expose similar early-selection
signals, such as threshold-based selection or multi-stage top-k selection. This also suggests an
opportunity to co-design training-time or training-free sparse attention with prefetch-friendly selection
mechanisms, which we leave to future work." BLASST is a threshold-based, block-wise rule on GQA models,
which is exactly the case they name and do not do.

### 3.3 Where it sits relative to Ideas 1, 2, 3

- **Idea 1 (cheap scoring so QK^T is skipped).** Orthogonal. ECHO takes the scorer as given and never tries
  to skip it; its only idea in this space is overlapping data movement with the scorer. One useful point for
  Idea 1's "certified skip" sub-direction: ECHO's discipline of separating a *speculative* step (prefetch
  on a prediction) from a *guaranteed* step (exact recall) is the same structure as "skip on a cheap bound
  only when the bound certifies it, otherwise fall back to the exact QK^T".
- **Idea 2 (reuse masks across layers).** Related but on a different axis. ECHO exploits predictability
  along **time** (decode step t-1 to t, Fig. 7) and along **query position** (prefill block i to i+1,
  Sec. 5.2), and explicitly does *not* reuse across layers (per-layer GPU pools). It also shows the
  cross-step stability is not uniform across layers (layer 17 hit rate 0.88 vs 0.97-0.99 elsewhere), which
  supports our per-(layer) guard band and refresh ideas. The "guard band" sub-direction has a direct analog:
  ECHO's prefetch cap and its EMA threshold are a guard band on how much to speculatively fetch.
  `directions_idea2.md` Direction 5 (reuse BLASST verdicts across decode steps, GPU only) is the closest
  existing plan, but it is framed as *lossy* reuse (skip K loads on a prediction, audit a subset). ECHO
  suggests the *lossless* version: never skip on a prediction, only prefetch on it.
- **Idea 3 (per-layer/head/modality thresholds).** Weakly related. ECHO's per-layer management and the
  per-layer hit-rate spread are system facts, not threshold choices. Nothing on modality.

### 3.4 What ECHO shows that our project currently lacks

Our results so far are accuracy versus % skipped, plus FLOP-style arguments about the QK^T floor. ECHO is
entirely about **the serving-level consequences of a sparse rule**: how much KV must be resident, how many
bytes move per step, whether the decode step can be captured in a CUDA graph, and throughput versus request
rate under a memory cap. BLASST's paper does not address memory capacity at all, and our RESULTS.md says
explicitly that our timing numbers are not speed results. ECHO gives a template for a systems-flavored
claim we *can* make with a PyTorch simulation: counts of resident bytes and transferred bytes per step,
and hit rates, which ECHO itself reports from traces (Figs. 7, 8, 17).

---

## 4. Why Prof. Lai might have recommended it (best guess)

These are guesses; we have not asked him.

1. **He wants a systems angle.** This is an ML systems course and the paper is OSDI. Our three ideas are all
   "better sparsity at the same accuracy". ECHO's thesis is that for long-context serving the binding
   constraint is GPU memory capacity and host-GPU bandwidth, not attention FLOPs, and that sparse attention
   "does not shrink the KV cache itself". The likely message: ask what BLASST's rule buys at the system
   level (resident KV, bytes per step, graph-friendliness), not only in % skipped.
2. **Threshold rules are prefetch-friendly, and BLASST is a threshold rule.** ECHO had to *manufacture* a
   threshold (EMA-predicted k-th score) to prefetch under top-k. BLASST has one natively. He may be pointing
   at the co-design opportunity the paper leaves as future work: a threshold-based block-sparse rule plus an
   offloading/prefetching system.
3. **"Lossless" as a design discipline.** ECHO refuses any speculation that can change the output; it gets
   its speed from overlap and capacity, not from dropping work. Our Idea 1 (proxy scores) and Idea 2 (mask
   reuse) both drop work on a guess. He may want us to separate the speculative part from the guaranteed
   part the way ECHO does, which is the certified-skip sub-direction we already listed.
4. **Evaluation methodology.** Throughput versus request rate under a fixed memory budget, latency
   breakdown in milliseconds per component, per-layer hit rates, and an ablation per mechanism (Figs.
   18-20) with the end-to-end gain honestly reported as "up to 4%". A model for our final report.
5. **Awareness that production is moving to trained sparsity.** DSA (DeepSeek-V3.2, GLM-5) is the sparse
   attention that is actually deployed at scale. A training-free method like BLASST has to justify itself
   by applying to existing models (Llama, Qwen) that have no indexer; ECHO's own caveat that training-free
   methods "incur accuracy degradation" is a reminder to report accuracy honestly.

Reading 1 and 2 together: the most useful takeaway is probably **"measure what a BLASST-style threshold
mask would do for an offloading serving system, the way ECHO measures DSA's top-k."**

---

## 5. What it leaves open that we could do in 5 weeks

All of these are PyTorch simulations on DeltaAI (GH200, 96 GB). GPU-hour estimates use 5 s per 8K prompt
and 15 s per 16K prompt for our reference attention function, and the team's own estimate of about 50 ms
per decode step in the Python-level simulation (`directions_idea2.md` Direction 5). None needs a CUDA
kernel; none needs accuracy runs, because everything below is lossless by construction (the attended set
is never changed). Total for all four: about 3 to 6 GPU-hours out of 538.

One engineering prerequisite for Experiments 1-3: add the **decode branch** to our recorder in
`blasst_attn.py`. BLASST's decode rule is one query row per head against 128-key tiles with a left-to-right
running max (NVIDIA's reference, `modelopt_and_hf_interface.md` line 187), so the margin trick works
unchanged: record per (layer, head, key tile) the tile max minus the running max, once per decode step.
At 16K that is 32 layers x 32 heads x 128 tiles = 131,072 fp16 values = 256 KB per step. Decode currently
runs plain SDPA (RESULTS.md Sec. 8), so this is new code, but small. Note BLASST uses a separate decode
threshold (NVIDIA's default decode lambda is 1e-4; calibrated decode a is 4.6-11.7 for 50% in the paper's
Table 12), so the decode sweep needs its own calibration, which the margin trick gives for free.

### Experiment 1: step-to-step stability of BLASST decode verdicts (ECHO Fig. 7, 8 and 17 for BLASST)

- **What we measure.** For each (layer, head, key tile) at decode step t: is the verdict (skip or keep) the
  same as at step t-1? Report (a) the **hit rate** = kept tiles at t that were also kept at t-1, divided by
  kept tiles at t, per layer (ECHO's Fig. 17 analog; ECHO: 0.97-0.99, two layers lower); (b) the same at
  the **KV-head level** for GQA, where a V tile is needed if *any* of the 4 query heads sharing that KV head
  keeps it; (c) how the hit rate changes with the number of steps since the last exact decision (reuse
  distance 1, 2, 4, 8); (d) for comparison on the same dump, the **cross-layer** overlap at the same step
  (Idea 2's axis), so we can say which axis of the BLASST mask is more predictable on Llama-3.1-8B: time
  or depth.
- **Comparison.** ECHO's token-level hit rates for DSA (0.97-0.99); AMEND's 89.2% unchanged verdicts;
  our own cross-layer overlap numbers from the prefill margins (teammate C).
- **Runs.** 50 prompts at 8K and 50 at 16K from RULER rows we have not used for calibration or testing
  (rows 150+), 128 decode steps each with the end-of-sequence token disabled so every prompt yields 128
  steps (flag this: post-answer tokens are filler; alternatively use prompts with naturally long answers).
  Per 8K prompt about 5 + 128 x 0.05 = 11.4 s; per 16K about 21.4 s; total about 1,640 s = 27 min.
  Storage: 128 steps x 256 KB x 100 prompts = 3.3 GB on `/work/nvme`. **About 1 GPU-hour on `ghx4`, 2
  including a debugging session on the interactive partition** (charge factor 2).
- **Why it is a contribution.** Nobody has measured the temporal stability of a *threshold* tile mask on an
  open GQA model. ECHO measures it for a trained token-level top-k on a 671B MoE; AMEND measures it on a
  GPU-PIM simulator. It also settles, with numbers, which reuse axis Idea 2 should target. It is a
  prerequisite for Experiment 2 and costs almost nothing.

### Experiment 2 (the single best one): lossless V-prefetch for BLASST decode, traffic model and measured copy time on GH200

- **The design being evaluated.** In BLASST decode, K is read for every tile (QK^T is never skipped) but V
  is read only for kept tiles. So a serving system could keep **K resident on the GPU and V in host
  memory**, exactly ECHO's split between the always-resident indexer K cache and the offloaded MLA KV. Per
  step: (i) while QK^T runs over the resident K, prefetch the V tiles kept at step t-1 (plus a guard band:
  tiles whose margin at t-1 was at least ln(lambda) - delta); (ii) after the margins are known, do a
  guaranteed recall of kept V tiles that were not prefetched. The attended set is exactly BLASST's, so no
  accuracy run is needed. For Llama-3.1-8B in bf16, K and V are each 64 KiB per token (32 layers x 8 KV
  heads x 128 dims x 2 bytes), so K-resident/V-offloaded halves resident KV per token from 128 KiB to
  64 KiB plus the cached V tiles.
- **What we measure (replayed offline from Experiment 1's dump, no model needed).** Per decode step, at
  calibrated 50% and 70% skipped (decode calibration): bytes of V needed (kept tiles x 32 KiB per tile at
  the KV-head level); bytes recalled with no prefetch versus with prefetch at guard bands delta = 0, 0.5,
  1, 2 nats; over-fetched bytes (prefetched but not kept); misses. Then the **measured** piece: a PyTorch
  microbenchmark on GH200 that gathers that many 32 KiB V tiles from pinned host memory to the GPU
  (`index_select` on a pinned tensor, `.cuda(non_blocking=True)`), reporting milliseconds per step for the
  observed byte counts. Compare against a measured dense decode step time for the 8B model at 8K and 16K.
  Back-of-envelope to be replaced by measurement: at 16K with 50% skipped, about 512 MiB of V is needed
  per step; at a 97% hit rate about 15 MiB must be recalled after the exact decision. At 64 GB/s (PCIe
  Gen5, ECHO's testbed) 512 MiB takes about 8 ms and 15 MiB about 0.25 ms; at the 900 GB/s the paper quotes
  for Grace NVLink-C2C the same transfers take about 0.6 ms and 0.02 ms. A dense decode step for an 8B
  model is roughly 15-25 ms, so without prefetch the recall is a large fraction of a step on PCIe machines
  and small on our GH200; with prefetch it is negligible on both. The experiment replaces every number in
  this paragraph with a measured one.
- **Also report a capacity table** (arithmetic from the traces, ECHO Fig. 1 analog): resident bytes per
  request at 8K, 16K, 32K, 128K for dense versus K-resident/V-offloaded, and how many concurrent requests
  fit in the roughly 75 GiB left on a 96 GB GH200 after 16 GB of weights.
- **Comparison.** ECHO's own per-step costs (recall 0.145 ms, 37 tokens recalled per step at 0.97-0.99 hit
  rate) and its 1.29x / 1.51x microbenchmark gains at hit rates 0.5 / 0.9; the dense decode step time.
- **Cost.** Replay is CPU work over the 3.3 GB dump. Microbenchmark about 0.1 GPU-hour. Optional end-to-end
  check (actually keeping V on the host for 20 prompts at 16K, about 20 x 21 s = 7 min) to confirm the
  outputs match the all-on-GPU run: about 0.5 GPU-hour. **Budget 1 to 2 GPU-hours.**
- **Why it is a contribution.** ECHO's prefetch framework applied to a training-free, block-wise, GQA,
  threshold rule is the case its Discussion lists as future work. BLASST's own paper has no memory-capacity
  story; AMEND's decode-step reuse is lossy and PIM-only; our Direction 5 was framed as lossy reuse with an
  audit and needed 15-20 GPU-hours of accuracy runs. This version is lossless, so the whole study is a
  traffic and hit-rate measurement plus one measured copy time, at about 2 GPU-hours. It also produces a
  concrete systems figure for the final report: bytes per decode step with and without prefetch versus
  guard band, with milliseconds on real hardware.

### Experiment 3: is a threshold rule more prefetch-friendly than top-k at the same % skipped?

- **What we measure.** From the same decode dump, build two masks per (layer, head, step): the BLASST mask
  at calibrated lambda, and a **top-k tile mask** with k equal to the number of tiles BLASST kept in that
  (layer, head) averaged over the prompt, so both have the same % skipped. Measure the step-to-step hit rate
  of each (Experiment 1's metric), and the fraction of tiles whose verdict is "decidable early", i.e.
  known before the whole row is scored (always 100% for the threshold rule; for top-k only tiles whose score
  is above the previous step's k-th score, which is ECHO's EMA trick).
- **Comparison.** ECHO's Fig. 8 (candidate tokens above the EMA-predicted threshold stay close to 2,048)
  versus BLASST's calibrated rule, which needs no prediction. Also matched-sparsity accuracy is already
  known for BLASST from our Fig. 2 and for top-k methods from the literature, so no new accuracy runs.
- **Cost.** 0 GPU-hours (offline replay).
- **Why it is a contribution.** This is the direct quantitative test of ECHO's co-design sentence. Nobody
  has compared threshold and top-k rules on the *predictability* axis at matched sparsity; the literature
  compares them on accuracy. If the threshold mask is more stable step to step, that is a systems argument
  for BLASST's rule independent of accuracy, and it is new.

### Experiment 4: query-tile-axis predictability in prefill (ECHO's inter-query prefetch) and the GQA union, from the saved margins, 0 GPU-hours

- **What we measure.** On the saved prefill margins (3 GB, 105 prompts at 4K-32K): for consecutive query
  tiles i and i+1 at the same (layer, head), the overlap of kept key tiles (ECHO's inter-query prefetch
  assumes this overlap exists and admits its 1.1x microbenchmark ignored it). Also the **KV-head-level %
  skipped**: a V tile is skippable only if all 4 query heads of the GQA group skip it; report this next to
  our query-head-level 50.82% at 8K and 53.38% at 16K, which is the number that actually governs V bytes
  in a K-resident/V-offloaded prefill with chunked prefill streaming V from the host.
- **Comparison.** Our own query-head-level % skipped (Fig. 1 table); ECHO's statement that during prefill
  "the total selection is the union set of all prefill tokens", which is why ECHO disables offloading in
  prefill; if the query-tile overlap is high for BLASST, prefill offloading looks less hopeless for a
  threshold rule than for top-k.
- **Cost.** CPU only, an afternoon of scripting against the existing margin files.
- **Why it is a contribution.** The KV-head-level number is one nobody reports for BLASST, and it is the
  honest sparsity for any byte-saving claim on a GQA model. The query-axis overlap is the prefill
  counterpart of Experiment 1 and completes a three-axis "predictability map" (query, layer, step) of the
  BLASST mask, where ECHO and Kascade each assume one axis without measuring the others on the same model.

### What not to do

- Do not try to reproduce ECHO: it needs 8 x H20, 1.5 TB RAM, DeepSeek-V3.2, and its claims are about a
  model and a serving stack we are not studying.
- Do not build a real offloading serving system: ECHO's cache manager is Triton + CUDA kernels inside
  SGLang; five weeks of PyTorch cannot match it, and the course does not require it. Report bytes, hit
  rates and measured copy times, and label any wall-clock projection as a projection.

### What to put in the Nov 4 report

One paragraph in related work (Sec. 3): ECHO shows the long-context serving bottleneck is KV capacity and
host-GPU bandwidth, that threshold selection enables lossless prefetching, and that it leaves
training-free block-wise threshold rules on GQA models as future work. One sentence in the revised plan
(Sec. 6): Experiments 1-2 above as a planned systems evaluation of BLASST's rule, about 3 GPU-hours.

---

## 6. Caveats

- I read the complete text (22 pages including references and the artifact appendix), but figures were
  read through their captions and the surrounding text, not the plots themselves. All numbers above are
  as stated in the text. Table 1's cell contents did not survive text extraction; the lossy/lossless
  distinction for InfiniGen/FreeKV/CLO versus ECHO is reconstructed from the prose in Secs. 1 and 2.4.
- I did not search for an arXiv version because the USENIX PDF downloaded complete; metadata was
  cross-checked on the USENIX abstract page.
- Every mapping from DSA/ECHO to BLASST in Sec. 3.2 and Sec. 5 is my analogy. ECHO never mentions BLASST,
  tiles, Llama, or running maxes; the only GQA/block mention is the one Discussion paragraph quoted above.
- ECHO's hit rates and prefetch gains are for a trained token-level top-k on a 671B MoE at 80K-100K
  context. BLASST's tile-level threshold rule on an 8B model at 8K-16K may behave differently; that is the
  point of Experiment 1, and it may come out less stable than 0.97.
- GPU-hour estimates use the caller's 5 s / 15 s per prompt and the team's 50 ms per decode step figure
  from `directions_idea2.md`; re-estimate after the first decode-recording run. The capacity and transfer
  arithmetic in Experiment 2 uses nominal bandwidths (64 GB/s PCIe Gen5 from the paper; 900 GB/s NVLink-C2C
  as the paper quotes for Grace systems); measured bandwidth will be lower, and the "15-25 ms dense decode
  step" is a rough figure for an 8B model, not a measurement.
- The reasons in Sec. 4 are guesses about Prof. Lai's intent. The cheapest way to find out is to ask him
  at the Nov 11 check-in which of (a) systems angle, (b) prefetch-friendly co-design, (c) evaluation
  methodology he had in mind.
- Experiment 1's "disable end-of-sequence, run 128 steps" choice makes post-answer margins less
  meaningful; if that worries us, use a long-answer task (summarization, or Qwen3-8B with thinking on, as
  Direction 5 already suggested) at a higher cost per prompt.
