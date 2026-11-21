# Idea 2, revisited: open directions for "reuse skip decisions instead of recomputing them"

Written 2026-11-01 for the CS 498 project (BLASST baseline). This follows up `related_work.md` § Idea 2,
which found that the original Idea 2 (reuse one layer's skip mask in the next few layers) belongs to a
published family: Kascade, IndexCache, TidalDecode, YOIO, Autonomy-of-Heads/H2Share, LongCat LSA,
SharePrefill, LiteAttention, AMEND.

**How the quotes were obtained.** For Kascade, TidalDecode, IndexCache, YOIO, LongCat, LiteAttention,
AMEND, Autonomy-of-Heads, SharePrefill, RouteRelay, HEART/HASTE and BLASST, the arXiv HTML was downloaded
with curl and read as plain text. Everything in quotation marks is copied exactly from that text; only
inline math was rewritten as plain text. Where I only saw an abstract through a summarizing fetch tool
(Chorus II, PRR, STS, MapFlow, RBS-Attention, Certified Sparse Attention via Perturbed Retrieval, BlindSight),
I paraphrase and do not quote. Numbers marked "my projection" are my arithmetic, not paper results.

## 0. Words used in this note

These are the project's words (see `00_team_plan.md` § 2), plus a few new ones.

- **Tile, skip rule, λ, margin, % skipped, dense, calibrated**: same meaning as in the team plan.
  - A tile is skipped when its **margin** (the largest value, over its 128 rows, of tile row-max minus
    running row-max) is below ln λ.
  - A **kept tile** is a tile the skip rule does not skip.
- **Anchor layer**: a layer that computes its own skip decisions in the normal BLASST way.
- **Reuse layer**: a layer that borrows an anchor's decisions. For a tile the borrowed mask says to skip,
  the reuse layer computes nothing at all, so it also avoids **QK^T**. That is the whole point: BLASST
  alone always computes QK^T.
- **Recall of kept tiles**: of the tiles that layer l's own BLASST pass would keep, the fraction that the
  borrowed mask also keeps.
  - 1 − recall is the **wrong-skip rate**: tiles BLASST would have kept that the reuse layer threw away.
- **% QK^T avoided**: the number of tiles whose QK^T is never computed, divided by all causal tiles.
- **Refresh**: a reuse layer gives up the borrowed mask and recomputes its own. It may also become the
  new anchor for later layers.
- **Drift check**: a cheap test that decides whether to refresh.
- **Top-k / top-p selection**: keep a fixed number k of keys or blocks (top-k), or the smallest set whose
  attention mass reaches p (top-p).
  - BLASST is neither. It is **threshold-based**: how many tiles survive depends on the input, so there is
    no fixed budget.
- **GQA (grouped-query attention)**: several query heads share one key/value head.
  - Llama-3.1-8B: 32 query heads and 8 KV heads, 32 layers.
  - Qwen3-8B: 32 query heads and 8 KV heads, 36 layers.
  - Qwen2.5-VL-7B language model: 28 query heads and 4 KV heads, 28 layers.
  - These shapes are from memory; check each model's `config.json`.
- **PIM (processing-in-memory)**: logic placed inside the DRAM that can do simple math, such as dot
  products, next to the data.

---

## 1. What each paper admits, and what it did not test

### 1.1 Kascade (arXiv 2512.16391; v2, 31 Aug 2026; text identical to v1)

**What it does.**
- Anchor layers compute exact post-softmax attention, pool it over 128-query tiles, and keep the top-k
  keys for each KV head.
- Reuse layers load only those keys ("we load the keys by consulting the Top-k indices").
- Each reuse head is mapped to its most similar anchor head; the mapping "can be a many-to-one mapping".
- The budget is a fixed 10% of the context.

**Admitted limitations** (§5 Conclusion, verbatim):
> "There are a few limitations of this work. First, this technique requires a development set to compute
> the anchor layers and head mappings. It is possible that this biases the technique towards the data in
> the development set. However, in the experiments we have done, we have found the selections to be robust
> to different datasets. Second, while Kascade reduces attention latency, it doesn't reduce the memory
> capacity requirements for attention. … Last, architectures which are trained with sparsity like Team et
> al. (2024); Agarwal et al. (2025) will benefit less with this scheme."

The "robust to different datasets" claim comes with no data.

Other admissions, verbatim:
- §3.6: "We see that the recomputation in the second pass, for prefill, is a significant cost."
- §4.3: "Prefill speedup takes a hit primarily because of the recomputation of attention weights in the
  second pass of anchor layers."
- Layer 0: "The only exception is layer 0, where the distribution is considerably flatter. Kascade therefore
  always computes full dense attention in layer 0".

**Ablations and numbers.**
- Head remapping on MuSiQue: "No remapping is the worst."
- On the LongBench average, pooling all heads together actually **beats** remapping, and the paper does
  not comment on it:
  - Llama: 45.70 vs 45.02.
  - Qwen3: 45.02 vs 44.57.
- Anchors (all from a dynamic-programming search on MuSiQue):
  - Llama-3.1-8B: [0, 2, 8, 13, 14].
  - Qwen3-8B: [0, 2, 7, 14, 23].
- Kernel times on one H100 at 10% top-k:
  - Prefill at 128K, batch 1: 215.76 ms (FA3) vs 98.55 ms (Kascade).
  - Prefill at 8K: 0.76 ms vs 0.62 ms.
  - Decode at 128K, batch 64: 11.68 ms vs 2.83 ms.

**Not ablated:**
- Number of anchors (always 5).
- Reuse distance.
- DP-chosen anchors vs hand-picked anchors.
- Choice of development set.

**Not evaluated:**
- Threshold-based selection. The word "threshold" never appears.
- Multimodal inputs.
- Reuse quality vs context length. The similarity matrix comes from MuSiQue, which has a 2.3K average
  context, and there is no RULER.
- Reuse quality per task.
- Any runtime drift check or refresh. The schedule is fixed offline: "we pass the Top-k indices from the
  previous anchor layer, and the head mapping computed on the development set".
- The cost of head remapping, measured separately.
- Reuse across decode steps or across requests.

### 1.2 IndexCache (arXiv 2603.12201)

**What it does.** It reuses DeepSeek-Sparse-Attention (DSA) lightning-indexer top-k indices
(k = 2048) across layers. Only some layers keep their indexer; the rest reuse the nearest kept layer's
indices.

There is no Limitations section. The closest statements, verbatim:
- §5.2: "the core principle extends to any sparse attention method that does not rely on a fixed sparse
  pattern but rather involves a dynamic token selection step: for instance, the block-level selection in
  MoBA (Lu et al., 2025) and NSA (Yuan et al., 2025) could similarly benefit from cross-layer reuse."
  This is an extension they name but did not test.
- On calibration: "Results are stable across different calibration sets, indicating that this importance
  ranking is an intrinsic model property".
- **App. C (an important negative result for any drift-check idea):** "local metrics—whether cosine
  similarity of attention outputs or top-k index overlap—lack the discriminative power to identify the
  optimal sharing pattern, necessitating end-to-end evaluation."
- App. A: "overlap is an aggregate metric: it counts how many tokens are shared but not which ones differ."

**Numbers.**
- Adjacent-layer overlap is "0.7-1.0"; early vs late layers "≤ 0.4".
- Training-free long-context average (original DSA = 50.2):

  | Indexers kept | Uniform interleaving | Searched pattern |
  |---|---|---|
  | 1/2 | 47.4 | 50.3 |
  | 1/4 | 43.0 | 49.9 |
  | 1/8 | 35.3 | 46.1 |

  Their summary: "which indexer layers are retained matters far more than how many."
- Prefill at 200K on H100: 19.5 s (DSA), 13.7 s (keep 1/2), 10.7 s (keep 1/4).

**Not evaluated:**
- Threshold or variable budget.
- Multimodal inputs.
- Any model without a trained indexer.
- Quality as a function of context length (RULER is reported only as one aggregate).
- Runtime refresh.
- Head-level reuse.
- Reuse across decode steps. Verbatim: "the decode phase in DSA involves a per-token indexer pass over
  the full context".
- Reuse across requests.

### 1.3 TidalDecode (arXiv 2410.05076, ICLR 2025)

**What it does.** Decode only. Two "token selection" layers pick top-k token positions, and the following
layers reuse them. Prefill is full attention: "TidalDecode uses the same prefilling mechanism as existing
systems and performs full attention to compute the key-value (KV) cache for all prompt tokens."

**Admissions,** verbatim:
- "our preliminary experimentation shows that using a single token set for all Transformer layers reduces
  the LLM's predictive performance by a large margin since distant Transformer layers are less correlated
  compared to consecutive layers"
- "However, deciding which layer to perform token re-selection is critical to model performance."
- "The choice of T can be at the level of thousands of decoding steps but also depend on different models
  and tasks." (T is the cache-correction period.)

**Numbers.**
- With a single selection layer at layer 3, recall is "less than 20%". Adding re-selection at layer 13
  brings it to "almost 40%".
  - Measured on Llama-3-8B, NIAH, 100K context, top-256.
- The re-selection layer was picked by hand sweeps:
  - Llama-3.1-8B at 10K: layer 13 gives 100% NIAH at K = 32 to 256.
  - Layer 12 gives only 8 / 10 / 28 / 40%.

**Not evaluated:**
- Threshold selection.
- Prefill.
- Multimodal inputs.
- Any runtime drift check.
- Head remapping.
- Reuse across decode steps. Selection is redone at every step.

### 1.4 LiteAttention (arXiv 2511.11062, video diffusion transformers)

**What it does.** This is the only paper that reuses a **threshold-based** online-softmax skip decision.
- The skip test is SpargeAttention's: skip a tile if `max(m_local − m_ij) ≤ −ε`. That is the same test
  as BLASST with ln λ = −ε. BLASST itself is not cited.
- Skip decisions are carried **across diffusion timesteps**, not across layers.
- Skipped tiles skip QK^T and the K/V loads: "If a tile is marked as skippable, the producer omits loading
  the corresponding K_j and V_j tiles from global memory, allowing the consumer to bypass all computation
  for the entire j'th iteration."
- The mask only ever grows. Once a tile is skipped, it is never un-skipped: "As the diffusion process
  progresses, the number of tiles marked for skipping gradually increases."

**Admissions,** verbatim:
- "Nevertheless, our skipping scheme only relies on the slow-evolution assumption to bind the tiles
  eligible for skipping. Tiles that are not skipped are recomputed from scratch and may undergo large
  changes." This means there is no check that a reused skip is still valid.
- "Video quality degrades sharply beyond 70% sparsity."

**Numbers.**
- Wan2.1-14B, total time for one video: 1707 s (FA3) → 902 s, at 42% sparsity.
- Self-attention time vs sparsity: 695 s at 0%, 418 s at 42%, 163 s at 77%.

**Not evaluated:**
- LLMs and VLMs.
- Reuse across layers.
- Any refresh, or any measurement of how many skipped tiles later become important.
- Only 12 prompts were used.

### 1.5 AMEND (arXiv 2609.09823, "Audited Margins Enable Nonblocking Drops in GPU–PIM LLM Decoding")

**What it does.** It predicts each BLASST skip decision at decode step t from steps up to t−1.
- AMEND calls a skip decision a "verdict" and a skipped block a "drop".
- The predictor is an EMA (exponential moving average: a running average that weights recent steps more)
  of each block's margin, plus a hysteresis counter (it must agree for a few steps before it flips).
- Every predicted drop is **audited** at every step on the PIM side: the real score is computed and used
  to update the predictor.

It deliberately uses thresholds instead of top-k, verbatim: "a max-relative verdict is a per-block
comparison whose history can be tracked on its own, whereas a Top-K keep depends on the ranking of every
other block."

**Admissions,** verbatim:
- "A predicted drop is irreversible for the current token, so incorrect drops must be bounded, not
  repaired."
- "Prediction alone, however, opens an observation hole: a dropped block is never scored again, so an
  incorrect drop removes the very evidence that could correct it"
- On hardware: "the GPU–PIM mapping of Section 4.3 is modeled in this simulator". There are no measured
  GPU kernel timings.

**Numbers.**
- "89.2% of eligible verdicts remain unchanged at the next step".
- Across the full evaluation: "false-drop rates of 1.50–2.24% at the 2% target and drop recall of
  72.60–90.75%".
- Llama recall by context length: 85.30% (8K), 88.26% (16K), 90.75% (32K), 90.50% (64K).
- Without the audit, the rolling false-drop rate is 0.087; with it, 0.028.
- LongBench average for Llama-3.1-8B: 49.69 (dense), 49.88 (BLASST), 49.76 (AMEND).
- BLASST skips 0% of K bytes; AMEND skips 83.57–87.99% of GPU-side K bytes on Llama.

**Not evaluated:**
- Prefill. The word appears 0 times.
- Reuse across layers.
- A GPU-only variant. The audit relies on PIM.
- Long reasoning generations.
- Absolute decode times. Only simulated ratios are given.

### 1.6 SharePrefill (arXiv 2505.19578)

**What it does.** Within one prefill pass, one "pivotal" head per offline cluster (the clusters span
layers) builds a top-p block mask with γ = 0.9.
- Other heads in the cluster reuse it **only if a drift check passes**: "If the JS distance is below a
  given similarity threshold τ, we share the pivotal pattern with the current head. Otherwise, we fall back
  to a conservative vertical-slash pattern".
- JS distance is the Jensen–Shannon distance between two attention distributions; τ = 0.2.

**Future work and limitations,** verbatim:
- "The principle of similarity between heads and the proposed pattern-sharing mechanism holds the
  potential for accelerating the decoding phase and extending to multi-modular systems, which will be
  explored in future work."
- "the underlying explanation for the highly consistent similarity relationships among heads across
  different inputs remains unclear."

**Number.** At 128K, removing sharing leaves latency about the same (17.01 s vs 16.92 s) but drops
Retr.PassKey from 100 to 22. So sharing bought accuracy, not speed.

**Not evaluated:**
- Threshold masks.
- Multimodal inputs.
- Decode.
- How often the JS check rejects reuse.

### 1.7 Autonomy-of-Heads / H2Share (arXiv 2608.06849)

**What it does.** H2Share is only Appendix J of the "Autonomy-of-Heads" paper.
- Anchor layers record top-k token indices.
- Reuse heads are remapped by a weights-only similarity: the Frobenius inner product of each head's
  W_K^T W_Q matrix.
- Verbatim: "Naively giving every Reuse Retrieval Head the indices selected by the same-indexed Anchor Head
  assumes that head indices have identical functions across adjacent layers. This assumption is
  unreliable."

**Limitations,** verbatim:
- "we have not exhaustively tested it on encoder-decoder models, multimodal LLMs, or models with heavily
  modified attention mechanisms."
- "AoH uses a fixed sparsity budget … We leave automatic budget selection and finer-grained head policies
  for future work."

**Numbers.** H2Share results appear only as a figure; no k, no block size, no latency.

### 1.8 LongCat Sparse Attention (arXiv 2608.01662) and YOIO (arXiv 2606.06467)

Both are **trained** architectures that reuse fixed top-k (2048) indices across layers. Neither applies to
an off-the-shelf model, so they matter here only as warnings.

**LongCat.**
- §3.2: "Naively reusing indices without training adaptation degrades performance".
- Measured: needle-in-a-haystack at 128K is 70% with training-free reuse (group size N = 2), vs 96%
  after distillation.
- "Beyond 32K context, N=4 suffers a clear accuracy degradation that grows with sequence length".
- "Adjacent layers share 57.4% of their Top-K budget on average… still captures 93.2% of the target
  layer's attention mass."
- It uses a fixed "uniform interleave pattern". Its stated limitation is the KV-cache footprint.

**YOIO.**
- One single-head indexer is shared by all 16 cross-decoder layers.
- Quality is measured only up to 32K, while the speedups are reported at 128K.
- It has no Limitations section.

### 1.9 Papers found during this search (not in our earlier survey)

| Paper | Axis of reuse | Selection rule | Drift check? | Scope / what is missing |
|---|---|---|---|---|
| **RouteRelay** (2609.07306, Sep 2026) | Across layers | Top-k chunks | **Yes: "near-miss" and random "sentinel" chunks are re-scored; a row is re-routed only if a sentinel beats its weakest kept chunk** | **Synthetic data only, CPU only, no accuracy** (details below) |
| **HEART** (2605.14513; v1 titled HASTE) | Across diffusion timesteps | Top-p | **Yes: per-head query-key drift**; refresh "only when the drift exceeds a prescribed threshold" | Video diffusion only (details below) |
| **Chorus II** (2606.25040) | Across requests | Not stated in abstract | Not stated in abstract | Image-to-video diffusion. Per the abstract, it reuses sparse masks from similar earlier requests; 2.16x speedup. No LLMs |
| **PRR** (2606.30389) | Across decode steps | Top-k blocks | Repairs missed blocks afterwards | Per the abstract: an EMA predictor plus repair using online-softmax statistics; up to 40% lower per-token latency |
| **LServe** (2502.14866) | Across decode steps | Top-k pages | No | Re-selects pages every ~4 steps (from secondary summaries, not checked in the paper) |
| **NOSA** (2510.13602) | Across decode steps | Top-k | No | Measures that ≥ 0.8 of selected tokens repeat from the previous step in most layers (from a search snippet) |
| **STS** (2605.15508), **SpecAttn** (2510.27641), **MapFlow** (2610.14688) | Across models (draft → target, or sender → receiver) | Various | — | Draft-model or other-model attention used to predict the target's sparsity |
| **RBS-Attention** (2609.20971) | None | Thresholds on centroid and radius scores | — | Uses a key-block radius as a *risk score*, not a certificate; prefill; H100 |
| **Certified Sparse Attention via Perturbed Retrieval** (2610.24991) | None (fresh each step) | Threshold on perturbed scores | Probabilistic certificate | Decode only, CPU timings |

**RouteRelay details.**
- Setup, verbatim: "We generate 32 layers, 384 query rows, and 192 key chunks. Each row has persistent
  preferred chunks together with layer-local Gaussian drift."
- Result: recall ≥ 99.99% while re-routing 25.0% / 55.4% / 78.2% of rows at drift 0.01 / 0.03 / 0.06.
  At drift 0.03, a fixed stride of 4 drops to 94.9% recall.
- "Equation 10 does not guarantee perfect recall, but makes the overhead–risk trade-off explicit."
- "The slower unfused CPU path shows that these arithmetic savings require a fused kernel to translate
  into latency gains."
- No real model, benchmark or GPU appears anywhere.

**HEART details.**
- The query-key drift signal has Spearman correlation "− 0.7914 and − 0.7568 with adjacent-step mask IoU"
  (full-token and mean-pooled versions).
  - Spearman correlation is a rank correlation; −1 means "more drift always means less overlap".
  - IoU (intersection over union) is the overlap between two masks.
- Wan2.1 models; on top of XAttention and SVG2.

### 1.10 BLASST's own words that bear on reuse

- **Appendix B (error bound), verbatim:** "When BLASST skips block j, the criterion m̃(j) − m(j) < ln λ
  guarantees exp(m̃(j) − m(j)) < λ … Since m(j) ≤ M, every score in a skipped block satisfies
  exp(s_jk − M) ≤ exp(m̃(j) − M) ≤ exp(m̃(j) − m(j)) < λ."
  - So any skip based on an **upper bound** on the tile max, compared against **any lower bound** on the
    row max, keeps the same guarantee. Direction 3 uses this.
  - A plain reused mask loses the guarantee.
- **Fig. 7 caption:** "Substantial head-level and layer-level variance motivates adaptive thresholding
  strategies." The number of kept tiles under one λ differs a lot by layer. A mask copied from layer a
  therefore carries layer a's density, not layer l's.
- **App. A.4, tile reordering:** processing tiles in reverse order (local window first) "has a negligible
  impact on model accuracy at a given sparsity level" (Fig. 9 caption), "but provides noticeable
  improvements on FWE". This weakens any direction built on a better starting running max (see § 4).

---

## 2. The gap matrix

Columns:
- **Threshold?** Is selection threshold-based (variable budget) rather than top-k/top-p?
- **Axis.** What the decision is reused across.
- **Runtime drift check?** Does the method decide at run time whether reuse is still valid?
- **Guarantee?** Is there a per-tile error bound on skipped work?
- **Real GPU time?** Are measured GPU timings reported?

| Method | Threshold? | Axis | Phase | Runtime drift check? | Guarantee? | Multimodal? | Real GPU time? |
|---|---|---|---|---|---|---|---|
| Kascade | No (top-k) | Layers (+ head remap) | Prefill + decode | No | No | No | Yes (H100) |
| IndexCache | No | Layers | Both | No | No | No | Yes (H100) |
| TidalDecode | No | Layers | Decode | No | No | No | Yes (A100) |
| H2Share | No | Layers (+ remap) | Decode (ambiguous) | No | No | No (admitted) | No |
| LongCat / YOIO | No | Layers (trained) | Both | No | No | No | Yes |
| SharePrefill | No (top-p) | Heads (clusters span layers) | Prefill | **Yes (JS distance)** | No | No (future work) | Yes (A100) |
| RouteRelay | No | Layers | Prefill | **Yes (sentinels)** | Probabilistic miss bound | No | **No (CPU, synthetic)** |
| LiteAttention | **Yes** | Diffusion timesteps | n/a (DiT) | No | No | Video generation | Yes (H200) |
| HEART | No (top-p) | Diffusion timesteps | n/a | **Yes (QK drift)** | No | Video generation | Yes |
| AMEND | **Yes (BLASST)** | Decode steps | Decode | **Yes (audit on PIM)** | Statistical (2% false-drop target) | No | **No (simulated)** |
| **Open cell** | **Yes** | **Layers / heads / frames** | **Prefill** | **To test** | **To test** | **To test (Qwen2.5-VL)** | Simulation + optional kernel |

The empty region: **threshold-based (BLASST) skip decisions reused across layers in LLM prefill**, with
or without a drift check or certificate, on real models, has not been published as far as these searches
show. Neither has any reuse of skip decisions inside a VLM.

---

## 3. Candidate directions

All experiments assume the team's setup:
- A PyTorch simulation of BLASST registered through transformers' `AttentionInterface`
  (`code/blasst_attn.py`). It records per-tile margins of shape `[layers, heads, q tiles, k tiles]` in a
  dense pass, so that one pass gives the mask for every λ.
- One GH200 (96 GB) on DeltaAI.
- Llama-3.1-8B-Instruct, Qwen3-8B, Qwen2.5-VL-7B.
- RULER, LongBench, Video-MME.

**Cost unit.** The simulation has never been run, so the per-sample time is an assumption to replace after
the smoke test (Project `CLAUDE.md`, step 4).
- Assume about 5 s per 8K prefill and 15 s per 16K prefill.
- One **RULER accuracy configuration** = niah_multikey + vt + fwe × 50 samples × {8K, 16K} = 300 runs.
  That is 150 × 5 s + 150 × 15 s = 3,000 s, plus generation, so **about 1 GPU-hour**.
- One **margin-recording batch** = 20 samples × {4K, 8K, 16K}, about 0.2 GPU-hours. Kevin's baseline
  already produces this.
- The group allocation had 541 hours left on Nov 1.

**Recommended recorder addition (useful for directions 1–3).** Next to each margin, store the **tile
mass**: the largest share, over the tile's 128 rows, of that row's softmax mass that falls inside the
tile, in fp16 with the same shape.
- Margins say whether BLASST would keep a tile. Mass says how much a wrong skip would cost.
- Kascade's own similarity measure is mass-based, so this makes our numbers comparable.
- It costs one extra softmax sum per chunk, because the dense pass already holds the full score row.

### Direction 1: reuse a threshold mask across layers, find what breaks, and add a guard band

**Idea in plain words.**
- In anchor layers, run BLASST as usual and save the mask and margins.
- In reuse layers, skip every tile the anchor skipped, so no QK^T is computed for them.
- Optionally run BLASST's own test on the surviving tiles as well. The running max is then computed over
  survivors only, so it is lower than the true one, which can only make that test skip less (AMEND §5.2
  proves the same "no additional false drop" property for its survivor test).

**Three ways a threshold mask breaks that a top-k index set does not:**
1. **Density mismatch.** Layer a's number of kept tiles is not layer l's (BLASST Fig. 7). A denser target
   layer silently loses tiles.
2. **No guarantee.** BLASST App. B no longer covers the reused skips.
3. **Length scaling.** Calibrated λ = a/L moves with context length, so agreement between layers could
   change with length. Kascade measured only at a 2.3K average context.
   - Evidence that length matters for reuse: LongCat: "Beyond 32K context, N=4 suffers a clear accuracy
     degradation that grows with sequence length". AMEND's decode-step recall instead *rises* with length,
     from 85.30% to 90.75%.

**The fix to test: a guard band.** Keep a tile in layer l if the anchor's margin is at least ln λ − δ,
rather than at least ln λ.
- δ is calibrated offline for each (anchor, layer) pair so that recall of kept tiles reaches 99% (and
  99.9%) on niah.
- It is then tested on vt, fwe, LongBench and other lengths.
- This is RouteRelay's "guard margin" idea, carried over from top-k to thresholds.
- Also compare three head mappings:
  - same index;
  - Kascade-style remap, learned on the niah calibration samples;
  - union of the anchor masks inside a GQA group, i.e. keep a tile if any head sharing that KV head keeps
    it. This also covers direction 6.

**Gap it fills.**
- Every cross-layer method is top-k: Kascade, IndexCache, TidalDecode, H2Share, LongCat, YOIO. IndexCache
  names only "MoBA … and NSA" as next targets.
- The one threshold-mask reuser, LiteAttention, reuses across diffusion timesteps, has no refresh, and
  admits it "only relies on the slow-evolution assumption".
- AMEND explains why thresholds suit reuse ("a max-relative verdict is a per-block comparison whose history
  can be tracked on its own") but applies this only across decode steps.

**Prior-art status: appears open.**
- Searches run:
  - "cross-layer reuse threshold-based sparse attention mask variable budget prefill"
  - "\"previous layer\" skip mask FlashAttention tiles reuse LLM prefill online softmax threshold"
  - "diffusion transformer sparse attention mask shared across adjacent layers"
- None found a threshold or skip-softmax mask reused across layers in an LLM.
- Closest: LiteAttention (different axis); Kascade and IndexCache (top-k).

**Experiment.**
1. **Offline (no new GPU time beyond the margins Kevin records).**
   - For Llama-3.1-8B and Qwen3-8B, 20 samples × 4K/8K/16K, λ at calibrated 50% and 75% targets.
   - For every pair (a, l) with l − a = 1 … 4, and each head-mapping option, compute:
     - recall of kept tiles;
     - largest lost tile mass per row;
     - % QK^T avoided;
     - all of these split by length and by task.
   - Then fit δ and recompute on held-out tasks and lengths.
2. **Online accuracy.** Add a "mask cache" to `blasst_attn.py`: anchor layers store the mask, reuse
   layers apply it.
   - Anchors: Kascade's [0, 2, 8, 13, 14] (Llama) and [0, 2, 7, 14, 23] (Qwen3), plus "every 4th layer".
   - Configurations: dense; BLASST 50%; BLASST 75%; reuse with δ = 0; reuse with the fitted δ;
     two anchor sets.
   - Report RULER accuracy, % skipped, and % QK^T avoided.
   - Optionally add a LongBench subset (6 tasks × 50).

**Metrics.**
- Recall of kept tiles, and largest lost mass per row.
- % QK^T avoided.
- RULER accuracy at matched % skipped versus plain BLASST.

**Upper bound (my projection).** At 50% skipped, with 27 of Llama's 32 layers reusing, at most
0.5 × 27/32 ≈ 42% of all QK^T tiles are avoided. The guard band can only lower this.

**Cost.**
- About 0.2–0.5 GPU-hours for margins (shared with the baseline).
- About 6–8 GPU-hours for 6–8 accuracy configurations.
- **About 3–4 engineering days**: 1–2 for the mask cache, 1–2 for the analysis. Teammate C's planned
  reuse simulator covers part of this.

**Real kernel needed?** No, for the claim itself: the simulation measures recall, accuracy and tiles
avoided exactly.
- Wall-clock time needs a kernel that skips QK^T for masked tiles.
- Cheapest real path: PyTorch **FlexAttention with a `BlockMask`** built from the anchor mask (128×128
  blocks; blocks that are fully masked are skipped). That is 2–3 extra days. I have not checked that it
  runs on DeltaAI's torch 2.11 or on GH200.
- A cost model based on NVIDIA's kernel timings (QK^T is 54–61% of BLASST attention time,
  `blasst_paper.md` § 4.3) can give an estimate in the meantime.

**Kill criterion.** If adjacent-layer recall stays below about 90% even with remapping, or if 99% recall
needs a δ that leaves under about 10% QK^T avoided, report it as a negative result. That is still a
clean finding: "threshold masks do not transfer the way top-k indices do."

### Direction 2: a cheap runtime check that decides when to refresh the reused mask

**Idea in plain words.** Instead of a fixed anchor schedule chosen on a development set (Kascade,
IndexCache, TidalDecode, LongCat), let each reuse layer test cheaply whether the borrowed mask still fits
this input, and refresh only when it does not. Three candidate checks:
- **(a) Probe tile row.** Compute the true QK^T and BLASST decisions for one or two query-tile rows, for
  example the last row, which sees every key tile. Refresh the (layer, head) if the borrowed mask's recall
  on that row falls below τ.
  - Cost at 16K: one row of 128 tiles out of 8,256 causal tiles, i.e. 1.55% of the layer's QK^T.
- **(b) Near-miss sentinels,** RouteRelay carried over to thresholds. For each query-tile row, also
  compute the few tiles whose anchor margin was just below ln λ, plus a few random tiles. If any of them
  passes BLASST's test in layer l, refresh only that row.
- **(c) A residual-stream signal.** For example, cosine similarity between the layer's input and the
  anchor's input. It costs almost nothing, but IndexCache warns that such local signals "lack the
  discriminative power".

**Gap it fills.**
- Kascade admits the development set "biases the technique towards the data in the development set" and
  shows no data for its robustness claim.
- TidalDecode: "deciding which layer to perform token re-selection is critical".
- LongCat fixes "a uniform interleave pattern".
- Drift-gated reuse does exist on other axes, but not here:
  - HEART: diffusion timesteps, using query-key drift. That signal does not carry over to layers, because
    each layer has its own Q/K projection space.
  - SharePrefill: heads, using JS distance.
  - AMEND: decode steps, using a PIM audit.
  - RouteRelay is the only cross-layer one, and it was tested only on "layer-local Gaussian drift" in
    synthetic data, on CPU, without any accuracy measurement.

**Prior-art status: partially done.**
- The concept is done (RouteRelay for top-k on synthetic data; HEART and SharePrefill on other axes).
- **Left:** threshold masks, real LLMs, accuracy results, a comparison of the three signals at matched
  QK^T cost, and a test of whether the best schedule depends on the task. IndexCache says it does not;
  Kascade asserts robustness without data.
- Searches run:
  - "input-adaptive dynamic decision when to recompute sparse attention indices across layers similarity
    check refresh"
  - "cross-layer reuse threshold-based …" (this surfaced RouteRelay)
  - "diffusion … mask … reuse" (this surfaced HEART)

**Experiment.**
- **Almost fully offline.** The recorder stores the true margins of *every* layer, so any refresh policy
  can be replayed on CPU from the saved margins.
  - Policies (a) and (b) read the target layer's true margins only on the tiles they probe, and the probes
    are counted as QK^T work.
  - **Approximation:** inside a real reuse layer, the running max would come from kept tiles plus probes
    only, so it would be lower. That makes probes trigger refreshes more often, which errs on the safe side.
  - Signal (c) needs per-layer hidden-state statistics, which are cheap to add to the recorder.
- Compare against Kascade's fixed anchors and against fixed strides **at equal QK^T work**.
- Calibrate τ on niah; test on vt, fwe and LongBench.
- Then run accuracy for the best one or two policies.

**Metrics.**
- Curve of recall of kept tiles (or lost mass) against % QK^T avoided, counting the probe cost.
- Refresh rate by task and by length.
- RULER and LongBench accuracy at matched QK^T avoided.

**Cost.**
- Offline replay: CPU only.
- About 4–5 GPU-hours for 4 accuracy configurations, plus the margins from direction 1.
- **About 4–6 engineering days.** It builds directly on direction 1's mask cache.

**Real kernel needed?** No, for the scientific claim: refresh rates and accuracy at matched QK^T work.
- Per-row refresh (b) creates irregular work, and only a fused kernel turns it into time. RouteRelay
  states the same about its own method.

**Risk.** IndexCache App. C found that local similarity did not pick good sharing patterns. That is why
the end-to-end accuracy runs are required, not optional.

### Direction 3: "certified" reuse, where a cheap upper bound proves each reused skip is safe

**Idea in plain words.** BLASST's guarantee (App. B) needs only two things:
- an *upper bound* on the skipped tile's largest score;
- a *lower bound* on the row's true maximum.

In a reuse layer, the running max over the tiles we did compute is already a lower bound. For each tile
the borrowed mask wants to skip, compute a cheap upper bound U on the tile's largest score q·k.

- **Box bound.** Store, per key tile, the per-channel minimum and maximum of K (k_min, k_max). Then
  U = q⁺·k_max + q⁻·k_min, where q⁺ and q⁻ are the positive and negative parts of q. That is two dot
  products per row instead of 128, i.e. 1/64 of the tile's QK^T.
- **Ball bound** (center c, radius r of the keys): U = q·c + ‖q‖·r.
- **Low-rank bound.** Project onto the top few principal directions of K and bound the remainder.
  Loki (2406.02542, not re-checked this session) reports that keys are low rank.

If U minus the running max is below ln λ for every row of the tile, the skip is **certified**: it obeys
exactly the same bound as a native BLASST skip. Otherwise, compute that tile normally.

The reused mask supplies *which tiles to try*; the bound supplies *safety*. This ties to soundness
arguments in NN verification: it is the same "sound upper bound before pruning" logic.

**Gap it fills.**
- Every reuse method gives up per-tile guarantees:
  - AMEND: "incorrect drops must be bounded, not repaired", using a statistical false-drop budget.
  - LiteAttention: "only relies on the slow-evolution assumption".
  - RouteRelay: "does not guarantee perfect recall".
- BLASST's own appeal is that its skips come with App. B's bound.

**Prior-art status: appears open** for certifying reused skips in prefill.
- Bounds exist elsewhere, but only for *ranking* or *risk scoring*:
  - Quest: per-page min/max bounds to rank pages in decode.
  - RBS-Attention: radius used as a "rescue" risk score; per its abstract it does not claim a certificate.
- The only "certified" sparse attention found (2610.24991) is probabilistic, decode-only, and timed on CPU.
- Searches run:
  - "upper bound attention score per key block channel min max … skip block prefill exact guarantee"
  - "provable error guarantee sparse attention skip key blocks upper bound q·k certify"

**High risk; run a one-day kill test first.** My back-of-envelope, assuming random isotropic keys
(every direction alike), d = 128, 128 keys per tile:
- The box bound overshoots the true tile max by about 2.6·(0.8·√d − 1)·s ≈ 21·s nats, where s is the
  standard deviation of the scaled logits.
- That is far larger than |ln λ| ≈ 7–10 nats, so it would certify almost nothing.
- Real keys are strongly anisotropic: they have a large shared mean, outlier channels, and nearly constant
  low-frequency RoPE channels. Those effects tighten the bound, so the true certification rate is an
  **empirical question**.

**Experiment.**
1. **Kill test.**
   - On 2–5 samples where Q and K were saved (the team plan already saves Q/K for about 5 samples at 8K),
     or computed inside the attention function: for the tiles BLASST skips anyway, what fraction does
     each bound (box, ball, low-rank with r = 8/16/32, box over 16-key sub-tiles) certify, using the
     *true* row max?
   - This is an upper bound on what certified reuse could ever avoid.
   - Report it by layer and head.
2. If the rate is meaningful, run **direction 1's reuse with certification**:
   - % QK^T avoided after certification, minus bound cost, at zero uncertified skips;
   - one RULER sanity configuration.
   Accuracy should match BLASST at the same λ by construction, so the result is the work saved.

**Metrics.**
- Fraction of reused skips certified.
- Net QK^T FLOPs avoided after paying for the bounds.
- Results by layer and head.

**Cost.**
- About 0.5 GPU-hours for the kill test.
- About 2–3 GPU-hours if it continues.
- **About 3–4 engineering days.**

**Real kernel needed?** No. FLOP counting in the simulation shows whether certification pays. A real
kernel would be needed only for wall-clock time.

### Direction 4: modality-aware reuse in a VLM (Qwen2.5-VL-7B), across layers and across frames

**Idea in plain words.** Run direction 1's analysis on Qwen2.5-VL with long Video-MME videos, and split
every number by tile type (text or vision query × text or vision key). Two sub-questions:
1. **Layers.** Are vision-key tiles more stable across layers than text-key tiles? If so, reuse only
   vision-key decisions and recompute text tiles.
2. **Frames.** Within one layer, does a query tile in frame-group g+1 have nearly the same key-tile
   decisions as frame-group g? Video tokens repeat in a frame-periodic layout. If so, a query tile could
   borrow its neighbour frame's decisions instead of computing QK^T.
   - Choose the resolution so that one frame-group is a whole number of 128-token tiles; for example,
     256 tokens per two-frame group (`multimodal_qwen25vl.md` § 1).

**Gap it fills.**
- H2Share: "we have not exhaustively tested it on … multimodal LLMs".
- SharePrefill names "extending to multi-modular systems" as future work.
- Kascade, IndexCache, TidalDecode, LongCat and AMEND have no multimodal results.
- MMInference shows text-derived sparse indices "fail to generalize to visual ones", but never reuses
  decisions.
- LiteAttention is video *generation*, not video understanding.

**Prior-art status: appears open** for layers. **Partially done** for frames: MMInference's "Grid"
pattern fixes frame periodicity as a static pattern whose stride is found online, but nobody reuses
per-tile skip decisions across frames.
- Searches run:
  - "video LLM prefill sparse attention reuse mask across frames or layers Qwen2.5-VL"
  - "multimodal LLM cross-layer sparse attention index reuse anchor layers … VLM prefill"
  - "temporal redundancy consecutive frames attention pattern reuse video large language model prefill"
- Nearest results: ViCA (vision tokens skip layers), BlindSight (static per-head templates for
  multi-image), and frame dropping. None reuses skip decisions.

**Experiment.**
- Teammate D's pipeline: language model on "blasst", vision tower on SDPA.
- Record margins on 20 Video-MME long videos at about 8.6K and 15.8K tokens, with subtitles so that whole
  text tiles exist.
- Run direction 1's reuse analysis split by tile type, and the frame-shift agreement.
- Then Video-MME accuracy on about 100 questions for:
  - dense;
  - BLASST 50%;
  - reuse everything;
  - reuse vision-key tiles only.

**Metrics.**
- Recall of kept tiles by tile type.
- % QK^T avoided.
- Video-MME accuracy.

**Cost.**
- About 1 GPU-hour for margins.
- About 2–4 GPU-hours for accuracy (about 400 runs × about 20 s).
- **About 3–4 engineering days**, after teammate D's dense pipeline runs.

**Real kernel needed?** No. This is a measurement and accuracy study.

### Direction 5: reuse BLASST skip decisions across decode steps on a GPU only, with no PIM

**Idea in plain words.** In decode, BLASST still loads every K tile, because it must compute QK^T to
decide; it saves only V loads (AMEND: BLASST skips "0%" of K bytes). If step t reuses step t−1's
decisions, skipped tiles need neither K nor V loaded.
- AMEND does this with an audit on PIM, so on a plain GPU the audit would cost the very K loads it is
  trying to save.
- The GPU-only question: audit a *rotating subset* of predicted skips each step (say 1/8), or audit every
  T steps. How do false skips and bytes saved trade off?
- Also test long reasoning generations of thousands of tokens (Qwen3-8B with thinking on, MATH500).
  AMEND's evaluation is LongBench and RULER, and its persistence trace is 128 steps.

**Gap it fills.**
- AMEND: "the GPU–PIM mapping of Section 4.3 is modeled in this simulator". It has no GPU-only variant,
  no prefill, and no long-generation workload.
- The GPU-side methods that reuse across decode steps all use top-k pages or blocks: LServe, PRR, NOSA,
  GVR.

**Prior-art status: partially done by AMEND** (same idea, PIM, simulated). **Left:** a GPU-only audit
policy and long generations.
- Searches run:
  - "reuse sparse attention block selection across consecutive decoding steps … GPU kernel skip"
  - "\"skip softmax\" OR BLASST decode reuse previous step mask"
  - "LServe reusable page selector"

**Experiment.**
- Record decode-step margins (one query row per head, or one row per GQA group) over 10–20 long
  generations.
- Replay policies offline:
  - keep last step's decisions;
  - EMA (exponential moving average) of margins;
  - a rotating audit.
- Measure recall of kept tiles, K+V bytes avoided, and false-skip rate.
- Then accuracy on about 100 MATH500 problems for 2–3 policies.

**Metrics.**
- K+V bytes avoided.
- False-skip rate.
- MATH500 / RULER accuracy.

**Cost.**
- Decode in a Python-level simulation is slow. Assume 40–60 ms per token × 36 layers' worth of work:
  about 3–4 minutes per 4K-token generation.
- About 1 GPU-hour for margins, plus about 15 GPU-hours for accuracy (100 problems × 3 configurations).
- **About 5–7 engineering days.**
- Per-step margins at 16K: 36 × 32 × 128 fp16 values, about 0.3 MB per step, or about 1.2 GB per
  4K-token sample. Store them only for about 10 samples, on `/work/nvme`.

**Real kernel needed?** Bytes avoided can be counted in simulation, which is what AMEND reports as well.
Real wall-clock time needs a custom decode kernel, which is hard in 5 weeks.

### Direction 6: share BLASST decisions across the query heads of one GQA group

**Idea in plain words.** Llama-3.1-8B has 4 query heads per KV head. Compute QK^T and the BLASST mask for
one "leader" head, or the union over a cheap subset, and reuse it for the other three. That would save up
to 75% of QK^T in that layer.

**Prior-art status: mostly done.**
- Trained, group-shared top-k: NSA, MiniMax Sparse Attention, SPLA.
- Union of masks within a GQA group in chunked prefill: CompactAttention (2605.16839).
- Cross-head reuse with a JS check: SharePrefill.
- Max over the GQA heads for verdicts: AMEND.
- Search run: "GQA group query heads share block sparse mask compute one head reuse others sparse
  attention prefill training-free".

**Left:** only the threshold version, which is a small step. **Recommendation:** fold it into direction 1
as the "GQA union" head-mapping option. It costs no separate experiment, and the offline numbers come free
from the margins.

---

## 4. Considered and set aside

- **Seeding BLASST's running max from the previous layer, or ordering tiles by the previous layer's best
  tiles.**
  - BLASST App. A.4 already tested a better early running max (reverse order, local window first) and
    found "a negligible impact on model accuracy at a given sparsity level" on VT, though better on FWE.
  - CoSA (2607.25291) orders tiles by a proxy so that tiles holding the row max come first.
  - VFA (2604.12798) seeds the max from key-block summaries.
  - Searches for a cross-layer seed found nothing ("appears open"), but the expected gain is small and
    disputed.
  - If anyone wants it: it costs one offline number from the recorded margins, namely % skipped with the
    true final row max vs the running max. It can go in direction 1's analysis as a single figure.
- **Reuse across requests that share a prefix (prefix caching).**
  - Chorus II reuses masks across *image-to-video* requests.
  - For LLMs, no paper was found. But new suffix queries differ from cached ones, so what would be reused
    is unclear; it would have to be key-tile importance, not decisions.
- **Reuse across models** (a small draft model predicting the big model's sparsity): done by STS,
  SpecAttn and MapFlow.

---

## 5. Final ranked table

Ranked by (how open) × (feasible in about 5 weeks with one GH200 and the PyTorch simulation) × (how
interesting). GPU-hours assume the cost unit above and must be re-estimated after the smoke test.

| Rank | Direction | Gap it fills (paper + quote) | Prior-art status | Experiment | Cost | Needs real kernel? |
|---|---|---|---|---|---|---|
| 1 | **Threshold-mask reuse across layers + guard band δ** (what breaks: density mismatch, lost guarantee, length scaling) | All cross-layer methods are top-k; IndexCache names only "MoBA … and NSA" as next targets; LiteAttention (only threshold reuser) "only relies on the slow-evolution assumption" across timesteps; BLASST Fig. 7 "Substantial head-level and layer-level variance" | **Appears open** (closest: LiteAttention on timesteps; Kascade/IndexCache top-k) | Offline recall / lost mass / % QK^T avoided for layer pairs, by length and task, on recorded margins; then RULER with Kascade anchors, δ = 0 vs fitted δ, three head mappings | ~7–9 GPU-h; 3–4 eng-days | No (simulation gives recall, accuracy, tiles avoided); FlexAttention BlockMask optional for timing (+2–3 days) |
| 2 | **Runtime refresh check** (probe tile row; near-miss sentinels; residual signal) vs fixed anchors | Kascade: dev set "biases the technique towards the data in the development set"; TidalDecode: layer choice "is critical"; LongCat: "uniform interleave pattern"; RouteRelay tested only on synthetic "layer-local Gaussian drift" | **Partially done** (RouteRelay: top-k, synthetic, CPU; HEART: timesteps; SharePrefill: heads; AMEND: decode). Left: thresholds, real LLMs, accuracy, signal comparison | Replay policies offline from all-layer margins at equal QK^T work; calibrate on niah, test on vt/fwe/LongBench; accuracy for the best 1–2 | ~4–5 GPU-h; 4–6 eng-days (on top of #1) | No for the claim; fused kernel only for wall-clock |
| 3 | **Certified reuse** (box / ball / low-rank upper bound proves each reused skip satisfies BLASST App. B) | AMEND: "incorrect drops must be bounded, not repaired"; RouteRelay: "does not guarantee perfect recall"; BLASST App. B shows any upper/lower-bound pair preserves the bound | **Appears open** (bounds used only for ranking: Quest; risk scores: RBS; probabilistic decode-only certificate: 2610.24991) | Day-1 kill test: fraction of BLASST-skipped tiles each bound certifies with the true row max; if meaningful, certified reuse in direction 1 | ~0.5 GPU-h kill test, ~3 total; 3–4 eng-days | No (FLOP counts suffice) |
| 4 | **Modality-aware reuse in Qwen2.5-VL** (vision vs text tiles across layers; across frames) | H2Share: "not exhaustively tested it on … multimodal LLMs"; SharePrefill: "extending to multi-modular systems … future work"; MMInference: text indices "fail to generalize to visual ones" | **Appears open** (layers); **partially done** (frames, by MMInference's static Grid pattern) | Direction 1's analysis split by tile type on 20 Video-MME videos; frame-shift agreement; ~100-question Video-MME accuracy | ~3–5 GPU-h; 3–4 eng-days after teammate D's pipeline | No |
| 5 | **GPU-only decode-step reuse of BLASST decisions** (skip K loads too; rotating audit; long reasoning) | AMEND: GPU–PIM "modeled in this simulator", no GPU-only variant, no long generations; BLASST skips 0% of K bytes | **Partially done by AMEND**; top-k versions done (LServe, PRR, NOSA) | Record decode margins on long generations; replay policies; MATH500 accuracy | ~15–20 GPU-h; 5–7 eng-days | Bytes in simulation; real decode kernel for time (hard) |
| 6 | **Cross-head sharing within a GQA group** | (none specific) | **Mostly done** (NSA, MSA, SPLA, CompactAttention, SharePrefill, AMEND) | Fold into direction 1 as the "GQA union" mapping | ~0 extra | No |

**Suggested combination for the final report.** Directions 1 and 2 are one story: measure, then refresh
adaptively. They share all the infrastructure, and direction 1 is a prerequisite for 2, 3 and 4.
- Direction 3's kill test costs one day and should run early.
- Direction 4 is the natural bridge to the team's modality idea (Idea 3).
- Combined compute for 1 + 2 + 3: about 15–20 GPU-hours.

---

## 6. Could not verify

- Per-sample simulation time on GH200 is unmeasured, so every GPU-hour figure scales with it.
- FlexAttention `BlockMask` availability and its block-skipping speed on DeltaAI (torch 2.11, aarch64)
  were not checked.
- Chorus II, PRR, STS, MapFlow, RBS-Attention, 2610.24991 and BlindSight: only abstracts or search
  snippets were seen, not full text.
- LServe's 4-step re-selection window and NOSA's ≥ 0.8 locality come from secondary summaries or snippets.
- Loki (2406.02542, low-rank keys) is cited from memory, not re-checked.
- The model shapes in § 0 are from memory; check each `config.json`.
- The isotropic-key estimate in direction 3 is my own arithmetic, not a measurement.
- Whether LiteAttention's tile test is exactly BLASST's (all-rows vote) was confirmed from its Eq. 7 text
  as quoted by a subagent. I did not re-read the equation myself.

## Sources

- Kascade https://arxiv.org/html/2512.16391 ; IndexCache https://arxiv.org/html/2603.12201 ; TidalDecode https://arxiv.org/html/2410.05076 and the ICLR 2025 camera-ready
- LiteAttention https://arxiv.org/html/2511.11062 ; AMEND https://arxiv.org/html/2609.09823 ; SharePrefill https://arxiv.org/html/2505.19578
- Autonomy-of-Heads / H2Share https://arxiv.org/html/2608.06849 ; LongCat LSA https://arxiv.org/html/2608.01662 ; YOIO https://arxiv.org/html/2606.06467
- RouteRelay https://arxiv.org/html/2609.07306 ; HEART/HASTE https://arxiv.org/html/2605.14513v2
- Chorus II https://arxiv.org/abs/2606.25040 ; PRR https://arxiv.org/abs/2606.30389 ; LServe https://arxiv.org/abs/2502.14866 ; NOSA https://arxiv.org/abs/2510.13602
- STS https://arxiv.org/abs/2605.15508 ; SpecAttn https://arxiv.org/abs/2510.27641 ; MapFlow https://arxiv.org/abs/2610.14688
- RBS-Attention https://arxiv.org/abs/2609.20971 ; Certified Sparse Attention via Perturbed Retrieval https://arxiv.org/abs/2610.24991
- VFA https://arxiv.org/abs/2604.12798 ; CoSA https://arxiv.org/abs/2607.25291 ; BlindSight https://arxiv.org/abs/2507.09071 ; CompactAttention https://arxiv.org/abs/2605.16839
- BLASST https://arxiv.org/html/2512.12087 (v3; App. A.4 and App. B quoted)
