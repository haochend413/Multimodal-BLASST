# One global threshold, a different statistic: what the literature does

Written 2026-11-03 for the CS 498 project (desk research only; no runs, no cluster). Question from a teammate:
instead of one lambda per region, keep ONE lambda but change *what is measured* (the statistic, its reference
point, or a normalization) so the same lambda behaves well on T->T, T->V, V->T and V->V tiles.

**How to read the evidence tags.** *Read* = the arXiv HTML full text was read this session (by me or a helper
agent; the fetch tool returns a summary, so quotes are near-verbatim). *Abstract* = only the abstract or
venue page. *Snippet* = search result only, do not cite numbers. **Inference** = my own reasoning, not in any
paper. Earlier notes (`related_work.md`, `paper_2607.02980.md`, `multimodal_qwen25vl.md`) are not repeated.

---

## 0. Words

Ours (unchanged): **tile** (128 query rows x 128 keys), **skip rule**, **lambda**, **margin** (max over the
128 rows of tile row max minus running max), **% skipped**, **dense**, **calibrated**, **running max**,
regions **T->T / T->V / V->T / V->V**, **weight lost**, **the 50% lambda**, **question->video**, **answer
position**, layouts **A / B / C** (all as in `results/VLM_REGIONS_2.md`).

New terms, one sentence each:
- **Statistic**: the number the rule compares with the threshold. BLASST: the tile row max.
- **Reference point**: what the statistic is measured against. BLASST: the running max.
- **Row mass** Z: the softmax denominator, sum of exp(score) over a row's keys. **LSE** (log-sum-exp) = ln Z.
  **Tile mass**: the same sum over one tile's 128 keys; its log lies between the tile max and tile max + ln 128
  (ln 128 = 4.85).
- **Top-p (cumulative-mass) rule**: keep the fewest tiles whose share of the row mass reaches p (e.g. 0.95).
- **Min-p (relative-to-max) rule**: keep items whose weight is at least a fraction of the largest weight.
  From LLM sampling: min-p keeps tokens with p >= p_base * p_max (2407.01082, ICLR 2025, read). BLASST is a
  min-p rule on tiles; top-p rules are its "nucleus" counterpart.
- **Attention sink**: a token (usually the first) that takes a large share of the softmax weight but adds
  almost nothing to the output because its value vector is near zero.
- **z-score**: (score - row mean) / row standard deviation.

---

## 1. The four knobs of a skip rule, and where BLASST already sits

BLASST skips a tile iff margin < ln lambda, with lambda = a / L. Rewrite it as

    skip  iff  margin + ln L  <  ln a.

So BLASST **already** uses "one threshold on a transformed statistic": the `+ ln L` term is a length
normalization, and it is the reason one calibrated `a` works across context lengths (Inference from BLASST's
lambda = a/L fit; the intuition is that L keys each at weight ~lambda relative to the max add up to ~L*lambda).
The teammate's proposal is to add a region-aware term in the same place, instead of making lambda a table.

Every rule in the literature is a choice on four knobs:

| Knob | BLASST's choice | Alternatives found in the literature (section 2) |
|---|---|---|
| 1. Tile statistic | tile row max | tile mass / pooled energy (FlashPrefill, XAttention, HiLS), pooled score (Sol-Attn, AnchorAttention), output error (SpargeAttn, SALE, vAttention) |
| 2. Reference point | running max (includes the sink once tile 0 is seen) | running LSE (DynSparse), sink+local max or mass (AnchorAttention, SALE), best block in the row (FlashPrefill), estimated final max (VFA), row mean and std (Sol-Attn), row total mass (all top-p methods) |
| 3. Transform | + ln L | per-row solved threshold (entmax / AdaSplash), Gaussian tail (Sol-Attn), quantile (Top-theta) |
| 4. Threshold granularity | one per model and phase | per head (SALE, Top-theta, XAttention option), per layer (SpargeAttn), per modality (MoDES, MoE experts only); one threshold on per-modality-normalized mass (MadaKV, KV cache only) |

---

## 2. Survey: what is thresholded, and does one threshold transfer?

"Transfer" = does the paper show one setting working across heads / layers / lengths / models / modalities.

### 2a. Relative to a max (BLASST's family)

| Method (ID, venue, tag) | Statistic vs reference | Threshold, calibration | Transfer evidence | Key numbers |
|---|---|---|---|---|
| **BLASST** (2512.12087, MLSys 2026, read) | tile row max vs running max; all rows must agree | one lambda = a/L per model and phase, fit on ~1,000 RULER sequences | across lengths via 1/L; Fig. 7 shows large head/layer spread under one lambda | see `related_work.md` |
| **SpargeAttn stage 2** (2502.18137, ICML 2025, read) | same as BLASST, per warp slice | per layer, grid search under an L1 output-error bound | none claimed; "different hyperparameters for each layer and head is necessary" | Llama-3.1 128K: 0.54 sparsity, the lambda stage adds ~2.8 points |
| **VFA / VSA** (2604.12798, arXiv, read) | BLASST rule; running max is *seeded* from key-block summaries; sink and local blocks get exact max updates | lambda swept | none | MMLU comp-sec: lambda 1e-2 -> 52.8% skipped, acc 0.84 (dense 0.81); 1e-1 -> 84.6%, 0.78 |
| **AnchorAttention** (2505.23520, EMNLP 2025 per snippet, read) | pooled score vs "anchor" = max over the initial (sink) block + local window; keep if anchor - score <= theta | one theta = 12 (logit units), all datasets, 4K-128K | "consistent hyperparameter settings" across datasets; cross-model untested | 128K: 89% sparsity, 82.8% recall; 1.44x over FlexPrefill |
| **FlashPrefill** (2603.06199, arXiv, read) | block energy (sum of exp over full queries x **mean-pooled key**), rescaled to a common per-row max and normalized; keep if Score >= alpha * best block in row | one alpha per model (0.18 Llama-3.1-8B, 0.08 Qwen2.5-7B, 0.12 Qwen3-30B), set for ~70% density at 4K; per head applied | density falls with length at fixed alpha (70.4% at 4K -> 5.8% at 256K); argues max-based beats top-p (top-p's cumsum is "sequential", fixed budgets keep low blocks) | **Qwen2.5-VL-7B Video-MME 63.22 vs 63.74 dense** (MInference 61.56, FlexPrefill 62.15, XAttention 61.51); no density or modality breakdown for the VLM. Llama RULER-128K density 4.5% (max) vs 14.0% (top-p), score 75.31 vs 72.83 |
| **AMEND** (2609.09823, arXiv, read) | BLASST margin with ln(a/L) (a = 64), reduced over a GQA group; predicts this decode step's verdict from an EMA of past margins + volatility (mu + z*sigma < -delta) | knobs by Bayesian optimization on 50 examples | decode only | simulated H100: 1.40-3.63x decode at 8K-64K |

### 2b. Relative to the row's mass (LSE or top-p)

| Method | Statistic vs reference | Threshold, calibration | Transfer evidence | Key numbers |
|---|---|---|---|---|
| **DynSparse / cumulative energy filtering** (2606.16317, arXiv, Huawei, read) | tile max vs **running LSE** = running max + ln(running sum): skip iff m~ - (m + ln l) < lambda | "a global threshold"; recommended lambda = ln(P_real) + ln(B_c / L) | Wan 2.2 (video diffusion) only; claims less sparsity spread over layers and timesteps; no sinks, no text | Wan 2.2 14B: 61.42% (BLASST) -> 82.14% skipped, 1.32x -> 1.61x; VBench drop "<5%" claimed, IQ drops ~6% relative |
| **XAttention** (2503.16428, ICML 2025, read) | strided antidiagonal sums per block, softmax over blocks, keep fewest blocks reaching tau | one tau (0.9) by default; optional per-head tau by a DP (x0.9 per step) | same tau on RULER and LongBench; fixed tau for video | Qwen2-VL-7B Video-MME at tau = 0.9: 63.3 vs 63.7 dense (no subs), 69.1 vs 69.2 (subs); per-head tau: RULER 88.47 at 20.97% density vs 84.96 at 26.13% for fixed tau |
| **FlexPrefill** (2502.20766, ICLR 2025 oral, read) | JS distance picks pattern (tau = 0.1); then add blocks until cumulative weight reaches gamma | tau shared by 4 models; gamma 0.9 or 0.95 per model | tau transfers across models; gamma not shown to | Llama-3.1-8B 128K: gamma 0.95 -> 2.43x, RULER 89.18 vs 88.70 |
| **SampleAttention** (2406.15486, MLSys 2025, read) | "cumulative residual attention" kept per query, on sampled query blocks; separate alpha for columns and stripes | per model and length range, small RULER subset | across tasks yes; 128K setting reused for longer; across models "inherently model-specific" | 89.6% sparsity, up to 5.29x TTFT at 1M |
| **Twilight** (2502.02770, NeurIPS 2025, read) | top-p over tokens the base selector kept, weights from INT4 keys | one p per model (0.95 Llama, 0.85 Longchat) | p is "less influenced by the different distributions ... for different heads/layers/queries" than k; not tested across models | decode: up to 15.8x self-attention vs FA2 |
| **Tactic** (2502.12216, read) | selected mass / estimated full-row mass (K-means + fitted tail) >= P | P swept 50-90% | adaptive per head/query; no transfer test | RULER 84.6 at P = 90% vs 86.8 dense (decode) |
| **Double-P** (2602.05191, arXiv, read) | hierarchical top-p over key clusters | (p1, p2) per model | fixed k = 256 "violates" p = 0.95 "in over 20% of heads" | RULER 89.08 vs 89.54 (decode) |
| **SALE** (2505.24179, arXiv, read) | 4-bit score vs **sink+local** max and mass: exp(s - m_sl) / l_sl >= tau | per head, tau halved from 0.008 until L1 output error < theta (0.4 Llama, 2.0 Qwen) | per model | 3.36x at 64K (Llama-3.1-8B) |
| SpargeAttn stage 1, CoSA, ProxyAttn | top-CDF / top-p on a pooled map | per layer / per model | see `related_work.md` | see `related_work.md` |

### 2c. Other statistics

| Method | Statistic | Threshold, calibration | Transfer / numbers |
|---|---|---|---|
| **Sol-Attn** (2607.24027, arXiv, NVIDIA, read) | pooled-Q x pooled-K logit vs **row mean + beta * row std** (a z-score cut) | one shared beta controls a model's sparsity; logits "near-Gaussian within each model" | per-query density "tightly concentrated across models" vs top-p's "broadly varying"; video/image diffusion only. Wan2.1-14B: 85.12% sparsity, 2.02x |
| **Top-theta** (2502.08363, ICANN 2026, read) | score vs a calibrated per-row quantile (+ alpha * std), with compensation for dropped mass | per layer x head x row length | calibrated on ARC-C, works on HumanEval and LongBench: tied to the model, not the data. LLaMA only |
| **EntropyInfer** (2606.09508, arXiv, read) | max row entropy per head per segment -> budget (rigid vs dynamic heads) | fixed constants | LongBench Llama 48.55 vs 48.94 dense |
| **vAttention** (2510.05688, arXiv, read) | sink + local + top-k + random sample sized for an (eps, delta) error **relative to the full output** (row mass estimated from the sample) | grid search per target sparsity | decode; RULER-32K-hard 86.56 vs 88.74 at 10% |
| **AdaSplash** (2502.12082, ICML 2025, read) | alpha-entmax: subtract a per-row tau solved so weights sum to 1 | alpha global; needs training | the per-row solved tau is the cleanest "self-normalizing" threshold, but not plug-in |
| **SeerAttention** (2410.13276, read) | learned gate, softmax per row, > threshold | one threshold per benchmark (5e-4 RULER, 2e-3 LongBench) | "identical threshold across all tests in RULER"; trained gate |

**Warning from the field (read):** The Sparse Frontier (2504.17768, Sec. 4.2): in prefill, threshold-based
FlexPrefill "matches or underperforms" uniform budgets, which the authors put down to the "attention sink
phenomenon": "threshold-based selection captures high-attention tokens while missing information in the
distribution's long tail." That is our T->V story in text-only form.

### 2d. Multimodal / video-LLM prefill

| Method (ID, venue, tag) | Stage | What is thresholded | How modality is handled | Key numbers |
|---|---|---|---|---|
| **MMInference** (2504.16083, ICML 2025, read) | prefill, training-free | per-head pattern (Grid / A-shape / Vertical-Slash), index from the last 64 queries; budget matched to an A-shape of 1K sink + 4K local | **per region**: rows permuted by modality; "Q-Boundary" heads build a separate index per query modality; "2D-Boundary" heads one block per (query modality, key modality) pair | Qwen2.5-VL-7B, 256 frames (33,950 tokens), 41.3% FLOPs: Video-MME no-subs 64.7 -> 65.1, with subs 71.3 -> 70.9. VLMs need the top 5.78% of weights for 95% recall vs 1.79% for text-only LLMs; "the sparse index achieves high recall for textual regions but fails to generalize to visual ones" |
| **XAttention** (2503.16428, ICML 2025, read) | prefill | one tau on cumulative block mass | none | Qwen2-VL-7B, tau = 0.9: Video-MME 63.3 vs 63.7 (no subs), 69.1 vs 69.2 (subs); sparsity not reported |
| **UniSparse** (2512.14082, arXiv, read) | prefill | Q and K average-pooled by 8, softmax, keep blocks up to cumulative mass P | none; calls its proxy "modality-agnostic" | Qwen2.5-VL-7B Video-MME (dense 65.3 / 69.6 no-subs / subs): P = 0.9 -> 57.7% sparsity, 64.6; P = 0.95 -> 45.9%, 65.0. XAttention in their table: tau = 0.9 -> 60.2%, 63.0 |
| **RRAttention** (2602.05853, arXiv, read) | prefill | top-tau cumulative mass, round-robin query sampling across heads | none | Qwen2-VL-7B Video-MME (dense 63.9): tau = 0.9 -> 47.0% sparsity, 64.0 |
| **FlashPrefill** (2603.06199) | prefill | alpha x best block (2a) | none | Qwen2.5-VL-7B Video-MME 63.22 vs 63.74 |
| **BlindSight** (2507.09071, arXiv, read) | prefill, static template mask | per-head category chosen by normalized MSE < 0.1 on MMIU | mask from vision start/end tokens; "text tokens continue to attend to all the previous tokens" (text rows dense) | 1.8-3.2x attention at 36K-300K; -0.78% average accuracy |
| **VideoNSA** (2510.02295, ICLR 2026, read) | prefill, **trained** (216K videos) | NSA top-k blocks (1 block = 1 frame) + window + learned gates | video queries sparse, **text queries dense** | "3.6% of the full attention budget" |
| **OmniSparse** (2511.12201, AAAI 2026, partial) | trained | vision queries "lazy/active" by comparison with a sink key and a pooled visual key | only vision rows/keys sparsified | 2.7x prefill |
| **ZipVL** (2410.08584, read) | prefill + KV | per layer, fewest tokens whose column-summed mass >= tau (0.975) | none | LongVA-7B Video-MME, 128 frames: 82.3% attention FLOPs cut, 52.6 = dense 52.6 |
| **VL-Cache** (2410.23317, ICLR 2025, read) | KV cache | layer sparsity = share of weights **< 1% of the row max** (i.e. score - row max < ln 0.01 = -4.6, BLASST's form) sets layer budgets; tokens ranked by post-vision text queries only | ranking by text queries; "visual tokens attend to other visual tokens rather uniformly while language tokens only pay attention to a few visual tokens with high concentration" | 10% KV ~ full cache |
| **MadaKV** (2506.15724, ACL 2025, read) | KV cache | per head: keep fewest vision tokens whose mass >= **theta x (that head's total mass on vision tokens)**, same theta for text | **one theta on per-modality-normalized mass**; budget split by each head's modality shares | LLaVA-1.5, Qwen2.5-VL-7B, MileBench: 80-95% KV cut, 1.3-1.5x decode. "textual tokens display sharp attention concentration, whereas visual tokens exhibit diffuse attention allocation" |
| **MEDA** (2502.17599, NAACL 2025, read) | KV cache | layer entropy of T->V and V->T attention | T->V computed as a **softmax over vision keys only** (per-region normalization), used to set layer budgets | 72% KV cut, 2.82x decode |
| **MoDES** (2511.15690, CVPR 2026, read) | MoE expert skipping | router prob x per-layer factor | **separate text and vision thresholds** | see Q1 |

Video diffusion (different setting; only transferable ideas): Sparse VideoGen (2502.01776) always keeps
text-prompt and first-frame keys dense; Sparse VideoGen2 (2505.18875) clusters Q and K and runs top-p on
centroid scores; Sol-Attn's z-score and DynSparse's running LSE (above) are both diffusion-only so far.

Two patterns: (1) one-threshold prefill methods (XAttention, UniSparse, RRAttention, FlashPrefill) reach
46-60% sparsity on Video-MME within about 1 point of dense, but report no per-region numbers, and their
sparsity is dominated by video query rows, which are most of the tiles. (2) Methods that treat modality
explicitly either split by region (MMInference) or keep text query rows dense (BlindSight, VideoNSA,
OmniSparse). Our "never skip question->video" setting (64 / 90) is the latter by design.

---

## 3. The text-vs-vision score gap, outside sparse attention

- **Norm gap, LLaVA-1.5-7B** (2503.17349, arXiv, read): at the LLM input, vision tokens have L2 norms 10^1-10^3
  vs 0.3-1 for text, persisting to ~layer 15. Fix: RMS-normalize vision embeddings to text-like norms, with
  **retraining**. "The Unseen Bias" (ICLR 2026, abstract): same gap in "mainstream MLLMs", one LayerNorm
  after the projector, LLaVA-1.5. Neither measures Qwen2.5-VL or sparsity.
- **Key-space gap** (2510.26721, arXiv, read): image and text keys form separate distributions in LLaVA-1.5-7B
  and **Qwen2.5-VL-7B** (MMD 0.408 cross-modal vs ~0.01-0.05 within a modality; for Qwen the gap nearly
  vanishes in late layers, MMD ~0.01). It does **not** measure q.k scores, so it does not show which way the
  logits shift.
- **Attention share** (AdaptVis, 2503.01773, read): in LLaVA-1.5, image tokens are ~90% of the input but get
  ~10% of the attention. AdaptVis multiplies the last token's logits over image tokens by alpha in
  {0.5 ... 2.0}, chosen by output confidence (training-free; for spatial reasoning, not speed). PAI
  (2407.21771, ECCV 2024, abstract) similarly amplifies image-token attention against hallucination.
- **Shape of text->vision attention** (FasterVLM, 2412.01818, read, LLaVA-1.5-7B): "attention shift" (text
  attends more to *later* visual tokens) and "attention dispersion" (flatter: top 20% of visual tokens get
  >60% of last-token attention vs ~80% for the vision encoder's [CLS]).
- **Sinks carry weight but no value**: VATP (2406.12335, EMNLP 2024, read) scores tokens by attention x
  ||v||_1 and finds sink value norms "close to 0" in most heads of layers 3-31 (LLaMA2-7B); Gu et al.
  (2410.10781, ICLR 2025, abstract) say sinks act "like key biases" and do not contribute to the value
  computation; visual sinks exist too (2503.03321, abstract: removing them "does not impact" performance).
- **When text reads the image** (2411.18620, CVPR 2025, abstract): image -> question flow in lower layers
  (general) and middle layers (object-specific), question -> last position in higher layers.

---

## 4. Answers

### Q1. Has anyone used ONE threshold on a normalized statistic specifically to remove a text-vs-vision gap?

**Not in sparse attention or prefill, as far as I found** (about 20 targeted searches plus the papers
above). The closest precedent is in **KV-cache eviction**:
- **MadaKV** (2506.15724, ACL 2025, read) uses one theta on a per-modality-normalized statistic: in each head,
  keep the fewest vision tokens holding theta of *that head's vision mass*, and the same for text. Because
  the mass is renormalized within each modality, a head that puts 5% of its weight on vision still keeps
  theta of that 5%. Evaluated on Qwen2.5-VL-7B (decode, MileBench). This is the nearest thing to "one
  threshold that behaves the same on text and vision".
- **MEDA** (2502.17599, NAACL 2025, read) computes T->V attention as a softmax over vision keys only, a
  per-region normalization, but only to set per-layer KV budgets.

Other near misses:
- **MoDES** (2511.15690, CVPR 2026, read) does the *opposite* for MoE expert skipping: separate text and
  vision thresholds ("dual-modality thresholding"), because the two token types respond differently. Its
  ablation is exactly our planned experiment: Kimi-VL at 83% of experts skipped, one threshold vs two:
  ChartQA 76.74 vs 82.94, MMBench 65.48 vs 79.42. Not attention.
- **MMInference** handles modality with per-head patterns, a separate index per (query modality, key
  modality) region, and permutation; no thresholds (2d).
- **XAttention** (tau on cumulative mass) and **FlashPrefill** (alpha on max-normalized block energy) run
  one threshold on Video-MME with near-dense accuracy, but report no per-modality sparsity and do not claim
  their normalization fixes a modality gap.
- Outside sparsity, the gap is "fixed" by renormalizing vision embeddings (retrained) or by rescaling
  image-token logits for accuracy (AdaptVis, PAI). None of these is a skip rule.

### Q2. Which statistics would make T->V tiles look less skippable than BLASST's rule?

Our setting: a text row's running max is set by the sink and the earlier transcript long before the video
tiles are reached (tile 0 is always visited first), so "running max vs final max" barely matters for T->V.
Rows run roughly from "makes T->V more skippable" to "less skippable" (all predicted effects are
**Inference**; the Precedent column names where the statistic itself comes from):

| Candidate statistic (one lambda) | Precedent | Effect on T->V vs BLASST | Cost in our kernel |
|---|---|---|---|
| tile max - running LSE | DynSparse | **more** skippable: the LSE includes the sink's large mass, so the reference only rises | free |
| tile max - (sink + local) max or mass | AnchorAttention, SALE, VFA's sink-first order | same or **more** skippable | free |
| top-p on row mass | XAttention, FlexPrefill, Twilight | probably **no better**: text rows put only 4.7-6.5% of weight on video, so a 5% drop budget can be filled by video tiles | needs a pass over the row |
| tile mass instead of tile max (vs best block) | FlashPrefill, XAttention, HiLS's max-vs-mass argument | **less** skippable if video tiles are flatter than text tiles (FasterVLM's "dispersion"); a flat tile has up to ln 128 = 4.85 more log-mass than its max shows, the same size as the median T->V margin of -4.85 | exp of the tile (the cost we skip) unless estimated |
| **tile log-mass lower bound**: max(tile row max, q . mean key / sqrt(d) + ln 128) - running max | none found; q . mean key / sqrt(d) is exactly the row's mean score over the tile (mean-pooling, as in MInference / HiLS's zeroth order), and log tile mass >= mean + ln 128 by Jensen's inequality | **less** skippable for flat tiles, unchanged for peaky ones; since it is >= the tile max, at the same lambda it only skips tiles BLASST also skips | one dot product per row per tile with a precomputed mean key (~1/128 of the tile's QK^T) |
| **sink-excluded reference**: tile max - running max over non-sink keys | none found (VATP, Gu et al. justify it: sinks add ~no value) | **less** skippable in sink-heavy rows (text rows) by (sink max - non-sink max); whether it helps T->V more than T->T must be measured | a second running max |
| per-region reference (running max, or mass, over keys of the same region) | not in a skip rule; MadaKV and MEDA normalize per modality in KV-cache work; MMInference isolates regions by permutation | **much less** skippable (the best video tile gets margin 0); in effect an automatic per-region lambda | one running max per region |
| value-weighted: + ln(max value norm in tile) | VATP (KV cache) | unknown: depends on video vs text value norms in Qwen2.5-VL | precomputed per key tile |
| z-score of the tile max within the row | Sol-Attn (diffusion, pooled proxy) | unclear; needs the spread of text rows' scores | row mean and std (proxy) |

**Is text->video attention flat or peaky?** This decides whether the "tile mass" rows help, and the papers
disagree: FasterVLM (LLaVA-1.5) and MadaKV (LLaVA-1.5, Qwen2.5-VL) say attention over visual tokens is
diffuse; VL-Cache says language tokens attend to "a few visual tokens with high concentration". Our own
numbers lean toward spread-out: at the 50% lambda, skipping 84% of T->V tiles removes 52-57% of the
video weight, so the skipped tiles hold about half of it between them (Inference). Measuring the gap
between tile log-mass and tile max per region on a dense pass settles it.

### Q3. What is still open (a fair course-project contribution)?

1. **A per-region comparison of statistics.** No paper reports % skipped per region (T->T / T->V / V->T /
   V->V) for any skip rule. A table of 4-5 statistics from Q2 (BLASST, running LSE, sink-excluded reference,
   tile log-mass lower bound, per-region reference), each at the lambda that skips 50% overall, with weight
   lost on question rows and the answer position, would be new. It can be computed offline from dense passes,
   like the existing margin and weight-lost probe.
2. **Moving per-modality normalization from KV-cache eviction into an exact prefill skip rule.** MadaKV's
   idea (one threshold, mass renormalized per modality) has not been tried on tiles in a FlashAttention-style
   kernel; the per-region reference in Q2 is its natural tile version.
   **Two statistics with no precedent found**: the sink-excluded reference and the mean-key log-mass lower
   bound. Both keep a single lambda and are cheap inside a BLASST-style kernel. At equal lambda the second
   skips only tiles BLASST also skips; at matched 50% skipped it needs a larger lambda, so peaky tiles
   (likely T->T) get skipped more. That trade is the whole point and must be shown to be worth it.
3. **One normalized lambda vs per-region lambdas at matched % skipped**, MoDES-style, on attention tiles.
4. **An honest risk**: the published reference changes (DynSparse, AnchorAttention, SALE, VFA) all push
   toward *more* skipping of low-mass regions, and The Sparse Frontier says thresholds miss long tails. A
   negative result is plausible. Our accuracy pilot (65 / 61 / 64 of 90) cannot tell rules apart, so weight
   lost should be the main metric and accuracy a check on a larger set.

---

## 5. Not verified / caveats

- Venues marked "per snippet" (AnchorAttention EMNLP 2025, XAttention ICML 2025, Twilight NeurIPS 2025) were
  not opened on the proceedings site. Tactic's venue is unconfirmed. HASTE (2605.14513, per-head top-p by
  error-guided calibration) was seen in a search snippet only.
- Read only in part (method or abstract): OmniSparse, LOOK-M (2406.18139), SparseMM (2506.05344), PSA
  (2512.04025, cumulative mass picks among pooled KV levels; Qwen2.5-VL numbers not extracted).
- PAI's formula, "The Unseen Bias" (models, arXiv ID, whether it retrains) and Gu et al.'s details: abstract only.
- DynSparse: its abstract and tables disagree (speedup ratio stated as 1.18x, table gives 1.61 / 1.32 =
  1.22; "<5%" VBench drop vs ~6-7% on some metrics). Row aggregation over a tile is not stated.
- FlashPrefill: intro says up to 7.22x TTFT, its Table 7 maximum is 5.02x.
- The predictions in Q2 are untested. In particular, flatness of T->V tiles in Qwen2.5-VL is inferred from
  LLaVA results (FasterVLM) and from our weight-lost numbers, not measured.
