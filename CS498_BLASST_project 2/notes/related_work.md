# Related work: training-free sparse attention and prior work overlapping Ideas 1–3

Compiled 2026-10-31 for the CS 498 ML Systems project (BLASST reproduction + three extensions).
Sources: arXiv abstract/HTML pages, the TensorRT-LLM docs, the NVIDIA developer blog, the PyTorch blog, and web search. Every number below was read from one of those pages; anything I could not read directly is in the "Unverified" list at the end. Where a paper's own text is internally inconsistent I say so.

Vocabulary used throughout (matches our project): a **tile** is one Br x Bc block of the QK^T score matrix; the **skip rule** is BLASST's test "tile row-max minus running row-max < ln λ"; a **proxy** or **score** is any cheap estimate of a tile's importance computed *instead of* the full tile; a **mask** is the binary keep/skip decision per tile.

---

## 0. The baseline we reproduce: BLASST (arXiv 2512.12087, MLSys 2026)

- Authors: Jiayi Yuan, Cameron Shinn, Kai Xu, ... Song Han, Timmy Liu, Huizi Mao (22 authors; NVIDIA-led). v1 12 Dec 2025, v3 28 Apr 2026. MLSys 2026 (slides at mlsys.org).
- Skip rule (Algorithm 1): after computing S = QK^T for a tile, compute the tile's local row max m~ and compare to the running max m; skip when m~ − m < ln λ, i.e. exp(m~ − m) < λ. The kernel takes a warp-level VOTE so a tile is skipped only if every row handled by the warp agrees. The paper does **not** state a numeric tile size (B_c is left symbolic), so our "128x128" is an implementation choice, not a paper fact.
- What is skipped: exp, row-sum, the V load from HBM, and the PV GEMM (BMM2). What is **always** computed: QK^T (BMM1) for every tile. Verbatim: the kernel skips softmax and BMM2 "even as all QK^T (BMM1) operations are computed."
- Thresholds: one scalar per model for prefill and one for decode. Calibration fits λ·L = α·exp(β·s) over ~1,000 RULER sequences (4K–64K) and uses λ = α·exp(β·S)/L at inference (S = target sparsity, L = context length). Optimal λ is inversely proportional to L (example: ~1e-4 at 8K vs ~1e-5 at 64K for 75% sparsity). Table 12 gives a ≈ 900–1,100 for prefill and ≈ 4.6–11.7 for decode at 50% target sparsity.
- Per-layer / per-head: **no** per-layer or per-head thresholds. Figure 7 (Llama-8B, 8K NIAH, λ = 0.03) shows "substantial head-level and layer-level variance" in achieved sparsity under the single threshold; the caption says this "motivates adaptive thresholding strategies." No numeric ranges are given in the text. This is direct motivation for our Idea 3.
- Hardware/results: B200 and H200 vs FlashAttention-3 BF16. Prefill 1.52x at 71.9% sparsity (B200, 64K); decode 1.48x at 73.2% sparsity (B200, batch 148, 32K). At 0% sparsity: 1.00x prefill, 0.98x/0.96x decode (B200/H200). Prefill speedup "scale[s] nearly linearly with sparsity in the compute-bound regime."
- Accuracy: Llama-3.1-8B prefill at ~50% sparsity: RULER avg 92.87 vs 93.21 dense; LongBench 31.8 vs 31.4. Qwen3-8B decode ~50%: RULER-32K 91.55 vs 91.90; MATH500 96.23 vs 95.87.
- Combinations (Table 7, Qwen3-8B, RULER-16K / LongBench-16K): XAttention prefill + BLASST decode 92.89 / 28.8 vs dense 93.22 / 29.4; BLASST prefill + RocketKV decode 92.60 / 29.4.
- Compared against MInference, FlexPrefill, XAttention; the paper claims "the effectiveness of our threshold-based pruning over proxy-based importance estimation." SpargeAttention is discussed as using "a separate prediction step."
- Multimodal: not discussed. Code: integrated into TensorRT-LLM ("Skip Softmax Attention") and FlashInfer (per paper); artifact repo cameronshinn/blasst-ae-mlsys26.
- TensorRT-LLM docs (blog16) state the QK^T floor explicitly: "Due to the design that BMM1 (Q·K^T) in the attention kernel is not skipped, the performance gain is capped to 1.8x at kernel level." Supported on Hopper (fmha_v2 prefill, XQA decode) and Blackwell (trtllm-gen). Threshold = threshold_scale_factor / context_length, set separately for prefill and decode; ModelOpt calibration. End-to-end on Qwen3-30B-A3B-Instruct-2507, LongBench V2 (avg ISL 130K), sparsity 0.9: TTFT 16487 → 12508 ms on H200 (~24%), 6991 → 5277 ms on B200 (~25%); LongBench V1 accuracy 47.79 (0.0) → 46.87 (0.5) → 45.56 (0.7) → 39.84 (0.9).
- NVIDIA developer blog (16 Dec 2025): sparsity "beyond 60%" often causes sharp accuracy drops; ~50% is the stated safe zone. Llama 3.3 70B on GB200 NVL72: "estimated" 1.4x end-to-end prefill at 128K, "projected" 1.36x decode.

---

## 1. Survey table

Columns: how tiles/blocks are scored | avoids full QK^T for skipped tiles? | per-layer/head adaptive? | reuses across layers? | multimodal? | open code + GPU | key reported result.

| Method (arXiv, venue) | Year | How tiles/blocks are scored | Avoids full QK^T? | Per-layer/head adaptive? | Reuses across layers? | Multimodal? | Code + GPU | Key reported result |
|---|---|---|---|---|---|---|---|---|
| **BLASST** (2512.12087, MLSys 2026) | 2025 | Exact tile row-max vs running max, threshold ln λ, inside FlashAttention | **No** (BMM1 always computed; skips exp/V-load/PV) | No: one λ per model (prefill, decode); Fig. 7 shows large head/layer variance under one λ | No | No | TRT-LLM, FlashInfer, artifact repo; B200/H200 | 1.52x prefill @71.9% sparsity, 1.48x decode @73.2%; RULER 92.87 vs 93.21 (Llama-3.1-8B, ~50%) |
| **Block Sparse Flash Attention** (2512.07011) | 2025 | Exact QK^T, per-block max vs stored threshold; diagonal always kept | No (computes all scores) | **Yes**: threshold tensor indexed by (layer, head, position, sparsity level), calibrated to a fixed top-k per query | No | No | github Danielohayon/Block-Sparse-Flash-Attention; A100, RTX A6000, H100 | attention up to 1.38x, TTFT up to 1.13–1.16x (Llama-3.1-8B, 64K); "skipped PV ... ≈50% of attention FLOPs and 50% of HBM bandwidth" |
| **MInference 1.0** (2407.02490, NeurIPS 2024 spotlight) | 2024 | A-shape static; Vertical-Slash: last 64 queries x all K, top-k columns/diagonals; Block-Sparse: **mean-pool Q and K in blocks of 64**, pooled QK^T, top-k blocks | Yes for skipped blocks (proxy is pooled) | Pattern per head chosen offline (~15 min on A100); budgets fixed | No | No (MMInference extends) | aka.ms/MInference; A100 | 10x prefill at 1M (30 min → 3 min); index build 5–20% of latency (block-sparse ~25%) |
| **SpargeAttention** (2502.18137, ICML 2025) | 2025 | Stage 1: mean-pool each Q/K block to one token, keep only "self-similar" blocks (cosine-similarity test, threshold θ), pooled softmax, top-CDF τ per row. Stage 2: in-kernel warp-level skip of PV when max(m_local − m_ij) is below λ (BLASST-like) | Yes for Stage-1 skips; Stage 2 still computes QK^T | **Yes**: τ, θ, λ grid-searched **per attention layer** against an L1 error bound; concludes "setting different hyperparameters for each layer and head is necessary" | No | Yes: Llama-3.1, CogVideoX, Mochi, Flux, SD3.5 (text + video + image gen) | github thu-ml/SpargeAttn (SageAttention-based CUDA); RTX 4090, L40 | Mochi 1897 s → 1037 s (1.83x); Llama-3.1 128K 52 s → 29.98 s; sparsity grows with length (6.8% @8K → 54% @128K at fixed error) |
| **XAttention** (2503.16428, ICML 2025) | 2025 | **Antidiagonal**: strided rows Q[i::S] and columns K[i::S] within each block, softmax of their product, sum = block score; keep minimal block set whose cumulative score ≥ τ (0.9) | Yes (scoring cost ~1/S of full) | Optional per-head "minimum threshold prediction" DP (thresholds reduced 10%/step); not mandatory | No | Yes: Qwen2-VL-7B (Video-MME), HunyuanVideo (VBench) | github mit-han-lab/x-attention; GPU not named in paper | up to 13.5x attention at 256K vs FlashInfer FA; RULER avg 88.47 (S=8) vs 87.52 full; pattern selection 24.9x / 5.9x faster than MInference / FlexPrefill |
| **FlexPrefill** (2502.20766, ICLR 2025 oral) | 2025 | Last `block_size` (128) queries as representative set; JS-divergence (τ=0.1) picks query-aware vs vertical-slash head; cumulative-attention γ (0.9/0.95) sets per-head budget | Yes (representative queries only) | **Yes**: per-head, per-input sparsity ratio; "large differences in the sparsity rates required for different attention heads" (App. I) | No | No | github bytedance/FlexPrefill (Triton); A100 | 3.49x attention at 128K (γ=0.9, Llama-3.1) vs 3.05x MInference; RULER avg 89.18 vs 88.70 full |
| **SeerAttention** (2410.13276) | 2024 | **Learned** gate: pool Q and K along sequence, learnable linear layers, product → block gate | Yes | Learned per head | No | No | github microsoft/SeerAttention | Trains only gate params by self-distillation; (training-based, not a direct competitor) |
| **CoSA** (2607.25291) | 2026 | **Stage 1 KAP: stride-downsample Q and K** (stride s, value not given), softmax, **MaxPool over keys per 128-token block**, sum over sampled query rows → ordered block list + "have-rowmax" flag; top-p (0.95 Qwen; calibrated table for Llama). **Stage 2 OSK**: in-kernel BLASST/SpargeAttn-style skip when local rowmax − running rowmax < ln(Δ/N), visiting HRM blocks first | Yes for Stage-1 skips; Stage 2 computes QK^T | Single Δ per model (2000 Qwen, 200 Llama); p per model | No | No | github Tencent/AngelSlim; **H20**, vs FlashAttention-2 | 4.93x attention, 2.53x TTFT at 128K; RULER Qwen3-8B 88.65 @22% budget vs 89.20 dense |
| **ProxyAttn** (2509.24745, ICLR 2026) | 2025 | **Pool Q and K across heads** in a group (mean), keep every 4th token (stride 4), softmax, max-pool to blocks; per-head dynamic budget from last-block queries + cumulative γ | Yes (g/(n·stride²) of full; "roughly one percent") | **Yes**: per-head dynamic budget; Fig. 2(c): "some heads are very sparse ... while others are dense" | No | No | github wyxstriker/ProxyAttn; H800 | 10.3x attention at 256K; 2.4x TTFT at RULER-128K; Llama-70B 128K TTFT 91.84 s → 44.23 s |
| **SALE** (2505.24179) | 2025 | **4-bit quantized QK^T** estimates fine-grained scores; "Relative Attention Score"; then full-precision block-sparse attention | Partly (low-bit QK^T computed, not BF16) | Not stated on abstract page | No | No | Custom CUDA kernel; GPU not stated | ≥3.36x on Llama-3.1-8B beyond 64K; estimation overhead ~11% of full attention |
| **MMInference** (2504.16083, ICML 2025) | 2025 | Per-head offline pattern search over intra-/inter-modality patterns (Grid, Q-Boundary, 2D-Boundary); Grid stride/phase found online from last 64 queries; modality-aware permutation inside kernel | Yes (last-q sampling) | Pattern per head (offline, ~15 min on A100); no per-modality thresholds | No | **Yes**: LongVILA, LLaVA-Video, VideoChat-Flash, Qwen2.5-VL-7B | aka.ms/MMInference; A100 | 8.3x prefill at 1M vs FA2; Qwen2.5-VL avg 59.4 vs 59.5 at 41.3% FLOPs. Finds "the sparse index achieves high recall for textual regions but fails to generalize to visual ones"; VLMs need top 5.78% of weights for 95% recall vs 1.79% for text LLMs |
| **Sparse VideoGen** (2502.01776, ICML 2025) | 2025 | Online profiling on **1% of query rows**: compute full, spatial-mask, temporal-mask attention, pick lower MSE per head | Yes (1% rows) | Per head and per denoising step; "a certain head can be a spatial head for one prompt while being a temporal head given another" | No | Video DiT (CogVideoX-v1.5, HunyuanVideo, Wan 2.1) | github svg-project/Sparse-VideoGen; H100 | 2.28x CogVideoX T2V, 2.33x HunyuanVideo (FP8); profiling overhead ~3% |
| **AdaSpa** (2502.21079, ICCV 2025) | 2025 | Blockified "dynamic pattern" + online precise search; "Fused LSE-Cached Search" | Partly | Head-adaptive; patterns/LSE vary with layers and heads | **Across denoising steps** (not layers): pattern and LSE "remain invariant across denoising steps" | Video DiT (HunyuanVideo) | Code not linked on abstract | 1.59x–2.04x (ICCV version per search snippet) |
| **DraftAttention** (2505.14708) | 2025 | Low-resolution "draft" attention map from downsampled (pooled) Q/K over frames; reorder Q/K/V to make sparsity block-structured | Yes | Not stated | No | Video DiT | github shawnricecake/draft-attention | up to 1.75x end-to-end |
| **TidalDecode** (2410.05076, ICLR 2025) | 2024 | Full attention in first 2 layers + 2 "token selection" layers (layer after full + one middle layer, e.g. L13 for Llama-3.1-8B); top-k token indices by q·K | n/a (decode) | No | **Yes: token-index set reused by all following layers** (decode) | No | github DerrickYLJ/TidalDecode; A100 | decode latency up to 2.1x; attention up to 5.56x vs full; recall <20% without mid-layer re-selection, ~40% with |
| **Kascade** (2512.16391) | 2025 | Anchor layers compute exact post-softmax weights, pool per 128-query tile, top-k keys per key-head; reuse layers map each head to its most similar anchor head | n/a for anchors; reuse layers skip scoring | Head-aware remap ("critical for high accuracy") | **Yes: top-k key indices reused across layers**, prefill and decode; anchors chosen by DP; Llama-3.1-8B anchors [0,2,8,13,14], Qwen3-8B [0,2,7,14,23] | No | github microsoft/kascade ("will be available"); **H100 vs FA3** | 2.2x prefill attention, 4.1x decode attention; adjacent-layer similarity "close to 1" (most pairs > 0.98, Fig. 3); LongBench 45.02 vs 45.92 dense (Llama, 10% top-k) |
| **IndexCache** (2603.12201) | 2026 | DSA lightning-indexer top-k indices; "Full" layers index, "Shared" layers reuse nearest Full layer | n/a (indexer) | No | **Yes: index reuse across layers**; training-free greedy search or training-aware distillation | No | No repo linked; H100 (SGLang) | 1.82x prefill at 200K, 1.48x decode; adjacent-layer index overlap 0.7–1.0, early-vs-late ≤ 0.4; uniform interleaving degrades quality |
| **SharePrefill** (2505.19578) | 2025 | Pivotal head computes block-average QK, softmax, cumulative γ=0.9 mask; other heads reuse if JS distance < 0.2 and sparsity distance < 0.3, else FlexPrefill vertical-slash | Yes for non-pivotal heads | Clusters **heads across layers** (autoencoder + hierarchical clustering, offline) | **Yes in effect**: cluster may span layers, but the paper frames sharing as head-level | No | "code upon publication"; A100 | 128K latency 16.92 s; InfiniteBench Llama 39.05 vs 38.16 FA |
| **LessIsMore** (2508.07101, ICML 2026) | 2025 | Cross-head unified token selection + recency window (decode, reasoning) | n/a | Unified across heads | Token set "can be reused across layers" (per abstract) | No | — | AIME-24 Qwen3-8B 2K budget 73.75% vs TidalDecode 53.33%; up to 1.6x decode |
| **LiSA** (2408.01890, TACL) | 2024 | n/a (replaces attention in deep layers with aligned previous-layer attention + low-rank diff) | n/a | n/a | **Yes, attention weights** (not masks); needs training of ~0.5–1.6% new params on ~4.2B tokens | No | github takagi97/lisa | JS divergence of adjacent layers "below the 0.05 threshold" (LLaMA3-8B; layers 1–2, 11–14 differ); 53–84% of layers shared |
| **CLA** (2405.12981) | 2024 | n/a | n/a | n/a | **KV heads shared between adjacent layers** (trained from scratch) | No | — | 2x KV cache beyond MQA, 1B/3B models |
| **MiniCache** (2405.14366) | 2024 | n/a | n/a | n/a | **KV states merged across adjacent middle-to-deep layers** (training-free) | No | — | up to 5.02x compression (4-bit, LLaMA-2-7B) |
| **KVSharer** (2410.18517) | 2024 | n/a | n/a | n/a | **KV cache shared across *dissimilar* layers** (plug-and-play) | No | — | ~30% KV compute cut, ≥1.3x generation |
| **YOIO** (2606.06467) | 2026 | One indexer computes token top-k; index + KV shared across layers (YOCO-style) | n/a | n/a | Yes (architectural, trained) | No | — | up to 7.6x decode at 128K |
| **HySparse2** (2609.26368) | 2026 | Self-decoder/cross-decoder KV bridging + token-level sparsity | n/a | n/a | Yes (architectural) | No | — | 80B-A3B MoE results, no numbers on abstract |
| **DuoAttention** (2410.10819, ICLR 2025) | 2024 | Retrieval vs streaming heads identified by optimization on synthetic data | n/a | **Per head** (binary: full KV vs sink+recent) | No | No | github mit-han-lab/duo-attention; A100 | prefill up to 1.73x (MHA)/1.63x (GQA); memory 2.55x |
| **RazorAttention** (2407.15891) | 2024 | Retrieval heads keep full cache; others drop distant tokens + compensation token | n/a | Per head | No | No | — | >70% KV reduction |
| **HeadKV** (2410.19258, ICLR 2025) | 2024 | Head-level KV budget by retrieval-reasoning score | n/a | Per head | No | No | github FYYFU/HeadKV | 1.5% KV keeps ~97% QA performance |
| **Ada-KV** (2407.11550, NeurIPS 2025) | 2024 | Top-B over concatenated head scores per layer → per-head budgets | n/a | Per head; Fig. 1: "most heads retain nearly all attention weights with a small cache (e.g., top 5%)" while "dispersed heads require larger cache proportions" | No | No | github FFY0/AdaKV | Retained attention 2.26 → 2.48 vs uniform |
| **PyramidKV** (2406.02069) | 2024 | More KV budget to lower layers, less to upper ("pyramidal information funneling") | n/a | **Per layer** | No | No | — | 12% KV matches full on LongBench |
| **Tactic** (2502.12216) | 2025 | Cumulative-attention target; clustering + distribution fitting to estimate | Partly (decode) | Per head/layer/context adaptive budget | No | No | — | 7.29x decode attention |
| **Twilight** (2502.02770, NeurIPS 2025 spotlight) | 2025 | Hierarchical top-p pruning on top of existing sparse methods | n/a | Adaptive budget | No | No | — | 15.4x self-attention, 3.9x per-token latency (decode) |
| **Quest** (2406.10774, ICML 2024) | 2024 | Per-page min/max key bounds vs query → top-K pages (decode) | Yes (bounds only) | No | No | No | github mit-han-lab/Quest | 2.23x self-attention, 7.03x latency |
| **SparQ** (2312.04985) | 2023 | Top-r largest-magnitude query components approximate scores; fetch full keys only for top tokens (decode) | Yes (head-dim downsampling) | No | No | No | — | up to 8x attention data-transfer savings |
| **MoBA** (2502.13189) | 2025 | Mean-pooled key per block, gating top-k (trained; Kimi) | Yes | No | No | No | github MoonshotAI/MoBA | deployed in Kimi |
| **DSA / DeepSeek-V3.2** (2512.02556) | 2025 | Learned "lightning indexer" with few heads, FP8, top-k 2048 tokens | Yes | No | No (IndexCache adds it) | No | HF/vLLM/SGLang | (trained) |
| **TriangleMix** (2507.21526) | 2025 | Static: dense in some layers, Triangle pattern in others (chosen by decoding-time contribution) | Yes | **Per layer** (static) | No | No | aka.ms/TriangleMix | 15.3x attention in Triangle layers at 128K; +6–19% TTFT on top of dynamic methods |
| **AnchorAttention** (2505.23520) | 2025 | Pattern-based anchor (near-max scores) + difference-aware stripe sparsity | Yes | Not stated | No | No | — | 1.44x over prior SOTA at 128K |
| **SampleAttention** (2406.15486) | 2024 | Fixed-percentage local window + two-stage query-guided KV filtering for column stripes | Yes | Per head at runtime | No | No | — | TTFT up to 2.42x vs FA |
| **VSPrefill** (2603.04460) | 2026 | Trained VSIndexer predicts vertical/slash importance; adaptive cumulative threshold allocates **per-layer** budgets | Yes | Per layer | No | No | — | 4.95x at 128K, 98.35% accuracy retained (lightweight training) |
| **FlashPrefill V2** (2608.19758) | 2026 | "Instantaneous pattern discovery" + max-based dynamic thresholding + mean correction (details not on abstract) | Yes | Not stated | Not stated | No | H20; SGLang backend | 27.19x BF16 / 47.26x FP8 vs FA2 at 128K |
| **Elastic Attention** (2601.17367) | 2026 | Trained Attention Router assigns each head a computation mode per input | n/a | Per head, per input (trained 12 h on 8xA800) | No | No | — | (no numbers on abstract) |
| **VFA** (2604.12798) | 2026 | Not a scorer: seeds running max from key-block summaries, freezes it; VSA = VFA + BLASST skipping | n/a | n/a | No | No | — | ~2x vs C16V32 baseline; notes "simple Q and K block summaries fail because of intra-block heterogeneity" |
| **The Sparse Frontier** (2504.17768) | 2025 | Study of 6 training-free methods, ≤128K, sparsity ≤0.95 | — | — | — | No | github (per abstract) | Fine-grained per-query estimation "impractical" in prefill; longer sequences tolerate more sparsity |
| **FastV** (2403.06764, ECCV 2024 oral) | 2024 | n/a (prunes visual tokens after layer 2) | n/a | n/a | No | **Yes**: finds "attention computation over visual tokens is of extreme inefficiency in the deep layers" | github pkunlp-icler/FastV | 45% FLOPs cut on LLaVA-1.5-13B |
| **Visual Attention Sink** (2503.03321) | 2025 | n/a | n/a | n/a | No | **Yes**: certain visual tokens get high attention regardless of relevance (massive activations) | — | VAR redistribution, training-free |
| **SparseMM** (ICCV 2025) | 2025 | n/a (per-head KV budgets by visual-head score) | n/a | Per head; <5% of heads are "visual heads" | No | **Yes** | — | (per search snippet only) |
| **StreamingLLM** (2309.17453, ICLR 2024) | 2023 | n/a (attention sinks + window) | n/a | n/a | No | No | released | 22.2x vs sliding-window recompute |

---

## 2. Idea-by-idea positioning

### Idea 1 — score tiles with downsampled Q and K (mean-pool / stride / max-pool / antidiagonal) so skipped tiles never need the full QK^T

**Closest prior work (ranked):**

1. **CoSA (2607.25291)** is the nearest in spirit to "cheap proxy + BLASST-style in-kernel skip". Its Stage-1 proxy *stride-downsamples* Q and K, takes a softmax, max-pools over keys within each 128-token block, and emits an ordered block list; Stage 2 then applies a running-max skip rule "following" BLASST/SpargeAttn, with the ordering chosen so blocks that contain the row max are visited first (so the running max rises early and the skip is "both safer and more aggressive"). It does **not** ablate stride vs mean vs antidiagonal, does not report the stride value, and was measured on an H20 against FlashAttention-2 (4.93x attention at 128K).
2. **XAttention (2503.16428)** is the antidiagonal precedent exactly: within each block it multiplies Q[i::S] by K[i::S]^T (S = 8 or 16 in main tables; 4 and 64 in ablations), sums, softmax-normalizes, and keeps the smallest block set reaching τ = 0.9. Up to 13.5x attention at 256K; pattern selection 24.9x / 5.9x cheaper than MInference / FlexPrefill. BLASST already combined XAttention-prefill with BLASST-decode (Table 7).
3. **MInference block-sparse head (2407.02490)** and **SpargeAttention Stage 1 (2502.18137)** are the mean-pool precedents: MInference "perform[s] mean pooling on both query and key vectors in blocks of 64" and argues pooling and matmul commute; SpargeAttention additionally drops the pooled estimate for blocks whose tokens are not "self-similar" (cosine-similarity test), because a mean is only a faithful summary when the block is homogeneous. VFA (2604.12798) independently notes "simple Q and K block summaries fail because of intra-block heterogeneity."

Also relevant: **ProxyAttn** pools across *heads* and keeps every 4th token (stride 4), cost ≈ g/(n·stride²) of full attention; **SALE** estimates with 4-bit QK^T (low precision instead of low resolution); **SparQ** downsamples along the *head dimension* (top-r query components); **Quest** uses per-page min/max key bounds; **DraftAttention** uses a pooled low-res attention map for video DiTs; **Sparse VideoGen** profiles on 1% of query rows; **SeerAttention / MoBA / DSA** are the learned-pooling versions. **Block Sparse Flash Attention (2512.07011)** is the "exact QK^T, skip PV" twin of BLASST and states the ceiling plainly: "the skipped PV multiplication and value loads constitute approximately 50% of attention FLOPs and 50% of HBM bandwidth."

**The QK^T floor, as the literature states it:** TRT-LLM docs: "Due to the design that BMM1 (Q·K^T) in the attention kernel is not skipped, the performance gain is capped to 1.8x at kernel level." BLASST measured 1.52x at ~72% sparsity on B200/H200. Why the cap is not simply 2x: on H100 "there are 512x more matmul FLOPS compared to exponential operations, but the exponential has 256x lower throughput" (989 TFLOPS FP16 vs 3.9 TFLOPS special functions; FlashAttention-3), and on Blackwell "softmax's exp() is now as expensive as the matrix multiplies" (PyTorch FA4 blog, Mar 2026) — so skipping exp buys more than its FLOP share suggests, but BMM1 remains a hard floor. No paper I found reports a measured BMM1/softmax/BMM2 time split for a skip-softmax kernel; that measurement would itself be a contribution.

**What is already shown:** cheap proxies (mean-pool, stride, antidiagonal, head-pool, low-bit) can select blocks for a prefill kernel at <10% overhead with near-dense accuracy; a proxy stage can be stacked in front of a BLASST-style in-kernel skip (CoSA; BLASST+XAttention Table 7).

**What our project would add (honest):** the proxy-then-skip pipeline itself is not new. What is missing is (a) a controlled comparison of the four pooling operators (mean / stride / max / antidiagonal) as the *same* front-end to the *same* BLASST threshold, holding tile size and kernel fixed, and (b) a measurement of how much of BLASST's time is the unskippable QK^T floor and how much of that floor a proxy recovers. CoSA and XAttention each report one operator; nobody reports the operator sweep, and nobody ties it to the 1.8x cap.

### Idea 2 — reuse the binary tile-skip mask across neighboring layers

**Closest prior work (ranked):**

1. **Kascade (2512.16391, Microsoft Research India)** — training-free, prefill *and* decode, H100 vs FA3. Anchor layers compute exact post-softmax weights, pool them per 128-query tile, and take top-k keys per key head; the layers in between reuse the nearest anchor's indices with a head-to-head remap (which "is critical for high accuracy"). Cross-layer similarity (share of layer b's attention mass captured by layer a's top-k) is "close to 1" for most adjacent pairs (most > 0.98 in Fig. 3) and "decays with layer distance." Anchors for Llama-3.1-8B: [0, 2, 8, 13, 14]; Qwen3-8B: [0, 2, 7, 14, 23]. 2.2x prefill attention (the paper also says 2.1x), 4.1x decode attention; LongBench 45.02 vs 45.92 dense at 10% top-k.
2. **IndexCache (2603.12201, Z.ai/Tsinghua)** — reuses DSA lightning-indexer top-k indices across layers; adjacent-layer index overlap 0.7–1.0, early-vs-late ≤ 0.4; a training-free greedy search picks which layers keep an indexer (uniform interleaving "degrades quality"); 1.82x prefill at 200K on a 30B DSA model. Only applies to models that already have a learned indexer.
3. **TidalDecode (2410.05076, ICLR 2025)** — decode only; two token-selection layers (the one after the first two full layers and one middle layer) pick top-k token positions that all later layers reuse; recall drops below 20% without the mid-layer re-selection, ~40% with it.

Also relevant: **SharePrefill** clusters heads *across layers* and lets a non-pivotal head reuse a pivotal head's block mask if JS distance < 0.2; **LessIsMore** builds a cross-head token set that "can be reused across layers" (decode, reasoning); **LiSA** gives the cleanest similarity statistic (JS divergence of adjacent-layer attention "below the 0.05 threshold" on LLaMA3-8B; layers 1–2 and 11–14 are exceptions; "shallow layers are sensitive") but shares *attention weights* and needs training; **CLA / MiniCache / KVSharer / YOIO / HySparse2** share *KV cache or indices*, not masks, and CLA/YOIO/HySparse2 are trained architectures.

**What is already shown:** top-k key *indices* and indexer outputs are stable enough across adjacent layers to be reused training-free, with ~1–2% accuracy cost at 10% budgets; layer 0 and a few middle layers must recompute; head remapping matters.

**What our project would add (honest):** reusing a *threshold-derived binary tile mask* (rather than a top-k index set) across layers has not been reported, and the two objects differ in a way that matters: a BLASST mask has no fixed budget, so a reused mask inherits layer i's sparsity level at layer i+1 even if layer i+1 is denser under the same λ (BLASST Fig. 7 shows the per-layer sparsity varies a lot). The open questions we can answer are whether mask-level reuse is as stable as index-level reuse, how much of the skip-decision + QK^T work it saves (since a reused mask lets us skip BMM1 too, which Kascade's reuse layers also do), and whether a cheap "mask drift" check (e.g., recompute on a sampled row block) is enough to decide when to refresh. The Kascade anchor sets are a sensible starting schedule.

### Idea 3 — different thresholds per layer / head / modality (text vs vision tokens in Qwen2.5-VL)

**Closest prior work (ranked):**

1. **Block Sparse Flash Attention (2512.07011)** already stores a threshold tensor "indexed by layer, head, and position" (plus a sparsity-level dimension), calibrated so each query row keeps a fixed top-k; it explicitly contrasts itself with BLASST's "single global threshold per (model, target-sparsity)." Results are modest (1.13–1.16x TTFT, 1.38x attention-only on A100) and text-only. This is the direct per-layer/per-head-threshold precedent for a BLASST-style kernel.
2. **SpargeAttention (2502.18137)** grid-searches τ, θ, λ *per attention layer* under an L1 error bound and concludes "setting different hyperparameters for each layer and head is necessary" (from CogVideoX analysis); it also shows sparsity rising with diffusion timestep and with sequence length (6.8% @8K → 54% @128K at fixed error).
3. **MMInference (2504.16083)** is the modality precedent: it finds VLMs are less sparse than text LLMs (top 5.78% of weights for 95% recall vs 1.79%), that "attention across modalities diverges significantly from intra-modality attention," and that "the sparse index achieves high recall for textual regions but fails to generalize to visual ones"; LLaVA-Video "primarily uses Vertical-Slash pattern for purely textual inputs" and "transitions to a Grid pattern" once video is appended. It handles this with per-head *patterns* and in-kernel permutation, **not** with per-modality thresholds or budgets, and it does not report separate text-vs-vision sparsity numbers.

Also relevant: BLASST's own Figure 7 ("Substantial head-level and layer-level variance motivates adaptive thresholding strategies"); FlexPrefill App. I ("large differences in the sparsity rates required for different attention heads"); ProxyAttn Fig. 2(c); Ada-KV Fig. 1 (most heads keep nearly all attention in the top 5% of cache, dispersed heads need much more); DuoAttention / RazorAttention / HeadKV (binary or scored per-head budgets); PyramidKV (per-layer budgets, lower layers broader); Tactic / Twilight (cumulative-attention targets instead of fixed budgets); TriangleMix / VSPrefill (per-layer static or learned budgets). On the modality side: FastV ("attention computation over visual tokens is of extreme inefficiency in the deep layers"), Visual Attention Sink (visual sink tokens receive high attention regardless of relevance), SparseMM (<5% of heads are "visual heads"), and in video DiTs Sparse VideoGen / AdaSpa (per-head spatial vs temporal types, per-step). NVIDIA's TRT-LLM reportedly applies Skip Softmax to Wan 2.2 with skipping disabled for early denoising timesteps (unverified; see list below).

**What is already shown:** under one threshold, achieved sparsity varies widely across heads and layers (BLASST, FlexPrefill, ProxyAttn, Ada-KV); per-layer/per-head calibration is feasible and helps for block-sparse kernels (BSFA, SpargeAttention); VLM attention is denser than text attention and text-estimated sparse indices fail on vision regions (MMInference).

**What our project would add (honest):** per-layer and per-head thresholds for a BLASST-style kernel are largely done by BSFA (though with a top-k-matching calibration rather than BLASST's sparsity-vs-ln λ fit, and on A100). The genuinely open piece is the *modality* axis: no paper I found assigns different skip thresholds to text-query vs vision-query tiles (or text-key vs vision-key tiles) in a VLM, and none reports the per-modality sparsity achieved by a single BLASST λ on Qwen2.5-VL. MMInference tells us the effect exists but measures it only through pattern choice. A clean contribution is: measure tile-skip rates by (query modality x key modality) under one λ, then show whether two or four thresholds recover accuracy at equal sparsity. The per-layer/per-head part should be framed as a replication of BSFA/SpargeAttention findings inside BLASST's calibration framework, not as new.

---

## 3. BibTeX-ready references

```bibtex
@inproceedings{yuan2026blasst,
  title     = {{BLASST}: Dynamic BLocked Attention Sparsity via Softmax Thresholding},
  author    = {Yuan, Jiayi and Shinn, Cameron and Xu, Kai and Cui, Jingze and Klimiashvili, George and Xiao, Guangxuan and Zheng, Perkz and Li, Bo and Zhou, Yuxin and Ye, Zhouhai and You, Weijie and Zheng, Tian and Brown, Dominic and Wang, Pengbo and Hoehnerbach, Markus and Cai, Richard and Demouth, Julien and Owens, John D. and Hu, Xia and Han, Song and Liu, Timmy and Mao, Huizi},
  booktitle = {Proceedings of Machine Learning and Systems (MLSys)},
  year      = {2026},
  note      = {arXiv:2512.12087}
}

@article{ohayon2025bsfa,
  title   = {Block Sparse Flash Attention},
  author  = {Ohayon, Daniel and Lamprecht, Itay and Hubara, Itay and Cohen, Israel and Soudry, Daniel and Elata, Noam},
  journal = {arXiv preprint arXiv:2512.07011},
  year    = {2025}
}

@inproceedings{jiang2024minference,
  title     = {{MInference} 1.0: Accelerating Pre-filling for Long-Context {LLMs} via Dynamic Sparse Attention},
  author    = {Jiang, Huiqiang and Li, Yucheng and Zhang, Chengruidong and Wu, Qianhui and Luo, Xufang and Ahn, Surin and Han, Zhenhua and Abdi, Amir H. and Li, Dongsheng and Lin, Chin-Yew and Yang, Yuqing and Qiu, Lili},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  note      = {Spotlight. arXiv:2407.02490}
}

@inproceedings{zhang2025spargeattn,
  title     = {{SpargeAttention}: Accurate and Training-free Sparse Attention Accelerating Any Model Inference},
  author    = {Zhang, Jintao and Xiang, Chendong and Huang, Haofeng and Wei, Jia and Xi, Haocheng and Zhu, Jun and Chen, Jianfei},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  note      = {arXiv:2502.18137}
}

@inproceedings{xu2025xattention,
  title     = {{XAttention}: Block Sparse Attention with Antidiagonal Scoring},
  author    = {Xu, Ruyi and Xiao, Guangxuan and Huang, Haofeng and Guo, Junxian and Han, Song},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  note      = {arXiv:2503.16428}
}

@inproceedings{lai2025flexprefill,
  title     = {{FlexPrefill}: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inference},
  author    = {Lai, Xunhao and Lu, Jianqiao and Luo, Yao and Ma, Yiyuan and Zhou, Xun},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  note      = {Oral. arXiv:2502.20766}
}

@article{gao2024seerattention,
  title   = {{SeerAttention}: Learning Intrinsic Sparse Attention in Your {LLMs}},
  author  = {Gao, Yizhao and Zeng, Zhichen and Du, Dayou and Cao, Shijie and Zhou, Peiyuan and Qi, Jiaxing and Lai, Junjie and So, Hayden Kwok-Hay and Cao, Ting and Yang, Fan and Yang, Mao},
  journal = {arXiv preprint arXiv:2410.13276},
  year    = {2024}
}

@article{xue2026cosa,
  title   = {{CoSA}: Accelerating Long-Context Inference via Proxy-Kernel Co-Designed Sparse Attention},
  author  = {Xue, Yufei and Niu, Lin and Liu, Hong and Liu, Siran and Shao, Hanyong and Liu, Wei and Yu, Guanghua and Zhu, Jianchen and Zhang, Jun},
  journal = {arXiv preprint arXiv:2607.25291},
  year    = {2026}
}

@inproceedings{wang2026proxyattn,
  title     = {{ProxyAttn}: Guided Sparse Attention via Representative Heads},
  author    = {Wang, Yixuan and He, Huang and Bao, Siqi and Wu, Hua and Wang, Haifeng and Zhu, Qingfu and Che, Wanxiang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {arXiv:2509.24745}
}

@article{ji2025sale,
  title   = {{SALE}: Low-bit Estimation for Efficient Sparse Attention in Long-context {LLM} Prefilling},
  author  = {Ji, Xiaodong and Zhang, Hailin and Fu, Fangcheng and Cui, Bin},
  journal = {arXiv preprint arXiv:2505.24179},
  year    = {2025}
}

@inproceedings{li2025mminference,
  title     = {{MMInference}: Accelerating Pre-filling for Long-Context {VLMs} via Modality-Aware Permutation Sparse Attention},
  author    = {Li, Yucheng and Jiang, Huiqiang and Zhang, Chengruidong and Wu, Qianhui and Luo, Xufang and Ahn, Surin and Abdi, Amir H. and Li, Dongsheng and Gao, Jianfeng and Yang, Yuqing and Qiu, Lili},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  note      = {arXiv:2504.16083}
}

@inproceedings{xi2025svg,
  title     = {Sparse {VideoGen}: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity},
  author    = {Xi, Haocheng and Yang, Shuo and Zhao, Yilong and Xu, Chenfeng and Li, Muyang and Li, Xiuyu and Lin, Yujun and Cai, Han and Zhang, Jintao and Li, Dacheng and Chen, Jianfei and Stoica, Ion and Keutzer, Kurt and Han, Song},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  note      = {arXiv:2502.01776}
}

@inproceedings{xia2025adaspa,
  title     = {Training-free and Adaptive Sparse Attention for Efficient Long Video Generation},
  author    = {Xia, Yifei and Ling, Suhan and Fu, Fangcheng and Wang, Yujie and Li, Huixia and Xiao, Xuefeng and Cui, Bin},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025},
  note      = {arXiv:2502.21079}
}

@article{shen2025draftattention,
  title   = {{DraftAttention}: Fast Video Diffusion via Low-Resolution Attention Guidance},
  author  = {Shen, Xuan and Han, Chenxia and Zhou, Yufa and Xie, Yanyue and Gong, Yifan and Wang, Quanyi and Wang, Yiwei and Wang, Yanzhi and Zhao, Pu and Gu, Jiuxiang},
  journal = {arXiv preprint arXiv:2505.14708},
  year    = {2025}
}

@inproceedings{yang2025tidaldecode,
  title     = {{TidalDecode}: Fast and Accurate {LLM} Decoding with Position Persistent Sparse Attention},
  author    = {Yang, Lijie and Zhang, Zhihao and Chen, Zhuofu and Li, Zikun and Jia, Zhihao},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  note      = {arXiv:2410.05076}
}

@article{deshmukh2025kascade,
  title   = {Kascade: A Practical Sparse Attention Method for Long-Context {LLM} Inference},
  author  = {Deshmukh, Dhruv and Goyal, Saurabh and Kwatra, Nipun and Ramjee, Ramachandran},
  journal = {arXiv preprint arXiv:2512.16391},
  year    = {2025}
}

@article{bai2026indexcache,
  title   = {{IndexCache}: Accelerating Sparse Attention via Cross-Layer Index Reuse},
  author  = {Bai, Yushi and Dong, Qian and Jiang, Ting and Lv, Xin and Du, Zhengxiao and Zeng, Aohan and Tang, Jie and Li, Juanzi},
  journal = {arXiv preprint arXiv:2603.12201},
  year    = {2026}
}

@article{peng2025shareprefill,
  title   = {Accelerating Prefilling for Long-Context {LLMs} via Sparse Pattern Sharing},
  author  = {Peng, Dan and Fu, Zhihui and Ye, Zewen and Song, Zhuoran and Wang, Jun},
  journal = {arXiv preprint arXiv:2505.19578},
  year    = {2025}
}

@inproceedings{yang2026lessismore,
  title     = {Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention},
  author    = {Yang, Lijie and Zhang, Zhihao and others},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {arXiv:2508.07101 (v1 title: Less Is More: Training-Free Sparse Attention with Global Locality for Efficient Reasoning). Full author list not verified.}
}

@article{mu2025lisa,
  title   = {Cross-layer Attention Sharing for Pre-trained Large Language Models},
  author  = {Mu, Yongyu and Wu, Yuzhang and Fan, Yuchun and Wang, Chenglong and Li, Hengyu and Zeng, Jiali and He, Qiaozhi and Yang, Murun and Meng, Fandong and Zhou, Jie and Xiao, Tong and Zhu, Jingbo},
  journal = {Transactions of the Association for Computational Linguistics},
  year    = {2025},
  note    = {arXiv:2408.01890}
}

@article{brandon2024cla,
  title   = {Reducing Transformer Key-Value Cache Size with Cross-Layer Attention},
  author  = {Brandon, William and Mishra, Mayank and Nrusimha, Aniruddha and Panda, Rameswar and Ragan-Kelly, Jonathan},
  journal = {arXiv preprint arXiv:2405.12981},
  year    = {2024}
}

@article{liu2024minicache,
  title   = {{MiniCache}: {KV} Cache Compression in Depth Dimension for Large Language Models},
  author  = {Liu, Akide and Liu, Jing and Pan, Zizheng and He, Yefei and Haffari, Gholamreza and Zhuang, Bohan},
  journal = {arXiv preprint arXiv:2405.14366},
  year    = {2024}
}

@article{yang2024kvsharer,
  title   = {{KVSharer}: Efficient Inference via Layer-Wise Dissimilar {KV} Cache Sharing},
  author  = {Yang, Yifei and Cao, Zouying and Chen, Qiguang and Qin, Libo and Yang, Dongjie and Zhao, Hai and Chen, Zhi},
  journal = {arXiv preprint arXiv:2410.18517},
  year    = {2024}
}

@article{sun2026yoio,
  title   = {You Only Index Once: Cross-Layer Sparse Attention with Shared Routing},
  author  = {Sun, Yutao and Zhang, Yanqi and Dong, Li and Wang, Jianyong and Wei, Furu},
  journal = {arXiv preprint arXiv:2606.06467},
  year    = {2026}
}

@article{wei2026hysparse2,
  title   = {{HySparse2}: Hybrid Sparse Attention with Two-Level {KV} Sharing},
  author  = {Wei, Jianyu and Gao, Yizhao and Zhang, Qihao and Chen, Shimao and Tang, Zhengju and Cheng, Yu and Zhou, Shengjie and Jiang, Zihan and Song, Yifan and Zhang, Hailin and Zhao, Liang and Yang, Bo and Wang, Gang and Cao, Shijie and Luo, Fuli},
  journal = {arXiv preprint arXiv:2609.26368},
  year    = {2026}
}

@inproceedings{xiao2025duoattention,
  title     = {{DuoAttention}: Efficient Long-Context {LLM} Inference with Retrieval and Streaming Heads},
  author    = {Xiao, Guangxuan and Tang, Jiaming and Zuo, Jingwei and Guo, Junxian and Yang, Shang and Tang, Haotian and Fu, Yao and Han, Song},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  note      = {arXiv:2410.10819}
}

@article{tang2024razorattention,
  title   = {{RazorAttention}: Efficient {KV} Cache Compression Through Retrieval Heads},
  author  = {Tang, Hanlin and Lin, Yang and Lin, Jing and Han, Qingsen and Hong, Shikuan and Yao, Yiwu and Wang, Gongyi},
  journal = {arXiv preprint arXiv:2407.15891},
  year    = {2024}
}

@inproceedings{fu2025headkv,
  title     = {Not All Heads Matter: A Head-Level {KV} Cache Compression Method with Integrated Retrieval and Reasoning},
  author    = {Fu, Yu and Cai, Zefan and Asi, Abedelkadir and Xiong, Wayne and Dong, Yue and Xiao, Wen},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  note      = {arXiv:2410.19258}
}

@inproceedings{feng2025adakv,
  title     = {{Ada-KV}: Optimizing {KV} Cache Eviction by Adaptive Budget Allocation for Efficient {LLM} Inference},
  author    = {Feng, Yuan and Lv, Junlin and Cao, Yukun and Xie, Xike and Zhou, S. Kevin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  note      = {arXiv:2407.11550; venue per arXiv comments field}
}

@article{cai2024pyramidkv,
  title   = {{PyramidKV}: Dynamic {KV} Cache Compression based on Pyramidal Information Funneling},
  author  = {Cai, Zefan and Zhang, Yichi and Gao, Bofei and Liu, Yuliang and Li, Yucheng and Liu, Tianyu and Lu, Keming and Xiong, Wayne and Dong, Yue and Hu, Junjie and Xiao, Wen},
  journal = {arXiv preprint arXiv:2406.02069},
  year    = {2024}
}

@article{zhu2025tactic,
  title   = {Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context {LLMs}},
  author  = {Zhu, Kan and Tang, Tian and Xu, Qinyu and Gu, Yile and Zeng, Zhichen and Kadekodi, Rohan and Zhao, Liangyu and Li, Ang and Krishnamurthy, Arvind and Kasikci, Baris},
  journal = {arXiv preprint arXiv:2502.12216},
  year    = {2025}
}

@inproceedings{lin2025twilight,
  title     = {Twilight: Adaptive Attention Sparsity with Hierarchical Top-$p$ Pruning},
  author    = {Lin, Chaofan and Tang, Jiaming and Yang, Shuo and Wang, Hanshuo and Tang, Tian and Tian, Boyu and Stoica, Ion and Han, Song and Gao, Mingyu},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  note      = {Spotlight. arXiv:2502.02770}
}

@inproceedings{tang2024quest,
  title     = {Quest: Query-Aware Sparsity for Efficient Long-Context {LLM} Inference},
  author    = {Tang, Jiaming and Zhao, Yilong and Zhu, Kan and Xiao, Guangxuan and Kasikci, Baris and Han, Song},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year      = {2024},
  note      = {arXiv:2406.10774}
}

@article{ribar2023sparq,
  title   = {{SparQ} Attention: Bandwidth-Efficient {LLM} Inference},
  author  = {Ribar, Luka and Chelombiev, Ivan and Hudlass-Galley, Luke and Blake, Charlie and Luschi, Carlo and Orr, Douglas},
  journal = {arXiv preprint arXiv:2312.04985},
  year    = {2023}
}

@article{lu2025moba,
  title   = {{MoBA}: Mixture of Block Attention for Long-Context {LLMs}},
  author  = {Lu, Enzhe and Jiang, Zhejun and Liu, Jingyuan and Du, Yulun and Jiang, Tao and Hong, Chao and Liu, Shaowei and He, Weiran and Yuan, Enming and Wang, Yuzhi and Huang, Zhiqi and Yuan, Huan and Xu, Suting and Xu, Xinran and Lai, Guokun and Chen, Yanru and Zheng, Huabin and Yan, Junjie and Su, Jianlin and Wu, Yuxin and Zhang, Neo Y. and Yang, Zhilin and Zhou, Xinyu and Zhang, Mingxing and Qiu, Jiezhong},
  journal = {arXiv preprint arXiv:2502.13189},
  year    = {2025}
}

@article{deepseek2025v32,
  title   = {{DeepSeek-V3.2}: Pushing the Frontier of Open Large Language Models},
  author  = {{DeepSeek-AI}},
  journal = {arXiv preprint arXiv:2512.02556},
  year    = {2025}
}

@article{he2025trianglemix,
  title   = {Accelerating Prefilling via Decoding-time Contribution Sparsity},
  author  = {He, Zhiyuan and Zhang, Yike and Zhang, Chengruidong and Jiang, Huiqiang and Yang, Yuqing and Qiu, Lili},
  journal = {arXiv preprint arXiv:2507.21526},
  year    = {2025},
  note    = {TriangleMix}
}

@article{zhang2025anchorattention,
  title   = {{AnchorAttention}: Difference-Aware Sparse Attention with Stripe Granularity},
  author  = {Zhang, Yu and Guo, Dong and Wu, Fang and Zhu, Guoliang and Ding, Dian and Zhang, Yiming},
  journal = {arXiv preprint arXiv:2505.23520},
  year    = {2025}
}

@article{zhu2024sampleattention,
  title   = {{SampleAttention}: Near-Lossless Acceleration of Long Context {LLM} Inference with Adaptive Structured Sparse Attention},
  author  = {Zhu, Qianchao and Duan, Jiangfei and Chen, Chang and Liu, Siran and Feng, Guanyu and Lv, Xin and Chuanfu, Xiao and Lin, Dahua and Yang, Chao},
  journal = {arXiv preprint arXiv:2406.15486},
  year    = {2024}
}

@article{chen2026vsprefill,
  title   = {{VSPrefill}: Vertical-Slash Sparse Attention with Lightweight Indexing for Long-Context Prefilling},
  author  = {Chen, Guanzhong},
  journal = {arXiv preprint arXiv:2603.04460},
  year    = {2026}
}

@article{fan2026flashprefillv2,
  title   = {{FlashPrefill V2}: Block-Sparse Prefill Attention for Long-Context {LLM} Serving},
  author  = {Fan, Qihang and Huang, Huaibo and Wu, Zhiying and Wang, Bingning and He, Ran},
  journal = {arXiv preprint arXiv:2608.19758},
  year    = {2026}
}

@article{tang2026elastic,
  title   = {Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers},
  author  = {Tang, Zecheng and Qiu, Quantong and Yang, Yi and Hong, Zhiyi and Xiang, Haiya and Liu, Kebin and Dang, Qingqing and Li, Juntao and Zhang, Min},
  journal = {arXiv preprint arXiv:2601.17367},
  year    = {2026}
}

@article{sun2026vfa,
  title   = {{VFA}: Relieving Vector Operations in Flash Attention with Global Maximum Pre-computation},
  author  = {Sun, Yupeng and Li, Yanzhao and Zou, Zhiqiang and Du, Bai and Zhang, Zhiyuan and Dong, Hui and Fan, Gaoyige and Wang, Hui},
  journal = {arXiv preprint arXiv:2604.12798},
  year    = {2026}
}

@article{nawrot2025sparsefrontier,
  title   = {The Sparse Frontier: Sparse Attention Trade-offs in Transformer {LLMs}},
  author  = {Nawrot, Piotr and Li, Robert and Huang, Renjie and Ruder, Sebastian and Marchisio, Kelly and Ponti, Edoardo M.},
  journal = {arXiv preprint arXiv:2504.17768},
  year    = {2025}
}

@inproceedings{chen2024fastv,
  title     = {An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models},
  author    = {Chen, Liang and Zhao, Haozhe and Liu, Tianyu and Bai, Shuai and Lin, Junyang and Zhou, Chang and Chang, Baobao},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024},
  note      = {Oral. arXiv:2403.06764 (FastV)}
}

@article{kang2025visualsink,
  title   = {See What You Are Told: Visual Attention Sink in Large Multimodal Models},
  author  = {Kang, Seil and Kim, Jinyeong and Kim, Junhyeok and Hwang, Seong Jae},
  journal = {arXiv preprint arXiv:2503.03321},
  year    = {2025}
}

@inproceedings{wang2025sparsemm,
  title     = {{SparseMM}: Head Sparsity Emerges from Visual Concept Responses in {MLLMs}},
  author    = {Wang, Jiahui and others},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025},
  note      = {Author list not verified; seen only via search snippet of the CVF open-access PDF}
}

@inproceedings{xiao2024streamingllm,
  title     = {Efficient Streaming Language Models with Attention Sinks},
  author    = {Xiao, Guangxuan and Tian, Yuandong and Chen, Beidi and Han, Song and Lewis, Mike},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  note      = {arXiv:2309.17453}
}

@inproceedings{yang2025lserve,
  title     = {{LServe}: Efficient Long-sequence {LLM} Serving with Unified Sparse Attention},
  author    = {Yang, Shang and Guo, Junxian and Tang, Haotian and Hu, Qinghao and Xiao, Guangxuan and Tang, Jiaming and Lin, Yujun and Liu, Zhijian and Lu, Yao and Han, Song},
  booktitle = {Proceedings of Machine Learning and Systems (MLSys)},
  year      = {2025},
  note      = {arXiv:2502.14866}
}

@article{shah2024flashattention3,
  title   = {{FlashAttention-3}: Fast and Accurate Attention with Asynchrony and Low-precision},
  author  = {Shah, Jay and Bikshandi, Ganesh and Zhang, Ying and Thakkar, Vijay and Ramani, Pradeep and Dao, Tri},
  journal = {arXiv preprint arXiv:2407.08608},
  year    = {2024}
}

@misc{nvidia2025skipsoftmax_blog,
  title        = {Accelerating Long-Context Inference with Skip Softmax in {NVIDIA} {TensorRT} {LLM}},
  author       = {Tewari, Laikh and Xu, Kai and Li, Bo and Oh, Fred},
  howpublished = {NVIDIA Developer Blog},
  year         = {2025},
  month        = dec,
  note         = {https://developer.nvidia.com/blog/accelerating-long-context-inference-with-skip-softmax-in-nvidia-tensorrt-llm/}
}

@misc{nvidia2025skipsoftmax_docs,
  title        = {Accelerating Long Context Inference with Skip Softmax Attention (TensorRT-LLM tech blog 16)},
  author       = {{NVIDIA TensorRT-LLM}},
  howpublished = {https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md},
  year         = {2025}
}

@misc{pytorch2026flexfa4,
  title        = {{FlexAttention} + {FlashAttention-4}: Fast and Flexible},
  author       = {{PyTorch Team}},
  howpublished = {PyTorch Blog, https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/},
  year         = {2026},
  month        = mar
}
```

---

## 4. Unverified / could not read

- **TensorRT-LLM tech blog 28** (Skip Softmax on Wan 2.2 video generation, reportedly `target_sparsity=0.75`, `disabled_until_timestep=0.86`): both the GitHub and nvidia.github.io URLs returned 404 from this network. The settings above come only from a search snippet; do not cite numbers from it without opening the page.
- **BLASST tile size**: the paper does not give B_r/B_c numerically; "128x128" in our proposal is our assumption. BLASST also does not report a BMM1 / softmax / BMM2 time breakdown; the "1.8x cap" is from the TRT-LLM docs, not the paper.
- **BLASST FlashInfer integration**: stated in the paper; I did not open FlashInfer's repo to confirm the kernel exists there.
- **CoSA stride value** and KAP overhead: not stated in the HTML I read; no ablation of stride vs mean-pool vs antidiagonal exists in the paper.
- **SpargeAttention Stage-2 skip direction**: the paper's text and Algorithm 1 disagree on "<" vs ">"; I did not resolve which is a typo.
- **XAttention GPU** for timing: not named in the HTML; block size B also not stated (only an 8x8 illustration).
- **Kascade prefill speedup**: abstract says 2.2x, contributions list says 2.1x. Venue not confirmed (v2 dated Aug 2026). Code repo "will be available."
- **Block Sparse Flash Attention date**: arXiv ID is 2512.* but the HTML stamp reads 29 Sep 2026 (v2); the paper's threshold comparison direction is also internally inconsistent.
- **SeerAttention venue**: could not confirm any peer-reviewed venue; cited as arXiv.
- **PyramidKV, RazorAttention, CLA, MiniCache, KVSharer, Tactic, SALE, DraftAttention, AnchorAttention, AdaSpa venues**: only the arXiv page was read (AdaSpa's ICCV 2025 acceptance is per CVF search results, not the arXiv page). AdaSpa's 1.59x–2.04x figure is from a search snippet of the ICCV version.
- **LessIsMore** author list and the "reused across layers" claim: from the arXiv abstract via search summary; I did not open the full text.
- **SparseMM** author list: not read; cited only via the CVF open-access PDF URL in search results.
- **Attn-QAT** "softmax ≈ two BF16 GEMMs per 128x128 tile": appeared in a search summary but the abstract page does not contain it; not cited in the body.
- **DeepSeek DSA indexer config** (64 heads, head dim 128, top-k 2048): from the Transformers docs mirror, not the DeepSeek report.
- **No paper found** that (a) reuses a threshold-derived binary tile mask (as opposed to top-k indices) across layers, (b) sets separate skip thresholds per modality in a VLM, or (c) measures the per-modality sparsity achieved by a single BLASST threshold. Absence of evidence after ~6 targeted searches, not proof of absence; a direct arXiv listing scan for Sept–Oct 2026 would be the next check.
- **Sparse VideoGen 2 (2505.18875)**, **HySparse v1**, **VideoNSA**, **SparseVLM(+)**: seen in search results only; not read, not relied on.

---

## 5. Update Nov 1, 2026: is Idea 1 already done? (yes)

Checked every paper that cites BLASST (Semantic Scholar, 15 citing papers) plus web searches.
Cheap scoring before QK^T, combined with a BLASST-style skip, is published:

- **CoSA** (2607.25291, Jul 2026, code in Tencent AngelSlim): stride-downsampled Q/K proxy picks blocks,
  then an in-kernel running-max skip. 4.93x attention, 2.53x TTFT at 128K. This is Idea 1 end to end.
- **AMEND** (2609.09823, Sep 2026): "predicts each block's BLASST verdict from margins audited at earlier
  steps" (decode, GPU + processing-in-memory).
- **VFA / VSA** (2604.12798, Apr 2026): cheap running-max estimate from key-block summaries, combined with
  BLASST skipping.
- Older proxies of the same QK^T: MInference (mean-pool), SpargeAttention (mean-pool + similarity check),
  XAttention (antidiagonal), ProxyAttn (head-pool + stride 4), SALE (4-bit QK^T).
- Also new: **cumulative energy filtering** (2606.16317) replaces BLASST's fixed threshold with a dynamic
  one and reports 61.42% -> 82% sparsity at similar accuracy (video diffusion).

Conclusion: Idea 1 is not a contribution. At most it is a baseline or a controlled operator comparison.

## 6. Update Nov 1, 2026: are Ideas 2 and 3 already done?

**Idea 2 (reuse skip mask across layers): largely done.**
- Kascade (2512.16391, Dec 2025, Microsoft): training-free, prefill and decode, Llama-3.1-8B. Anchor layers
  compute exact top-k keys per 128-query tile; layers in between reuse them with head remapping.
- IndexCache (2603.12201, Mar 2026): reuses top-k indices across layers, 70-100% adjacent-layer overlap,
  training-free greedy choice of which layers recompute; 1.82x prefill at 200K.
- You Only Index Once (2606.06467, Jun 2026), H2Share / Autonomy-of-Heads (2608.06849, Aug 2026),
  LongCat Sparse Attention cross-layer indexing (2608.01662, Aug 2026): more cross-layer index reuse.
- LiteAttention (2511.11062, Nov 2025): reuses skip decisions for tiles across diffusion timesteps
  (across time, not layers), on a FlashAttention kernel.
- Remaining gap: reusing a threshold-based BLASST mask instead of a top-k index set. Thin.

**Idea 3 (different thresholds): per-layer/head done; per-modality open.**
- Per-layer/head thresholds: Block Sparse Flash Attention (2512.07011), SpargeAttention per-layer search,
  cumulative energy filtering (2606.16317, dynamic threshold, 61.42% -> 82% sparsity vs BLASST).
- Modality: MMInference (2504.16083, ICML 2025) shows VLM sparsity differs by modality but handles it with
  permutation and per-head patterns, not thresholds. VideoNSA (2510.02295) keeps text dense and video
  sparse but is trained. UniSparse (2512.14082) reports Qwen2.5-VL-7B on Video-MME at 46-59% sparsity
  without per-modality treatment.
- Not found in 6 searches: separate BLASST-style thresholds for text vs vision, or BLASST skip rates
  measured by modality. Open, but confirm with the instructor at the Nov 11 check-in.
