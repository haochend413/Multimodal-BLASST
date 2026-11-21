# Idea-1 scope: open directions for making BLASST's skip decision cheaper, more aggressive, or more accurate

Written 2026-11-01. Scope: BLASST's per-tile decision **inside one layer**. Cross-layer and cross-step reuse (Kascade, IndexCache, AMEND as reuse, LiteAttention as reuse) is covered in a separate note and is not repeated here.

**How this was done.** I downloaded the arXiv HTML of each paper and converted it to text locally. Every quote below was copied from that text. Math is written as plain text, and "[...]" marks a cut. Each candidate direction was then checked with 1–3 web searches. "Appears open" means none of those searches found the idea; it does not prove nobody has done it.

**Vocabulary** (same words as `00_team_plan.md` and `blasst_paper.md`):
- **Tile**: a 128 query-row x 128 key-column block of the score matrix.
- **Skip rule**: BLASST's test. A tile is skipped if `tile row-max - running row-max < ln λ`.
- **Running max**: the row max over the tiles visited so far.
- **Margin**: `max over rows of (tile row-max - running row-max)`. A tile is skipped exactly when `margin < ln λ`.
- **All-rows rule**: a tile is skipped only if all 128 rows pass.
- **% skipped**: skipped causal tiles divided by all causal tiles.
- **BMM1**: QK^T. **BMM2**: PV.
- New terms are defined where they first appear: oracle, query group, certified skip, LSE.

---

## 0. One-paragraph answer

The proxy-before-QK^T idea is taken; the CoSA/AMEND/VFA notes cover why. Five narrower directions inside one layer still look testable. Three are the strongest:

1. **Certified skip** (rank 1, appears open). A provable upper bound on a tile's max score lets us skip BMM1 and the K load for a tile only when BLASST itself would have skipped it. The output is therefore identical to BLASST's.
2. **Exact-score seeding of the running max** (rank 2, appears open). Start each row's running max at a few exact scores instead of `-inf`, so later tiles can be skipped. BLASST's error bound still holds.
3. **Smaller query groups for the vote** (rank 3, partially done). Replace the all-rows rule with 64-, 32- or 16-row groups. Nobody has measured this on long-context LLMs.

Two more are weaker:
- **LSE (log-sum-exp) reference instead of the running max** (rank 4). Done only for video diffusion models.
- **Visiting order with a global-max upper bound** (rank 5). Mostly done by three papers, but needed as the upper bound for directions 2 and 3.

All five can be measured with the team's PyTorch simulation. Speed claims for directions 1 and 3 need a real kernel.

---

## 1. What the papers admit, and what they did not test

### 1.1 BLASST (2512.12087, v3)
Quotes:
- Order. Sec. 5: "We also investigate whether permuting the tile-row processing order could improve pruning accuracy by establishing a better running maximum earlier. Results in Appendix A (Figure 9) show that the effect is dataset-dependent but generally negligible, confirming BLASST's robustness to processing order."
- Order. App. A.4: "reordering yields similar performance on VT but provides noticeable improvements on FWE." Fig. 9 caption: "Llama 3.1 8B (ctx=8192). We compare Standard Cummax (processing tiles sequentially) with Reordered Cummax (processing tiles in reverse order)."
- BMM1 is never skipped. Sec. 4.1: "even as all QK^T (BMM1) operations are computed, the kernel dynamically skips compute-heavy softmax and attention-value multiplication (BMM2)".
- The vote. Sec. 4: "(2) issuing a VOTE instruction to determine if all threads within a warp agree to skip, and (3) a single ATOMIC instruction to shared memory issued by one thread per warp to coordinate the block-level decision across the softmax warpgroup."
- Fig. 7 caption: "Substantial head-level and layer-level variance motivates adaptive thresholding strategies."
- Conclusion: "the combination of hardware-aware sparse patterns, learned sparsity through training, and adaptive hybrid methods will be the key to unlocking the full potential".

Not evaluated:
- Reordering was tried at one length (8K), on one model, on two tasks (VT, FWE). The result is shown as an image only, and only reverse order was tried.
- No upper bound using the true global max.
- No analysis of how often one row vetoes a tile under the all-rows rule.
- No reference other than the running max.
- No way to skip BMM1.

### 1.2 CoSA (2607.25291): proxy plus ordered in-kernel skip
Quotes:
- Intro: "Though the skipping strategy is oracle-aligned, it still needs to compute the full query-key interactions. Moreover, it applies a conservative skip rule, which leaves much of the potential speedup on the table."
- Limitation 2: "(Running max ≠ global max) It tests each block against a running maximum that need not yet equal the global maximum;" and "(Bucket effect) Its per-row AND (∧) lets a single outlier row veto the skip of an otherwise negligible block."
- Default orders: "a rowmax can lie anywhere, not necessarily in the first (sink) or last (most-recent) block. Yet most high-performance backends still traverse KV blocks in a fixed ascending [...] or descending [...] order."
- Their kernel keeps the 128-row vote: "Two consumer warp groups evaluate 64 query rows each and combine their partial predicates into a single tile-wide vote" (App. D.1: "This realizes the per-128 skip semantics of Eq. (5)").
- Reordering alone (App. D.4, Stage-1 budget 100%, so no proxy pruning): "remapping reaches an approximately 52% skip ratio at Δ=500, whereas the non-remapped execution requires roughly Δ=3000 for a similar ratio and incurs a higher PPL."
- Future work: "we will extend CoSA beyond prefilling to decoding".

Scope and gaps:
- Prefill only ("leaves decoding dense"), H20 only, Qwen3-8B and Llama-3.1-8B.
- The bucket effect is fixed only by reordering with a proxy. The 128-row vote is never relaxed.
- The order study used perplexity on 128 LongBench-v2 samples, not RULER. There is no zero-cost static order and no global-max upper bound.
- BLASST alone is not a row in the main tables. The baselines are MInference, FlexPrefill and XAttention.

### 1.3 AMEND (2609.09823): decode, GPU plus processing-in-memory (PIM)
Quotes:
- "at 90% block sparsity, K still accounts for about 91% of its remaining logical K+V traffic."
- "BLASST uses only the sign of γ*; its magnitude separates a block far below threshold from one barely below it".
- "A predicted drop is irreversible for the current token, so incorrect drops must be bounded, not repaired."

Scope:
- Decode only, 16-token blocks.
- Quality measured in BF16 on A100. Speed measured in an AttAcc-based simulator.
- Realized false-drop rates of 1.50–2.24%: blocks dropped that BLASST would have kept.
- Relevant here because it shows that the K read in decode is the cost BLASST cannot remove, and that its fix makes false drops. Direction 1 below removes K reads with no false drops.

### 1.4 VFA / VSA (2604.12798): seeding and freezing the running max
Quotes:
- Abstract: "naive block summaries of Q/K are insufficient due to intra-block heterogeneity" and "m-initialization is necessary to cover cases where maxima occur in middle blocks".
- Limitations: "workloads with highly non-local or adversarial attention distributions may reduce the effectiveness of reordering and max-freezing. A promising direction is to design adaptive schedules that detect stabilization online".
- "its approximation quality can vary across layers, heads, and tasks. Future work includes exploring stronger yet still lightweight representations (e.g., multi-prototype or norm-aware summaries)".
- "a broader evaluation covering more models, longer-context tasks, and additional generation settings would strengthen the conclusions."

Appendix Table 4 is the only BLASST-variant data. It covers one short task, MMLU_computer_security; the model is not named, though the dense score 0.81 matches their Qwen3-30B row.
- **Blasst_SWA** (sink and local blocks visited first), % skipped:
  - λ=0.1: 32.3% (plain BLASST) → 37.4% (SWA)
  - λ=0.3: 44.0% → 48.9%
  - λ=0.5: 49.5% → 54.1%
  - λ=0.9: 56.1% → 60.5%
- **Blasst_Rowskip**: "refine BLASST from block-level decisions to row-level decisions". It reports "row-wise sparsity" of 62.3–73.5%. Rows are only masked; compute is saved only when every row in the tile is skipped.

The VFA seed (`Q_i · sabsmax(K_j)`) is an approximation, not an exact score, so it does not keep BLASST's error bound. No long-context evaluation.

### 1.5 DynSparse, "cumulative energy filtering" (2606.16317)
Quotes:
- Rule: `t = tile max - LSE^(j-1)` (Eq. 5): the tile max minus the log of the probability mass accumulated so far.
- Abstract: "compared to the BLASST algorithm which is also integrated with FA, our dynamic thresholding strategy enhances sparsity from 61.42% to 82% with a VBench metric drop of less than 5%."
- "In low-entropy (spiky) distributions, once a global maximum ("winner") is encountered, LSE rises sharply, causing all subsequent low-scoring blocks' t_{i,j} to immediately fall below λ".

Scope: only Wan 2.2 5B/14B video diffusion (non-causal attention, VBench). No LLM, causal attention, RULER, decode, or tile order.

### 1.6 Block Sparse Flash Attention, BSFA (2512.07011)
Quotes:
- "BSFA targets prefill (decode is complementary [...]) and extends FA-2, leaving FA-3 integration to future work."
- Says BLASST "loses more accuracy at matched sparsity".
- "Diagonal blocks bypass the gate".

Scope: thresholds per (layer, head, position) calibrated for a fixed top-k. Llama-3.1 and Qwen2.5; A100, A6000, H100. Thresholds per layer or head are out of scope here.

### 1.7 SpargeAttention (2502.18137)
- Its in-kernel skip is **per warp**: "If max(m_local[I_w] − m_ij[I_w]) < λ [...] we will skip the computation of P̃_ij[I_w] V_j". Here `I_w` is the warp's slice of `b_q / c_w` rows.
- The group size is never ablated.
- Table 6 (Llama-3.1, NIAH 128K): in-kernel skip alone gives 27.7% sparsity; the proxy alone gives 51.2%; both together give 54%.
- Hyperparameters per layer: "setting different hyperparameters for each layer and head is necessary".

### 1.8 Others read for limitations
- **LiteAttention** (2511.11062, video diffusion): "its precision is further limited by the use of the cumulative row maximum m_ij instead of the true global row maximum." and "radial-centric ordering improved performance by reaching the global maximum faster."
  - Hopper kernel detail: "On the H100, each matrix multiply–accumulate (MMA) operation is executed by a warpgroup of four warps, and enforcing cross-warp agreement on the skip predicate before continuation would stall otherwise overlapping execution."
- **RBS-Attention** (2609.20971): uses the Cauchy–Schwarz bound `max_k q·k ≤ q·c + ||q||·r`.
  - "For β_b<1, it is a risk score and need not upper-bound every token logit in the block."
  - "Selecting a fixed number of blocks by upper bound does not by itself guarantee recovery of the globally highest-scoring keys: a loose bound can rank an irrelevant block above a relevant one."
  - Scope: prefill only, a ranking proxy with relative thresholds, H100. It does not combine the bound with a BLASST-style threshold.
- **MInference** (2407.02490), App. A: "with a 10k context, the time spent on building the index increases from 5% to 30%". Proxy overhead hurts at short lengths.
- **SALE** (2505.24179), App. A: "it may lose its performance advantage on hardware that does not support efficient 4-bit matrix multiplication."
- **ProxyAttn** (2509.24745): "We will further investigate the utility of the proxy head specifically within the decoding phase of LLMs."
- **XAttention** (2503.16428): its ablation compares antidiagonal, random and diagonal patterns. It has no stated limitations section.

### 1.9 Already done: do not propose these as new

| Idea | Done by |
|---|---|
| Cheap downsampled-Q/K proxy, then BLASST-style skip | CoSA (stride + max-pool), SpargeAttn (mean-pool), XAttention (antidiagonal) |
| Proxy-guided order ("rows' max tile first") | CoSA |
| Running max seeded from key-block summaries | VFA (`sabsmax`), approximate |
| Visiting sink and local tiles first | VFA appendix (Blasst_SWA, short MMLU); BLASST App. A.4 (reverse order, 8K) |
| Thresholds per layer, head or position | BSFA, SpargeAttn |
| Predicting the verdict from the previous decode step | AMEND |
| LSE / cumulative-mass skip rule | DynSparse (video diffusion only) |
| 4-bit QK^T estimate | SALE |

---

## 2. Shared measurement setup (used by every direction)

**Simulation hooks.** `blasst_attn.py` already holds, per 128-row query chunk:
- `s_t` with shape `[B, H, rows, n_kt, 128]`: the scaled scores for every tile;
- `tile_max` with shape `[B, H, rows, n_kt]`: each row's max inside each tile;
- the cumulative max (`cummax`) of `tile_max`: the running max.

Every variant below is a few lines operating on these tensors.

**One pass gives every λ.** For the rules in directions 1, 2, 3 and 5, a skipped tile never raises the running max, because its max is below the running max. So the running-max trajectory, and therefore the margin, does not depend on λ. One forward pass with skipping off gives the skip mask for every λ, as with the existing margin trick. The LSE rule (direction 4) is the exception; see there.

**Matched sparsity.** Compare variants at the same % skipped, not at the same λ. For each variant, choose λ from its saved margins so it hits 50%, 70% and 80% skipped. This needs no GPU time.

**Use a low-noise metric first.** RULER with 50 samples per task is noisy. At 90% accuracy over 150 samples, the standard error is about 2.4 points, so differences of 1–2 points cannot be detected.
- Primary metric: the dense model's log-probability of the gold answer when the answer is fed in (teacher forcing, one forward pass, no generation). Report the change in negative log-likelihood (NLL) versus dense. Alternatively, report the relative L1 error of the attention output (SpargeAttn's metric).
- Confirm with RULER accuracy only at the end.
- CoSA used perplexity for the same reason.

**Cost assumptions.** These are guesses; replace them with the timing from CLAUDE.md step 4.
- Assumed time per RULER sample on one GH200: about 5 s at 8K and about 15 s at 16K (simulated prefill plus up to 128 generated tokens).
- One "RULER config" = 3 tasks x 50 samples at 8K and 16K: 150 x 5 s + 150 x 15 s = 3,000 s ≈ 0.85 GPU-h.
- One margin-recording pass over 20 samples at each of 4K, 8K and 16K: under 0.5 GPU-h.
- Teacher-forced NLL runs take about a third of a RULER config.
- Kevin's DeltaAI balance is 541 h, shared with his research group. Every direction below fits in roughly 3–15 GPU-h.

**Suggested shared figure ("where BLASST's skippable tiles go").** At one λ, plot % skipped under:
1. the oracle: every row uses its true global row max, with the all-rows rule;
2. BLASST: running max, ascending order, all-rows rule;
3. the oracle with 64-row and 16-row votes;
4. the LSE rule.

The gaps between the bars are exactly what directions 2, 3 and 4 try to recover. About 2 engineering days. It also makes a good report figure.

---

## 3. Candidate directions

### Direction A: certified skip with a provable bound, skipping BMM1 with output identical to BLASST (cheaper)

**Idea in plain steps.**
1. For every key tile `j`, store two cheap summaries computed once per layer: the centroid `c_j` (mean key) and the radius `r_j = max over keys ||k − c_j||`. Optionally also the per-dimension min/max box `[lo_j, hi_j]`, as in Quest.
2. Before computing tile `j` for query row `q`, compute an upper bound `U` on the row's tile max.
   - Ball bound (Cauchy–Schwarz: a dot product is at most the product of the lengths): `U = scale·(q·c_j + ||q||·r_j)`.
   - Box bound: `U = scale·(q⁺·hi_j + q⁻·lo_j)`, where `q⁺` keeps the positive entries of `q` and `q⁻` the negative ones.
   - Use the smaller of the two; both are valid upper bounds.
3. **Certified**: skip the tile without loading K or computing BMM1 if `max over rows (U − m_prev) < ln λ`, where `m_prev` is the running max before tile `j`.
4. Why the output is unchanged:
   - BLASST skips tile `j` exactly when `true tile max − m_prev < ln λ` for every row. The running max is updated first, but because `ln λ < 0` the test reduces to this.
   - `U ≥ true tile max`, so every certified tile is one BLASST would also skip.
   - BLASST would not update the running max for that tile either, since its max is below `m_prev`. So the whole trajectory, mask and output are identical to BLASST.
   - BLASST's Appendix B error bound is untouched. This is the "soundness" idea from NN verification (interval and linear bounds) applied to `q·k`.
5. Using a stale running max, for example from 2–3 tiles earlier because the kernel prefetches K ahead, is still sound. An older running max is lower, so certification only becomes rarer, never wrong.

**Gap it fills.**
- CoSA, Limitation 1: "The QK multiplication is therefore unavoidable, and skipping trims only the value-side steps."
- AMEND: "at 90% block sparsity, K still accounts for about 91% of its remaining logical K+V traffic."
- AMEND's fix creates false drops: "incorrect drops must be bounded, not repaired"; realized 1.50–2.24%. Certification has zero false drops.
- RBS-Attention uses the same bound only as a ranking score: "a loose bound can rank an irrelevant block above a relevant one."
- `blasst_paper.md` Sec. 7 noted that a proxy breaks BLASST's Appendix B guarantee. An upper bound keeps it.

**Prior-art status: appears open as a combination.** The pieces exist:
- Quest uses the box bound to pick top-K pages in decode.
- RBS-Attention uses a weakened ball bound as a ranking score in prefill.

Three searches found nobody using a sound bound as a pre-check inside BLASST's test:
- "sound upper bound key block centroid radius skip QK^T exact attention guarantee running maximum";
- "Cauchy-Schwarz bound attention block pruning certified [...]";
- "lossless skip QK matmul blocks key norm bound skip softmax".

One search snippet mentioned a tinygrad feature request, "dynamic block pruning for attention (cauchy-schwarz bounds)", said to be closed as not planned. I could not open it (the repo has issues disabled), so it is unverified.

**Experiment (simulation).**
1. In the prefill loop, compute `c_j`, `r_j`, `lo_j`, `hi_j` per key tile from the post-RoPE `key32`, plus `U` with shape `[B, H, rows, n_kt]`.
2. Record the **certificate margin** `cm = max over rows (U − m_prev)` next to the existing margin. It does not depend on λ.
3. Sanity check: `cm < ln λ` must imply `margin < ln λ` for every tile. Any violation is a bug.
4. Variants:
   - ball, box, and the smaller of the two;
   - sub-block bounds: split each 128-key tile into 8 x 16-key sub-blocks, bound each, and take the max of the sub-block bounds. This is tighter but needs 8 times the metadata;
   - combined with direction B's seeding, which raises `m_prev` earlier;
   - with a stale running max (lag of 1, 2 and 4 tiles).
5. Decode variant:
   - implement decode BLASST (one query row, 128-key tiles, sequential running max);
   - certify a K tile for a KV head only if every query head in its grouped-query attention (GQA) group is certified, like AMEND's GQA reduction;
   - measure K bytes not read per token.
6. Models: Llama-3.1-8B, then Qwen3-8B. RULER 4K/8K/16K, 20 samples per length.

**Metrics.**
- **Certification rate** = certified tiles / BLASST-skipped tiles, at λ for 50% and 70% skipped.
- **BMM1 FLOPs avoided** = certified tiles / causal tiles.
- Bound cost: a `[128 x d] x [d x n_kt]` product per query tile, about 1/128 of BMM1 for the ball bound and 2/128 for the box bound.
- Decode: K bytes avoided as a % of K traffic.
- Accuracy: identical to BLASST by construction. Check with one RULER config: the output should match.

**Cost.**
- Simulation: 3–4 engineering days (prefill 2, decode 1–2).
- GPU: about 2–3 GPU-h (margin passes, about 1 h of decode simulation, 1 sanity config). No accuracy sweeps are needed.
- Optional Triton kernel, extending ModelOpt's `triton_fa.py`:
  - compute the bounds for a query tile with one small matrix multiply before the loop;
  - in the loop, compare against the current running max before the `tl.load` of K;
  - effort 8–10 engineering days, about 5–10 GPU-h of benchmarking on the GH200.

**Needs a real kernel?** Yes, for any speed claim. The whole point is skipping BMM1 and K loads, which the PyTorch simulation cannot time. Without a kernel, report certification rate, FLOPs and bytes avoided, and accuracy (identical). Label any time savings as a ceiling, not a measurement.

**Risk / go–no-go (2 days).** The bounds may be too loose in 128 dimensions. BLASST-skipped tiles with a margin only slightly below `ln λ` will rarely be certified; tiles far below it are the candidates. If the certification rate stays under about 10% of skipped tiles even with 16-key sub-blocks and seeding, stop. A negative result is still reportable as "how loose are sound attention bounds on real Q/K".

### Direction B: exact-score seeding of the running max (more aggressive, still keeps BLASST's guarantee)

**Idea in plain steps.**
1. BLASST starts each row's running max at `-inf`. Early tiles can never be skipped, and later tiles are judged against a running max that is too low (CoSA's "running max ≠ global max").
2. Instead, start each row's running max at the largest of a few **exact** scores for that row, computed with `q·k` in this layer. Seed sources:
   - (i) the sink key (key 0) and the row's own key (the diagonal);
   - (ii) the key that had the row's highest score in the previous layer, same head;
   - (iii) the union of those keys across all heads of the previous layer, about 32 keys per row.
3. Why it is safe: every seed is a real score in this row, so `seed ≤ global row max`. BLASST's bound `p < λ` (Appendix B) still holds, and online softmax remains exact for any starting `m` (the normalizer `ℓ` starts at 0).
4. Cost: a handful of dot products per row, for example 32 x d per row against BMM1's full context length x d. That is about 0.2% of BMM1 at 16K with 32 seeds.
5. Why this differs from VFA: VFA's seed `Q_i·sabsmax(K_j)` is an estimate, not an exact score, and can overshoot, which breaks the bound.
6. Why this differs from cross-layer reuse (Kascade and others): nothing is reused as a mask. Every tile is still tested on its own exact scores. Only the starting reference changes.

**Gap it fills.**
- CoSA: "a rowmax can lie anywhere, not necessarily in the first (sink) or last (most-recent) block". Its fix needs a stride proxy plus a page reorder.
- VFA: "m-initialization is necessary to cover cases where maxima occur in middle blocks". Its seed is approximate, and it asks for "stronger yet still lightweight representations".
- LiteAttention: "precision is further limited by the use of the cumulative row maximum".

Seeding per row also attacks the bucket effect at its root: CoSA notes that rows' maxima are "scattered across blocks", so no single tile order can serve every row.

**Prior-art status: appears open.** Searches:
- "initialize online softmax running max from previous layer attention argmax key positions";
- "heavy hitter key positions reused across layers seed attention maximum [...]";
- "cross-layer reuse skip-softmax BLASST mask or running maximum from previous layer".

The only hit for a non-default running-max start was VFA, with an approximate seed.

**Experiment (simulation).**
1. In layer `l−1`, store each row's argmax key index, `s.argmax(-1)` with shape `[B, H, Sq]`: 2 MB per layer at 16K.
2. In layer `l`, gather the exact scores at the seed keys from `s` (free in the simulation) and set `running = max(cummax(tile_max), seed)`.
3. Variants: no seed (BLASST); seed source (i); (ii); (iii); and the oracle `seed = global row max` (direction E's upper bound).
4. Diagnostics:
   - **seed hit rate**: the fraction of rows whose seed equals the final row max;
   - the mean seed gap (`global max − seed`, in nats);
   - % skipped at fixed λ.
5. Accuracy: teacher-forced NLL change and RULER at 50% and 70% skipped, for Llama-3.1-8B and Qwen3-8B.
6. VLM check (hypothesis, not a known result): in Qwen2.5-VL-7B on Video-MME, the high-attention "visual sink" tokens sit in the middle of the sequence (Visual Attention Sink, 2503.03321). A static sink-first order should miss them; argmax seeding does not care about position.

**Metrics.**
- Teacher-forced NLL change (then RULER accuracy) versus % skipped, as a frontier per variant.
- % skipped at fixed λ.
- How much of the gap between BLASST and the oracle each seed source closes.

**Cost.**
- 3 engineering days, plus 1 for the VLM check.
- About 6–10 GPU-h (6 RULER/NLL configs on 2 models at about 0.85 h each, plus about 2–3 h for Qwen2.5-VL).

**Needs a real kernel?** Mostly no.
- The kernel change is to start `m_i` at the seed instead of `-inf`, plus a small gather-and-dot step before the main loop. Sparsity gains then convert to time through BLASST's own measured time-vs-sparsity line (`blasst_paper.md` Sec. 4.3).
- Sources (ii) and (iii) need the previous layer to output a per-row argmax, an extra reduction. Measuring that cost needs a kernel.

**Risk / go–no-go.** The first thing to measure, with no extra GPU, is the oracle (global row max) versus BLASST frontier at matched sparsity.
- BLASST found reordering "generally negligible" at a given sparsity, while CoSA found lower perplexity at a matched skip ratio after reordering.
- If the oracle does not beat BLASST, seeding cannot help accuracy-vs-sparsity. Its only remaining value would be raising the running max early to feed direction A.

### Direction C: smaller query groups for the vote (more aggressive)

**Idea in plain steps.**
1. Under the all-rows rule, one row with a non-negligible score vetoes all 128 rows (CoSA's "bucket effect").
2. Split the 128 rows into **query groups** of `G` rows. Each group of `G` rows x 128 keys is skipped on its own vote, for `G` = 128, 64, 32, 16 or 1.
3. Smaller `G` always skips a superset of tiles at the same λ, so compare at matched work skipped, counted in row-blocks.
4. Hardware mapping:
   - `G = 64` is the natural unit on Hopper. CoSA's FA3-style kernel has "Two consumer warp groups [...] 64 query rows each" and then deliberately ANDs them. The PV matrix multiply (WGMMA, Hopper's warpgroup-wide tensor-core instruction) is issued per warpgroup.
   - `G = 16` matches an Ampere warp (one `mma.sync`, a single warp's matrix-multiply instruction) and SpargeAttn's per-warp rule.
   - `G = 1` is VFA's Rowskip. It only masks rows; nothing is saved unless every row skips.
   - BLASST's Blackwell kernel uses one softmax warpgroup per 128-row tile, so `G = 128` is native there.

**Gap it fills.**
- CoSA names the problem ("lets a single outlier row veto the skip of an otherwise negligible block") but keeps "a single tile-wide vote".
- SpargeAttn votes per warp but never ablates the group size.
- VFA's Rowskip covers one short MMLU task and reports row-wise sparsity, which is not comparable.
- BLASST never analyses vetoes.

**Prior-art status: partially done.** Group sizes have been used (SpargeAttn per warp, VFA per row) but never measured against each other on long-context LLMs. Searches:
- "skip-softmax sparse attention finer query granularity per-warp vote all rows conservative [...]";
- "threshold-based block skipping attention query block size ablation 64 vs 128 rows";
- "bucket effect sparse attention skip rowmax query tile split warpgroup granularity".

None found a study of group size versus vetoes. The closest are block-size ablations for proxy methods (FlexPrefill, MiniMax MSA), which are about key blocks and top-k.

**Experiment (simulation).**
1. `margin_G = diff.unflatten(2, (rows // G, G)).amax(3)`, then mask per group.
2. Record margins for all `G` in one pass.
3. **Veto anatomy**: for each tile kept at `G = 128`, count the vetoing rows. Then ask whether they are special tokens (start-of-sequence, newline, chat-template tokens), sit at particular positions inside the query tile, or concentrate in particular heads.
4. Accuracy: NLL change and RULER at 50% and 70% (matched work), for Llama-3.1-8B and Qwen3-8B.

**Metrics.**
- Work skipped (row-blocks x 128 keys) at fixed λ, and the gain over `G = 128`.
- NLL change and RULER versus work skipped.
- Vetoes per kept tile (distribution).

**Cost.**
- 1.5 engineering days, plus 1 for the veto anatomy.
- About 6–8 GPU-h.

**Needs a real kernel?** Yes, for speed.
- On Hopper only `G ≥ 64` can actually skip the PV WGMMA. LiteAttention even reports that acting on the predicate immediately "would stall otherwise overlapping execution" on the H100.
- A per-warpgroup skip needs an FA3-style CUDA kernel. Triton cannot express two warpgroups voting separately inside one program, and `BLOCK_M = 64` changes the baseline efficiency.
- The simulation gives the skippable work and the accuracy; any speed number is a cost-model ceiling.

### Direction D: LSE (cumulative-energy) reference on causal LLMs (more accurate at matched sparsity?)

**Idea in plain steps.**
1. Compare the tile max against the running LSE `= m + ln ℓ` instead of the running max `m`.
   - LSE (log-sum-exp) is the log of the total exponentiated score mass seen so far.
   - `ℓ` is the running sum of exponentiated scores, which FlashAttention already keeps.
2. The LSE is always at least the running max, so this rule is more aggressive at the same λ.
3. It also bounds each element's true normalized probability, not just its weight relative to the max element. That is a more meaningful guarantee.
4. DynSparse showed 61.42% → 82% sparsity on Wan 2.2 at matched quality.
5. Variant (global budget): skip while the summed bound on skipped mass stays under ε·ℓ. This is LiteAttention's "global condition" idea, applied to a single layer.

**Gap it fills.**
- DynSparse was evaluated only on video diffusion.
- Its own mechanism suggests the size of the gain depends on how peaked attention is: "once a global maximum ("winner") is encountered, LSE rises sharply".
- Testable prediction: in LLM heads dominated by the attention sink, `ln ℓ` after the sink tile is small, so LSE ≈ max and the two rules coincide. In flat heads the LSE rule should differ. Nobody has measured this.

**Prior-art status: partially done** (DynSparse, video diffusion only). Searches:
- "DynSparse cumulative energy log-sum-exp threshold skip blocks FlashAttention large language model long-context RULER";
- "in-kernel block skipping using running softmax denominator accumulated attention mass instead of running max LLM".

No LLM study found.

**Experiment (simulation).**
1. `tile_lse = logsumexp(s_t, -1)`; take the prefix LSE up to but not including tile `j` (exclusive), computed with `logcumsumexp`; rule `t = tile_max − prefix_LSE`, with the all-rows rule.
2. Caveat: this one-pass version includes the mass of tiles that were skipped, so it is slightly more aggressive than the online rule, and its margins depend on λ.
   - Sparsity scans: use the one-pass version.
   - Accuracy runs: use a true sequential loop over key tiles. That is about 128 x 128 x 32 small steps at 16K, slower; budget about 30 s per 16K sample.
   - Measure the difference between the two on 5 samples.
3. Also measure the distribution of `ln ℓ = LSE − max` per head.
4. Test DynSparse's length-scaled threshold (`λ' = ln P + ln(B_c/L)`) against BLASST's `a/L` calibration for stability of % skipped across 4K, 8K and 16K.

**Metrics.**
- NLL change and RULER versus % skipped (50/70/80%).
- % skipped spread across lengths with one threshold.
- Per-head `ln ℓ` distribution.

**Cost.** 2 engineering days; about 6–8 GPU-h (the sequential accuracy runs are slower).

**Needs a real kernel?** No.
- `m` and `ℓ` are already in registers; the only addition is one `log` per row per tile, compared with 16K `exp` operations.
- Sparsity converts to time through BLASST's measured line.

### Direction E: visiting order, with the global row max as an upper bound (mostly done; keep as the measuring stick)

**Idea.** Measure how much sparsity the order of visiting tiles controls. Compare, at matched sparsity:
- ascending (BLASST Alg. 1);
- descending (the native order of FA2, FA3 and FA4 per CoSA, and BLASST A.4's "reverse");
- sink first, then diagonal, then the rest (VFA's SWA);
- the **oracle**: every row compared against its true global max. This is the best any order or seed can achieve under the max rule.

It is cheap and it exposes a hidden variable: BLASST's reported sparsity depends on which kernel order is used.

**Gap.**
- BLASST App. A.4 covered one length, one model, two tasks, and one alternative order, shown as an image only.
- VFA's SWA covered one short MMLU task.
- CoSA's order needs a proxy and was judged by perplexity.
- Nobody reports the oracle upper bound on RULER.

**Prior-art status: partially done** (BLASST A.4, VFA appendix, CoSA, LiteAttention's "radial-centric ordering" for video). The search "skip softmax attention tile visiting order running maximum sink diagonal first BLASST" found only these.

**Experiment.** Permute the tile axis before `cummax` and undo the permutation afterwards; oracle `running = tile_max.amax(-1, keepdim=True)`.

**Metrics.** % skipped at fixed λ; NLL change and RULER at matched sparsity; fraction of rows whose running max has reached its final value after the sink and diagonal tiles.

**Cost.** 1–2 engineering days; about 5–6 GPU-h.

**Needs a real kernel?** No. BLASST says reordering is supported "at the kernel scheduling level with negligible overhead".

**Value.** Low novelty on its own. Its oracle is the go–no-go test for direction B and the "headroom" bar for C and D.

---

## 4. Suggested 5-week order

| Week | Work |
|---|---|
| 1 | Shared margin passes with every variant recorded (directions A, B, C, E in one pass). Oracle vs BLASST frontier (go–no-go for B). Certification-rate test (go–no-go for A). |
| 2 | NLL/RULER at matched sparsity for whichever of A, B, C passed. Decode certification for A. |
| 3–4 | If A passes: Triton prototype and timing on the GH200. Otherwise deepen B and C, plus D. |
| 5 | Qwen3-8B replication, Qwen2.5-VL check for B, write-up. |

Total GPU time for all five directions in simulation: roughly 25–40 GPU-h, plus about 10 GPU-h if the Triton kernel is built. These are estimates under the assumptions in Sec. 2.

---

## 5. Final ranked table

Ranked by open x feasible x interesting.

| Rank | Direction | Gap it fills (paper + quote) | Prior-art status | Experiment | Cost | Needs real kernel? |
|---|---|---|---|---|---|---|
| 1 | **A. Certified skip**: a provable bound on the tile max (centroid + radius, or min/max box) checked against BLASST's running max; skip BMM1 and the K load; output identical to BLASST | CoSA: "The QK multiplication is therefore unavoidable, and skipping trims only the value-side steps." AMEND: "K still accounts for about 91% of its remaining logical K+V traffic." RBS uses the bound only to rank: "a loose bound can rank an irrelevant block above a relevant one." | **Appears open as a combination.** Bounds exist (Quest: box, top-K pages in decode; RBS-Attention: weakened ball bound, ranking in prefill); nobody uses them as a certificate inside BLASST's test (3 searches) | Record the certificate margin next to the margin; certification rate at 50/70% skipped; ball/box/16-key sub-blocks, with seeding, with a stale running max; decode K bytes avoided; check certified ⊆ skipped | 3–4 eng-days sim, ~2–3 GPU-h; Triton +8–10 days, ~5–10 GPU-h | **Yes for speed** (simulation gives exact certification rates and FLOPs/bytes avoided; accuracy is identical by construction) |
| 2 | **B. Exact-score seeding** of the running max (sink + own key, previous-layer argmax keys); BLASST's bound still holds | CoSA: "a rowmax can lie anywhere, not necessarily in the first (sink) or last (most-recent) block". VFA: "m-initialization is necessary to cover cases where maxima occur in middle blocks" (its seed is approximate). LiteAttention: "precision is further limited by the use of the cumulative row maximum" | **Appears open** (3 searches; only VFA's approximate seed) | Seeds gathered from `s`; seed hit rate, seed gap; % skipped at fixed λ; NLL/RULER frontier vs BLASST and the oracle; Qwen2.5-VL visual-sink check | 3–4 eng-days, ~6–10 GPU-h | Mostly no (initial `m` plus a tiny gather); argmax tracking cost needs a kernel |
| 3 | **C. Query-group vote** with `G` = 64/32/16 instead of the all-rows rule; veto anatomy | CoSA: "Its per-row AND (∧) lets a single outlier row veto the skip of an otherwise negligible block", yet it keeps "a single tile-wide vote" | **Partially done**: SpargeAttn votes per warp (no ablation); VFA Rowskip (one short MMLU task, row-wise sparsity); no long-context study (3 searches) | Margins per `G` in one pass; work skipped vs `G`; NLL/RULER at matched work; vetoing-row statistics | 2.5 eng-days, ~6–8 GPU-h | **Yes for speed** (only `G ≥ 64` can skip WGMMA on Hopper; needs FA3-style CUDA) |
| 4 | **D. LSE reference** (`tile max − running LSE`) and a summed skipped-mass budget, on causal LLMs | DynSparse (video diffusion only): "enhances sparsity from 61.42% to 82% with a VBench metric drop of less than 5%"; its gain depends on peakedness ("once a global maximum ... is encountered, LSE rises sharply") | **Partially done** (DynSparse on Wan 2.2 only; no LLM study found, 2 searches) | One-pass and sequential LSE rule; NLL/RULER at 50/70/80%; `ln ℓ` per head; % skipped stability across lengths | 2 eng-days, ~6–8 GPU-h | No (uses `m` and `ℓ` already kept; one extra log per row per tile) |
| 5 | **E. Visiting order** (ascending / descending / sink + diagonal first) plus the global-row-max **oracle** | BLASST A.4: "dataset-dependent but generally negligible"; CoSA: backends "traverse KV blocks in a fixed ascending [...] or descending [...] order" | **Partially done** (BLASST A.4 at 8K, VFA SWA on MMLU, CoSA with a proxy, LiteAttention for video); oracle on RULER not reported | Permute tiles before `cummax`; oracle; NLL/RULER at matched sparsity | 1–2 eng-days, ~5–6 GPU-h | No (reordering costs nothing in a kernel per BLASST) |

---

## 6. Could not verify

1. **tinygrad issue #14247** ("dynamic block pruning for attention (cauchy-schwarz bounds)"). Seen only in a search snippet saying it was closed as not planned; the repository has issues disabled, so I could not read it. It may be an earlier, unpublished proposal of direction A.
2. **Hopper hardware details for direction C** come from CoSA's description of its own SM90 kernel ("each consumer warp group responsible for 64 query rows") and from LiteAttention's H100 text. I did not check FA3's source. Whether a 64-row PV skip saves time without stalling the pipeline is unverified.
3. **The default KV order of NVIDIA's production Hopper prefill kernel** (`fmha_v2`) is not stated anywhere I read. BLASST's Algorithm 1 is ascending; CoSA says FA4 and FlashInfer are descending.
4. **Which model VFA's appendix Table 4 used.** It is not stated; the 0.81 dense score matches their Qwen3-30B row.
5. **Figures available only as images**: BLASST Fig. 9, CoSA Figs. 6/B/C, VFA Fig. 2. Only the numbers in their text were used.
6. **GPU-hour numbers** rest on assumed per-sample times (about 5 s at 8K, about 15 s at 16K); none were measured.
7. **How tight the bounds are** (direction A) and **how much the oracle helps** (directions B and E) are open empirical questions. Nothing in the literature predicts them for Llama/Qwen at 8K–16K.
8. **"Appears open" is limited by search.** Each was checked with only 1–3 searches, and arXiv papers from the last few weeks may be missed.

---

## Sources

- BLASST: https://arxiv.org/abs/2512.12087 (v3 HTML)
- CoSA: https://arxiv.org/abs/2607.25291
- AMEND: https://arxiv.org/abs/2609.09823
- VFA / VSA: https://arxiv.org/abs/2604.12798
- DynSparse (cumulative energy filtering): https://arxiv.org/abs/2606.16317
- Block Sparse Flash Attention: https://arxiv.org/abs/2512.07011
- SpargeAttention: https://arxiv.org/abs/2502.18137
- XAttention: https://arxiv.org/abs/2503.16428
- ProxyAttn: https://arxiv.org/abs/2509.24745
- SALE: https://arxiv.org/abs/2505.24179
- MInference: https://arxiv.org/abs/2407.02490
- LiteAttention: https://arxiv.org/abs/2511.11062
- RBS-Attention: https://arxiv.org/abs/2609.20971
- MiniMax Sparse Attention (block-size ablation, top-k): https://arxiv.org/abs/2606.13392
- ROCm/aiter BLASST Triton PR #5868 (per-tile vote, compares against the running max from before the tile; 45.0% elision on Qwen3-8B RULER-32K): https://github.com/ROCm/aiter/pull/5868
- vLLM-Omni skip-softmax design note ("the kernel-level speedup stays well under 2×"): https://docs.vllm.ai/projects/vllm-omni/en/stable/design/feature/skip_softmax/
- Visual Attention Sink (cited for the direction B VLM hypothesis): https://arxiv.org/abs/2503.03321
