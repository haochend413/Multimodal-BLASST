# One threshold, a different measurement: can one lambda treat text and video alike?

Written 2026-11-03. Desk research and reasoning only: no experiment was run and no cluster was touched.
Question from a teammate: instead of a **per-region lambda**, keep **one lambda** and change *what is measured*
(a transform of the scores, a different reference point, a different statistic), so that one lambda behaves
well on text and video. Our numbers come from `results/VLM_REGIONS.md` and `results/VLM_REGIONS_2.md`.

## 0. Bottom line

1. **Only one family attacks our finding head-on:** judge a video tile against the best *video* key seen so far,
   not against the sink and transcript (**own-modality reference**, candidate 1). Done as a **cross-modal-first
   visiting order** it costs nothing in a kernel, keeps BLASST's guarantee, and reduces exactly to BLASST on text.
2. **Second choice: a "useful-key" reference** (6b): keys with near-zero value vectors (the sink) are left out of the
   reference. A sink exclusion that sets itself from the data, in every region.
3. **Per-region lambda in disguise:** a per-modality score shift or temperature in the skip test (exactly), and
   per-head z-scores (per-head lambda). Per-modality key centering also *breaks* the bound. Inside the real
   softmax, any of these changes the model.
4. **Mass rules** (LSE reference, tile mass, skipped-mass budget) target the *total* weight lost, not T->V. Text
   rows really do put little mass on each video tile, so I expect them to skip T->V at least as much as now.
   Useful as yardsticks, not as the fix.
5. **Honest caveat:** even the best one-threshold design may pick almost the *same* tiles as a well-tuned
   per-region lambda (exact test in Section 3). Then its value is self-calibration: no offline lambda table, and
   it adapts to the layout. Our region gap moves from 16 to 40 points with layout, which suggests that matters.

## 1. Words

- Ours: **tile**, **skip rule**, **lambda**, **margin**, **% skipped**, **dense**, **calibrated**, **running max**,
  regions **T->T / T->V / V->T / V->V**, **weight lost** (dense softmax weight sitting on skipped tiles).
- **Reference**: the number a tile's row max is compared against. BLASST's reference is the running max m_i.
- **Row difference** d_ij = (tile row max M_ij) - m_i <= 0, so margin_j = max over the 128 rows of d_ij.
- **Offset**: what a design adds to d_ij. New rule: skip iff max_i (d_ij + c_ij) < ln lambda.
- **Sink**: a token (usually the first) that takes a large share of attention without carrying content.
  **Massive activation**: a few hidden-state coordinates hundreds of times larger than the rest.
- **Value norm** ||v_k||: length of key k's value vector. Large weight with ||v|| ~ 0 adds almost nothing.
- **LSE** (log-sum-exp) = m_i + ln(l_i), with l_i FlashAttention's running softmax denominator (relative to m_i).
- **Oracle**: a rule using information a one-pass kernel cannot have (e.g. the final row max); only for measuring.

## 2. (A) Why text rows score low on video keys: the evidence

**Verified facts** (read on the arXiv abstract/HTML pages, 2026-11-03):

| Mechanism | Source | What it shows |
|---|---|---|
| Attention sinks | StreamingLLM, 2309.17453 (ICLR 2024) | Initial tokens draw strong attention with little meaning; keeping their KV "will largely recover" window attention. |
| Sinks from massive activations | Massive Activations, 2402.17762 (COLM 2024) | A few huge activations act as fixed biases and concentrate attention on their tokens; also in ViTs. |
| Sinks are no-ops | When Attention Sink Emerges, 2410.10781 (ICLR 2025 spotlight) | The sink "acts more like key biases, storing extra attention scores, which could be non-informative and not contribute to the value computation". |
| Sink values are tiny | VATP, 2406.12335 (EMNLP 2024) | LLaMA2-7B sink value norms are "much smaller than other tokens", ~0 in layers 3-31; VATP must force-keep the first 20-40 tokens. |
| Weight is not contribution | Kobayashi et al., 2004.10102 (EMNLP 2020) | Output depends on weight x norm of the transformed value; seen this way, BERT "pays poor attention to special tokens". |
| Little attention on images, deep layers | FastV, 2403.06764 (ECCV 2024 oral) | Deep layers: per token, image tokens get 0.21% of the score system-prompt tokens get (50% in the first two layers); system prompt takes 85% of attention while image tokens are 64% of the input; "anchor tokens are not image tokens". |
| Visual attention sinks | Kang et al., 2503.03321 (ICLR 2025) | Some visual tokens get high attention even when irrelevant, via massive activations in the *same* sink dimensions as text sinks (tau = 20); masking them barely matters, masking random visual tokens hurts. Tested on Qwen2-VL-7B, not 2.5-VL. |
| Visual sinks in Qwen2.5-VL | SinkPruner, 2609.01004 (EMNLP 2026 Findings) | High-norm visual tokens are redundant yet kept by pruners; its sanitizer "alleviates attention sink and attention dispersion"; evaluated on Qwen2.5-VL. |
| Norm gap | The Unseen Bias, 2512.08374 (preprint) | Qwen2.5-VL visual tokens after the projector: norm 56.88 vs 0.86 for text embeddings. Argues slow visual updates leave related pairs misaligned in angle, so text-to-image attention spreads out. |
| Modalities stay apart | The Narrow Gate, 2412.06646 (NeurIPS 2025) | Native multimodal VLMs keep image and text "more separated within the residual stream"; adapter VLMs (like Qwen2.5-VL) pass image information through many tokens. |
| Distance decay | CCA, 2410.15926 (NeurIPS 2024) | RoPE long-term decay: models "hallucinate more when relevant visual cues are distant from instruction tokens". |
| Video positions follow time | Qwen2.5-VL report, 2502.13923 | Temporal M-RoPE component aligned "with absolute time". |
| Where image information flows | Cross-modal Information Flow, 2411.18620 (CVPR 2025) | LLaVA: image -> question tokens in lower and middle layers, -> last position in higher layers. |
| Low image attention is not optimal | PAI, 2407.21771 | Training-free amplification of image-token attention reduces hallucination ("text inertia"); changing per-modality attention changes outputs. |

**My inference: how these produce our numbers.**
- A text row's running max is set before the first video key arrives: by the sink, then by transcript part 1.
  Video keys rarely beat it (1.7% of T->V tiles have margin 0), so the median T->V margin is -4.85 vs -2.04 for V->V.
- What pushes video scores down: the modality gap in *direction*, not norm (Qwen2.5-VL is pre-norm, so RMSNorm removes
  the norm gap before Q/K/V; *The Unseen Bias* does not reconcile this), distance decay, and FastV-style disinterest
  in deep layers.
- So the bar for a video tile is set by another modality's keys, some of them no-ops (the sink). That is about the
  **reference**, not the threshold, which is why a reference fix with one lambda is plausible at all.
- It also explains the layout effect (gap 40 / 33 / 16 points for A / B / C): more text before or around the video
  means a higher bar for video keys.
- To check in our pipeline (unverified): the M-RoPE gap between question and early frames depends on the fps we pass
  to the processor, because temporal IDs follow seconds.

## 3. When is a "one-threshold" design just per-region lambda? (the disguise test)

Every design below has the form: skip tile j iff max_i (d_ij + c_ij) < ln lambda.
- **Exact disguise:** if c_ij = c_r for every row and tile of region r, the rule is margin < ln lambda - c_r, i.e.
  per-region lambda with lambda_r = lambda * exp(-c_r). The same holds for any increasing f_r applied to all rows
  alike, since max_i f_r(d) = f_r(max_i d). Example: a per-region temperature d/tau_r gives lambda_r = lambda^tau_r.
- **Genuinely different** only if c_ij varies across a tile's rows or across a region's tiles. Then the design
  changes *which* tiles inside a region are skipped, not just how many.
- **Self-calibrating but equivalent:** varying offsets can still pick nearly the same tiles as some per-region lambda.
  **Test:** take the design's per-region % skipped, build the per-region-lambda rule with the same % per region
  from saved margins, and report per region the Jaccard overlap (shared / union) of skipped sets and the difference
  in weight lost. Jaccard ~1 = "automatic per-region lambda"; lower weight lost = "genuinely better".
- **Softmax facts used below:** a per-row constant changes nothing (softmax and d_ij both, as M and m shift together).
  A per-key or per-modality constant inside the real softmax changes the model (no longer sparsity); used only in the
  skip test, it changes only the decisions.
- **Exact error of skipping** (row i, skipped set S, delta = weight lost, o = dense output, õ = output over kept keys,
  ō_S = weighted mean of skipped values): `õ - o = delta * (õ - ō_S)`, from o = (1 - delta) õ + delta ō_S.
  Error = weight lost x how far the skipped values sit from the kept output. Skipping a sink (ō_S ~ 0, large delta)
  is catastrophic; skipping redundant video keys whose values resemble the output is cheap.
- **"BLASST's guarantee" at our lambdas:** each skipped key's weight < lambda x the max key's, so a tile holds
  < 128 lambda. That is above 1 (says nothing) once lambda > 1/128 = 0.0078: all our Qwen lambdas (0.08-0.23) and
  Llama at 8K (0.013). Only Llama 16K (0.0052) gives a non-trivial per-tile bound (0.67). So "keeps a BLASST-style
  bound" is a formal property; **weight lost is what actually measures damage.**

## 4. (B) Candidates

**1. Own-modality reference.** *Plain words:* compare a video tile with the best video key this row has seen so far,
and a text tile with the best text key.
- Exact version: two running maxes per row (m_i^T, m_i^V); offset c_ij = m_i - m_i^{mod(j)} >= 0.
- **Free version, cross-modal-first order:** for each query tile, visit the other modality's key tiles first, then
  its own. Text rows then judge video tiles against the video-only max, and the following text tiles against
  max(video, text), about the text max for them. In layouts A-C this changes only T->V (video rows already see
  the earlier text first, or have none).
- (i) *Disguise?* Not exactly: the offset varies per row and tile and is set at runtime from token types alone, no
  tuning. It nears per-region lambda if every text row has a similar text-minus-video gap (unknown; Jaccard test).
- (ii) *Bound:* kept, tighter (m^V <= m): a skipped key's weight < lambda x the best *same-modality* key so far.
- (iii) *Kernel:* a per-query-tile permutation of key-tile indices, an index lookup like a paged-KV block table
  (BLASST App. A.4 already ran a reversed order); keep the causal diagonal tile last. Two-register version: +1
  register per row, plus a masked max on the 1-2 mixed tiles per query tile.
- (iv) *Prediction:* at fixed lambda, T->V % skipped falls toward V->V's, V->V unchanged, and the T->V minus V->V gap
  varies much less across layouts A/B/C (the transcript no longer sets the bar).
- Risk 1: with time-ordered positions and distance decay, later frames may score higher for the question. In
  ascending order the video max keeps rising, many tiles are new maxima, all kept. Descending order (latest frames
  first) avoids this; record both.
- Risk 2: a visual sink early in the video sets the bar too high again. Fix: combine with 6b.

**2. Sink-excluded reference.** Always keep the sink tile; leave the first few tokens out of the running max.
- (i) Per-row offset m_i - m_i^{-sink}: large in sink-dominated rows, 0 otherwise; not a disguise. (ii) Bound kept,
  tighter. (iii) Free as a "sink-last" order, or a masked max on the first tile only.
- (iv) Helps every region whose bar is the sink, V->V included. In layout A the T->V bar is still transcript part 1,
  so the gap mostly stays; in C it should shrink more.
- Weaker than 6b: hand-picked sink position, and it misses visual sinks inside the video.

**3. Per-modality score shift, temperature, or key centering.** *Verdict: per-region lambda in disguise, or unsafe; drop.*
- A fixed shift c_V in the skip test is exactly per-region lambda when a text key holds the max (nearly otherwise),
  and it needs an offline value, the very tuning we wanted to avoid.
- Centering only video keys (k - mu_V) shifts row i's score by -q_i . mu_V / sqrt(d): per row and data-set, but it
  can be positive, making the effective lambda lambda * exp(q . mu_V / sqrt(d)). **No bound.**
- One global key mean is a per-row constant: no effect. Inside the real softmax, any of these is a model change (PAI).

**4. Mass-based statistics.**
- (a) **LSE reference:** skip iff max_i (M_ij - m_i - ln l_i) < ln lambda (Direction D in `directions_idea1.md`;
  DynSparse 2606.16317 did it for video diffusion). Offset -ln l_i, per row and growing: skips more where mass is
  spread out. Bound: normalized weight p < lambda per key, as BLASST. Cost: one log per row per tile.
- (b) **Tile mass share:** skip if Z_ij / l_i < eps for every row (Z_ij = the tile's mass). Bound: skipped tile
  weight < eps. Cost: needs the exp of the whole tile, the work BLASST saves (prefill skips softmax and PV but still
  loads V), so only PV is saved. The cheap bound 128 exp(M - m) turns it back into (a) with lambda = eps/128.
- (c) **Per-row budget:** skip while (mass already skipped + bound on this tile) < eps * l_i. The only design that
  *bounds weight lost directly*: <= eps per row, since l only grows. One register per row. Top-p rules need the
  whole row's mass first, i.e. a prediction step, not BLASST's single pass: FlexPrefill 2502.20766 (ICLR 2025),
  Twilight 2502.02770 (NeurIPS 2025), and ZipVL 2410.08584 for VLM tokens.
- (iv) Text rows put only 4.7-6.5% of their attention on 47-110 video tiles, so each holds very little of their mass.
  Mass rules will call it negligible: T->V skipped as much or more. (c) cuts the long tail (13% of question rows
  lose > 10%) by capping every row, not by sparing video.
- A variant dividing by the *useful* (non-sink) mass is better motivated: by Section 3, the relative output error is
  about (weight lost) / (1 - sink weight).

**5. Final-row-max reference.** Use the row's true max m_i^final instead of the running max.
- (i) Offset m_i - m_i^final <= 0, varying per row and tile; not a disguise. (ii) Cleanest bound: p < lambda x the
  row's largest weight, whatever the order.
- (iii) Exact version needs a second QK^T pass (about half the dense work), defeating the purpose. Exact-score seeding
  (Direction B: sink, own key, previous layer's argmax) approaches it from below and keeps the bound.
- (iv) Removes order artifacts: margin-0 tiles become skippable (47% of T->T, 37% of V->T), so those regions skip
  more. T->V changes only if text rows' final max comes later (transcript part 2 or the question): unclear, measure.
- *Use as oracle and diagnostic, not as the fix.*

**6. Output-contribution statistics.**
- **6a. Weight x value norm in the tile test** (max_k of s_ik + ln||v_k||): can skip a *large-weight*, small-||v|| key
  such as a mid-video visual sink, which shrinks the softmax denominator and inflates the whole output (Section 3).
  **Reject.**
- **6b. Useful-key reference.** Tile test stays the plain max M_ij; the *reference* is the running max over keys with
  non-negligible value norm, e.g. ||v_k|| >= 10% of the head's median (fix the cut once, beforehand).
  - (i) Per-row offset set from the data; catches the text sink and any visual sinks with small values (shown for
    text sinks in LLaMA2 by VATP; unverified for Qwen2.5-VL). (ii) Bound kept, tighter: skipped weight < lambda x
    the best *useful* key. (iii) ln||v|| precomputed once per layer (O(N d)); per tile, one extra masked row max, no
    exp. (iv) Like 2 in every region without a hand-picked position. With 1: "own-modality useful-key reference".

**7. Per-row or per-head z-scores.** Skip iff d_ij / sigma_i < -kappa, i.e. per-row lambda_i = exp(-kappa sigma_i).
In flat rows (small sigma), lambda_i nears 1 and keys almost as large as the max get skipped, exactly where many keys
matter. Per-head sigma fixed offline is per-head lambda (an existing direction). Two extra reductions per tile.
*Verdict: drop.*

**Summary.**

| # | Design | Per-region lambda in disguise? | Bound | Kernel cost | Effect on T->V vs V->V |
|---|---|---|---|---|---|
| 1 | Own-modality reference / cross-modal-first order | No (runtime, per row); may *act* like it | Kept, tighter | Free (order) | Gap shrinks; layout-robust |
| 2 | Sink-excluded reference | No | Kept | Free | Both kept more; gap stays in A |
| 3 | Modality shift / temperature / centering | **Yes** (shift, temperature); centering unsafe | lambda_r / none | Tiny | Whatever you tune |
| 4 | LSE / tile mass / budget | No | p < lambda; tile < eps; row <= eps | LSE free; exact mass pays exp | T->V skipped as much or more |
| 5 | Final max | No | Cleanest | Two passes (oracle only) | T->T, V->T skip more; T->V unclear |
| 6b | Useful-key reference | No | Kept | ~1 masked max per tile | Like 2, automatic |
| 7 | z-score | Per-head yes; per-row no | Unsafe in flat rows | 2 reductions | Unclear |

## 5. (C) How to test the top candidates cheaply

**Data already there.** Saved margins (10 Qwen prompts) give only the BLASST margin; every new rule needs per-row values
the margins have lost. The 5.4 GB per-row arrays are probably per-row totals at the two saved lambdas (my reading of
their size: ~135 MB per prompt, a few numbers per layer, head and row; check `vlm2_weights.py`). If so, they cannot
score a new rule. So: **one new dense pass with extra recording.**

**Step 1: one recording pass (GPU, ~0.5 GPU-h).** The same 5 videos in layouts A, B, C at 8K and 16K: 30 prefills.
- Record in fp16, per (layer, head, query tile, key tile):
  - (a) BLASST margin (sanity check against the saved one);
  - (b) own-modality margins: two-register, cross-modal-first ascending and descending, own-modality *final*-max oracle;
  - (c) useful-key (6b) and sink-excluded (2) margins; (d) global final-max margin (5); (e) LSE margin (4a, one pass);
  - (f) **tile mass** S = sum over the tile's 128 rows of each row's dense weight on its keys. Mean weight lost for
    *any* skip set is a sum of S, so this one tensor scores every rule;
  - (g) S weighted by ||v|| (a proxy for output error).
- Per row: tile weights for the question rows and the answer position (row-group weight lost and tails).
- Diagnostics: for each text row after the video, which block holds the running max when the video starts (first
  block, other text, none); value norms by token type (first 4 tokens, other text, video, video tokens over the
  tau = 20 massive-activation criterion).
- Size: ~13 MB per tile tensor at 16K (28 x 28 x 8,256 tiles x 2 bytes), ~3 GB in all. Time: 3 s / 11 s dense, assume
  2-3x with recording, ~15 min on `ghx4`, plus interactive debugging (charged x2).

**Step 2: offline analysis (CPU, no GPU-hours).** Per prompt, set lambda (or eps) for 50% and 75% of all tiles
skipped. Report per-region % skipped, mean weight lost (all rows, from S), question-row and answer-position weight
lost, share of question rows losing > 10%, the **disguise test** (Section 3), and the **layout test**
(question->video minus V->V gap in A / B / C; one lambda gives 16-40 points). Compare against four yardsticks:
1. one lambda;
2. **best per-region lambda**: per-region thresholds minimizing mean weight lost at 50% overall (greedy over regions,
   using S in margin order);
3. "T->V never skipped";
4. **mass oracle**: skip the 50% of tiles with the smallest S, the lowest weight lost *any* tile rule can reach on the
   dense pass. If it barely beats the best per-region lambda, no statistic can beat per-region lambda on weight lost,
   and the question is calibration alone.

Text control: on text the own-modality rule *is* BLASST (no regression to test). Rules 2, 4, 5 run on the saved
Llama Q/K (~0.1 GPU-h); 6b needs V, not saved (small Llama pass, ~0.05 GPU-h).

**Step 3: accuracy for the winner only (GPU, ~1.5 GPU-h).** Implement it live in `blasst_regions.py` (order version =
key-tile permutation; 6b = ln||v|| per key). Reuse the 90 Video-MME pilot questions (layout C, 16K) and their dense
(65/90), one-lambda (61/90) and T->V-never (64/90) arms. New arms: the winner at 50%, plus the winner and one lambda at
75% (more answers change). Each arm ~18 min live + ~18 min dense margins to pick its lambda: ~0.5-0.6 GPU-h. Only 3-5
of 90 answers changed last time, so accuracy is a sanity check; the decision rests on Step 2.

**Total: about 2-2.5 GPU-hours** (project so far: about 4.4).

**Decision rule (write down before Step 2).** At 50% on the dense pass, adopt the one-threshold design as the main
method if: its question-row and all-row weight lost are no worse than the best per-region lambda; its layout gap varies
less than one lambda's (16-40 points); and Step 3 shows no loss against one lambda. Otherwise per-region lambda stays
the main method and this becomes an ablation paragraph.

## 6. Recommendation

- **Test candidate 1 (own-modality reference)** as the cross-modal-first order (ascending and descending) plus the exact
  two-register version. **Add 6b (useful-key reference)** as the sink-robust partner and test the 1 + 6b combination.
- Use the mass oracle and best per-region lambda as yardsticks; run the row budget (4c) only if the oracle shows real
  room inside regions. Drop 3 and 7. Keep 5 as a diagnostic (its exact-seed version is Direction B).
- **Story.** Per-region lambda is simple and direct, but needs offline per-region (and, from our layout check, probably
  per-layout) calibration. The own-modality reference uses one lambda, costs nothing in a kernel, keeps BLASST's
  guarantee, is BLASST on text, and has a mechanism: cross-modal tiles are judged against another modality's keys,
  no-op sinks included.
  - If it matches the best per-region lambda and holds across layouts, it is the **stronger** story, even if the
    disguise test shows mostly the same tiles. Then say it plainly: "a free, self-calibrating per-region lambda".
  - If it does not close the gap, it is the **weaker** one and per-region lambda stays.
- Skipping T->V may be harmless (FastV), and our accuracy evidence is weak (64 vs 61 of 90, p = 0.25). The case rests
  on weight lost: 52-57% of the text rows' video weight is removed, 69-72% at the answer position.
- For tomorrow's mid-semester report (Nov 4), this is at most one "next step" sentence.

**Sources not re-checked this session** (from our earlier notes): DynSparse 2606.16317; LiteAttention 2511.11062;
MMInference 2504.16083 (ICML 2025); BlindSight 2507.09071. These are the closest modality-aware sparse attention work,
and both choose offline patterns, not a threshold. One search found no paper with a per-modality or sink-excluded
*reference* for threshold-based tile skipping: a hint that the direction is open, not proof.
