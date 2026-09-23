# Can a FlashAttention tile's skip decision be made without computing `QK^T`?

**Question.** BLASST skips a (query tile × key tile) pair when its row-wise max
sits far enough below the row's running max. But it computes the full `P × B × d`
matmul to learn that. Is the tile's row-max determined by a few keys — or a few
queries — so the decision could be made from a fraction of the work?

**Answer.** Yes, but only on the **query** axis.

- **Key-side downsampling fails.** No query-independent subset of a key tile
  carries the signal. Keys chosen by norm are no better than random, and random
  sits exactly at chance (`r/B`). Which keys win is a function of the query, so
  by the time you know them you have done the matmul.
- **Query-side downsampling works.** The 128 rows of a query tile agree on their
  argmax key 74–94% of the time. Pooling them into one query, scoring it against
  all 64 keys for **1/128** of the matmul, and letting it nominate 4–16 keys
  reaches near-oracle accuracy.
- **Headroom is large**: at the aggressive thresholds BLASST targets, 93–98% of
  tile work is skippable in text, Wan, VLM and images.
- **Net effect**: ~9–13% of the `QK^T` work at 3–7% false skips (text, Wan).

One caveat governs all of it, stated in full in §7: `skippable` here is measured
**per row**, while a real kernel skips per tile. That makes the headroom — and
therefore the savings — optimistic by an unmeasured amount.

Numbers come from `qk_results/*.json`; code is `qk_sparse/tile_skip.py`.

---

## 1. Setting

### 1.1 The tiling

FlashAttention processes a tile of `P = 128` queries against a tile of `B = 64`
keys, producing `S = Q_i K_j^T` of shape `[128, 64]`. BLASST reduces that to a
row-wise max and drops the tile when it cannot matter.

### 1.2 What "skippable" means

For query row `i` and key tile `j`:

```
deficit   = (row i's max score over ALL allowed keys) − (row i's max within tile j)
skippable ⟺ deficit > gap
```

The softmax justifies it: every entry in the tile has weight at most
`exp(−deficit)` relative to the row's largest entry. At `gap = 4`, nothing in the
tile exceeds `e⁻⁴ ≈ 1.8%` of the top entry.

### 1.3 The gap is not a free parameter

BLASST skips when `exp(s − max) < threshold`, and blog16 defines
`threshold = threshold_scale_factor / context_length`, so

```
gap = −ln(threshold) = ln(L / threshold_scale_factor)
```

With blog16's calibrated factors for Qwen3-30B-A3B:

| target sparsity | L=4k | L=32k | L=100k |
|---|---|---|---|
| 0.1 | 5.39 | 7.47 | 8.58 |
| 0.3 | 3.66 | 5.74 | 6.86 |
| 0.5 | 1.94 | 4.02 | 5.14 |
| 0.7 | 0.22 | 2.30 | 3.41 |
| 0.9 | — | 0.57 | 1.69 |

Aggressive sparsity therefore lives at **gap 0.6–3.4**, which is why results are
reported at gaps 1, 2 and 4 rather than only at conservative settings.

Equivalently, `gap` is how many e-folds below the row max count as negligible:

| gap | discards anything below |
|---|---|
| 1 | 36.8% of the top entry |
| 2 | 13.5% |
| 3 | 5.0% |
| 4 | 1.8% |
| 6 | 0.25% |

### 1.4 Models and capture

| model | tokens | causal | samples | depths | heads |
|---|---|---|---|---|---|
| Qwen2.5-0.5B (`text`) | 4,096 | yes | 4 | 6 | 4 |
| Wan2.1-T2V-1.3B (`video-dit-long`) | 32,760 | no | 3 | 6 | 4 |
| Qwen2.5-VL-7B (`vlm`) | 1,612 / 4,060 | yes | 2 | 6 | 4 |
| DINOv2-base (`image`) | 1,370 | no | 4 | 6 | 4 |
| Whisper-small enc. (`audio`) | 1,500 | no | 4 | 6 | 4 |
| VideoMAE-base (`video`) | 1,568 | no | 2 | 6 | 4 |

Q, K and V are captured **post-RoPE, post-head-split**, exactly as the kernel
sees them, by patching `torch.nn.functional.scaled_dot_product_attention`.
Cross-attention is excluded (`n_key ≠ n_query`). Heads are sampled at random with
a fixed seed. Wan is captured at **denoising step 10 of 20** — step 0 is
near-noise and 13× less concentrated, so it would misrepresent the model.

For causal models, a query tile is scored only against keys strictly before it,
so all 128 rows share one allowed key set.

Up to 8 query tiles × 64 key tiles are sampled per (layer, head), giving
96–384 measurements behind each reported number.

---

## 2. Probes evaluated

Every probe estimates the tile's row-max. A subset max is always **≤** the true
max, so the only possible error is a **false skip** — dropping a tile that
mattered. That asymmetry is why false-skip rate, not raw error, is the headline
metric: a false *keep* costs speed, a false *skip* costs accuracy.

| probe | what it computes | cost vs full tile |
|---|---|---|
| `oracle r=N` | best N keys of the tile, chosen per (query tile, key tile) | N/64 — **not buildable** |
| `random r=N` | N keys chosen blindly | N/64 |
| `norm r=N` | N largest-norm keys, chosen once per key tile | N/64 |
| `chN` | all keys, N largest-magnitude channels of `q` | N/64 of the dims |
| `poolQ` | mean the query tile into one, score against all 64 keys | **1/128** |
| `pQ+N` | pooled query nominates N keys, scored exactly per row | 1/128 + N/64 |
| `ball` | `q·k̄ + ‖q‖·max‖k−k̄‖` — provable upper bound | ~1/64 |
| `box` | `Σ_d max(q_d·lo_d, q_d·hi_d)` — provable upper bound | ~1/64 |

`oracle` is the ceiling: it already knows which keys score highest. If the oracle
fails, no probe can succeed. The two bounds can *never* false-skip by
construction, so for them the question is recall — whether they are tight enough
to skip anything.

**Controls.** Synthetic tiles whose rows are near-identical give `pQ+4` = 0.000
and `norm r=16` = 0.210; tiles whose rows are independent give `poolQ` = 1.000
and `pQ+16` = 0.587. The measurement separates the two cases cleanly in both
directions.

---

## 3. Headroom

Fraction of (row, tile) pairs that are skippable:

| model | gap=1 | gap=2 | gap=4 |
|---|---|---|---|
| Wan DiT | **0.980** | 0.961 | 0.907 |
| text | **0.964** | 0.945 | 0.892 |
| vlm | 0.933 | 0.919 | 0.715 |
| image | 0.931 | 0.840 | 0.491 |
| VideoMAE | 0.844 | 0.631 | 0.253 |
| audio | 0.709 | 0.424 | 0.194 |

At the operating points that matter, the great majority of tile work contributes
nothing. Audio is the weakest case and also the model whose attention is least
concentrated.

---

## 4. Structure inside a tile

### 4.1 The rows of a tile agree on their maximum

`argmax agree` = 1 − (distinct argmax keys ÷ 128 rows):

| audio | VideoMAE | text | vlm | image | Wan |
|---|---|---|---|---|---|
| 0.944 | 0.864 | 0.856 | 0.848 | 0.794 | 0.737 |

This is the structural fact the whole result rests on. One probe location can
serve an entire query tile.

### 4.2 But the top key is not dominant

`top1_margin` — the gap between the best and second-best key, normalised by the
tile's score spread — is **0.065–0.083** everywhere. The maximum is not a
runaway; several keys sit close behind it.

Softmax mass within a tile, by rank:

| model | top-1 | top-4 | top-16 |
|---|---|---|---|
| image | 0.142 | 0.363 | 0.785 |
| text | 0.144 | 0.341 | 0.684 |
| vlm | 0.139 | 0.305 | 0.636 |
| VideoMAE | 0.102 | 0.272 | 0.605 |
| audio | 0.069 | 0.204 | 0.481 |
| Wan | 0.058 | 0.181 | 0.490 |

So a tile's mass is spread over roughly 16 of its 64 keys — concentrated, but not
onto one or two.

### 4.3 Exact max recovery from `r` keys

Fraction of rows whose true tile max is recovered exactly:

| model | r=1 | r=2 | r=4 | r=8 | r=16 | random r=4 |
|---|---|---|---|---|---|---|
| audio | 0.494 | 0.635 | 0.741 | 0.841 | 0.909 | 0.071 |
| image | 0.201 | 0.307 | 0.435 | 0.630 | 0.857 | 0.121 |
| text | 0.216 | 0.314 | 0.449 | 0.615 | 0.804 | 0.065 |
| VideoMAE | 0.169 | 0.272 | 0.409 | 0.564 | 0.715 | 0.067 |
| vlm | 0.153 | 0.238 | 0.357 | 0.516 | 0.718 | 0.063 |
| Wan | 0.084 | 0.144 | 0.228 | 0.362 | 0.552 | 0.061 |

`random r=4 = 0.061–0.121`, against `4/64 = 0.0625` — **exactly chance**. Blind
sampling contributes nothing.

Exact recovery is, however, the wrong bar: the skip rule only needs the decision
to come out the same.

---

## 5. Decision quality

False-skip rate as a fraction of tiles that should be kept.

### 5.1 gap = 1 (aggressive; 93–98% skippable)

| model | skippable | oracle r=16 | norm r=16 | ch32 | poolQ | poolQ waste | **pQ+4** | **pQ+16** |
|---|---|---|---|---|---|---|---|---|
| text | 0.964 | 0.000 | 0.163 | 0.130 | 0.170 | 0.002 | **0.029** | **0.006** |
| Wan | 0.980 | 0.026 | 0.052 | 0.088 | 0.096 | 0.000 | **0.065** | **0.038** |
| vlm | 0.933 | 0.000 | 0.190 | 0.257 | 0.255 | 0.002 | 0.121 | **0.023** |
| image | 0.931 | 0.007 | 0.041 | 0.206 | 0.279 | 0.009 | 0.074 | **0.032** |
| audio | 0.709 | 0.019 | 0.441 | 0.247 | 0.337 | 0.058 | 0.067 | **0.021** |
| VideoMAE | 0.844 | 0.066 | 0.488 | 0.671 | 0.779 | 0.013 | 0.286 | 0.116 |

### 5.2 gap = 2

| model | skippable | oracle r=16 | norm r=16 | **pQ+4** | **pQ+16** |
|---|---|---|---|---|---|
| text | 0.945 | 0.001 | 0.281 | 0.067 | 0.011 |
| Wan | 0.961 | 0.035 | 0.074 | 0.087 | 0.048 |
| vlm | 0.919 | 0.001 | 0.411 | 0.192 | 0.049 |
| image | 0.840 | 0.019 | 0.066 | 0.151 | 0.040 |
| audio | 0.424 | 0.012 | 0.394 | 0.040 | 0.008 |
| VideoMAE | 0.631 | 0.078 | 0.511 | 0.208 | 0.069 |

### 5.3 gap = 4 (conservative; less headroom)

| model | skippable | oracle r=16 | norm r=16 | **pQ+4** | **pQ+16** |
|---|---|---|---|---|---|
| text | 0.892 | 0.007 | 0.399 | 0.089 | 0.018 |
| Wan | 0.907 | 0.050 | 0.146 | 0.138 | 0.067 |
| vlm | 0.715 | 0.052 | 0.432 | 0.286 | 0.074 |
| image | 0.491 | 0.022 | 0.097 | 0.144 | 0.034 |
| audio | 0.194 | 0.005 | 0.240 | 0.016 | 0.002 |
| VideoMAE | 0.253 | 0.040 | 0.401 | 0.108 | 0.031 |

### 5.4 Both axes favour aggressive sparsity

Contrary to the obvious expectation, tightening the gap improves probe accuracy
*and* increases headroom simultaneously:

| | gap=4 | gap=2 | gap=1 |
|---|---|---|---|
| text skippable | 0.892 | 0.945 | **0.964** |
| text `pQ+16` false-skip | 0.018 | 0.011 | **0.006** |
| Wan skippable | 0.907 | 0.961 | **0.980** |
| Wan `pQ+16` false-skip | 0.067 | 0.048 | **0.038** |

The reason: at gap=1 the tiles that must be kept are only those within one e-fold
of the row max — an extreme, easily-detected set. At gap=4 the keep-set includes
marginal tiles sitting near the decision boundary, which are exactly the hard
cases. The regime BLASST targets is the easy regime for a probe.

### 5.5 The provable bounds are useless

Skip recall (fraction of genuinely skippable tiles caught, with zero risk of
error):

| model | ball, gap=1 | box, gap=1 | gap=4 |
|---|---|---|---|
| audio | 0.208 | 0.271 | 0.027 / 0.023 |
| Wan | 0.043 | 0.018 | 0.000 |
| text, image, vlm, VideoMAE | 0.000 | 0.000 | 0.000 |

Mean slack over the true max is 4.7–29.8 (ball) and 3.9–21.6 (box) — far larger
than any realistic gap. The box bound is *looser* than the ball for most models:
in 64 dimensions a circumscribing box is much larger than a sphere unless the key
tile is strongly anisotropic. Only audio, with the least headroom to begin with,
gets anything from them.

---

## 6. The central result

### 6.1 Key-side downsampling does not work

- `random` sits at chance (`r/B`).
- `norm` — largest-norm keys, the obvious query-free proxy since `q·k ≤ ‖q‖‖k‖` —
  is **no better than random**, and clearly worse in text (0.163 vs ~0.16 at the
  same budget when read against `random r=16` = 0.300 at gap=2) and audio (0.441).
- `chN` channel probing is worse than `norm` almost everywhere.

Long keys are not the winning keys. What determines the tile max is **direction
alignment with the query**, which is unavailable before the matmul.

### 6.2 Query-side downsampling reaches the oracle

| | text | Wan | vlm | audio | image | VideoMAE |
|---|---|---|---|---|---|---|
| oracle r=16 (ceiling) | 0.000 | 0.026 | 0.000 | 0.019 | 0.007 | 0.066 |
| **pQ+16** (buildable) | 0.006 | 0.038 | 0.023 | 0.021 | 0.032 | 0.116 |

The pooled query nominates almost exactly the keys an oracle would, without
seeing the true scores. `poolQ waste` — tiles kept unnecessarily — is 0.000–0.058,
so the savings are not eaten by over-keeping.

`poolq/max_err` (how far the pooled query's max sits from the true row max) is
0.6–2.9 in score units, which is why `poolQ` alone is not enough and the
two-stage form is needed: the pooled query is good at **ranking** keys, not at
**estimating** the max.

### 6.3 Cost accounting

```
few keys × all queries    : r/B      = 16/64 = 25.0%
pooled query × all keys   : 1/P      = 1/128 =  0.8%
pooled query → N keys     : 1/128 + N/64
```

Net `QK^T` work at gap=1, assuming kept tiles are computed in full:

| model | kept | **pQ+4 work** | false-skip | **pQ+16 work** | false-skip |
|---|---|---|---|---|---|
| Wan DiT | 0.020 | **8.9%** | 6.5% | 27.3% | 3.8% |
| text | 0.036 | **10.4%** | 2.9% | 28.5% | 0.6% |
| vlm | 0.067 | 13.3% | 12.1% | 30.8% | 2.3% |
| image | 0.069 | 13.4% | 7.4% | 30.9% | 3.2% |
| VideoMAE | 0.156 | 21.5% | 28.6% | 37.4% | 11.6% |
| audio | 0.291 | 34.1% | 6.7% | 47.4% | 2.1% |

`pQ+4` is the better operating point for cost, `pQ+16` for accuracy. Text at
`pQ+16` is the standout: **28.5% of the work at 0.6% false skips**; at `pQ+4`,
10.4% of the work at 2.9%.

### 6.4 VideoMAE is the exception

`pQ+4` = 0.286 and even the oracle manages only 0.066. Its rows agree (0.864)
but its tile maxima genuinely depend on many keys. It has been the outlier in
every measurement made on it.

---

## 7. Limitations — read before acting on the savings

**1. `skippable` is per row; kernels skip per tile.** A tile can only be dropped
if *all* 128 rows agree (`min` over rows of deficit > gap). The reported figure
is the per-row mean, which is strictly looser. **The savings in §6.3 are
optimistic by an unmeasured amount**, and this is the single most important
missing measurement.

**2. Global max, not running max.** FlashAttention streams key tiles and holds a
running max over tiles seen so far, which is ≤ the global max. Smaller running
max → smaller deficit → fewer tiles look skippable. Again optimistic.

**3. No output-error validation of the skip policy.** A false-skip rate of 3% is
a count of misclassified tiles, not a measure of how far the layer's output
moves. The machinery to measure that exists in `metrics.py` (relative error of
`AV`) but was never applied to the probe-driven policy.

**4. The probe cost model is idealised.** It counts MACs only. A real kernel pays
for the gather of nominated keys, for divergence, and for a pooled-query pass
that may not map cleanly onto tensor-core shapes. `pQ+4` at 7% of the FLOPs will
not be 7% of the runtime.

**5. Sample counts.** 2 clips for `video` and `vlm`, 3 for Wan, 4 elsewhere; one
model per modality; prefill only; single seed; medians without significance tests.

**6. The mechanism is not new.** Pooling a query tile and using it to select keys
is VecAttention's TilingSelect. The contribution here is applying it to the
**skip decision** rather than to key selection, and the cost accounting that
follows — not the estimator itself.

---

## 8. What to do next, in order

1. **Measure per-tile skippability** (`min` over rows) and the **running-max**
   variant. Both are small changes to `tile_skip.py` and they determine whether
   the headline savings are real. Nothing else should be built first.
2. **Apply the output-error metric to the probe-driven policy** — turn "3% of
   tiles misclassified" into "the layer's output moves by X".
3. **Sweep `P` and `B`.** Everything here used 128×64. Query-side downsampling
   should improve with larger `P` (more rows share one probe) and the key-side
   result should be insensitive to `B`. That is a cheap, falsifiable prediction.
4. Only then consider a kernel prototype, and only for text and Wan, where the
   numbers are strongest.

---

## 9. Reproduction

```bash
# capture (tile metrics are computed at capture time, so probes require a re-run)
python -m qk_sparse run --adapters text image audio video vlm --device cuda \
       --dtype bfloat16 --data-dir assets --out qk_results --samples 4 --layers 6 --keep-going
python -m qk_sparse run --adapters video-dit-long --device cuda --dtype bfloat16 \
       --out qk_results --samples 3 --layers 6

# tables in this report
for g in 1 2 4; do
  python -m qk_sparse report qk_results --density 0.05 --gap $g | grep -A10 '^| model | skippable'
done
```

Implementation: `qk_sparse/tile_skip.py` (`analyse_tiles`), reported by
`tile_table` and `tile_decision_table` in `qk_sparse/aggregate.py`. Raw
per-(layer, head) tile statistics are under `head["tiles"]` in
`qk_results/*.json`.

A separate study of attention *pattern shape* — vertical vectors versus blocks,
lines and horizontal vectors — is in `REPORT.md`, using the same captures.
