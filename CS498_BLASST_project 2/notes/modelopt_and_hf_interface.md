# BLASST reproduction: NVIDIA Model-Optimizer reference, HF AttentionInterface, RULER

Research notes, 2026-10-31. Sources actually read (not recalled from memory):

- `nvidia-modelopt` **0.47.0** wheel downloaded from PyPI and unzipped; every file quoted under
  `modelopt/torch/sparsity/attention_sparsity/` and `modelopt/torch/kernels/` is the real 0.47.0 source.
- `examples/llm_sparsity/attention_sparsity/{README.md,hf_sa.py,download_ruler_data.sh}` fetched from
  `raw.githubusercontent.com` (worked directly; no mirror needed this time).
- `transformers` **5.9.0** installed locally on this Windows machine
  (`...\Python313\Lib\site-packages\transformers\`), plus the live docs page
  `huggingface.co/docs/transformers/main/en/attention_interface` (its content matches 5.9.0 source).
- NVIDIA/RULER `main`: `README.md`, `scripts/run.sh`, `scripts/config_tasks.sh`, `scripts/synthetic.yaml`,
  `scripts/data/prepare.py`, `scripts/data/template.py`, `scripts/data/synthetic/constants.py`,
  `scripts/eval/synthetic/constants.py`.
- BLASST paper HTML `arxiv.org/html/2512.12087v1` (read through a summarizer; quotes marked as such).

Things I could NOT verify are collected in the last section.

---

## PART 1 - NVIDIA Model-Optimizer attention-sparsity example

### 1a. What the example does end to end

**Directory** `examples/llm_sparsity/attention_sparsity/` contains exactly four files:
`.gitignore`, `README.md`, `download_ruler_data.sh`, `hf_sa.py`. There is no `hf_sparse_attn.py`
and no separate calibration script; calibration happens inside `mtsa.sparsify()`.

**Library module** `modelopt/torch/sparsity/attention_sparsity/`:

```
__init__.py              config.py (presets + pydantic configs)   conversion.py (wraps modules, sets attn_implementation, export)
mode.py                  model_sparsify.py (sparsify / calibrate)  sparse_attention.py (SparseAttentionModule wrapper)
stats_manager.py         utils.py
calibration/calibrate.py  calibration/calibrator.py (DynamicThresholdCalibrator)  calibration/ruler_dataset.py
methods/registry.py  methods/flash_skip_softmax.py (the PyTorch simulation)  methods/triton_skip_softmax.py
methods/triton_sparse_softmax.py  methods/vsa.py (video)   plugins/huggingface.py  plugins/vllm.py
```

The Triton kernels live in `modelopt/torch/kernels/common/attention/{triton_fa.py, hf_triton_attention.py,
decode_attention.py}` and `modelopt/torch/kernels/sparsity/attention/calibrate.py`.

**Two methods, two backends** (README, verbatim):

> - **Skip-softmax** (`flash_skip_softmax`): Skips attention tiles whose contribution is negligible, based on a threshold. Based on the BLASST algorithm.
> - **N:M sparse softmax** (`triton_sparse_softmax`): For every M consecutive key positions, keeps the top-N attention scores and sets the rest to -inf before softmax.
> - **pytorch** (default): Patches `F.softmax` to apply skip-softmax sparsity (requires `attn_implementation="eager"`)
> - **triton**: Uses a fused Triton Flash Attention kernel with in-kernel sparsity (uses `attn_implementation="modelopt_triton"`)

**`hf_sa.py` flow** (read from source):

1. `mto.enable_huggingface_checkpointing()`; requires CUDA (`raise OSError("GPU is required for inference.")`).
2. `AutoModelForCausalLM.from_pretrained(path, attn_implementation="eager", dtype="auto", device_map="auto")`.
3. Generates a "before" sample. The README says it "Loads a test sample from the NarrativeQA dataset", but the
   code actually uses a fixed prompt list (`"What is the capital of France? Answer:"`, ...) - README is stale.
4. Deep-copies the chosen preset, applies CLI overrides, then `model = mtsa.sparsify(model, config=sparse_config)`
   (this runs calibration if the config has a `"calibration"` key).
5. Generates an "after" sample, prints both (first 300 chars) and whether they are identical.
6. Optional `export_hf_checkpoint(model, export_dir=...)`.

**CLI flags** (from `argparse` in `hf_sa.py`):

| flag | default | meaning |
|---|---|---|
| `--pyt_ckpt_path` | required | HF model id or local path |
| `--sparse_attn` | `skip_softmax_calib` | one of `skip_softmax_calib`, `skip_softmax_calib_sparse24`, `skip_softmax_triton_calib`, `sparse_softmax` |
| `--backend` | `None` (= preset's) | `pytorch` or `triton`; overrides `backend` in every layer config |
| `--seq_len` | 2048 | max prompt tokens for the before/after generation |
| `--max_new_tokens` | 50 | |
| `--do_sample`, `--temperature` | off, 0.7 | |
| `--export_dir` | `None` | |
| `--target_sparse_ratio` | `None` | float; sets **both** `prefill` and `decode` targets |
| `--calib_samples`, `--calib_max_seqlen`, `--calib_chunk_size` | `None` | override `calibration.{samples,max_seqlen,chunk_size}` |
| `--calib_data_dir` | `None` -> `<script dir>/data` | directory containing `essays/*.txt` |

**Supported models.** No list. `plugins/huggingface.py::register_sparse_attention_on_the_fly` walks
`model.named_modules()` and wraps every module whose class name contains `"attention"` (case-insensitive) or
ends with `Attention`/`SelfAttention`, for any `transformers.PreTrainedModel` (or diffusers model, or plain
`nn.Module`). The README uses `Qwen/Qwen3-8B` and `meta-llama/Llama-3.1-8B` as examples. `LlamaAttention` and
`Qwen3Attention` both match. Note the preset pattern is `"*attn*"`, matched against module *names*
(`model.layers.0.self_attn`), so the wildcard config also matches.

**Does it run on an A100?**

- `backend="pytorch"`: pure PyTorch (`F.softmax` patch inside HF eager attention). No architecture requirement.
  Yes on A100. The constraint is memory, see 1b.
- `backend="triton"`: needs `pip install triton` and CUDA; `kernels/common/attention/__init__.py` only imports the
  kernels if `torch.cuda.is_available()`. I grepped the kernel sources for `get_device_capability`, `sm_80/90`,
  `Hopper`, `Ampere`: **no architecture gate exists**, and the kernels are ordinary Triton FA2-style kernels
  (`triton.Config({"BLOCK_M": block_m, "BLOCK_N": 32}, num_stages=2, num_warps=4)`, `_MEASURE_BLOCK_M = 128`,
  `_MEASURE_BLOCK_N = 128`). So it should run on sm80, but I could not test it.

### 1b. How the PyTorch "skip-softmax" simulation works

**Mechanism.** `conversion.py::_set_attn_implementation` forces `model.config._attn_implementation = "eager"`
for the pytorch backend. Every attention module is replaced by a `SparseAttentionModule` (a `DynamicModule`
subclass) whose `forward` runs the original forward inside a context that monkey-patches
`torch.nn.functional.softmax`:

```python
# methods/flash_skip_softmax.py  (FlashSkipSoftmax.get_sparse_context)
original_softmax = F.softmax

def sparse_softmax(input, dim=-1, *args, **kwargs):
    sparse_mask, stats = self.calculate_sparsity(input)
    module._last_stats = stats
    if not self._calibration_mode:
        input = self.apply_sparsity(input, sparse_mask)
    return original_softmax(input, dim, *args, **kwargs)
...
stack.enter_context(replace_function(torch.nn.functional, "softmax", sparse_softmax))
```

Because HF's `eager_attention_forward` computes `attn_weights = (Q @ K^T) * scaling + attention_mask` and then
calls `nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)`, the tensor the patch sees is the
**scaled** score matrix (`1/sqrt(d)` already applied, natural-log domain) with the additive causal/padding mask
already added, shape `[batch, heads, seq_q, seq_k]`, and **already repeated across the GQA groups**
(`repeat_kv` runs before the matmul). This settles two things for the team:

- the threshold lambda is on the probability scale relative to the running max: a tile is dropped when
  `exp(blockmax - runningmax) <= lambda`, computed on scaled scores;
- the skip decision is **per query head**, never shared within a KV group (nothing in the package reduces over
  `num_key_value_groups`; grep confirmed). The Triton calibration kernel also indexes `kv_head_idx = head_idx // kv_group_num`
  per query head.

**The core routine** (`FlashSkipSoftmax.calc_correction_factor_and_p`, prefill branch, verbatim core):

```python
# thresholds: either fixed list, or calibrated
if use_calibration_params:
    a = calibration_params[phase]["a"]; b = calibration_params[phase]["b"]
    target_sparsity = target_sparse_ratio.get(phase, 0.5)
    scale_factor = a * np.exp(b * target_sparsity)
    log_thresholds = [np.log(scale_factor / seq_k)]
else:
    log_thresholds = [np.log(t) for t in self.thresholds]

if phase == "prefill":
    blocked_attn, num_block_rows, num_block_cols, padded_seq_q, padded_seq_k = (
        self._reshape_to_blocks(attn_weights, self.br, self.bc)          # pad to multiples with finfo.min,
    )                                                                     # view [B, H, nr, br, nc, bc]
    block_max = blocked_attn.max(dim=-1)[0]                               # [B, H, nr, br, nc]  (per row, per key tile)
    block_max_cummax = block_max.cummax(dim=-1)[0]                        # running max over key tiles, left->right, inclusive
    block_max_larger = torch.ones_like(block_max)
    block_max_larger[..., 1:] = block_max[..., 1:] > block_max_cummax[..., :-1]
    correction_factor = (block_max_larger.sum() / block_max_larger.numel()).item()   # diagnostic only
    if self.is_causal:
        num_causal_blocks = num_block_rows * (2 * num_block_cols - num_block_rows + 1) // 2
        total_valid_blocks = batch_size * num_heads * num_causal_blocks
        total_blocks = num_causal_blocks
    ...
    block_diff = block_max - block_max_cummax
    for i, log_threshold in enumerate(log_thresholds):
        block_mask = (block_diff > log_threshold).any(dim=-2)             # [B, H, nr, nc]: keep tile if ANY of its br rows keeps it
        dense_blocks_list.append(block_mask.sum().item())
        if i == 0 and not self._calibration_mode:
            block_mask_0 = block_mask
    # expand block_mask_0 -> element_mask [B, H, seq_q, seq_k]
...
sparsity_list = [1.0 - d / total_valid_blocks for d in dense_blocks_list]
stats = {"correction_factor": ..., "sparsity": sparsity_list, "phase": phase,
         "total_blocks": total_blocks, "sparse_blocks": [...], "sample_length": seq_k}
```

and `apply_sparsity` is simply

```python
mask_value = torch.finfo(attention_scores.dtype).min
return attention_scores.masked_fill(~sparse_mask, mask_value)
```

Details that matter for a faithful re-implementation:

- **Block sizes**: `br = 128` (query rows), `bc = 128` (key columns), both configurable
  (`SparseAttentionAttributeConfig.br/bc`, default 128). Padding uses `finfo.min`, not `-inf`.
- **Tile decision**: row-wise diff `blockmax_row - runningmax_row` is compared to `ln(lambda)`; the tile
  `(query-block i, key-block j)` is kept if **any** of its 128 rows exceeds the threshold (`.any(dim=-2)`).
  Because `cummax` is inclusive, a tile that sets a new running max has diff 0 and is always kept.
- **Causal mask**: not handled explicitly. The additive HF mask (value `finfo.min`) is already in the scores,
  so fully-masked future tiles get `block_max = finfo.min`, fall far below `ln(lambda)` and are "skipped" for
  free. The only explicit causal handling is the statistics denominator (`num_causal_blocks`, the lower
  triangle of tiles). The formula assumes the query rows are the *last* `seq_q` rows of the `seq_k` axis,
  which is exactly what chunked prefill with a KV cache produces (checked: for q=4096, kv=8192 it gives 1552
  tiles, equal to the direct count).
- **Phase**: `_infer_phase` returns `"decode"` iff `attention_scores.shape[2] == 1`, otherwise `"prefill"`.
  So a chunked-prefill chunk is "prefill". Thresholds are per phase: `thresholds = {"prefill": [...], "decode": [...]}`.
- **Decode branch**: `_reshape_to_blocks(attn_weights, 1, self.bc)` - one query row, key tiles of 128;
  same cummax test, no `.any` needed. `sample_length = seq_k` (full KV length at that step).
- **GQA**: see above (per query head, K/V already repeated by HF).
- **Numerics**: eager HF path does softmax in fp32 then casts to the model dtype; P @ V in bf16.
- **Memory (why calibration chunks)**: with eager attention the full `[1, 32, S, S]` score tensor is
  materialized. For Llama-3.1-8B / Qwen3-8B (32 query heads) at S = 16384: 32 x 16384 x 16384 x 2 B = 17.2 GB in
  bf16, plus the fp32 softmax copy (34.4 GB) - impossible on a 40 GB A100 next to 16 GB of weights. With
  `chunk_size = 4096` and 16 K context the biggest tensors are the chunk's scores (32 x 4096 x 16384 x 2 B =
  4.3 GB bf16), its fp32 softmax (8.6 GB), the bool element mask (2.1 GB) and the masked copy (4.3 GB), so
  roughly 20 GB of activations on top of the weights - feasible but tight on 40 GB. Budget accordingly, or
  loop over heads inside your own function.

### 1c. Calibration procedure

**Config** (`CalibrationConfig`, pydantic, verbatim docstring):

> Calibration fits an Exponential model to determine dynamic thresholds that achieve target sparsity. The model learns parameters a and b per phase:
> `scale_factor = a * exp(b * target_sparsity)`
> At inference time, the threshold is computed as: `threshold = scale_factor / sequence_length`

| field | default | notes |
|---|---|---|
| `target_sparse_ratio` | `{"prefill": 0.5, "decode": 0.5}` | fraction of *tiles* to skip; 0.0 disables that phase |
| `samples` | 24 | total RULER-style samples, spread over length bins and the 6 tasks |
| `max_seqlen` | 32768 | length bins = `[max, max/2, max/4, max/8]` (>= 1024), `num_length_bins = 4` |
| `chunk_size` | 2048 | chunked prefill with KV cache to avoid OOM; `-1` = full prefill |
| `num_decode_tokens` | 10 | greedy decode steps per sample for decode calibration |
| `threshold_trials` | `None` | code default is 20 values `1e-6 ... 0.99` (docstring says 16 - stale) |
| `fit_logspace` | `False` | fit `log(scale_factor)` instead; meant for diffusion |
| `cache_dir`, `data_dir` | `None` | JSON cache of generated samples; RULER essays dir |

The shipped preset:

```python
SKIP_SOFTMAX_CALIB = {
    "sparse_cfg": {
        "calibration": {"target_sparse_ratio": {"prefill": 0.5, "decode": 0.5},
                        "samples": 64, "max_seqlen": 16384, "chunk_size": 4096},
        "*attn*": {"method": "flash_skip_softmax", "br": 128, "bc": 128,
                   "backend": "pytorch", "collect_stats": True, "enable": True},
        "default": {"enable": False},
    },
}
```

and the fixed-threshold preset `SKIP_SOFTMAX_DEFAULT` uses `"thresholds": {"prefill": [1e-3], "decode": [1e-4]}`.

**Algorithm** (`DynamicThresholdCalibrator.calibrate`, one forward pass per phase):

1. Set `method.thresholds = threshold_trials` on every wrapped attention module and switch on calibration mode.
   In calibration mode the mask is *computed but not applied* (dense forward; see `sparse_softmax` above), and
   each softmax call records a `sparsity` list with one entry per trial threshold.
2. Run the forward loop over all samples. For prefill: tokenizer(sample, truncation to `sample["length"]`), then
   chunked prefill `model(chunk, past_key_values=..., use_cache=True)`. For decode: prefill with
   `_attn_implementation = "sdpa"` (no measurement), then restore eager and run 10 greedy steps
   `model(next_token, past_key_values=...)`, measuring only the decode steps.
3. Per sample, average the sparsity list across all layers, then unpack into points
   `(scale_factor_ij = lambda_j * L_i, sparsity_ij)` where `L_i = sample_length = seq_k`.
4. Keep points with `0.10 <= sparsity <= 0.90`, fit `scale_factor = a * exp(b * sparsity)` with
   `scipy.optimize.curve_fit` (`p0=[1.0, 5.0]`, bounds `a >= 0`, `0 <= b <= 20`), report `r_squared`.
5. Write `calibration_params = {"prefill": {"a","b",...}, "decode": {...}}` and `target_sparse_ratio` onto every
   module's `_sparse_method_instance`. At inference: `lambda = a * exp(b * S*) / seq_k`.

**Relation to the paper.** BLASST (Eq. 3) fits `lambda = a / L` with "a model-specific constant", by linear
regression on `(1/L_k, lambda_best)` pairs for one target sparsity. Model-Optimizer keeps the `1/L` shape but
replaces the constant by `a * exp(b * S*)`, so the target-sparsity knob `S*` can be moved at runtime without
recalibrating (`module._sparse_method_instance.target_sparse_ratio = {"prefill": X, "decode": Y}`).

**Calibration data.** Generated in-process by `calibration/ruler_dataset.py` ("Copied and Adapted from
NVIDIA/RULER"), using the model's own tokenizer (`model.config._name_or_path`). `RULER_TASKS` contains exactly
six tasks: `niah_multikey_2`, `niah_multikey_3`, `vt`, `fwe`, `qa_1`, `qa_2` (templates identical to RULER's
`constants.py`; `niah_multikey_1` is **not** included). Both NIAH tasks use `type_haystack: "needle"`, VT pads
with `VAR X123 = Y456` lines, FWE with coded words, and the QA tasks use *synthetic* documents
(`_generate_document_text`), not SQuAD/HotpotQA. Sample allocation: `samples_per_length = samples // 4` bins,
then `max(per_bin // 6, 1)` per task, so the preset's 64 becomes 2 per task per bin = 48 samples at
{16384, 8192, 4096, 2048}. The Paul-Graham essays (`download_ruler_data.sh`) are only read when
`type_haystack == "essay"`; by my reading of the source none of the six default tasks hits that path, but the
README and `hf_sa.py` still insist on the download and the loader raises if `data_dir/essays` is missing when
it *is* needed - download anyway (it curls ~200 small `.txt` files from `raw.githubusercontent.com`; from a
mainland-China network, run it through a mirror). NIAH generation also calls `nltk.download("punkt")` /
`"punkt_tab"` at runtime - pre-download or expect a network error.

**Outputs.** Console only: fitted `a`, `b`, `R^2`, observed sparsity range, a table of scale factors for
targets 0.3-0.8, a per-threshold table (`Threshold | Avg SF | Avg Sparsity | Samples`), then
`print_sparse_attention_summary` (e.g. `flash_skip_softmax: 36 layers, calibrated(prefill: a=..., b=..., target=50%)`).
**No plots** and no "% skipped vs lambda" file are written. `calibrate_sparse_attention()` returns
`{"calibration_params", "target_sparse_ratio", "calibration_results"}` but `sparsify()` discards it; read
`module._sparse_method_instance.calibration_params` afterwards or call `mtsa.calibrate` yourself. The export
writes into `config.json`:

```json
"sparse_attention_config": {"config_groups": {"group_0": {"algorithm": "skip_softmax",
   "targets": ["LlamaAttention"],
   "threshold_scale_factor": {"formula": "a * exp(b * target_sparsity)", "prefill": {"a": 7.93, "b": 8.61}, "decode": {"a": 0.12, "b": 9.85}},
   "target_sparsity": {"prefill": 0.5, "decode": 0.5}}},
 "producer": {"name": "modelopt", "version": "0.47.0"}}
```

(the numbers above are the docstring's example, not a measurement).

### 1d. Installation

```bash
pip install nvidia-modelopt[hf]      # 0.47.0 on PyPI (also 0.43 ... 0.46.1); use a domestic PyPI mirror
pip install triton                   # only for backend="triton"
bash ./download_ruler_data.sh        # essays; hits raw.githubusercontent.com ~200 times
```

From the wheel `METADATA`: `Requires-Python >=3.10,<3.15`; core: `torch>=2.8`, `numpy`, `scipy`, `pydantic>=2`,
`safetensors`, `nvidia-ml-py>=12`, `pulp<4`, `omegaconf`, `PyYAML`, `regex`, `rich`, `ninja`, `tqdm`; extra `hf`:
`transformers>=4.57,<5.15`, `accelerate>=1.0`, `datasets>=3`, `diffusers>=0.32.2`, `peft>=0.17`,
`huggingface_hub>=0.24`, `nltk`, `wonderwords`, `sentencepiece`, `tiktoken`, `deepspeed>=0.9.6` (Linux only).
No CUDA version is pinned by the package; the Triton kernels are JIT-compiled, so any CUDA that your torch build
supports works. The locally installed transformers 5.9.0 is inside the supported window.

### 1e. Per-layer / per-head masks and stats

What is exposed:

- Per wrapped module (= per layer): `module.get_stats()` ->
  `{"module", "total_calls", "average_sparsity": [one per threshold], "phase_distribution"}` (aggregated over
  all calls, all heads, all batch items). Iterate with
  `modelopt.torch.sparsity.attention_sparsity.utils.get_named_sparse_attention_modules(model)`.
- Per call: `module._last_stats` (cleared right after collection) and, in calibration mode,
  `module._stats_manager.per_sample_stats` (list of `{"module","sparsity","sample_length","phase"}`).
- `module.get_threshold_info()` gives `a`, `b`, `scale_factor` and example thresholds for 1K-128K.

What is **not** exposed: per-head numbers and the masks themselves. The tile mask `block_mask`
(`[B, H, n_row_tiles, n_col_tiles]` in prefill, `[B, H, 1, 1, n_col_tiles]` in decode) is deleted inside
`calc_correction_factor_and_p` after it is expanded. Cheapest way to get per-layer, per-head masks out of the
reference implementation without forking it: subclass and re-register the method with a higher version (the
registry resolves the latest version automatically):

```python
from modelopt.torch.sparsity.attention_sparsity.methods import register_sparse_method
from modelopt.torch.sparsity.attention_sparsity.methods.flash_skip_softmax import FlashSkipSoftmax

@register_sparse_method("flash_skip_softmax", version="v2")
class RecordingFlashSkipSoftmax(FlashSkipSoftmax):
    def calc_correction_factor_and_p(self, attn_weights, phase):
        element_mask, stats = super().calc_correction_factor_and_p(attn_weights, phase)
        if element_mask is not None:            # None in calibration mode
            # per-head kept fraction; or save the bool tile mask (element_mask[..., ::128, ::128]) to CPU
            stats["per_head_density"] = element_mask.float().mean(dim=(0, 2, 3)).cpu()
        return element_mask, stats
```

`sparse_softmax` already stores `stats` into `module._last_stats`, so a forward hook on each wrapped module can
pick the extra keys up (the stock `SparseAttentionStatsManager.collect` ignores unknown keys). For the project
it is probably simpler to log masks from your own `AttentionInterface` function (Part 2), using the exact tile
logic quoted above as the oracle.

---

## PART 2 - transformers `AttentionInterface` (verified on 5.9.0 source)

### 2a. Registering and selecting a custom attention function

```python
from transformers import AttentionInterface, AttentionMaskInterface          # both re-exported at top level
from transformers.masking_utils import eager_mask, sdpa_mask

AttentionInterface.register("blasst", blasst_attention_forward)   # classmethod: cls._global_mapping.update({key: value})
AttentionMaskInterface.register("blasst", eager_mask)              # SAME name; see 2c for why this is mandatory

model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="blasst", dtype=torch.bfloat16)
# or later:
model.set_attn_implementation("blasst")
```

Register **before** `from_pretrained`: `get_correct_attn_implementation` rejects anything not in
`["eager"] + ALL_ATTENTION_FUNCTIONS.valid_keys()`. `ALL_ATTENTION_FUNCTIONS["blasst"] = fn` also works but only
sets a per-instance local override, so prefer `.register`.

How the models call it (identical in `modeling_llama.py`, `modeling_qwen3.py`, `modeling_qwen2_5_vl.py`):

```python
attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
    self.config._attn_implementation, eager_attention_forward
)
attn_output, attn_weights = attention_interface(
    self,                      # the attention nn.Module (has .num_key_value_groups, .is_causal, .layer_idx, .scaling, .config)
    query_states,              # [batch, num_attention_heads, q_len, head_dim]   (after RoPE)
    key_states,                # [batch, num_key_value_heads, kv_len, head_dim]  (after RoPE, after cache update)
    value_states,              # [batch, num_key_value_heads, kv_len, head_dim]
    attention_mask,            # see 2c
    dropout=0.0 if not self.training else self.attention_dropout,
    scaling=self.scaling,      # head_dim ** -0.5
    sliding_window=self.sliding_window,   # Qwen3 / Qwen2.5-VL only ("# diff with Llama"); None for Qwen3-8B
    **kwargs,
)
attn_output = attn_output.reshape(*input_shape, -1).contiguous()   # input_shape = (batch, q_len)
```

So the required signature and contract are:

```python
def blasst_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,                    # [B, Hq, Sq, D]
    key: torch.Tensor,                      # [B, Hkv, Skv, D]
    value: torch.Tensor,                    # [B, Hkv, Skv, D]
    attention_mask: torch.Tensor | None,    # [B, 1, Sq, Skv] float (eager_mask) / bool (sdpa_mask) / None
    scaling: float,                         # passed by keyword
    dropout: float = 0.0,                   # passed by keyword
    **kwargs,                               # must accept: position_ids, use_cache, sliding_window, is_causal, output_attentions, ...
) -> tuple[torch.Tensor, torch.Tensor | None]:
    ...
    return attn_output, attn_weights        # attn_output MUST be [B, Sq, Hq, D] (i.e. .transpose(1, 2).contiguous()); attn_weights may be None
```

`scaling` and `dropout` are passed as keywords, so their order does not matter but their names do. Both
built-ins end with `attn_output.transpose(1, 2).contiguous()`. Head counts for the two target models
(from the HF configs; both have `head_dim = 128`): Llama-3.1-8B 32 query / 8 KV heads, Qwen3-8B 32 / 8, so
`module.num_key_value_groups == 4` for both.

### 2b. GQA: K/V are NOT pre-repeated

The attention module hands over `key/value` with `num_key_value_heads` heads. Each backend repeats them itself.
`modeling_llama.py` (same code copied into every model file):

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """... (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights
```

`integrations/sdpa_attention.py` does the same unless it can hand GQA to the kernel:

```python
if hasattr(module, "num_key_value_groups") and module.num_key_value_groups > 1:
    if not use_gqa_in_sdpa(attention_mask, key, value):      # CUDA: only when attention_mask is None and head_dim <= 256
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    else:
        sdpa_kwargs["enable_gqa"] = True
```

Note that `repeat_kv` is `repeat_interleave` semantics: query head `h` uses KV head `h // n_rep`. Keep that
mapping if you avoid materializing the repeat (e.g. `query.view(B, Hkv, n_rep, Sq, D)` and broadcast).

### 2c. What `attention_mask` you receive, and causality

`create_causal_mask` (called in `LlamaModel.forward` / `Qwen3Model.forward`) looks up
`ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]`. Three cases:

1. **Your name is not registered in `AttentionMaskInterface`** -> `_preprocess_mask_arguments` returns early:
   `if config._attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping: return True, None, ...`.
   You get `attention_mask=None` **always**, even with padding. The docs warn: "Your attention function must
   handle causal, padding, packing, or sliding-window constraints itself, or those constraints can be silently dropped."
2. **Registered with `eager_mask`** -> always a materialized 4D **float** tensor `[B, 1, Sq, Skv]`, dtype =
   `inputs_embeds.dtype` (bf16 for you), `0.0` = attend, `torch.finfo(dtype).min` (not `-inf`) = masked. Causal
   offset, padding and sliding window (for `sliding_attention` layers) are all baked in. Never `None`
   (`allow_is_causal_skip=False`). Cost: 1 x 1 x 16384 x 16384 x 2 B = 512 MB at 16 K - acceptable.
3. **Registered with `sdpa_mask`** -> 4D **bool** `[B, 1, Sq, Skv]`, `True` = attend, **or `None`** whenever
   `_ignore_causal_mask_sdpa` says the kernel can use `is_causal` instead: no padding and
   (`Sq == 1` or `Sq == Skv`) and not tracing. That is exactly plain prefill and plain decode, so with
   `sdpa_mask` you must implement causality yourself; mirror SDPA's rule
   `is_causal = q_len > 1 and attention_mask is None and is_causal`, with
   `is_causal = kwargs.get("is_causal", getattr(module, "is_causal", True))`.

A user-supplied 4D mask is passed through untouched (early exit). `is_causal` can also arrive as a forward
kwarg (`model(**inputs, is_causal=False)`) and overrides the config for that call.

Recommendation for the simulation: register `eager_mask`. Adding the float mask before thresholding reproduces
the reference path (Part 1b) exactly, including how masked tiles fall out of the skip test.

### 2d. Prefill vs decode, and the KV cache

- Phase: `q_len = query.shape[2]`; `q_len == 1` is decode (same rule as Model-Optimizer). `kv_len = key.shape[2]`.
- The attention module updates the cache **before** calling you:
  `key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)`.
  `DynamicLayer.update` does `self.keys = torch.cat([self.keys, key_states], dim=-2); return self.keys, self.values`,
  so you receive the **full** cached K/V `[B, Hkv, past + q_len, D]`; your queries are the last `q_len` positions
  (for `DynamicCache`, `kv_offset = 0`, so the causal diagonal for query row `i` is at key index
  `kv_len - q_len + i`). With a `StaticCache` you would instead get the full pre-allocated length with a
  materialized mask; stay on `DynamicCache` (the default in `generate()`).
- `generate()` therefore calls you once with `q_len = prompt_len` (prefill) and then once per token with
  `q_len = 1`; chunked prefill (calling `model(chunk, past_key_values=cache, use_cache=True)` yourself) gives
  `1 < q_len < kv_len` - the reference treats that as prefill with `sample_length = kv_len`.
- Extra kwargs that reach you from Llama/Qwen3: `position_ids` (absolute positions, `[B, Sq]`; `[3, B, Sq]`
  mrope positions for Qwen2.5-VL), `use_cache`, plus anything the user passed to `model(...)`
  (`output_attentions`, `is_causal`, `cu_seq_lens_*`, ...). `past_key_values` itself is *not* forwarded.

### 2e. Which models route through the interface

| model file | class | `_supports_attention_backend` | extra kwargs it passes |
|---|---|---|---|
| `models/llama/modeling_llama.py` | `LlamaAttention` | `True` | - |
| `models/qwen3/modeling_qwen3.py` | `Qwen3Attention` (q_norm/k_norm before RoPE) | `True` | `sliding_window=self.sliding_window` (None for Qwen3-8B: `use_sliding_window=False` -> all `layer_types == "full_attention"`) |
| `models/qwen2_5_vl/modeling_qwen2_5_vl.py` | `Qwen2_5_VLAttention` (text) | `True` | `sliding_window`, `position_ids` (3-D mrope) |
| same file | `Qwen2_5_VLVisionAttention` (vision) | | calls the interface with `attention_mask=None, is_causal=False`, per image chunk |

So Llama-3.1 and Qwen3 route through it, and Qwen2.5-VL's language-model attention does too. For Qwen2.5-VL
use the per-backbone dict so the vision tower keeps SDPA:
`attn_implementation={"text_config": "blasst", "vision_config": "sdpa"}`; if you ever give the vision tower
your function it must honour `is_causal=False` and `attention_mask=None`.

### 2f. Pitfalls

- **Mask registration** (2c) is the big one: forgetting `AttentionMaskInterface.register` silently removes
  causality and padding.
- **dropout**: always `0.0` in eval; ignore it or apply `F.dropout(..., training=module.training)`.
- **`output_attentions=True`**: in v5 the model declares `_can_record_outputs = {"attentions": LlamaAttention}`
  and captures the attention module's output tuple via forward hooks, i.e. the second element *you* return.
  Return `None` normally; return the masked probabilities only for small analysis runs (it is `[B, 32, S, S]`).
- **`sliding_window` kwarg**: accept it in `**kwargs`. For Qwen3-8B it is `None`. If a model does use sliding
  layers, `eager_mask` already encodes the window, so adding the mask is sufficient.
- **torch.compile / tracing**: shape-dependent Python branches (`q_len == 1`) and `.item()` calls (the reference
  uses `.item()` for stats) cause graph breaks; the reference is not compiled either. Keep `is_causal` a Python bool.
- **bf16 matching against SDPA**: eager/your path does `softmax(fp32).to(bf16)` then `P_bf16 @ V_bf16`; SDPA's
  flash kernel keeps P in registers with online rescaling and fp32 accumulation. Expect absolute output
  differences of order 1e-2 in bf16 and occasional argmax flips on long inputs. Validate in two steps:
  (i) your function with lambda = 0 (nothing skipped) against `attn_implementation="eager"` - should be bitwise
  or near-bitwise identical since it is the same arithmetic; (ii) against `"sdpa"` with
  `torch.testing.assert_close(atol=..., rtol=...)` on logits, or compare greedy outputs / RULER scores rather
  than tensors. Run the oracle comparison in fp32 on a short prompt once to separate "sparsity error" from
  "bf16 error".
- **`finfo.min` vs `-inf`**: HF masks use `finfo(dtype).min`; adding two of them in bf16 is still finite, so no
  NaNs. If you build your own mask, do the same (a fully masked row with `-inf` gives NaN after softmax).
- **Memory**: see 1b; at 16 K-32 K context you must chunk queries or loop over heads inside the function.
- **Model-Optimizer interaction**: `mtsa.sparsify()` overwrites `config._attn_implementation` to `"eager"`
  (or `"modelopt_triton"`). If you load the reference and your function in the same process, re-set
  `model.set_attn_implementation("blasst")` after sparsify, or use two processes.
- **Version pin**: everything above is 5.9.0; `AttentionMaskInterface`/`masking_utils` exist from ~4.53 on,
  Model-Optimizer 0.47 requires `4.57 <= transformers < 5.15`. Pin one version on the A100 box and re-check
  `eager_attention_forward` there before trusting these notes.

Minimal skeleton that follows the reference tile logic (prefill; add the `q_len == 1` branch with `br = 1`):

```python
import math, torch
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import eager_mask
from transformers.models.llama.modeling_llama import repeat_kv

def blasst_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0,
                             lam=1e-3, br=128, bc=128, **kwargs):
    key = repeat_kv(key, module.num_key_value_groups); value = repeat_kv(value, module.num_key_value_groups)
    s = torch.matmul(query, key.transpose(2, 3)) * scaling            # [B,H,Sq,Sk], scaled scores
    if attention_mask is not None:
        s = s + attention_mask                                         # float mask from eager_mask
    B, H, Sq, Sk = s.shape
    if Sq == 1: br = 1
    pq, pk = -(-Sq // br) * br, -(-Sk // bc) * bc
    sp = torch.nn.functional.pad(s, (0, pk - Sk, 0, pq - Sq), value=torch.finfo(s.dtype).min)
    blk = sp.view(B, H, pq // br, br, pk // bc, bc)
    bmax = blk.max(-1).values                                          # [B,H,nr,br,nc]
    keep = ((bmax - bmax.cummax(-1).values) > math.log(lam)).any(-2)   # [B,H,nr,nc] tile kept if any row keeps it
    keep = keep[:, :, :, None, :, None].expand(B, H, pq // br, br, pk // bc, bc).reshape(B, H, pq, pk)[:, :, :Sq, :Sk]
    s = s.masked_fill(~keep, torch.finfo(s.dtype).min)
    p = torch.softmax(s, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(p, value).transpose(1, 2).contiguous()          # [B,Sq,H,D]
    return out, None

AttentionInterface.register("blasst", blasst_attention_forward)
AttentionMaskInterface.register("blasst", eager_mask)
```

---

## PART 3 - RULER basics for this project

### Tasks

The `synthetic` benchmark = 13 tasks (`scripts/config_tasks.sh`): `niah_single_1/2/3`, `niah_multikey_1/2/3`,
`niah_multivalue`, `niah_multiquery`, `vt`, `cwe`, `fwe`, `qa_1`, `qa_2`. Categories (README): Retrieval
(`niah`), Multi-hop Tracing (`variable_tracking`), Aggregation (`common_words_extraction`,
`freq_words_extraction`), QA. The five the paper calls "challenging subsets", from `scripts/synthetic.yaml`:

```yaml
niah_multikey_1:   {task: niah, args: {type_haystack: essay,  type_needle_k: words, type_needle_v: numbers, num_needle_k: 4, num_needle_v: 1, num_needle_q: 1}}
niah_multikey_2:   {task: niah, args: {type_haystack: needle, type_needle_k: words, type_needle_v: numbers, num_needle_k: 1, num_needle_v: 1, num_needle_q: 1}}
niah_multikey_3:   {task: niah, args: {type_haystack: needle, type_needle_k: uuids, type_needle_v: uuids,   num_needle_k: 1, num_needle_v: 1, num_needle_q: 1}}
vt:                {task: variable_tracking, args: {type_haystack: noise, num_chains: 1, num_hops: 4}}
fwe:               {task: freq_words_extraction, args: {alpha: 2.0}}
```

Meaning (README option table: `words` = adjective-noun, `numbers` = 7 digits, `uuids` = 32 hex digits;
haystack `essay` = Paul Graham essays, `needle` = distracting needles, `repeat/noise` = repeated noise sentences):

- **niah_multikey_1**: Paul-Graham-essay haystack with 4 different keys (one queried, 3 distractors), each
  mapped to a 7-digit number; return the number for the queried key.
- **niah_multikey_2**: the haystack itself is a long list of distractor key -> number needles (no essays);
  one word key is queried.
- **niah_multikey_3**: same as 2 but keys and values are 32-hex UUIDs (hard for tokenizers).
- **vt** (variable tracking): one chain of 4 hops, e.g. `VAR A123 = 12345`, `VAR B456 = A123`, ... buried in
  noise; "Find all variables that are assigned the value {query}" -> the 5 variable names of the chain.
- **fwe** (frequent words extraction): a long "coded text" of synthetic words sampled from a Zipf-like
  distribution with `alpha = 2.0`; return the 3 most frequent coded words.

Prompt templates (RULER `scripts/data/synthetic/constants.py`, identical copy in Model-Optimizer):

```
niah: "Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
      answer_prefix " The special magic {type_needle_v} for {query} mentioned in the provided text are"      tokens_to_generate 128
vt:   "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\nQuestion: Find all variables that are assigned the value {query} in the text above."
      answer_prefix " Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assigned the value {query}, they are: "   tokens_to_generate 30
fwe:  "Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. {context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text?"
      answer_prefix " Answer: According to the coded text above, the three most frequently appeared words are:"   tokens_to_generate 50
```

### Scoring (`scripts/eval/synthetic/constants.py`, verbatim)

```python
def string_match_part(preds, refs):
    score = sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

def string_match_all(preds, refs):
    score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

TASKS = {'niah': string_match_all, 'variable_tracking': string_match_all, 'common_words_extraction': string_match_all,
         'freq_words_extraction': string_match_all, 'qa': string_match_part}
```

Case-insensitive substring match of each reference in the prediction. `niah_multikey_*` has one reference
value -> 0/100 per sample; `vt` has 5 references and `fwe` 3 -> partial credit per sample. Scores are
percentages averaged over samples.

### Data generation (`scripts/data/prepare.py`)

```
python data/prepare.py --save_dir DIR --benchmark synthetic --task niah_multikey_1 \
    --tokenizer_path meta-llama/Llama-3.1-8B-Instruct --tokenizer_type hf \
    --max_seq_length 16384 --model_template_type meta-llama3 --num_samples 500 [--remove_newline_tab]
```

Flags: `--save_dir`, `--benchmark` (`synthetic`), `--task`, `--subset` (`validation`), `--tokenizer_path`,
`--tokenizer_type` (`nemo|hf|openai`, default `nemo` - set `hf`), `--max_seq_length` (includes generated tokens),
`--num_samples` (500), `--random_seed` (42), `--model_template_type` (keys in `template.py`: `base`, `meta-chat`,
`meta-llama3`, `vicuna-chat`, `Phi3`, `jamba`, `nemotron5-instruct`, ...; there is **no Qwen template** - use
`base` or add one), `--chunk_idx/--chunk_amount` (parallel generation). `prepare.py` wraps the task template in
the chat template, then shells out to `synthetic/{niah,variable_tracking,...}.py` with
`--tokens_to_generate` from `constants.py`; the task scripts grow the haystack until the tokenized prompt fits
`max_seq_length - tokens_to_generate`, so the **tokenizer is required** and the data is tokenizer-specific.

What it needs: a HF tokenizer; for essay haystacks `python scripts/data/synthetic/json/download_paulgraham_essay.py`
(URL list `PaulGrahamEssays_URLs.txt` -> ~200 `raw.githubusercontent.com` files - mirror needed in China);
`bash download_qa_dataset.sh` (SQuAD + HotpotQA) only for `qa_1/qa_2`; `nltk` punkt; `wonderwords` (adjective /
noun lists for word needles). The README marks this pipeline deprecated and points to the `rulerv1-ns` /
`rulerv2-ns` branches (NeMo-Skills based); the deprecated docker is `nvcr.io/nvidia/pytorch:23.10-py3` with
`docker/requirements.txt`.

### Sequence lengths

`run.sh` loops `for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"` but **`SEQ_LENGTHS` is not defined anywhere in the
`main` branch files I fetched** (`run.sh`, `config_tasks.sh`) - you set it yourself. The leaderboard uses
4K, 8K, 16K, 32K, 64K, 128K; the BLASST paper calibrates on about 1000 RULER sequences at 4K/8K/16K/32K/64K
and reports RULER-32K in Table 1 (Llama-3.1-8B dense 92.33 vs 91.67 at ~75 % sparsity; Qwen3-8B 91.90 vs
92.11 prefill-only, as read through the summarizer). On a 40 GB A100 with an eager/simulated kernel, 4K-16K is
the realistic range for the mid-semester report.

### Pre-generated datasets (avoid regenerating)

- **`self-long/RULER-llama3-1M`** (Hugging Face, MIT): all 13 tasks x 9 lengths (`4k`..`1M`), 500 rows for
  4k-64k, config name `f"{task}_{length}"`, split `validation`, columns `input`, `answers`, `index`, `length`
  (+ `predictions` from the uploader's models). Built with the Llama-3 chat template, completion-style prompts.
  `ds = load_dataset("self-long/RULER-llama3-1M", "niah_multikey_1_8k", split="validation")`. Llama-3 and
  Llama-3.1 share the tokenizer, so this is directly usable for Llama-3.1-8B-Instruct; for Qwen3-8B the token
  counts will differ (re-tokenize and check `length` against your budget).
- **`lighteval/RULER-{4096|16384|32768}-{model}`** (e.g. `RULER-32768-Qwen3-1.7B`, `RULER-16384-Qwen2.5-0.5B`,
  `RULER-4096-Falcon-H1-3B-Base`): 500 examples per task split, generated per tokenizer by lighteval. Qwen3
  sizes share one tokenizer, so the Qwen3-1.7B set is a candidate for Qwen3-8B - I did not open the card to
  confirm the exact task list and template.
- `MaxJeblick/Ruler`: gated and the card is empty - not useful.
- ModelScope: a web search found no RULER dataset; search modelscope.cn directly. For Hugging Face from the
  mainland use `HF_ENDPOINT=https://hf-mirror.com`.

---

## What I could not verify / discrepancies to keep in mind

- NVIDIA docs page `nvidia.github.io/TensorRT-Model-Optimizer/guides/_attention_sparsity.html` returned 404;
  the reference pages on `nvidia.github.io/Model-Optimizer/` only list field names. Everything above comes
  from the wheel source instead.
- Whether the Triton backend (`modelopt_triton`) actually runs on an A100: no arch gate in the code, untested.
- Whether `download_ruler_data.sh` is really needed for the six default calibration tasks (source says the essay
  path is never taken; README says required). Download it anyway.
- RULER `prepare.py` output field names (`index/input/outputs/length`) were not re-read from the task scripts.
- `SEQ_LENGTHS` undefined on RULER `main`; the paper's exact per-length lambda / `a` values are not reported.
- The paper does not state tile sizes, whether lambda is applied to scaled scores, or per-head vs per-group GQA
  decisions; the Model-Optimizer source settles all three for the reference implementation (128 x 128,
  scaled, per query head).
- `CalibrationConfig.threshold_trials` docstring lists 16 defaults; `DynamicThresholdCalibrator` code uses 20.
- README's "NarrativeQA sample" vs the fixed prompts in `hf_sa.py`.
- Local transformers is 5.9.0 on this Windows laptop; the A100 box may differ. Re-check `eager_attention_forward`
  and `masking_utils` there.
