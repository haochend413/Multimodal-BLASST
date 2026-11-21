# Qwen2.5-VL-7B-Instruct + custom (BLASST-style) attention on long videos: practical guide

Written 2026-10-31 for the CS 498 BLASST-reproduction project. Goal of this note: let one teammate run
Qwen2.5-VL-7B-Instruct on 10-20 long videos with our tile-skipping attention simulation (128x128 QK^T
tiles, skip when every row's tile-max is below the running row-max by more than ln lambda), and report
the % of tiles skipped for TEXT tiles vs VISION tiles under one threshold.

Everything below was checked against the sources listed at the end on 2026-10-31 unless it is in the
"Unverified" list (section 10). Version pins that matter: **transformers 4.57.3** (last 4.x line; all
file citations are to that tag unless stated) and **qwen-vl-utils 0.0.11** (the Qwen2.5-VL-era release).

---

## 0. TL;DR

| Question | Answer |
|---|---|
| Video tokens per frame-pair | `(H/28) * (W/28)` after resize to multiples of 28; two consecutive frames share one group of tokens (`temporal_patch_size = 2`). Total video tokens = `(nframes/2) * (H/28) * (W/28)`. |
| Default sampling | `fps = 2.0`, `min_frames = 4`, `max_frames = 768`, per-frame cap `768 * 28 * 28` px (768 tokens), total budget `128000 * 28 * 28 * 0.9` px. 60 s of 720p at defaults = **43,200 video tokens** (exceeds the 32K context). |
| Special token ids | `<\|vision_start\|>` = **151652**, `<\|vision_end\|>` = **151653**, `<\|vision_pad\|>` = 151654, `<\|image_pad\|>` = **151655**, `<\|video_pad\|>` = **151656**. `is_vision = (input_ids == 151656) \| (input_ids == 151655)`. |
| Does `attn_implementation="my_custom"` reach the LLM? | **Yes, from transformers v4.53.0 on** (`Qwen2_5_VLAttention` dispatches through `ALL_ATTENTION_FUNCTIONS`). v4.52.4 and earlier use hard-coded `QWEN2_5_VL_ATTENTION_CLASSES` (eager / sdpa / flash only). |
| Can it be restricted to the LLM? | Yes: `attn_implementation={"text_config": "blasst", "vision_config": "sdpa"}`. A plain string applies to the ViT too (it would call your function with `is_causal=False` on 64-patch windows and on 65K-patch full-attention layers). |
| M-RoPE | Applied to q/k **before** the attention function is called; your function sees post-RoPE tensors. `position_ids` is `(3, B, L)` and is forwarded as a kwarg you can ignore. |
| GQA | 28 query heads, 4 KV heads (`num_key_value_groups = 7`), head_dim 128, 28 layers, hidden 3584. K/V arrive **un-repeated**; call `repeat_kv` yourself. |
| 16K prefill on A100-40GB? | Fits: ~16.6 GB weights + ~0.9 GB KV cache + ~2-3 GB transients + 0.24 GB per 128-row fp32 score chunk. Must: chunk scores (full 16K x 16K fp32 matrix = 30 GB -> OOM), keep ViT on sdpa/flash, pass `logits_to_keep=1` (full logits = 5 GB). |
| Best video source | **Video-MME** (`lmms-lab/Video-MME`, not gated, long = 30-60 min, `duration` column, subtitles shipped). Fallback: Blender open movies on archive.org (CC BY 3.0). |

---

## 1. Video tokenization in Qwen2.5-VL

### 1.1 The pipeline (what turns a video file into `<|video_pad|>` tokens)

Two stages, two different pieces of code. Both matter for the token count.

**Stage A - `qwen_vl_utils.process_vision_info` (the Qwen helper, runs on CPU):**
decodes the video, picks `nframes` frames, resizes them, returns a tensor `(nframes, 3, H, W)` plus
`video_kwargs = {"fps": [sample_fps]}` (when `return_video_kwargs=True`).

**Stage B - the HF `Qwen2_5_VLProcessor` / `Qwen2VLVideoProcessor` (transformers):**
re-runs its own `smart_resize` with its own pixel limits (!), cuts 14x14 patches, groups two frames per
patch (`temporal_patch_size = 2`), merges 2x2 spatial patches (`merge_size = 2`), and replaces the single
`<|video_pad|>` in the chat-template text with `video_grid_thw.prod() // 4` copies of it.

Verified constants (`config.json` / `preprocessor_config.json` of `Qwen/Qwen2.5-VL-7B-Instruct`):
`patch_size = 14`, `spatial_merge_size = 2` (`merge_size = 2`), `temporal_patch_size = 2`,
`vision_config.tokens_per_second = 2`, `window_size = 112`, `fullatt_block_indexes = [7, 15, 23, 31]`,
ViT depth 32, ViT hidden 1280, 16 ViT heads, `out_hidden_size = 3584`.

So one LLM token covers a **28 x 28 pixel area over 2 frames** (14 px patch x 2 merge = 28). Per
temporal group:

```
tokens_per_group = (H_resized / 28) * (W_resized / 28)
video_tokens     = (nframes / 2) * tokens_per_group        # nframes is always even (FRAME_FACTOR = 2)
```

Exactly: `video_grid_thw = [nframes // 2, H // 14, W // 14]` and `num_video_tokens = prod(video_grid_thw) // 4`
(`processing_qwen2_5_vl.py`, `merge_length = merge_size**2`). If `nframes` is odd the last frame is
duplicated (`video_processing_qwen2_vl.py`).

### 1.2 Frame sampling: `smart_nframes` (qwen-vl-utils 0.0.11, `vision_process.py` at commit a30e36f)

Module constants (verbatim):

```
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200
VIDEO_MIN_PIXELS = 128 * 28 * 28          # 100,352 px  (128 tokens per group)
VIDEO_MAX_PIXELS = 768 * 28 * 28          # 602,112 px  (768 tokens per group)
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))   # 90,316,800 px
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
```

Note the env var that sets the *total* budget is confusingly named `VIDEO_MAX_PIXELS`. (In qwen-vl-utils
0.0.13+ the file was rewritten for Qwen3-VL: same numbers, but expressed as `VIDEO_MIN_TOKEN_NUM = 128`,
`VIDEO_MAX_TOKEN_NUM = 768`, total = `MODEL_SEQ_LEN * 28 * 28 * 0.9` with env var `MODEL_SEQ_LEN`, and
`process_vision_info(..., image_patch_size=14)` must stay at 14 for Qwen2.5-VL.)

Frame count, given the message dict `ele` and the file's `total_frames`, `video_fps`:

```
if "nframes" in ele:   nframes = round_by_factor(ele["nframes"], 2)
else:
    fps        = ele.get("fps", 2.0)
    min_frames = ceil_by_factor(ele.get("min_frames", 4), 2)
    max_frames = floor_by_factor(ele.get("max_frames", min(768, total_frames)), 2)
    nframes    = total_frames / video_fps * fps
    nframes    = floor_by_factor(min(max(nframes, min_frames), max_frames, total_frames), 2)
```

`fps` and `nframes` are mutually exclusive (assert). The tech report confirms the cap: "we capped the
maximum number of frames analyzed per video at 768, with the total number of video tokens not exceeding
24,576" (arXiv 2502.13923, Table/Sec. on video).

### 1.3 Per-frame resize: `fetch_video` + `smart_resize`

```
min_pixels   = ele.get("min_pixels", VIDEO_MIN_PIXELS)
total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
max_pixels   = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
max_pixels   = min(ele.get("max_pixels", max_pixels), max_pixels)
H, W = smart_resize(h, w, factor=28, min_pixels=min_pixels, max_pixels=max_pixels)
```

`smart_resize` rounds each side to a multiple of 28; if the area exceeds `max_pixels` it scales both sides
down by `beta = sqrt(h*w / max_pixels)` and **floors** to a multiple of 28; if below `min_pixels` it scales
up and ceils.

Two consequences worth internalising:

1. `total_pixels` is really a **budget on total video tokens x 784**: because the per-frame cap is
   `total_pixels / nframes * 2`, the sum over `nframes/2` groups is at most `total_pixels / 784` tokens
   (floor rounding usually lands 5-15 % under).
2. The `int(min_pixels * 1.05)` floor (105,369 px, about 134 tokens per group) **silently overrides**
   `total_pixels` once `nframes` is large. With `nframes = 768` the budget would need ~103K tokens; if you
   ask for 16K via `total_pixels` you still get ~46K. For long videos you must lower `min_pixels` or
   cap `nframes` / `max_frames` explicitly.

### 1.4 Stage B pitfall: the HF video processor resizes again

`Qwen2VLVideoProcessor` (transformers 4.57.3, `video_processing_qwen2_vl.py`) has
`size = {"shortest_edge": 128 * 28 * 28, "longest_edge": 768 * 28 * 28}`, `do_resize = True`,
`do_sample_frames = False`, `min_frames = 4`, `max_frames = 768`, and in `_preprocess` calls
`smart_resize(h, w, factor=28, min_pixels=size["shortest_edge"], max_pixels=size["longest_edge"])` on
the frames it receives. So if Stage A produced frames with fewer than 100,352 px (e.g. 224x420 = 94,080,
which Stage A happily emits because its own floor rounding lands under `min_pixels`), Stage B **upscales**
them back to >= 128 tokens per group. The fix is to make Stage B's limits match what you asked for:

```python
processor.video_processor.size = {"shortest_edge": 32 * 28 * 28, "longest_edge": 768 * 28 * 28}
```

and then **always verify** with `inputs["video_grid_thw"]` and `(inputs["input_ids"] == 151656).sum()`
rather than trusting arithmetic.

### 1.5 Worked examples (computed with the formulas above; 1280x720 source at 30 fps)

| Case | Message dict | nframes | frame after resize | tokens/group | video tokens |
|---|---|---|---|---|---|
| A. 60 s, defaults | `{}` | 120 | 560x1008 | 720 | **43,200** (does not fit 32K) |
| B. 60 s, ~8K | `{"total_pixels": 9216*28*28}` | 120 | 252x448 | 144 | **8,640** |
| C. 60 s, ~16K | `{"total_pixels": 16384*28*28}` | 120 | 336x616 | 264 | **15,840** |
| D. 30 min, defaults | `{}` | 768 | 336x644 | 276 | **105,984** (way over) |
| E. 30 min, naive 16K | `{"max_frames": 256, "total_pixels": 16384*28*28}` | 256 | 224x420 -> HF upscales to 252x448 | 144 | **18,432** (would be 15,360 without the Stage B re-resize) |
| F. 30 min, ~8K | `{"nframes": 128, "max_pixels": 128*28*28, "min_pixels": 32*28*28}` + processor `shortest_edge = 32*28*28` | 128 | 224x420 | 120 | **7,680** |
| G. 30 min, ~16K | `{"nframes": 128, "max_pixels": 256*28*28}` | 128 | 336x588 | 252 | **16,128** |
| H. 30 min, ~8K, higher res | `{"nframes": 64, "max_pixels": 256*28*28}` | 64 | 336x588 | 252 | **8,064** |

Recommendation for the experiment: fix `nframes` (even) and `max_pixels` explicitly (cases F/G/H), set
`min_pixels` and the processor `shortest_edge` low, and record `(nframes, H, W, video_tokens)` per video.
Sampling 128 frames of a 30-min video means one frame every ~14 s; that is the price of a 16K budget.
Remember the `fps` you pass to the processor (`video_kwargs["fps"]`) is `nframes / total_frames * video_fps`,
which the model uses for absolute-time M-RoPE (`second_per_grid_ts = temporal_patch_size / fps`).

A reproducible calculator lives at
`C:\Users\kevin\AppData\Local\Temp\claude\C--Users-kevin-coursework-local-CS-498\47c86247-3162-42a4-93d5-661e77e534e7\scratchpad\qwen_tokens.py`
(copy it into the project if useful; it re-implements the verified logic of both stages).

---

## 2. Sequence layout inside the LLM; text vs vision positions; M-RoPE

### 2.1 Special tokens (from `tokenizer_config.json`, `added_tokens_decoder`, all `special: true`)

| id | token | role |
|---|---|---|
| 151643 | `<\|endoftext\|>` | pad / eos (base) |
| 151644 | `<\|im_start\|>` | chat turn start |
| 151645 | `<\|im_end\|>` | chat turn end |
| 151646-151651 | `<\|object_ref_start\|>` ... `<\|quad_end\|>` | grounding, unused here |
| **151652** | `<\|vision_start\|>` | one before every image/video block |
| **151653** | `<\|vision_end\|>` | one after every image/video block |
| 151654 | `<\|vision_pad\|>` | unused placeholder |
| **151655** | `<\|image_pad\|>` | one per image token (`image_token_id`) |
| **151656** | `<\|video_pad\|>` | one per video token (`video_token_id`) |

`config.json` agrees: `image_token_id = 151655`, `video_token_id = 151656`, `vision_start_token_id =
151652`, `vision_end_token_id = 151653`. The model scatters ViT outputs with
`inputs_embeds.masked_scatter(input_ids == config.video_token_id, video_embeds)` (`Qwen2_5_VLModel.forward`),
so **the set of vision positions is exactly `input_ids == 151656`** (plus 151655 if images are present).

### 2.2 What the prompt looks like

The chat template renders a video item as `<|vision_start|><|video_pad|><|vision_end|>` (with an optional
`Video N:` prefix when `add_vision_id` is set); the processor then expands the single `<|video_pad|>` to
`N_video` copies. A typical single-video prompt is therefore:

```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
<|im_start|>user\n<|vision_start|> <|video_pad|> x N_video <|vision_end|> QUESTION TEXT <|im_end|>\n
<|im_start|>assistant\n
```

Implication for the tile experiment: with 128x128 tiles, a "text tile" needs >= 128 contiguous text
positions on the query side (and on the key side). A default prompt has only ~20-40 text tokens, so you
would get essentially **zero pure-text tiles**. Put a few hundred to a few thousand text tokens in the
prompt: Video-MME ships `subtitle.zip` and LongVideoBench ships `subtitles.tar`; pasting the subtitle
text (or a long instruction) *before* the video gives a clean text block. Then classify each
(query-tile, key-tile) pair by the modality of its rows and columns: text->text, vision->vision,
text->vision, vision->text, and mixed (tiles straddling a boundary; report separately or drop).

Also decide up front whether `<|vision_start|>`/`<|vision_end|>` and the chat-control tokens count as
text; they are single positions, so it only affects the boundary tile.

### 2.3 M-RoPE position ids and why the custom attention function does not care

`get_rope_index` builds `position_ids` of shape `(3, B, L)`: rows are (temporal, height, width). Text
tokens get the same scalar in all three rows; video tokens get grid coordinates, with the temporal row
scaled by `second_per_grid_t * tokens_per_second` so that it tracks absolute time (this is the "absolute
time alignment" from the tech report).

In `Qwen2_5_VLAttention.forward` (transformers 4.57.3) the order is:

1. `q_proj / k_proj / v_proj`
2. `query_states, key_states = apply_multimodal_rotary_pos_emb(query_states, key_states, cos, sin, self.rope_scaling["mrope_section"])`
3. `past_key_values.update(...)`
4. `attention_interface(self, query_states, key_states, value_states, attention_mask, dropout=..., scaling=self.scaling, sliding_window=self.sliding_window, position_ids=position_ids, **kwargs)`

So RoPE (including the 3-axis M-RoPE mixing, `mrope_section = [16, 24, 24]`) is baked into q and k before
your function runs. `position_ids` is forwarded only "for FA2"; accept and ignore it via `**kwargs`.
`sliding_window` is `None` for this checkpoint (`use_sliding_window = false`).

---

## 3. Does the LLM route through `AttentionInterface`? Which versions? What about the ViT?

### 3.1 Version matrix (checked file-by-file on GitHub tags)

| transformers | `Qwen2_5_VLAttention` (LLM) | `Qwen2_5_VLVisionAttention` (ViT) | custom `attn_implementation` works? |
|---|---|---|---|
| v4.51.3 | `QWEN2_5_VL_ATTENTION_CLASSES = {"eager", "flash_attention_2", "sdpa"}` hard-coded classes | same pattern (`QWEN2_5_VL_VISION_ATTENTION_CLASSES`) | **No** (KeyError on unknown name) |
| v4.52.4 | still class dict; `Qwen2_5_VLTextModel` + `config.text_config` already exist | class dict | **No** |
| **v4.53.0** | `attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]` (falls back to `eager_attention_forward` when "eager") | same | **Yes** |
| v4.57.3 (recommended pin) | same as 4.53; verified call signature above | same; non-flash backends split by `cu_seqlens` and call `attention_interface(..., attention_mask=None, is_causal=False)` per chunk | **Yes** |
| main (v5.x) | `ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)`; `apply_rotary_pos_emb` with M-RoPE handled inside the rotary module | same | **Yes** (but v5 renamed some kwargs, e.g. `dtype=` instead of `torch_dtype=`) |

Model layout from v4.53 on: `Qwen2_5_VLForConditionalGeneration.model` is a `Qwen2_5_VLModel` holding
`self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)` and
`self.language_model = Qwen2_5_VLTextModel._from_config(config.text_config)`; `lm_head` sits on the
outer class. The Hub `config.json` is the old flat format (`transformers_version: 4.41.2`), and
`Qwen2_5_VLConfig` builds `text_config` from the flat keys when the `text_config` key is missing
(`configuration_qwen2_5_vl.py`), so the published checkpoint loads fine.

### 3.2 Restricting the custom function to the LLM

`Qwen2_5_VLConfig.sub_configs = {"vision_config": Qwen2_5_VLVisionConfig, "text_config": Qwen2_5_VLTextConfig}`.
Per the transformers attention-interface docs, `attn_implementation` may be a dict keyed by sub-config name;
omitted keys fall back to SDPA; a single string sets all backbones; `model.set_attn_implementation(...)`
switches after loading.

```python
attn_implementation={"text_config": "blasst", "vision_config": "sdpa"}
```

Why not a plain string: the ViT would then call your function (a) in the 28 windowed layers on chunks of
<= 64 patches (window 112 px = 8x8 patches), (b) in the 4 full-attention layers (`[7, 15, 23, 31]`) on the
whole patch sequence (65,536 patches for a 16K-token video), with `is_causal=False`, `attention_mask=None`,
head_dim 80, 16 heads. A causal tile-skipping simulation applied there would corrupt the vision features
and would be dramatically slow. Keep the ViT on `sdpa` (or `flash_attention_2` if installed).

### 3.3 What your function receives (LLM side)

Signature contract, same as `sdpa_attention_forward` in `integrations/sdpa_attention.py`:

```python
def blasst_attention_forward(module, query, key, value, attention_mask,
                             dropout=0.0, scaling=None, is_causal=None, **kwargs):
    # query: (B, 28, L, 128)   key/value: (B, 4, L, 128)   <- NOT repeated for GQA
    # returns (attn_output of shape (B, L, 28, 128), attn_weights_or_None)
```

- **GQA:** `module.num_key_value_groups == 7`; do `key = repeat_kv(key, 7)` (from
  `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv`) or index KV head `h // 7`.
- **Mask:** the docs state that if no mask function is registered for a custom attention name, mask
  creation is skipped and `attention_mask=None` is passed. Even if you register `sdpa_mask` via
  `AttentionMaskInterface.register("blasst", sdpa_mask)`, it returns `None` for an unpadded causal batch
  (`allow_is_causal_skip=True`). So **your function must apply causality itself** (`module.is_causal` is
  `True`); with batch size 1 there is no padding to worry about.
- **Scaling:** `scaling = 1/sqrt(128)` is passed; use it.
- **Output layout:** transpose to `(B, L, H, D)` and `.contiguous()` before returning; the caller reshapes to
  `(B, L, 3584)` and applies `o_proj`.
- **Layer id:** `module.layer_idx` (0..27) for per-layer statistics.
- `output_attentions` is not meaningful for a custom function; return `None` for weights.

### 3.4 LLM config (Qwen2.5-VL-7B-Instruct `config.json`)

`hidden_size 3584`, `num_hidden_layers 28`, `num_attention_heads 28`, `num_key_value_heads 4`
(head_dim 128, 7 query heads per KV head), `intermediate_size 18944`, `max_position_embeddings 128000`
(model card: "config.json is set for context length up to 32,768 tokens"; for long videos the card says
`max_position_embeddings` "can be directly modified to a larger value, such as 64k" and warns against YaRN
for temporal/spatial tasks), `rope_theta 1e6`, `rope_scaling = {type: mrope, mrope_section: [16, 24, 24]}`,
`vocab_size 152064`, `tie_word_embeddings false`, `use_sliding_window false`, `torch_dtype bfloat16`.
Total parameters (HF API, safetensors): **8,292,166,656**, all BF16; on-disk 16,584,414,560 bytes.

---

## 4. Memory on one A100 40 GB (bf16, batch 1)

Static: weights 16.58 GB (8.29B x 2 B). CUDA context ~0.5 GB.

Per-sequence-length numbers (computed from the config; L = LLM sequence length):

| L | KV cache (28 layers x 2 x L x 4 x 128 x 2 B) | one MLP intermediate tensor (L x 18944 x 2 B) | hidden (L x 3584 x 2 B) | 128-row fp32 score chunk (28 x 128 x L x 4 B) | full score matrix bf16 / fp32 (28 x L^2) | all-position logits bf16 (L x 152064 x 2 B) |
|---|---|---|---|---|---|---|
| 8,192 | 0.47 GB | 0.31 GB | 59 MB | 117 MB | 3.8 / 7.5 GB | 2.5 GB |
| **16,384** | **0.94 GB** | **0.62 GB** | **117 MB** | **235 MB** | **15.0 / 30.1 GB** | **5.0 GB** |
| 32,768 | 1.88 GB | 1.24 GB | 235 MB | 470 MB | 60 / 120 GB | 10.0 GB |

Verdict for 16K: **fits with roughly 15-19 GB of headroom** (16.6 weights + 0.9 KV + ~2-3 GB of layer
transients: gate/up/activation MLP tensors ~1.9 GB, q/k/v/o ~0.3 GB, two 128-row chunks of scores and
probabilities ~0.5 GB), provided:

1. **Chunk the scores.** A full `28 x 16384 x 16384` matrix is 15 GB in bf16 and 30 GB in fp32
   (`eager_attention_forward` softmaxes in fp32); 16.6 + 30 > 40 -> OOM. The 128-query-row chunk loop is
   0.24 GB per chunk in fp32; keep the running-max/sum state in fp32 as `(28, 128)` vectors. Per-tile
   skip bookkeeping for all layers is tiny: 28 layers x 28 heads x 128 x 128 tiles = 12.8 M booleans.
2. **Vision tower on sdpa/flash.** The ViT runs on 4 x 16,384 = 65,536 patches; its four full-attention
   layers under eager would need 16 x 65536^2 x 2 B = 137 GB. SDPA's memory-efficient kernel handles it.
3. **`logits_to_keep=1`** in `model(...)` / `model.forward` (supported in 4.57.3:
   `logits_to_keep: Union[int, torch.Tensor] = 0` -> `logits = self.lm_head(hidden_states[:, slice_indices, :])`).
   Default 0 computes 5 GB of logits for all 16K positions. Alternatively call `model.model(...)` for
   hidden states only.
4. `torch.inference_mode()`; `use_cache=False` if you do not generate (saves the 0.94 GB).
5. Do not keep per-layer attention maps for plotting; if you want one, dump one head of one layer as a
   downsampled tile-level matrix (128 x 128 floats).

32K would also fit (~24 GB) under the same rules, so the headroom is real.

Time (not memory) is the practical limit of a pure-PyTorch simulation: 28 layers x 128 chunks x 28 heads of
`(128 x 128) @ (128 x 16384)` matmuls per forward; expect tens of seconds per 16K prefill, which is fine
for 10-20 videos but argues against sweeping many thresholds per forward.

---

## 5. Where to get 10-20 long videos (from a US CloudLab node)

| Dataset | HF id / access | Length | Size and layout | License / terms |
|---|---|---|---|---|
| **Video-MME** (recommended) | `lmms-lab/Video-MME` - no gate shown on the card | 900 videos, 254 h total; short < 2 min, medium 4-15 min, **long 30-60 min** | `videos_chunked_01..20.zip` (~5 GB each, 101 GB total), `subtitle.zip` (7.5 MB), `videomme/test-00000-of-00001.parquet` (405 KB) with columns `video_id, duration (short/medium/long), domain, sub_category, url, videoID, question_id, task_type, question, options, answer` | GitHub README: "only used for academic research. Commercial use in any form is prohibited." |
| LongVideoBench | `longvideobench/LongVideoBench` - gated (accept terms) | 3,763 videos, groups 8-15 s, 15-60 s, 180-600 s, **900-3600 s** | `videos.tar.*` split parts that must be concatenated (162 GB total), `subtitles.tar`, `lvb_val.json` | CC-BY-NC-SA 4.0, non-commercial only |
| MLVU | `MLVU/MVLU` - gated | 3 min - 2 h | 430 GB | CC-BY-NC-SA 4.0, research only |
| Fallback: Blender open movies on archive.org | no account; direct links | Big Buck Bunny ~10 min (720p mp4, 59 MB): `https://archive.org/download/BigBuckBunny_124/Content/big_buck_bunny_720p_surround.mp4`; Sintel 14:48 (mp4 ~270 MB): `https://archive.org/download/Sintel/sintel-2048-stereo.mp4` | tiny | CC BY 3.0 |

Practical recipe for Video-MME: download one or two zip chunks (10 GB), unzip, list the `.mp4` names,
join on `videoID` with the parquet, keep rows with `duration == "long"`, and use the matching `subtitle.zip`
entries as the text block. The chunk-to-duration mapping is not published, so expect to need 1-2 chunks for
10-20 long videos (each chunk holds ~45 videos). Do not rely on the `url` column (YouTube) from a
datacenter node; yt-dlp is routinely blocked there.

Note on "long": with a 16K budget you only feed 64-128 frames regardless of source length, so the Blender
shorts are perfectly adequate for the sparsity measurement; the benchmark videos mainly buy credibility and
subtitles.

---

## 6. Model mirrors and ids

| Model | Hugging Face id | ModelScope id (verified via `modelscope.cn/api/v1/models/...`) | License |
|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | `Qwen/Qwen2.5-VL-7B-Instruct` (not gated) | `Qwen/Qwen2.5-VL-7B-Instruct` | apache-2.0 |
| Qwen3-8B | `Qwen/Qwen3-8B` | `Qwen/Qwen3-8B` | apache-2.0 |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` (gated on HF; see Unverified) | `LLM-Research/Meta-Llama-3.1-8B-Instruct` | Llama 3.1 Community License |

hf-mirror fallback (from hf-mirror.com): `export HF_ENDPOINT=https://hf-mirror.com` (PowerShell:
`$env:HF_ENDPOINT = "https://hf-mirror.com"`), then `huggingface-cli download --resume-download
Qwen/Qwen2.5-VL-7B-Instruct --local-dir ...`; gated repos need `--token hf_...` obtained on the official site.
ModelScope: `pip install modelscope` then `from modelscope import snapshot_download;
snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")` (standard API; exact CLI flags not re-verified).
On the CloudLab node plain HF works; the laptop in China should use ModelScope or hf-mirror.

---

## 7. Prior work to cite and differentiate from

Our claim: a single BLASST-style softmax threshold skips a different fraction of 128x128 tiles for
text-query/text-key, vision/vision and cross tiles, motivating per-modality thresholds. Closest prior work:

1. **BLASST** (arXiv 2512.12087, v1 Dec 2025, v3 Apr 2026; MLSys 2026). Training-free dynamic block
   sparsity from a fixed scalar threshold on online-softmax statistics; works for prefill/decode and
   MHA/GQA/MQA/MLA; 1.52x prefill at 71.9 % sparsity. Reports that the optimal threshold shrinks as context
   grows (one threshold per model per phase). **Evaluated on LLMs only; no multimodal input** - our gap.
2. **MMInference** (arXiv 2504.16083, ICML 2025). Modality-aware permutation sparse attention for
   long-context VLM prefill: video tokens induce a *Grid* pattern from spatio-temporal locality; it detects
   Query-boundary and 2D-boundary patterns at modality boundaries and applies a modality-wise permutation to
   isolate intra-modality regions; per-head patterns found offline; up to 8.3x at 1M tokens; evaluated on
   Qwen2.5-VL among others; 52.3 % of heads need < 2 % attention recalled. Differentiation: it designs
   per-modality *patterns* offline; we ask whether a single *threshold* already yields modality-dependent
   skip rates, which is the pattern-free analogue.
3. **BlindSight** (arXiv 2507.09071, v3 Jan 2026). Input-template-aware sparsity for multi-image VLMs:
   heads classified as Dense / Sink / Intra-Image / Intra-Image+Sink from the prompt layout; no inter-image
   attention in many layers; 1.8-3.2x attention speedup at 36K-300K; generalises to Qwen2.5-VL; reports
   attention = 53 % of prefill time at 8K tokens on Qwen2.5-VL-32B. Closest to "sparsity depends on token
   modality/position", but static masks rather than threshold-driven.
4. **VideoNSA** (arXiv 2510.02295, ICLR 2026). Trains Qwen2.5-VL with a hybrid: dense attention for text,
   Native Sparse Attention for video tokens; scales to 128K. Explicit modality-dependent sparsity, but
   *trained* and token-selection based; we are training-free and tile-threshold based.
5. **Sparse VideoGen** (arXiv 2502.01776). Spatial vs temporal attention heads in video *diffusion*
   transformers with online profiling; 2.3x. Different setting (DiT generation, no text/vision asymmetry in
   an LLM), cite as evidence that video attention has structured sparsity.
6. **FastV** (arXiv 2403.06764). Visual tokens receive very inefficient attention in deep layers of LLaVA-1.5
   / QwenVL-Chat / Video-LLaVA; prunes half the image tokens after layer 2; 45 % FLOP reduction. Token
   pruning, not attention sparsity, but supports the hypothesis that vision-key tiles fall below threshold
   in later layers.
7. **SparseVLM** (arXiv 2410.04417, ICML 2025). Text-guided visual token pruning; motivates with visual
   tokens carrying less information than text tokens; 54 % FLOPs reduction at 97 % accuracy.
8. **VisionZip** (arXiv 2412.04467). Visual tokens from CLIP/SigLIP are highly redundant; selects informative
   tokens before the LLM; ~8x faster prefill.
9. **SparseVILA** (arXiv 2510.17777). Decouples visual sparsity: query-agnostic pruning in prefill,
   query-aware retrieval in decoding; up to 4.0x prefill on long video.

Differentiation statement for the report: items 6-9 remove or select *tokens* (sequence length shrinks);
items 2-4 choose *which blocks or tokens each head may attend to* using patterns or learned selection; BLASST
decides per tile from the softmax statistics with a single threshold. None of them measure how one such
threshold behaves across text vs vision tiles in the same forward pass, which is what we report.

---

## 8. Code sketches

### 8a. Loading with a custom attention function on the LLM only (transformers 4.57.3)

```python
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AttentionInterface
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv

BLASST_CTX = {"is_vision": None, "stats": []}   # set per forward from the driver

def blasst_attention_forward(module, query, key, value, attention_mask,
                             dropout=0.0, scaling=None, is_causal=None, **kwargs):
    # query (B, 28, L, 128); key/value (B, 4, L, 128). GQA: expand KV to 28 heads.
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    B, H, L, D = query.shape
    assert B == 1, "simulation assumes batch 1 (no padding mask)"
    scaling = scaling if scaling is not None else D ** -0.5
    causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    out = torch.empty_like(query)
    TILE = 128
    for q0 in range(0, L, TILE):                       # 128-query-row chunk
        q1 = min(q0 + TILE, L)
        k_end = q1 if causal else L                    # causal: keys <= last query of chunk
        s = torch.matmul(query[:, :, q0:q1], key[:, :, :k_end].transpose(2, 3)).float() * scaling
        if causal:                                     # mask inside the diagonal tile only
            qi = torch.arange(q0, q1, device=s.device)[:, None]
            kj = torch.arange(k_end, device=s.device)[None, :]
            s = s.masked_fill(kj > qi, float("-inf"))
        # --- BLASST tile rule goes here: per (head, key-tile) compare tile row-max with running row-max,
        # --- record skip flags, then zero the skipped tiles' contribution. Label tiles with
        # --- BLASST_CTX["is_vision"][q0:q1] (query side) and BLASST_CTX["is_vision"][:k_end] (key side),
        # --- store into BLASST_CTX["stats"] together with module.layer_idx.
        p = torch.softmax(s, dim=-1).to(query.dtype)
        out[:, :, q0:q1] = torch.matmul(p, value[:, :, :k_end])
    return out.transpose(1, 2).contiguous(), None      # (B, L, H, D)

AttentionInterface.register("blasst", blasst_attention_forward)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation={"text_config": "blasst", "vision_config": "sdpa"},  # ViT stays dense
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
# make stage-B limits match what we ask qwen_vl_utils for (see section 1.4)
processor.video_processor.size = {"shortest_edge": 32 * 28 * 28, "longest_edge": 768 * 28 * 28}
```

Sanity checks after loading: `model.config.text_config._attn_implementation == "blasst"` and
`model.config.vision_config._attn_implementation == "sdpa"`; run one short text-only prompt with
`"blasst"` vs `"sdpa"` and compare logits (max abs diff should be at bf16 noise level when no tiles are
skipped).

### 8b. Building the inputs and the per-position `is_vision` mask

```python
from qwen_vl_utils import process_vision_info   # qwen-vl-utils==0.0.11

VIDEO_PAD, IMAGE_PAD = 151656, 151655
VISION_START, VISION_END = 151652, 151653

subtitle_text = open("subtitles/xxx.srt").read()[:6000]      # gives a real text block (section 2.2)
messages = [{"role": "user", "content": [
    {"type": "text", "text": "Transcript:\n" + subtitle_text},
    {"type": "video", "video": "file:///data/videomme/xxx.mp4",
     "nframes": 128, "max_pixels": 256 * 28 * 28, "min_pixels": 32 * 28 * 28},   # case G, ~16K video tokens
    {"type": "text", "text": "Summarize what happens in the video."},
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                   padding=True, return_tensors="pt", **video_kwargs).to(model.device)

ids = inputs["input_ids"][0]                                   # (L,)
is_vision = (ids == VIDEO_PAD) | (ids == IMAGE_PAD)            # bool (L,)
print("L =", ids.numel(), "vision =", int(is_vision.sum()), "grid_thw =", inputs["video_grid_thw"].tolist())
assert int(is_vision.sum()) == int(inputs["video_grid_thw"].prod(-1).sum() // 4)

# tile-level labels for 128x128 tiles (query rows, key cols)
TILE = 128
n_tiles = (ids.numel() + TILE - 1) // TILE
frac_vision = torch.stack([is_vision[t*TILE:(t+1)*TILE].float().mean() for t in range(n_tiles)])
tile_kind = torch.where(frac_vision == 1, 2, torch.where(frac_vision == 0, 0, 1))   # 0 text, 1 mixed, 2 vision

BLASST_CTX["is_vision"] = is_vision
BLASST_CTX["stats"].clear()
with torch.inference_mode():
    out = model(**inputs, use_cache=False, logits_to_keep=1)   # prefill only
```

Then aggregate `BLASST_CTX["stats"]` into skip-rate by (layer, query-tile kind, key-tile kind).

### 8c. Environment pins

```
pip install "transformers==4.57.3" "qwen-vl-utils==0.0.11" accelerate torchvision   # or decord / torchcodec for faster decode
```

Set `FORCE_QWENVL_VIDEO_READER=torchvision` (or `decord`) to pin the decoder; the Qwen README warns decord
can hang on some files.

---

## 9. Known pitfalls (collected)

- Default video settings blow the context: 60 s of 720p at defaults = 43,200 tokens (> 32,768).
- `min_pixels * 1.05` floor in `fetch_video` overrides `total_pixels` for large `nframes`.
- The HF video processor re-resizes with `shortest_edge = 128*28*28`; set `processor.video_processor.size`.
- A plain string `attn_implementation="blasst"` also hits the ViT (non-causal, windowed); use the dict form.
- Custom attention functions receive `attention_mask=None`; implement causality yourself.
- K/V arrive with 4 heads; `repeat_kv` to 28.
- `eager` or any full-matrix path at 16K is 30 GB of fp32 scores -> OOM; chunk by 128 rows.
- Pass `logits_to_keep=1` or you pay 5 GB for logits.
- transformers < 4.53 silently has no AttentionInterface path for this model.
- qwen-vl-utils 0.0.13+ changed the env var name (`MODEL_SEQ_LEN`) and signature (`image_patch_size`).

---

## 10. Unverified (flagged, not confirmed by a primary source today)

- Video-MME having exactly 300 videos per duration class, and whether its zip chunks are uniformly mixed by
  duration (the 1-2 chunk estimate assumes they are).
- The `videoID`-to-filename convention inside `videos_chunked_XX.zip` (assumed `<videoID>.mp4`).
- `meta-llama/Llama-3.1-8B-Instruct` as the HF id and its gated status (standard, but not re-fetched).
- ModelScope CLI flags (`modelscope download --model ... --local_dir ...`); the Python `snapshot_download`
  API is the usual route.
- Whether `Qwen2_5_VLProcessor(..., do_resize=False)` is accepted for videos in 4.57.3 (the Qwen3-VL README
  recommends it for its processor; for Qwen2.5-VL I verified only the `size` attribute path).
- qwen-vl-utils 0.0.14 producing byte-identical frames to 0.0.11 for Qwen2.5-VL (same constants, rewritten
  code; not executed).
- The exact `smart_resize` rounding in qwen-vl-utils at commit a30e36f ("fix bug of smart_resize"); I used
  the `max(factor, floor_by_factor(...))` form that transformers 4.57.3 implements. Differences would be
  one 28-px step at most.
- Headroom figures in section 4 are arithmetic from the config, not measured; run `torch.cuda.max_memory_allocated()` once.
- BLASST v3 details (block size, exact lambda rule) were not re-read; the team already has the paper.
- Durations of individual LongVideoBench / MLVU videos beyond the published group ranges.
- The Blender archive.org mp4 frame rates and exact resolutions (listed as 720p / 2048-wide).

---

## 11. Sources consulted (2026-10-31)

- Qwen2.5-VL-7B-Instruct Hub files: `config.json`, `preprocessor_config.json`, `tokenizer_config.json`,
  `README.md`; `https://huggingface.co/api/models/Qwen/Qwen2.5-VL-7B-Instruct`.
- `QwenLM/Qwen2.5-VL` at commit `a30e36facd0a5131d9ed59e93210c7ac5de75adb` (2025-04-23):
  `qwen-vl-utils/src/qwen_vl_utils/vision_process.py`, `README.md`; current `main` (Qwen3-VL rewrite) for
  comparison; PyPI `qwen-vl-utils` release list (0.0.11 = 2025-04-21, 0.0.14 = 2025-09-23).
- transformers tags v4.51.3, v4.52.4, v4.53.0, v4.57.3 and `main`:
  `models/qwen2_5_vl/modeling_qwen2_5_vl.py`, `configuration_qwen2_5_vl.py`, `processing_qwen2_5_vl.py`,
  `models/qwen2_vl/video_processing_qwen2_vl.py`, `image_processing_qwen2_vl.py`,
  `docs/source/en/attention_interface.md`; Context7 excerpts of `integrations/sdpa_attention.py`,
  `masking_utils.py`, `modeling_utils.py`.
- Qwen2.5-VL Technical Report, arXiv 2502.13923 (HTML).
- Datasets: `lmms-lab/Video-MME` card + file tree; `BradyFU/Video-MME` README; `longvideobench/LongVideoBench`
  card; `MLVU/MVLU` card; archive.org items `BigBuckBunny_124`, `Sintel`.
- ModelScope API records for the three model ids; hf-mirror.com front page.
- arXiv abstracts: 2512.12087 (BLASST), 2504.16083 (MMInference), 2507.09071 (BlindSight),
  2510.02295 (VideoNSA), 2502.01776 (Sparse VideoGen), 2403.06764 (FastV), 2410.04417 (SparseVLM),
  2412.04467 (VisionZip), 2510.17777 (SparseVILA).
