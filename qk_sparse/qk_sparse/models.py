"""Model adapters, one per modality.

Each adapter's only job is: load a small pretrained transformer, and hand the
runner a callable that does one forward pass over a realistic input.  The Q/K
capture and all metrics are shared, so adding a modality means adding ~30 lines
here and nothing else.

Deliberate choices:

* Everything is pretrained.  Attention structure in a randomly initialised model
  is noise, and measuring sparsity in noise is a way to get a confident wrong
  answer.
* Everything is small enough to run on one GPU, and most of it on CPU.
* ``query_block``/``key_block`` defaults scale with the sequence length a model
  actually produces.  VecAttention's 128 is meaningless for a model whose whole
  sequence is 42 tokens, so each adapter states its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch


@dataclass
class Sample:
    """One input to run a forward pass on."""

    sample_id: str
    run: Callable[[], None]


@dataclass
class AdapterSpec:
    """Static description of an adapter, readable without loading weights."""

    name: str
    modality: str
    model_id: str
    causal: bool
    query_block: int = 128
    key_block: int = 64
    min_seq_len: int = 256
    note: str = ""
    group_strides: Sequence[int] = ()
    """Strides at which to try a non-contiguous query group.

    Set for grid-structured modalities: the same patch one row down (image) or
    one tubelet later (video). Empty means "contiguous grouping is the only
    sensible one", as for plain text.
    """

    deltas: Optional[Sequence[int]] = None
    """Query offsets to probe for key-set agreement.

    Sequence distance is not the same thing as *semantic* distance once tokens
    carry a grid. A ViT at 518px has 37 patches per row, so the vertical
    neighbour of a query sits 37 positions away; VideoMAE tiles 14x14 patches
    per tubelet, so the same patch one step later in time is 196 away. Probing
    only 1/2/4/8 measures the raster direction and calls everything else
    "distant", which would hide exactly the structure video is supposed to have.
    """


class Adapter:
    """Base class.  Subclasses implement ``load`` and ``samples``."""

    spec: AdapterSpec

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        self.device = device
        self.dtype = dtype
        self.model = None
        self.load_warnings: List[str] = []

    def load(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def samples(self, count: int, data_dir: Optional[str]) -> Iterator[Sample]:
        raise NotImplementedError

    def _load_checked(self, cls, model_id: str, **extra):
        """``from_pretrained`` + a check that nothing important was invented.

        ``output_loading_info=True`` makes HF return what it could not match;
        we refuse to let that pass silently for attention weights.
        """
        model, info = cls.from_pretrained(
            model_id, output_loading_info=True, **self._hf_kwargs(), **extra
        )
        self.load_warnings.extend(check_loaded_weights(model_id, info))
        return model

    # Shared helper: HF models must route through SDPA or capture sees nothing.
    def _hf_kwargs(self) -> Dict[str, object]:
        return {"attn_implementation": "sdpa", **_dtype_kwarg(self.dtype)}


def check_loaded_weights(model_id: str, info: Dict[str, object]) -> List[str]:
    """Turn HuggingFace's loading info into hard warnings about Q/K integrity.

    A checkpoint whose key names have drifted from the current modelling code
    loads *almost* everything and silently random-initialises the rest.  When
    the rest includes attention projections or their biases, every number this
    package produces for that model is measuring noise.  VideoMAE's
    ``q_bias``/``v_bias`` parameterisation does exactly this against recent
    transformers.

    Returns a list of warnings; the ones mentioning attention are fatal to the
    interpretation and say so.
    """
    warnings: List[str] = []
    missing = [str(k) for k in info.get("missing_keys", [])]
    if not missing:
        return warnings
    attn = [k for k in missing if any(
        t in k for t in ("attention", "attn", ".q_", ".k_", "query", "key")
    )]
    if attn:
        warnings.append(
            f"{model_id}: {len(attn)} ATTENTION parameters were randomly "
            f"initialised (e.g. {attn[0]}). Q/K are partly random, so results "
            f"for this model are NOT valid. Pick a checkpoint that matches this "
            f"transformers version, or pin the version."
        )
    other = [k for k in missing if k not in attn]
    if other:
        warnings.append(
            f"{model_id}: {len(other)} non-attention parameters were randomly "
            f"initialised (e.g. {other[0]}); harmless for Q/K metrics."
        )
    return warnings


def _dtype_kwarg(dtype: torch.dtype) -> Dict[str, object]:
    """``torch_dtype`` was renamed to ``dtype`` in transformers 5.

    Both spellings have to work: a TensorRT-LLM container pins its own
    transformers, and passing the wrong one is either a hard TypeError or,
    worse, a silently ignored kwarg that loads the model in float32.
    """
    try:
        from transformers import __version__ as version

        major = int(version.split(".")[0])
    except Exception:
        major = 5
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------


class CausalLMAdapter(Adapter):
    """Decoder-only language model over long documents."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B", seq_len: int = 4096, **kw):
        super().__init__(**kw)
        self.seq_len = seq_len
        self.spec = AdapterSpec(
            name="text",
            modality="text",
            model_id=model_id,
            causal=True,
            query_block=128,
            key_block=64,
            min_seq_len=1024,
            note="long-form text, causal self-attention",
        )

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id)
        self.model = self._load_checked(
            AutoModelForCausalLM, self.spec.model_id
        ).to(self.device).eval()

    def samples(self, count: int, data_dir: Optional[str]) -> Iterator[Sample]:
        from .data import long_text_chunks

        for i, text in enumerate(long_text_chunks(count, self.seq_len, data_dir)):
            ids = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=self.seq_len
            ).input_ids.to(self.device)
            if ids.shape[1] < self.spec.min_seq_len:
                continue

            def run(ids=ids):
                with torch.inference_mode():
                    self.model(input_ids=ids)

            yield Sample(f"text-{i}", run)


# ---------------------------------------------------------------------------
# image
# ---------------------------------------------------------------------------


class VisionAdapter(Adapter):
    """Vision transformer over natural images (bidirectional, 2-D token grid)."""

    def __init__(self, model_id: str = "facebook/dinov2-base", image_size: int = 518, **kw):
        super().__init__(**kw)
        self.image_size = image_size
        self.spec = AdapterSpec(
            name="image",
            modality="image",
            model_id=model_id,
            causal=False,
            query_block=64,
            key_block=32,
            min_seq_len=512,
            note="ViT patches in raster order; neighbouring queries are "
            "horizontal neighbours in the image, not 2-D neighbours",
            # 518/14 = 37 patches per row: 37 = the patch directly below.
            deltas=(1, 2, 4, 8, 16, 37, 74, 111, 296),
            group_strides=(37,),
        )

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(
            self.spec.model_id, size={"height": self.image_size, "width": self.image_size},
            crop_size={"height": self.image_size, "width": self.image_size},
        )
        self.model = self._load_checked(
            AutoModel, self.spec.model_id
        ).to(self.device).eval()

    def samples(self, count: int, data_dir: Optional[str]) -> Iterator[Sample]:
        from .data import natural_images

        for i, image in enumerate(natural_images(count, data_dir)):
            pixel = self.processor(images=image, return_tensors="pt").pixel_values
            pixel = pixel.to(self.device, self.dtype)

            def run(pixel=pixel):
                with torch.inference_mode():
                    self.model(pixel_values=pixel)

            yield Sample(f"image-{i}", run)


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------


class VideoAdapter(Adapter):
    """Video transformer: the setting VecAttention's claim comes from.

    Tokens are a space-time grid, so neighbouring queries are spatial
    neighbours within a frame and, every ``H*W`` positions, the same patch one
    frame later.  If vertical vectors exist anywhere, they exist here.
    """

    def __init__(self, model_id: str = "MCG-NJU/videomae-base-finetuned-kinetics", num_frames: int = 16, **kw):
        super().__init__(**kw)
        self.num_frames = num_frames
        self.spec = AdapterSpec(
            name="video",
            modality="video",
            model_id=model_id,
            causal=False,
            query_block=128,
            key_block=64,
            min_seq_len=512,
            note="space-time tokens; VecAttention's home turf",
            # 14 patches per row, 196 per tubelet: 14 = below, 196 = same
            # patch one time step later -- the axis video sparsity lives on.
            deltas=(1, 2, 4, 8, 14, 28, 98, 196, 392, 784),
            group_strides=(14, 196),
        )

    def _repair_beit_style_biases(self) -> None:
        """Restore VideoMAE's q_bias/v_bias into the current bias layout.

        VideoMAE (like BEiT) stores attention bias as separate ``q_bias`` and
        ``v_bias`` tensors and gives the key projection **no bias at all** --
        the original implementation builds ``cat([q_bias, zeros, v_bias])``.
        Recent transformers expects plain ``query/key/value.bias``, does not
        convert, and random-initialises all three.

        So the repair is exact rather than a guess: q_bias -> query.bias,
        v_bias -> value.bias, key.bias -> zeros, which is what the checkpoint
        actually means.
        """
        import re

        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(self.spec.model_id, "model.safetensors")
            ckpt = load_file(path)
        except Exception as exc:
            self.load_warnings.append(
                f"{self.spec.model_id}: could not repair attention biases "
                f"({type(exc).__name__}); results remain invalid"
            )
            return

        state = dict(self.model.state_dict())
        repaired = 0
        for key, tensor in ckpt.items():
            match = re.search(r"encoder\.layer\.(\d+)\.attention\.attention\.([qv])_bias$", key)
            if not match:
                continue
            layer, which = match.group(1), match.group(2)
            target = f"encoder.layer.{layer}.attention.attention." \
                     f"{'query' if which == 'q' else 'value'}.bias"
            if target in state and state[target].shape == tensor.shape:
                state[target] = tensor.to(state[target].dtype)
                repaired += 1

        # Key bias is genuinely absent from the checkpoint: it must be zero,
        # not the random values nn.Linear initialised it with.
        zeroed = 0
        for name in list(state):
            if name.endswith("attention.attention.key.bias"):
                state[name] = torch.zeros_like(state[name])
                zeroed += 1

        if repaired:
            self.model.load_state_dict(state, strict=False)
            self.load_warnings = [
                w for w in self.load_warnings if "NOT valid" not in w
            ]
            self.load_warnings.append(
                f"{self.spec.model_id}: repaired {repaired} q/v biases from the "
                f"checkpoint and zeroed {zeroed} key biases (BEiT-style layout); "
                f"Q/K now match the pretrained model."
            )

    def load(self) -> None:
        from transformers import AutoModel, AutoVideoProcessor

        try:
            self.processor = AutoVideoProcessor.from_pretrained(self.spec.model_id)
        except Exception:  # older transformers
            from transformers import AutoImageProcessor

            self.processor = AutoImageProcessor.from_pretrained(self.spec.model_id)
        self.model = self._load_checked(
            AutoModel, self.spec.model_id
        ).to(self.device).eval()
        if any("NOT valid" in w for w in self.load_warnings):
            self._repair_beit_style_biases()

    def samples(self, count: int, data_dir: Optional[str]) -> Iterator[Sample]:
        from .data import video_clips

        for i, frames in enumerate(video_clips(count, self.num_frames, data_dir)):
            inputs = self.processor(list(frames), return_tensors="pt")
            pixel = inputs["pixel_values"].to(self.device, self.dtype)

            def run(pixel=pixel):
                with torch.inference_mode():
                    self.model(pixel_values=pixel)

            yield Sample(f"video-{i}", run)


# ---------------------------------------------------------------------------
# audio  (long sequences, non-causal, 1-D)
# ---------------------------------------------------------------------------


class AudioAdapter(Adapter):
    """Speech encoder: 1500 tokens of 1-D signal, no 2-D structure.

    Useful as a contrast to video: if vertical vectors show up here too, the
    effect is about sequence locality in general rather than anything specific
    to space-time tokens.
    """

    def __init__(self, model_id: str = "openai/whisper-small", **kw):
        super().__init__(**kw)
        self.spec = AdapterSpec(
            name="audio",
            modality="audio",
            model_id=model_id,
            causal=False,
            query_block=128,
            key_block=64,
            min_seq_len=512,
            note="Whisper encoder, fixed 1500-token context",
        )

    def load(self) -> None:
        from transformers import AutoProcessor, WhisperModel

        self.processor = AutoProcessor.from_pretrained(self.spec.model_id)
        self.model = self._load_checked(
            WhisperModel, self.spec.model_id
        ).encoder.to(self.device).eval()

    def samples(self, count: int, data_dir: Optional[str]) -> Iterator[Sample]:
        from .data import speech_clips

        for i, (audio, rate) in enumerate(speech_clips(count, data_dir)):
            feats = self.processor(
                audio, sampling_rate=rate, return_tensors="pt"
            ).input_features.to(self.device, self.dtype)

            def run(feats=feats):
                with torch.inference_mode():
                    self.model(feats)

            yield Sample(f"audio-{i}", run)


# ---------------------------------------------------------------------------
# time series
# ---------------------------------------------------------------------------


class TimeSeriesAdapter(Adapter):
    """Patch-based forecaster.

    Caveat worth stating up front: PatchTST turns a 512-step window into ~42
    patch tokens.  That is two orders of magnitude shorter than the video case,
    so block sizes drop to 8/4 and the results answer "do nearby queries agree"
    at a completely different scale.  Treat this as a probe of whether the
    effect is universal, not as a comparable data point.
    """

    def __init__(self, model_id: str = "ibm-granite/granite-timeseries-patchtst", **kw):
        super().__init__(**kw)
        self.spec = AdapterSpec(
            name="timeseries",
            modality="timeseries",
            model_id=model_id,
            causal=False,
            query_block=8,
            key_block=4,
            min_seq_len=32,
            note="short token sequences (~42 patches); not scale-comparable",
        )

    def load(self) -> None:
        from transformers import PatchTSTForPrediction

        self.model = self._load_checked(
            PatchTSTForPrediction, self.spec.model_id
        ).to(self.device).eval()
        self.context_length = int(self.model.config.context_length)
        self.num_channels = int(self.model.config.num_input_channels)

    def samples(self, count: int, data_dir: Optional[str]) -> Iterator[Sample]:
        import glob as _glob
        import os as _os

        from .data import SYNTHETIC_TIMESERIES_NOTE, time_series_windows

        # The loader silently falls back to a synthetic signal when no CSV is
        # present. That is fine for a sanity run and fatal for a claim about
        # real time series, so the record has to say which one it was.
        has_csv = bool(data_dir and _glob.glob(_os.path.join(data_dir, "*.csv")))
        if not has_csv:
            self.load_warnings.append(SYNTHETIC_TIMESERIES_NOTE)

        for i, window in enumerate(
            time_series_windows(count, self.context_length, self.num_channels, data_dir)
        ):
            values = window.unsqueeze(0).to(self.device, self.dtype)

            def run(values=values):
                with torch.inference_mode():
                    self.model(past_values=values)

            yield Sample(f"timeseries-{i}", run)


# ---------------------------------------------------------------------------
# synthetic control
# ---------------------------------------------------------------------------


class SyntheticAdapter(Adapter):
    """Ground-truth control that needs no weights and no network.

    Generates Q/K with a *known* amount of local query correlation, so the
    metrics can be checked against an answer we already know:

    * ``coherence=1.0`` -- queries inside a block are identical, so vertical
      vectors must be exactly as good as the per-row oracle.
    * ``coherence=0.0`` -- queries are independent, so vertical vectors must
      collapse towards the block/line patterns and top-key overlap must sit at
      the chance line.

    Any real-model number is only believable if this control behaves.
    """

    def __init__(self, coherence: float = 0.9, seq_len: int = 2048, heads: int = 4,
                 head_dim: int = 64, **kw):
        super().__init__(**kw)
        self.coherence = coherence
        self.seq_len = seq_len
        self.heads = heads
        self.head_dim = head_dim
        self.spec = AdapterSpec(
            name=f"synthetic-c{coherence:g}",
            modality="synthetic",
            model_id="none",
            causal=False,
            query_block=128,
            key_block=64,
            min_seq_len=256,
            note=f"control with local query coherence={coherence:g}",
        )

    def load(self) -> None:
        self.model = torch.nn.Identity()

    def samples(self, count: int, data_dir: Optional[str]) -> Iterator[Sample]:
        import torch.nn.functional as F

        for i in range(count):
            g = torch.Generator(device="cpu").manual_seed(1234 + i)
            n, p = self.seq_len, self.spec.query_block
            shared = torch.randn(self.heads, n // p, self.head_dim, generator=g)
            shared = shared.repeat_interleave(p, dim=1)[:, :n]
            noise = torch.randn(self.heads, n, self.head_dim, generator=g)
            c = self.coherence
            q = (c * shared + (1 - c) * noise).unsqueeze(0)
            k = torch.randn(1, self.heads, n, self.head_dim, generator=g)
            # V must be real content, not zeros: the output-error metric
            # compares A@V against a sparse A@V, and a zero V makes every
            # pattern look perfect.
            v = torch.randn(1, self.heads, n, self.head_dim, generator=g)
            q, k, v = (t.to(self.device, self.dtype) for t in (q, k, v))

            def run(q=q, k=k, v=v):
                with torch.inference_mode():
                    F.scaled_dot_product_attention(q, k, v)

            yield Sample(f"synthetic-{i}", run)


# ---------------------------------------------------------------------------
# multimodal LLM (video understanding) -- VecAttention's VLM setting
# ---------------------------------------------------------------------------


class VideoLLMAdapter(Adapter):
    """Qwen2.5-VL over video: the model family VecAttention reports VLM results on.

    Two things make this worth the GPU it costs. It is a *decoder* over a mixed
    video+text sequence, so unlike VideoMAE the attention is causal and the
    video tokens sit next to instruction tokens. And it reaches tens of
    thousands of tokens, which is the scale the vertical-vector claim was made
    at -- an order of magnitude past anything else in this suite.

    Grid strides are read from the processor's ``video_grid_thw`` rather than
    hardcoded: after 2x2 spatial merging a row is ``W/2`` tokens and a frame is
    ``H*W/4``, and both depend on the resolution the processor happened to
    choose for the clip.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 num_frames: int = 32, max_pixels: int = 200704,
                 video_max_pixels: Optional[int] = None, **kw):
        super().__init__(**kw)
        self.num_frames = num_frames
        self.max_pixels = max_pixels  # per frame; caps sequence length
        self.video_max_pixels = video_max_pixels
        self.spec = AdapterSpec(
            name="vlm",
            modality="video-llm",
            model_id=model_id,
            causal=True,
            query_block=128,
            key_block=64,
            # Qwen's processor resamples video by fps and can return far fewer
            # frames than handed to it, so the prefill may be shorter than the
            # arithmetic suggests. Gate low and let the shape census report what
            # actually happened rather than silently capturing nothing.
            min_seq_len=1024,
            note="causal decoder over video+text tokens; grid strides derived "
                 "from video_grid_thw at runtime",
            deltas=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048),
        )

    def load(self) -> None:
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as _Model
        except ImportError as exc:  # older transformers
            raise ImportError(
                "Qwen2.5-VL needs transformers>=4.49: pip install -U transformers"
            ) from exc

        self.processor = AutoProcessor.from_pretrained(
            self.spec.model_id, max_pixels=self.max_pixels
        )

        # Qwen's *video* processor carries its own pixel budget, independent of
        # the image `max_pixels` above -- which is why a clip comes back resized
        # far below the requested resolution (1612 tokens instead of ~20k). The
        # attribute has moved between transformers versions, so set whichever
        # exists and report what actually took effect rather than assuming.
        if self.video_max_pixels:
            targets = [
                getattr(self.processor, "video_processor", None),
                getattr(self.processor, "image_processor", None),
            ]
            applied = []
            for obj in targets:
                if obj is None:
                    continue
                for attr in ("max_pixels", "video_max_pixels"):
                    if hasattr(obj, attr):
                        setattr(obj, attr, self.video_max_pixels)
                        applied.append(f"{type(obj).__name__}.{attr}")
                size = getattr(obj, "size", None)
                if isinstance(size, dict) and "longest_edge" in size:
                    size["longest_edge"] = self.video_max_pixels
                    applied.append(f"{type(obj).__name__}.size.longest_edge")
            self.load_warnings.append(
                f"video pixel budget -> {self.video_max_pixels} via "
                f"{', '.join(applied) if applied else 'NOTHING (attribute not found)'}"
            )
        self.model = self._load_checked(_Model, self.spec.model_id).to(
            self.device).eval()

    def samples(self, count: int, data_dir: Optional[str]) -> Iterator[Sample]:
        from PIL import Image

        from .data import video_clips

        for i, frames in enumerate(video_clips(count, self.num_frames, data_dir)):
            images = [Image.fromarray(f).convert("RGB") for f in frames]
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": "Describe what happens in this video."},
                ],
            }]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], videos=[images], return_tensors="pt"
            ).to(self.device)

            # Derive the token grid so strided grouping probes the temporal axis.
            n_tokens = int(inputs["input_ids"].shape[-1])
            grid = inputs.get("video_grid_thw")
            if grid is not None:
                self.load_warnings.append(
                    f"sample {i}: {n_tokens} prefill tokens, "
                    f"video_grid_thw={[int(v) for v in grid[0]]}"
                )
            if grid is not None:
                merge = int(getattr(self.processor.image_processor, "merge_size", 2))
                _, h, w = (int(v) for v in grid[0])
                row, frame = max(1, w // merge), max(1, (h * w) // (merge * merge))
                self.spec.group_strides = (row, frame)
                self.spec.deltas = tuple(sorted({
                    1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                    row, 2 * row, frame, 2 * frame,
                }))

            def run(inputs=inputs):
                with torch.inference_mode():
                    self.model(**inputs)

            yield Sample(f"vlm-{i}", run)


# ---------------------------------------------------------------------------
# video diffusion transformer -- VecAttention's generation setting
# ---------------------------------------------------------------------------


class VideoDiTAdapter(Adapter):
    """Wan text-to-video through diffusers: a real video DiT.

    This is the closest thing in the suite to what VecAttention actually
    measures -- bidirectional self-attention over a space-time latent grid,
    thousands to tens of thousands of tokens, at a mid-trajectory denoising
    step. Diffusers' attention processors call
    ``F.scaled_dot_product_attention``, so the existing capture works unchanged
    and none of their kernel stack is needed.

    Cross-attention to the text embedding also passes through SDPA, but it has
    ``n_key != n_query`` and the capture filters it out: sharing keys between
    neighbouring *video* queries is the question, and a 512-token text
    conditioning sequence would only dilute it.
    """

    def __init__(self, model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                 height: int = 480, width: int = 832, num_frames: int = 17,
                 steps: int = 20, capture_step: int = 10, **kw):
        super().__init__(**kw)
        self.height, self.width = height, width
        self.num_frames, self.steps = num_frames, steps
        # Attention structure changes a lot along the denoising trajectory:
        # early steps are near-noise and diffuse, later ones have structure.
        # VecAttention dumps at step 5 or 25 of 50, so capturing step 0 (what
        # steps=2 gave) measured the least representative point of the run.
        self.capture_step = capture_step
        self.spec = AdapterSpec(
            name="video-dit",
            modality="video-dit",
            model_id=model_id,
            causal=False,
            query_block=128,
            key_block=64,
            min_seq_len=2048,
            note="Wan T2V latent tokens; strides computed from the latent grid",
            deltas=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
        )

    def _latent_grid(self) -> Tuple[int, int, int]:
        """(frames, rows, cols) of patch tokens for the configured resolution.

        Wan's VAE downsamples 8x spatially and 4x temporally, and the DiT then
        patchifies 2x2, so a token covers 16 pixels per side.
        """
        rows = self.height // 16
        cols = self.width // 16
        frames = (self.num_frames - 1) // 4 + 1
        return frames, rows, cols

    def load(self) -> None:
        try:
            from diffusers import DiffusionPipeline
        except ImportError as exc:
            raise ImportError(
                "the video-dit adapter needs diffusers: pip install diffusers"
            ) from exc

        self.pipe = DiffusionPipeline.from_pretrained(
            self.spec.model_id, torch_dtype=self.dtype
        )
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.model = self.pipe

        frames, rows, cols = self._latent_grid()
        self.spec.group_strides = (cols, rows * cols)
        # The probe must sample the grid's own offsets, or the overlap table
        # reports only d=1 and every grid-aligned distance comes back blank.
        self.spec.deltas = tuple(sorted({
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
            cols, 2 * cols, rows * cols, 2 * rows * cols,
        }))
        self.load_warnings.append(
            f"latent token grid {frames}x{rows}x{cols} = {frames * rows * cols} "
            f"tokens; strides row={cols}, frame={rows * cols}"
        )

    def samples(self, count: int, data_dir: Optional[str]) -> Iterator[Sample]:
        prompts = [
            "A cat walks across a sunlit kitchen floor, camera slowly panning.",
            "Waves crash against dark rocks at sunset, seabirds circling above.",
            "A cyclist rides down a tree-lined street, leaves falling.",
            "Steam rises from a cup of coffee on a wooden table by a window.",
        ]
        from .capture import set_capture_enabled

        class _StopAfterCaptureStep(Exception):
            """Abort the pipeline once the target step has been recorded."""

        for i in range(min(count, len(prompts))):
            def run(prompt=prompts[i]):
                target = max(0, min(self.capture_step, self.steps - 1))

                def on_step_end(pipe, step, timestep, kwargs):
                    # Record only the target step: enable just before it runs,
                    # disable and bail out as soon as it has.
                    if step == target - 1:
                        set_capture_enabled(True)
                    elif step >= target:
                        set_capture_enabled(False)
                        raise _StopAfterCaptureStep
                    return kwargs

                set_capture_enabled(target == 0)
                try:
                    with torch.inference_mode():
                        self.pipe(
                            prompt=prompt,
                            height=self.height,
                            width=self.width,
                            num_frames=self.num_frames,
                            num_inference_steps=self.steps,
                            callback_on_step_end=on_step_end,
                        )
                except _StopAfterCaptureStep:
                    pass
                finally:
                    set_capture_enabled(True)

            yield Sample(f"video-dit-{i}", run)


# ---------------------------------------------------------------------------
# replay of externally dumped Q/K
# ---------------------------------------------------------------------------


class QKDumpAdapter(Adapter):
    """Analyse Q/K tensors dumped by another codebase.

    VecAttention's ``dump_qk_layers.sh`` writes, per prompt and per layer,
    ``<dir>/<input_id>/layer_<i>_qk.pt`` holding
    ``{"query_states": [B,H,N,D], "key_states": [B,H,N,D]}`` -- the real
    post-RoPE tensors of Wan / HunyuanVideo / Qwen2.5-VL at a chosen diffusion
    step. That is the exact setting the vertical-vector claim was made in, and
    at 30k-100k tokens rather than the 1.5k our small models reach.

    Reading that layout costs nothing and imports nothing: it is plain
    ``torch.save`` output. This adapter replays those tensors through the same
    metrics as every other modality, so a video DiT becomes just another row in
    the table.

    Point ``--data-dir`` at either the per-prompt directory or its parent.
    """

    def __init__(self, dump_dir: Optional[str] = None, causal: bool = False,
                 modality: str = "video-dit",
                 group_strides: Sequence[int] = (), **kw):
        super().__init__(**kw)
        self.dump_dir = dump_dir
        self.spec = AdapterSpec(
            name="qk-dump",
            modality=modality,
            model_id="external dump",
            causal=causal,
            query_block=128,
            key_block=64,
            min_seq_len=512,
            note="Q/K replayed from an external dump (e.g. VecAttention's "
                 "dump_qk_layers.sh); scale and layout come from the file",
            deltas=(1, 2, 4, 8, 16, 32, 64, 128, 196, 256, 392, 512, 1024, 4096),
            group_strides=group_strides or (),
        )

    def load(self) -> None:
        self.model = None  # nothing to load; tensors come from disk

    @staticmethod
    def _find_layer_files(root: "Path") -> List["Path"]:
        files = sorted(root.glob("layer_*_qk.pt"))
        if files:
            return files
        # parent of one or more per-prompt directories
        for child in sorted(p for p in root.iterdir() if p.is_dir()):
            files = sorted(child.glob("layer_*_qk.pt"))
            if files:
                return files
        return []

    def iter_captures(self, samples: int, layers: int, heads: Optional[int],
                      data_dir: Optional[str]):
        """Yield ``AttentionCall``s straight from disk.

        The runner uses this instead of its two-pass SDPA capture when an
        adapter provides it -- there is no forward pass to intercept here, and
        running one on a 100k-token tensor just to observe it would be absurd.
        """
        import re
        from pathlib import Path

        from .capture import AttentionCall

        root = Path(data_dir or self.dump_dir or "")
        if not root.is_dir():
            raise FileNotFoundError(
                f"Q/K dump directory not found: {root!s}. Pass --data-dir "
                f"pointing at a directory containing layer_<i>_qk.pt files."
            )
        files = self._find_layer_files(root)
        if not files:
            raise FileNotFoundError(f"no layer_*_qk.pt under {root!s}")

        def layer_index(path) -> int:
            match = re.search(r"layer_(\d+)_qk", path.name)
            return int(match.group(1)) if match else 0

        files.sort(key=layer_index)
        if layers < len(files):
            step = (len(files) - 1) / max(1, layers - 1)
            chosen = sorted({files[min(len(files) - 1, int(round(i * step)))]
                             for i in range(layers)}, key=layer_index)
        else:
            chosen = files

        for path in chosen:
            blob = torch.load(path, map_location="cpu")
            q = blob["query_states"]
            k = blob["key_states"]
            if q.dim() == 4:
                q, k = q[0], k[0]  # [B,H,N,D] -> [H,N,D]
            if q.shape[0] > q.shape[1]:  # [N,H,D] layouts appear in some dumps
                q, k = q.transpose(0, 1), k.transpose(0, 1)
            if heads:
                q, k = q[:heads], k[:heads]
            if k.shape[0] != q.shape[0] and q.shape[0] % k.shape[0] == 0:
                k = k.repeat_interleave(q.shape[0] // k.shape[0], dim=0)
            yield AttentionCall(
                index=layer_index(path),
                query=q.to(self.device),
                key=k.to(self.device),
                is_causal=self.spec.causal,
                had_mask=False,
                scale=None,
                )

    def samples(self, count: int, data_dir: Optional[str]) -> Iterator[Sample]:
        return iter(())  # unused; the runner takes the iter_captures path


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

DEFAULT_DELTAS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

REGISTRY: Dict[str, Callable[..., Adapter]] = {
    "text": CausalLMAdapter,
    "text-small": lambda **kw: CausalLMAdapter(model_id="HuggingFaceTB/SmolLM2-135M", **kw),
    "image": VisionAdapter,
    "image-vit": lambda **kw: VisionAdapter(model_id="google/vit-base-patch16-224", image_size=224, **kw),
    "video": VideoAdapter,
    "video-timesformer": lambda **kw: VideoAdapter(
        model_id="facebook/timesformer-base-finetuned-k400", **kw
    ),
    "audio": AudioAdapter,
    "timeseries": TimeSeriesAdapter,
    "vlm": VideoLLMAdapter,
    "vlm-3b": lambda **kw: VideoLLMAdapter(model_id="Qwen/Qwen2.5-VL-3B-Instruct", **kw),
    # ~4x the tokens: raises the video processor's own pixel budget, which the
    # image-level max_pixels does not control.
    "vlm-long": lambda **kw: VideoLLMAdapter(video_max_pixels=602112, **kw),
    "video-dit": VideoDiTAdapter,
    # Wan's native 81-frame config: 21 latent frames instead of 5, so a
    # temporal-stride query group finally has enough rows to be compared
    # fairly against a contiguous one (5 rows could never say anything).
    "video-dit-long": lambda **kw: VideoDiTAdapter(num_frames=81, **kw),
    "video-dit-square": lambda **kw: VideoDiTAdapter(
        height=256, width=256, num_frames=81, **kw),
    "qk-dump": QKDumpAdapter,
    "qk-dump-grid": lambda **kw: QKDumpAdapter(group_strides=(196,), **kw),
    "qk-dump-causal": lambda **kw: QKDumpAdapter(causal=True, modality="vlm", **kw),
    "synthetic-high": lambda **kw: SyntheticAdapter(coherence=0.95, **kw),
    "synthetic-low": lambda **kw: SyntheticAdapter(coherence=0.0, **kw),
}

DEFAULT_SUITE = ["synthetic-high", "synthetic-low", "text", "image", "video", "audio", "timeseries"]


def build(name: str, **kwargs) -> Adapter:
    if name not in REGISTRY:
        raise KeyError(f"unknown adapter {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
