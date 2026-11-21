"""
Shared helpers for the BLASST experiments (smoke test, Fig. 1 sweep, Fig. 2 RULER runs).

  load_model()            Llama-3.1-8B-Instruct in bf16 on the GPU
  load_ruler()            RULER samples from the local parquet copy of self-long/RULER-llama3-1M
  encode()                prompt text -> token ids (the prompt already contains the chat template)
  generate_answer()       greedy decoding, returns only the new text
  string_match_all()      RULER's score for niah / vt / fwe
  stack_margins()         per-layer margins recorded by blasst_attn -> one [layers, heads, q tiles, k tiles] tensor
  sparsity_curve()        % of causal tiles skipped at each lambda, computed from margins (no extra forward pass)
"""
from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Sequence

import pandas as pd
import torch

import blasst_attn as ba

W = "/work/nvme/beqg/zyang21/blasst"
DATA = f"{W}/data/ruler"
RESULTS = f"{W}/results"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# tokens to generate per task family (RULER constants.py)
GEN_TOKENS = {"niah": 128, "vt": 30, "fwe": 50}


def family(task: str) -> str:
    """'niah_multikey_2' -> 'niah', 'vt' -> 'vt', 'fwe' -> 'fwe'."""
    return "niah" if task.startswith("niah") else task


# ----------------------------------------------------------------------------- model and data
def load_model(attn: str = "sdpa"):
    """Tokenizer and model. attn = 'sdpa' (PyTorch standard attention) or 'blasst' (ours).
    Switch later with ba.use_blasst(model) or model.set_attn_implementation('sdpa')."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation=attn, device_map="cuda")
    model.eval()
    return tok, model


def load_ruler(task: str, length: str, n: int | None = None, start: int = 0) -> List[Dict]:
    """task e.g. 'niah_multikey_2', 'vt', 'fwe'; length '4k', '8k', '16k'.
    Returns n rows starting at row `start`, as dicts with keys row, index, input, answers, length.
    Convention: rows 0-49 are the accuracy test set (Fig. 2); calibration uses rows >= 100."""
    path = f"{DATA}/{task}_{length}/validation-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    df = df.iloc[start:] if n is None else df.iloc[start:start + n]
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        rows.append({"task": task, "length_name": length, "row": start + i, "index": int(r["index"]),
                     "input": r["input"], "answers": [str(a) for a in r["answers"]],
                     "length": int(r["length"])})
    return rows


def encode(tok, text: str) -> torch.Tensor:
    """Token ids [1, S] on the GPU. The RULER prompt already starts with <|begin_of_text|>,
    so do not let the tokenizer add a second one."""
    add_bos = not text.startswith(tok.bos_token)
    return tok(text, return_tensors="pt", add_special_tokens=add_bos).input_ids.cuda()


@torch.no_grad()
def generate_answer(tok, model, ids: torch.Tensor, max_new_tokens: int) -> str:
    """Greedy decoding (the paper uses greedy for long-context tasks). Returns the new text only."""
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                         do_sample=False, temperature=None, top_p=None, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def timed_prefill(model, ids: torch.Tensor) -> tuple[torch.Tensor, float]:
    """One prefill (no KV cache kept), returns (last-token logits [vocab] fp32, seconds)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model(ids, logits_to_keep=1, use_cache=False).logits[0, -1].float()
    torch.cuda.synchronize()
    return logits, time.perf_counter() - t0


def string_match_all(pred: str, refs: Sequence[str]) -> float:
    """RULER score for one sample: fraction of reference strings found in the prediction
    (case-insensitive), times 100."""
    return 100.0 * sum(r.lower() in pred.lower() for r in refs) / len(refs)


# ----------------------------------------------------------------------------- margins -> % skipped
def stack_margins() -> tuple[torch.Tensor, torch.Tensor]:
    """After a recorded forward pass: (margins [layers, heads, q tiles, k tiles] fp32 CPU,
    causal [q tiles, k tiles] bool). Assumes batch size 1 and one prefill call per layer."""
    m = ba.get_margins()
    layers = sorted(m)
    margins = torch.stack([m[l][0] for l in layers])               # drop the batch axis
    meta = ba.STATE["record_meta"][layers[0]]
    causal = ba.causal_tiles(meta["Sq"], meta["Skv"], meta["offset"])
    return margins, causal


def sparsity_curve(margins: torch.Tensor, causal: torch.Tensor, lams: Sequence[float]) -> List[float]:
    """Fraction of causal tiles skipped at each lambda: a tile is skipped iff margin < ln(lambda).
    Sorting the margins once makes this cheap for hundreds of lambdas."""
    vals = margins[..., causal].flatten().float().sort().values
    n = vals.numel()
    return [torch.searchsorted(vals, math.log(l), right=False).item() / n for l in lams]


def per_layer_sparsity(margins: torch.Tensor, causal: torch.Tensor, lam: float) -> List[float]:
    """Fraction of causal tiles skipped per layer at one lambda."""
    skip = margins[..., causal] < math.log(lam)                    # [layers, heads, causal tiles]
    return skip.float().mean(dim=(1, 2)).tolist()


def save_margins(path: str, margins: torch.Tensor, sample: Dict, n_tokens: int) -> None:
    """Hand-off file for teammates B and C: fp16 margins + sample id and length."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"margins": margins.half(), "task": sample["task"], "index": sample["index"],
                "length_name": sample["length_name"], "row": sample.get("row"), "n_tokens": n_tokens, "tile": ba.TILE,
                "note": "skip tile at lambda iff margin < ln(lambda); non-causal tiles are -inf"}, path)
