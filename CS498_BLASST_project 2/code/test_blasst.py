"""
Correctness tests for blasst_attn.py. Runs on CPU or a small GPU with a tiny
randomly initialised Llama (GQA) so no download is needed.

  python test_blasst.py            # all tests
  python test_blasst.py --cpu      # force CPU

Why the "sink": with purely random Q/K no key tile stands out, and the skip rule
needs ALL 128 rows of a tile to agree, so random data skips ~0% at every lambda
and a skip test would pass without testing anything. Real LLMs have an
"attention sink": almost every query puts a large score on the first token. We
plant one (plant_sink) so that tiles really get skipped.

Checks:
  T1  functional: with no skipping the kernel equals SDPA (fp32 tight, bf16 to bf16 precision),
      with a sequence length that is NOT a multiple of 128 and with GQA.
  T2  skip rule on sink data: every query tile keeps the tile holding its row maxima;
      % skipped is monotone in lambda and NOT always 0; margins recorded with skipping OFF
      reproduce the online decisions; skip mask AND output equal an independent,
      unchunked reference (oracle_attention). Run at S=1024 (MHA) and S=700 (GQA, partial tiles).
  T3  model-level: Llama through transformers with attn_implementation 'blasst' equals
      'sdpa' with no skipping; generate() (prefill + decode path) matches; with a planted
      sink and skipping on, the model's logits and % skipped equal the oracle's.
  T4  causal-tile accounting matches a brute-force count.
  T5  calibrated rule: set_threshold(a=...) equals fixed lambda = a/L; lambda > 1 is capped (no NaN).
"""
import argparse
import math

import torch
import torch.nn.functional as F

import blasst_attn as ba

TILE = 128


def _dev(force_cpu: bool) -> torch.device:
    return torch.device("cpu") if force_cpu or not torch.cuda.is_available() else torch.device("cuda")


class _Mod:  # stands in for the attention module (only layer_idx is read)
    def __init__(self, layer_idx=0):
        self.layer_idx = layer_idx


def sdpa_ref(q, k, v, scale):
    n_rep = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(n_rep, dim=1)
    v = v.repeat_interleave(n_rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale).transpose(1, 2)


# ----------------------------------------------------------------------------- reference
def oracle_attention(q, k, v, scale, lam):
    """Independent, unchunked implementation of the skip rule (plain prefill, Sq == Skv).

    Builds the whole [S, S] score matrix at once, cuts it into 128 x 128 tiles,
    and keeps a tile if ANY of its query rows has (tile row max - running row max) >= ln(lambda).
    Returns (out [B, S, Hq, D] fp32, keep [B, H, n_tiles, n_tiles] bool, causal [n_tiles, n_tiles] bool).
    """
    B, Hq, S, D = q.shape
    n_rep = Hq // k.shape[1]
    k = k.repeat_interleave(n_rep, dim=1).float()
    v = v.repeat_interleave(n_rep, dim=1).float()
    s = (q.float() @ k.transpose(-1, -2)) * scale                      # [B, H, S, S]
    future = torch.ones(S, S, dtype=torch.bool, device=q.device).triu(1)
    s = s.masked_fill(future, float("-inf"))

    n = math.ceil(S / TILE)
    P = n * TILE
    sp = F.pad(s, (0, P - S, 0, P - S), value=float("-inf"))          # pad both axes to a multiple of 128
    tiles = sp.view(B, Hq, n, TILE, n, TILE)                          # [B, H, q tile, row, k tile, col]
    tmax = tiles.amax(dim=-1)                                         # [B, H, q tile, row, k tile]
    run = tmax.cummax(dim=-1).values                                  # running max, includes current tile
    diff = tmax - run                                                 # NaN only for padded rows (-inf - -inf)
    diff = torch.nan_to_num(diff, nan=float("-inf"))                  # padded rows never vote "keep"
    lnl = math.log(lam) if lam is not None else float("-inf")
    keep = (diff >= lnl).any(dim=3)                                   # [B, H, q tile, k tile]

    keep_el = keep[:, :, :, None, :, None].expand(B, Hq, n, TILE, n, TILE).reshape(B, Hq, P, P)[:, :, :S, :S]
    s = s.masked_fill(~keep_el, float("-inf"))
    out = torch.softmax(s, dim=-1) @ v
    causal = ba.causal_tiles(S, S, 0).to(q.device)
    return out.transpose(1, 2), keep, causal


def plant_sink(q, k, strength=10.0, seed=0):
    """Give every query a large score on key 0, like the attention sink of real LLMs.

    q += strength * u and k[key 0] = strength * u, with u a random unit vector per KV head.
    Then q . k0 = strength^2 + strength * (q . u): a large, nearly row-independent score.
    Different strengths per head give different % skipped per head.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    Hq, Hkv, D = q.shape[1], k.shape[1], q.shape[-1]
    u = torch.randn(Hkv, D, generator=g)
    u = (u / u.norm(dim=-1, keepdim=True)).to(q.device, q.dtype)
    if torch.is_tensor(strength):
        st = strength.to(q.device, q.dtype)
    else:
        st = torch.full((Hkv,), float(strength), device=q.device, dtype=q.dtype)
    uk = st[:, None] * u                                              # [Hkv, D]
    uq = uk.repeat_interleave(Hq // Hkv, dim=0)                       # query head h uses KV head h // n_rep
    q = q + uq[None, :, None, :]
    k = k.clone()
    k[:, :, 0, :] = uk[None]
    return q, k


# ----------------------------------------------------------------------------- tests
def t1_functional(dev):
    torch.manual_seed(0)
    B, Hq, Hkv, S, D = 1, 8, 2, 700, 32          # 700 is not a multiple of 128
    scale = 1 / math.sqrt(D)
    for dtype, atol in [(torch.float32, 1e-5), (torch.bfloat16, 3e-2)]:
        q = torch.randn(B, Hq, S, D, device=dev, dtype=dtype)
        k = torch.randn(B, Hkv, S, D, device=dev, dtype=dtype)
        v = torch.randn(B, Hkv, S, D, device=dev, dtype=dtype)
        ba.set_threshold(None)
        ba.reset_records(record=False)
        out, _ = ba.blasst_attention_forward(_Mod(), q, k, v, None, scale)
        ref = sdpa_ref(q, k, v, scale)
        err = (out.float() - ref.float()).abs().max().item()
        assert out.shape == (B, S, Hq, D), out.shape
        assert err < atol, f"T1 {dtype}: max abs err {err:.3e} > {atol}"
        print(f"T1 ok  {str(dtype):16s} max|blasst - sdpa| = {err:.2e}  (no skipping)")
        st = ba.sparsity_from_stats()
        assert st["overall"] == 0.0, st["overall"]


def t2_skip_rule(dev, S, Hq, Hkv, D, seed):
    torch.manual_seed(seed)
    B = 1
    scale = 1 / math.sqrt(D)
    q = torch.randn(B, Hq, S, D, device=dev)
    k = torch.randn(B, Hkv, S, D, device=dev)
    v = torch.randn(B, Hkv, S, D, device=dev)
    q, k = plant_sink(q, k, strength=torch.linspace(7, 12, Hkv), seed=seed)

    # margins with skipping off
    ba.set_threshold(None)
    ba.reset_records(record=True)
    ba.blasst_attention_forward(_Mod(0), q, k, v, None, scale)
    margin = ba.get_margins()[0]                         # [B, H, n_qt, n_kt]
    causal = ba.causal_tiles(S, S, 0)
    assert margin.shape[-2:] == causal.shape

    # every query tile has a key tile with margin exactly 0 (the tile holding a row's max), so it keeps >= 1 tile
    best = margin[0].amax(dim=-1)                         # [H, n_qt]
    assert torch.allclose(best, torch.zeros_like(best)), best.min()
    assert torch.all(torch.isinf(margin[0][:, ~causal])), "non-causal tiles should be -inf"

    prev, partial = -1.0, False
    for lam in [1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 0.5]:
        offline = (ba.masks_from_margins(margin, lam)[0][:, causal]).double().mean().item()
        ba.set_threshold(lam)
        ba.reset_records(record=False)
        out, _ = ba.blasst_attention_forward(_Mod(0), q, k, v, None, scale)
        online = ba.sparsity_from_stats()["overall"]
        assert abs(offline - online) < 1e-9, (lam, offline, online)
        assert online >= prev - 1e-12, "skip fraction must be monotone in lambda"
        assert torch.isfinite(out).all(), "no row may lose all its tiles"

        # independent reference: same skip mask, same output
        ref_out, keep, _ = oracle_attention(q, k, v, scale, lam)
        skip_ours = ba.masks_from_margins(margin, lam)[0]              # [H, n_qt, n_kt], CPU
        keep = keep[0].cpu()
        assert torch.equal(skip_ours[:, causal], ~keep[:, causal]), f"skip mask differs from oracle at {lam}"
        err = (out - ref_out).abs().max().item()
        assert err < 1e-4, f"T2 output differs from oracle at lambda={lam}: {err:.2e}"

        partial |= 0.0 < online < 1.0
        prev = online
        print(f"T2 ok  S={S} Hq={Hq} Hkv={Hkv} lambda={lam:<7g} skipped {100*online:5.1f}% "
              f"(online == offline == oracle; max|out - oracle| {err:.1e})")
    assert partial and prev > 0.2, "test data never produced partial skipping: the test would be vacuous"
    ba.set_threshold(None)


def t3_model_level(dev):
    from transformers import AttentionInterface, LlamaConfig, LlamaForCausalLM
    torch.manual_seed(2)
    cfg = LlamaConfig(vocab_size=1000, hidden_size=256, intermediate_size=512, num_hidden_layers=3,
                      num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=4096,
                      tie_word_embeddings=False)
    model = LlamaForCausalLM(cfg).to(dev, torch.float32).eval()
    model.generation_config.pad_token_id = cfg.eos_token_id   # silences a harmless generate() warning
    ids = torch.randint(0, 1000, (1, 650), device=dev)

    model.set_attn_implementation("sdpa")
    with torch.no_grad():
        ref = model(ids).logits
        gen_ref = model.generate(ids, max_new_tokens=8, do_sample=False)

    ba.use_blasst(model)
    ba.set_threshold(None)
    ba.reset_records(record=True)
    with torch.no_grad():
        out = model(ids).logits
    err = (out - ref).abs().max().item()
    assert err < 1e-3, f"T3 logits differ: {err}"
    margins = ba.get_margins()
    assert sorted(margins) == [0, 1, 2], f"layer_idx not propagated: {sorted(margins)}"
    assert margins[0].shape == (1, 8, 6, 6), margins[0].shape
    print(f"T3 ok  model logits max err {err:.2e} vs sdpa; margins recorded for layers {sorted(margins)}, "
          f"shape {tuple(margins[0].shape)}")

    ba.reset_records(record=False)
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=8, do_sample=False)
    assert torch.equal(gen, gen_ref), "generate() mismatch (prefill+decode path)"
    print("T3 ok  generate() identical to sdpa (prefill via blasst, decode via SDPA)")

    # Skipping ON through the whole model. A random model has no sink, so plant one inside the
    # attention call (same planting for both), then compare our function with the oracle.
    lam = 0.05
    oracle_counts = {"skipped": 0, "causal": 0}

    def with_sink_blasst(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        query, key = plant_sink(query, key, strength=10.0, seed=module.layer_idx)
        return ba.blasst_attention_forward(module, query, key, value, attention_mask, scaling, dropout, **kw)

    def with_sink_oracle(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        query, key = plant_sink(query, key, strength=10.0, seed=module.layer_idx)
        o, keep, causal = oracle_attention(query, key, value, scaling, lam)
        oracle_counts["skipped"] += int((~keep[:, :, causal]).sum())
        oracle_counts["causal"] += int(causal.sum()) * keep.shape[0] * keep.shape[1]
        return o.to(query.dtype).contiguous(), None

    AttentionInterface.register("test_sink_blasst", with_sink_blasst)
    AttentionInterface.register("test_sink_oracle", with_sink_oracle)

    model.set_attn_implementation("test_sink_oracle")
    with torch.no_grad():
        ref_skip = model(ids).logits
    model.set_attn_implementation("test_sink_blasst")
    ba.set_threshold(lam)
    ba.reset_records(record=False)
    with torch.no_grad():
        out_skip = model(ids).logits
    st = ba.sparsity_from_stats()
    oracle_sp = oracle_counts["skipped"] / oracle_counts["causal"]
    err = (out_skip - ref_skip).abs().max().item()
    assert torch.isfinite(out_skip).all()
    assert 0.0 < st["overall"] < 1.0, st["overall"]
    assert abs(st["overall"] - oracle_sp) < 1e-9, (st["overall"], oracle_sp)
    assert err < 1e-3, f"T3 logits with skipping differ from oracle: {err}"
    print(f"T3 ok  sink + lambda={lam}: model skips {100*st['overall']:.1f}% (oracle {100*oracle_sp:.1f}%), "
          f"logits max err vs oracle {err:.2e}; per-layer {[round(x, 3) for x in st['per_layer'].values()]}")
    ba.set_threshold(None)
    ba.use_blasst(model)


def t4_causal_count():
    S = 700
    ct = ba.causal_tiles(S, S, 0)
    n = math.ceil(S / 128)
    bf = torch.zeros(n, n, dtype=torch.bool)
    for qt in range(n):
        for kt in range(n):
            q_last = min(qt * 128 + 127, S - 1)
            bf[qt, kt] = kt * 128 <= q_last
    assert torch.equal(ct, bf)
    print(f"T4 ok  causal tiles for S={S}: {int(ct.sum())} of {n*n}")


def t5_calibrated(dev):
    torch.manual_seed(5)
    B, H, S, D = 1, 4, 1000, 64
    scale = 1 / math.sqrt(D)
    q, k = plant_sink(torch.randn(B, H, S, D, device=dev), torch.randn(B, H, S, D, device=dev), 10.0)
    v = torch.randn(B, H, S, D, device=dev)

    a = 50.0                                      # lambda = a / L = 0.05 at L = 1000
    ba.set_threshold(a=a)
    ba.reset_records(record=False)
    out_a, _ = ba.blasst_attention_forward(_Mod(0), q, k, v, None, scale)
    sp_a = ba.sparsity_from_stats()["overall"]
    ba.set_threshold(a / S)
    ba.reset_records(record=False)
    out_l, _ = ba.blasst_attention_forward(_Mod(0), q, k, v, None, scale)
    sp_l = ba.sparsity_from_stats()["overall"]
    assert sp_a == sp_l and torch.equal(out_a, out_l), (sp_a, sp_l)

    ba.set_threshold(a=1e9)                        # lambda = 1e6 would skip everything -> capped at 1
    ba.reset_records(record=False)
    out_big, _ = ba.blasst_attention_forward(_Mod(0), q, k, v, None, scale)
    assert torch.isfinite(out_big).all(), "lambda > 1 must be capped, not produce NaN"
    try:
        ba.set_threshold(2.0)
        raise AssertionError("set_threshold(2.0) should raise")
    except ValueError:
        pass
    print(f"T5 ok  a={a:g} at L={S} == fixed lambda={a/S:g} ({100*sp_a:.1f}% skipped); "
          f"lambda>1 capped (finite output, {100*ba.sparsity_from_stats()['overall']:.1f}% skipped)")
    ba.set_threshold(None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    dev = _dev(args.cpu)
    print(f"device: {dev}; torch {torch.__version__}")
    t4_causal_count()
    t1_functional(dev)
    t2_skip_rule(dev, S=1024, Hq=4, Hkv=4, D=64, seed=1)
    t2_skip_rule(dev, S=700, Hq=8, Hkv=2, D=64, seed=3)
    t5_calibrated(dev)
    t3_model_level(dev)
    print("ALL TESTS PASSED")
