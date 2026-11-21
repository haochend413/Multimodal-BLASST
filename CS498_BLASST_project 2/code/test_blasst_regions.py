"""
Tests for blasst_regions.py (runs on CPU in a few seconds; no model needed).

 R1  one lambda everywhere: same output and same number of skipped tiles as blasst_attn.py.
 R2  a different lambda per region: the skipped tiles are exactly {margin < ln(lambda of the tile's region)},
     computed offline from the recorded margins, and the output equals a brute-force masked softmax.
 R3  probe mode: the output is dense (equals blasst_attn with skipping off), and the weight each row would
     lose at lambda equals a brute-force sum of the dense softmax over the tiles that lambda skips,
     split into video keys and other keys.
"""
import math
from types import SimpleNamespace

import torch

import blasst_attn as ba
import blasst_regions as br
from test_blasst import plant_sink

torch.manual_seed(0)
S, Hq, Hkv, D = 700, 4, 2, 64                       # 700: the last tile is partial
q = torch.randn(1, Hq, S, D)
k = torch.randn(1, Hkv, S, D)
v = torch.randn(1, Hkv, S, D)
q, k = plant_sink(q, k, strength=torch.linspace(7, 12, Hkv))
mod = SimpleNamespace(layer_idx=0)
scale = 1 / math.sqrt(D)
is_vision = torch.zeros(S, dtype=torch.bool)
is_vision[200:520] = True                           # blocks: 0 text, 1 mixed, 2-3 video, 4 mixed, 5 text
kinds = br.block_kinds(is_vision)
assert kinds.tolist() == [0, 1, 2, 2, 1, 0], kinds.tolist()


def brute_force(skip):                              # skip: [H, nq, nk] tile mask -> attention output
    kr, vr = k.repeat_interleave(Hq // Hkv, 1), v.repeat_interleave(Hq // Hkv, 1)
    s = (q @ kr.transpose(-1, -2))[0] * scale       # [H, S, S]
    s = s.masked_fill(torch.ones(S, S, dtype=torch.bool).triu(1), float("-inf"))
    m = skip.repeat_interleave(128, 1).repeat_interleave(128, 2)[:, :S, :S]
    p = torch.softmax(s.masked_fill(m, float("-inf")), -1)
    return (p @ vr[0]).transpose(0, 1)[None], torch.softmax(s, -1)   # output [1, S, H, D], dense p


# R1: one lambda everywhere == blasst_attn
for lam in (1e-3, 3e-2, 0.3):
    ba.set_threshold(lam); ba.reset_records(record=False)
    ref, _ = ba.blasst_attention_forward(mod, q, k, v, None, scaling=scale)
    n_ref = int(ba.STATE["stats"][0]["skipped"].sum())
    br.set_kinds(is_vision); br.set_table(default=lam); br.set_probe(); br.reset()
    out, _ = br.blasst_regions_forward(mod, q, k, v, None, scaling=scale)
    n_out = int(br.region_counts()["skipped"].sum())
    assert torch.equal(out, ref) and n_out == n_ref, (lam, n_out, n_ref)
    print(f"R1 ok  lambda {lam}: {n_out} tiles skipped, output identical to blasst_attn")
assert n_out > 0

# R2: per-region lambdas
table = {(br.VIDEO, br.VIDEO): 0.3, (br.TEXT, br.VIDEO): None, (br.TEXT, br.TEXT): 1e-2, (br.VIDEO, br.TEXT): 0.1,
         (br.MIXED, br.VIDEO): 0.05}
br.set_table(table, default=1e-3); br.reset(record=True)
out, _ = br.blasst_regions_forward(mod, q, k, v, None, scaling=scale)
margin = br.get_margins()[0]                        # [H, nq, nk]
ln_tab = br.STATE["ln_lam_table"]
expect = margin < ln_tab[kinds[:, None], kinds[None, :]][None]
want, _ = brute_force(expect)
assert torch.allclose(out, want, atol=1e-5), (out - want).abs().max()
causal = torch.isfinite(margin[0])
cnt = br.region_counts()
for qk in range(3):
    for kk in range(3):
        sel = causal & (kinds[:, None] == qk) & (kinds[None, :] == kk)
        assert int(cnt["skipped"][qk, kk]) == int((expect & sel).sum()), (qk, kk)
        assert int(cnt["causal"][qk, kk]) == int(sel.sum()) * Hq
assert int(cnt["skipped"][br.TEXT, br.VIDEO]) == 0
print(f"R2 ok  per-region lambdas: {int(expect[:, causal].sum())} tiles skipped, matches offline margins "
      f"and brute force (max diff {(out - want).abs().max():.1e}); text->video skipped: 0")

# R2b: relabel the query side (block 4, mixed, called "text"): its video tiles are then protected too
qk_ = kinds.clone(); qk_[4] = br.TEXT
br.set_kinds(is_vision, q_kinds=qk_)
br.set_table({(br.TEXT, br.VIDEO): None}, default=0.3); br.reset(record=True)
out, _ = br.blasst_regions_forward(mod, q, k, v, None, scaling=scale)
margin = br.get_margins()[0]
expect = margin < br.STATE["ln_lam_table"][qk_[:, None], kinds[None, :]][None]
assert not expect[:, 4, 2:4].any() and torch.allclose(out, brute_force(expect)[0], atol=1e-5)
assert int(br.region_counts()["skipped"][br.TEXT, br.VIDEO]) == 0
br.set_kinds(is_vision)
print(f"R2b ok query-side relabel: block 4 -> video tiles protected, {int(expect[:, causal].sum())} tiles skipped")

# R3: probe, on a weaker sink so the tiles a lambda skips still carry noticeable weight
torch.manual_seed(1)
k = torch.randn(1, Hkv, S, D)
k[:, :, 256:512] *= 0.1                             # key tiles 2-3 score low: skippable but not weightless
q, k = plant_sink(torch.randn(1, Hq, S, D), k, strength=torch.linspace(3.5, 5, Hkv))
lams = [0.3, 0.6]
br.set_table(None); br.set_probe(lams); br.reset(record=True)
out, _ = br.blasst_regions_forward(mod, q, k, v, None, scaling=scale)
ba.set_threshold(None); ba.reset_records(record=False)
ref, _ = ba.blasst_attention_forward(mod, q, k, v, None, scaling=scale)
assert torch.equal(out, ref)
margin = br.get_margins()[0]
_, p = brute_force(torch.zeros_like(margin, dtype=torch.bool))
res = br.STATE["probe"][0]
assert torch.allclose(res["dense_vid"].float(), p[..., is_vision].sum(-1), atol=2e-3)
for j, lam in enumerate(lams):
    skip_tok = (margin < math.log(lam)).repeat_interleave(128, 1).repeat_interleave(128, 2)[:, :S, :S]
    lost_v = (p * (skip_tok & is_vision[None, None, :])).sum(-1)
    lost_o = (p * (skip_tok & ~is_vision[None, None, :])).sum(-1)
    assert torch.allclose(res["lost_vid"][j].float(), lost_v, atol=2e-3)
    assert torch.allclose(res["lost_other"][j].float(), lost_o, atol=2e-3)
    print(f"R3 ok  probe lambda {lam}: mean weight lost per row {100 * (lost_v + lost_o).mean():.2f}% "
          f"(video keys {100 * lost_v.mean():.2f}%), {int((margin[:, causal] < math.log(lam)).sum())} tiles, matches brute force")
    assert (lost_v + lost_o).mean() > 1e-3, "probe test would be vacuous"
print("all blasst_regions tests passed")
