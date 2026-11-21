"""
How % skipped depends on the tile shape (needs the Q/K saved by fig1_sweep.py).

The skip rule is the same everywhere: a tile is skipped iff every query row in it has
(tile row max - running row max) < ln(lambda). What changes is the tile:
  Br = query rows per tile (all of them must agree), Bc = keys per tile.
Br = 128, Bc = 128 is our setup and NVIDIA's reference code. Br = 1 means every row decides alone.

  python tile_sensitivity.py results/qk/8k/*.pt
Writes results/tile_sensitivity_<length>.csv (% of causal tiles skipped per Br, Bc, lambda).
"""
import csv
import math
import sys

import torch
import torch.nn.functional as F

BRS = [128, 64, 32, 16, 1]
BCS = [128, 64]
LAMS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
CHUNK = 128                                     # rows processed at once (a multiple of every Br)
ln_lams = torch.tensor([math.log(l) for l in LAMS], device="cuda")

skipped = {(br, bc): torch.zeros(len(LAMS), dtype=torch.float64) for br in BRS for bc in BCS}
total = {(br, bc): 0 for br in BRS for bc in BCS}
files = sys.argv[1:]
for path in files:
    d = torch.load(path)
    for layer, (q, k) in sorted(d["qk"].items()):
        q = q.cuda().float()                              # [1, 32, S, 128]
        k = k.cuda().float().repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        S, D = q.shape[2], q.shape[3]
        scale = 1 / math.sqrt(D)
        kpos = torch.arange(S, device="cuda")
        for r0 in range(0, S, CHUNK):
            r1 = min(r0 + CHUNK, S)
            s = (q[:, :, r0:r1] @ k.transpose(-1, -2)) * scale            # [1, H, r, S]
            qpos = torch.arange(r0, r1, device="cuda")
            s = s.masked_fill(kpos[None, :] > qpos[:, None], float("-inf"))
            for bc in BCS:
                nk = math.ceil(S / bc)
                sp = F.pad(s, (0, nk * bc - S), value=float("-inf"))
                tmax = sp.view(*sp.shape[:3], nk, bc).amax(-1)              # [1, H, r, nk]
                run = tmax.cummax(-1).values
                diff = torch.where(torch.isfinite(run), tmax - run, torch.full_like(tmax, float("-inf")))
                k_first = torch.arange(nk, device="cuda") * bc
                for br in BRS:
                    r = r1 - r0
                    nr = math.ceil(r / br)
                    dpad = F.pad(diff, (0, 0, 0, nr * br - r), value=float("-inf"))   # padded rows never veto
                    margin = dpad.view(1, dpad.shape[1], nr, br, nk).amax(3)          # [1, H, nr, nk]
                    q_last = torch.clamp(r0 + (torch.arange(nr, device="cuda") + 1) * br - 1, max=r1 - 1)
                    causal = k_first[None, :] <= q_last[:, None]                       # [nr, nk]
                    m = margin[0][:, causal]                                           # [H, n causal]
                    skipped[(br, bc)] += (m[None] < ln_lams[:, None, None]).sum(dim=(1, 2)).double().cpu()
                    total[(br, bc)] += m.numel()
    print(f"done {path}", flush=True)

length = files[0].rstrip("/").split("/")[-2]           # results/qk/<length>/<file>.pt
out = f"/work/nvme/beqg/zyang21/blasst/results/tile_sensitivity_{length}.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["Br", "Bc", "lam", "pct_skipped", "n_files"])
    for (br, bc), v in skipped.items():
        for lam, x in zip(LAMS, v.tolist()):
            w.writerow([br, bc, lam, f"{100 * x / total[(br, bc)]:.3f}", len(files)])
print("\n% of causal tiles skipped, averaged over", len(files), "samples. lambdas:", LAMS)
for (br, bc), v in skipped.items():
    print(f"  Br={br:3d} Bc={bc:3d}  " + " ".join(f"{100 * x / total[(br, bc)]:6.2f}" for x in v.tolist()))
print("wrote", out)
