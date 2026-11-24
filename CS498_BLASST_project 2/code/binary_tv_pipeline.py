"""Propagate binary T→V attention decisions across layers of one prompt."""
from collections import Counter

import torch
import torch.nn.functional as F

from region_max import row_tile_margins


BINARY_METRICS = (
    "tv_qk_pairs", "tv_qk_pairs_bypassed", "tv_fallback_rows",
    "tv_query_rows", "vv_qk_pairs", "vv_qk_pairs_bypassed",
)
BINARY_TV_COUNTS = tuple(f"{phase}_{metric}" for phase in ("prefill", "decode") for metric in BINARY_METRICS)


def tile_max(scores, tile_size):
    return F.pad(scores, (0, -scores.shape[-1] % tile_size), value=-torch.inf).unflatten(
        -1, (-1, tile_size)).amax(-1)


def keep_tiles(maxima, cutoff):
    return maxima.isfinite() & (row_tile_margins(maxima) >= cutoff)


def visible_keys(mask, rows, length, heads):
    positions = torch.arange(length, device=rows.device)
    causal = (positions <= rows[:, None]).expand(heads, -1, -1)
    if mask is None:
        return causal, None
    mask = mask[0].expand(heads, len(rows), length)
    if mask.dtype == torch.bool:
        return causal & mask, None
    return causal & (mask > torch.finfo(mask.dtype).min), mask.float()


class BinaryTVPipeline:
    """Run staged BLASST and keep device-local masks for cached text positions."""

    def __init__(self, visual, tile_size, *, gate_vv):
        self.visual = visual.bool().cpu()
        self.tile_size = tile_size
        self.gate_vv = gate_vv
        self.visual_groups = [(start // tile_size, tile.nonzero().flatten() + start)
                              for start, tile in zip(range(0, len(visual), tile_size), self.visual.split(tile_size))
                              if tile.any()]
        self.layers = {}
        self.counts = Counter(dict.fromkeys(BINARY_TV_COUNTS, 0))

    def start(self, layer, heads, length, device):
        prompt_length = len(self.visual)
        if length == prompt_length:
            self.layers[layer] = (
                torch.zeros(heads, length, len(self.visual_groups), dtype=torch.bool, device=device),
                torch.zeros(heads, length, dtype=torch.bool, device=device),
            )
        else:
            old_links, old_history = self.layers[layer]
            if old_links.shape[1] != length - 1:
                raise ValueError("Binary T→V requires sequential decode with an append-only KV cache")
            self.layers[layer] = (F.pad(old_links, (0, 0, 0, 1)), F.pad(old_history, (0, 1)))
        return self.layers[layer]

    def key_layout(self, length, device):
        visual = F.pad(self.visual.to(device), (0, length - len(self.visual)))
        visual_groups = [(tile, columns.to(device)) for tile, columns in self.visual_groups]
        return visual, visual_groups

    def record_warmup(self, q_start, tiles, mask, cutoff, phase):
        """Record the row-level visual decisions without changing layer 0 output."""
        links, history = self.layers[0]
        heads, rows = tiles.shape[1:3]
        length = links.shape[1]
        visual, groups = self.key_layout(length, tiles.device)
        positions = torch.arange(q_start, q_start + rows, device=tiles.device)
        allowed, _ = visible_keys(mask, positions, length, heads)
        scores = tiles[0].flatten(-2)[..., :length].masked_fill(~allowed, -torch.inf)
        tv_allowed = allowed & visual & ~visual[positions, None]
        idx = [tile for tile, _ in groups]
        visual_max = tile_max(scores.masked_fill(~tv_allowed, -torch.inf), self.tile_size)[..., idx]
        running = tile_max(scores, self.tile_size).cummax(-1).values[..., idx]
        links[:, positions] = visual_max.isfinite() & (visual_max - running >= cutoff)
        history[:, positions] = tv_allowed.any(-1)
        self.counts[f"{phase}_tv_qk_pairs"] += int(tv_allowed.sum())
        self.counts[f"{phase}_tv_query_rows"] += int(tv_allowed.any(-1).sum())
        if phase == "prefill":
            self.counts["prefill_vv_qk_pairs"] += int((allowed & visual & visual[positions, None]).sum())

    def forward(self, layer, query, key, value, mask, scaling, cutoff, phase):
        heads, rows, length = query.shape[1], query.shape[2], key.shape[-2]
        device = query.device
        current, history = self.start(layer, heads, length, device)
        prev_links, prev_history = (item.to(device) for item in self.layers[layer - 1])
        if prev_links.shape[1] != length:
            raise ValueError("Binary T→V requires the previous layer's mask for the current cache")
        visual, groups = self.key_layout(length, device)
        tile_ids = [tile for tile, _ in groups]
        key_tiles = torch.arange(length, device=device) // self.tile_size
        text_keys = (~visual).nonzero().flatten()
        bounds = torch.arange(length - rows, length, device=device)
        prev_history = prev_history[:, text_keys]
        prev_links = prev_links[:, text_keys].float()
        is_text_query = ~visual[bounds]
        q, k = query[0].float(), key[0].float()
        out = torch.zeros_like(query[0])
        # Assemble physical skip statistics from the two modality slices in each query tile.
        physical_kept = torch.zeros(heads, (rows + self.tile_size - 1) // self.tile_size,
                                    (length + self.tile_size - 1) // self.tile_size,
                                    dtype=torch.bool, device=device)

        def geometry(local):
            positions = bounds[local]
            if mask is None or mask.shape[-2] == 1:
                subset = mask
            else:
                subset = mask.index_select(-2, local)
            return positions, *visible_keys(subset, positions, length, heads)

        def compute_text(qs, allowed, bias):
            scores = qs.new_full((*qs.shape[:-1], length), -torch.inf)
            tt = qs @ k[:, text_keys].transpose(-1, -2) * scaling
            if bias is not None:
                tt += bias[..., text_keys]
            scores[..., text_keys] = tt.masked_fill(~allowed[..., text_keys], -torch.inf)
            return scores

        def compute_visual(scores, qs, allowed, selected, bias):
            # Each dot-product batch includes only head/query rows selected for this visual tile.
            for j, (_, cols) in enumerate(groups):
                valid = allowed[..., cols] & selected[..., j, None]
                h, r = valid.any(-1).nonzero(as_tuple=True)
                if not h.numel():
                    continue
                logits = torch.bmm(qs[h, r, None], k[h[:, None], cols].transpose(-1, -2)).squeeze(1) * scaling
                if bias is not None:
                    logits += bias[h[:, None], r[:, None], cols]
                scores[h[:, None], r[:, None], cols] = logits.masked_fill(~valid[h, r], -torch.inf)

        def save_output(local, query_tile, scores):
            probs = scores.softmax(-1).nan_to_num().to(value.dtype)
            out[:, local] = probs @ value[0]
            physical_kept[:, query_tile] |= tile_max(scores, self.tile_size).isfinite().any(1)

        counts = torch.zeros(6, dtype=torch.long, device=device)
        for start in range(0, rows, self.tile_size):
            local = is_text_query[start:start + self.tile_size].nonzero().flatten() + start
            for indices in local.split(32):
                positions, allowed, bias = geometry(indices)
                qs = q[:, indices]
                scores = compute_text(qs, allowed, bias)
                retained = keep_tiles(tile_max(scores, self.tile_size), cutoff)[..., key_tiles] & scores.isfinite()
                usable = retained[..., text_keys] & prev_history[:, None, :]
                fallback = ~usable.any(-1)
                candidates = (usable.float() @ prev_links) > 0
                candidates |= fallback.unsqueeze(-1)
                tv_allowed = allowed & visual
                scores.masked_fill_(~retained, -torch.inf)
                compute_visual(scores, qs, tv_allowed, candidates, bias)
                visual_max = tile_max(scores.masked_fill(~visual, -torch.inf), self.tile_size)[..., tile_ids]
                running = tile_max(scores, self.tile_size).cummax(-1).values[..., tile_ids]
                links = visual_max.isfinite() & (visual_max - running >= cutoff)
                current[:, positions] = links
                history[:, positions] = tv_allowed.any(-1)
                keep = retained.clone()
                computed = torch.zeros_like(tv_allowed)
                for j, (_, cols) in enumerate(groups):
                    keep[..., cols] = links[..., j, None]
                    computed[..., cols] = candidates[..., j, None] & tv_allowed[..., cols]
                counts[0] += tv_allowed.sum()
                counts[1] += (tv_allowed & ~computed).sum()
                has_vision = tv_allowed.any(-1)
                counts[2] += (fallback & has_vision).sum()
                counts[3] += has_vision.sum()
                save_output(indices, start // self.tile_size, scores.masked_fill(~keep, -torch.inf))

        if phase == "prefill":
            consumed = current.any(dim=1)
            enabled = torch.zeros(heads, len(visual), dtype=torch.bool, device=device)
            for j, (_, columns) in enumerate(groups):
                enabled[:, columns] = consumed[:, j, None] if self.gate_vv else True
            for start in range(0, rows, self.tile_size):
                local = torch.where(visual[start:start + self.tile_size])[0] + start
                if not local.numel():
                    continue
                chunks = []
                for indices in local.split(32):
                    positions, allowed, bias = geometry(indices)
                    qs = q[:, indices]
                    scores = compute_text(qs, allowed, bias)
                    selected = enabled[:, positions, None]
                    vv_allowed = allowed & visual
                    compute_visual(scores, qs, vv_allowed, selected.expand(-1, -1, len(groups)), bias)
                    counts[4] += vv_allowed.sum()
                    counts[5] += (vv_allowed & ~selected).sum()
                    chunks.append(scores)
                scores = torch.cat(chunks, dim=1)
                maxima = tile_max(scores, self.tile_size)
                keep = (row_tile_margins(maxima).amax(dim=1) >= cutoff)[..., key_tiles]
                save_output(local, start // self.tile_size, scores.masked_fill(~keep[:, None, :], -torch.inf))

        self.counts.update({f"{phase}_{name}": amount for name, amount in zip(BINARY_METRICS, counts.tolist())})
        return out.transpose(0, 1).unsqueeze(0).contiguous(), physical_kept
