"""Use prefill text-to-vision links to select visual keys during decode."""
from collections import Counter

import torch

from region_max import row_tile_margins


DECODE_LINK_COUNTS = (
    "decode_head_steps", "decode_link_fallback_heads", "decode_visual_tiles",
    "decode_visual_tiles_bypassed", "decode_visual_keys", "decode_visual_keys_bypassed",
)


class DecodeVisualLinks:
    """Own per-prompt boolean masks indexed by layer, head, position, and visual tile."""

    def __init__(self, visual, tile_size):
        self.visual = visual.bool().cpu()
        self.tile_size = tile_size
        self.layers = {}
        self.counts = Counter(dict.fromkeys(DECODE_LINK_COUNTS, 0))

    def start_layer(self, layer, heads, device):
        visual = self.visual.to(device)
        visual_positions = visual.nonzero().flatten()
        visual_tile_ids, visual_tile_columns = torch.unique(
            visual_positions // self.tile_size, return_inverse=True
        )
        self.layers[layer] = (
            torch.zeros(heads, len(visual), len(visual_tile_ids), dtype=torch.bool, device=device),
            torch.zeros(heads, len(visual), dtype=torch.bool, device=device),
            visual_positions, visual_tile_ids, visual_tile_columns,
        )

    def record(self, layer, start, tiles, ln_lambda, mask):
        links, history, _, visual_tiles, _ = self.layers[layer]
        scores = tiles[0].flatten(-2)
        length = len(self.visual)
        visual = self.visual.to(scores.device)
        if mask is not None and mask.dtype != torch.bool:
            unmasked = mask[0] > torch.finfo(mask.dtype).min
            scores = scores[..., :length].masked_fill(~unmasked, -torch.inf)
            tiles = torch.nn.functional.pad(scores, (0, -length % self.tile_size), value=-torch.inf)
            tiles = tiles.unflatten(-1, (-1, self.tile_size)).unsqueeze(0)
        visual_scores = scores[..., :length].masked_fill(~visual, -torch.inf)
        padded = torch.nn.functional.pad(visual_scores, (0, -length % self.tile_size), value=-torch.inf)
        visual_max = padded.unflatten(-1, (-1, self.tile_size)).amax(-1).index_select(-1, visual_tiles)
        running = tiles[0].amax(-1).cummax(-1).values.index_select(-1, visual_tiles)
        valid = torch.isfinite(visual_max) & ~visual[start:start + scores.shape[-2], None]
        end = start + scores.shape[-2]
        links[:, start:end] = valid & (visual_max - running >= ln_lambda)
        history[:, start:end] = valid.any(-1)

    def decode(self, layer, query, key, value, mask, scaling, ln_lambda, score_tiles):
        """Return one joint attention output and the physical-tile skip mask."""
        heads, length = query.shape[1], key.shape[-2]
        device = query.device
        prompt_length = len(self.visual)
        n_tiles = (length + self.tile_size - 1) // self.tile_size
        links, history, visual_positions, visual_tiles, visual_columns = self.layers[layer]
        visual_positions = visual_positions.to(device)
        positions = torch.arange(length, device=device)
        is_visual = torch.zeros(length, dtype=torch.bool, device=device)
        is_visual[visual_positions] = True
        text_positions = positions[~is_visual]
        q = query[0].float()
        k = key[0]
        text_scores = (q @ k.index_select(-2, text_positions).float().transpose(-1, -2)).squeeze(1) * scaling

        if mask is None:
            allowed = torch.ones(heads, length, dtype=torch.bool, device=device)
        else:
            mask = mask.expand(1, heads, 1, length)[0, :, 0]
            allowed = mask if mask.dtype == torch.bool else mask > torch.finfo(mask.dtype).min

        def apply_mask(scores, head_indices, key_indices):
            if mask is not None and mask.dtype != torch.bool:
                scores = scores + mask[head_indices, key_indices].float()
            return scores.masked_fill(~allowed[head_indices, key_indices], -torch.inf)

        head_ids = torch.arange(heads, device=device)
        text_scores = apply_mask(text_scores, head_ids[:, None], text_positions)
        full = q.new_full((heads, length), -torch.inf)
        full[:, text_positions] = text_scores
        allowed_positions = torch.ones_like(positions, dtype=torch.bool)
        text_tiles = score_tiles(full, allowed_positions)
        text_max = text_tiles.amax(-1)
        text_kept = torch.isfinite(text_max) & (row_tile_margins(text_max) >= ln_lambda)
        kept_keys = text_kept[:, positions // self.tile_size] & torch.isfinite(full)
        usable = kept_keys[:, :prompt_length] & history
        fallback = ~usable.any(-1)
        selected = (links & usable.unsqueeze(-1)).any(dim=1)
        visual_allowed = allowed[:, visual_positions]
        compute = visual_allowed & (selected[:, visual_columns] | fallback[:, None])
        visual_heads, visual_keys = compute.nonzero(as_tuple=True)
        columns = visual_positions[visual_keys]
        visual_scores = (q[visual_heads] * k[visual_heads, columns].float().unsqueeze(1)).sum(-1).squeeze(1) * scaling
        full[visual_heads, columns] = apply_mask(visual_scores, visual_heads, columns)

        full_tiles = score_tiles(full, allowed_positions)
        ordinary_keep = row_tile_margins(full_tiles.amax(-1)) >= ln_lambda
        visual_kept = compute & (~fallback[:, None] | ordinary_keep[:, visual_tiles[visual_columns]])
        kept_keys[:, visual_positions] = visual_kept
        scores = full.masked_fill(~kept_keys, -torch.inf)
        # Masked rows contribute zero, as in SDPA.
        probs = torch.softmax(scores, dim=-1).nan_to_num().unsqueeze(1)
        output = probs.to(value.dtype) @ value[0]

        pair_tiles = score_tiles(scores, allowed_positions)
        skip = ~torch.isfinite(pair_tiles.amax(-1))
        tile_flags = torch.zeros((2, heads, n_tiles), dtype=torch.int32, device=device)
        tile_flags.scatter_reduce_(2, visual_tiles[visual_columns].expand(2, heads, -1),
                                   torch.stack((visual_allowed, compute)).int(), reduce="amax")
        eligible_tiles, computed_tiles = tile_flags.sum(dim=(1, 2)).tolist()
        eligible_keys = int(visual_allowed.sum())
        self.counts.update(dict(zip(DECODE_LINK_COUNTS, (
            heads, int(fallback.sum()), eligible_tiles, eligible_tiles - computed_tiles,
            eligible_keys, eligible_keys - len(columns),
        ))))
        return output.unsqueeze(0), skip
