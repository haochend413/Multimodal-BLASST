"""Previous-layer text votes for the causal V→V query-tile policy."""
from __future__ import annotations

import torch


SWEEP_SCHEMA = "blasst-tv-guided-vv-v1"


class TVGuidedVV:
    """Own the votes for one prompt so each layer reads only its predecessor."""

    def __init__(self, visual, tile_size, threshold, *, replay=None):
        self.visual = visual.cpu().bool()
        self.tile_size = tile_size
        self.threshold = threshold
        self.replay = replay
        self.text_rows = [~tile for tile in self.visual.split(tile_size)]
        self.pure_visual = [index for index, text in enumerate(self.text_rows) if not text.any()]
        self.visual_set = set(self.pure_visual)
        self.previous = None
        self.layers = []

    def start_layer(self, layer, device):
        if layer == 0:
            self.previous = None
            self.layers.clear()
        previous = self.previous if self.previous and self.previous["layer"] == layer - 1 else None
        return TVLayer(self, layer, device, previous)


class TVLayer:
    def __init__(self, policy, layer, device, previous):
        self.policy = policy
        self.layer = layer
        self.previous = previous
        self.kept = torch.zeros(len(policy.text_rows), dtype=torch.long, device=device)
        self.eligible = torch.zeros_like(self.kept)
        self.bypassed = {}
        self.protected = set()
        self.mask_fallback = set()
        self.dropped = set()
        if policy.replay is not None:
            if layer not in policy.replay:
                raise ValueError(f"V→V gate replay has no entry for layer {layer}")
            self.dropped = set(policy.replay[layer])
            if not self.dropped <= policy.visual_set or (previous is None and self.dropped):
                raise ValueError(f"Invalid V→V gate replay for layer {layer}")
        elif previous is not None and policy.threshold is not None:
            self.dropped = {vote["tile"] for vote in previous["votes"]
                            if vote["eligible"] and vote["kept"] / vote["eligible"] < policy.threshold}

    def bypass_keys(self, query_tile):
        if query_tile not in self.dropped:
            return []
        keys = [index for index in self.policy.pure_visual if index <= query_tile]
        # If the prompt starts with visual tiles, leave every query a valid key.
        if len(keys) == query_tile + 1:
            self.protected.add(query_tile)
            keys = keys[1:]
        return keys

    def observe_text(self, query_tile, row_margins, cutoff):
        text = self.policy.text_rows[query_tile]
        if not text.any() or not self.policy.pure_visual:
            return
        margins = row_margins[:, :, text.to(row_margins.device), :]
        valid = margins.isfinite()
        eligible = valid.any(-2)
        vote = (valid & (margins >= cutoff)).any(-2)
        self.eligible += eligible[0].sum(0)
        self.kept += vote[0].sum(0)

    def finish(self):
        kept, eligible = self.kept.tolist(), self.eligible.tolist()
        previous_votes = {vote["tile"]: vote for vote in self.previous["votes"]} if self.previous else {}
        votes = []
        for index in self.policy.pure_visual:
            prior = previous_votes.get(index)
            if self.policy.threshold is None:
                decision = "baseline"
            elif self.previous is None:
                decision = "warmup"
            else:
                decision = "skip" if index in self.dropped else "keep"
            votes.append(dict(tile=index, key_start=index * self.policy.tile_size,
                              kept=kept[index], eligible=eligible[index],
                              relevance=kept[index] / eligible[index] if eligible[index] else None,
                              previous_relevance=prior["relevance"] if prior else None,
                              decision=decision,
                              vv_tiles_bypassed=self.bypassed.get(index, 0),
                              protected_first_key=index in self.protected, mask_fallback=index in self.mask_fallback))
        vv_tiles = len(votes) * (len(votes) + 1) // 2
        record = dict(layer=self.layer, vote_layer=self.previous["layer"] if self.previous else None,
                      pure_vv_tiles=vv_tiles, vv_tiles_bypassed=sum(self.bypassed.values()), votes=votes)
        self.policy.layers.append(record)
        self.policy.previous = record
