"""Previous-layer text votes for the causal V→V query-tile policy."""
from __future__ import annotations

import math

import torch


SWEEP_SCHEMA = "blasst-tv-guided-vv-v1"


class TVGuidedVV:
    """Own the votes for one prompt so each layer reads only its predecessor."""

    def __init__(self, visual, tile_size, threshold, *, replay=None, top_percent=None):
        self.visual = visual.cpu().bool()
        self.tile_size = tile_size
        self.threshold = threshold
        self.top_percent = top_percent
        self.replay = replay
        self.text_rows = [~tile for tile in self.visual.split(tile_size)]
        self.pure_visual = [index for index, text in enumerate(self.text_rows) if not text.any()]
        self.visual_set = set(self.pure_visual)
        self.previous = None
        self.layers = []

    @property
    def enabled(self):
        return self.threshold is not None or self.top_percent is not None

    def start_layer(self, layer, device):
        if layer == 0:
            self.previous = None
            self.layers.clear()
        previous = self.previous if self.previous and self.previous["layer"] == layer - 1 else None
        return TVLayer(self, layer, device, previous)

    def distribution(self):
        """Count causal token pairs with and without the recorded QK bypass."""
        runs = []
        tokens = {"T": 0, "V": 0}
        causal = dict.fromkeys(("TT", "TV", "VT", "VV"), 0)
        for is_visual in self.visual.tolist():
            kind = "V" if is_visual else "T"
            if not runs or runs[-1]["modality"] != kind:
                runs.append(dict(modality=kind, tokens=0))
            runs[-1]["tokens"] += 1
            tokens[kind] += 1
            for key, count in tokens.items():
                causal[kind + key] += count

        tile_lengths = [len(tile) for tile in self.text_rows]
        pure_rank = {tile: rank for rank, tile in enumerate(self.pure_visual)}
        layers = []
        for layer in self.layers:
            bypassed = 0
            for vote in layer["votes"]:
                count = vote["vv_tiles_bypassed"]
                if not count:
                    continue
                query = vote["tile"]
                end = pure_rank[query] + 1
                # The gate drops a suffix of the eligible pure-V key tiles.
                for key in self.pure_visual[end - count:end]:
                    q_len = tile_lengths[query]
                    bypassed += q_len * (q_len + 1) // 2 if query == key else q_len * tile_lengths[key]
            layers.append(dict(layer=layer["layer"], vv_pairs_bypassed=bypassed,
                               visual_query_tiles=len(layer["votes"]),
                               visual_query_tiles_selected=sum(vote["decision"] != "skip" for vote in layer["votes"])))

        before = {region: count * len(layers) for region, count in causal.items()}
        after = dict(before, VV=before["VV"] - sum(layer["vv_pairs_bypassed"] for layer in layers))
        return dict(prompt_runs=runs, prompt_tokens=tokens, pairs_before=before, pairs_after=after, layers=layers)


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
        elif previous is not None and policy.top_percent is not None:
            quota = math.ceil(len(policy.pure_visual) * policy.top_percent / 100)
            ranked = sorted(previous["votes"], key=lambda vote: (
                -(vote["kept"] / vote["eligible"]) if vote["eligible"] else math.inf,
                vote["tile"],
            ))
            self.dropped = {vote["tile"] for vote in ranked[quota:]}
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
            if not self.policy.enabled:
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
