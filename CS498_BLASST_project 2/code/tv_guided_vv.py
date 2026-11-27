"""Previous-layer text votes for the causal V→V query-tile policy."""
from __future__ import annotations

import math
from decimal import Decimal
from random import Random

import torch


SWEEP_SCHEMA = "blasst-tv-guided-vv-v1"
TOKEN_GATE_COUNTS = ("vv_pairs", "vv_pairs_bypassed", "visual_query_tokens", "visual_query_tokens_selected")
VISUAL_QUERY_COUNTS = ("vt_pairs", "vt_pairs_bypassed")


class TVGuidedVV:
    """Own the votes for one prompt so each layer reads only its predecessor."""

    def __init__(self, visual, tile_size, threshold, *, replay=None, top_percent=None, token_percent=None,
                 token_gate=None, token_drop_percent=None, random_drop=False, drop_seed="", drop_highest=False):
        self.visual = visual.cpu().bool()
        self.tile_size = tile_size
        self.threshold = threshold
        self.top_percent = top_percent
        self.token_percent = token_percent
        self.token_gate = token_gate
        self.token_drop_percent = token_drop_percent
        self.random_drop = random_drop
        self.drop_seed = drop_seed
        self.drop_highest = drop_highest
        self.visual_prefix = self.visual.cumsum(0)
        self.text_prefix = (~self.visual).cumsum(0)
        self.replay = replay
        self.text_rows = [~tile for tile in self.visual.split(tile_size)]
        self.pure_visual = [index for index, text in enumerate(self.text_rows) if not text.any()]
        self.visual_set = set(self.pure_visual)
        self.previous = None
        self.layers = []

    @property
    def enabled(self):
        return any(value is not None for value in (self.threshold, self.top_percent, self.token_percent, self.token_gate))

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
            if "vv_pairs_bypassed" in layer:
                layers.append({key: layer[key] for key in ("layer", *TOKEN_GATE_COUNTS)})
                continue
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
        self.token_drops = None
        if policy.token_percent is not None or policy.token_gate is not None:
            self.token_kept = torch.zeros(len(policy.visual), dtype=torch.long, device=device)
            self.token_eligible = torch.zeros_like(self.token_kept)
            self.token_drops = torch.zeros_like(policy.visual)
            if policy.replay is not None:
                positions = policy.replay.get(layer)
                if positions is None or (previous is None and positions):
                    raise ValueError(f"Invalid token gate replay for layer {layer}")
                if any(pos < 0 or pos >= len(policy.visual) for pos in positions) or not policy.visual[positions].all():
                    raise ValueError(f"Token gate replay contains a nonvisual position at layer {layer}")
                self.token_drops[positions] = True
            elif previous is not None and policy.token_drop_percent is not None:
                positions = policy.visual.nonzero().flatten().tolist()
                count = int(len(positions) * Decimal(str(policy.token_drop_percent)) / 100)
                if count:
                    # The same seeded order supplies the random mask and breaks frequency ties.
                    Random(f"{policy.drop_seed}:{layer}").shuffle(positions)
                    if not policy.random_drop:
                        positions.sort(key=previous["token_relevance"].__getitem__, reverse=policy.drop_highest)
                    self.token_drops[positions[:count]] = True
            elif previous is not None and policy.token_gate is not None:
                self.token_drops = policy.visual & ~previous["token_had_vote"]
            elif previous is not None:
                quota = math.ceil(int(policy.visual.sum()) * policy.token_percent / 100)
                ranked = sorted(policy.visual.nonzero().flatten().tolist(),
                                key=lambda pos: (-previous["token_relevance"][pos], pos))
                self.token_drops[ranked[quota:]] = True
        elif policy.replay is not None:
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

    def observe_tokens(self, start, score_tiles, cutoff, mask):
        if self.policy.random_drop:
            return
        length = len(self.policy.visual)
        text = ~self.policy.visual[start:start + score_tiles.shape[-3]]
        if not text.any():
            return
        scores = score_tiles[0, :, text.to(score_tiles.device)]
        running = scores.amax(-1).cummax(-1).values
        valid = torch.isfinite(scores)
        if mask is not None and mask.dtype != torch.bool:
            present = (mask[0] > torch.finfo(mask.dtype).min).expand_as(score_tiles[0].flatten(-2))
            present = present[:, text.to(scores.device)]
            pad = -length % self.policy.tile_size
            valid &= torch.nn.functional.pad(present, (0, pad)).unflatten(-1, (-1, self.policy.tile_size))
        eligible = valid.flatten(-2)[..., :length]
        kept = (valid & (scores - running.unsqueeze(-1) >= cutoff)).flatten(-2)[..., :length]
        self.token_eligible += eligible.sum(dim=(0, 1))
        self.token_kept += kept.sum(dim=(0, 1))

    def finish(self):
        if self.token_drops is not None:
            policy = self.policy
            visual_tokens = int(policy.visual.sum())
            record = dict(layer=self.layer, vote_layer=self.previous["layer"] if self.previous else None,
                          visual_query_tokens=visual_tokens,
                          visual_query_tokens_selected=visual_tokens - int(self.token_drops.sum()),
                          vv_pairs=visual_tokens * (visual_tokens + 1) // 2,
                          vv_pairs_bypassed=int(policy.visual_prefix[self.token_drops].sum()),
                          pure_vv_tiles=0, vv_tiles_bypassed=0, votes=[],
                          dropped_token_positions=self.token_drops.nonzero().flatten().tolist())
            if policy.token_gate is not None:
                record["vt_pairs"] = int(policy.text_prefix[policy.visual].sum())
                record["vt_pairs_bypassed"] = (
                    int(policy.text_prefix[self.token_drops].sum()) if policy.token_gate == "whole" else 0
                )
            if policy.token_drop_percent is not None:
                record["visual_query_tokens_gateable"] = visual_tokens if self.previous is not None else 0
            if policy.random_drop:
                policy.previous = record
            elif policy.token_gate is not None and policy.token_drop_percent is None:
                policy.previous = dict(record, token_had_vote=(self.token_kept > 0).cpu())
            else:
                relevance = torch.where(self.token_eligible > 0,
                                        self.token_kept / self.token_eligible, -1.).tolist()
                self.policy.previous = dict(record, token_relevance=relevance)
            policy.layers.append(record)
            return
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
