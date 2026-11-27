"""Matched random masks and synchronized generation timing for binary attention."""
from collections import defaultdict
from functools import lru_cache
from math import ceil
from random import Random
from time import perf_counter

import torch


def random_prefill_gate(visual_positions, guided, rng):
    """Match dropped rows and the sum of their causal visual ranks, per layer."""
    ranks = {position: i + 1 for i, position in enumerate(visual_positions)}
    n = len(ranks)
    result = {}
    for layer, dropped in guided.items():
        target = sum(ranks[position] for position in dropped)
        chosen = sorted(rng.sample(range(1, n + 1), len(dropped)))
        delta = target - sum(chosen)
        if delta > 0:
            limit = n
            for i in reversed(range(len(chosen))):
                increase = min(delta, limit - chosen[i])
                chosen[i] += increase
                delta -= increase
                limit = chosen[i] - 1
        elif delta < 0:
            limit = 1
            for i in range(len(chosen)):
                decrease = min(-delta, chosen[i] - limit)
                chosen[i] -= decrease
                delta += decrease
                limit = chosen[i] + 1
        assert delta == 0
        result[layer] = [visual_positions[rank - 1] for rank in chosen]
    return result


class DecodeBudgets:
    """Record or replay budgets by layer, decode step, head, and visual tile width."""

    def __init__(self, rng=None, source=None):
        self.steps = defaultdict(list)
        self.source = source
        self.rng = rng
        self.tail_head_steps = 0
        self.tiles_reassigned = 0

    def select(self, layer, allowed, columns, selected, fallback):
        widths = torch.zeros_like(selected, dtype=torch.long)
        widths.scatter_add_(1, columns.expand_as(allowed), allowed.long())
        widths = widths.cpu().tolist()
        steps = self.steps[layer]
        if self.source is None:
            chosen = (selected | fallback[:, None]).cpu().tolist()
            steps.append((widths, chosen, fallback.cpu().tolist()))
            return selected, fallback

        reference = self.source.steps[layer]
        if not reference:
            # Guided generation stopped at its first token. There is no matched decode budget.
            self.tail_head_steps += len(widths)
            return torch.ones_like(selected), torch.ones_like(fallback)
        if len(steps) >= len(reference):
            self.tail_head_steps += len(widths)
        prior_widths, chosen, prior_fallback = reference[min(len(steps), len(reference) - 1)]
        steps.append(None)
        if widths != prior_widths:
            raise ValueError("Random decode control requires the same visible visual tile geometry")
        randomized = torch.zeros_like(selected, device="cpu")
        for h, (row, keep) in enumerate(zip(widths, chosen)):
            groups = defaultdict(list)
            for index, width in enumerate(row):
                if width:
                    groups[width].append(index)
            for indices in groups.values():
                sampled = self.rng.sample(indices, sum(keep[i] for i in indices))
                randomized[h, sampled] = True
                self.tiles_reassigned += sum(not keep[i] for i in sampled)
        return randomized.to(selected.device), torch.tensor(prior_fallback, device=fallback.device)


def case_rng(seed, index, variant):
    return Random(f"{seed}:{index}:{variant}")


@lru_cache(maxsize=4)
def modelopt_attention(prefill_lambda, decode_lambda, tile):
    from modelopt.torch.sparsity.attention_sparsity.config import SparseAttentionAttributeConfig
    from modelopt.torch.sparsity.attention_sparsity.methods.flash_skip_softmax import FlashSkipSoftmax

    simulator = FlashSkipSoftmax(SparseAttentionAttributeConfig(
        thresholds={"prefill": [prefill_lambda], "decode": [decode_lambda]},
        br=tile, bc=tile, backend="pytorch", is_causal=True, collect_stats=False,
    ).model_dump())

    def forward(*, query, key, value, attention_mask, scaling, **_):
        heads_per_kv = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(heads_per_kv, dim=1)
        value = value.repeat_interleave(heads_per_kv, dim=1)
        scores = (query.float() @ key.float().transpose(-1, -2)) * scaling
        rows, keys = scores.shape[-2:]
        causal = torch.arange(keys, device=query.device) <= torch.arange(
            keys - rows, keys, device=query.device)[:, None]
        allowed = causal
        if attention_mask is not None:
            mask = attention_mask[..., -rows:, :keys]
            if mask.dtype == torch.bool:
                allowed = allowed & mask
            else:
                allowed = allowed & (mask > torch.finfo(mask.dtype).min)
                scores = scores + mask
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        keep, _ = simulator.calculate_sparsity(scores)
        scores = simulator.apply_sparsity(scores, keep).masked_fill(~(keep & allowed), -torch.inf)
        probs = scores.softmax(-1).nan_to_num().to(value.dtype)
        return (probs @ value).transpose(1, 2).contiguous()

    return forward


def timed_generation(model, inputs, tokens):
    """Measure batch-one model.generate wall time with fixed output length."""
    device = inputs.input_ids.device
    first_token = None
    calls = 0

    def before_forward(_module, _args):
        nonlocal calls, first_token
        if calls == 1:
            torch.cuda.synchronize(device)
            first_token = perf_counter()
        calls += 1

    hook = model.register_forward_pre_hook(before_forward)
    try:
        torch.cuda.synchronize(device)
        start = perf_counter()
        generated = model.generate(**inputs, min_new_tokens=tokens, max_new_tokens=tokens, do_sample=False)
        torch.cuda.synchronize(device)
        end = perf_counter()
    finally:
        hook.remove()
    if first_token is None or generated.shape[-1] - inputs.input_ids.shape[-1] != tokens:
        raise ValueError("Latency run requires fixed-length autoregressive generation of at least two tokens")
    return generated, dict(first_token_seconds=first_token - start, decode_seconds=end - first_token,
                           generation_seconds=end - start)


def latency_summary(cases, variants):
    summary = {}
    for variant in variants:
        rows = [case["results"][variant] for case in cases]
        result = {}
        for field in ("first_token_seconds", "decode_seconds", "generation_seconds"):
            values = sorted(row[field] for row in rows)
            for percentile in (50, 95):
                result[f"{field}_p{percentile}"] = values[ceil(len(values) * percentile / 100) - 1]
        tokens = sum(row["generated_tokens"] for row in rows)
        result["generated_tokens_per_second"] = tokens / sum(row["generation_seconds"] for row in rows)
        result["decode_tokens_per_second"] = (tokens - len(rows)) / sum(row["decode_seconds"] for row in rows)
        summary[variant] = result
    return summary
