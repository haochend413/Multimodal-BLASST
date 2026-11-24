"""Running-max reductions shared by live attention and offline region audits."""
from __future__ import annotations

from typing import NamedTuple

import torch


def row_tile_margins(tile_max):
    running = torch.cummax(tile_max, dim=-1).values
    return torch.where(torch.isfinite(running), tile_max - running, -torch.inf)


class Span(NamedTuple):
    start: int
    end: int
    region: int


def axis_tiles(visual, size, *, packed):
    """Start each contiguous modality run at a tile boundary, padding only its tail."""
    end = len(visual)
    starts = [0]
    if packed:
        starts.extend((visual[1:] != visual[:-1]).nonzero().flatten().add(1).tolist())
    return [Span(start, min(start + size, hi), region)
            for region, (lo, hi) in enumerate(zip(starts, [*starts[1:], end]))
            for start in range(lo, hi, size)]


def region_margins(tile_max, keys):
    """Reset at contiguous key-run transitions; the first valid tile establishes the bar."""
    cuts = [0, *(i for i in range(1, len(keys)) if keys[i - 1].region != keys[i].region), len(keys)]
    return torch.cat([row_tile_margins(tile_max[..., lo:hi])
                      for lo, hi in zip(cuts, cuts[1:])], dim=-1)


def key_segments(physical_tiles, visual):
    """Find contiguous modality spans inside each physical key tile."""
    runs = iter(axis_tiles(visual, len(visual), packed=True))
    run = next(runs)
    groups = []
    for lo, hi, _ in physical_tiles:
        segments = []
        while lo < hi:
            if lo == run.end:
                run = next(runs)
            end = min(hi, run.end)
            segments.append(Span(lo, end, run.region))
            lo = end
        groups.append(segments)
    return groups


def physical_reset_margins(scores, tile_max, segments_by_tile):
    """Reduce regional margins to one margin per row and physical tile."""
    segments, segment_maxima, stops = [], [], [0]
    for index, group in enumerate(segments_by_tile):
        segments.extend(group)
        segment_maxima.append(tile_max[..., index:index + 1] if len(group) == 1 else row_max(scores, group))
        stops.append(len(segments))
    margins = region_margins(torch.cat(segment_maxima, dim=-1), segments)
    # Every visible segment must pass the test for the physical tile to skip.
    return torch.stack([margins[..., lo:hi].amax(-1) for lo, hi in zip(stops, stops[1:])], dim=-1)


def row_max(scores, keys):
    return torch.stack([scores[..., lo:hi].amax(-1) for lo, hi, _ in keys], dim=-1)


class RegionMax:
    """Cache prompt-local physical tile segments for each key length."""

    def __init__(self, visual, tile_size):
        self.visual = visual.cpu().bool()
        self.tile_size = tile_size
        self.segments = {}

    def row_margins(self, tiles, key_length):
        if key_length not in self.segments:
            visual = torch.zeros(key_length, dtype=torch.bool)
            visual[:len(self.visual)] = self.visual[:key_length]
            keys = axis_tiles(visual, self.tile_size, packed=False)
            self.segments[key_length] = key_segments(keys, visual)
        return physical_reset_margins(tiles.flatten(-2), tiles.amax(-1), self.segments[key_length])

    def margins(self, tiles, key_length):
        return self.row_margins(tiles, key_length).amax(-2)


DecodeRegionMax = RegionMax
