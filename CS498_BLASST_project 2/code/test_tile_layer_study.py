"""Regression checks for one reused key per row and key tile."""
import torch

from tile_layer_study import TileIndexTracker


def test_replays_each_querys_own_key_and_rebuilds_the_running_max():
    tracker = TileIndexTracker()
    source = torch.tensor([[[6., 3., 4., 1.], [6., 2., 1., 5.]]])
    target = torch.tensor([[[3., 6., 0., 5.9], [6., 2., 1., 3.]]])
    _, _, old_keys, first = tracker.capture("chunk", 0, source, 2)
    assert first is None
    assert old_keys[0][1] == [0, 1]

    _, snapshots, new_keys, replay = tracker.capture("chunk", 1, target, 2)
    assert new_keys[0][1] == [1, 1]
    assert replay[0][1] == [1., 1., 2., -3.]
    tile_score, running_score, _ = snapshots[0][1]
    torch.testing.assert_close(torch.tensor(tile_score - running_score), torch.tensor(-.1))


def test_ignores_masked_rows_and_counts_score_ties_separately():
    tracker = TileIndexTracker()
    source = torch.tensor([[[4., 1.], [-torch.inf, -torch.inf]]])
    target = torch.tensor([[[4., 4.], [-torch.inf, -torch.inf]]])
    tracker.capture("chunk", 0, source.flip(-1), 2)
    _, _, keys, replay = tracker.capture("chunk", 1, target, 2)
    assert keys[0][0] == [0, 255]
    assert replay[0][0] == [0., 1., 1., 0.]
    assert tracker.capture("chunk", 3, target, 2)[3] is None
