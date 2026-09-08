"""Paper-specific feature/reward interpretations are explicit, testable contracts."""

import pytest
import torch

from dht.bandits.features import FrozenStandardizer, pooled_features
from dht.policies.controller import RouteDecision
from dht.policies.rewards import RewardHistory
from dht.training.queues import DepthQueues


def test_pooling_values_dimensions_and_padding():
    hidden = torch.tensor([[1.0, 3.0], [5.0, 7.0], [100.0, 100.0]])
    mask = torch.tensor([True, True, False])
    result = pooled_features(hidden, mask, (1, 2), (1, 2))
    assert result.numel() == 9
    torch.testing.assert_close(
        result, torch.tensor([4.0, 3.0, 5.0, 2.0, 6.0, 1.0, 3.0, 5.0, 7.0], dtype=torch.float64)
    )
    assert (
        pooled_features(
            torch.ones(5, 768), torch.ones(5, dtype=torch.bool), (1, 2), (4, 8, 16)
        ).numel()
        == 84
    )


def test_standardizer_constant_dimensions_stay_finite():
    norm = FrozenStandardizer.fit([torch.tensor([1.0, 2.0]), torch.tensor([1.0, 4.0])], 1e-6)
    torch.testing.assert_close(norm.transform(torch.tensor([1.0, 3.0])), torch.zeros(2))
    assert norm.scale[0] == 1


def test_previous_batch_history_not_current_batch_and_missing_groups():
    history = RewardHistory(cap=None)
    assert history.observe_batch([{"depth": 4.0, "head:1": 2.0}, {"depth": 8.0}]) == [
        {"depth": None, "head:1": None},
        {"depth": None},
    ]
    rewards = history.observe_batch([{"depth": 3.0}, {"depth": 12.0, "head:2": 4.0}])
    assert rewards == [{"depth": 2.0}, {"depth": 0.5, "head:2": None}]
    assert history.previous["head:1"] == 2  # Not executed; no invented observation.
    assert history.previous["depth"] == 7.5


def test_reward_stability_feedback_and_roundtrip():
    history = RewardHistory(epsilon=1e-8, cap=10)
    history.observe_batch([{"depth": 1.0}])
    assert history.observe_batch([{"depth": 0.0}])[0]["depth"] == 10
    assert history.clipped == 1
    assert history.binary_feedback(2, "paper_literal") == 0
    assert history.binary_feedback(2, "improvement") == 1
    assert history.binary_feedback(1, "paper_literal") == 1
    other = RewardHistory(epsilon=1e-8, cap=10)
    other.load_state_dict(history.state_dict())
    assert other.state_dict() == history.state_dict()
    with pytest.raises(ValueError):
        history.observe_batch([{"depth": float("nan")}])


def test_queue_fifo_tail_flush_and_context_roundtrip():
    queues = DepthQueues((1, 2), 2)
    for index, depth in [(0, 2), (1, 1), (2, 2)]:
        queues.push(RouteDecision(index, depth, 0, torch.tensor([float(index)])))
    assert [item.sample_id for item in queues.pop()] == [0, 2]
    assert queues.pop() is None
    restored = DepthQueues((1, 2), 2)
    restored.load_state_dict(queues.state_dict())
    tail = restored.pop(flush=True)
    assert [item.sample_id for item in tail] == [1]
    assert torch.equal(tail[0].route_features, torch.tensor([1.0]))
    assert restored.pop(flush=True) is None
