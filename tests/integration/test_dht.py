"""End-to-end policy tests: gradients, delayed updates, queues and exact resume."""

import pytest
import torch

from dht.cli import train
from dht.config import ExperimentConfig
from dht.data import synthetic_split
from dht.models.transformer import TransformerClassifier
from dht.runtime import seed_everything
from dht.training.checkpointing import read_checkpoint
from dht.training.trainer import DHTTrainer, restore_controller


def configuration(mode="tree"):
    return ExperimentConfig.model_validate(
        {
            "steps": 5,
            "batch_size": 3,
            "model": {"dropout": 0.15},
            "data": {"train_size": 7, "validation_size": 4, "test_size": 4},
            "dht": {
                "mode": mode,
                "depths": [2, 3],
                "head_counts": [2, 4],
                "calibration_samples": 4,
            },
        }
    )


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            # Performance counters naturally differ; numerical/research state must not.
            if key not in ("routing_seconds", "calibration_seconds"):
                assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("mode", ["tree", "integrated"])
def test_dht_updates_bandits_and_resumes_exactly(tmp_path, mode):
    config = configuration(mode)
    train(config, tmp_path / "full")
    train(config, tmp_path / "partial", stop_after=2)
    train(config, tmp_path / "resumed", tmp_path / "partial/checkpoint.pt")
    full = read_checkpoint(tmp_path / "full/checkpoint.pt")
    resumed = read_checkpoint(tmp_path / "resumed/checkpoint.pt")
    for key in ("model", "optimizer", "training_state"):
        assert_nested_equal(full[key], resumed[key])
    controller = full["training_state"]["controller"]
    assert controller["route_ucb"]["counts"].sum() > 0
    assert controller["ts"]["counts"].sum() > 0
    if mode == "tree":
        assert controller["head_ucb"]["counts"].sum() > 0
    # Also prove a checkpoint carrying a pending queue can round-trip.
    partial = read_checkpoint(tmp_path / "partial/checkpoint.pt")
    assert partial["training_state"]["cursor"] > 0


def test_routing_no_gradient_execution_matches_selected_spec_and_eval_is_frozen():
    config = configuration()
    seed_everything(config.seed, config.bandit.seed)
    model = TransformerClassifier(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    tokens, labels = synthetic_split(config, "train")
    trainer = DHTTrainer(config, model, optimizer, tokens, labels, torch.device("cpu"))
    route = trainer.controller.route(model, tokens[:1], 0)
    assert model.training
    assert not route.route_features.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())
    model.eval()
    execution = trainer.controller.execute(model, tokens[:1], route, explore=False)
    torch.testing.assert_close(
        execution.logits, model(tokens[:1], architecture=execution.architecture)
    )
    before = trainer.controller.state_dict()
    one, specs = trainer.controller.predict(model, tokens)
    two, _ = trainer.controller.predict(model, tokens)
    torch.testing.assert_close(one, two, rtol=0, atol=0)
    assert len(specs) == len(tokens)
    assert_nested_equal(before, trainer.controller.state_dict())


def test_epoch_tail_no_samples_dropped():
    config = configuration()
    model = TransformerClassifier(config.model)
    tokens, labels = synthetic_split(config, "train")
    trainer = DHTTrainer(
        config, model, torch.optim.AdamW(model.parameters()), tokens, labels, torch.device("cpu")
    )
    seen = []
    # next_batch routes without training; consume until all original samples released.
    while len(seen) < len(tokens):
        batch = trainer.next_batch()
        assert len({item.total_depth for item in batch}) == 1
        seen.extend(item.sample_id for item in batch)
    assert sorted(seen) == list(range(len(tokens)))
    assert trainer.epoch == 0


def test_checkpoint_policy_can_evaluate_without_calibration(tmp_path, monkeypatch):
    config = configuration()
    train(config, tmp_path / "run", stop_after=2)
    state = read_checkpoint(tmp_path / "run/checkpoint.pt")
    from dht.bandits.features import FeatureBank

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation must not recalibrate")

    monkeypatch.setattr(FeatureBank, "calibrate", forbidden)
    controller = restore_controller(config, state)
    assert controller.route_ucb.counts.sum() > 0


def test_different_head_subsets_within_same_depth_batch(monkeypatch):
    config = ExperimentConfig.model_validate(
        {
            "dht": {"depths": [2], "head_counts": [2], "calibration_samples": 2},
        }
    )
    model = TransformerClassifier(config.model).eval()
    tokens, labels = synthetic_split(config, "train")
    trainer = DHTTrainer(
        config, model, torch.optim.AdamW(model.parameters()), tokens, labels, torch.device("cpu")
    )
    subsets = iter([(0, 1), (2, 3)])
    monkeypatch.setattr(trainer.controller.ts, "select", lambda *args, **kwargs: next(subsets))
    outputs = []
    for index in range(2):
        route = trainer.controller.route(model, tokens[index : index + 1], index)
        outputs.append(trainer.controller.execute(model, tokens[index : index + 1], route))
    assert outputs[0].architecture.head_indices[1] != outputs[1].architecture.head_indices[1]
    for index, item in enumerate(outputs):
        torch.testing.assert_close(
            item.logits, model(tokens[index : index + 1], architecture=item.architecture)
        )


@pytest.mark.parametrize("mode", ["tree", "integrated"])
def test_multi_layer_groups_keep_heads_and_integrated_count(mode):
    config = ExperimentConfig.model_validate(
        {
            "model": {"total_depth": 5, "prefix_depth": 1, "group_size": 2},
            "dht": {"mode": mode, "depths": [5], "head_counts": [2], "calibration_samples": 2},
        }
    )
    model = TransformerClassifier(config.model).eval()
    tokens, labels = synthetic_split(config, "train")
    trainer = DHTTrainer(
        config, model, torch.optim.AdamW(model.parameters()), tokens, labels, torch.device("cpu")
    )
    route = trainer.controller.route(model, tokens[:1], 0)
    execution = trainer.controller.execute(model, tokens[:1], route, explore=False)
    spec = execution.architecture
    assert len(execution.groups) == 2
    assert spec.head_indices[1] == spec.head_indices[2]
    assert spec.head_indices[3] == spec.head_indices[4]
    assert all(len(heads) == 2 for heads in spec.head_indices.values())
    assert spec.heads_for(0, 4) == (0, 1, 2, 3)
    torch.testing.assert_close(execution.logits, model(tokens[:1], architecture=spec))
