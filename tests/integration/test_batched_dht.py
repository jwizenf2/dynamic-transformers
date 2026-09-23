"""D017: batched DHT execution must be the SAME algorithm as the per-sample path.

The batched path only changes how work is laid out on the device. These tests
pin every observable: routing arms and stored contexts, per-group count arms,
selected heads, TS random-stream consumption, logits, parameter gradients, and
multi-step training trajectories. Float64 models make comparisons exact enough
that any logic difference (not summation order) would fail loudly.
"""

import copy

import pytest
import torch

from dht.cli import train
from dht.config import ExperimentConfig
from dht.data import synthetic_split
from dht.models.attention import SelectableAttention
from dht.models.transformer import TransformerClassifier
from dht.runtime import seed_everything
from dht.training.checkpointing import read_checkpoint
from dht.training.queues import DepthQueues
from dht.training.trainer import DHTTrainer


def configuration(mode="tree", **dht):
    return ExperimentConfig.model_validate(
        {
            "steps": 6,
            "batch_size": 3,
            "model": {"dropout": 0.0, "total_depth": 5, "prefix_depth": 1, "group_size": 2},
            "data": {"train_size": 12, "validation_size": 6, "test_size": 6},
            "dht": {
                "mode": mode,
                "depths": [1, 3, 5],
                "head_counts": [2, 3, 4],
                "calibration_samples": 6,
                **dht,
            },
        }
    )


def trainer_for(config):
    streams = seed_everything(config.seed, config.bandit.seed)
    del streams
    model = TransformerClassifier(config.model).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens, labels = synthetic_split(config, "train")
    return DHTTrainer(config, model, optimizer, tokens, labels, torch.device("cpu"))


def test_per_sample_head_attention_matches_individual_samples():
    torch.manual_seed(0)
    attention = SelectableAttention(24, 4).double()
    hidden = torch.randn(3, 5, 24, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[1, 3:] = False
    heads = [(0, 2), (1, 3), (2, 3)]
    batched = attention(hidden, mask, heads)
    single = torch.cat([attention(hidden[b : b + 1], mask[b : b + 1], heads[b]) for b in range(3)])
    torch.testing.assert_close(batched, single, rtol=0, atol=1e-12)
    parameters = [hidden, *attention.parameters()]
    for a, b in zip(
        torch.autograd.grad(batched.square().sum(), parameters),
        torch.autograd.grad(single.square().sum(), parameters),
        strict=True,
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="equally many"):
        attention(hidden, mask, [(0, 1), (2,), (1, 3)])


@pytest.mark.parametrize("mode", ["tree", "integrated"])
@pytest.mark.parametrize("explore", [True, False])
def test_execute_batch_matches_per_sample_decisions_logits_and_gradients(mode, explore):
    trainer = trainer_for(configuration(mode))
    controller, model = trainer.controller, trainer.model
    tokens = trainer.tokens[:6]
    routes = [controller.route(model, tokens[i : i + 1], i) for i in range(6)]
    batched_routes = controller.route_batch(model, tokens, list(range(6)))
    for single, batched in zip(routes, batched_routes, strict=True):
        assert (single.route_arm, single.total_depth, single.head_count) == (
            batched.route_arm,
            batched.total_depth,
            batched.head_count,
        )
        torch.testing.assert_close(single.route_features, batched.route_features)
    # Force a mixed-depth batch so exits and head-count partitions are exercised.
    depths = controller.depths if mode == "tree" else [p[0] for p in controller.pairs]
    for index, route in enumerate(routes):
        arm = index % len(depths)
        route.route_arm = arm
        route.total_depth = depths[arm]
        if mode == "integrated":
            route.head_count = controller.pairs[arm][1]
    assert len({r.total_depth for r in routes}) > 1

    rng = controller.ts.generator.get_state()
    model.zero_grad()
    reference = [
        controller.execute(model, tokens[i : i + 1], route, explore=explore)
        for i, route in enumerate(routes)
    ]
    torch.cat([e.logits for e in reference]).square().sum().backward()
    reference_grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    reference_rng = controller.ts.generator.get_state()

    controller.ts.generator.set_state(rng)
    model.zero_grad()
    batched = controller.execute_batch(model, tokens, routes, explore=explore)
    torch.cat([e.logits for e in batched]).square().sum().backward()
    batched_grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    assert torch.equal(controller.ts.generator.get_state(), reference_rng)

    for one, many in zip(reference, batched, strict=True):
        assert one.architecture.to_dict() == many.architecture.to_dict()
        assert [(g.start_layer, g.count_arm, g.head_indices) for g in one.groups] == [
            (g.start_layer, g.count_arm, g.head_indices) for g in many.groups
        ]
        torch.testing.assert_close(one.logits, many.logits, rtol=0, atol=1e-10)
    for a, b in zip(reference_grads, batched_grads, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-10)
    # Coverage: mixed head counts in one group force the multi-partition path.
    # (An untrained frozen tree UCB picks one count for everyone; that case is
    # still an exact-equivalence check, just not a partition-coverage one.)
    if explore or mode == "integrated":
        assert len({len(g.head_indices) for e in batched for g in e.groups}) > 1


@pytest.mark.parametrize("mode", ["tree", "integrated"])
def test_batched_training_follows_the_per_sample_trajectory(mode):
    reference = trainer_for(configuration(mode))
    batched = trainer_for(configuration(mode, execution="batched"))
    for _ in range(6):
        one, many = reference.step(), batched.step()
        assert [s["architecture"] for s in one["samples"]] == [
            s["architecture"] for s in many["samples"]
        ]
        assert one["loss"] == pytest.approx(many["loss"], abs=1e-9)
    for a, b in zip(reference.model.parameters(), batched.model.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-9)
    torch.testing.assert_close(reference.controller.ts.mean, batched.controller.ts.mean)
    torch.testing.assert_close(reference.controller.route_ucb.A, batched.controller.route_ucb.A)


def test_chunked_routing_conserves_samples_and_order():
    config = configuration("tree", execution="batched", routing_batch_size=4)
    trainer = trainer_for(config)
    seen = []
    while trainer.epoch == 0:
        batch = trainer.next_batch()
        seen.extend(item.sample_id for item in batch)
        if trainer.cursor == len(trainer.tokens) and not any(trainer.queues.pending.values()):
            break
    assert sorted(seen) == list(range(len(trainer.tokens)))
    assert trainer.routed_samples == len(trainer.tokens)


def test_queue_overfill_releases_fifo_capacity_batches():
    queues = DepthQueues((1, 3), capacity=2)
    from dht.policies.controller import RouteDecision

    for sample in range(5):
        queues.push(RouteDecision(sample, 1, 0, torch.zeros(1)))
    assert [r.sample_id for r in queues.pop()] == [0, 1]
    restored = DepthQueues((1, 3), capacity=2)
    restored.load_state_dict(copy.deepcopy(queues.state_dict()))
    assert [r.sample_id for r in restored.pop()] == [2, 3]
    assert [r.sample_id for r in restored.pop(flush=True)] == [4]


def test_batched_predict_matches_per_sample_predict(tmp_path):
    config = configuration("tree")
    train(config, tmp_path / "run")
    state = read_checkpoint(tmp_path / "run/checkpoint.pt")
    fast = config.model_copy(update={"dht": config.dht.model_copy(update={"execution": "batched"})})
    from dht.training.trainer import restore_controller

    model = TransformerClassifier(config.model)
    model.load_state_dict(state["model"])
    tokens, _ = synthetic_split(config, "validation")
    slow_logits, slow_specs = restore_controller(config, state).predict(model, tokens)
    fast_logits, fast_specs = restore_controller(fast, state).predict(model, tokens)
    assert [s.to_dict() for s in slow_specs] == [s.to_dict() for s in fast_specs]
    torch.testing.assert_close(slow_logits, fast_logits, rtol=1e-5, atol=1e-5)


def test_device_pooling_matches_float64_reference():
    """D018: prefix-sum batched pooling reproduces adaptive pooling per valid length."""
    from dht.bandits.features import pooled_features, pooled_features_batch

    generator = torch.Generator().manual_seed(4)
    hidden = torch.randn(5, 11, 24, generator=generator)
    lengths = [11, 1, 2, 7, 5]
    mask = torch.arange(11)[None] < torch.tensor(lengths)[:, None]
    batched = pooled_features_batch(hidden, mask, (1, 2, 4), (1, 3, 8))
    for row in range(5):
        reference = pooled_features(hidden[row], mask[row], (1, 2, 4), (1, 3, 8))
        torch.testing.assert_close(batched[row].double(), reference, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="right padding"):
        pooled_features_batch(hidden, mask.flip(1), (1,), (1,))


def test_ucb_select_many_matches_per_sample_select():
    from dht.bandits.linear_ucb import LinearUCB

    generator = torch.Generator().manual_seed(8)
    ucb = LinearUCB(4, 6, ridge=1.0, exploration=0.7)
    for _ in range(40):
        arm = int(torch.randint(4, (1,), generator=generator))
        ucb.update(
            arm, torch.randn(6, generator=generator, dtype=torch.float64), float(torch.rand(1))
        )
    contexts = torch.randn(25, 4, 6, generator=generator, dtype=torch.float64)
    for explore in (True, False):
        assert ucb.select_many(contexts, explore=explore) == [
            ucb.select(c, explore=explore) for c in contexts
        ]


@pytest.mark.parametrize("mode", ["tree", "integrated"])
def test_device_pooling_mode_matches_exact_decisions_on_synthetic_data(mode):
    exact = trainer_for(configuration(mode, execution="batched"))
    device = trainer_for(configuration(mode, execution="batched", context_pooling="device"))
    for _ in range(4):
        one, many = exact.step(), device.step()
        assert [s["architecture"] for s in one["samples"]] == [
            s["architecture"] for s in many["samples"]
        ]
    with pytest.raises(ValueError, match="requires execution"):
        configuration(mode, context_pooling="device")
