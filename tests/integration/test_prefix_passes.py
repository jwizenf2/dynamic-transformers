"""Instrument Figure 1's TWO real passes, not just a no-gradient context tensor.

Use the published b=c=3 and 12 head identities, but tiny hidden width and synthetic
tokens. Single-action policies make the expected trace deterministic without
replacing route(), execute(), the queue, or the optimizer with test doubles.
"""

import pytest
import torch

from dht.config import ExperimentConfig
from dht.data import synthetic_split
from dht.models.transformer import TransformerClassifier
from dht.runtime import seed_everything
from dht.training.trainer import DHTTrainer


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("mode", ["tree", "integrated"])
@pytest.mark.parametrize("depth", [3, 6])
def test_prefix_routed_without_grad_then_reexecuted_and_updated(device, mode, depth):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    config = ExperimentConfig.model_validate(
        {
            "batch_size": 2,
            "model": {
                "total_depth": 9,
                "prefix_depth": 3,
                "group_size": 3,
                "num_heads": 12,
                "dropout": 0.1,
            },
            "data": {"train_size": 2},
            "dht": {
                "mode": mode,
                "depths": [depth],
                "head_counts": [8],
                "calibration_samples": 2,
            },
        }
    )
    seed_everything(config.seed, config.bandit.seed)
    target = torch.device(device)
    model = TransformerClassifier(config.model).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens, labels = synthetic_split(config, "train")
    trainer = DHTTrainer(config, model, optimizer, tokens, labels, target)
    before = {key: value.detach().cpu().clone() for key, value in model.named_parameters()}
    calls, activations = [], []

    def observe(index):
        def hook(layer, arguments, output):
            calls.append((index, torch.is_grad_enabled(), layer.training, arguments[2]))
            activations.append(output)
            if torch.is_grad_enabled():
                output.retain_grad()
            else:
                # No parameter freezing toggle: no_grad suppresses the GRAPH, and
                # eval disables dropout. Neither the weights nor gradients change.
                assert output.grad_fn is None and not output.requires_grad
                for name, parameter in model.named_parameters():
                    assert parameter.requires_grad and parameter.grad is None
                    assert torch.equal(parameter.detach().cpu(), before[name])

        return hook

    handles = [layer.register_forward_hook(observe(i)) for i, layer in enumerate(model.layers)]
    try:
        row = trainer.step()  # Real routing -> queue -> two forwards -> backward -> update.
    finally:
        for handle in handles:
            handle.remove()

    full = tuple(range(12))
    assert calls[:6] == [(i, False, False, full) for _ in range(2) for i in range(3)]
    expected_layers = list(range(3)) * 2 + list(range(depth)) * 2
    assert [call[0] for call in calls] == expected_layers
    assert len({id(output) for output in activations}) == len(activations)
    for (index, grad_enabled, training, heads), output in zip(
        calls[6:], activations[6:], strict=True
    ):
        assert grad_enabled and training
        assert output.grad is not None and torch.isfinite(output.grad).all()
        assert output.grad.abs().sum() > 0
        assert heads == full if index < 3 else len(heads) == 8
    for index in range(3):
        # Check a substantial task-gradient tensor in EACH prefix layer; key bias
        # has a softmax shift invariance and is not a sensible nonzero-gradient probe.
        parameter = model.layers[index].attention.query.weight
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
        assert not torch.equal(
            parameter.detach().cpu(), before[f"layers.{index}.attention.query.weight"]
        )
    assert model.tokens.weight.grad.abs().sum() > 0
    for index in range(depth, 9):
        for name, parameter in model.layers[index].named_parameters():
            assert parameter.grad is None
            assert torch.equal(parameter.detach().cpu(), before[f"layers.{index}.{name}"])
    assert row["batch_size"] == 2 and trainer.routed_samples == 2
    assert not any(trainer.queues.pending.values())
    assert model.training and all(p.requires_grad for p in model.parameters())
