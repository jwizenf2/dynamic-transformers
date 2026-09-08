import random

import numpy as np
import pytest
import torch

from dht.cli import train
from dht.config import ExperimentConfig
from dht.models.transformer import TransformerClassifier
from dht.runtime import seed_everything
from dht.training.checkpointing import read_checkpoint, restore_checkpoint, save_checkpoint


def test_all_rng_streams_and_optimizer_round_trip(tmp_path):
    config = ExperimentConfig()
    streams = seed_everything(config.seed, config.bandit.seed)
    model = TransformerClassifier(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    model(torch.tensor([[1, 2, 3]])).sum().backward()
    optimizer.step()
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, optimizer, config, 1, streams)

    def sample():
        return (
            random.random(),
            np.random.rand(),
            torch.rand(1).item(),
            streams.numpy.random(),
            torch.rand(1, generator=streams.policy).item(),
        )

    expected = sample()
    with torch.no_grad():
        next(model.parameters()).add_(1)
    state = read_checkpoint(path)
    assert restore_checkpoint(state, model, optimizer, config, streams) == 1
    assert sample() == expected
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state["model"][key], rtol=0, atol=0)
    assert optimizer.state_dict()["state"]
    with pytest.raises(ValueError, match="configuration"):
        restore_checkpoint(state, model, optimizer, ExperimentConfig(seed=8), streams)


def test_resume_matches_uninterrupted_training_with_dropout(tmp_path):
    config = ExperimentConfig.model_validate(
        {
            "steps": 4,
            "model": {"dropout": 0.2},
            "architecture": {"total_depth": 2, "head_indices": {"1": [1, 3]}},
        }
    )
    complete = train(config, tmp_path / "complete")
    train(config, tmp_path / "partial", stop_after=2)
    resumed = train(config, tmp_path / "resumed", tmp_path / "partial/checkpoint.pt")
    full_state = read_checkpoint(tmp_path / "complete/checkpoint.pt")
    resume_state = read_checkpoint(tmp_path / "resumed/checkpoint.pt")
    for key, value in full_state["model"].items():
        torch.testing.assert_close(value, resume_state["model"][key], rtol=0, atol=0)
    assert complete["validation"] == resumed["validation"]
    assert complete["test"] == resumed["test"]
    for key, value in full_state["optimizer"]["state"].items():
        for field, tensor in value.items():
            torch.testing.assert_close(tensor, resume_state["optimizer"]["state"][key][field])
    with pytest.raises(FileExistsError):
        train(config, tmp_path / "complete")
