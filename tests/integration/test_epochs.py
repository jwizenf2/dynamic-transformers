"""Long-run contracts exercised on tiny offline data, including queue tails/resume."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from dht.cli import train
from dht.config import ExperimentConfig
from dht.training.checkpointing import read_checkpoint
from dht.training.epochs import scheduled_learning_rate


def config_for(mode="fixed", epochs=2, evaluate_test=False):
    return ExperimentConfig.model_validate(
        {
            "batch_size": 4,
            "model": {"dropout": 0.2},
            "evaluate_test": evaluate_test,
            "training": {"epochs": epochs, "checkpoint_every_steps": 1},
            "data": {"train_size": 11, "validation_size": 6, "test_size": 6},
            **({"dht": {"mode": mode, "calibration_samples": 4}} if mode != "fixed" else {}),
        }
    )


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            if not str(key).endswith("seconds") and key != "samples_per_second":
                assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            assert_same(left, right)
    else:
        assert a == b


@pytest.mark.parametrize("mode", ["fixed", "tree", "integrated"])
@pytest.mark.parametrize("after_validation", [False, True])
def test_epoch_exposure_tail_flush_and_exact_resume(tmp_path, mode, after_validation, monkeypatch):
    config = config_for(mode)
    complete = train(config, tmp_path / "complete")
    stop = complete["validation_history"][0]["step"] + 1 if after_validation else 2
    partial = train(config, tmp_path / "partial", stop_after=stop)
    assert not partial["complete"]
    assert partial["test"] is None
    if mode != "fixed":
        from dht.bandits.features import FeatureBank

        def forbidden(*args, **kwargs):
            raise AssertionError("resume must NOT recalibrate")

        monkeypatch.setattr(FeatureBank, "calibrate", forbidden)
    resumed = train(config, tmp_path / "resumed", tmp_path / "partial/checkpoint.pt")
    full_state = read_checkpoint(tmp_path / "complete/checkpoint.pt")
    resumed_state = read_checkpoint(tmp_path / "resumed/checkpoint.pt")
    for key in ("model", "optimizer", "rng", "training_state"):
        assert_same(full_state[key], resumed_state[key])
    assert_same(complete["validation_history"], resumed["validation_history"])
    assert complete["best_step"] == resumed["best_step"]
    assert complete["trained_samples"] == resumed["trained_samples"] == 22
    if mode != "fixed":
        assert not any(resumed["pending_queues"].values())
        assert resumed["routed_samples"] == 22
    assert Path(resumed["best_checkpoint"]).is_file()
    rows = [
        json.loads(line) for line in (tmp_path / "complete/metrics.jsonl").read_text().splitlines()
    ]
    for epoch in (1, 2):
        observed = []
        for row in rows:
            if row["epoch"] == epoch:
                observed.extend(
                    row.get(
                        "sample_ids", [sample["sample_id"] for sample in row.get("samples", [])]
                    )
                )
        assert Counter(observed) == Counter(range(11))
    assert_same(
        read_checkpoint(tmp_path / "complete/best.pt")["model"],
        read_checkpoint(tmp_path / "resumed/best.pt")["model"],
    )


def test_validation_selects_earliest_tie_and_test_uses_best(tmp_path, monkeypatch):
    values, captures = iter([0.8, 0.5, 0.8]), []

    def evaluate(model, config, device, split):
        captures.append((split, {key: value.clone() for key, value in model.state_dict().items()}))
        return {"macro_f1": next(values) if split == "validation" else 0.7}

    monkeypatch.setattr("dht.cli.evaluate_model", evaluate)
    result = train(config_for(epochs=3, evaluate_test=True), tmp_path / "run")
    assert result["best_step"] == result["validation_history"][0]["step"]
    assert result["best_validation_score"] == 0.8
    assert [split for split, _ in captures] == ["validation"] * 3 + ["test"]
    assert_same(captures[0][1], captures[-1][1])
    assert any(not torch.equal(captures[0][1][k], captures[2][1][k]) for k in captures[0][1])


def test_crash_after_publishing_new_best_keeps_old_resume_reference(tmp_path, monkeypatch):
    from dht.training import epochs

    original = epochs.save_checkpoint

    def fail_latest(path, *args, **kwargs):
        if path.name == "checkpoint.pt" and args[3] == 6:
            raise RuntimeError("simulated crash before latest rename")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(epochs, "save_checkpoint", fail_latest)
    scores = iter([0.5, 0.9, 0.9, 0.7])
    monkeypatch.setattr("dht.cli.evaluate_model", lambda *args: {"macro_f1": next(scores)})
    config = config_for(epochs=3)
    with pytest.raises(RuntimeError, match="simulated crash"):
        train(config, tmp_path / "failed")
    last = read_checkpoint(tmp_path / "failed/checkpoint.pt")
    assert last["step"] == 5 and last["run_state"]["best_step"] == 3
    assert Path(last["run_state"]["best_checkpoint"]).is_file()
    monkeypatch.setattr(epochs, "save_checkpoint", original)
    resumed = train(config, tmp_path / "resumed", tmp_path / "failed/checkpoint.pt")
    assert resumed["complete"] and resumed["best_step"] == 6


def test_sample_schedule_boundaries_and_conflicting_budgets():
    config = ExperimentConfig.model_validate(
        {"data": {"train_size": 100}, "training": {"epochs": 1, "warmup_fraction": 0.1}}
    )
    assert scheduled_learning_rate(config, 0) == pytest.approx(config.learning_rate / 10)
    assert scheduled_learning_rate(config, 9) == pytest.approx(config.learning_rate)
    assert scheduled_learning_rate(config, 54) == pytest.approx(config.learning_rate / 2)
    assert scheduled_learning_rate(config, 99) == 0
    with pytest.raises(ValueError, match="remove the legacy steps"):
        ExperimentConfig(steps=100, training={"epochs": 3})
