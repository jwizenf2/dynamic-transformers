"""Class-balanced calibration must be deterministic, train-only, and fully recorded."""

import pytest
import torch

from dht.config import ExperimentConfig
from dht.training.calibration import calibration_indices


def test_stratified_selection_balances_and_does_not_consume_global_rng():
    config = ExperimentConfig(dht={"calibration_samples": 12, "calibration_sampling": "stratified"})
    labels = torch.tensor([0] * 20 + [1] * 20 + [2] * 20)
    before = torch.get_rng_state().clone()
    indices, record = calibration_indices(labels, config)
    assert torch.equal(before, torch.get_rng_state())
    assert record["class_counts"] == [4, 4, 4]
    assert len(set(indices.tolist())) == 12
    assert record["training_row_indices"] == indices.tolist()
    repeated, _ = calibration_indices(labels, config)
    assert torch.equal(indices, repeated)
    assert indices.min() >= 0 and indices.max() < len(labels)


def test_stratified_shortfalls_fail_instead_of_silent_four_example_fallback():
    config = ExperimentConfig(dht={"calibration_samples": 12, "calibration_sampling": "stratified"})
    with pytest.raises(ValueError, match="fit training data"):
        calibration_indices(torch.tensor([0, 1, 2]), config)
    with pytest.raises(ValueError, match="insufficient training records"):
        calibration_indices(torch.tensor([0] * 10 + [1] * 10), config)


def test_legacy_first_calibration_still_works():
    config = ExperimentConfig(dht={"calibration_samples": 20})
    indices, record = calibration_indices(torch.tensor([0, 1, 2]), config)
    assert indices.tolist() == [0, 1, 2]
    assert record["actual_samples"] == 3
