import pytest
from pydantic import ValidationError

from dht.config import ExperimentConfig, ModelConfig, load_config
from dht.models.architecture import ArchitectureSpec


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_depth": 0},
        {"total_depth": True},
        {"total_depth": 2, "head_indices": {1: ()}},
        {"total_depth": 2, "head_indices": {1: (1, 1)}},
        {"total_depth": 2, "head_indices": {1: (-1,)}},
        {"total_depth": 2, "head_indices": {1: (True,)}},
        {"total_depth": 2, "head_indices": {"1": (1,)}},
    ],
)
def test_invalid_spec(kwargs):
    with pytest.raises(ValueError):
        ArchitectureSpec(**kwargs)


@pytest.mark.parametrize(
    "spec",
    [
        ArchitectureSpec(4),
        ArchitectureSpec(2, {0: (1,)}),
        ArchitectureSpec(2, {1: (4,)}),
        ArchitectureSpec(2, {2: (1,)}),
    ],
)
def test_invalid_model_compatibility(spec, model_config):
    with pytest.raises(ValueError):
        spec.validate(model_config)


def test_prefix_and_group_boundary():
    config = ModelConfig(total_depth=5, prefix_depth=1, group_size=2)
    ArchitectureSpec(1).validate(config)
    ArchitectureSpec(3, {1: (0, 2), 2: (2, 0)}).validate(config)
    with pytest.raises(ValueError, match="complete groups"):
        ArchitectureSpec(2).validate(config)
    with pytest.raises(ValueError, match="same heads"):
        ArchitectureSpec(3, {1: (0, 2)}).validate(config)


def test_immutable_copied_spec():
    original = {1: [3, 1]}
    spec = ArchitectureSpec(2, original)
    original[1].append(0)
    assert spec.head_indices[1] == (1, 3)
    with pytest.raises(TypeError):
        spec.head_indices[1] = (0,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 25},
        {"prefix_depth": 4},
        {"group_size": 3},
        {"dropout": 1.0},
        {"hidden_size": True},
        {"layer_norm_eps": float("nan")},
        {"unknown": 2},
    ],
)
def test_invalid_model_config(kwargs):
    with pytest.raises(ValidationError):
        ModelConfig(**kwargs)


def test_experiment_cross_validation():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"data": {"sequence_length": 20}})
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {"architecture": {"total_depth": 2, "head_indices": {"01": [1]}}}
        )
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"learning_rate": float("inf")})


def test_checked_in_configs():
    for name in ("smoke", "smoke_dynamic"):
        config = load_config(f"configs/experiment/{name}.toml")
        ArchitectureSpec.from_config(config).validate(config.model)
