"""Trusted local step-boundary checkpoints, including optional complete DHT state.

Checkpoints are written to a temporary file then renamed so an interrupted
write cannot leave a half-written final checkpoint. Never load untrusted files.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dht.config import ExperimentConfig
from dht.runtime import RandomStreams

SCHEMA_VERSION = 3


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    step: int,
    streams: RandomStreams,
    training_state: dict | None = None,
    run_state: dict | None = None,
) -> None:
    """Save weights, optimizer, all random streams, and caller-owned DHT state.

    DHT training_state includes calibration, posterior statistics, reward history,
    pending queues, epoch/data cursor and cumulative execution counters.
    """
    state = {
        "schema_version": SCHEMA_VERSION,
        "scope": "dht" if training_state is not None else "deterministic_scaffold",
        "training_state": training_state,
        "run_state": run_state,
        "config": config.model_dump(mode="json"),
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_training": model.training,
        "rng": {
            "python": random.getstate(),
            "numpy_global": np.random.get_state(),
            "numpy_policy": streams.numpy.bit_generator.state,
            "torch_cpu": torch.get_rng_state(),
            "torch_policy": streams.policy.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def read_checkpoint(path: Path) -> dict:
    """Read trusted schema1/2 legacy or schema3 epoch-run checkpoints on CPU."""
    # Contains Python/NumPy RNG state. Only load checkpoints you created/trust.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema_version") not in (1, 2, SCHEMA_VERSION) or state.get("scope") not in (
        "deterministic_scaffold",
        "dht",
    ):
        raise ValueError("unsupported checkpoint schema or scope")
    return state


def restore_checkpoint(
    state: dict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    streams: RandomStreams,
) -> int:
    """Restore common state; the DHT trainer restores training_state separately."""
    if ExperimentConfig.model_validate(state["config"]).model_dump(
        mode="json"
    ) != config.model_dump(mode="json"):
        raise ValueError("resume requires the saved experiment configuration")
    model.load_state_dict(state["model"])
    model.train(state["model_training"])
    optimizer.load_state_dict(state["optimizer"])
    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy_global"])
    streams.numpy.bit_generator.state = rng["numpy_policy"]
    torch.set_rng_state(rng["torch_cpu"])
    streams.policy.set_state(rng["torch_policy"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    if rng["mps"] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(rng["mps"])
    return int(state["step"])
