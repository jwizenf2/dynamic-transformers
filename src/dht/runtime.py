"""Explicit device selection, random streams, and local run provenance."""

import hashlib
import importlib.metadata
import platform
import random
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dht.config import ExperimentConfig


@dataclass
class RandomStreams:
    numpy: np.random.Generator
    policy: torch.Generator


def seed_everything(seed: int, policy_seed: int) -> RandomStreams:
    """Seed global libraries and create separate reproducible controller streams.

    Reproducibility is guaranteed only on a verified backend/software stack; enabling
    deterministic algorithms does not make different devices numerically identical.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return RandomStreams(
        np.random.default_rng(policy_seed), torch.Generator().manual_seed(policy_seed)
    )


def select_device(requested: str, allow_fallback: bool = True) -> torch.device:
    """Prefer CUDA/MPS/CPU for auto; make an unavailable explicit device visible."""
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if requested == "auto":
        requested = next(name for name in ("cuda", "mps", "cpu") if available[name])
    if requested not in available:
        raise ValueError(f"unknown device: {requested}")
    if not available[requested]:
        if not allow_fallback:
            raise RuntimeError(f"requested device {requested} is unavailable")
        warnings.warn(
            f"requested device {requested} unavailable; falling back to CPU", stacklevel=2
        )
        requested = "cpu"
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    """Wait for accelerator work before reading wall-clock timers; CPU is synchronous."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def metadata(config: ExperimentConfig, device: torch.device) -> dict:
    """Capture effective config, dependency versions and exact source hashes.

    Git revision alone is insufficient during development with uncommitted edits.
    File hashes preserve which source/configuration/lock content produced a run.
    """
    root = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    files = [
        *sorted((root / "src").rglob("*.py")),
        *sorted((root / "configs").rglob("*.toml")),
        root / "uv.lock",
        root / "pyproject.toml",
    ]
    result = {
        "config": config.model_dump(mode="json"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "file_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
            if p.is_file()
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pydantic", "dynamic-transformers")
        },
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.machine(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_threads": torch.get_num_threads(),
        "cuda_runtime": torch.version.cuda,
        "mps_available": torch.backends.mps.is_available(),
    }
    if config.data.name == "ag_news":
        import json

        result["dataset_manifest"] = json.loads(
            (Path(config.data.prepared_directory) / "manifest.json").read_text()
        )
    for name in ("tokenizers", "safetensors"):
        try:
            result["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result
