"""Versioned synthetic data for offline correctness tests; no NLP claims."""

import torch
from torch import Tensor

from dht.config import ExperimentConfig


def load_split(config: ExperimentConfig, split: str) -> tuple[Tensor, Tensor]:
    """Select a configured offline source; real-data preparation is a separate CLI."""
    if config.data.name == "ag_news":
        from dht.ag_news import load_prepared

        return load_prepared(config, split)
    return synthetic_split(config, split)


def synthetic_split(config: ExperimentConfig, split: str) -> tuple[Tensor, Tensor]:
    """Generate a reproducible offline task with independent split random streams.

    Token 1 is a synthetic CLS marker; token 0 is reserved for padding. The label
    depends on the second token, so the model needs attention to classify from CLS.
    This is a software test dataset, not a stand-in for AG News performance.
    """
    sizes = {
        "train": (config.data.train_size, 0),
        "validation": (config.data.validation_size, 1),
        "test": (config.data.test_size, 2),
    }
    if split not in sizes:
        raise ValueError(f"unsupported split: {split}")
    size, offset = sizes[split]
    generator = torch.Generator().manual_seed(config.seed + offset)
    tokens = torch.randint(
        2, config.model.vocab_size, (size, config.data.sequence_length), generator=generator
    )
    tokens[:, 0] = 1
    labels = (tokens[:, 1] - 2) % config.model.num_classes
    return tokens, labels
