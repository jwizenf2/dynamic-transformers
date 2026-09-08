"""Reproducible training-only calibration selection (our protocol, D013).

The paper describes all-data descriptors but leaves the sampling/refresh details
unclear (§4.2 pp.442–443). A larger balanced subset is a bounded approximation,
NOT a claim to recover the authors' exact arm features or normalization procedure.
"""

import torch
from torch import Tensor

from dht.config import ExperimentConfig


def calibration_indices(labels: Tensor, config: ExperimentConfig) -> tuple[Tensor, dict]:
    """Choose distinct training row IDs with a private RNG; never touch task RNG.

    Labels only control class-balanced sampling. They are not inputs to FeatureBank
    or bandit rewards during calibration. The selected IDs/counts are checkpointed.
    Stratified selection fails rather than silently using a tiny or unbalanced set.
    Legacy `first` preserves previous runs' bounded-prefix convention.
    """
    options = config.dht
    if options is None or labels.ndim != 1 or not len(labels):
        raise ValueError("calibration requires DHT and nonempty training labels")
    if options.calibration_sampling == "first":
        indices = torch.arange(min(len(labels), options.calibration_samples))
    else:
        count, classes = options.calibration_samples, config.model.num_classes
        if count > len(labels) or count < classes:
            raise ValueError("stratified calibration must fit training data and cover all classes")
        generator = torch.Generator().manual_seed(options.calibration_seed)
        selected = []
        for label in range(classes):
            candidates = torch.where(labels.cpu() == label)[0]
            required = count // classes + int(label < count % classes)
            if len(candidates) < required:
                raise ValueError(f"insufficient training records for calibration class {label}")
            order = torch.randperm(len(candidates), generator=generator)
            selected.append(candidates[order[:required]])
        indices = torch.cat(selected)
        indices = indices[torch.randperm(len(indices), generator=generator)]
    return indices, {
        "method": options.calibration_sampling,
        "seed": options.calibration_seed,
        "requested_samples": options.calibration_samples,
        "actual_samples": len(indices),
        "training_row_indices": indices.tolist(),
        "class_counts": torch.bincount(
            labels.cpu()[indices], minlength=config.model.num_classes
        ).tolist(),
    }
