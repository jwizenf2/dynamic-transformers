"""Diagnose pretrained CPU/MPS activations before any architecture-policy training.

Small batch-size/backend checks detect device-specific failures that a CPU unit
suite cannot. This is a read-only check: weights and policies are never updated.
"""

import argparse
import json
from pathlib import Path

import torch

from dht.bandits.features import pooled_features
from dht.config import load_config
from dht.data import load_split
from dht.models.pretrained import build_model
from dht.runtime import seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    torch.set_num_threads(config.cpu_threads)
    seed_everything(config.seed, config.bandit.seed)
    model = build_model(config).eval()
    tokens, _ = load_split(config, "train")
    with torch.no_grad():
        for device in ("cpu", "mps"):
            if device == "mps" and not torch.backends.mps.is_available():
                continue
            model.to(device)
            for batch_size in (1, 8):
                hidden, mask = model.embed(tokens[:batch_size].to(device))
                for index in range(config.model.prefix_depth + 1):
                    values = hidden.cpu()
                    valid = values[mask.cpu()]
                    # Also check the exact device-index/float64 path used by contexts.
                    pooled_input = hidden[0].detach()[mask[0]].to("cpu", dtype=torch.float64)
                    corrected = pooled_features(hidden[0], mask[0], (1, 2), (4, 8, 16))
                    reference = pooled_features(values[0], mask[0].cpu(), (1, 2), (4, 8, 16))
                    torch.testing.assert_close(corrected, reference, rtol=0, atol=0)
                    print(
                        json.dumps(
                            {
                                "device": device,
                                "batch_size": batch_size,
                                "after_layers": index,
                                "all_finite": torch.isfinite(values).all().item(),
                                "valid_finite": torch.isfinite(valid).all().item(),
                                "pooled_input_finite": torch.isfinite(pooled_input).all().item(),
                                "corrected_pool_matches_cpu": True,
                                "max_abs": values.abs().max().item(),
                            },
                            allow_nan=True,
                        ),
                        flush=True,
                    )
                    if index < config.model.prefix_depth:
                        hidden = model.layers[index](
                            hidden, mask, tuple(range(config.model.num_heads))
                        )
