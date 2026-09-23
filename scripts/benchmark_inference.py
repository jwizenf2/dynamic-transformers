"""Frozen-inference throughput: fixed BERT vs per-sample vs batched DHT (D017).

Loads trusted local validation-selected checkpoints, runs the same validation
rows through each model, and checks that batched DHT returns EXACTLY the same
architectures as the per-sample reference (and near-identical logits). Timing
includes routing, bandit policy, task forward and CPU transfers. Validation rows
only; the public test split is never touched. Run on an otherwise idle device.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from dht.config import ExperimentConfig
from dht.data import load_split
from dht.models.architecture import ArchitectureSpec
from dht.models.pretrained import build_model
from dht.runtime import select_device, synchronize
from dht.training.checkpointing import read_checkpoint
from dht.training.trainer import restore_controller


def load(path: Path, device: torch.device):
    state = read_checkpoint(path)
    config = ExperimentConfig.model_validate(state["config"])
    model = build_model(config, initialize=False).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return state, config, model


def timed(function, batches, device, warmup: int):
    with torch.inference_mode():
        for batch in batches[:warmup]:
            function(batch)
        synchronize(device)
        started = time.perf_counter()
        outputs = [function(batch) for batch in batches]
        synchronize(device)
    return time.perf_counter() - started, outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--dht", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = select_device(args.device, allow_fallback=False)
    _, fixed_config, fixed = load(args.fixed, device)
    torch.set_num_threads(fixed_config.cpu_threads)
    tokens, _ = load_split(fixed_config, "validation")
    tokens = tokens[: args.samples]
    results = {"device": str(device), "samples": len(tokens), "results": []}
    spec = ArchitectureSpec(fixed_config.model.total_depth)
    for size in args.batch_sizes:
        batches = [tokens[i : i + size].to(device) for i in range(0, len(tokens), size)]
        seconds, _ = timed(lambda b: fixed(b, architecture=spec), batches, device, args.warmup)
        results["results"].append(
            {"model": "fixed", "batch_size": size, "samples_per_second": len(tokens) / seconds}
        )
        print(json.dumps(results["results"][-1]), flush=True)
    for path in args.dht:
        state, config, model = load(path, device)
        for size in args.batch_sizes:
            batches = [tokens[i : i + size].to(device) for i in range(0, len(tokens), size)]
            runs = {}
            variants = {
                "per_sample": {"execution": "per_sample"},
                "batched": {"execution": "batched"},
                "batched_device_pooling": {"execution": "batched", "context_pooling": "device"},
            }
            for name, update in variants.items():
                variant = config.model_copy(update={"dht": config.dht.model_copy(update=update)})
                controller = restore_controller(variant, state)
                runs[name] = timed(
                    lambda b, c=controller, m=model: c.predict(m, b), batches, device, args.warmup
                )
            _, reference = runs["per_sample"]
            reference_specs = [s.to_dict() for batch in reference for s in batch[1]]
            for name, (seconds, outputs) in runs.items():
                specs = [s.to_dict() for batch in outputs for s in batch[1]]
                results["results"].append(
                    {
                        "model": f"{config.dht.mode}:{path.parent.name}",
                        "execution": name,
                        "batch_size": size,
                        "samples_per_second": len(tokens) / seconds,
                        "architecture_agreement": sum(
                            a == b for a, b in zip(specs, reference_specs, strict=True)
                        )
                        / len(specs),
                        "identical_architectures": specs == reference_specs,
                        "max_logit_difference": max(
                            float((a[0] - b[0]).abs().max())
                            for a, b in zip(outputs, reference, strict=True)
                        ),
                    }
                )
                print(json.dumps(results["results"][-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
