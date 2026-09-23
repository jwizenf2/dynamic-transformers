"""Measure end-to-end TRAINING throughput: fixed BERT vs per-sample vs batched DHT (D017).

Each variant starts from the same pinned pretrained BERT-base weights and trains
on the same AG News training slice for ``--steps`` optimizer updates after
``--warmup`` untimed updates. Timed work is everything a real run does per step:
routing (DHT), task forward, backward, AdamW, and bandit updates. Calibration
(one-off), validation and checkpoint I/O are excluded and reported separately.

Nothing is written to artifacts/runs; this is a benchmark, not an experiment.
Run on an otherwise idle device: concurrent jobs make timings meaningless.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from dht.config import load_config
from dht.data import load_split
from dht.models.architecture import ArchitectureSpec
from dht.models.pretrained import build_model
from dht.runtime import seed_everything, select_device, synchronize
from dht.training.trainer import DHTTrainer


def dht_variant(config, execution: str, routing_batch_size: int, pooling: str = "cpu_float64"):
    return config.model_copy(
        update={
            "dht": config.dht.model_copy(
                update={
                    "execution": execution,
                    "routing_batch_size": routing_batch_size,
                    "context_pooling": pooling,
                }
            )
        }
    )


def time_fixed(config, device, tokens, labels, steps, warmup) -> dict:
    seed_everything(config.seed, config.bandit.seed)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    spec = ArchitectureSpec(config.model.total_depth)
    model.train()
    size = config.batch_size
    started = None
    for step in range(warmup + steps):
        if step == warmup:
            synchronize(device)
            started = time.perf_counter()
        batch = (torch.arange(size) + step * size) % len(tokens)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(
            model(tokens[batch].to(device), architecture=spec), labels[batch].to(device)
        )
        loss.backward()
        optimizer.step()
    synchronize(device)
    seconds = time.perf_counter() - started
    return {
        "samples": steps * size,
        "seconds": seconds,
        "samples_per_second": steps * size / seconds,
    }


def time_dht(config, device, tokens, labels, steps, warmup) -> dict:
    seed_everything(config.seed, config.bandit.seed)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    trainer = DHTTrainer(config, model, optimizer, tokens, labels, device)
    for _ in range(warmup):
        trainer.step()
    synchronize(device)
    started = time.perf_counter()
    rows = [trainer.step() for _ in range(steps)]
    synchronize(device)
    seconds = time.perf_counter() - started
    samples = sum(r["batch_size"] for r in rows)
    parts = [
        "routing_seconds",
        "forward_including_head_policy_seconds",
        "backward_optimizer_seconds",
        "controller_update_seconds",
    ]
    return {
        "samples": samples,
        "seconds": seconds,
        "samples_per_second": samples / seconds,
        "calibration_seconds_excluded": trainer.calibration_seconds,
        "ms_per_sample": {p: 1000 * sum(r[p] for r in rows) / samples for p in parts},
        "mean_relative_task_flops": sum(
            r["selections"]["relative_task_forward_matmul_flops"] * r["batch_size"] for r in rows
        )
        / samples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="full-data DHT TOML")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = load_config(args.config)
    device = select_device(args.device, allow_fallback=False)
    torch.set_num_threads(base.cpu_threads)
    tokens, labels = load_split(base, "train")
    # Enough rows that queues never wrap: DHT routes ahead of what it trains.
    limit = (args.steps + args.warmup) * base.batch_size * 4 + base.dht.calibration_samples
    tokens, labels = tokens[:limit], labels[:limit]
    results = {
        "config": str(args.config),
        "device": str(device),
        "batch_size": base.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "fixed_bert": time_fixed(base, device, tokens, labels, args.steps, args.warmup),
    }
    variants = [
        ("dht_per_sample", "per_sample", 1, "cpu_float64"),
        ("dht_batched", "batched", base.batch_size, "cpu_float64"),
        ("dht_batched_device_pooling", "batched", base.batch_size, "device"),
    ]
    for name, execution, chunk, pooling in variants:
        results[name] = time_dht(
            dht_variant(base, execution, chunk, pooling),
            device,
            tokens,
            labels,
            args.steps,
            args.warmup,
        )
        print(json.dumps({name: results[name]["samples_per_second"]}), flush=True)
    fixed = results["fixed_bert"]["samples_per_second"]
    for name, *_ in variants:
        results[name]["speed_vs_fixed"] = results[name]["samples_per_second"] / fixed
    print(json.dumps(results, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
