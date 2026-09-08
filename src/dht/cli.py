"""Local fixed/DHT train, evaluate and benchmark entry points.

Presence of a [dht] configuration section selects the learned controller workflow.
The fixed path remains useful as a matched baseline and a debugging reference.
"""

import argparse
import csv
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.nn import functional as F

from dht.config import ExperimentConfig, load_config
from dht.data import load_split
from dht.evaluation.flops import forward_matmul_flops
from dht.evaluation.metrics import classification_metrics
from dht.models.architecture import ArchitectureSpec
from dht.models.pretrained import build_model
from dht.models.transformer import TransformerClassifier
from dht.runtime import metadata, seed_everything, select_device, synchronize
from dht.training.checkpointing import read_checkpoint, restore_checkpoint, save_checkpoint


def write_json(path: Path, value: dict | list) -> None:
    """Write readable finite JSON; NaN/Infinity should fail rather than pollute results."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@torch.no_grad()
def evaluate_model(
    model: TransformerClassifier, config: ExperimentConfig, device: torch.device, split: str
) -> dict[str, float]:
    """Evaluate the fixed baseline without dropout, updates, or train-mode side effects."""
    was_training = model.training
    model.eval()
    tokens, labels = load_split(config, split)
    predictions = []
    spec = ArchitectureSpec.from_config(config)
    try:
        for start in range(0, len(tokens), config.batch_size):
            logits = model(tokens[start : start + config.batch_size].to(device), architecture=spec)
            predictions.append(logits.argmax(-1).cpu())
        return classification_metrics(torch.cat(predictions), labels, config.model.num_classes)
    finally:
        model.train(was_training)


def train(
    config: ExperimentConfig,
    output: Path,
    resume: Path | None = None,
    stop_after: int | None = None,
) -> dict:
    """Dispatch a bounded run; [dht] enables the independently implemented policy."""
    if config.training is not None:
        from dht.training.epochs import train_epochs

        return train_epochs(config, output, resume, stop_after)
    if config.dht is not None:
        from dht.training.trainer import train_dynamic

        return train_dynamic(config, output, resume, stop_after)
    if stop_after is not None and not 1 <= stop_after <= config.steps:
        raise ValueError("stop_after must be in [1, configured steps]")
    output.mkdir(parents=True, exist_ok=False)  # never overwrite an existing run
    torch.set_num_threads(config.cpu_threads)
    streams = seed_everything(config.seed, config.bandit.seed)
    device = select_device(config.device, config.allow_device_fallback)
    model = build_model(config, initialize=resume is None).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    step = 0
    if resume is not None:
        step = restore_checkpoint(read_checkpoint(resume), model, optimizer, config, streams)
    start_step = step
    target_step = config.steps if stop_after is None else stop_after
    if target_step <= step:
        raise ValueError("resume target must exceed checkpoint step")
    run_metadata = metadata(config, device)
    run_metadata["resume_from"] = str(resume.resolve()) if resume else None
    run_metadata["started_at"] = datetime.now(UTC).isoformat()
    write_json(output / "metadata.json", run_metadata)
    write_json(output / "config.json", config.model_dump(mode="json"))
    spec = ArchitectureSpec.from_config(config)
    tokens, labels = load_split(config, "train")
    rows = []
    model.train()
    synchronize(device)
    start_time = time.perf_counter()
    for step in range(start_step, target_step):
        # Defined cyclic ordering: cursor reconstructs from completed optimizer steps.
        indices = (torch.arange(config.batch_size) + step * config.batch_size) % len(tokens)
        inputs, targets = tokens[indices].to(device), labels[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs, architecture=spec), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite task loss")
        loss.backward()
        optimizer.step()
        rows.append({"step": step + 1, "loss": loss.detach().item()})
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(rows[-1], allow_nan=False) + "\n")
        if (step + 1) % config.log_every == 0 or step == start_step:
            print(json.dumps(rows[-1]), flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - start_time
    save_checkpoint(output / "checkpoint.pt", model, optimizer, config, target_step, streams)
    write_json(output / "metrics.json", rows)
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "loss"])
        writer.writeheader()
        writer.writerows(rows)
    full = ArchitectureSpec(config.model.total_depth)
    selected_flops = forward_matmul_flops(config.model, spec, config.data.sequence_length)
    counts = [len(spec.heads_for(i, config.model.num_heads)) for i in range(spec.total_depth)]
    summary = {
        "scope": f"fixed_{config.data.name}",
        "completed_steps": target_step,
        "segment_start_step": start_step,
        "segment_training_seconds": elapsed,
        "complete": target_step == config.steps,
        "architecture": spec.to_dict(),
        "average_total_depth": spec.total_depth,
        "average_heads_per_executed_layer": sum(counts) / len(counts),
        "head_counts_by_layer": counts,
        "forward_matmul_flops_per_sample": selected_flops,
        "relative_forward_matmul_flops": selected_flops
        / forward_matmul_flops(config.model, full, config.data.sequence_length),
        "validation": evaluate_model(model, config, device, "validation"),
        "test": evaluate_model(model, config, device, "test")
        if target_step == config.steps and config.evaluate_test
        else None,
        "checkpoint": str((output / "checkpoint.pt").resolve()),
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    """CLI arguments choose files/devices; research settings remain in TOML."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--output", type=Path)
    training.add_argument("--resume", type=Path)
    training.add_argument("--stop-after", type=int)
    for name in ("evaluate", "benchmark"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
        command.add_argument("--output", type=Path)
        if name == "evaluate":
            command.add_argument("--split", choices=["validation", "test"], default="validation")
        else:
            command.add_argument("--repeats", type=int, default=10)
            command.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.command == "train":
        config = load_config(args.config)
        output = args.output or Path("artifacts/runs") / datetime.now(UTC).strftime(
            "%Y%m%dT%H%M%S_%fZ"
        )
        result = train(config, output, args.resume, args.stop_after)
        result["run_directory"] = str(output)
    else:
        state = read_checkpoint(args.checkpoint)
        config = ExperimentConfig.model_validate(state["config"])
        torch.set_num_threads(config.cpu_threads)
        seed_everything(config.seed, config.bandit.seed)
        device = select_device(args.device)
        model = build_model(config, initialize=False).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        controller = None
        if config.dht is not None:
            from dht.training.trainer import evaluate_dynamic, restore_controller

            controller = restore_controller(config, state)
        if args.command == "evaluate":
            result = {
                "split": args.split,
                "metrics": evaluate_dynamic(model, controller, config, device, args.split)
                if controller
                else evaluate_model(model, config, device, args.split),
            }
        else:
            if args.repeats < 1 or args.warmup < 0:
                parser.error("repeats must be positive and warmup nonnegative")
            tokens, _ = load_split(config, "validation")
            batch = tokens[: config.batch_size].to(device)
            spec = ArchitectureSpec.from_config(config)
            timings = []
            with torch.inference_mode():
                for iteration in range(args.warmup + args.repeats):
                    synchronize(device)
                    started = time.perf_counter()
                    if controller:
                        controller.predict(model, batch)
                    else:
                        model(batch, architecture=spec)
                    synchronize(device)
                    if iteration >= args.warmup:
                        timings.append(time.perf_counter() - started)
            result = {
                "scope": f"{config.data.name}_dht_routing_policy_and_forward"
                if controller
                else f"{config.data.name}_forward_only",
                "warmup": args.warmup,
                "repeats": args.repeats,
                "batch_size": len(batch),
                "sequence_length": batch.shape[1],
                "batch_seconds": timings,
                "mean_batch_seconds": sum(timings) / len(timings),
                "samples_per_second": len(batch) * len(timings) / sum(timings),
                "includes_routing_or_controller": controller is not None,
                "peak_memory_bytes": None,
                "peak_memory_note": "not instrumented; not reported as zero",
            }
        result["checkpoint"] = str(args.checkpoint.resolve())
        result["metadata"] = metadata(config, device)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
