"""Run bounded fixed/tree/integrated diagnostics with live JSON logs.

The observation hooks call original functions unchanged. They inspect detached
losses/gradients and check policy state; they do not change actions, rewards,
optimizer settings, or RNG state. Instrumented training times are NOT benchmarks.
Use the normal CLI benchmark separately for latency measurements.

Example:
    uv run python scripts/audit_training.py --output artifacts/runs/audit-001
"""

import argparse
import hashlib
import json
import math
import statistics
import traceback
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from dht.cli import train
from dht.config import ExperimentConfig, load_config
from dht.training.trainer import DHTTrainer


def run_audit(config: ExperimentConfig, output: Path, log_every: int) -> dict:
    """Run unmodified training under read-only diagnostic observation hooks."""
    losses, events = [], []
    optimizer_steps = 0
    trained_samples = 0
    max_gradient = 0.0
    original_optimizer_step = torch.optim.AdamW.step
    original_loss = F.cross_entropy
    original_dht_step = DHTTrainer.step
    log_path = output.parent / f"{output.name}.live.jsonl"

    with log_path.open("x") as log:

        def emit(event: dict) -> None:
            event = {"run": output.name, **event}
            line = json.dumps(event, allow_nan=False)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)

        def observe_loss(*args, **kwargs):
            result = original_loss(*args, **kwargs)
            value = float(result.detach().mean())
            if not math.isfinite(value):
                raise FloatingPointError("observed nonfinite task loss")
            losses.append(value)
            return result

        def observe_optimizer(optimizer, *args, **kwargs):
            nonlocal optimizer_steps, max_gradient
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        if not torch.isfinite(parameter.grad).all():
                            raise FloatingPointError("nonfinite task gradient")
                        max_gradient = max(max_gradient, float(parameter.grad.abs().max()))
            result = original_optimizer_step(optimizer, *args, **kwargs)
            optimizer_steps += 1
            for group in optimizer.param_groups:
                if any(not torch.isfinite(p).all() for p in group["params"]):
                    raise FloatingPointError("nonfinite model parameter")
            if config.dht is None and (optimizer_steps % log_every == 0 or optimizer_steps == 1):
                emit(
                    {
                        "step": optimizer_steps,
                        "recent_mean_loss": statistics.mean(losses[-log_every:]),
                        "max_abs_gradient_so_far": max_gradient,
                    }
                )
            return result

        def observe_dht(trainer):
            nonlocal trained_samples
            row = original_dht_step(trainer)
            trained_samples += row["batch_size"]
            controller = trainer.controller
            pending = sum(len(q) for q in trainer.queues.pending.values())
            if trainer.routed_samples != trained_samples + pending:
                raise AssertionError("routed samples do not equal trained plus pending")
            for bandit in [controller.route_ucb, controller.head_ucb]:
                if bandit is not None:
                    if not torch.isfinite(bandit.A).all() or not torch.isfinite(bandit.b).all():
                        raise FloatingPointError("nonfinite UCB statistics")
            if not torch.isfinite(controller.ts.mean).all() or (
                not torch.isfinite(controller.ts.precision).all()
                or (controller.ts.precision <= 0).any()
            ):
                raise FloatingPointError("invalid logistic TS posterior")
            if any(not fit["converged"] for fit in controller.ts.last_fit.values()):
                raise AssertionError("unconverged TS posterior update")
            events.append(row)
            if row["step"] % log_every == 0 or row["step"] == 1:
                emit(
                    {
                        "step": row["step"],
                        "epoch": row["epoch"],
                        "recent_mean_loss": statistics.mean(losses[-log_every:]),
                        "batch_depths": row["selections"]["depth_counts"],
                        "route_observations": controller.route_ucb.counts.tolist(),
                        "head_observations": controller.head_ucb.counts.tolist()
                        if controller.head_ucb
                        else None,
                        "ts_observations": controller.ts.counts.tolist(),
                        "reward_clips": controller.rewards.clipped,
                        "pending": pending,
                        "max_abs_gradient_so_far": max_gradient,
                    }
                )
            return row

        emit({"event": "start", "config": config.model_dump(mode="json")})
        try:
            with (
                patch.object(torch.optim.AdamW, "step", observe_optimizer),
                patch.object(F, "cross_entropy", observe_loss),
                patch.object(DHTTrainer, "step", observe_dht),
            ):
                summary = train(config, output)
            audit = {
                "status": "completed",
                "optimizer_steps": optimizer_steps,
                "finite_losses_gradients_parameters": True,
                "max_abs_gradient": max_gradient,
                "first_25_mean_loss": statistics.mean(losses[:25]),
                "last_25_mean_loss": statistics.mean(losses[-25:]),
                "training_loss_decreased": statistics.mean(losses[-25:])
                < statistics.mean(losses[:25]),
                "queue_conservation_and_finite_converged_bandits": True if config.dht else None,
                "losses": losses,
                "summary": summary,
            }
            if events:
                audit["first_25_depth_counts"] = combine_depths(events[:25])
                audit["last_25_depth_counts"] = combine_depths(events[-25:])
            emit(
                {
                    "event": "completed",
                    "first_25_loss": audit["first_25_mean_loss"],
                    "last_25_loss": audit["last_25_mean_loss"],
                    "validation": summary["validation"],
                    "test": summary["test"],
                }
            )
        except Exception as error:
            audit = {
                "status": "failed",
                "optimizer_steps": optimizer_steps,
                "losses": losses,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "last_completed_dht_step": events[-1] if events else None,
            }
            emit({"event": "failed", "optimizer_steps": optimizer_steps, "error": repr(error)})
        (output.parent / f"{output.name}.audit.json").write_text(
            json.dumps(audit, indent=2, allow_nan=False) + "\n"
        )
        return audit


def combine_depths(events: list[dict]) -> dict:
    """Count observed sample depths over a training window."""
    counts = {}
    for row in events:
        for depth, count in row["selections"]["depth_counts"].items():
            counts[depth] = counts.get(depth, 0) + count
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["fixed", "prefix", "tree", "integrated"],
        default=["fixed", "tree", "integrated"],
        help="prefix is a diagnostic fixed-one-layer control, not a paper variant",
    )
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error("log-every must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for mode in args.modes:
        source = Path(f"configs/experiment/audit_{mode}.toml")
        for seed in args.seeds:
            settings = load_config(source).model_dump(mode="json")
            settings.update(seed=seed, name=f"audit_{mode}_seed{seed}")
            settings["bandit"]["seed"] = seed + 1000
            config = ExperimentConfig.model_validate(settings)
            result = run_audit(config, args.output / f"{mode}-seed{seed}", args.log_every)
            records.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "config_source": str(source),
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "status": result["status"],
                    "optimizer_steps": result["optimizer_steps"],
                }
            )
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    if any(record["status"] != "completed" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
