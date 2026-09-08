"""Small epoch-based fixed/DHT runner for the main reproduction (decision D013).

This is orchestration, not another DHT implementation: DHTTrainer still owns
routing, queues, task gradients and observed bandit updates. Legacy step-budget
smoke commands remain unchanged. This runner adds equal corpus exposure, a shared
sample-based LR schedule, streaming logs, and resumable latest/best checkpoints.
"""

import csv
import json
import shutil
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from dht.ag_news import sha256
from dht.config import ExperimentConfig
from dht.data import load_split
from dht.evaluation.flops import forward_matmul_flops
from dht.models.architecture import ArchitectureSpec
from dht.models.pretrained import build_model
from dht.runtime import metadata, seed_everything, select_device, synchronize
from dht.training.checkpointing import read_checkpoint, restore_checkpoint, save_checkpoint
from dht.training.trainer import (
    DHTTrainer,
    evaluate_dynamic,
    prefix_matmul_flops,
    restore_controller,
    selection_summary,
)


def scheduled_learning_rate(config: ExperimentConfig, trained_samples: int) -> float:
    """LR for the next update, indexed by the NEXT sample's 1-based exposure.

    Use actual trained samples, not routing calls or queue releases. Both models
    have the same schedule as a function of corpus exposure even if their batches
    differ. Rate is constant within one update. The first rate is positive during
    warmup; zero warmup starts near the peak. This exact convention is our choice,
    not an inferred formula from §7.3 p.449. State is the saved sample counter, so
    resume cannot accidentally restart a scheduler or use a different step horizon.
    """
    options = config.training
    if options is None or trained_samples < 0:
        raise ValueError("nonnegative exposure and epoch configuration required")
    if options.schedule == "constant":
        return config.learning_rate
    budget = options.epochs * config.data.train_size
    fraction = min((trained_samples + 1) / budget, 1.0)
    warmup = options.warmup_fraction
    multiplier = fraction / warmup if fraction < warmup else (1 - fraction) / (1 - warmup)
    return config.learning_rate * multiplier


def _merge_selection(total: dict, selected: dict) -> None:
    """Bounded-memory cumulative counters, rather than retaining every sample graph/log."""
    total["samples"] += selected["samples"]
    total["task_forward_matmul_flops"] += selected["task_forward_matmul_flops"]
    layers = sum(int(depth) * count for depth, count in selected["depth_counts"].items())
    total["layer_total"] += layers
    total["head_total"] += round(selected["average_heads_per_executed_layer"] * layers)
    for key in ("depth_counts", "dynamic_group_head_count_counts"):
        for action, count in selected[key].items():
            total[key][action] = total[key].get(action, 0) + count
    for group, counts in selected["head_selection_frequencies_by_group"].items():
        previous = total["head_selection_frequencies_by_group"].setdefault(group, [0] * len(counts))
        total["head_selection_frequencies_by_group"][group] = [
            a + b for a, b in zip(previous, counts, strict=True)
        ]


def _publish_best(directory: Path, candidate: Path) -> None:
    """Atomically point best.pt at an immutable candidate, without duplicating GBs.

    A hard link uses the same underlying file within this run directory. Candidates
    are never overwritten. Therefore a crash while publishing a NEW best cannot
    invalidate the older best file referenced by the last periodic checkpoint.
    Keep candidates (at most one per epoch) until the experiment is archived.
    """
    temporary = directory / ".best-link.tmp"
    temporary.hardlink_to(candidate)
    temporary.replace(directory / "best.pt")


def train_epochs(
    config: ExperimentConfig,
    output: Path,
    resume: Path | None = None,
    stop_after: int | None = None,
) -> dict:
    """Train complete epochs, or stop safely at an absolute optimizer-step boundary.

    `checkpoint.pt` is the latest RESUMABLE state; `best.pt` contains the complete
    state selected by epoch-end validation (strictly larger score; earliest tie).
    Resume writes a new directory and carries forward its best checkpoint; previous
    runs are never overwritten. Public-test evaluation is opt-in and uses BEST, not
    last. --stop-after saves at a completed update boundary. An arbitrary Ctrl-C
    leaves the previous atomic periodic checkpoint intact, never serializes a
    partially mutated model/policy pair as if it were a completed update.
    """
    from dht.cli import evaluate_model

    if config.training is None or (stop_after is not None and stop_after < 1):
        raise ValueError("epoch training requires [training] and a positive stop boundary")
    state = read_checkpoint(resume) if resume else None
    if state:
        if ExperimentConfig.model_validate(state["config"]) != config or not state.get("run_state"):
            raise ValueError("epoch resume requires the same config and an epoch-run checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.cpu_threads)
    streams = seed_everything(config.seed, config.bandit.seed)
    device = select_device(config.device, config.allow_device_fallback)
    model = build_model(config, initialize=state is None).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    tokens, labels = load_split(config, "train")
    trainer = (
        DHTTrainer(
            config,
            model,
            optimizer,
            tokens,
            labels,
            device,
            state["training_state"] if state else None,
        )
        if config.dht
        else None
    )
    progress = (
        state["run_state"]
        if state
        else {
            "completed_epochs": 0,
            "completed_steps": 0,
            "trained_samples": 0,
            "fixed_cursor": 0,
            "best_score": None,
            "best_step": None,
            "best_checkpoint": None,
            "best_checkpoint_sha256": None,
            "validation_history": [],
            "training_step_seconds": 0.0,
            "selections": {
                "samples": 0,
                "layer_total": 0,
                "head_total": 0,
                "task_forward_matmul_flops": 0,
                "depth_counts": {},
                "dynamic_group_head_count_counts": {},
                "head_selection_frequencies_by_group": {},
            },
        }
    )
    if state:
        step = restore_checkpoint(state, model, optimizer, config, streams)
        if step != progress["completed_steps"] or (trainer and trainer.completed_steps != step):
            raise ValueError("checkpoint step counters disagree")
        if progress["best_step"] is not None:
            # Resuming BEST itself must use this file, not a possibly superseded
            # best-path pointer embedded inside that older checkpoint snapshot.
            candidate = (
                resume if step == progress["best_step"] else Path(progress["best_checkpoint"])
            )
            if candidate != resume and sha256(candidate) != progress["best_checkpoint_sha256"]:
                raise ValueError("saved best-checkpoint artifact changed or is missing")
            carried = output / f"best-step-{progress['best_step']:08d}.pt"
            shutil.copyfile(candidate, carried)
            _publish_best(output, carried)
            progress["best_checkpoint"] = str(carried.resolve())
            progress["best_checkpoint_sha256"] = sha256(carried)
        del state  # Release CPU copies of large model/optimizer tensors before training.
    if progress["completed_epochs"] >= config.training.epochs or (
        stop_after is not None and stop_after <= progress["completed_steps"]
    ):
        raise ValueError("resume must request additional training")

    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    provenance = metadata(config, device)
    provenance["resume_from"] = str(resume.resolve()) if resume else None
    write("metadata.json", provenance)
    write("config.json", config.model_dump(mode="json"))
    if trainer:
        write("calibration.json", trainer.calibration_selection)

    def save(path):
        save_checkpoint(
            path,
            model,
            optimizer,
            config,
            progress["completed_steps"],
            streams,
            trainer.state_dict() if trainer else None,
            progress,
        )

    def evaluate(split):
        return (
            evaluate_dynamic(model, trainer.controller, config, device, split)
            if trainer
            else evaluate_model(model, config, device, split)
        )

    # Initial safe checkpoint includes calibrated descriptors; no repeat calibration
    # is needed after a later interruption, even before the first periodic interval.
    save(output / "checkpoint.pt")
    start_step = progress["completed_steps"]
    fixed_spec = ArchitectureSpec.from_config(config)
    fields = ["step", "epoch", "batch_size", "loss", "learning_rate", "trained_samples"]
    with (
        (output / "metrics.jsonl").open("a") as log,
        (output / "metrics.csv").open("a", newline="") as csv_file,
    ):
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        while progress["completed_epochs"] < config.training.epochs:
            if stop_after is not None and progress["completed_steps"] >= stop_after:
                break
            rate = scheduled_learning_rate(config, progress["trained_samples"])
            for group in optimizer.param_groups:
                group["lr"] = rate
            synchronize(device)
            started = time.perf_counter()
            if trainer:
                row = trainer.step()
                epoch_finished = trainer.cursor == len(tokens) and not any(
                    trainer.queues.pending.values()
                )
            else:
                start = progress["fixed_cursor"]
                end = min(start + config.batch_size, len(tokens))
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(
                    model(tokens[start:end].to(device), architecture=fixed_spec),
                    labels[start:end].to(device),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite fixed task loss")
                loss.backward()
                optimizer.step()
                progress["fixed_cursor"] = end
                row = {
                    "loss": loss.detach().item(),
                    "batch_size": end - start,
                    "sample_ids": list(range(start, end)),
                    "selections": selection_summary(config, [fixed_spec] * (end - start)),
                }
                epoch_finished = end == len(tokens)
            synchronize(device)
            progress["training_step_seconds"] += time.perf_counter() - started
            progress["trained_samples"] += row["batch_size"]
            progress["completed_steps"] += 1
            row.update(
                {
                    "step": progress["completed_steps"],
                    "epoch": progress["completed_epochs"] + 1,
                    "trained_samples": progress["trained_samples"],
                    "learning_rate": rate,
                }
            )
            _merge_selection(progress["selections"], row["selections"])
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            writer.writerow(row)
            csv_file.flush()
            if row["step"] % config.log_every == 0 or row["step"] == start_step + 1:
                print(json.dumps({key: row[key] for key in fields}), flush=True)
            if epoch_finished:
                progress["completed_epochs"] += 1
                progress["fixed_cursor"] = 0
                assert progress["trained_samples"] == progress["completed_epochs"] * len(tokens)
                if trainer:
                    assert trainer.routed_samples == progress["trained_samples"]
                metrics = evaluate("validation")
                score = metrics[config.training.selection_metric]
                record = {
                    "epoch": progress["completed_epochs"],
                    "step": row["step"],
                    "trained_samples": progress["trained_samples"],
                    "metrics": metrics,
                }
                progress["validation_history"].append(record)
                if progress["best_score"] is None or score > progress["best_score"]:
                    candidate = output / f"best-step-{row['step']:08d}.pt"
                    progress.update(
                        {
                            "best_score": score,
                            "best_step": row["step"],
                            "best_checkpoint": str(candidate.resolve()),
                            "best_checkpoint_sha256": None,
                        }
                    )
                    save(candidate)
                    progress["best_checkpoint_sha256"] = sha256(candidate)
                    _publish_best(output, candidate)
                write("validation.json", progress["validation_history"])
                print(
                    json.dumps(
                        {"stage": "validation", **record, "best_score": progress["best_score"]}
                    ),
                    flush=True,
                )
            if epoch_finished or row["step"] % config.training.checkpoint_every_steps == 0:
                save(output / "checkpoint.pt")
    save(output / "checkpoint.pt")
    complete = progress["completed_epochs"] == config.training.epochs
    selected_validation = next(
        (
            row["metrics"]
            for row in progress["validation_history"]
            if row["step"] == progress["best_step"]
        ),
        None,
    )
    test = None
    if complete and config.evaluate_test:
        best = read_checkpoint(output / "best.pt")
        model.load_state_dict(best["model"])
        test = (
            evaluate_dynamic(model, restore_controller(config, best), config, device, "test")
            if trainer
            else evaluate_model(model, config, device, "test")
        )
    selections = dict(progress["selections"])
    count = selections["samples"]
    full_flops = forward_matmul_flops(
        config.model, ArchitectureSpec(config.model.total_depth), config.data.sequence_length
    )
    selections.update(
        {
            "average_total_depth": selections["layer_total"] / count,
            "average_heads_per_executed_layer": selections["head_total"]
            / selections["layer_total"],
            "relative_task_forward_matmul_flops": selections["task_forward_matmul_flops"]
            / (count * full_flops),
        }
    )
    summary = {
        "scope": f"epoch_{'dht' if trainer else 'fixed'}_{config.data.name}",
        "complete": complete,
        "completed_steps": progress["completed_steps"],
        "completed_epochs": progress["completed_epochs"],
        "trained_samples": progress["trained_samples"],
        "target_samples": config.training.epochs * len(tokens),
        "segment_start_step": start_step,
        "cumulative_training_step_seconds": progress["training_step_seconds"],
        "best_step": progress["best_step"],
        "best_validation_score": progress["best_score"],
        "selection_metric": config.training.selection_metric,
        "validation": selected_validation,
        "validation_history": progress["validation_history"],
        "test": test,
        "checkpoint": str((output / "checkpoint.pt").resolve()),
        "best_checkpoint": progress["best_checkpoint"],
        "training_architectures": selections,
    }
    if trainer:
        summary.update(
            {
                "calibration": trainer.calibration_selection,
                "calibration_seconds": trainer.calibration_seconds,
                "calibration_forward_matmul_flops": trainer.calibration_flops,
                "routed_samples": trainer.routed_samples,
                "routing_forward_matmul_flops": trainer.routed_samples
                * prefix_matmul_flops(config),
                "pending_queues": {str(k): len(v) for k, v in trainer.queues.pending.items()},
                "clipped_rewards": trainer.controller.rewards.clipped,
            }
        )
    write("summary.json", summary)
    return summary
