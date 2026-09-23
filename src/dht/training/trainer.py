"""End-to-end DHT training, written in the order of Figure 1 (printed p.445).

This module intentionally uses a small explicit loop rather than a framework.
Reading order: DHTTrainer.__init__ -> next_batch -> step -> state_dict.
The separate train_dynamic function handles local files and CLI orchestration.
"""

import csv
import json
import time
from collections import Counter
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from dht.bandits.features import FeatureBank
from dht.config import ExperimentConfig
from dht.data import load_split
from dht.evaluation.flops import forward_matmul_flops
from dht.evaluation.metrics import classification_metrics
from dht.models.architecture import ArchitectureSpec
from dht.models.pretrained import build_model
from dht.models.transformer import TransformerClassifier
from dht.policies.controller import DHTController
from dht.runtime import metadata, seed_everything, select_device, synchronize
from dht.training.calibration import calibration_indices
from dht.training.checkpointing import read_checkpoint, restore_checkpoint, save_checkpoint
from dht.training.queues import DepthQueues


def selection_summary(config: ExperimentConfig, specs: list[ArchitectureSpec]) -> dict:
    """Summarize observed architectures; count per-layer heads including full prefix.

    Head frequencies are keyed by dynamic group start. Forward FLOPs below include
    the task path; routing is reported separately so its duplicated prefix is visible.
    """
    depth_counts, count_counts, head_frequency = Counter(), Counter(), {}
    flops, head_total, layer_total = 0, 0, 0
    for spec in specs:
        depth_counts[str(spec.total_depth)] += 1
        for layer in range(spec.total_depth):
            heads = spec.heads_for(layer, config.model.num_heads)
            head_total += len(heads)
            layer_total += 1
            if (
                layer >= config.model.prefix_depth
                and (layer - config.model.prefix_depth) % config.model.group_size == 0
            ):
                count_counts[str(len(heads))] += 1
                group = head_frequency.setdefault(str(layer), [0] * config.model.num_heads)
                for head in heads:
                    group[head] += 1
        flops += forward_matmul_flops(config.model, spec, config.data.sequence_length)
    count = len(specs)
    full = forward_matmul_flops(
        config.model, ArchitectureSpec(config.model.total_depth), config.data.sequence_length
    )
    return {
        "samples": count,
        "depth_counts": dict(depth_counts),
        "dynamic_group_head_count_counts": dict(count_counts),
        "head_selection_frequencies_by_group": head_frequency,
        "average_total_depth": sum(s.total_depth for s in specs) / count if count else None,
        "average_heads_per_executed_layer": head_total / layer_total if layer_total else None,
        "task_forward_matmul_flops": flops,
        "relative_task_forward_matmul_flops": flops / (count * full) if count else None,
    }


def prefix_matmul_flops(config: ExperimentConfig) -> int:
    """Routing/calibration prefix excludes the classifier, which is not executed."""
    return (
        forward_matmul_flops(
            config.model, ArchitectureSpec(config.model.prefix_depth), config.data.sequence_length
        )
        - 2 * config.model.hidden_size * config.model.num_classes
    )


def restore_controller(config: ExperimentConfig, state: dict) -> DHTController:
    """Build a policy directly from saved calibration; never use evaluation data."""
    if state.get("scope") != "dht" or state.get("training_state") is None:
        raise ValueError("a DHT checkpoint is required")
    features = FeatureBank(config)
    features.load_state_dict(state["training_state"]["controller"]["features"])
    controller = DHTController(config, features)
    controller.load_state_dict(state["training_state"]["controller"])
    return controller


@torch.no_grad()
def evaluate_dynamic(
    model: TransformerClassifier,
    controller: DHTController,
    config: ExperimentConfig,
    device: torch.device,
    split: str,
) -> dict:
    """Report classification and architecture choices with completely frozen policies."""
    tokens, labels = load_split(config, split)
    predicted, specs = [], []
    synchronize(device)
    started = time.perf_counter()
    for start in range(0, len(tokens), config.batch_size):
        logits, architectures = controller.predict(
            model, tokens[start : start + config.batch_size].to(device)
        )
        predicted.append(logits.argmax(-1).cpu())
        specs.extend(architectures)
    synchronize(device)
    seconds = time.perf_counter() - started
    return {
        **classification_metrics(torch.cat(predicted), labels, config.model.num_classes),
        "architectures": selection_summary(config, specs),
        # Batched inference reuses the routing prefix for execution (D017), so the
        # separate routing pass costs nothing extra; per-sample reruns it.
        "routing_forward_matmul_flops": 0
        if config.dht.execution == "batched"
        else len(tokens) * prefix_matmul_flops(config),
        "execution": config.dht.execution,
        "end_to_end_seconds": seconds,
        "samples_per_second": len(tokens) / seconds,
        "timing_scope": "routing + per-sample policy + task forward + transfers + predictions",
    }


class DHTTrainer:
    """State machine for depth queues, optimizer steps and observed bandit feedback.

    steps counts completed optimizer updates, not epochs. cursor counts routed
    training samples in the current epoch. Pending queues survive checkpoint/resume.
    At the end of the dataset, queues are flushed before a new epoch starts.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: TransformerClassifier,
        optimizer: torch.optim.Optimizer,
        tokens: Tensor,
        labels: Tensor,
        device: torch.device,
        saved: dict | None = None,
    ):
        self.config, self.model, self.optimizer = config, model, optimizer
        self.tokens, self.labels, self.device = tokens, labels, device
        features = FeatureBank(config)
        self.calibration_seconds = 0.0
        self.calibration_flops = 0
        if saved is None:
            indices, self.calibration_selection = calibration_indices(labels, config)
            synchronize(device)
            started = time.perf_counter()
            features.calibrate(
                model,
                tokens[indices],
                device,
                progress=(
                    lambda done, total: print(
                        json.dumps({"stage": "calibration", "completed": done, "total": total}),
                        flush=True,
                    )
                )
                if config.training
                else None,
            )
            synchronize(device)
            self.calibration_seconds = time.perf_counter() - started
            # Calibration reuses prefix activations between count-specific paths.
            prefix_cost = prefix_matmul_flops(config)
            cost = prefix_cost
            full_heads = tuple(range(config.model.num_heads))
            for count in sorted(set(features.counts) | {config.model.num_heads}):
                # Identities do not affect this architecture operation estimate.
                chosen = full_heads[:count]
                spec = ArchitectureSpec(
                    config.model.total_depth,
                    {
                        layer: chosen
                        for layer in range(config.model.prefix_depth, config.model.total_depth)
                    },
                )
                cost += forward_matmul_flops(config.model, spec, config.data.sequence_length) - (
                    prefix_cost + 2 * config.model.hidden_size * config.model.num_classes
                )
            self.calibration_flops = len(indices) * cost
        else:
            self.calibration_selection = saved.get(
                "calibration_selection", {"method": "legacy_unrecorded"}
            )
            features.load_state_dict(saved["controller"]["features"])
        self.controller = DHTController(config, features)
        self.queues = DepthQueues(features.depths, config.batch_size)
        self.cursor, self.epoch, self.completed_steps = 0, 0, 0
        self.routed_samples = 0
        self.routing_seconds = 0.0
        if saved is not None:
            self.controller.load_state_dict(saved["controller"])
            self.queues.load_state_dict(saved["queues"])
            for name in (
                "cursor",
                "epoch",
                "completed_steps",
                "routed_samples",
                "routing_seconds",
                "calibration_seconds",
                "calibration_flops",
            ):
                setattr(self, name, saved[name])
            if not 0 <= self.cursor <= len(tokens):
                raise ValueError("checkpoint data cursor exceeds training dataset")

    def next_batch(self) -> list:
        """Route samples until a depth queue is ready, or flush a partial epoch tail."""
        while True:
            ready = self.queues.pop(flush=self.cursor == len(self.tokens))
            if ready:
                return ready
            if self.cursor == len(self.tokens):
                # All tails have now trained: start a new pass in fixed dataset order.
                self.cursor = 0
                self.epoch += 1
            synchronize(self.device)
            started = time.perf_counter()
            chunk = min(self.config.dht.routing_batch_size, len(self.tokens) - self.cursor)
            if chunk == 1:
                routes = [
                    self.controller.route(
                        self.model,
                        self.tokens[self.cursor : self.cursor + 1].to(self.device),
                        self.cursor,
                    )
                ]
            else:  # D017: one no-grad prefix pass routes the whole chunk.
                routes = self.controller.route_batch(
                    self.model,
                    self.tokens[self.cursor : self.cursor + chunk].to(self.device),
                    list(range(self.cursor, self.cursor + chunk)),
                )
            synchronize(self.device)
            self.routing_seconds += time.perf_counter() - started
            for route in routes:
                self.queues.push(route)
            self.cursor += chunk
            self.routed_samples += chunk

    def step(self) -> dict:
        """One queue release -> sample-specific forwards -> mean task loss -> updates.

        No per-sample optimizer steps: the shared model takes ONE mean-loss gradient
        step for the entire released depth batch, even when masks differ within it.
        """
        routing_before = self.routing_seconds
        batch = self.next_batch()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        synchronize(self.device)
        started = time.perf_counter()
        if self.config.dht.execution == "batched":
            ids = [item.sample_id for item in batch]
            executions = self.controller.execute_batch(
                self.model, self.tokens[ids].to(self.device), batch
            )
        else:
            executions = [
                self.controller.execute(
                    self.model,
                    self.tokens[item.sample_id : item.sample_id + 1].to(self.device),
                    item,
                )
                for item in batch
            ]
        logits = torch.cat([execution.logits for execution in executions])
        targets = self.labels[[item.sample_id for item in batch]].to(self.device)
        individual_losses = F.cross_entropy(logits, targets, reduction="none")
        if not torch.isfinite(individual_losses).all():
            raise FloatingPointError("nonfinite DHT task loss")
        observed_losses = individual_losses.detach().cpu().tolist()
        synchronize(self.device)
        forward_seconds = time.perf_counter() - started
        started = time.perf_counter()
        individual_losses.mean().backward()
        self.optimizer.step()
        synchronize(self.device)
        optimizer_seconds = time.perf_counter() - started
        started = time.perf_counter()
        rewards = self.controller.update(executions, observed_losses)
        controller_update_seconds = time.perf_counter() - started
        self.completed_steps += 1
        # Traces contain no logits/graphs or large feature tensors. Full decision
        # contexts live in bandit statistics or pending queues, not metric logs.
        return {
            "step": self.completed_steps,
            "epoch": self.epoch,
            "loss": sum(observed_losses) / len(observed_losses),
            "batch_size": len(batch),
            "routing_seconds": self.routing_seconds - routing_before,
            "forward_including_head_policy_seconds": forward_seconds,
            "backward_optimizer_seconds": optimizer_seconds,
            "controller_update_seconds": controller_update_seconds,
            "selections": selection_summary(
                self.config, [item.architecture for item in executions]
            ),
            "samples": [
                {
                    "sample_id": item.route.sample_id,
                    "loss": loss,
                    "architecture": item.architecture.to_dict(),
                    "rewards": reward,
                }
                for item, loss, reward in zip(executions, observed_losses, rewards, strict=True)
            ],
        }

    def state_dict(self) -> dict:
        """Save everything needed to resume at this completed optimizer boundary."""
        return {
            "controller": self.controller.state_dict(),
            "calibration_selection": self.calibration_selection,
            "queues": self.queues.state_dict(),
            **{
                name: getattr(self, name)
                for name in (
                    "cursor",
                    "epoch",
                    "completed_steps",
                    "routed_samples",
                    "routing_seconds",
                    "calibration_seconds",
                    "calibration_flops",
                )
            },
        }


def train_dynamic(
    config: ExperimentConfig,
    output: Path,
    resume: Path | None = None,
    stop_after: int | None = None,
) -> dict:
    """Run a bounded local experiment and save metrics, provenance and full state.

    Synthetic or explicitly prepared AG News data use the SAME controller code.
    No data/model downloads occur implicitly. See the real-data protocol for the
    choices (initialization, text preprocessing, split, budget) absent from DHT.
    """
    target = config.steps if stop_after is None else stop_after
    if not 1 <= target <= config.steps:
        raise ValueError("stop_after must be in [1, configured steps]")
    state = read_checkpoint(resume) if resume else None
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.cpu_threads)
    streams = seed_everything(config.seed, config.bandit.seed)
    device = select_device(config.device, config.allow_device_fallback)
    model = build_model(config, initialize=state is None).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    tokens, labels = load_split(config, "train")
    if state is not None and state["scope"] != "dht":
        raise ValueError("cannot resume a fixed checkpoint as DHT")
    trainer = DHTTrainer(
        config,
        model,
        optimizer,
        tokens,
        labels,
        device,
        state["training_state"] if state else None,
    )
    if state:
        step = restore_checkpoint(state, model, optimizer, config, streams)
        if step != trainer.completed_steps:
            raise ValueError("checkpoint trainer/model step mismatch")
    if target <= trainer.completed_steps:
        raise ValueError("resume target must exceed checkpoint step")

    def write(name: str, value: dict | list) -> None:
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    provenance = metadata(config, device)
    provenance["resume_from"] = str(resume.resolve()) if resume else None
    write("metadata.json", provenance)
    write("config.json", config.model_dump(mode="json"))
    start_step = trainer.completed_steps
    started = time.perf_counter()
    rows = []
    while trainer.completed_steps < target:
        rows.append(trainer.step())
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(rows[-1], allow_nan=False) + "\n")
        if trainer.completed_steps % config.log_every == 0 or len(rows) == 1:
            print(
                json.dumps(
                    {key: rows[-1][key] for key in ("step", "loss", "batch_size", "selections")}
                ),
                flush=True,
            )
    synchronize(device)
    segment_seconds = time.perf_counter() - started
    save_checkpoint(
        output / "checkpoint.pt", model, optimizer, config, target, streams, trainer.state_dict()
    )
    write("metrics.json", rows)
    with (output / "metrics.csv").open("w", newline="") as stream:
        fields = [
            "step",
            "epoch",
            "loss",
            "batch_size",
            "routing_seconds",
            "forward_including_head_policy_seconds",
            "backward_optimizer_seconds",
            "controller_update_seconds",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    specs = [
        ArchitectureSpec(
            sample["architecture"]["total_depth"],
            {int(k): tuple(v) for k, v in sample["architecture"]["head_indices"].items()},
        )
        for row in rows
        for sample in row["samples"]
    ]
    summary = {
        "scope": f"independent_dht_{config.data.name}",
        "mode": config.dht.mode,
        "completed_steps": target,
        "segment_start_step": start_step,
        "complete": target == config.steps,
        "segment_training_seconds": segment_seconds,
        "calibration_seconds": trainer.calibration_seconds,
        "calibration_forward_matmul_flops": trainer.calibration_flops,
        "cumulative_routed_samples": trainer.routed_samples,
        "cumulative_routing_seconds": trainer.routing_seconds,
        "cumulative_routing_forward_matmul_flops": trainer.routed_samples
        * prefix_matmul_flops(config),
        "training_segment_architectures": selection_summary(config, specs),
        "route_bandit_observations": trainer.controller.route_ucb.counts.tolist(),
        "head_count_bandit_observations": trainer.controller.head_ucb.counts.tolist()
        if trainer.controller.head_ucb
        else None,
        "ts_head_observations": trainer.controller.ts.counts.tolist(),
        "clipped_rewards": trainer.controller.rewards.clipped,
        "pending_queues": {str(k): len(v) for k, v in trainer.queues.pending.items()},
        "validation": evaluate_dynamic(model, trainer.controller, config, device, "validation"),
        "test": evaluate_dynamic(model, trainer.controller, config, device, "test")
        if target == config.steps and config.evaluate_test
        else None,
        "checkpoint": str((output / "checkpoint.pt").resolve()),
        "limitations": "Matmul estimates exclude backward/optimizer/controller FLOPs. "
        "Per-sample execution prioritizes correctness; no paper/runtime speedup is claimed.",
    }
    write("summary.json", summary)
    return summary
