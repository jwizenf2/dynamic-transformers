"""Validate completed checkpoints/logs and curate a paper-separated pilot record.

These checks are POST-RUN checks, not a claim that every gradient was inspected
during training. The unit/GPU tests establish gradient correctness separately.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from dht.ag_news import sha256
from dht.config import ExperimentConfig
from dht.models.architecture import ArchitectureSpec
from dht.training.checkpointing import read_checkpoint


def require_finite(value, location="checkpoint") -> int:
    """Walk saved model/optimizer/controller tensors, reporting which field failed."""
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite tensor at {location}")
        return 1
    if isinstance(value, dict):
        return sum(require_finite(item, f"{location}.{key}") for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return sum(require_finite(item, f"{location}.{i}") for i, item in enumerate(value))
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite scalar at {location}")
    return 0


def collect(directory: Path) -> dict:
    """Check logs match the checkpoint and returned architectures remain valid."""

    def read(name):
        return json.loads((directory / name).read_text())

    summary = read("summary.json")
    state = read_checkpoint(directory / "checkpoint.pt")
    config = ExperimentConfig.model_validate(state["config"])
    rows = read("metrics.json")
    live = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    assert rows == live
    assert summary["complete"] and summary["completed_steps"] == state["step"] == len(rows)
    assert summary["test"] is None and not config.evaluate_test
    tensor_count = sum(
        require_finite(state[key], key) for key in ("model", "optimizer", "training_state")
    )
    assert all(math.isfinite(row["loss"]) for row in rows)
    checks = {
        "finite_saved_tensors_checked": tensor_count,
        "live_logs_match_final_metrics": True,
        "first_16_mean_loss": sum(row["loss"] for row in rows[:16]) / 16,
        "last_16_mean_loss": sum(row["loss"] for row in rows[-16:]) / 16,
    }
    if config.dht:
        training = state["training_state"]
        ts = training["controller"]["ts"]
        assert (ts["precision"] > 0).all()
        assert all(fit["converged"] for fit in ts["last_fit"].values())
        trained = sum(row["batch_size"] for row in rows)
        pending = sum(len(queue) for queue in training["queues"]["pending"].values())
        assert training["routed_samples"] == trained + pending
        for row in rows:
            for sample in row["samples"]:
                spec = sample["architecture"]
                ArchitectureSpec(
                    spec["total_depth"], {int(k): tuple(v) for k, v in spec["head_indices"].items()}
                ).validate(config.model)
        checks.update(
            {
                "trained_samples": trained,
                "pending_samples": pending,
                "routed_samples": training["routed_samples"],
                "queue_conservation": True,
                "last_ts_fits_converged": True,
                "all_logged_architectures_valid": True,
                "training_stage_seconds": {
                    key: sum(row[key] for row in rows)
                    for key in (
                        "routing_seconds",
                        "forward_including_head_policy_seconds",
                        "backward_optimizer_seconds",
                        "controller_update_seconds",
                    )
                },
            }
        )
    else:
        checks["trained_samples"] = len(rows) * config.batch_size
    benchmark = read("benchmark.json")
    return {
        "directory": str(directory),
        "summary": summary,
        "checks": checks,
        "training_metadata": read("metadata.json"),
        "benchmark": {key: value for key, value in benchmark.items() if key != "metadata"},
        "full_validation": read("full_validation.json"),
        "checkpoint_sha256": sha256(directory / "checkpoint.pt"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    runs = {"fixed": collect(args.fixed), "tree": collect(args.tree)}
    fixed, tree = [runs[key]["training_metadata"]["config"] for key in ("fixed", "tree")]
    assert {key: value for key, value in fixed.items() if key not in {"name", "dht"}} == {
        key: value for key, value in tree.items() if key not in {"name", "dht"}
    }
    with Path("references/paper_results.csv").open() as stream:
        authors = [
            row
            for row in csv.DictReader(stream)
            if row["dataset"] == "ag_news" and row["model"] in ("12-VaBERT", "T-DHT")
        ]
    assert len(authors) == 4  # two models × the two original paper tables
    result = {
        "provenance": "our_measured_pilot_not_full_reproduction",
        "runs": runs,
        "matched_config_verified": True,
        "collector_sha256": sha256(Path(__file__)),
        "authors_reported_separate": authors,
        "physical_execution": json.loads((args.tree / "execution.json").read_text()),
        "physical_execution_pruned_policy": json.loads(
            (args.tree / "execution-pruned.json").read_text()
        ),
        "limitations": [
            "One training seed; only 1024 training records, 128 updates.",
            "Validation is not the public test; paper F1 averaging unspecified.",
            "DHT trained 1018 records with six pending, versus fixed 1024.",
            "Timing snapshots are not repeated-run speedup evidence.",
            "Forward matrix counts exclude non-matmul and optimizer work.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({name: run["checks"] for name, run in runs.items()}, indent=2))
