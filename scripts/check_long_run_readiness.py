"""Bounded real-backbone checks before expensive full-data training (D013).

Run fixed, literal-feedback T-DHT, and the named alternative on identical small
data/protocol. Capture a periodic literal checkpoint without touching model/RNG
state, resume it, and compare the final model/optimizer/controller/RNG tensors.
The primary full-data config stays literal regardless of this diagnostic's scores.
No test split is evaluated and no full-data training configuration is launched.
"""

import argparse
import gc
import json
import math
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dht.ag_news import sha256
from dht.cli import train
from dht.config import load_config
from dht.policies.rewards import RewardHistory
from dht.training import epochs
from dht.training.checkpointing import read_checkpoint


def compare_state(left, right, path="state") -> None:
    """Require bitwise CPU-loaded state equality; exclude measured wall-clock fields."""
    if isinstance(left, torch.Tensor):
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise AssertionError(f"nonfinite {path}")
        if not torch.equal(left, right):
            raise AssertionError(f"resume differs at {path}")
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            if not str(key).endswith("seconds") and key != "samples_per_second":
                compare_state(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            compare_state(a, b, f"{path}.{index}")
    elif left != right:
        raise AssertionError(f"resume differs at {path}")


def summarize_logs(directory: Path, rule: str | None) -> dict:
    """Check exactly one visit per record/epoch and tally actual observed TS labels."""
    exposures, binary = {}, Counter()
    minimum, maximum, count = math.inf, -math.inf, 0
    reward = RewardHistory()
    with (directory / "metrics.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            assert math.isfinite(row["loss"])
            minimum, maximum = min(minimum, row["loss"]), max(maximum, row["loss"])
            count += 1
            samples = row.get("samples", [])
            ids = row.get("sample_ids", [sample["sample_id"] for sample in samples])
            exposures.setdefault(row["epoch"], Counter()).update(ids)
            for sample in samples:
                for key, ratio in sample["rewards"].items():
                    if key.startswith("head:") and ratio is not None:
                        binary[str(reward.binary_feedback(ratio, rule))] += 1
    for visits in exposures.values():
        assert visits == Counter(range(1024)), "incomplete/duplicated epoch exposure"
    return {
        "epochs_checked": len(exposures),
        "steps": count,
        "minimum_loss": minimum,
        "maximum_loss": maximum,
        "binary_feedback_group_counts": dict(binary),
        "exact_once_per_epoch": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configs = {
        name: load_config(f"configs/experiment/ag_news_readiness_{name}.toml")
        for name in ("fixed", "literal", "improvement")
    }
    literal, alternative = [
        configs[name].model_dump(mode="json") for name in ("literal", "improvement")
    ]
    literal.pop("name")
    alternative.pop("name")
    literal["dht"].pop("feedback_rule")
    alternative["dht"].pop("feedback_rule")
    assert literal == alternative, "diagnostic must change only the feedback rule"
    results = {
        "scope": "bounded_readiness_not_full_reproduction",
        "script_sha256": sha256(Path(__file__)),
        "runs": {},
    }
    original_save = epochs.save_checkpoint
    snapshot = args.output / "literal/resume-step-00000064.pt"

    def capture(path, *arguments, **kwargs):
        original_save(path, *arguments, **kwargs)
        if path == args.output / "literal/checkpoint.pt" and arguments[3] == 64:
            # Atomic future saves replace checkpoint.pt's inode. This hard link
            # retains the completed step64 artifact without another GB copy.
            snapshot.hardlink_to(path)

    for name, config in configs.items():
        print(json.dumps({"starting": name}), flush=True)
        with patch.object(epochs, "save_checkpoint", capture):
            summary = train(config, args.output / name)
        assert summary["complete"] and summary["trained_samples"] == 2048
        assert summary["test"] is None
        if config.dht:
            assert not any(summary["pending_queues"].values())
            assert summary["calibration"]["class_counts"] == [128] * 4
        results["runs"][name] = {
            "summary": summary,
            "log_checks": summarize_logs(
                args.output / name, config.dht.feedback_rule if config.dht else None
            ),
            "metadata": json.loads((args.output / name / "metadata.json").read_text()),
        }
        (args.output / "readiness.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n"
        )
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    print(json.dumps({"starting": "literal_resume_from_64"}), flush=True)
    resumed = train(configs["literal"], args.output / "resumed", snapshot)
    full_state = read_checkpoint(args.output / "literal/checkpoint.pt")
    resumed_state = read_checkpoint(args.output / "resumed/checkpoint.pt")
    for field in ("model", "optimizer", "training_state", "rng"):
        compare_state(full_state[field], resumed_state[field], field)
    compare_state(
        full_state["run_state"]["validation_history"],
        resumed_state["run_state"]["validation_history"],
        "validation_history",
    )
    assert full_state["run_state"]["best_step"] == resumed_state["run_state"]["best_step"]
    results["resume"] = {"exact_state_match": True, "from_step": 64, "summary": resumed}
    results["primary_feedback_rule_unchanged"] = (
        load_config("configs/experiment/ag_news_tree_full.toml").dht.feedback_rule
        == "paper_literal"
    )
    (args.output / "readiness.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "ready_checks_passed": True,
                "validation_macro_f1": {
                    name: result["summary"]["validation"]["macro_f1"]
                    for name, result in results["runs"].items()
                },
                "exact_mps_resume": True,
            }
        ),
        flush=True,
    )
