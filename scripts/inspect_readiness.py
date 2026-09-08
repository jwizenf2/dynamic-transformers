"""Read-only follow-up to a readiness run, including a FAILED bitwise replay check.

Never rewrite a failed check as exact: report every floating-state discrepancy,
require discrete action/sample/RNG equality, and inspect live MPS checkpoint
restoration before ANY training computation. No tolerance is fitted to the observed
weights and no new training or test evaluation is performed. Output is a separate
inspection record, leaving the original readiness result and checkpoints intact.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from dht.ag_news import sha256
from dht.config import load_config
from dht.data import load_split
from dht.models.pretrained import build_model
from dht.runtime import seed_everything
from dht.training.calibration import calibration_indices
from dht.training.checkpointing import read_checkpoint, restore_checkpoint
from dht.training.trainer import restore_controller


def differences(left, right, path="state"):
    """Quantify finite floating differences; discrete values must match exactly."""
    rows = []
    if isinstance(left, torch.Tensor):
        left, right = left.detach().cpu(), right.detach().cpu()
        assert left.shape == right.shape and left.dtype == right.dtype, path
        assert torch.isfinite(left).all() and torch.isfinite(right).all(), path
        if not torch.equal(left, right):
            assert left.is_floating_point(), path
            delta = (left.double() - right.double()).abs()
            rows.append(
                {
                    "path": path,
                    "different_elements": int(torch.count_nonzero(delta)),
                    "max_absolute_difference": float(delta.max()),
                    "mean_absolute_difference": float(delta.mean()),
                }
            )
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            if not str(key).endswith("seconds") and key != "samples_per_second":
                rows.extend(differences(left[key], right[key], f"{path}.{key}"))
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            rows.extend(differences(a, b, f"{path}.{index}"))
    elif left != right:
        assert isinstance(left, float) and isinstance(right, float), path
        rows.append({"path": path, "max_absolute_difference": abs(left - right)})
    return rows


def inspect(directory: Path, mps: bool):
    """Check saved trajectories, finite posteriors, exposure and full-data readiness."""
    torch.set_num_threads(4)
    original = read_checkpoint(directory / "literal/checkpoint.pt")
    resumed = read_checkpoint(directory / "resumed/checkpoint.pt")
    output = {
        "scope": "bounded_readiness_inspection_not_full_reproduction",
        "script_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": {
            name: sha256(directory / name / "checkpoint.pt") for name in ("literal", "resumed")
        },
        "floating_state_differences": {
            field: differences(original[field], resumed[field], field)
            for field in ("model", "optimizer", "training_state", "rng")
        },
    }
    assert output["floating_state_differences"]["rng"] == []
    output["bitwise_resume_match"] = not any(output["floating_state_differences"].values())
    history_delta = differences(
        original["run_state"]["validation_history"],
        resumed["run_state"]["validation_history"],
    )
    assert not history_delta
    assert original["run_state"]["best_step"] == resumed["run_state"]["best_step"]
    output["validation_history_and_best_step_exact"] = True
    del original, resumed
    rows = {
        name: [json.loads(line) for line in (directory / name / "metrics.jsonl").open()]
        for name in ("literal", "resumed")
    }
    source = {row["step"]: row for row in rows["literal"]}
    deltas = []
    for replay in rows["resumed"]:
        deltas.extend(differences(source[replay["step"]], replay, f"step.{replay['step']}"))
    output["log_differences"] = deltas
    output["sample_ids_actions_and_rng_exact"] = True
    output["resume_steps_checked"] = len(rows["resumed"])
    output["runs"] = json.loads((directory / "readiness.json").read_text())["runs"]
    for name in ("literal", "improvement"):
        state = read_checkpoint(directory / name / "checkpoint.pt")
        posterior = state["training_state"]["controller"]["ts"]
        output["runs"][name]["ts_state_keys"] = list(posterior)
        output["runs"][name]["rewards"] = state["training_state"]["controller"]["rewards"]
        # The recursive comparison also rejects nonfinite floating tensors.
        assert differences(state["training_state"], state["training_state"]) == []
        del state
    config = load_config("configs/experiment/ag_news_tree_full.toml")
    tokens, labels = load_split(config, "train")
    indices, selection = calibration_indices(labels, config)
    output["full_data_preflight"] = {
        "training_shape": list(tokens.shape),
        "calibration": selection,
        "unique_calibration_records": len(indices.unique()),
        "primary_feedback_rule": config.dht.feedback_rule,
    }
    assert len(tokens) == 114000 and selection["class_counts"] == [128] * 4
    assert config.dht.feedback_rule == "paper_literal"
    del tokens, labels
    if mps:
        assert torch.backends.mps.is_available(), "MPS required for requested live check"
        config = load_config("configs/experiment/ag_news_readiness_literal.toml")
        snapshot = read_checkpoint(directory / "literal/resume-step-00000064.pt")
        streams = seed_everything(config.seed, config.bandit.seed)
        model = build_model(config, initialize=False).to("mps")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        restore_checkpoint(snapshot, model, optimizer, config, streams)
        assert differences(snapshot["model"], model.state_dict()) == []
        assert differences(snapshot["optimizer"], optimizer.state_dict()) == []
        assert torch.equal(snapshot["rng"]["mps"], torch.mps.get_rng_state())
        controller = restore_controller(config, snapshot)
        assert differences(snapshot["training_state"]["controller"], controller.state_dict()) == []
        output["live_mps_restore_before_computation_exact"] = True
        del snapshot, controller, optimizer
        tokens, _ = load_split(config, "validation")
        logits, specs = [], []
        for name in ("literal", "resumed"):
            state = read_checkpoint(directory / name / "checkpoint.pt")
            model.load_state_dict(state["model"])
            controller = restore_controller(config, state)
            prediction, architectures = controller.predict(model, tokens[:32].to("mps"))
            logits.append(prediction.cpu())
            specs.append(architectures)
            del state, controller
            gc.collect()
        assert specs[0] == specs[1]
        assert torch.equal(logits[0].argmax(-1), logits[1].argmax(-1))
        output["live_mps_validation_probe"] = {
            "samples": 32,
            "architectures_and_predictions_exact": True,
            "max_absolute_logit_difference": float((logits[0] - logits[1]).abs().max()),
        }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mps", action="store_true")
    args = parser.parse_args()
    report = inspect(args.run, args.mps)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key
                not in (
                    "runs",
                    "floating_state_differences",
                    "log_differences",
                    "full_data_preflight",
                )
            }
        ),
        flush=True,
    )
