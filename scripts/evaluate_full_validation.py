"""Read-only full held-out validation evaluation of a bounded pilot checkpoint.

The pilot's 128-example validation score is noisy. This widens ONLY evaluation to
the already held-out 6000 training-source records, without retraining, recalibration,
checkpoint selection, or public-test access. Both compared models use the same
records. Effective evaluation config and original checkpoint config are both saved.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from dht.cli import evaluate_model
from dht.config import ExperimentConfig
from dht.models.pretrained import build_model
from dht.runtime import metadata, seed_everything, select_device, synchronize
from dht.training.checkpointing import read_checkpoint
from dht.training.trainer import evaluate_dynamic, restore_controller

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    args = parser.parse_args()
    state = read_checkpoint(args.checkpoint)
    original = ExperimentConfig.model_validate(state["config"])
    if original.data.name != "ag_news":
        parser.error("requires a prepared AG News checkpoint")
    manifest = json.loads((Path(original.data.prepared_directory) / "manifest.json").read_text())
    effective = original.model_dump(mode="json")
    effective["data"]["validation_size"] = manifest["splits"]["validation"]["rows"]
    config = ExperimentConfig.model_validate(effective)
    torch.set_num_threads(config.cpu_threads)
    seed_everything(config.seed, config.bandit.seed)
    device = select_device(args.device, allow_fallback=False)
    model = build_model(config, initialize=False).to(device).eval()
    model.load_state_dict(state["model"])
    controller = restore_controller(config, state) if config.dht else None
    synchronize(device)
    started = time.perf_counter()
    metrics = (
        evaluate_dynamic(model, controller, config, device, "validation")
        if controller
        else evaluate_model(model, config, device, "validation")
    )
    synchronize(device)
    result = {
        "scope": "full_validation_only_no_training_or_selection",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_config": original.model_dump(mode="json"),
        "samples": config.data.validation_size,
        "metrics": metrics,
        "evaluation_seconds_including_data_loading": time.perf_counter() - started,
        "metadata": metadata(config, device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"samples": result["samples"], "metrics": metrics}), flush=True)
