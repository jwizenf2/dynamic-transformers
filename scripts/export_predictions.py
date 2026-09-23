"""Export per-sample predictions and frozen DHT decision traces for analysis notebooks.

Post-hoc ANALYSIS of an already validation-selected final checkpoint, not tuning:
nothing is trained, selected or updated. The aggregate macro F1 is recomputed and
checked against the run's recorded ``final_test.json`` so the per-sample table is
provably the same evaluation that produced the reported number (same device).

Fixed models are run in batches. DHT models use ``DHTController.explain`` one
sample at a time: the same frozen route/execute code as ``predict`` plus the UCB
scores and TS head probabilities that caused every choice. Output is JSONL with
one record per sample; ``source_index`` joins back to the raw AG News CSV row.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dht.config import ExperimentConfig
from dht.data import load_split
from dht.evaluation.metrics import classification_metrics
from dht.models.architecture import ArchitectureSpec
from dht.models.pretrained import build_model
from dht.runtime import seed_everything, select_device
from dht.training.checkpointing import read_checkpoint


@torch.no_grad()
def export(checkpoint: Path, split: str, device_name: str, output: Path) -> dict:
    state = read_checkpoint(checkpoint)
    config = ExperimentConfig.model_validate(state["config"])
    seed_everything(config.seed, config.bandit.seed)
    device = select_device(device_name, allow_fallback=False)
    model = build_model(config, initialize=False).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    tokens, labels = load_split(config, split)
    directory = Path(config.data.prepared_directory)
    with np.load(directory / f"{split}.npz", allow_pickle=False) as arrays:
        source_indices = arrays["source_indices"][: len(labels)].tolist()
    lengths = (tokens != 0).sum(-1).tolist()  # [PAD]=0; includes [CLS]/[SEP]

    records: list[dict] = []
    if config.dht is None:
        spec = ArchitectureSpec.from_config(config)
        for start in range(0, len(tokens), config.batch_size):
            batch = tokens[start : start + config.batch_size].to(device)
            logits = model(batch, architecture=spec).float().cpu()
            for offset, row in enumerate(logits):
                records.append({"logits": row, "trace": None, "index": start + offset})
    else:
        from dht.training.trainer import restore_controller

        controller = restore_controller(config, state)
        for index in range(len(tokens)):
            logits, trace = controller.explain(model, tokens[index : index + 1].to(device))
            records.append({"logits": logits[0].float().cpu(), "trace": trace, "index": index})

    predicted = torch.stack([r["logits"] for r in records]).argmax(-1)
    metrics = classification_metrics(predicted, labels, config.model.num_classes)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        for record in records:
            index = record["index"]
            logits = record["logits"]
            probabilities = logits.softmax(-1)
            row = {
                "index": index,
                "source_index": int(source_indices[index]),
                "label": int(labels[index]),
                "prediction": int(logits.argmax()),
                "probabilities": probabilities.tolist(),
                "loss": float(F.cross_entropy(logits[None], labels[index : index + 1])),
                "token_length": int(lengths[index]),
                "trace": record["trace"],
            }
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return {
        "checkpoint": str(checkpoint),
        "split": split,
        "device": str(device),
        "samples": len(records),
        "mode": config.dht.mode if config.dht else "fixed",
        "metrics": metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expect",
        type=Path,
        help="recorded evaluation JSON whose macro F1 must be reproduced exactly",
    )
    args = parser.parse_args()
    summary = export(args.checkpoint, args.split, args.device, args.output)
    if args.expect:
        recorded = json.loads(args.expect.read_text())["metrics"]["macro_f1"]
        summary["recorded_macro_f1"] = recorded
        summary["matches_recorded"] = recorded == summary["metrics"]["macro_f1"]
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
