"""Verify paper-sized physical skipping and optionally a learned checkpoint trace.

Run on CPU even if training used MPS: PyTorch's CPU profiler exposes the actual
matrix dimensions/FLOP estimates. This is operator verification, not GPU timing.
"""

import argparse
import json
from pathlib import Path

import torch

from dht.config import ExperimentConfig, load_config
from dht.data import load_split
from dht.evaluation.execution import profile_forward, verify_execution
from dht.models.architecture import ArchitectureSpec
from dht.models.pretrained import build_model
from dht.runtime import metadata, seed_everything
from dht.training.checkpointing import read_checkpoint
from dht.training.trainer import restore_controller

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Validation record to trace; use an observed non-prefix path to inspect head skipping",
    )
    parser.add_argument(
        "--find-pruned",
        action="store_true",
        help="Read-only search for an actual frozen-policy pruned-head path",
    )
    args = parser.parse_args()
    state = read_checkpoint(args.checkpoint) if args.checkpoint else None
    config = ExperimentConfig.model_validate(state["config"]) if state else load_config(args.config)
    torch.set_num_threads(config.cpu_threads)
    seed_everything(config.seed, config.bandit.seed)
    model = build_model(config, initialize=state is None).eval()
    if state:
        model.load_state_dict(state["model"])
    tokens, _ = load_split(config, "validation")
    if args.find_pruned:
        if state is None or config.dht is None:
            parser.error("--find-pruned requires a learned DHT checkpoint")
        search_controller = restore_controller(config, state)
        for index in range(len(tokens)):
            # Mean-policy prediction does not fit, update, or sample. These search
            # forwards are outside the profiler and are not a latency measurement.
            _, specs = search_controller.predict(model, tokens[index : index + 1])
            if any(len(heads) < config.model.num_heads for heads in specs[0].head_indices.values()):
                args.sample_index = index
                break
        else:
            raise ValueError("no actual pruned-head path in this validation subset")
    if not 0 <= args.sample_index < len(tokens):
        parser.error("sample index outside the configured validation subset")
    tokens = tokens[args.sample_index : args.sample_index + 1]
    # Paper's 9-layer/8-head candidate, with deliberately noncontiguous head IDs.
    depth = config.model.total_depth - config.model.group_size
    heads = tuple(i for i in range(config.model.num_heads) if i % 3 != 1)
    selected = ArchitectureSpec(depth, {i: heads for i in range(config.model.prefix_depth, depth)})
    result = verify_execution(model, tokens[:1], selected)
    result["validation_sample_index"] = args.sample_index
    if state and config.dht:
        controller = restore_controller(config, state)
        chosen = []

        def predict():
            logits, specs = controller.predict(model, tokens[:1])
            chosen.extend(specs)
            return logits

        result["learned_policy"] = profile_forward(model, predict)
        spec = chosen[0]
        expected_layers = [*range(config.model.prefix_depth), *range(spec.total_depth)]
        assert [row["layer"] for row in result["learned_policy"]["layers"]] == expected_layers
        task_layers = result["learned_policy"]["layers"][config.model.prefix_depth :]
        assert all(
            row["head_indices"] == list(spec.heads_for(row["layer"], config.model.num_heads))
            for row in task_layers
        )
        result["learned_policy"]["selected_architecture"] = spec.to_dict()
    result["metadata"] = metadata(config, torch.device("cpu"))
    result["checkpoint"] = str(args.checkpoint) if args.checkpoint else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sliced_to_full_matmul_ratio": result["sliced_to_full_matmul_ratio"],
                "sliced_to_dense_masked_matmul_ratio": result[
                    "sliced_to_dense_masked_matmul_ratio"
                ],
                "max_logit_difference": result["max_sliced_dense_logit_difference"],
            }
        )
    )
