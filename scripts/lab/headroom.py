"""Phase 0 headroom probe: tool-call accuracy of a frozen Qwen3.5/3.8 model under layer skipping.

Explicit opt-in lab script (reads the HF cache filled by ``setup_check.py``; never trains):

    python scripts/lab/headroom.py --model Qwen/Qwen3.5-0.8B --n 100 --modes full singles

Data: xLAM validation ids from a fixed seeded split (``--val-size`` / ``--split-seed``); the
first ``--n`` validation ids are used. BFCL is NOT used here (it is the final test set).

Modes (each architecture = a set of skipped layers; group 0 always runs):
* ``full``     no skipping.
* ``singles``  for every group g >= 1: skip the whole group, only its attention layer, and only
               its three DeltaNet layers (sub-question 2: which layer type is skippable).
* ``greedy``   static frontier: repeatedly add the whole group whose removal keeps accuracy
               highest (ties -> cheaper, then lower group), up to ``--greedy-steps`` groups.
               Architectures already measured (e.g. by ``singles``) are reused, not rerun.

Requires mlx-lm >= 0.32.0: 0.31.3 corrupted left-padded prompts in batched generation (found
2026-10-13 by tests/unit/test_mlx_routing.py; the first lab run's numbers were discarded).

Writes per-example rows (``examples.jsonl``: architecture, id, correct, reason, output tokens)
and a summary (``summary.json``) under ``--out``; per-example rows are what the later oracle /
held-out probe analysis needs. Generation is greedy (temperature 0), thinking disabled.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from mlx_lm import batch_generate, load

from dht.toolcall import mlx_routing
from dht.toolcall.checker import check_example
from dht.toolcall.data import load_xlam, split_ids
from dht.toolcall.parser import parse_tool_calls
from dht.toolcall.schema import to_openai_tool

XLAM = "Salesforce/xlam-function-calling-60k"
XLAM_REVISION = "26d14ebfe18b1f7b524bd39b404b50af5dc97866"  # setup_manifest.json, 2026-10-13
# Model snapshots recorded by setup_check.py on the lab (2026-10-13). Loading by commit keeps the
# evaluated weights identical to the manifest even if the upstream repo changes.
MODEL_REVISIONS = {
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen/Qwen3.5-27B": "fc05daec18b0a78c049392ed2e771dde82bdf654",
    "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
}
WEIGHT_PATTERNS = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "tokenizer*"]


def validation_examples(n: int, val_size: int, seed: int):
    root = Path(snapshot_download(XLAM, repo_type="dataset", revision=XLAM_REVISION))
    examples = {e.id: e for e in load_xlam(root / "xlam_function_calling_60k.json")}
    _, val_ids = split_ids(list(examples), val_size, seed)
    return [examples[i] for i in val_ids[:n]]


def encode(tokenizer, example) -> list[int]:
    text = tokenizer.apply_chat_template(
        example.messages,
        tools=[to_openai_tool(f) for f in example.functions],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def evaluate(model, tokenizer, examples, prompts, skipped, args, label, rows_file):
    """Run one architecture over all examples; append per-example rows; return a summary."""
    start = time.perf_counter()
    texts: list[str] = []
    token_counts: list[int] = []
    with mlx_routing.skip_layers(model, skipped):
        for i in range(0, len(prompts), args.batch):
            response = batch_generate(
                model,
                tokenizer,
                prompts[i : i + args.batch],
                max_tokens=args.max_tokens,
                return_token_ids=True,
            )
            texts.extend(response.texts)
            token_counts.extend(map(len, response.token_ids))
    seconds = time.perf_counter() - start
    correct = 0
    out_tokens = 0
    reasons: dict[str, int] = {}
    for example, text, n_tok in zip(examples, texts, token_counts, strict=True):
        ok, why = check_example(
            parse_tool_calls(text, example.functions), example.functions, example.ground_truth
        )
        correct += ok
        out_tokens += n_tok
        reasons[why] = reasons.get(why, 0) + 1
        rows_file.write(
            json.dumps(
                {
                    "arch": label,
                    "skipped": sorted(skipped),
                    "id": example.id,
                    "correct": ok,
                    "reason": why,
                    "out_tokens": n_tok,
                    "text": text if not ok else None,
                }
            )
            + "\n"
        )
    rows_file.flush()
    summary = {
        "arch": label,
        "skipped": sorted(skipped),
        "accuracy": correct / len(examples),
        "relative_cost": mlx_routing.relative_cost(model, skipped),
        "seconds": round(seconds, 2),
        "out_tokens": out_tokens,
        "reasons": reasons,
    }
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--modes", nargs="+", default=["full", "singles"])
    parser.add_argument("--greedy-steps", type=int, default=6)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.n < 1 or args.batch < 1 or args.max_tokens < 1:
        parser.error("--n, --batch and --max-tokens must be positive")
    if args.model not in MODEL_REVISIONS:
        parser.error(f"no pinned revision for {args.model}; add it to MODEL_REVISIONS")

    out = Path(args.out or f"results/headroom/{args.model.split('/')[-1]}")
    out.mkdir(parents=True, exist_ok=True)
    revision = MODEL_REVISIONS[args.model]
    model, tokenizer = load(
        snapshot_download(args.model, revision=revision, allow_patterns=WEIGHT_PATTERNS)
    )
    examples = validation_examples(args.n, args.val_size, args.split_seed)
    prompts = [encode(tokenizer, e) for e in examples]
    groups = range(1, mlx_routing.num_groups(model))
    print(
        f"model={args.model} groups={mlx_routing.num_groups(model)} n={len(examples)} "
        f"mean_prompt_tokens={sum(map(len, prompts)) / len(prompts):.0f}",
        flush=True,
    )

    summaries = []
    by_arch: dict[frozenset[int], dict] = {}  # each architecture is generated at most once
    with open(out / "examples.jsonl", "w") as rows:

        def run(skipped, label):
            key = frozenset(skipped)
            if key not in by_arch:
                by_arch[key] = evaluate(
                    model, tokenizer, examples, prompts, skipped, args, label, rows
                )
                summaries.append(by_arch[key])
            return by_arch[key]

        if "full" in args.modes:
            run(set(), "full")
        if "singles" in args.modes:
            for g in groups:
                run(mlx_routing.group_layers(model, [g]), f"group{g}")
                run(mlx_routing.attention_layers(model, [g]), f"attn{g}")
                run(mlx_routing.deltanet_layers(model, [g]), f"deltanet{g}")
        if "greedy" in args.modes:
            chosen: list[int] = []
            for step in range(1, args.greedy_steps + 1):
                trials = [
                    (g, run(mlx_routing.group_layers(model, [*chosen, g]), f"greedy{step}_g{g}"))
                    for g in groups
                    if g not in chosen
                ]
                if not trials:
                    break
                best_g, best = max(
                    trials, key=lambda t: (t[1]["accuracy"], -t[1]["relative_cost"], -t[0])
                )
                chosen.append(best_g)
                print(
                    json.dumps(
                        {"greedy_step": step, "chosen_groups": chosen, "accuracy": best["accuracy"]}
                    ),
                    flush=True,
                )

    meta = {
        "model": args.model,
        "model_revision": revision,
        "xlam_revision": XLAM_REVISION,
        "args": vars(args),
    }
    (out / "summary.json").write_text(json.dumps({"meta": meta, "runs": summaries}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
