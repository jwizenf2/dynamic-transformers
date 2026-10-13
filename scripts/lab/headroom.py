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
               highest (ties -> cheaper), up to ``--greedy-steps`` groups.

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
    with mlx_routing.skip_layers(model, skipped):
        for i in range(0, len(prompts), args.batch):
            response = batch_generate(
                model, tokenizer, prompts[i : i + args.batch], max_tokens=args.max_tokens
            )
            texts.extend(response.texts)
    seconds = time.perf_counter() - start
    correct = 0
    out_tokens = 0
    reasons: dict[str, int] = {}
    for example, text in zip(examples, texts, strict=True):
        ok, why = check_example(
            parse_tool_calls(text, example.functions), example.functions, example.ground_truth
        )
        n_tok = len(tokenizer.encode(text, add_special_tokens=False))
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

    out = Path(args.out or f"results/headroom/{args.model.split('/')[-1]}")
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load(args.model)
    examples = validation_examples(args.n, args.val_size, args.split_seed)
    prompts = [encode(tokenizer, e) for e in examples]
    groups = range(1, mlx_routing.num_groups(model))
    print(
        f"model={args.model} groups={mlx_routing.num_groups(model)} n={len(examples)} "
        f"mean_prompt_tokens={sum(map(len, prompts)) / len(prompts):.0f}",
        flush=True,
    )

    summaries = []
    with open(out / "examples.jsonl", "w") as rows:

        def run(skipped, label):
            summaries.append(
                evaluate(model, tokenizer, examples, prompts, skipped, args, label, rows)
            )
            return summaries[-1]

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
                    run(mlx_routing.group_layers(model, [*chosen, g]), f"greedy{step}_g{g}")
                    for g in groups
                    if g not in chosen
                ]
                best = max(trials, key=lambda s: (s["accuracy"], -s["relative_cost"]))
                chosen = sorted(
                    set(chosen)
                    | {g for g in groups if g * mlx_routing.interval(model) in best["skipped"]}
                )
                print(json.dumps({"greedy_step": step, "chosen_groups": chosen}), flush=True)

    meta = {"model": args.model, "xlam_revision": XLAM_REVISION, "args": vars(args)}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "runs": summaries}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
