"""Dependency-depth test: Ouro accuracy at each loop count on chained (NESTful) tool calls.

    python scripts/lab/ouro_nestful.py --n 360

Hypothesis (findings doc, section 7, step 3): the number of loop passes a request needs grows
with its dependency depth, the longest chain of calls that feed into each other. xLAM requests
are depth 1 (single or parallel calls); NESTful's math subset covers depth 2-12.

Method (same probe as ``ouro_headroom.py``, so results are comparable):
* Data: NESTful at a pinned GitHub commit, SHA256-checked (``fetch``). Only the math subset is
  used (all gold calls in ``basic_functions.py``; 1,416 of 1,861 samples), because only those
  functions are executed. See ``dht.toolcall.nestful``.
* Order: requests are grouped by gold depth (2, 3, 4, 5, 6, 7+), each group shuffled with
  ``--seed``, then interleaved round-robin. The first 360 therefore hold ~60 per depth, and a
  partial run stays balanced. ``--start`` continues the same order (e.g. router training data).
* Prompt: Ouro-2.6B is not tool-call tuned, so it is prompted few-shot with NESTful's own three
  ICL examples (``src/icl_examples.json``, not in the evaluation file), each shown with its
  own tool list, in the same ``### Tools / ### Query / ### Calls`` layout as the xLAM probe.
* Grading: the predicted sequence is executed and its final value compared with the gold
  answer (NESTful's win rate; primary). Exact sequence match is also logged.
* Router features: identical to the xLAM probe (pass-1 states, gate/entropy/margin per pass),
  plus gold depth and call count for analysis only (a router may not use them).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from ouro_mlx.model import OuroConfig, OuroForCausalLM
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ouro_headroom import OURO_REVISIONS, generate, prefill_signals  # noqa: E402

from dht.toolcall.nestful import (  # noqa: E402
    answers_match,
    basic_function_names,
    dependency_depth,
    exact_match,
    execute,
    load_functions,
    load_nestful,
    parse_sequence,
)

NESTFUL_COMMIT = "fc2c4123e73500a56185a5fb354f05d1c8b4890c"
NESTFUL_FILES = {
    "data_v2/nestful_data.jsonl": (
        "e7c9999e2eaaa2bea87776bc55e2b0913a0cf089f4b6817c7644554547df22c1"
    ),
    "data_v2/executable_functions/basic_functions.py": (
        "db8048658f7ccd930c691a36dd0831a7d46dbffa2a275932f60af43792128212"
    ),
    "src/icl_examples.json": "fa10032a95bb1d1177f25ddbb6bc102140cac0b41e4d19ea78eecc8b249b18e7",
}
INSTRUCTIONS = (
    "You are a function-calling assistant. Given the available tools and a user query, answer "
    "with a JSON list of the function calls needed to compute the answer, in order. Each call "
    'is {"name": ..., "label": "$var_N", "arguments": {...}}. To use the result of an earlier '
    'call as an argument, write "$var_N.result$".\n\n'
)


def fetch(cache: Path) -> Path:
    """Download the pinned NESTful files once; every file is checked against its SHA256."""
    root = cache / NESTFUL_COMMIT
    for rel, sha in NESTFUL_FILES.items():
        path = root / rel
        if not path.exists():
            url = f"https://raw.githubusercontent.com/IBM/NESTFUL/{NESTFUL_COMMIT}/{rel}"
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (pinned https URL)
                data = r.read()
            if hashlib.sha256(data).hexdigest() != sha:
                raise ValueError(f"SHA256 mismatch for {rel}")
            path.write_bytes(data)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError(f"cached {rel} does not match its SHA256")
    return root


def tools_json(tools) -> str:
    return json.dumps(tools, ensure_ascii=False, separators=(",", ":"))


def block(tools, query: str) -> str:
    return f"### Tools:\n{tools_json(tools)}\n### Query:\n{query}\n### Calls:\n"


def build_prompt(shots, example) -> str:
    text = INSTRUCTIONS
    for s in shots:
        text += block(s["tools"], s["input"]) + json.dumps(s["output"]) + "\n\n"
    return text + block(example.tools, example.query)


def depth_bucket(depth: int) -> str:
    return str(depth) if depth < 7 else "7+"


def ordered(examples, seed: int):
    groups = defaultdict(list)
    for e in examples:
        groups[depth_bucket(dependency_depth(e.calls))].append(e)
    rng = random.Random(seed)
    lists = []
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda e: e.id)
        rng.shuffle(g)
        lists.append(g)
    out = []
    for i in range(max(len(g) for g in lists)):
        out.extend(g[i] for g in lists if i < len(g))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B")
    parser.add_argument("--n", type=int, default=360)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", default=str(Path.home() / ".cache" / "dht" / "nestful"))
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    args = parser.parse_args()
    if args.n < 1 or not args.steps or min(args.steps) < 1:
        parser.error("--n and --steps must be positive")

    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))  # see ouro_headroom.py
    root = fetch(Path(args.cache))
    functions_file = root / "data_v2/executable_functions/basic_functions.py"
    basic = basic_function_names(functions_file)
    functions = load_functions(functions_file)
    pool = [
        e
        for e in load_nestful(root / "data_v2/nestful_data.jsonl")
        if all(c["name"] in basic for c in e.calls)
    ]
    shots = json.loads((root / "src/icl_examples.json").read_text(encoding="utf-8"))
    evals = ordered(pool, args.seed)[args.start : args.start + args.n]

    snap = Path(snapshot_download(args.model, revision=OURO_REVISIONS[args.model]))
    cfg = OuroConfig.from_json(snap / "config.json")
    model = OuroForCausalLM(cfg, ut_steps=max(args.steps))
    model.load_weights(str(snap / "model.safetensors"), strict=True)
    mx.eval(model.parameters())
    model.eval()
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(snap / "tokenizer.json"))
    tokenizer.eos_token_id = 0  # config.json: eos_token_id = 0

    tag = f"nestful_{args.start}_{args.start + len(evals)}"
    out = Path(f"results/ouro_nestful/{args.model.split('/')[-1]}/{tag}")
    out.mkdir(parents=True, exist_ok=True)
    win: dict[int, dict[str, bool]] = {r: {} for r in args.steps}
    exact: dict[int, dict[str, bool]] = {r: {} for r in args.steps}
    depth = {e.id: dependency_depth(e.calls) for e in evals}
    features: dict[str, list] = {
        k: []
        for k in (
            "id",
            "pool1",
            "last1",
            "gate",
            "entropy",
            "margin",
            "prompt_tokens",
            "n_tools",
            "depth",
            "n_calls",
        )
    }
    start = time.perf_counter()
    with open(out / "examples.jsonl", "w") as rows:
        for k, e in enumerate(evals, 1):
            prompt = build_prompt(shots, e)
            ids_ = tokenizer(prompt)["input_ids"]
            sig = prefill_signals(model, ids_, max(args.steps))
            features["id"].append(e.id)
            for key in ("pool1", "last1", "gate", "entropy", "margin"):
                features[key].append(sig[key])
            features["prompt_tokens"].append(len(ids_))
            features["n_tools"].append(len(e.tools))
            features["depth"].append(depth[e.id])
            features["n_calls"].append(len(e.calls))
            for r in args.steps:
                model.ut_steps = r
                text, n_tok, gates, secs = generate(model, tokenizer, prompt, args.max_tokens)
                pred = parse_sequence(text)
                ran, value = execute(pred, e.tools, functions)
                ok = ran and answers_match(value, e.answer)
                win[r][e.id] = ok
                exact[r][e.id] = exact_match(pred, e.calls)
                rows.write(
                    json.dumps(
                        {
                            "steps": r,
                            "id": e.id,
                            "depth": depth[e.id],
                            "correct": ok,
                            "exact": exact[r][e.id],
                            "reason": "ok" if ok else (value if not ran else "wrong answer"),
                            "n_pred_calls": len(pred),
                            "out_tokens": n_tok,
                            "seconds": round(secs, 3),
                            "gates": [round(g, 4) for g in gates],
                            "text": text[:600],
                        },
                        default=str,
                    )
                    + "\n"
                )
            rows.flush()
            mx.clear_cache()
            print(f"::progress {k}/{len(evals)} requests", flush=True)
            if k % 30 == 0 or k == len(evals):
                accs = {r: round(sum(win[r].values()) / k, 3) for r in args.steps}
                print(
                    json.dumps(
                        {
                            "done": k,
                            "of": len(evals),
                            "accuracy": accs,
                            "elapsed_s": round(time.perf_counter() - start),
                        }
                    ),
                    flush=True,
                )
                np.savez_compressed(
                    out / "features.npz", **{key: np.array(v) for key, v in features.items()}
                )

    def summarize(ids: list[str]) -> dict:
        smallest = []
        for i in ids:
            ok_rs = [r for r in sorted(args.steps) if win[r][i]]
            smallest.append(ok_rs[0] if ok_rs else None)
        solved = [s for s in smallest if s is not None]
        return {
            "n": len(ids),
            "accuracy_by_steps": {
                r: round(sum(win[r][i] for i in ids) / len(ids), 3) for r in args.steps
            },
            "exact_by_steps": {
                r: round(sum(exact[r][i] for i in ids) / len(ids), 3) for r in args.steps
            },
            "oracle_accuracy": round(len(solved) / len(ids), 3),
            "smallest_correct_histogram": {str(r): solved.count(r) for r in sorted(args.steps)},
            "mean_smallest_correct_on_solved": round(sum(solved) / len(solved), 3)
            if solved
            else None,
        }

    by_depth = defaultdict(list)
    for e in evals:
        by_depth[depth_bucket(depth[e.id])].append(e.id)
    summary = {
        "model": args.model,
        "revision": OURO_REVISIONS[args.model],
        "nestful_commit": NESTFUL_COMMIT,
        "math_subset_size": len(pool),
        "all": summarize([e.id for e in evals]),
        "by_depth": {key: summarize(v) for key, v in sorted(by_depth.items())},
        "args": vars(args),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
