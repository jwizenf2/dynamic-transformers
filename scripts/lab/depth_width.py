"""Depth vs width on multi-turn NESTful: loops per turn vs sampled attempts at matched compute.

    python scripts/lab/depth_width.py --backend dory --n 60
    python scripts/lab/depth_width.py --backend ouro --adapters <run>/adapters.safetensors --n 120

Both per-call oracle controls (Ouro v2: depth .308 vs sampling .325; Dory: depth .633 vs
sampling .617 vs best-of-4 sampled dialogues .700) say the per-step gain is search, not depth
allocation. The question this measures instead: for a looped model, is a fixed compute budget
better spent on more loops or on more attempts (majority vote over final answers)?

Per request, for each loop count d in 1..4, ``--k`` whole dialogues are sampled at depth d
(temperature ``--temperature``; turn t of sample s uses seed 1000·s + t; same chat format, tool
execution and grading as the oracle scripts, imported from them). Each attempt records whether it
is correct, a vote key for its final answer (the last executed call's result; numbers rounded to
6 significant digits; failed or missing final calls abstain) and its compute:
* Ouro: ``recompute_loop_tokens`` (4 x prompt tokens + d x generated tokens per turn);
* Dory: ``block_tokens`` (prefill at 4 loops, 96 blocks/token, generated tokens at 8 + 20d + 8).
``decode`` counts generated tokens only (d x tokens for Ouro, blocks(d) x tokens for Dory).

The summary gives, for every depth d and k' <= k: majority-vote accuracy over the first k'
samples (ties go to the earliest attempt), pass@k' (any correct; a search ceiling, not a
policy) and the mean summed compute, plus a vote over one sample at each depth 1..4. Rows are
appended per request; a restart skips done ids. All other strategies (mixed depths, other
budgets) can be recomputed offline from ``examples.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ouro_percall_oracle as opc  # noqa: E402
from ouro_nestful_multiturn import requests  # noqa: E402

DEPTHS = (1, 2, 3, 4)


def vote_key(d) -> str | None:
    """Canonical final answer for voting; None when the last call failed or nothing was called."""
    ok, value = d.last
    if not ok or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return f"{float(value):.6g}" if math.isfinite(value) else repr(value)
    if isinstance(value, tuple):
        value = list(value)
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return repr(value)


def majority(attempts: list[dict]) -> bool:
    """Majority vote over vote keys (ties: earliest attempt); correct iff the winner is."""
    counts: dict[str, int] = {}
    first: dict[str, dict] = {}
    for a in attempts:
        if a["key"] is not None:
            counts[a["key"]] = counts.get(a["key"], 0) + 1
            first.setdefault(a["key"], a)
    if not counts:
        return False
    best = max(counts.values())
    winner = next(a for a in attempts if a["key"] is not None and counts[a["key"]] == best)
    return bool(first[winner["key"]]["correct"])


def backend(args):
    """(start dialogue factory, env factory, compute fn, decode fn, model name)."""
    if args.backend == "ouro":
        from ouro_sft_eval import SYSTEM, load_model

        model, tok = load_model(args.model, None, args.adapters)
        stop_ids = {0, tok.convert_tokens_to_ids("<|im_end|>")}
        gen = opc.Generator(model, tok, args.max_tokens or 256, stop_ids, args.temperature)

        def start(ex):
            system = SYSTEM.format(tools=json.dumps(ex.tools, ensure_ascii=False))
            return opc.Dialogue(
                f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{ex.query}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        def env(ex, functions):
            return opc.Env(gen, ex, functions, args.max_turns)

        def compute(t):
            return 4 * t["prompt"] + t["depth"] * t["tokens"]

        def decode(t):
            return t["depth"] * t["tokens"]

        return gen, start, env, compute, decode, args.adapters

    import dory_nestful_multiturn as dn
    import dory_percall_oracle as dpo

    model, tok = dn.load()
    gen = dpo.DoryGenerator(model, tok, args.max_tokens or 1024, args.temperature)
    b4 = dn.blocks(model, 4)

    def start(ex):
        return dpo.DoryDialogue(
            [{"role": "system", "content": dn.SYSTEM}, {"role": "user", "content": ex.query}]
        )

    def env(ex, functions):
        return dpo.DoryEnv(gen, ex, functions, args.max_turns)

    def compute(t):
        return b4 * t["prompt"] + dn.blocks(model, t["depth"]) * t["tokens"]

    def decode(t):
        return dn.blocks(model, t["depth"]) * t["tokens"]

    return gen, start, env, compute, decode, f"{dn.DORY}@{dn.DORY_REVISION}"


def summarize(data: list[dict], k: int) -> dict:
    n = len(data)
    out: dict = {}
    for d in DEPTHS:
        for kk in range(1, k + 1):
            atts = [r["attempts"][str(d)][:kk] for r in data]
            out[f"d{d}_k{kk}"] = {
                "vote": round(sum(majority(a) for a in atts) / n, 4),
                "pass": round(sum(any(x["correct"] for x in a) for a in atts) / n, 4),
                "compute": round(sum(sum(x["compute"] for x in a) for a in atts) / n, 1),
                "decode": round(sum(sum(x["decode"] for x in a) for a in atts) / n, 1),
            }
    mixed = [[r["attempts"][str(d)][0] for d in DEPTHS] for r in data]
    out["mixed_1234_k1"] = {
        "vote": round(sum(majority(a) for a in mixed) / n, 4),
        "pass": round(sum(any(x["correct"] for x in a) for a in mixed) / n, 4),
        "compute": round(sum(sum(x["compute"] for x in a) for a in mixed) / n, 1),
        "decode": round(sum(sum(x["decode"] for x in a) for a in mixed) / n, 1),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=["ouro", "dory"], required=True)
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B")
    parser.add_argument("--adapters", default=None, help="Ouro LoRA adapters (required for ouro)")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--tasks", default=None, help="NESTful-format jsonl instead of NESTful")
    parser.add_argument("--k", type=int, default=4, help="sampled dialogues per depth")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=None, help="ouro 256, dory 1024")
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--cache-limit-gb", type=float, default=4.0)
    args = parser.parse_args()
    if args.backend == "ouro" and not args.adapters:
        parser.error("--adapters is required for --backend ouro")

    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    gen, start, make_env, compute, decode, model_name = backend(args)
    examples, functions = requests(args.n, args.tasks)
    suffix = f"_{Path(args.tasks).stem}" if args.tasks else ""
    out = Path(f"results/depth_width/{args.backend}/{args.tag}{suffix}")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "examples.jsonl"
    done = set()
    if path.exists():
        good = []
        for line in path.read_text().splitlines():
            try:
                done.add(json.loads(line)["id"])
            except json.JSONDecodeError:
                break
            good.append(line)
        path.write_text("".join(x + "\n" for x in good))

    t0 = time.perf_counter()
    with open(path, "a") as rows:
        for i, ex in enumerate(examples, 1):
            if ex.id in done:
                continue
            gen.reset()
            before = gen.calls
            env, first = make_env(ex, functions), start(ex)
            attempts: dict[str, list[dict]] = {}
            for d in DEPTHS:
                attempts[str(d)] = []
                for s in range(1, args.k + 1):
                    final = env.finish(first, lambda _t, d=d: d, lambda t, s=s: 1000 * s + t)
                    turns = final.turns
                    attempts[str(d)].append(
                        {
                            "correct": env.correct(final),
                            "key": vote_key(final),
                            "compute": sum(compute(t) for t in turns),
                            "decode": sum(decode(t) for t in turns),
                            "turns": len(turns),
                            "tokens": sum(t["tokens"] for t in turns),
                        }
                    )
            rows.write(
                json.dumps({"id": ex.id, "gold_calls": len(ex.calls), "attempts": attempts}) + "\n"
            )
            rows.flush()
            mx.clear_cache()
            print(f"::progress {i}/{len(examples)} requests", flush=True)
            print(
                json.dumps(
                    {
                        "done": i,
                        "generations": gen.calls - before,
                        "s": round(time.perf_counter() - t0),
                    }
                ),
                flush=True,
            )

    data = [json.loads(x) for x in path.read_text().splitlines()]
    summary = {
        "n": len(data),
        "model": model_name,
        "args": vars(args),
        "strategies": summarize(data, args.k),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
