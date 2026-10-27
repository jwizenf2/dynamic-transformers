"""Router smoke test, collection: per-request depth labels and pre-generation probe features.

    python scripts/lab/router_smoke.py \
        --adapters results/ouro_sft/chain-depth-v3/adapters.safetensors \
        --n 292
    python scripts/lab/router_smoke_analysis.py results/router_smoke/chain-depth-v3

Gate before building a per-request loop router on chain v3 (plan of 2026-10-25). Two questions:
(a) is "how many loops a request needs" a stable property of the request (greedy and sampled
runs agree), and (b) can signals available before generation predict it? Surface features of
the question did not (best length/number rule within 1-3 points of a random depth mix, on 240
requests); the remaining candidate is the model's own state.

Requests: the 292 "clean" MathQA-validation tasks (``make_mathqa_clean_val.py``: never trained on,
and no near-duplicate, same computation or exact copy among the train tasks, the relation
NESTful test has to train). The full validation set is mostly train templates with other
numbers (v3 at depth 4: .346 on the 26 clean vs .489 on the 94 other tasks of the first 120),
where depth need could track memorisation. Depth-balanced order of
``ouro_nestful_multiturn.requests``, slice ``[start, start + n)``; NESTful is not touched. The
greedy rows are also the v3 baseline of the v4 pilot (same protocol as
``ouro_nestful_multiturn.py --variant plain``).

Per request:
* ``probe``: Ouro's first ``--k`` answer tokens decoded greedily with all 4 loops on the first
  prompt, recording per token and loop the exit gate, entropy, top-2 margin, agreement with
  the loop-4 argmax and the relative state change (``ouro_early_features.probe``, [k, 4, 5]).
  A deployed probe-then-commit router would pay these k tokens at full depth.
* Dialogues at each depth in ``--depths``: greedy, and one sample (temperature
  ``--temperature``, seed ``crc32(id) + depth`` then split per token), with the protocol of
  ``ouro_nestful_multiturn.py`` (plain prompt; every pass of every turn at the request's depth,
  incremental KV cache; executed and graded like NESTful).
Rows are appended (``examples.jsonl``) and a restart skips done ids.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ouro_early_features import probe  # noqa: E402
from ouro_nestful_multiturn import requests  # noqa: E402
from ouro_sft_eval import SYSTEM, load_model, parse_hermes  # noqa: E402

from dht.toolcall.nestful import answers_match, run_dialogue  # noqa: E402


def make_assistant(model, tok, caches, first, stop_ids, max_tokens, key, temperature, log):
    """``run_dialogue`` callback on shared caches; greedy when ``key`` is None, else sampled."""
    state = {"key": key}

    def assistant(feedback: str | None) -> list[dict]:
        chunk = (
            first
            if feedback is None
            else f"<|im_end|>\n<|im_start|>tool\n{feedback}<|im_end|>\n<|im_start|>assistant\n"
        )
        ids = tok.encode(chunk, add_special_tokens=False)
        logits = model(mx.array([ids]), caches)[:, -1, :]
        out: list[int] = []
        for _ in range(max_tokens):
            if state["key"] is None:
                tid = int(mx.argmax(logits, axis=-1).item())
            else:
                state["key"], sub = mx.random.split(state["key"])
                tid = int(mx.random.categorical(logits / temperature, key=sub).item())
            if tid in stop_ids:
                break
            out.append(tid)
            logits = model(mx.array([[tid]]), caches)[:, -1, :]
        log["tokens"] += len(out)
        log["prompt_tokens"] += len(ids)
        text = tok.decode(out)
        log["texts"].append(text[:300])
        return [{"name": c["name"], "arguments": c["arguments"]} for c in parse_hermes(text)]

    return assistant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B")
    parser.add_argument("--adapters", required=True)
    parser.add_argument(
        "--tasks",
        default="results/mathqa/validation_clean_tasks.jsonl",
        help='NESTful-format jsonl, or "nestful" for the NESTful math subset (test set)',
    )
    parser.add_argument("--tag", default="", help="output folder suffix (e.g. nestful)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=292)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--modes", nargs="+", choices=["greedy", "sample"], default=["greedy", "sample"]
    )
    parser.add_argument("--k", type=int, default=8, help="probe tokens")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    args = parser.parse_args()

    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    model, tok = load_model(args.model, None, args.adapters)
    stop_ids = {0, tok.convert_tokens_to_ids("<|im_end|>")}
    tasks = None if args.tasks == "nestful" else args.tasks
    examples, functions = requests(args.start + args.n, tasks)
    examples = examples[args.start :]
    tag = f"_{args.tag}" if args.tag else ""
    out = Path(f"results/router_smoke/{Path(args.adapters).parent.name}{tag}")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "examples.jsonl"
    done, good = set(), []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                done.add(json.loads(line)["id"])
            except json.JSONDecodeError:
                break
            good.append(line)
        path.write_text("".join(x + "\n" for x in good))
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    t0 = time.perf_counter()
    with open(path, "a") as rows:
        for j, e in enumerate(examples, 1):
            if e.id in done:
                continue
            system = SYSTEM.format(tools=json.dumps(e.tools, ensure_ascii=False))
            first = (
                f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{e.query}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            model.ut_steps = 4
            feats = probe(model, tok.encode(first, add_special_tokens=False), args.k)
            runs: dict[str, dict] = {}
            for d in args.depths:
                for mode in args.modes:
                    model.ut_steps = d
                    caches = model.make_cache()
                    key = None if mode == "greedy" else mx.random.key(zlib.crc32(e.id.encode()) + d)
                    log = {"tokens": 0, "prompt_tokens": 0, "texts": []}
                    assistant = make_assistant(
                        model,
                        tok,
                        caches,
                        first,
                        stop_ids,
                        args.max_tokens,
                        key,
                        args.temperature,
                        log,
                    )
                    res = run_dialogue(assistant, e.tools, functions, args.max_turns)
                    runs[f"d{d}_{mode}"] = {
                        "correct": bool(res["ok"] and answers_match(res["value"], e.answer)),
                        "value": repr(res["value"])[:80],
                        **{k: res[k] for k in ("turns", "calls", "errors", "finished")},
                        **log,
                    }
                    mx.clear_cache()
            model.ut_steps = 4
            probe_rows = [
                [[None if math.isnan(x) else round(float(x), 5) for x in loop] for loop in t]
                for t in feats.tolist()
            ]
            row = {
                "id": e.id,
                "query": e.query,
                "gold_calls": len(e.calls),
                "probe": probe_rows,
                "runs": runs,
            }
            rows.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.flush()
            print(f"::progress {j}/{len(examples)} requests", flush=True)
            if j % 10 == 0:
                print(json.dumps({"done": j, "s": round(time.perf_counter() - t0)}), flush=True)


if __name__ == "__main__":
    main()
