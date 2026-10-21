"""Dory 1.0 4B at fixed loop counts on multi-turn NESTful (or MathQA-val) tool calling, in MLX.

    python scripts/lab/dory_nestful_multiturn.py --n 60 --loops 1 2 3 4
    python scripts/lab/dory_nestful_multiturn.py --n 60 \
        --tasks results/mathqa/validation_tasks.jsonl

Why: Ouro-2.6B after our fine-tunes solves ~20% of chained NESTful problems at 4 loops, which caps
any per-step loop-allocation result. NVIDIA's Dory 1.0 4B (released 2026-10-16) is a looped model
post-trained with SFT + RL for agentic tool use, with 1-4 loops selectable at run time. This is
the calibration that decides whether it is a better backbone: accuracy, generated tokens and
compute at each fixed loop count, on the same requests and grading as the Ouro runs.

Model: NVIDIA's BF16 checkpoint (``nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`` at a pinned
revision) loaded through the vendored community MLX port (``third_party/dory_mlx``; parity
checks in its README). The loop count is set per call; ``per_loop`` KV caches (NVIDIA's launch
setting), built once for 4 loops.

Protocol per request and loop count L (``ouro_nestful_multiturn.py`` analogue in Dory's own
format):
1. Messages rendered with Dory's chat template: system prompt = ``--system`` text, tools as
   OpenAI function schemas converted from the request's NESTful specs, thinking on unless
   ``--no-thinking``.
2. Each assistant turn: the whole rendered conversation is prefilled at L loops on fresh caches,
   then decoded at L loops until ``<|im_end|>`` or ``--max-tokens``. Sampling follows NVIDIA's
   recommendation (temperature 1.0, top-p 0.95) with a seed per (request, L, turn); ``--temp 0``
   is greedy.
3. Calls in the visible answer (after ``</think>``) are parsed (``dht.toolcall.dory_format``) and
   executed with NESTful's functions (``run_call``); results return as tool messages.
4. Ends at a turn without calls or after ``--max-turns`` turns with calls. Graded like NESTful:
   the last executed call's result must match the gold answer.
Per turn we log generated tokens, reasoning tokens and prompt tokens; compute is reported as
loop-tokens (L x (prompt + generated) per turn, prefill included). Rows are appended and a
restart skips graded (id, L) pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dory_mlx  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402
from ouro_nestful_multiturn import requests  # noqa: E402

from dht.toolcall.dory_format import openai_tool, parse_calls, split_thinking  # noqa: E402
from dht.toolcall.nestful import answers_match, dependency_depth, run_call  # noqa: E402

DORY = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
DORY_REVISION = "b366b87f762ea334ba3eb55912031baba1515810"
SYSTEM = (
    "You are a helpful assistant that solves the user's problem by calling the provided "
    "functions. Call functions to compute every intermediate value, use their results in later "
    "calls, and give the final answer after the last result."
)
STOP_IDS = {2, 11}


def load():
    path = Path(snapshot_download(DORY, revision=DORY_REVISION))
    model, tok = dory_mlx.load(str(path), loops=4)
    model.make_cache()  # builds the per_loop cache plan for 4 loops before any lower count is used
    return model, tok


def generate(model, tok, prompt: str, loops: int, max_tokens: int, sampler) -> tuple[str, int, int]:
    ids = tok.encode(prompt, add_special_tokens=False)
    model.args.n_recurrent_loops = loops
    caches = model.make_cache()
    logits = model(mx.array([ids]), caches)[:, -1, :]
    out: list[int] = []
    for _ in range(max_tokens):
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tid = int(sampler(logprobs).item())
        if tid in STOP_IDS:
            break
        out.append(tid)
        logits = model(mx.array([[tid]]), caches)[:, -1, :]
    model.args.n_recurrent_loops = 4
    return tok.decode(out), len(out), len(ids)


def dialogue(model, tok, ex, functions, loops, args, seed_base) -> dict:
    tools = [openai_tool(t) for t in ex.tools]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ex.query}]
    results: dict = {}
    last = (False, "no calls")
    turns, n_calls, errors, finished = [], 0, 0, False
    for t in range(args.max_turns + 1):
        prompt = tok.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=not args.no_thinking,
        )
        mx.random.seed(seed_base + t)
        sampler = make_sampler(temp=args.temp, top_p=args.top_p)
        text, n_gen, n_prompt = generate(model, tok, prompt, loops, args.max_tokens, sampler)
        reasoning, answer = ("", text.strip()) if args.no_thinking else split_thinking(text)
        calls = parse_calls(answer)
        turns.append(
            {
                "tokens": n_gen,
                "reasoning_tokens": len(tok.encode(reasoning, add_special_tokens=False))
                if reasoning
                else 0,
                "prompt": n_prompt,
                "calls": len(calls),
                "truncated": n_gen >= args.max_tokens,
            }
        )
        if not calls or t == args.max_turns:
            finished = not calls
            break
        msg = {
            "role": "assistant",
            "content": answer.split("<tool_call>")[0].strip(),
            "reasoning_content": reasoning,
            "tool_calls": [{"type": "function", "function": c} for c in calls],
        }
        messages.append(msg)
        for c in calls:
            ok, value = run_call(c, n_calls, ex.tools, functions, results)
            n_calls += 1
            errors += not ok
            last = (ok, value)
            content = (
                (list(value) if isinstance(value, tuple) else value) if ok else {"error": value}
            )
            messages.append({"role": "tool", "content": json.dumps(content, ensure_ascii=False)})
    correct = bool(last[0] and answers_match(last[1], ex.answer))
    gen = sum(x["tokens"] for x in turns)
    return {
        "correct": correct,
        "turns": len(turns),
        "calls": n_calls,
        "errors": errors,
        "finished": finished,
        "generated_tokens": gen,
        "reasoning_tokens": sum(x["reasoning_tokens"] for x in turns),
        "truncated_turns": sum(x["truncated"] for x in turns),
        "loop_tokens": loops * sum(x["tokens"] + x["prompt"] for x in turns),
        "decode_loop_tokens": loops * gen,
        "value": repr(last[1])[:80],
        "per_turn": turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--tasks", default=None, help="NESTful-format jsonl instead of NESTful")
    parser.add_argument("--loops", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=3072, help="per assistant turn")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--cache-limit-gb", type=float, default=4.0)
    args = parser.parse_args()

    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    model, tok = load()
    examples, functions = requests(args.n, args.tasks)
    name = Path(args.tasks).stem if args.tasks else "nestful"
    tag = args.tag or ("nothink" if args.no_thinking else "think")
    out = Path(f"results/dory_multiturn/{name}/{tag}")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "examples.jsonl"
    done = set()
    if path.exists():
        good = []
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                break
            good.append(line)
            done.add((r["id"], r["loops"]))
        path.write_text("".join(x + "\n" for x in good))
    t0 = time.perf_counter()
    with open(path, "a") as rows:
        for k, ex in enumerate(examples, 1):
            for loops in args.loops:
                if (ex.id, loops) in done:
                    continue
                seed_base = (zlib.crc32(ex.id.encode()) % 100_000) * 100 + loops * 10
                res = dialogue(model, tok, ex, functions, loops, args, seed_base)
                rows.write(
                    json.dumps(
                        {
                            "id": ex.id,
                            "loops": loops,
                            "gold_depth": dependency_depth(ex.calls),
                            "gold_calls": len(ex.calls),
                            **res,
                        }
                    )
                    + "\n"
                )
                rows.flush()
                mx.clear_cache()
            print(f"::progress {k}/{len(examples)} requests", flush=True)
            print(json.dumps({"done": k, "s": round(time.perf_counter() - t0)}), flush=True)

    data = [json.loads(x) for x in path.read_text().splitlines()]
    by = defaultdict(list)
    for r in data:
        by[r["loops"]].append(r)
    summary = {
        "model": DORY,
        "revision": DORY_REVISION,
        "args": vars(args),
        "by_loops": {
            L: {
                "n": len(rs),
                "accuracy": round(sum(r["correct"] for r in rs) / len(rs), 4),
                "generated_tokens": round(sum(r["generated_tokens"] for r in rs) / len(rs), 1),
                "reasoning_tokens": round(sum(r["reasoning_tokens"] for r in rs) / len(rs), 1),
                "loop_tokens": round(sum(r["loop_tokens"] for r in rs) / len(rs)),
                "decode_loop_tokens": round(sum(r["decode_loop_tokens"] for r in rs) / len(rs)),
                "turn_limit": round(sum(not r["finished"] for r in rs) / len(rs), 3),
                "errors_per_call": round(
                    sum(r["errors"] for r in rs) / max(1, sum(r["calls"] for r in rs)), 3
                ),
                "truncated_turns": sum(r["truncated_turns"] for r in rs),
            }
            for L, rs in sorted(by.items())
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
