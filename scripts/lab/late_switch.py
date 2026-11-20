"""Late switch: continue a 2-loop dialogue at 4 loops from a decision point (collection).

    python scripts/lab/late_switch.py --adapters A --tasks nestful --n 720 --tag nestful \
        --points 64 128

Question (docs/research/late_switch_2026-11-20.md): an early-signal router ranks "4 loops help"
better the longer it waits, but the requests it would rescue leave the 4-loop path as fast
(``router_window_hidden.py``). A switch before the 2- and 4-loop dialogues diverge reproduces the
4-loop dialogue exactly; whether a switch after the divergence still rescues the request was not
in the data. This measures it for every request and decision point.

Per request: (1) the 2-loop greedy dialogue with per-token loop-1/2 signals and the generated ids
of every turn (``router_trace.trace_one``, same protocol as all 2-loop runs); (2) for each decision
point T (in trace steps: generated tokens plus one stop step per turn, as in the window analysis),
the context the 2-loop run had built at T (first prompt, earlier turns' ids, their tool turns
re-executed, and the first j ids of the current turn) is prefilled at 4 loops on fresh caches,
and the dialogue continues greedily at 4 loops: the rest of the current turn, its calls executed,
then further turns as in ``run_dialogue`` (at most ``--max-turns`` turns with calls in total,
graded like NESTful: the last executed call's result). Prefilling the whole context at 4 loops
computes loops 1–2 again; their values equal the 2-loop run's (step-s states depend only on steps
<= s), so the outcome is that of a switch that reuses the loop-1/2 caches, and its compute is
charged analytically as 4 x (all tokens of the switched dialogue). A T beyond the end of the
2-loop dialogue is recorded as ``ended`` (a switch there is a restart, whose outcome is the 4-loop
run's). Rows: ``results/late_switch/<adapters parent>_<tag>/examples.jsonl`` with the trace (as
``router_trace``) and ``switch[T]`` = correct, value, finished, turns, context_tokens,
generated_after, prompt_tokens_after, decided_at (turn, offset). Appended; a restart skips done ids.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ouro_nestful_multiturn import requests  # noqa: E402
from ouro_sft_eval import SYSTEM, load_model, parse_hermes  # noqa: E402
from router_trace import trace_one  # noqa: E402

from dht.toolcall.nestful import answers_match, run_call, tool_responses  # noqa: E402

TOOL_CHUNK = "<|im_end|>\n<|im_start|>tool\n{}<|im_end|>\n<|im_start|>assistant\n"


def first_prompt(e) -> str:
    system = SYSTEM.format(tools=json.dumps(e.tools, ensure_ascii=False))
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{e.query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def locate(turn_steps: list[int], point: int) -> tuple[int, int] | None:
    """(turn, generated tokens of that turn before the switch) for a trace-step index, or None
    if the dialogue ended before it. ``turn_steps`` are the trace steps per turn (n_tokens + 1
    with the stop step; n_tokens for a turn cut at --max-tokens), as in the window analysis."""
    start = 0
    for k, n in enumerate(turn_steps):
        if point < start + n:
            return k, point - start
        start += n
    return None


def execute_turn(calls, n_calls, e, functions, results):
    outcomes, last = [], None
    for c in calls:
        ok, value = run_call(c, n_calls, e.tools, functions, results)
        n_calls += 1
        outcomes.append((c["name"], ok, value))
        last = (ok, value)
    return outcomes, last, n_calls


def continue_at_4(model, tok, e, functions, turn_ids, k, j, args, stop_ids) -> dict:
    """Rebuild the 2-loop context at (turn k, offset j) and finish the dialogue at 4 loops."""
    ids = tok.encode(first_prompt(e), add_special_tokens=False)
    results: dict = {}
    n_calls, last, turns_with_calls = 0, (False, "no calls"), 0
    for t in range(k):  # earlier turns, replayed exactly
        calls = [
            {"name": c["name"], "arguments": c["arguments"]}
            for c in parse_hermes(tok.decode(turn_ids[t]))
        ]
        ids += turn_ids[t]
        outcomes, lt, n_calls = execute_turn(calls, n_calls, e, functions, results)
        last = lt or last
        turns_with_calls += 1
        ids += tok.encode(TOOL_CHUNK.format(tool_responses(outcomes)), add_special_tokens=False)
    ids += turn_ids[k][:j]
    context = len(ids)
    model.ut_steps = 4
    caches = model.make_cache()
    gen = generated_after = 0
    prompt_after = 0
    current = list(turn_ids[k][:j])
    logits = model(mx.array([ids]), caches)[:, -1, :]
    finished = False
    while True:
        for _ in range(args.max_tokens - len(current)):
            tid = int(mx.argmax(logits, axis=-1).item())
            if tid in stop_ids:
                break
            current.append(tid)
            gen += 1
            logits = model(mx.array([[tid]]), caches)[:, -1, :]
        calls = [
            {"name": c["name"], "arguments": c["arguments"]}
            for c in parse_hermes(tok.decode(current))
        ]
        if not calls:
            finished = True
            break
        turns_with_calls += 1
        outcomes, lt, n_calls = execute_turn(calls, n_calls, e, functions, results)
        last = lt or last
        if turns_with_calls >= args.max_turns:
            break
        chunk = tok.encode(TOOL_CHUNK.format(tool_responses(outcomes)), add_special_tokens=False)
        prompt_after += len(chunk)
        logits = model(mx.array([chunk]), caches)[:, -1, :]
        current = []
    generated_after = gen
    mx.clear_cache()
    return {
        "correct": bool(last[0] and answers_match(last[1], e.answer)),
        "value": repr(last[1])[:80],
        "finished": finished,
        "turns": turns_with_calls,
        "context_tokens": context,
        "generated_after": generated_after,
        "prompt_tokens_after": prompt_after,
        "decided_at": [k, j],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="ByteDance/Ouro-2.6B")
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--tasks", default="nestful")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=720)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--points", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--cache-limit-gb", type=float, default=2.0)
    ap.add_argument(
        "--smoke", nargs="*", default=[], help="labelled router_smoke rows the tasks must match"
    )
    args = ap.parse_args()

    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    model, tok = load_model(args.model, None, args.adapters)
    stop_ids = {0, tok.convert_tokens_to_ids("<|im_end|>")}
    vocab = tok.convert_ids_to_tokens(list(range(len(tok))))
    has_digit = np.array([any(c in "0123456789" for c in (t or "")) for t in vocab])
    examples, functions = requests(
        args.start + args.n, None if args.tasks == "nestful" else args.tasks
    )
    examples = examples[args.start :]
    if args.smoke:  # same tasks as the stored labelled runs: query and gold chain length
        smoke = {}
        for f in args.smoke:
            smoke.update({r["id"]: r for r in map(json.loads, Path(f).read_text().splitlines())})
        for e in examples:
            r = smoke.get(e.id)
            assert r and (r["query"], r["gold_calls"]) == (e.query, len(e.calls)), e.id
    out = Path(f"results/late_switch/{Path(args.adapters).parent.name}_{args.tag}")
    (out / "hidden").mkdir(parents=True, exist_ok=True)
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
    trace_args = argparse.Namespace(max_tokens=args.max_tokens, max_turns=args.max_turns)
    t0 = time.perf_counter()
    with open(path, "a") as rows:
        for i, e in enumerate(examples, 1):
            if e.id in done:
                continue
            row, arr = trace_one(model, tok, e, functions, trace_args, stop_ids, has_digit)
            np.save(out / "hidden" / f"{e.id}.npy", arr)
            turn_ids = [s["ids"] for s in row["turn_signals"]]
            lens = [len(s["ent2"]) for s in row["turn_signals"]]
            row["switch"] = {}
            for p in args.points:
                loc = locate(lens, p)
                if loc is None:
                    row["switch"][str(p)] = {"ended": True}
                    continue
                row["switch"][str(p)] = continue_at_4(
                    model, tok, e, functions, turn_ids, loc[0], loc[1], args, stop_ids
                )
            model.ut_steps = 4
            rows.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.flush()
            mx.clear_cache()
            print(f"::progress {i}/{len(examples)} requests", flush=True)
            if i % 10 == 0:
                print(json.dumps({"done": i, "s": round(time.perf_counter() - t0)}), flush=True)


if __name__ == "__main__":
    main()
