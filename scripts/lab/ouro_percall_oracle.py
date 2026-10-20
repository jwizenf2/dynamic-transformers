"""Per-call depth oracle on multi-turn NESTful: does choosing loops per call beat per request?

    python scripts/lab/ouro_percall_oracle.py --adapters <run>/adapters.safetensors --n 60

Go/no-go for path B's RL depth policy (plan_2026-10-17_paper_paths.md, Gate B). The policy
would choose the loop count for each assistant turn (each round of tool calls) of a multi-turn
dialogue. Before training anything, this measures the headroom with oracles:

* fixed depth d = 1..4 for every turn (per request: the cheapest correct d is the per-request
  oracle);
* a per-call oracle (outcome-greedy): turn by turn, take the smallest depth whose turn, followed by
  a depth-4 rollout of the remaining turns, ends with the correct answer; if no depth does, take 4.
  It is an upper bound for a policy that sees the future, cheaper than exhaustive 4^T search.

Protocol per turn at depth d (same chat format and tool execution as
``ouro_nestful_multiturn.py``, ``plain`` system prompt): the conversation so far is prefilled at
all 4 loops (fresh caches; all tokens but the last), then the last prompt token and the generated
tokens run d loops, reading only loops ≤ d of the cache. So earlier turns, whatever depth produced
them, are re-encoded at full depth when the next turn starts (a "recompute" serving model; no
cache approximation). Cost is reported as mean loops per assistant turn and per generated token.
Graded like NESTful: the last executed call's result must match the gold answer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ouro_nestful_multiturn import requests  # noqa: E402
from ouro_sft_eval import SYSTEM, load_model, parse_hermes  # noqa: E402

from dht.toolcall.nestful import answers_match, run_call, tool_responses  # noqa: E402


class Dialogue:
    """A multi-turn tool-calling dialogue: prompt, turns, executed results."""

    def __init__(self, first: str):
        self.text = first  # ChatML text up to and including the current assistant header
        self.results: dict = {}
        self.n_calls = 0
        self.last: tuple[bool, object] = (False, "no calls")
        self.turns: list[dict] = []  # {"depth", "tokens", "calls"}
        self.done = False

    def clone(self) -> Dialogue:
        d = Dialogue(self.text)
        d.results = {k: dict(v) for k, v in self.results.items()}
        d.n_calls, d.last, d.done = self.n_calls, self.last, self.done
        d.turns = list(self.turns)
        return d


def gen_turn(model, tokenizer, text: str, depth: int, max_tokens: int, stop_ids) -> tuple[str, int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    model.ut_steps = 4
    caches = model.make_cache()
    model(mx.array([ids[:-1]]), caches)  # history at full depth
    model.ut_steps = depth
    logits = model(mx.array([ids[-1:]]), caches)[:, -1, :]
    out: list[int] = []
    for _ in range(max_tokens):
        tid = int(mx.argmax(logits, axis=-1).item())
        if tid in stop_ids:
            break
        out.append(tid)
        logits = model(mx.array([[tid]]), caches)[:, -1, :]
    model.ut_steps = 4
    return tokenizer.decode(out), len(out)


def step(dlg: Dialogue, model, tokenizer, depth, ex, functions, args, stop_ids) -> Dialogue:
    """Advance one assistant turn at ``depth`` (returns a new dialogue)."""
    d = dlg.clone()
    text, n_tok = gen_turn(model, tokenizer, d.text, depth, args.max_tokens, stop_ids)
    calls = [{"name": c["name"], "arguments": c["arguments"]} for c in parse_hermes(text)]
    d.turns.append({"depth": depth, "tokens": n_tok, "calls": len(calls)})
    if not calls or len([t for t in d.turns if t["calls"]]) > args.max_turns:
        d.done = True
        return d
    outcomes = []
    for c in calls:
        ok, value = run_call(c, d.n_calls, ex.tools, functions, d.results)
        d.n_calls += 1
        d.last = (ok, value)
        outcomes.append((c["name"], ok, value))
    d.text += (
        f"{text}<|im_end|>\n<|im_start|>tool\n{tool_responses(outcomes)}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return d


def correct(d: Dialogue, ex) -> bool:
    return bool(d.last[0] and answers_match(d.last[1], ex.answer))


def cost(d: Dialogue) -> dict:
    turns = d.turns or [{"depth": 4, "tokens": 1}]
    tok = sum(max(t["tokens"], 1) for t in turns)
    return {
        "loops_per_turn": sum(t["depth"] for t in turns) / len(turns),
        "loops_per_token": sum(t["depth"] * max(t["tokens"], 1) for t in turns) / tok,
        "turns": len(turns),
        "depths": [t["depth"] for t in turns],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B")
    parser.add_argument("--adapters", required=True)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    args = parser.parse_args()

    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    model, tokenizer = load_model(args.model, None, args.adapters)
    stop_ids = {0, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    examples, functions = requests(args.n)
    out = Path(f"results/ouro_percall_oracle/{Path(args.adapters).parent.name}")
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    if (out / "examples.jsonl").exists():
        done = {json.loads(x)["id"] for x in (out / "examples.jsonl").read_text().splitlines()}
    t0 = time.perf_counter()
    with open(out / "examples.jsonl", "a") as rows:
        for k, ex in enumerate(examples, 1):
            if ex.id in done:
                continue
            system = SYSTEM.format(tools=json.dumps(ex.tools, ensure_ascii=False))
            start = Dialogue(
                f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{ex.query}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            ctx = (model, tokenizer, ex, functions, args, stop_ids)

            def adv(dlg, depth, ctx=ctx):
                return step(dlg, ctx[0], ctx[1], depth, *ctx[2:])

            def finish(dlg, depth_of):
                while not dlg.done:
                    dlg = adv(dlg, depth_of(len(dlg.turns)))
                return dlg

            fixed = {}
            for depth in (1, 2, 3, 4):
                d = finish(start, lambda _t, depth=depth: depth)
                fixed[depth] = {"correct": correct(d, ex), **cost(d)}
            # outcome-greedy per-call oracle
            cur = start
            n_rollouts = 0
            while not cur.done:
                chosen = None
                for depth in (1, 2, 3, 4):
                    cand = adv(cur, depth)
                    fin = finish(cand, lambda _t: 4)
                    n_rollouts += 1
                    if correct(fin, ex):
                        chosen = cand
                        break
                cur = chosen if chosen is not None else adv(cur, 4)
            percall = {"correct": correct(cur, ex), **cost(cur)}
            rows.write(
                json.dumps(
                    {"id": ex.id, "gold_calls": len(ex.calls), "fixed": fixed, "percall": percall}
                )
                + "\n"
            )
            rows.flush()
            mx.clear_cache()
            print(f"::progress {k}/{len(examples)} requests", flush=True)
            print(
                json.dumps(
                    {"done": k, "rollouts": n_rollouts, "s": round(time.perf_counter() - t0)}
                )
            )

    data = [json.loads(x) for x in (out / "examples.jsonl").read_text().splitlines()]
    n = len(data)
    summary = {"n": n, "fixed": {}, "per_request_oracle": {}, "per_call_oracle": {}}
    for depth in ("1", "2", "3", "4"):
        summary["fixed"][depth] = {
            "accuracy": round(sum(r["fixed"][depth]["correct"] for r in data) / n, 4),
            "loops_per_token": round(
                sum(r["fixed"][depth]["loops_per_token"] for r in data) / n, 3
            ),
        }
    # per-request oracle: cheapest fixed depth that is correct, else depth 4
    pr = []
    for r in data:
        ok = [int(d) for d in ("1", "2", "3", "4") if r["fixed"][d]["correct"]]
        pick = str(min(ok)) if ok else "4"
        pr.append((bool(ok), r["fixed"][pick]["loops_per_token"]))
    summary["per_request_oracle"] = {
        "accuracy": round(sum(a for a, _ in pr) / n, 4),
        "loops_per_token": round(sum(c for _, c in pr) / n, 3),
    }
    summary["per_call_oracle"] = {
        "accuracy": round(sum(r["percall"]["correct"] for r in data) / n, 4),
        "loops_per_token": round(sum(r["percall"]["loops_per_token"] for r in data) / n, 3),
        "loops_per_turn": round(sum(r["percall"]["loops_per_turn"] for r in data) / n, 3),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
