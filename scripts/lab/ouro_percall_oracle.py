"""Per-call depth oracle on multi-turn NESTful, with the controls that decide whether it is real.

    python scripts/lab/ouro_percall_oracle.py --adapters <run>/adapters.safetensors --n 120 \
        --tag v2

Go/no-go for path B (per-turn loop allocation). Version 1 (commit cdc068a, 60 requests) found a
per-call oracle at .367 vs .267 for the per-request oracle. An independent review
(docs/research/review_2026-10-21_independent.md) showed that (a) the per-call-only wins look
like search (forced depth 4 early, a late shallow turn perturbing a failing trajectory, ~5x the
rollouts), (b) "late turns are shallow" is mostly the final no-call turn, which is never graded,
and (c) loops per generated token ignore re-reading the conversation. Version 2 measures, on the
same requests and with the same chat format and tool execution as ``ouro_nestful_multiturn.py``
(``plain`` system prompt):

Methods (all greedy unless stated; "rollout" = finish the dialogue greedily at depth 4):
* ``fixed-d``: every turn at depth d (d = 1..4).
* ``percall-depth``: outcome-greedy per-call oracle (v1): per turn, try depths 1..4 in order,
  each followed by a rollout; keep the first whose rollout is correct; else depth 4.
* ``percall-sample``: the same search budget over *samples* instead of depths: per turn, try 3
  temperature samples at depth 4 then the greedy depth-4 turn, each followed by a rollout; keep
  the first correct. If it matches ``percall-depth``, per-turn depth is a noise source, not an
  allocation lever.
* ``request-sample``: best of 4 whole dialogues at depth 4 (greedy + 3 sampled), the sampling
  counterpart of the per-request oracle (best of fixed-1..4).
* ``escalate-L`` (L = 1, 2, 3): training-free per-turn policy. Each turn runs at depth L; if any
  of its tool calls fails to execute (unknown tool or argument, unresolved reference, error,
  out of range), the turn is regenerated at depth 4 and the failed attempt's cost is kept.
  ``escalate-2s`` is sticky: after the first escalation every later turn runs at 4.

Turn protocol ("recompute"): the conversation so far is prefilled at 4 loops on fresh caches
(all tokens but the last), then the last prompt token and the generated tokens run d loops.
Turns are cached per request by (prompt, depth, sample seed), so repeated rollouts are free.

Costs per method (the trajectory the method actually serves; oracle search is reported apart):
* ``decode_lpt``: loops per generated token, all turns (v1's metric);
* ``decode_lpt_graded``: the same without the final no-call turn (its text is not graded);
* ``recompute_loop_tokens``: Σ over generated turns of 4 x prompt tokens + d x generated tokens;
* ``incremental_loop_tokens``: Σ 4 x tokens new since the previous turn + d x generated tokens
  (a KV-reuse server that prefills only the appended chunk; accuracy is still measured with
  full-depth history, so this is a cost model, not an accuracy result for KV reuse).
Escalation counts the discarded low-depth attempt in every cost. Graded like NESTful: the last
executed call's result must match the gold answer. Rows are appended; a restart skips done ids.
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

METHODS = (
    "fixed-1",
    "fixed-2",
    "fixed-3",
    "fixed-4",
    "percall-depth",
    "percall-sample",
    "request-sample",
    "escalate-1",
    "escalate-2",
    "escalate-3",
    "escalate-2s",
)


class Dialogue:
    """A multi-turn tool-calling dialogue: prompt, turns, executed results."""

    def __init__(self, first: str):
        self.text = first  # ChatML text up to and including the current assistant header
        self.results: dict = {}
        self.n_calls = 0
        self.last: tuple[bool, object] = (False, "no calls")
        # per generated turn: depth, generated tokens, calls, failed calls, prompt tokens,
        # new prompt tokens since the previous turn, and whether it was a discarded attempt
        self.turns: list[dict] = []
        self.prev_len = 0
        self.done = False

    def clone(self) -> Dialogue:
        d = Dialogue(self.text)
        d.results = {k: dict(v) for k, v in self.results.items()}
        d.n_calls, d.last, d.done, d.prev_len = self.n_calls, self.last, self.done, self.prev_len
        d.turns = list(self.turns)
        return d


class Generator:
    """Turn generation with a per-request cache keyed by (prompt text, depth, sample seed)."""

    def __init__(self, model, tokenizer, max_tokens: int, stop_ids, temperature: float):
        self.model, self.tok = model, tokenizer
        self.max_tokens, self.stop_ids, self.temperature = max_tokens, stop_ids, temperature
        self.cache: dict = {}
        self.calls = 0

    def reset(self) -> None:
        self.cache.clear()

    def __call__(self, text: str, depth: int, seed: int | None) -> tuple[str, int, int]:
        key = (text, depth, seed)
        if key not in self.cache:
            self.cache[key] = self._gen(text, depth, seed)
        return self.cache[key]

    def _gen(self, text: str, depth: int, seed: int | None) -> tuple[str, int, int]:
        self.calls += 1
        model = self.model
        ids = self.tok.encode(text, add_special_tokens=False)
        model.ut_steps = 4
        caches = model.make_cache()
        model(mx.array([ids[:-1]]), caches)  # history at full depth
        model.ut_steps = depth
        logits = model(mx.array([ids[-1:]]), caches)[:, -1, :]
        key = mx.random.key(seed) if seed is not None else None
        out: list[int] = []
        for _ in range(self.max_tokens):
            if key is None:
                tid = int(mx.argmax(logits, axis=-1).item())
            else:
                key, sub = mx.random.split(key)
                tid = int(mx.random.categorical(logits / self.temperature, key=sub).item())
            if tid in self.stop_ids:
                break
            out.append(tid)
            logits = model(mx.array([[tid]]), caches)[:, -1, :]
        model.ut_steps = 4
        return self.tok.decode(out), len(out), len(ids)


class Env:
    def __init__(self, gen: Generator, ex, functions, max_turns: int):
        self.gen, self.ex, self.functions, self.max_turns = gen, ex, functions, max_turns

    def step(self, dlg: Dialogue, depth: int, seed: int | None = None, discard=False) -> Dialogue:
        """Advance one assistant turn at ``depth`` (returns a new dialogue)."""
        d = dlg.clone()
        text, n_tok, n_prompt = self.gen(d.text, depth, seed)
        calls = [{"name": c["name"], "arguments": c["arguments"]} for c in parse_hermes(text)]
        turn = {
            "depth": depth,
            "tokens": n_tok,
            "calls": len(calls),
            "failed": 0,
            "prompt": n_prompt,
            "new": n_prompt - d.prev_len,
            "discarded": discard,
        }
        d.prev_len = n_prompt + n_tok
        if (
            not calls
            or len([t for t in d.turns if t["calls"] and not t["discarded"]]) >= self.max_turns
        ):
            d.turns.append(turn)
            d.done = True
            return d
        outcomes = []
        for c in calls:
            ok, value = run_call(c, d.n_calls, self.ex.tools, self.functions, d.results)
            d.n_calls += 1
            d.last = (ok, value)
            turn["failed"] += not ok
            outcomes.append((c["name"], ok, value))
        d.turns.append(turn)
        d.text += (
            f"{text}<|im_end|>\n<|im_start|>tool\n{tool_responses(outcomes)}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return d

    def finish(self, dlg: Dialogue, depth_of, seed_of=lambda _t: None) -> Dialogue:
        while not dlg.done:
            t = len(dlg.turns)
            dlg = self.step(dlg, depth_of(t), seed_of(t))
        return dlg

    def correct(self, d: Dialogue) -> bool:
        return bool(d.last[0] and answers_match(d.last[1], self.ex.answer))


def cost(d: Dialogue) -> dict:
    served = [t for t in d.turns if not t["discarded"]]
    gen = d.turns  # every generated turn costs, including discarded escalation attempts
    graded = [t for t in served if t["calls"]] or served[:1]  # without the final no-call turn

    def lpt(ts):
        tok = sum(max(t["tokens"], 1) for t in ts)
        return sum(t["depth"] * max(t["tokens"], 1) for t in ts) / tok

    return {
        "decode_lpt": lpt(gen),
        "decode_lpt_graded": lpt([t for t in gen if t["calls"]] or gen[:1]),
        "recompute_loop_tokens": sum(4 * t["prompt"] + t["depth"] * t["tokens"] for t in gen),
        "incremental_loop_tokens": sum(4 * t["new"] + t["depth"] * t["tokens"] for t in gen),
        "generated_tokens": sum(t["tokens"] for t in gen),
        "turns": len(served),
        "depths": [t["depth"] for t in served],
        "depths_graded": [t["depth"] for t in graded],
        "escalations": sum(t["discarded"] for t in gen),
    }


def run_methods(env: Env, start: Dialogue, methods, temperature_seeds) -> dict:
    out = {}
    for depth in (1, 2, 3, 4):
        name = f"fixed-{depth}"
        if name in methods:
            final = env.finish(start, lambda _t, depth=depth: depth)
            out[name] = {"correct": env.correct(final), **cost(final)}

    def search(candidates):
        cur, rollouts = start, 0
        while not cur.done:
            chosen = None
            for depth, seed in candidates:
                cand = env.step(cur, depth, seed)
                rollouts += 1
                if env.correct(env.finish(cand, lambda _t: 4)):
                    chosen = cand
                    break
            cur = chosen if chosen is not None else env.step(cur, 4)
        return cur, rollouts

    if "percall-depth" in methods:
        d, n = search([(1, None), (2, None), (3, None), (4, None)])
        out["percall-depth"] = {"correct": env.correct(d), "rollouts": n, **cost(d)}
    if "percall-sample" in methods:
        d, n = search([(4, s) for s in temperature_seeds] + [(4, None)])
        out["percall-sample"] = {"correct": env.correct(d), "rollouts": n, **cost(d)}
    if "request-sample" in methods:
        tries = [env.finish(start, lambda _t: 4)]
        for s in temperature_seeds:
            tries.append(env.finish(start, lambda _t: 4, lambda t, s=s: s * 1000 + t))
        ok = [env.correct(x) for x in tries]
        pick = tries[ok.index(True)] if any(ok) else tries[0]
        out["request-sample"] = {"correct": any(ok), "rollouts": len(tries), **cost(pick)}
    for low, sticky, name in (
        (1, False, "escalate-1"),
        (2, False, "escalate-2"),
        (3, False, "escalate-3"),
        (2, True, "escalate-2s"),
    ):
        if name not in methods:
            continue
        cur, escalated = start, False
        while not cur.done:
            depth = 4 if (sticky and escalated) else low
            nxt = env.step(cur, depth)
            if depth < 4 and nxt.turns[-1]["failed"]:
                escalated = True
                # the retry reuses the same prompt, so only the failed attempt's decode is extra
                failed = dict(nxt.turns[-1], discarded=True, prompt=0, new=0)
                nxt = env.step(cur, 4)
                nxt.turns.insert(len(cur.turns), failed)
            cur = nxt
        out[name] = {"correct": env.correct(cur), **cost(cur)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B")
    parser.add_argument("--adapters", required=True)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--tasks", default=None, help="NESTful-format jsonl instead of NESTful")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--tag", default="v2")
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    args = parser.parse_args()

    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    model, tokenizer = load_model(args.model, None, args.adapters)
    stop_ids = {0, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    examples, functions = requests(args.n, args.tasks)
    suffix = f"_{Path(args.tasks).stem}" if args.tasks else ""
    out = Path(f"results/ouro_percall_oracle/{Path(args.adapters).parent.name}/{args.tag}{suffix}")
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
    gen = Generator(model, tokenizer, args.max_tokens, stop_ids, args.temperature)
    seeds = list(range(1, args.samples + 1))
    t0 = time.perf_counter()
    with open(path, "a") as rows:
        for k, ex in enumerate(examples, 1):
            if ex.id in done:
                continue
            gen.reset()
            calls_before = gen.calls
            system = SYSTEM.format(tools=json.dumps(ex.tools, ensure_ascii=False))
            start = Dialogue(
                f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{ex.query}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            res = run_methods(Env(gen, ex, functions, args.max_turns), start, args.methods, seeds)
            rows.write(json.dumps({"id": ex.id, "gold_calls": len(ex.calls), **res}) + "\n")
            rows.flush()
            mx.clear_cache()
            print(f"::progress {k}/{len(examples)} requests", flush=True)
            print(
                json.dumps(
                    {
                        "done": k,
                        "generations": gen.calls - calls_before,
                        "s": round(time.perf_counter() - t0),
                    }
                ),
                flush=True,
            )

    data = [json.loads(x) for x in path.read_text().splitlines()]
    n = len(data)
    keys = ("decode_lpt", "decode_lpt_graded", "recompute_loop_tokens", "incremental_loop_tokens")
    summary: dict = {"n": n, "args": vars(args), "methods": {}}
    for m in args.methods:
        rs = [r[m] for r in data if m in r]
        summary["methods"][m] = {
            "accuracy": round(sum(r["correct"] for r in rs) / len(rs), 4),
            **{k: round(sum(r[k] for r in rs) / len(rs), 3) for k in keys},
        }
    # per-request oracle over fixed depths: cheapest correct fixed depth (by decode_lpt)
    if all(f"fixed-{d}" in data[0] for d in (1, 2, 3, 4)):
        pr = []
        for r in data:
            ok = [f"fixed-{d}" for d in (1, 2, 3, 4) if r[f"fixed-{d}"]["correct"]]
            pr.append(r[ok[0]] if ok else {**r["fixed-4"], "correct": False})
        summary["methods"]["request-depth-oracle"] = {
            "accuracy": round(sum(x["correct"] for x in pr) / n, 4),
            **{k: round(sum(x[k] for x in pr) / n, 3) for k in keys},
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
