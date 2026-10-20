"""Train and evaluate a per-call loop-depth policy (REINFORCE) on multi-turn tool chains.

    python scripts/lab/ouro_depth_policy.py --adapters <run>/adapters.safetensors \
        --train-n 200 --epochs 6 --lams 0.15 0.3 0.6 --eval-n 120

Environment: the frozen fine-tuned looped model in the multi-turn Hermes protocol (as
``ouro_nestful_multiturn.py``). At each assistant turn the conversation so far is prefilled at all
4 loops with fresh caches (a "recompute" serving model); the last prompt position's per-loop
logits and exit gates are the probe features; the policy picks a depth d; the first token is the
argmax of loop d's logits and the rest of the turn decodes with d loops (reading only loops ≤ d
of the cache). The turn's tool calls are executed and fed back. Policy and REINFORCE:
``dht.toolcall.depth_policy`` (reward = correct − λ·(mean loops/token − 1)/3).

Training uses synthetic executed chains (``dht.toolcall.chains``, seed 7, disjoint from the SFT
data seeds 0/1 and from NESTful's tools); NESTful is only evaluated. Generated turns are cached
by (conversation text, depth), so repeated states cost nothing; the cache is persisted.

Evaluation (first ``--eval-n`` NESTful math requests, same order as the other NESTful scripts):
fixed depths 1–4, two hand-written schedules (``deep-early``: 4,4,2,2…; ``decreasing``:
4,3,2,1,1…), and the greedy policy for each λ. Reported: accuracy, loops per generated token,
loops per turn, turns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ouro_nestful_multiturn import requests  # noqa: E402
from ouro_sft_eval import SYSTEM, load_model, parse_hermes  # noqa: E402

from dht.toolcall import chains  # noqa: E402
from dht.toolcall.depth_policy import (  # noqa: E402
    DepthPolicy,
    context_features,
    loop_stats,
    reward,
)
from dht.toolcall.nestful import answers_match, run_call, tool_responses  # noqa: E402

FEATURE_DIM = 16 + 10


class Env:
    """Model + cache of (conversation text, depth) → (turn text, tokens) and text → features."""

    def __init__(self, model, tokenizer, max_tokens: int, cache_path: Path):
        self.model, self.tokenizer, self.max_tokens = model, tokenizer, max_tokens
        self.stop_ids = {0, tokenizer.convert_tokens_to_ids("<|im_end|>")}
        self.cache_path = cache_path
        self.turns: dict[str, dict] = {}
        self.probes: dict[str, list] = {}
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                r = json.loads(line)
                if r["kind"] == "turn":
                    self.turns[r["key"]] = {"text": r["text"], "tokens": r["tokens"]}
                else:
                    self.probes[r["key"]] = r["stats"]
        self.log = open(cache_path, "a")
        self.generated = 0

    def _prefill(self, text: str):
        """Prefill at 4 loops; return caches and per-loop (logits, gate) at the last position."""
        model = self.model
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        model.ut_steps = 4
        caches = model.make_cache()
        h = model.model.embed_tokens(mx.array([ids]))
        n_layers = model.cfg.num_hidden_layers
        logits, gates = [], []
        for step in range(4):
            for i, layer in enumerate(model.model.layers):
                h = layer(h, "causal", caches[step * n_layers + i])
            h = model.model.norm(h)
            last = h[:, -1]
            logits.append(model.lm_head(last)[0])
            gates.append(mx.sigmoid(model.model.early_exit_gate(last.astype(mx.float32)))[0, 0])
        mx.eval(*logits, *gates)
        return caches, logits, gates, len(ids)

    def probe(self, text: str) -> list:
        key = hashlib.sha1(text.encode()).hexdigest()
        if key not in self.probes:
            _, logits, gates, n = self._prefill(text)
            stats = loop_stats(
                [np.array(z.astype(mx.float32)) for z in logits], [g.item() for g in gates]
            )
            self.probes[key] = [*stats.tolist(), n]
            self.log.write(
                json.dumps({"kind": "probe", "key": key, "stats": self.probes[key]}) + "\n"
            )
            self.log.flush()
        return self.probes[key]

    def turn(self, text: str, depth: int) -> dict:
        key = hashlib.sha1(f"{depth}\x00{text}".encode()).hexdigest()
        if key not in self.turns:
            caches, logits, _, _ = self._prefill(text)
            model = self.model
            model.ut_steps = depth
            tid = int(mx.argmax(logits[depth - 1]).item())
            out: list[int] = []
            for _ in range(self.max_tokens):
                if tid in self.stop_ids:
                    break
                out.append(tid)
                nxt = model(mx.array([[tid]]), caches)[:, -1, :]
                tid = int(mx.argmax(nxt, axis=-1).item())
            model.ut_steps = 4
            self.turns[key] = {"text": self.tokenizer.decode(out), "tokens": len(out)}
            self.log.write(json.dumps({"kind": "turn", "key": key, **self.turns[key]}) + "\n")
            self.log.flush()
            self.generated += 1
            mx.clear_cache()
        return self.turns[key]


def run_episode(env: Env, task: dict, choose, max_turns: int) -> dict:
    """One dialogue; ``choose(features, turn_index)`` returns a depth."""
    text = task["first"]
    results: dict = {}
    n_calls, last, last_error = 0, (False, "no calls"), False
    states, depths, tokens = [], [], []
    for t in range(max_turns + 1):
        stats = env.probe(text)
        feats = np.concatenate(
            [
                np.array(stats[:16]),
                context_features(t, n_calls, last_error, stats[16], task["n_tools"]),
            ]
        )
        d = choose(feats, t)
        gen = env.turn(text, d)
        states.append(feats)
        depths.append(d)
        tokens.append(max(gen["tokens"], 1))
        calls = [
            {"name": c["name"], "arguments": c["arguments"]} for c in parse_hermes(gen["text"])
        ]
        if not calls or t == max_turns:
            break
        outcomes = []
        last_error = False
        for c in calls:
            ok, value = task["execute"](c, n_calls, results)
            n_calls += 1
            last = (ok, value)
            last_error |= not ok
            outcomes.append((c["name"], ok, value))
        text += (
            f"{gen['text']}<|im_end|>\n<|im_start|>tool\n{tool_responses(outcomes)}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    correct = bool(last[0] and task["grade"](last[1]))
    loops = float(np.dot(depths, tokens) / np.sum(tokens))
    return {
        "states": states,
        "depths": depths,
        "correct": correct,
        "loops": loops,
        "turns": len(depths),
    }


def synthetic_tasks(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    tasks = []
    while len(tasks) < n:
        steps = chains.make_chain(rng, rng.randint(2, 6), 0.2)
        if len(steps) < 2:
            continue
        conv = chains.render(steps, rng, SYSTEM)
        expected = steps[-1].result

        def execute(call, _i, _res):
            tool = chains.BY_NAME.get(call["name"])
            if tool is None:
                return False, f"unknown function {call['name']}"
            try:
                return True, tool.fn(**call["arguments"])
            except Exception as exc:  # noqa: BLE001 (a failed predicted call is a miss)
                return False, type(exc).__name__

        def grade(value, expected=expected):
            if isinstance(expected, float) and isinstance(value, (int, float)):
                return abs(value - expected) < 1e-6
            return value == expected

        n_tools = len(json.loads(conv[0]["value"].split("<tools>\n")[1].split("\n</tools>")[0]))
        tasks.append(
            {
                "id": f"syn_{seed}_{len(tasks)}",
                "first": f"<|im_start|>system\n{conv[0]['value']}<|im_end|>\n<|im_start|>user\n"
                f"{conv[1]['value']}<|im_end|>\n<|im_start|>assistant\n",
                "execute": execute,
                "grade": grade,
                "n_tools": n_tools,
            }
        )
    return tasks


def nestful_tasks(n: int) -> list[dict]:
    examples, functions = requests(n)
    tasks = []
    for ex in examples:
        system = SYSTEM.format(tools=json.dumps(ex.tools, ensure_ascii=False))
        tasks.append(
            {
                "id": ex.id,
                "first": f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{ex.query}"
                "<|im_end|>\n<|im_start|>assistant\n",
                "execute": lambda c, i, res, ex=ex: run_call(c, i, ex.tools, functions, res),
                "grade": lambda v, ex=ex: answers_match(v, ex.answer),
                "n_tools": len(ex.tools),
            }
        )
    return tasks


def schedule(name: str):
    seqs = {"deep-early": [4, 4, 2], "decreasing": [4, 3, 2, 1]}
    seq = seqs[name]
    return lambda _f, t: seq[min(t, len(seq) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B")
    parser.add_argument("--adapters", required=True)
    parser.add_argument("--train-n", type=int, default=200)
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--lams", type=float, nargs="+", default=[0.15, 0.3, 0.6])
    parser.add_argument("--eval-n", type=int, default=120)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    args = parser.parse_args()

    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    model, tokenizer = load_model(args.model, None, args.adapters)
    out = Path(f"results/depth_policy/{Path(args.adapters).parent.name}")
    out.mkdir(parents=True, exist_ok=True)
    env = Env(model, tokenizer, args.max_tokens, out / "cache.jsonl")
    train = synthetic_tasks(args.train_n, args.train_seed)
    t0 = time.perf_counter()

    for lam in args.lams:
        path = out / f"policy_lam{lam:g}.json"
        if path.exists():
            continue
        pol = DepthPolicy(FEATURE_DIM, seed=0)
        rng = np.random.default_rng(0)
        order = list(range(len(train)))
        for epoch in range(args.epochs):
            random.Random(epoch).shuffle(order)
            for start in range(0, len(order), args.batch):
                batch = []
                for i in order[start : start + args.batch]:

                    def choose(f, _t, pol=pol, rng=rng):
                        pol.norm.update(f)
                        return pol.act(f, rng)

                    ep = run_episode(env, train[i], choose, args.max_turns)
                    batch.append(
                        (ep["states"], ep["depths"], reward(ep["correct"], ep["loops"], lam))
                    )
                stats = pol.update(batch, args.lr, args.entropy)
                rec = {
                    "lam": lam,
                    "epoch": epoch,
                    "step": start // args.batch,
                    **stats,
                    "generated": env.generated,
                    "s": round(time.perf_counter() - t0),
                }
                with open(out / "train_log.jsonl", "a") as log:
                    log.write(json.dumps(rec) + "\n")
                print(json.dumps(rec), flush=True)
        path.write_text(json.dumps(pol.to_dict()))

    evals = nestful_tasks(args.eval_n)
    choosers = {f"fixed-{d}": (lambda _f, _t, d=d: d) for d in (1, 2, 3, 4)}
    choosers.update({n: schedule(n) for n in ("deep-early", "decreasing")})
    for lam in args.lams:
        pol = DepthPolicy.from_dict(json.loads((out / f"policy_lam{lam:g}.json").read_text()))
        choosers[f"policy-lam{lam:g}"] = lambda f, _t, pol=pol: pol.act(f)
    done = set()
    if (out / "eval.jsonl").exists():
        done = {
            (r["method"], r["id"])
            for r in map(json.loads, (out / "eval.jsonl").read_text().splitlines())
        }
    with open(out / "eval.jsonl", "a") as rows:
        for k, task in enumerate(evals, 1):
            for name, choose in choosers.items():
                if (name, task["id"]) in done:
                    continue
                ep = run_episode(env, task, choose, args.max_turns)
                rows.write(
                    json.dumps(
                        {
                            "method": name,
                            "id": task["id"],
                            "correct": ep["correct"],
                            "loops": ep["loops"],
                            "turns": ep["turns"],
                            "depths": ep["depths"],
                        }
                    )
                    + "\n"
                )
                rows.flush()
            print(f"::progress {k}/{len(evals)} eval requests", flush=True)

    data = [json.loads(x) for x in (out / "eval.jsonl").read_text().splitlines()]
    summary = {}
    for name in choosers:
        r = [x for x in data if x["method"] == name]
        if r:
            summary[name] = {
                "n": len(r),
                "accuracy": round(sum(x["correct"] for x in r) / len(r), 4),
                "loops_per_token": round(float(np.mean([x["loops"] for x in r])), 3),
                "turns": round(float(np.mean([x["turns"] for x in r])), 2),
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
