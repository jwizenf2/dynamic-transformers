"""Per-call depth oracle vs same-budget sampling oracle for Dory 1.0 4B (reasoning off), NESTful.

    python scripts/lab/dory_percall_oracle.py --n 60

The control that decided per-step depth on Ouro (``ouro_percall_oracle.py`` v2: the per-call depth
oracle .308 vs a same-budget sampling oracle .325, i.e. the gain was search), rerun on Dory, which
is far stronger at the task (reasoning-off NESTful .27/.40/.38/.38 at 1-4 loops). Same methods and
search logic (imported from ``ouro_percall_oracle``): fixed depths 1-4, per-call depth oracle,
per-call sampling oracle (``--samples`` temperature samples at 4 loops + the greedy 4-loop turn),
best of 4 sampled 4-loop dialogues, optional escalation rules.

Dory specifics (as ``dory_nestful_multiturn.py``): NVIDIA BF16 weights through the vendored MLX
port, Dory's chat template with OpenAI-style tools and thinking disabled, Dory's XML tool calls,
NESTful execution and grading. Turn protocol ("recompute", as the Ouro oracle): the rendered
conversation is prefilled at 4 loops on fresh ``per_loop`` caches (all tokens but the last), then
the last prompt token and the generated tokens run d loops. Greedy decoding except the sampled
candidates (temperature ``--temperature``, top-p .95, seeded). Compute is also reported in
block-tokens (Dory loops only its 20 middle blocks: 8 + 20·L + 8 blocks per token): prefill at 4
loops (96 blocks) plus generated tokens at the turn's depth.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dory_nestful_multiturn as dn  # noqa: E402
import ouro_percall_oracle as opc  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402
from ouro_nestful_multiturn import requests  # noqa: E402

from dht.toolcall.dory_format import openai_tool, parse_calls  # noqa: E402
from dht.toolcall.nestful import run_call  # noqa: E402

METHODS = (
    "fixed-1",
    "fixed-2",
    "fixed-3",
    "fixed-4",
    "percall-depth",
    "percall-sample",
    "request-sample",
)


class DoryDialogue:
    """A Dory chat (messages) with executed results; same interface as ``opc.Dialogue``."""

    def __init__(self, messages: list[dict]):
        self.messages = messages
        self.results: dict = {}
        self.n_calls = 0
        self.last: tuple[bool, object] = (False, "no calls")
        self.turns: list[dict] = []
        self.prev_len = 0
        self.done = False

    def clone(self) -> DoryDialogue:
        d = DoryDialogue([dict(m) for m in self.messages])
        d.results = {k: dict(v) for k, v in self.results.items()}
        d.n_calls, d.last, d.done, d.prev_len = self.n_calls, self.last, self.done, self.prev_len
        d.turns = list(self.turns)
        return d


class DoryGenerator:
    """Turn generation cached per request by (rendered prompt, depth, sample seed)."""

    def __init__(self, model, tok, max_tokens: int, temperature: float):
        self.model, self.tok, self.max_tokens, self.temperature = (
            model,
            tok,
            max_tokens,
            temperature,
        )
        self.cache: dict = {}
        self.calls = 0

    def reset(self) -> None:
        self.cache.clear()

    def __call__(self, prompt: str, depth: int, seed: int | None) -> tuple[str, int, int]:
        key = (prompt, depth, seed)
        if key not in self.cache:
            self.cache[key] = self._gen(prompt, depth, seed)
        return self.cache[key]

    def _gen(self, prompt: str, depth: int, seed: int | None) -> tuple[str, int, int]:
        self.calls += 1
        model = self.model
        ids = self.tok.encode(prompt, add_special_tokens=False)
        model.args.n_recurrent_loops = 4
        caches = model.make_cache()
        model(mx.array([ids[:-1]]), caches)  # history at full depth
        model.args.n_recurrent_loops = depth
        logits = model(mx.array([ids[-1:]]), caches)[:, -1, :]
        if seed is None:
            sampler = make_sampler(temp=0.0)
        else:
            mx.random.seed(seed)
            sampler = make_sampler(temp=self.temperature, top_p=0.95)
        out: list[int] = []
        for _ in range(self.max_tokens):
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            tid = int(sampler(logprobs).item())
            if tid in dn.STOP_IDS:
                break
            out.append(tid)
            logits = model(mx.array([[tid]]), caches)[:, -1, :]
        model.args.n_recurrent_loops = 4
        return self.tok.decode(out), len(out), len(ids)


class DoryEnv(opc.Env):
    def __init__(self, gen: DoryGenerator, ex, functions, max_turns: int):
        super().__init__(gen, ex, functions, max_turns)
        self.tools = [openai_tool(t) for t in ex.tools]

    def step(self, dlg, depth: int, seed: int | None = None, discard=False):
        d = dlg.clone()
        prompt = self.gen.tok.apply_chat_template(
            d.messages,
            tools=self.tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        text, n_tok, n_prompt = self.gen(prompt, depth, seed)
        calls = parse_calls(text)
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
        served = [t for t in d.turns if t["calls"] and not t["discarded"]]
        if not calls or len(served) >= self.max_turns:
            d.turns.append(turn)
            d.done = True
            return d
        d.messages.append(
            {
                "role": "assistant",
                "content": text.split("<tool_call>")[0].strip(),
                "tool_calls": [{"type": "function", "function": c} for c in calls],
            }
        )
        for c in calls:
            ok, value = run_call(c, d.n_calls, self.ex.tools, self.functions, d.results)
            d.n_calls += 1
            d.last = (ok, value)
            turn["failed"] += not ok
            content = (
                (list(value) if isinstance(value, tuple) else value) if ok else {"error": value}
            )
            d.messages.append({"role": "tool", "content": json.dumps(content, ensure_ascii=False)})
        d.turns.append(turn)
        return d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--methods", nargs="+", choices=opc.METHODS, default=list(METHODS))
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--tag", default="v2")
    parser.add_argument("--cache-limit-gb", type=float, default=4.0)
    args = parser.parse_args()

    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    model, tok = dn.load()
    base_cost = opc.cost

    def cost(d) -> dict:  # Ouro's cost fields plus Dory block-tokens (prefill at 4 loops)
        out = base_cost(d)
        b4 = dn.blocks(model, 4)
        out["block_tokens"] = sum(
            b4 * t["prompt"] + dn.blocks(model, t["depth"]) * t["tokens"] for t in d.turns
        )
        out["decode_block_tokens"] = sum(
            dn.blocks(model, t["depth"]) * t["tokens"] for t in d.turns
        )
        return out

    opc.cost = cost
    examples, functions = requests(args.n, args.tasks)
    suffix = f"_{Path(args.tasks).stem}" if args.tasks else ""
    out = Path(f"results/dory_percall_oracle/{args.tag}{suffix}")
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
    gen = DoryGenerator(model, tok, args.max_tokens, args.temperature)
    seeds = list(range(1, args.samples + 1))
    t0 = time.perf_counter()
    with open(path, "a") as rows:
        for k, ex in enumerate(examples, 1):
            if ex.id in done:
                continue
            gen.reset()
            before = gen.calls
            start = DoryDialogue(
                [{"role": "system", "content": dn.SYSTEM}, {"role": "user", "content": ex.query}]
            )
            env = DoryEnv(gen, ex, functions, args.max_turns)
            res = opc.run_methods(env, start, args.methods, seeds)
            rows.write(json.dumps({"id": ex.id, "gold_calls": len(ex.calls), **res}) + "\n")
            rows.flush()
            mx.clear_cache()
            print(f"::progress {k}/{len(examples)} requests", flush=True)
            print(
                json.dumps(
                    {
                        "done": k,
                        "generations": gen.calls - before,
                        "s": round(time.perf_counter() - t0),
                    }
                ),
                flush=True,
            )

    data = [json.loads(x) for x in path.read_text().splitlines()]
    n = len(data)
    keys = ("decode_lpt", "decode_lpt_graded", "block_tokens", "decode_block_tokens")
    summary: dict = {
        "n": n,
        "model": dn.DORY,
        "revision": dn.DORY_REVISION,
        "args": vars(args),
        "methods": {},
    }
    for m in args.methods:
        rs = [r[m] for r in data if m in r]
        summary["methods"][m] = {
            "accuracy": round(sum(r["correct"] for r in rs) / len(rs), 4),
            **{k: round(sum(r[k] for r in rs) / len(rs), 3) for k in keys},
        }
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
