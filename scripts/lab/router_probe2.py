"""Router probe v2: per-loop signals at the decision tokens of the first answer (collection).

    python scripts/lab/router_probe2.py \
        --adapters /Users/jeremywizenfeld/AI/keep/ouro-chain-depth-v3/adapters.safetensors
    python scripts/lab/router_probe2_analysis.py

Why: the router smoke (``router_smoke.py``, 2026-10-25) probed the first 8 answer tokens. In all
292 clean tasks those are the boilerplate ``<tool_call>\\n{"name": "`` and every loop's argmax
equals loop 4's on every one of them, so the probe never saw the function name or the argument
values, where loop counts disagree. This re-runs only the probe, on the same 292 tasks in the
same order (the depth labels from the router smoke are reused), with ``--k`` = 48 tokens.

Per task: the first prompt (plain Hermes system prompt + query, as ``ouro_nestful_multiturn``)
is prefilled at 4 loops, then up to ``k`` tokens are decoded greedily along the loop-4 argmax
(stopping at ``<|im_end|>`` or end of text), recording per token and loop: Ouro's exit-gate
value, next-token entropy, top-2 log-prob margin, whether the loop's argmax equals loop 4's,
and the relative state change ‖h_p − h_{p−1}‖/‖h_{p−1}‖ (as ``ouro_early_features.probe``),
plus the loop-4 token ids, the prompt length and the system-prompt length (for the cost of the
probe with and without prefix caching). Rows are appended
to ``results/router_probe2/<model>/examples.jsonl``; a restart skips done ids.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ouro_nestful_multiturn import requests  # noqa: E402
from ouro_sft_eval import SYSTEM, load_model  # noqa: E402


def probe(model, ids: list[int], k: int, stop_ids: set[int]) -> tuple[np.ndarray, list[int]]:
    """([t, 4, 5] stats, loop-4 token ids) for the first t <= k greedy tokens at 4 loops.

    Stats per token and loop: (gate, entropy, margin, agrees_with_loop4, rel_state_change);
    loop 1's state change is NaN. Same computation as ``ouro_early_features.probe`` but it stops
    at any of ``stop_ids`` and returns the tokens.
    """
    n_layers = model.cfg.num_hidden_layers
    model.ut_steps = 4
    caches = model.make_cache()
    rows, out_ids = [], []
    tokens = mx.array([ids])
    for _ in range(k):
        h = model.model.embed_tokens(tokens)
        mask = "causal" if tokens.shape[1] > 1 else None
        prev = None
        stats, argmaxes = [], []
        for step in range(4):
            for i, layer in enumerate(model.model.layers):
                h = layer(h, mask, caches[step * n_layers + i])
            h = model.model.norm(h)
            last = h[0, -1].astype(mx.float32)
            z = model.lm_head(last)
            logp = z - mx.logsumexp(z)
            top2 = mx.topk(logp, 2)
            gate = mx.sigmoid(model.model.early_exit_gate(last))[0]
            ent = -(mx.exp(logp) * logp).sum()
            change = (
                mx.array(np.nan)
                if prev is None
                else mx.linalg.norm(last - prev) / (mx.linalg.norm(prev) + 1e-6)
            )
            stats.append(mx.stack([gate, ent, mx.abs(top2[1] - top2[0]), change]))
            argmaxes.append(mx.argmax(logp))
            prev = last
        s = np.array(mx.stack(stats))  # [4, 4]
        am = [int(a.item()) for a in argmaxes]
        row = np.empty((4, 5), dtype=np.float32)
        row[:, 0:3] = s[:, 0:3]
        row[:, 3] = [float(a == am[3]) for a in am]
        row[:, 4] = s[:, 3]
        rows.append(row)
        if am[3] in stop_ids:
            break
        out_ids.append(am[3])
        tokens = mx.array([[am[3]]])
    return np.stack(rows), out_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B")
    parser.add_argument("--adapters", required=True)
    parser.add_argument("--tasks", default="results/mathqa/validation_clean_tasks.jsonl")
    parser.add_argument("--n", type=int, default=292)
    parser.add_argument("--k", type=int, default=48)
    parser.add_argument(
        "--smoke",
        default="/Users/jeremywizenfeld/AI/keep/router_smoke_v3/examples.jsonl",
        help="router smoke rows whose tasks (ids, queries, order) this probe must match",
    )
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    args = parser.parse_args()

    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    model, tok = load_model(args.model, None, args.adapters)
    stop_ids = {0, tok.convert_tokens_to_ids("<|im_end|>")}
    examples, _ = requests(args.n, args.tasks)
    smoke = [json.loads(x) for x in Path(args.smoke).read_text().splitlines()]
    assert [(e.id, e.query) for e in examples] == [(r["id"], r["query"]) for r in smoke], (
        "tasks differ from the router smoke's (rebuild of results/mathqa drifted?)"
    )
    out = Path(f"results/router_probe2/{Path(args.adapters).parent.name}")
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
            ids = tok.encode(first, add_special_tokens=False)
            stats, out_ids = probe(model, ids, args.k, stop_ids)
            system_block = f"<|im_start|>system\n{system}<|im_end|>\n"
            row = {
                "id": e.id,
                "prompt_tokens": len(ids),
                # shared prefix a server would cache once per tool library (prefix caching)
                "system_tokens": len(tok.encode(system_block, add_special_tokens=False)),
                "probe": [
                    [[None if math.isnan(x) else round(float(x), 5) for x in loop] for loop in t]
                    for t in stats.tolist()
                ],
                "token_ids": out_ids,
                "text": tok.decode(out_ids),
            }
            rows.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.flush()
            mx.clear_cache()
            print(f"::progress {j}/{len(examples)} requests", flush=True)
            if j % 20 == 0:
                print(json.dumps({"done": j, "s": round(time.perf_counter() - t0)}), flush=True)


if __name__ == "__main__":
    main()
