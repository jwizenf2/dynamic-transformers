"""Router traces: loop-1/2 signals on every token of the actual 2-loop dialogue (collection).

    python scripts/lab/router_trace.py \
        --adapters /Users/jeremywizenfeld/AI/keep/ouro-chain-depth-v3/adapters.safetensors \
        --smoke results/router_smoke/ouro-chain-depth-v3/examples.jsonl --tag clean

Why: the escalation router decided from the first ~45 answer tokens, recorded along the 4-loop
trajectory. Phase A showed such early decisions are capped (AUC ~.77 for "4 loops helps" even with
4-sample success rates), and the errors we inspected occur later in the chain (wrong formula,
invented intermediate values). This records, from the 2-loop run a deployed router would actually
watch, signals over the whole dialogue, so routers deciding after the first tool result, after
turn t, or at the end (cascade) can be evaluated offline against the existing 4-loop outcomes.

Protocol: as ``router_smoke.py`` greedy at depth 2 (plain Hermes prompt, every pass at 2 loops,
incremental KV cache, NESTful execution and grading), with the forward pass unrolled so the
normed state after loop 1 is read too (``forward_2``; identical arithmetic to
``OuroForCausalLM.__call__``, so tokens and outcomes match ``d2_greedy`` up to MLX
nondeterminism; the analysis reports the match rate and uses each trace's own outcome as the
2-loop result, so features and label come from the same run).

Per generated token: loop-1 and loop-2 entropy, top probability, top-2 log-prob margin and exit
gate; KL(p_2 || p_1); whether the loop-1 and loop-2 argmax agree; log p_1 of the emitted token;
relative state change ‖h_2 − h_1‖/‖h_1‖; whether the token contains a digit. Per tool turn: length,
whether a call failed, and the loop-2 surprisal (−log p) of the tool-response tokens, excluding the
chat-template tokens around them (mean, max, mean over digit tokens). Per assistant turn: mean
loop-2 state and mean (h_2 − h_1) over its
tokens (``hidden/<id>.npy``, [turns, 2, H] float16). Rows are appended; a restart skips done ids.
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

from dht.toolcall.nestful import answers_match, run_dialogue  # noqa: E402


def forward_2(model, tokens: mx.array, caches) -> tuple[mx.array, mx.array]:
    """Normed states after loop 1 and loop 2 ([1, L, H] each); the model must have ut_steps 2."""
    h = model.model.embed_tokens(tokens)
    mask = "causal" if tokens.shape[1] > 1 else None
    n = model.cfg.num_hidden_layers
    states = []
    for step in range(2):
        for i, layer in enumerate(model.model.layers):
            h = layer(h, mask, caches[step * n + i])
        h = model.model.norm(h)
        states.append(h)
    return states[0], states[1]


def readout(model, h: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """For h [1, L, H]: (raw logits in model dtype, log-probs float32, exit gate), each [1, L, ...].

    lm_head runs on the 3-D tensor like ``OuroForCausalLM.__call__`` so the decoding decision
    (argmax of the raw logits) uses the same kernel and rounding as the original decode."""
    z = model.lm_head(h)
    zf = z.astype(mx.float32)
    logp = zf - mx.logsumexp(zf, axis=-1, keepdims=True)
    gate = mx.sigmoid(model.model.early_exit_gate(h.astype(mx.float32)))[..., 0]
    return z, logp, gate


def trace_one(model, tok, e, functions, args, stop_ids, has_digit):
    """One traced 2-loop dialogue: (row, per-turn hidden array [turns, 2, H] float16)."""
    system = SYSTEM.format(tools=json.dumps(e.tools, ensure_ascii=False))
    first = (
        f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{e.query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    model.ut_steps = 2
    caches = model.make_cache()
    turns: list[dict] = []
    hidden: list[np.ndarray] = []
    texts: list[str] = []
    log = {"tokens": 0, "prompt_tokens": 0}

    def assistant(feedback: str | None) -> list[dict]:
        if feedback is None:
            chunk = first
        else:
            chunk = f"<|im_end|>\n<|im_start|>tool\n{feedback}<|im_end|>\n<|im_start|>assistant\n"
        ids = tok.encode(chunk, add_special_tokens=False)
        log["prompt_tokens"] += len(ids)
        h1, h2 = forward_2(model, mx.array([ids]), caches)
        tool = None
        if feedback is not None:  # loop-2 surprisal of the tool-response tokens only
            pre = len(tok.encode("<|im_end|>\n<|im_start|>tool\n", add_special_tokens=False))
            suf = len(tok.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False))
            lo, hi = pre, max(pre + 1, len(ids) - suf)  # target positions inside the response
            _, lp, _ = readout(model, h2[:, lo - 1 : hi - 1])
            tgt = mx.array(ids[lo:hi])
            s = np.array(-mx.take_along_axis(lp[0], tgt[:, None], axis=-1)[:, 0])
            dig = has_digit[np.array(ids[lo:hi])]
            tool = {
                "len": len(ids),
                "error": int('"error"' in feedback or "error" in feedback.lower()),
                "surprisal_mean": round(float(s.mean()), 4),
                "surprisal_max": round(float(s.max()), 4),
                "surprisal_digits": round(float(s[dig].mean()), 4) if dig.any() else None,
                "surprisal": [round(float(x), 3) for x in s],
                "digit": [int(x) for x in dig],
            }
        # first decode step reads the full chunk's loop-2 states (as the original decode did,
        # lm_head over [1, L, H] then the last row); later steps are single tokens
        h1 = h1[:, -1:]
        feats = {
            k: []
            for k in (
                "ent1",
                "ent2",
                "top1",
                "top2",
                "mar1",
                "mar2",
                "gate1",
                "gate2",
                "kl21",
                "agree",
                "lp1",
                "change",
                "digit",
            )
        }
        hs2, hd = [], []
        out_ids: list[int] = []
        for _ in range(args.max_tokens):
            _, lp1, g1 = readout(model, h1)
            z2, lp2, g2 = readout(model, h2)
            lp1, g1, lp2, g2 = lp1[0, -1], g1[0, -1], lp2[0, -1], g2[0, -1]
            h2 = h2[:, -1:]
            p1, p2 = mx.exp(lp1), mx.exp(lp2)
            t1, t2 = mx.topk(lp1, 2), mx.topk(lp2, 2)
            a1, a2 = mx.argmax(lp1), mx.argmax(z2[0, -1])  # decision on raw logits
            vals = mx.stack(
                [
                    -(p1 * lp1).sum(),
                    -(p2 * lp2).sum(),
                    mx.exp(lp1.max()),
                    mx.exp(lp2.max()),
                    mx.abs(t1[1] - t1[0]),
                    mx.abs(t2[1] - t2[0]),
                    g1,
                    g2,
                    (p2 * (lp2 - lp1)).sum(),
                    (a1 == a2).astype(mx.float32),
                    lp1[a2],
                    mx.linalg.norm(h2[0, -1].astype(mx.float32) - h1[0, -1].astype(mx.float32))
                    / (mx.linalg.norm(h1[0, -1].astype(mx.float32)) + 1e-6),
                ]
            )
            v = np.array(vals)
            tid = int(a2.item())
            for key, x in zip(list(feats)[:-1], v, strict=True):
                feats[key].append(round(float(x), 4))
            feats["digit"].append(int(has_digit[tid]))
            hs2.append(np.array(h2[0, -1].astype(mx.float32)))
            hd.append(np.array((h2[0, -1] - h1[0, -1]).astype(mx.float32)))
            if tid in stop_ids:
                break
            out_ids.append(tid)
            h1, h2 = forward_2(model, mx.array([[tid]]), caches)
        log["tokens"] += len(out_ids)
        text = tok.decode(out_ids)
        texts.append(text)
        turns.append({"tool_before": tool, "n_tokens": len(out_ids), **feats})
        hidden.append(np.stack([np.mean(hs2, 0), np.mean(hd, 0)]))
        return [{"name": c["name"], "arguments": c["arguments"]} for c in parse_hermes(text)]

    res = run_dialogue(assistant, e.tools, functions, args.max_turns)
    model.ut_steps = 4
    arr = np.stack(hidden).astype(np.float16)
    assert np.isfinite(arr).all(), f"{e.id}: non-finite hidden state"
    row = {
        "id": e.id,
        "query": e.query,
        "correct": bool(res["ok"] and answers_match(res["value"], e.answer)),
        "value": repr(res["value"])[:80],
        **{k: res[k] for k in ("turns", "calls", "errors", "finished")},
        **log,
        "texts": texts,
        "turn_signals": turns,
    }
    return row, arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B")
    parser.add_argument("--adapters", required=True)
    parser.add_argument("--tasks", default="results/mathqa/validation_clean_tasks.jsonl")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=292)
    parser.add_argument("--smoke", required=True, help="router smoke rows the tasks must match")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    args = parser.parse_args()

    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    model, tok = load_model(args.model, None, args.adapters)
    stop_ids = {0, tok.convert_tokens_to_ids("<|im_end|>")}
    vocab = tok.convert_ids_to_tokens(list(range(len(tok))))
    has_digit = np.array([any(c in "0123456789" for c in (t or "")) for t in vocab])
    tasks = None if args.tasks == "nestful" else args.tasks
    examples, functions = requests(args.start + args.n, tasks)
    examples = examples[args.start :]
    smoke = {r["id"]: r for r in map(json.loads, Path(args.smoke).read_text().splitlines())}
    missing = [e.id for e in examples if e.id not in smoke]
    assert not missing, f"{len(missing)} tasks not in the router smoke rows, e.g. {missing[:3]}"
    for e in examples:  # same task as the labelled run: query and gold chain length
        r = smoke[e.id]
        assert (r["query"], r["gold_calls"]) == (e.query, len(e.calls)), f"{e.id} differs"
    out = Path(f"results/router_trace/{Path(args.adapters).parent.name}_{args.tag}")
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

    t0 = time.perf_counter()
    with open(path, "a") as rows:
        for j, e in enumerate(examples, 1):
            if e.id in done:
                continue
            row, arr = trace_one(model, tok, e, functions, args, stop_ids, has_digit)
            np.save(out / "hidden" / f"{e.id}.npy", arr)
            rows.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.flush()
            mx.clear_cache()
            print(f"::progress {j}/{len(examples)} requests", flush=True)
            if j % 20 == 0:
                print(json.dumps({"done": j, "s": round(time.perf_counter() - t0)}), flush=True)


if __name__ == "__main__":
    main()
