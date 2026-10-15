"""Ouro's own per-token exit gate during generation, on xLAM val or NESTful.

    python scripts/lab/ouro_token_gate.py --task xlam --n 200 --q 0.5 0.7 0.9
    python scripts/lab/ouro_token_gate.py --task nestful --n 360 --q 0.5 0.7 0.9

Why: our earlier gate baseline (router_analysis*.py) read the gate once, at the last prompt
token, as a request-level proxy. Ouro's actual mechanism decides per generated token, and
2608.18171 / SanSi (2610.07730) report it saving loops at near-full accuracy. This script
measures the real mechanism on our tasks, so the serving comparison uses the right baseline.

Exit rule (Ouro paper, "Q-exit"): at each token, with gate values λ_1..λ_4 (sigmoid of
``early_exit_gate`` on each pass's normed state at that position), the exit distribution is
p_k = λ_k ∏_{j<k}(1 − λ_j) for k < 4 and the remaining mass at pass 4. The token exits at the
first pass whose cumulative probability reaches q; its next token comes from that pass's logits.

KV handling: every pass is still computed for every token ("full KV"), so later tokens see a
complete cache at every pass. This gives the gate's accuracy and its *nominal* loop savings
(mean exit pass) without the cache approximations of 2610.09013; realised speed is measured
separately by the serving benchmark. Greedy decoding, same prompts, parsing and grading as
``ouro_headroom.py`` (xLAM) and ``ouro_nestful.py`` (NESTful).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from ouro_mlx.model import OuroConfig, OuroForCausalLM
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ouro_headroom as xh  # noqa: E402
import ouro_nestful as nf  # noqa: E402

from dht.toolcall.checker import check_example  # noqa: E402
from dht.toolcall.data import load_xlam, split_ids  # noqa: E402
from dht.toolcall.nestful import (  # noqa: E402
    answers_match,
    basic_function_names,
    dependency_depth,
    execute,
    load_functions,
    load_nestful_parquet,
    parse_sequence,
)


def all_passes(model: OuroForCausalLM, tokens: mx.array, caches) -> tuple[list, list]:
    """Run every pass; per pass, the last position's logits and exit-gate value."""
    h = model.model.embed_tokens(tokens)
    mask = "causal" if tokens.shape[1] > 1 else None
    n_layers = model.cfg.num_hidden_layers
    logits, gates = [], []
    for step in range(model.ut_steps):
        for i, layer in enumerate(model.model.layers):
            h = layer(h, mask, caches[step * n_layers + i])
        h = model.model.norm(h)
        last = h[:, -1]
        logits.append(model.lm_head(last))
        gates.append(mx.sigmoid(model.model.early_exit_gate(last))[0, 0])
    return logits, gates


def exit_pass(gates: list[float], q: float) -> int:
    """1-based exit pass under Ouro's Q-exit rule."""
    survive, total = 1.0, 0.0
    for k, g in enumerate(gates):
        total += g * survive if k < len(gates) - 1 else survive
        if total >= q:
            return k + 1
        survive *= 1 - g
    return len(gates)


def generate_token_exit(model, tokenizer, prompt: str, max_tokens: int, q: float):
    ids = tokenizer(prompt)["input_ids"]
    caches = model.make_cache()
    start = time.perf_counter()
    logits, gates = all_passes(model, mx.array([ids]), caches)
    out: list[int] = []
    exits: list[int] = []
    for _ in range(max_tokens):
        k = exit_pass([g.item() for g in gates], q)
        tid = int(mx.argmax(logits[k - 1], axis=-1).item())
        if tid == tokenizer.eos_token_id:
            break
        out.append(tid)
        exits.append(k)
        text = tokenizer.decode(out)
        if "\n\n" in text or "###" in text:
            break
        logits, gates = all_passes(model, mx.array([[tid]]), caches)
    return tokenizer.decode(out), exits, time.perf_counter() - start


def xlam_items(args):
    root = Path(snapshot_download(xh.XLAM, repo_type="dataset", revision=xh.XLAM_REVISION))
    examples = {e.id: e for e in load_xlam(root / "xlam_function_calling_60k.json")}
    train_ids, val_ids = split_ids(list(examples), 2000, 0)
    pool = [examples[i] for i in train_ids[:2000] if len(examples[i].ground_truth) == 1]
    pool.sort(key=lambda e: len(json.dumps(e.functions)))
    shots = random.Random(0).sample(pool[:50], 2)  # identical to ouro_headroom.py defaults
    for e in [examples[i] for i in val_ids[args.start : args.start + args.n]]:

        def grade(text, e=e):
            return check_example(xh.parse_calls(text), e.functions, e.ground_truth)[0]

        yield e.id, xh.build_prompt(shots, e), grade, {"n_calls": len(e.ground_truth)}


def nestful_items(args):
    snap = Path(
        snapshot_download(nf.NESTFUL_HF, repo_type="dataset", revision=nf.NESTFUL_HF_REVISION)
    )
    functions_file = nf.checked(nf.VENDORED / "basic_functions.py", "basic_functions.py")
    basic = basic_function_names(functions_file)
    functions = load_functions(functions_file)
    pool = [
        e
        for e in load_nestful_parquet(nf.checked(snap / nf.NESTFUL_PARQUET, nf.NESTFUL_PARQUET))
        if all(c["name"] in basic for c in e.calls)
    ]
    shots_file = nf.checked(nf.VENDORED / "icl_examples.json", "icl_examples.json")
    shots = json.loads(shots_file.read_text(encoding="utf-8"))
    for e in nf.ordered(pool, 0)[args.start : args.start + args.n]:

        def grade(text, e=e):
            ran, value = execute(parse_sequence(text), e.tools, functions)
            return bool(ran and answers_match(value, e.answer))

        info = {"depth": dependency_depth(e.calls), "n_calls": len(e.calls)}
        yield e.id, nf.build_prompt(shots, e), grade, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=["xlam", "nestful"], required=True)
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--q", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    args = parser.parse_args()
    max_tokens = args.max_tokens or (160 if args.task == "xlam" else 400)

    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    snap = Path(snapshot_download(args.model, revision=xh.OURO_REVISIONS[args.model]))
    model = OuroForCausalLM(OuroConfig.from_json(snap / "config.json"), ut_steps=4)
    model.load_weights(str(snap / "model.safetensors"), strict=True)
    mx.eval(model.parameters())
    model.eval()
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(snap / "tokenizer.json"))
    tokenizer.eos_token_id = 0

    items = list((xlam_items if args.task == "xlam" else nestful_items)(args))
    tag = f"{args.task}_{args.start}_{args.start + len(items)}"
    out = Path(f"results/ouro_token_gate/{args.model.split('/')[-1]}/{tag}")
    out.mkdir(parents=True, exist_ok=True)
    stats: dict[float, list] = defaultdict(list)
    t0 = time.perf_counter()
    with open(out / "examples.jsonl", "w") as rows:
        for k, (ex_id, prompt, grade, info) in enumerate(items, 1):
            for q in args.q:
                text, exits, secs = generate_token_exit(model, tokenizer, prompt, max_tokens, q)
                ok = grade(text)
                mean_exit = float(np.mean(exits)) if exits else 0.0
                stats[q].append((ok, mean_exit, len(exits), info))
                rows.write(
                    json.dumps(
                        {
                            "q": q,
                            "id": ex_id,
                            "correct": ok,
                            "mean_exit": round(mean_exit, 4),
                            "exit_hist": np.bincount(exits, minlength=5)[1:].tolist(),
                            "out_tokens": len(exits),
                            "seconds": round(secs, 3),
                            "text": text[:600],
                            **info,
                        }
                    )
                    + "\n"
                )
            rows.flush()
            mx.clear_cache()
            print(f"::progress {k}/{len(items)} requests", flush=True)
            if k % 25 == 0 or k == len(items):
                print(json.dumps({"done": k, "elapsed_s": round(time.perf_counter() - t0)}))
                for q in args.q:
                    acc = np.mean([s[0] for s in stats[q]])
                    print(json.dumps({"q": q, "accuracy": round(float(acc), 3)}), flush=True)

    def summarize(rows_: list) -> dict:
        ok = np.array([r[0] for r in rows_], float)
        mean_exit = np.array([r[1] for r in rows_])
        n_tok = np.array([r[2] for r in rows_], float)
        return {
            "n": len(rows_),
            "accuracy": round(float(ok.mean()), 3),
            "mean_exit_per_request": round(float(mean_exit.mean()), 3),
            "mean_exit_per_token": round(float((mean_exit * n_tok).sum() / max(n_tok.sum(), 1)), 3),
        }

    summary: dict = {"model": args.model, "task": args.task, "args": vars(args), "by_q": {}}
    for q in args.q:
        entry = summarize(stats[q])
        if args.task == "nestful":
            by = defaultdict(list)
            for r in stats[q]:
                by[nf.depth_bucket(r[3]["depth"])].append(r)
            entry["by_depth"] = {b: summarize(v) for b, v in sorted(by.items())}
        summary["by_q"][str(q)] = entry
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
