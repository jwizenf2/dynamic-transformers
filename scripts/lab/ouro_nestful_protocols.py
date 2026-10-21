"""Single-turn NESTful under three prompt formats, for fine-tuned Ouro (closes review gap G1).

    python scripts/lab/ouro_nestful_protocols.py --adapters <run>/adapters.safetensors \
        --n 240 --steps 2 4

Why: our Hermes-SFT models score .108-.117 at 4 loops on single-turn NESTful (``ouro_sft_eval.py
--task nestful``, zero-shot Hermes prompt), below the *base* model with NESTful's few-shot prompt
(.189 on 360, ``ouro_nestful.py``) and far below the .371 that Popescu et al. (2608.18171) report
for Ouro-2.6B after the same SFT recipe. Their NESTful prompt is not specified (our notes,
landscape 05). Before attributing the ceiling to model capacity, this measures how much of the
gap is the prompt format:

* ``hermes``: the current eval (Hermes function-calling system prompt with the request's tools,
  plus the sentence on ``"label"``/``"$var_N.result$"``; zero-shot; output ``<tool_call>`` blocks).
* ``hermes-icl``: the same, with NESTful's three ICL examples (``icl_examples.json``, not in the
  evaluation file) as earlier user/assistant turns; each demo user turn lists that example's tools
  and query, each demo answer is its gold calls as ``<tool_call>`` blocks with labels and
  ``$var`` references.
* ``raw-icl``: the plain-text few-shot prompt of ``ouro_nestful.py`` (NESTful's ICL examples in
  ``### Tools / ### Query / ### Calls`` layout, JSON list output), unchanged, no chat template.

Requests: the first ``--n`` of the depth-balanced order shared by every NESTful script. Grading:
execute the predicted sequence with NESTful's functions, compare the last value with the gold
answer (NESTful's win rate, as everywhere in this repo). Greedy decoding; ``--max-tokens`` per
answer; ``raw-icl`` also stops at a blank line or ``###`` (as the base-model probe).
Rows are appended; a restart skips graded (id, variant, steps) triples.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ouro_nestful as nf  # noqa: E402
from ouro_sft_eval import NESTFUL_EXTRA, SYSTEM, load_model, parse_hermes  # noqa: E402

from dht.toolcall.nestful import (  # noqa: E402
    answers_match,
    basic_function_names,
    dependency_depth,
    execute,
    load_functions,
    load_nestful_parquet,
    parse_sequence,
)

VARIANTS = ("hermes", "hermes-icl", "raw-icl")


def hermes_calls(calls: list[dict]) -> str:
    return "\n".join(
        "<tool_call>\n"
        + json.dumps(
            {"name": c["name"], "label": c.get("label"), "arguments": c["arguments"]},
            ensure_ascii=False,
        )
        + "\n</tool_call>"
        for c in calls
    )


def prompt_text(variant: str, shots: list[dict], e) -> str:
    if variant == "raw-icl":
        return nf.build_prompt(shots, e)
    system = SYSTEM.format(tools=json.dumps(e.tools, ensure_ascii=False)) + NESTFUL_EXTRA
    text = f"<|im_start|>system\n{system}<|im_end|>\n"
    if variant == "hermes-icl":
        for s in shots:
            user = f"Tools:\n{nf.tools_json(s['tools'])}\nQuery:\n{s['input']}"
            text += f"<|im_start|>user\n{user}<|im_end|>\n"
            text += f"<|im_start|>assistant\n{hermes_calls(s['output'])}<|im_end|>\n"
    return text + f"<|im_start|>user\n{e.query}<|im_end|>\n<|im_start|>assistant\n"


def decode(model, tokenizer, ids, passes, max_tokens, stop_ids, stop_text) -> tuple[str, int]:
    model.ut_steps = passes
    caches = model.make_cache()
    logits = model(mx.array([ids]), caches)[:, -1, :]
    out: list[int] = []
    for _ in range(max_tokens):
        tid = int(mx.argmax(logits, axis=-1).item())
        if tid in stop_ids:
            break
        out.append(tid)
        if stop_text and any(s in tokenizer.decode(out[-8:]) for s in stop_text):
            break
        logits = model(mx.array([[tid]]), caches)[:, -1, :]
    return tokenizer.decode(out), len(out)


def predicted_calls(variant: str, text: str) -> list[dict]:
    if variant == "raw-icl":
        return parse_sequence(text)
    return [
        {"name": c["name"], "label": c.get("label"), "arguments": c["arguments"]}
        for c in parse_hermes(text)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B")
    parser.add_argument("--adapters", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--steps", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    args = parser.parse_args()

    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    model, tokenizer = load_model(args.model, args.weights, args.adapters)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    snap = Path(
        snapshot_download(nf.NESTFUL_HF, repo_type="dataset", revision=nf.NESTFUL_HF_REVISION)
    )
    ffile = nf.checked(nf.VENDORED / "basic_functions.py", "basic_functions.py")
    basic, functions = basic_function_names(ffile), load_functions(ffile)
    pool = [
        e
        for e in load_nestful_parquet(nf.checked(snap / nf.NESTFUL_PARQUET, nf.NESTFUL_PARQUET))
        if all(c["name"] in basic for c in e.calls)
    ]
    shots = json.loads(
        nf.checked(nf.VENDORED / "icl_examples.json", "icl_examples.json").read_text()
    )
    evals = nf.ordered(pool, 0)[: args.n]

    trained = args.adapters or args.weights
    tag = Path(trained).parent.name if trained else "base"
    out = Path(f"results/ouro_nestful_protocols/{tag}")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "examples.jsonl"
    done = set()
    if path.exists():
        lines = path.read_text().splitlines()
        good = []
        for line in lines:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                break
            good.append(line)
            done.add((r["id"], r["variant"], r["steps"]))
        path.write_text("".join(x + "\n" for x in good))
    t0 = time.perf_counter()
    with open(path, "a") as rows:
        for k, e in enumerate(evals, 1):
            for variant in args.variants:
                ids = tokenizer.encode(prompt_text(variant, shots, e), add_special_tokens=False)
                stop_text = ("\n\n", "###") if variant == "raw-icl" else ()
                for r in args.steps:
                    if (e.id, variant, r) in done:
                        continue
                    text, n_tok = decode(
                        model, tokenizer, ids, r, args.max_tokens, {0, im_end}, stop_text
                    )
                    calls = predicted_calls(variant, text)
                    ran, value = execute(calls, e.tools, functions)
                    ok = bool(ran and answers_match(value, e.answer))
                    rows.write(
                        json.dumps(
                            {
                                "id": e.id,
                                "variant": variant,
                                "steps": r,
                                "correct": ok,
                                "depth": dependency_depth(e.calls),
                                "gold_calls": len(e.calls),
                                "n_pred_calls": len(calls),
                                "uses_refs": "$var" in text,
                                "reason": "ok" if ok else (str(value)[:80] if not ran else "wrong"),
                                "prompt_tokens": len(ids),
                                "out_tokens": n_tok,
                                "text": text[:600],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    rows.flush()
                mx.clear_cache()
            print(f"::progress {k}/{len(evals)} requests", flush=True)
            if k % 20 == 0:
                print(
                    json.dumps({"done": k, "elapsed_s": round(time.perf_counter() - t0)}),
                    flush=True,
                )

    rows_all = [json.loads(x) for x in path.read_text().splitlines()]
    acc: dict = defaultdict(lambda: defaultdict(list))
    for r in rows_all:
        acc[r["variant"]][r["steps"]].append(r)
    summary = {
        "adapters": args.adapters,
        "weights": args.weights,
        "n": len(evals),
        "accuracy": {
            v: {s: round(sum(x["correct"] for x in xs) / len(xs), 4) for s, xs in sorted(d.items())}
            for v, d in acc.items()
        },
        "uses_refs": {
            v: {
                s: round(sum(x["uses_refs"] for x in xs) / len(xs), 3)
                for s, xs in sorted(d.items())
            }
            for v, d in acc.items()
        },
        "mean_pred_calls": {
            v: {
                s: round(sum(x["n_pred_calls"] for x in xs) / len(xs), 2)
                for s, xs in sorted(d.items())
            }
            for v, d in acc.items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
