"""Multi-turn NESTful evaluation of the fine-tuned Ouro models (Hermes tool-response protocol).

    python scripts/lab/ouro_nestful_multiturn.py --adapters <run>/adapters.safetensors \
        --variant sequential --n 120 --steps 1 2 3 4

Why: in the single-turn NESTful eval (``ouro_sft_eval.py --task nestful``) the SFT models write
every call of a chain in one turn and fill in intermediate values they compute themselves
(0/120 outputs at d4 use labels, 31/120 use any ``$var_N.result$`` reference), so the score
measures one-shot arithmetic rather than tool chaining. The Hermes SFT data teaches the
multi-turn protocol instead: call, receive ``<tool_response>``, continue.

Protocol per request and fixed depth R (all passes of every turn use R loops):
1. ChatML system prompt = the Hermes function-calling prompt with the request's tools
   (``ouro_sft_eval.SYSTEM``); ``--variant sequential`` appends one sentence asking to make a
   call that needs another call's result only after receiving that result; ``plain`` does not.
2. Greedy decoding of an assistant turn until ``<|im_end|>`` (``--max-tokens`` per turn).
3. The turn's ``<tool_call>`` blocks are executed in order (``dht.toolcall.nestful.run_dialogue``,
   NESTful's basic functions, range-checked); the results go back as a ``tool`` turn of
   ``<tool_response>`` blocks and the next assistant turn starts.
4. Ends at the first turn without calls or after ``--max-turns`` turns with calls.
Graded like NESTful: the result of the last call made must match the gold answer.

The conversation is extended incrementally on the same KV caches (MLX's ``causal`` mask is
bottom-right aligned, so a multi-token chunk after cached tokens is exact). Request order and
subset are those of ``ouro_nestful.py`` (depth-balanced). Rows are appended and a restarted
run skips graded (id, steps) pairs.
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
from ouro_sft_eval import SYSTEM, load_done, load_model, parse_hermes  # noqa: E402

from dht.toolcall.nestful import (  # noqa: E402
    answers_match,
    basic_function_names,
    dependency_depth,
    load_functions,
    load_nestful_parquet,
    run_dialogue,
)

SEQUENTIAL = (
    "\nIf a call needs the result of another call, make the first call, wait for its result "
    "in <tool_response>, and only then make the next call."
)


def requests(n: int):
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
    return nf.ordered(pool, 0)[:n], functions


def decode(model, caches, new_ids: list[int], max_tokens: int, stop_ids: set[int]) -> list[int]:
    """Feed ``new_ids`` after what the caches hold, then greedy-decode until a stop token.

    The stop token itself is not fed, so the caller appends ``<|im_end|>`` text next.
    """
    logits = model(mx.array([new_ids]), caches)[:, -1, :]
    out = []
    for _ in range(max_tokens):
        tid = int(mx.argmax(logits, axis=-1).item())
        if tid in stop_ids:
            break
        out.append(tid)
        logits = model(mx.array([[tid]]), caches)[:, -1, :]
    return out


def make_assistant(model, tokenizer, caches, first: str, texts: list, max_tokens, stop_ids):
    """The ``run_dialogue`` callback: append the next chunk (first prompt or tool turn), decode
    one assistant turn on the shared caches, record its text, return its parsed calls."""

    def assistant(feedback: str | None) -> list[dict]:
        if feedback is None:
            chunk = first
        else:
            chunk = f"<|im_end|>\n<|im_start|>tool\n{feedback}<|im_end|>\n<|im_start|>assistant\n"
        ids = tokenizer.encode(chunk, add_special_tokens=False)
        text = tokenizer.decode(decode(model, caches, ids, max_tokens, stop_ids))
        texts.append(text)
        return [{"name": c["name"], "arguments": c["arguments"]} for c in parse_hermes(text)]

    return assistant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B")
    parser.add_argument("--adapters", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--variant", choices=["sequential", "plain"], default="sequential")
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256, help="per assistant turn")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    args = parser.parse_args()

    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    model, tokenizer = load_model(args.model, args.weights, args.adapters)
    stop_ids = {0, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    examples, functions = requests(args.n)

    trained = args.adapters or args.weights
    tag = args.tag or (Path(trained).parent.name if trained else "base")
    out = Path(f"results/ouro_nestful_mt/{tag}/{args.variant}")
    out.mkdir(parents=True, exist_ok=True)
    correct: dict[tuple[str, int], list[bool]] = defaultdict(list)
    done = load_done(out / "examples.jsonl", correct)
    t0 = time.perf_counter()
    with open(out / "examples.jsonl", "a") as rows:
        for k, e in enumerate(examples, 1):
            system = SYSTEM.format(tools=json.dumps(e.tools, ensure_ascii=False))
            if args.variant == "sequential":
                system += SEQUENTIAL
            first = (
                f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{e.query}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            for r in args.steps:
                if (str(e.id), r) in done:
                    continue
                model.ut_steps = r
                caches = model.make_cache()
                texts: list[str] = []

                assistant = make_assistant(
                    model, tokenizer, caches, first, texts, args.max_tokens, stop_ids
                )
                res = run_dialogue(assistant, e.tools, functions, args.max_turns)
                ok = bool(res["ok"] and answers_match(res["value"], e.answer))
                correct[("nestful", r)].append(ok)
                rows.write(
                    json.dumps(
                        {
                            "cat": "nestful",
                            "id": e.id,
                            "steps": r,
                            "correct": ok,
                            "gold_depth": dependency_depth(e.calls),
                            "gold_calls": len(e.calls),
                            **{key: res[key] for key in ("turns", "calls", "errors", "finished")},
                            "value": repr(res["value"])[:80],
                            "texts": [t[:400] for t in texts],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rows.flush()
                mx.clear_cache()
            print(f"::progress {k}/{len(examples)} requests", flush=True)
            if k % 10 == 0:
                print(json.dumps({"done": k, "elapsed_s": round(time.perf_counter() - t0)}))

    summary = {
        "adapters": args.adapters,
        "weights": args.weights,
        "variant": args.variant,
        "n": len(examples),
        "max_turns": args.max_turns,
        "accuracy": {
            r: round(sum(correct[("nestful", r)]) / max(1, len(correct[("nestful", r)])), 4)
            for r in args.steps
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
