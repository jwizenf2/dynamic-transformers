"""Evaluate base or fine-tuned Ouro in the Hermes chat format at fixed depths 1..4.

    python scripts/lab/ouro_sft_eval.py --weights results/ouro_sft/<name>/model.safetensors \
        --task bfcl --n 0
    python scripts/lab/ouro_sft_eval.py --task xlam --n 200          # base weights
    python scripts/lab/ouro_sft_eval.py --weights ... --task nestful --n 360
    python scripts/lab/ouro_sft_eval.py --adapters results/ouro_sft/<name>/adapters.safetensors \
        --task bfcl                                                   # exact, unmerged

``--adapters`` rebuilds the trained model exactly: base bf16 weights, LoRA modules with the run's
rank/alpha (from its config.json) and the trained A/B (and, for the Ouro loss, the fp32 gate).
``--weights`` loads the merged export, where W + ΔW is rounded to bf16 once.

Prompt (the format the SFT data uses, NousResearch hermes-function-calling-v1): a system message
"You are a function calling AI model. ... <tools> [OpenAI-style tool JSON] </tools>" plus the
Hermes instruction to answer with ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``
blocks, then the user turn, rendered with Ouro's ChatML template. Greedy decoding until
``<|im_end|>`` or EOS, at a fixed number of passes R for the whole answer (per-request depth).

Tasks and grading:
* ``bfcl``: BFCL v3 non-live single-turn AST categories simple, multiple, parallel,
  parallel_multiple (the set 2608.18171 reports), graded with our BFCL-style checker
  (``dht.toolcall.checker``; it reproduces 835/840 BFCL answer keys, but is not the official
  ``bfcl-eval`` package). ``--n`` per category (0 = all).
* ``xlam``: the same 200 validation requests as ``ouro_headroom.py``, same checker.
* ``nestful``: the NESTful math subset in the ``ouro_nestful.py`` order; the system prompt adds
  the label/reference convention, and each tool call may carry a "label"; graded by execution.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from ouro_mlx.model import OuroConfig, OuroForCausalLM
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ouro_headroom as xh  # noqa: E402
import ouro_nestful as nf  # noqa: E402
from ouro_lora_sft import add_lora  # noqa: E402

from dht.toolcall.checker import check_example  # noqa: E402
from dht.toolcall.data import load_bfcl, load_xlam, split_ids  # noqa: E402
from dht.toolcall.nestful import (  # noqa: E402
    answers_match,
    basic_function_names,
    execute,
    load_functions,
    load_nestful_parquet,
)
from dht.toolcall.parser import ToolCall  # noqa: E402
from dht.toolcall.schema import to_openai_tool  # noqa: E402

BFCL = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
BFCL_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple")
SYSTEM = (
    "You are a function calling AI model. You are provided with function signatures within "
    "<tools> </tools> XML tags. You may call one or more functions to assist with the user "
    "query. Don't make assumptions about what values to plug into functions.\n<tools>\n{tools}\n"
    "</tools>\nFor each function call return a json object with function name and arguments "
    "within <tool_call> </tool_call> tags with the following schema:\n<tool_call>\n"
    '{{"name": <function-name>, "arguments": <args-dict>}}\n</tool_call>'
)
NESTFUL_EXTRA = (
    '\nGive each call a "label" ("$var_1", "$var_2", ...). To use the result of an earlier call '
    'as an argument, write "$var_N.result$".'
)
TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def chat(tokenizer, system: str, user: str) -> list[int]:
    text = (
        f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return tokenizer.encode(text, add_special_tokens=False)


def parse_hermes(text: str) -> list[dict]:
    calls = []
    for block in TOOL_CALL.findall(text):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {}
            calls.append({"name": obj["name"], "arguments": args, "label": obj.get("label")})
    return calls


def generate(model, ids: list[int], passes: int, max_tokens: int, stop_ids: set[int]):
    model.ut_steps = passes
    caches = model.make_cache()
    logits = model(mx.array([ids]), caches)[:, -1, :]
    out = []
    for _ in range(max_tokens):
        tid = int(mx.argmax(logits, axis=-1).item())
        if tid in stop_ids:
            break
        out.append(tid)
        logits = model(mx.array([[tid]]), caches)[:, -1, :]
    return out


def items(args, tokenizer):
    if args.task == "bfcl":
        root = Path(snapshot_download(BFCL, repo_type="dataset", revision=BFCL_REVISION))
        for cat in BFCL_CATEGORIES:
            exs = load_bfcl(root, cat)
            for e in exs[: args.n or None]:
                tools = json.dumps([to_openai_tool(f) for f in e.functions], ensure_ascii=False)
                user = "\n".join(m["content"] for m in e.messages if m["role"] == "user")

                def grade(calls, e=e):
                    tc = [ToolCall(c["name"], c["arguments"]) for c in calls]
                    return check_example(tc, e.functions, e.ground_truth)[0]

                yield cat, e.id, chat(tokenizer, SYSTEM.format(tools=tools), user), grade
    elif args.task == "xlam":
        root = Path(snapshot_download(xh.XLAM, repo_type="dataset", revision=xh.XLAM_REVISION))
        examples = {e.id: e for e in load_xlam(root / "xlam_function_calling_60k.json")}
        _, val_ids = split_ids(list(examples), 2000, 0)
        for i in val_ids[: args.n or 200]:
            e = examples[i]
            tools = json.dumps([to_openai_tool(f) for f in e.functions], ensure_ascii=False)

            def grade(calls, e=e):
                tc = [ToolCall(c["name"], c["arguments"]) for c in calls]
                return check_example(tc, e.functions, e.ground_truth)[0]

            yield (
                "xlam",
                e.id,
                chat(tokenizer, SYSTEM.format(tools=tools), e.messages[0]["content"]),
                grade,
            )
    else:
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
        for e in nf.ordered(pool, 0)[: args.n or 360]:
            tools = json.dumps(e.tools, ensure_ascii=False)
            system = SYSTEM.format(tools=tools) + NESTFUL_EXTRA

            def grade(calls, e=e):
                ran, value = execute(calls, e.tools, functions)
                return bool(ran and answers_match(value, e.answer))

            yield "nestful", e.id, chat(tokenizer, system, e.query), grade


def load_model(model_id: str, weights: str | None, adapters: str | None):
    """Base Ouro, merged ``weights``, or base + exact LoRA ``adapters``; (model, tokenizer)."""
    snap = Path(snapshot_download(model_id, revision=xh.OURO_REVISIONS[model_id]))
    model = OuroForCausalLM(OuroConfig.from_json(snap / "config.json"), ut_steps=4)
    if weights and adapters:
        raise SystemExit("give --weights or --adapters, not both")
    model.load_weights(weights or str(snap / "model.safetensors"), strict=True)
    if adapters:
        run = json.loads((Path(adapters).parent / "config.json").read_text())
        add_lora(model, run["rank"], run["alpha"], 0.0)
        tensors = mx.load(adapters)
        expected = sum(1 for k in tensors if "lora_" in k)
        if expected != 7 * 2 * model.cfg.num_hidden_layers:
            raise SystemExit(f"{adapters}: {expected} LoRA tensors, expected 7·2·layers")
        model.load_weights(list(tensors.items()), strict=False)
    mx.eval(model.parameters())
    model.eval()
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(snap / "tokenizer.json"))
    return model, tokenizer


def load_done(path: Path, correct: dict) -> set[tuple[str, int]]:
    """(id, steps) pairs already graded in ``path``; adds their results to ``correct``.

    A partly written last line (process killed mid-write) is dropped and the file truncated
    to the last complete row, so appending continues from a clean line boundary.
    """
    if not path.exists():
        return set()
    done, good, size = set(), 0, 0
    with open(path, "rb") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if not line.endswith(b"\n"):
                break
            size += len(line)
            good += 1
            key = (str(row["id"]), int(row["steps"]))
            if key not in done:
                done.add(key)
                correct[(row["cat"], key[1])].append(bool(row["correct"]))
    with open(path, "r+b") as f:
        f.truncate(size)
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B")
    parser.add_argument("--weights", default=None, help="merged safetensors (default: base)")
    parser.add_argument("--adapters", default=None, help="trainer adapters.safetensors (exact)")
    parser.add_argument("--task", choices=["bfcl", "xlam", "nestful"], required=True)
    parser.add_argument("--n", type=int, default=0, help="per category for bfcl; 0 = all")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    args = parser.parse_args()

    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    model, tokenizer = load_model(args.model, args.weights, args.adapters)
    stop_ids = {0, tokenizer.convert_tokens_to_ids("<|im_end|>")}

    trained = args.adapters or args.weights
    tag = args.tag or (Path(trained).parent.name if trained else "base")
    out = Path(f"results/ouro_sft_eval/{tag}/{args.task}")
    out.mkdir(parents=True, exist_ok=True)
    work = list(items(args, tokenizer))
    correct: dict[tuple[str, int], list[bool]] = defaultdict(list)
    # Resume: rows already in examples.jsonl (from an attempt stopped by a preemption) are
    # kept and their (id, steps) pairs skipped, so a restart does not redo finished requests.
    done = load_done(out / "examples.jsonl", correct)
    if done:
        print(json.dumps({"resumed_rows": len(done)}), flush=True)
    t0 = time.perf_counter()
    with open(out / "examples.jsonl", "a") as rows:
        for k, (cat, ex_id, ids, grade) in enumerate(work, 1):
            for r in args.steps:
                if (str(ex_id), r) in done:
                    continue
                text = tokenizer.decode(generate(model, ids, r, args.max_tokens, stop_ids))
                calls = parse_hermes(text)
                ok = grade(calls)
                correct[(cat, r)].append(ok)
                rows.write(
                    json.dumps(
                        {
                            "cat": cat,
                            "id": ex_id,
                            "steps": r,
                            "correct": ok,
                            "n_calls": len(calls),
                            "text": text[:600],
                        }
                    )
                    + "\n"
                )
            rows.flush()
            mx.clear_cache()
            print(f"::progress {k}/{len(work)} requests", flush=True)
            if k % 25 == 0:
                print(
                    json.dumps({"done": k, "elapsed_s": round(time.perf_counter() - t0)}),
                    flush=True,
                )

    cats = sorted({c for c, _ in correct})
    summary = {
        "weights": args.weights or "base",
        "adapters": args.adapters,
        "task": args.task,
        "by_category": {
            c: {r: round(sum(correct[(c, r)]) / len(correct[(c, r)]), 4) for r in args.steps}
            for c in cats
        },
        "n_by_category": {c: len(correct[(c, args.steps[0])]) for c in cats},
    }
    # BFCL "Overall" as an unweighted mean over the four categories (2608.18171 does not say).
    summary["overall_mean_of_categories"] = {
        r: round(sum(summary["by_category"][c][r] for c in cats) / len(cats), 4) for r in args.steps
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
