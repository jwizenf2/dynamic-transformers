"""Huginn (LoRA tool fine-tune) on multi-turn NESTful-format tasks at fixed recurrence depths.

    python scripts/lab/huginn_nestful.py \
        --adapters results/huginn_sft/huginn-chain/adapters.safetensors \
        --tasks nestful --n 480 --steps 4 8 32 64 --tag nestful

Same protocol as ``router_smoke.py`` greedy dialogues for Ouro, so the Ouro analyses
(``replication_metrics.py``, ``depth_anatomy.py``) read the rows unchanged: the plain Hermes
function-calling system prompt with the request's tools (``ouro_sft_eval.SYSTEM``), every call
executed with NESTful's functions and returned as a tool turn, at most ``--max-turns`` turns with
calls and ``--max-tokens`` tokens per turn, graded like NESTful (the last executed call's result).
Differences forced by the model: Huginn's chat template (``<|begin_header|>{role}<|end_header|>
\\n\\n{text}<|end_turn|>``, assistant role "Huginn", tool results as role "tool", exactly as
``huginn_sft.py`` renders the training dialogues); ``num_steps`` recurrence steps for every token
of the dialogue; greedy decoding with the random initial state seeded per position
(``huginn_model.greedy``), so runs are reproducible. Costs are logged as tokens (prompt and
generated); a token at r steps runs 2 + 4r + 2 transformer blocks. Huginn's rotary table has 4,096
positions: a dialogue that would exceed it ends there (unfinished, ``truncated_at_block_size``).

Rows (``results/router_smoke/<adapters parent>_<tag>/examples.jsonl``): id, query, gold_calls and
``runs["d<r>_greedy"]`` with correct, value, turns, calls, errors, finished, tokens, prompt_tokens
and the assistant turns as ``texts``. Appended per request; a restart skips done ids.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from huginn_model import StepCache, greedy, load_huginn  # noqa: E402
from huginn_sft import add_lora  # noqa: E402
from ouro_nestful_multiturn import requests  # noqa: E402
from ouro_sft_eval import SYSTEM, parse_hermes  # noqa: E402

from dht.toolcall.nestful import answers_match, run_dialogue  # noqa: E402


def load_adapted(adapters: str | None):
    model, tok = load_huginn()
    if adapters:
        cfg = json.loads((Path(adapters).parent / "config.json").read_text())
        loras = add_lora(model, cfg["rank"], cfg["alpha"])
        state = load_file(adapters)
        for k, m in loras.items():
            m.lora_a.data.copy_(state[f"{k}.lora_a"])
            m.lora_b.data.copy_(state[f"{k}.lora_b"])
        assert len(state) == 2 * len(loras), "adapter file does not match the LoRA layout"
    return model.eval(), tok


BLOCK_SIZE = 4096  # Huginn's config.block_size: length of the precomputed rotary table


def header(role: str) -> str:
    return f"<|begin_header|>{role}<|end_header|>\n\n"


def dialogue(model, tok, e, functions, r: int, args, stop_ids) -> dict:
    system = SYSTEM.format(tools=json.dumps(e.tools, ensure_ascii=False))
    first = (
        tok.bos_token
        + header("system")
        + system.strip()
        + "<|end_turn|>"
        + header("user")
        + e.query.strip()
        + "<|end_turn|>"
        + header("Huginn")
    )
    cache = StepCache()
    log = {"tokens": 0, "prompt_tokens": 0, "texts": [], "truncated_at_block_size": False}

    def assistant(feedback: str | None) -> list[dict]:
        chunk = (
            first
            if feedback is None
            else (
                "<|end_turn|>"
                + header("tool")
                + feedback.strip()
                + "<|end_turn|>"
                + header("Huginn")
            )
        )
        ids = tok.encode(chunk, add_special_tokens=False)
        # Huginn's rotary table covers BLOCK_SIZE positions; a dialogue that reaches it ends
        # here (counted unfinished, flagged), where Ouro has no such limit
        room = BLOCK_SIZE - cache.get_seq_length() - len(ids)
        if room <= 0:
            log["truncated_at_block_size"] = True
            return []
        log["prompt_tokens"] += len(ids)
        out = greedy(model, ids, cache, r, min(args.max_tokens, room - 1), stop_ids)
        log["tokens"] += len(out)
        text = tok.decode(out)
        log["texts"].append(text)
        return [{"name": c["name"], "arguments": c["arguments"]} for c in parse_hermes(text)]

    res = run_dialogue(assistant, e.tools, functions, args.max_turns)
    return {
        "correct": bool(res["ok"] and answers_match(res["value"], e.answer)),
        "value": repr(res["value"])[:80],
        **{k: res[k] for k in ("turns", "calls", "errors", "finished")},
        **log,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapters", default=None, help="huginn_sft adapters (none: base model)")
    ap.add_argument("--tasks", default="nestful")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=480)
    ap.add_argument("--steps", type=int, nargs="+", default=[4, 8, 32, 64])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    model, tok = load_adapted(args.adapters)
    stop_ids = {tok.convert_tokens_to_ids(t) for t in ("<|end_turn|>", "<|end_text|>")}
    examples, functions = requests(
        args.start + args.n, None if args.tasks == "nestful" else args.tasks
    )
    examples = examples[args.start :]
    name = Path(args.adapters).parent.name if args.adapters else "huginn-base"
    out = Path(f"results/router_smoke/{name}_{args.tag}")
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
            runs = {}
            for r in args.steps:
                runs[f"d{r}_greedy"] = dialogue(model, tok, e, functions, r, args, stop_ids)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            row = {"id": e.id, "query": e.query, "gold_calls": len(e.calls), "runs": runs}
            rows.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.flush()
            print(f"::progress {j}/{len(examples)} requests", flush=True)
            if j % 10 == 0:
                print(json.dumps({"done": j, "s": round(time.perf_counter() - t0)}), flush=True)


if __name__ == "__main__":
    main()
