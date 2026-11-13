"""Huginn-0125 feasibility: one NESTful first turn at several depths, with timing.

    python scripts/lab/huginn_check.py --steps 4 8 16 32

Uses the vendored code (``huginn_model.load_huginn``) and Huginn's own chat template with the
Hermes function-calling system prompt of our Ouro runs (``ouro_sft_eval.SYSTEM``) and the first
NESTful request. Huginn is a base model (instruction data in pretraining, no tool tuning), so
the question is whether it emits parseable ``<tool_call>`` JSON at all, how its output changes
with depth, and how fast fixed-depth decoding with ``StepCache`` runs (prefill and per-token).

Result on a 32 GB MacBook (2026-11-12, first NESTful request, 96 tokens): loads in 22 s; zero-shot
output improves with depth but never solves the step (4 steps: rambling prose; 8: copies the tool
list; 16 and 32: valid ``<tool_call>`` JSON with the wrong operation). Timing at 32 steps there is
dominated by swap (24 GB used); measure on the lab.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from huginn_model import StepCache, greedy, load_huginn  # noqa: E402
from ouro_nestful_multiturn import requests  # noqa: E402
from ouro_sft_eval import SYSTEM, parse_hermes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--request", type=int, default=1)
    args = parser.parse_args()
    t0 = time.time()
    model, tok = load_huginn()
    print(json.dumps({"load_s": round(time.time() - t0, 1)}), flush=True)
    examples, _ = requests(args.request + 1)
    e = examples[args.request]
    messages = [
        {"role": "system", "content": SYSTEM.format(tools=json.dumps(e.tools, ensure_ascii=False))},
        {"role": "user", "content": e.query},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(prompt, add_special_tokens=False)
    stop = {tok.convert_tokens_to_ids(t) for t in ("<|end_turn|>", "<|end_text|>")}
    print(
        json.dumps({"query": e.query, "gold": e.calls[:2], "prompt_tokens": len(ids)}), flush=True
    )
    for r in args.steps:
        t1 = time.time()
        out = greedy(model, ids, StepCache(), r, args.max_tokens, stop)
        dt = time.time() - t1
        text = tok.decode(out)
        print(
            json.dumps(
                {
                    "steps": r,
                    "s": round(dt, 1),
                    "tokens": len(out),
                    "calls": parse_hermes(text),
                    "text": text[:400],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
