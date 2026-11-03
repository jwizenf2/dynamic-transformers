"""Nanbeige4.2-3B feasibility: load on MPS at several loop counts, one tool-calling prompt each.

    uv run python scripts/lab/nanbeige_check.py 2 1 3

Runs the model's own modeling code from its Hugging Face repo (``trust_remote_code=True``,
pinned revision); downloads ~8 GB on first use. Questions: does the remote code load under our
transformers version on Apple Silicon, does overriding ``config.num_loops`` (2 by default) to 1
or 3 run, and are the outputs sensible tool calls at each loop count. Prints the end of the
rendered prompt (to check the tool template) and one JSON line per loop count.
"""

from __future__ import annotations

import json
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO, REV = "Nanbeige/Nanbeige4.2-3B", "b82e54bd609793562a75cbf9337970a93369eab5"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {"arg_0": {"type": "number"}, "arg_1": {"type": "number"}},
                "required": ["arg_0", "arg_1"],
            },
        },
    }
]
QUESTION = "A box holds 12 eggs. How many eggs are in 7 boxes? Use the tools."


def main() -> None:
    tok = AutoTokenizer.from_pretrained(REPO, revision=REV, trust_remote_code=True)
    msgs = [{"role": "user", "content": QUESTION}]
    prompt = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False)
    print("PROMPT_TAIL", repr(prompt[-400:]), flush=True)
    for loops in [int(x) for x in sys.argv[1:]] or [2, 1, 3]:
        cfg = AutoConfig.from_pretrained(REPO, revision=REV, trust_remote_code=True)
        cfg.num_loops = loops
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            REPO, revision=REV, config=cfg, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        model = model.to("mps").eval()
        ids = tok(prompt, return_tensors="pt").to("mps")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=150, do_sample=False)
        text = tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=False)
        print(
            json.dumps({"loops": loops, "s": round(time.time() - t0, 1), "text": text}), flush=True
        )
        del model
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
