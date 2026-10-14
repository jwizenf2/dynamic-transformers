"""Sanity check that layer skipping really changes a loaded model's computation.

    python scripts/lab/verify_skip.py --model Qwen/Qwen3.8-27B --groups 1 4 8

Motivation: the Qwen3.8-27B headroom run (job 6cdfa60b) gave near-identical accuracy for every
single-group skip, unlike 4B/9B. This distinguishes "genuinely robust" from "skip not applied":
for a few xLAM validation prompts it reports, per architecture, the number of executed layers,
the max |Δ logit| and KL(full || skipped) of the next-token distribution at the end of the
prompt, the greedy first-token agreement, and whether the generated texts are identical.
A skip with exactly zero logit change means the skip did not take effect.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import generate, load

from dht.toolcall import mlx_routing

sys.path.insert(0, str(Path(__file__).resolve().parent))
from headroom import (  # noqa: E402
    MODEL_REVISIONS,
    WEIGHT_PATTERNS,
    encode,
    validation_examples,
)


def next_token_logprobs(model, prompt: list[int]) -> mx.array:
    logits = model(mx.array([prompt]))[0, -1].astype(mx.float32)
    return logits - mx.logsumexp(logits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--groups", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()

    path = snapshot_download(
        args.model, revision=MODEL_REVISIONS[args.model], allow_patterns=WEIGHT_PATTERNS
    )
    model, tokenizer = load(path)
    examples = validation_examples(args.n, 2000, 0)
    prompts = [encode(tokenizer, e) for e in examples]
    print(f"type={type(model).__name__} layers={len(mlx_routing.text_model(model).layers)}")

    reference = [next_token_logprobs(model, p) for p in prompts]
    ref_text = [generate(model, tokenizer, p, max_tokens=args.max_tokens) for p in prompts]
    archs = {"full": set()}
    for g in args.groups:
        archs[f"group{g}"] = mlx_routing.group_layers(model, [g])
        archs[f"attn{g}"] = mlx_routing.attention_layers(model, [g])
    for label, skipped in archs.items():
        with mlx_routing.skip_layers(model, skipped):
            executed = len(mlx_routing.text_model(model).layers)
            rows = []
            for p, ref, text in zip(prompts, reference, ref_text, strict=True):
                lp = next_token_logprobs(model, p)
                out = generate(model, tokenizer, p, max_tokens=args.max_tokens)
                rows.append(
                    {
                        "max_abs_dlogp": mx.max(mx.abs(lp - ref)).item(),
                        "kl_full_to_skip": mx.sum(mx.exp(ref) * (ref - lp)).item(),
                        "same_first_token": mx.argmax(lp).item() == mx.argmax(ref).item(),
                        "same_text": out == text,
                    }
                )
        print(json.dumps({"arch": label, "executed_layers": executed, "prompts": rows}), flush=True)


if __name__ == "__main__":
    main()
