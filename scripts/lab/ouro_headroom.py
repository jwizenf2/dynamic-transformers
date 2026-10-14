"""Gate test for loop-count routing: Ouro tool-call accuracy at each loop count, per example.

    python scripts/lab/ouro_headroom.py --model ByteDance/Ouro-2.6B --n 100

Question (docs/research/findings_2026-10-14_routing_and_looped_models.md, section 6): does the
best number of loop passes differ between requests? If one fixed count is best everywhere, a
per-request bandit has nothing to learn.

Method:
* Ouro via the ouro-mlx port (pinned git dependency). Ouro's HF weight names match the port, so
  the HF ``model.safetensors`` loads directly. The same loaded model is run with
  ``ut_steps = R`` for each R in ``--steps``; each pass reuses the shared 48-layer stack.
* Ouro has no tool-call fine-tuning (the looped tool-calling paper, 2608.18171, used LoRA SFT).
  For this zero-training gate we prompt the base model few-shot with xLAM-style JSON calls,
  using ``--shots`` short examples from the xLAM TRAIN split. Absolute accuracy will be lower
  than after SFT; the gate only needs per-example depth sensitivity.
* Generations are greedy, one request at a time (the port has no batching). Output is parsed
  as the first JSON list ``[{"name": ..., "arguments": {...}}]`` and graded with the same
  BFCL-style checker as the Qwen runs. Unparsable output counts as "no call".
* Ouro's exit gate: during prefill, the sigmoid of ``early_exit_gate`` at the last prompt token
  is logged after every pass. That gives Ouro's own per-request stopping signal for later
  comparison. (Ouro applies it per token; this is a request-level proxy.)

Outputs ``results/ouro_headroom/<model>/examples.jsonl`` (one row per example x R) and
``summary.json`` with per-R accuracy and the per-example oracle: the smallest R that is
correct, compared with the best fixed R.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from ouro_mlx.model import OuroConfig, OuroForCausalLM
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from headroom import XLAM, XLAM_REVISION  # noqa: E402

from dht.toolcall.checker import check_example  # noqa: E402
from dht.toolcall.data import load_xlam, split_ids  # noqa: E402
from dht.toolcall.parser import ToolCall  # noqa: E402

OURO_REVISIONS = {
    "ByteDance/Ouro-2.6B": "1ed04250da1a9936042725d302e81c8fa2ab5abd",
}
INSTRUCTIONS = (
    "You are a function-calling assistant. Given the available tools and a user query, answer "
    "with a JSON list of the function calls that answer the query, each as "
    '{"name": ..., "arguments": {...}}. Answer [] if no tool applies.\n\n'
)


def xlam_record_text(example) -> tuple[str, str]:
    """(tools JSON, query) in xLAM's original flat format."""
    tools = []
    for f in example.functions:
        props = (f.get("parameters") or {}).get("properties") or {}
        tools.append(
            {
                "name": f["name"],
                "description": f.get("description", ""),
                "parameters": {k: {kk: vv for kk, vv in v.items()} for k, v in props.items()},
            }
        )
    return json.dumps(tools, ensure_ascii=False), example.messages[0]["content"]


def reference_calls(example) -> str:
    calls = []
    for gt in example.ground_truth:
        ((name, args),) = gt.items()
        calls.append({"name": name, "arguments": {k: v[0] for k, v in args.items() if v[0] != ""}})
    return json.dumps(calls, ensure_ascii=False)


def block(tools: str, query: str) -> str:
    return f"### Tools:\n{tools}\n### Query:\n{query}\n### Calls:\n"


def build_prompt(shots, example) -> str:
    text = INSTRUCTIONS
    for s in shots:
        tools, query = xlam_record_text(s)
        text += block(tools, query) + reference_calls(s) + "\n\n"
    tools, query = xlam_record_text(example)
    return text + block(tools, query)


def parse_calls(text: str) -> list[ToolCall]:
    """First JSON list in the output; anything unparsable means no call."""
    match = re.search(r"\[.*?\](?=\s*(?:\n|$))", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    calls = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            calls.append(ToolCall(name=item["name"], arguments=args))
    return calls


def forward_with_gates(model: OuroForCausalLM, tokens: mx.array, caches) -> tuple[mx.array, list]:
    """Same computation as OuroForCausalLM.__call__, also returning the exit-gate value after
    each pass at the last position (sigmoid of the gate logit)."""
    h = model.model.embed_tokens(tokens)
    mask = "causal" if tokens.shape[1] > 1 else None
    n_layers = model.cfg.num_hidden_layers
    gates = []
    for step in range(model.ut_steps):
        for i, layer in enumerate(model.model.layers):
            h = layer(h, mask, caches[step * n_layers + i])
        h = model.model.norm(h)
        gates.append(mx.sigmoid(model.model.early_exit_gate(h[:, -1])).item())
    return model.lm_head(h), gates


def prefill_signals(model: OuroForCausalLM, ids: list[int], passes: int) -> dict:
    """Router inputs, computed on the prompt only (no generation).

    For each pass k = 1..passes, at the last prompt token: Ouro's exit-gate value, and the
    entropy and top-2 log-probability margin of the next-token distribution. Also the hidden
    state after pass 1, mean-pooled over the prompt and at the last token (the DHT-style
    "features after the first block"). Uses a fresh cache, so it does not affect generation.
    """
    saved = model.ut_steps
    model.ut_steps = passes
    caches = model.make_cache()
    h = model.model.embed_tokens(mx.array([ids]))
    n_layers = model.cfg.num_hidden_layers
    out: dict = {"gate": [], "entropy": [], "margin": []}
    for step in range(passes):
        for i, layer in enumerate(model.model.layers):
            h = layer(h, "causal", caches[step * n_layers + i])
        h = model.model.norm(h)
        last = h[:, -1]
        z = model.lm_head(last).astype(mx.float32)[0]
        logp = z - mx.logsumexp(z)
        top2 = mx.topk(logp, 2)
        out["gate"].append(mx.sigmoid(model.model.early_exit_gate(last)).item())
        out["entropy"].append(-(mx.exp(logp) * logp).sum().item())
        out["margin"].append(abs((top2[1] - top2[0]).item()))
        if step == 0:
            out["pool1"] = np.array(h[0].astype(mx.float32).mean(axis=0), dtype=np.float16)
            out["last1"] = np.array(last[0].astype(mx.float32), dtype=np.float16)
    model.ut_steps = saved
    return out


def generate(model, tokenizer, prompt: str, max_tokens: int) -> tuple[str, int, list, float]:
    ids = tokenizer(prompt)["input_ids"]
    caches = model.make_cache()
    start = time.perf_counter()
    logits, gates = forward_with_gates(model, mx.array([ids]), caches)
    logits = logits[:, -1, :]
    out: list[int] = []
    for _ in range(max_tokens):
        token = mx.argmax(logits, axis=-1)
        tid = int(token.item())
        if tid == tokenizer.eos_token_id:
            break
        out.append(tid)
        text = tokenizer.decode(out)
        if "\n\n" in text or "###" in text:
            break
        logits = model(token.reshape(1, 1), caches)[:, -1, :]
    return tokenizer.decode(out), len(out), gates, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--start", type=int, default=0, help="offset into the split's id list")
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    args = parser.parse_args()
    if args.n < 1 or not args.steps or min(args.steps) < 1:
        parser.error("--n and --steps must be positive")

    # MLX keeps freed GPU buffers in a reuse cache. Prompt lengths differ per request, so the
    # cache kept growing (~52 GB per job on 2026-10-14, causing heavy swap with two jobs).
    # Cap it, and clear it after every request below.
    mx.set_cache_limit(args.cache_limit_gb * 1024**3)
    snap = Path(snapshot_download(args.model, revision=OURO_REVISIONS[args.model]))
    cfg = OuroConfig.from_json(snap / "config.json")
    model = OuroForCausalLM(cfg, ut_steps=max(args.steps))
    model.load_weights(str(snap / "model.safetensors"), strict=True)
    mx.eval(model.parameters())
    model.eval()
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(snap / "tokenizer.json"))
    tokenizer.eos_token_id = 0  # config.json: eos_token_id = 0

    root = Path(snapshot_download(XLAM, repo_type="dataset", revision=XLAM_REVISION))
    examples = {e.id: e for e in load_xlam(root / "xlam_function_calling_60k.json")}
    train_ids, val_ids = split_ids(list(examples), args.val_size, args.split_seed)
    # Few-shot examples: the shortest single-call training records (fixed seed order).
    pool = [examples[i] for i in train_ids[:2000] if len(examples[i].ground_truth) == 1]
    pool.sort(key=lambda e: len(json.dumps(e.functions)))
    shots = random.Random(args.split_seed).sample(pool[:50], args.shots)
    if args.split == "val":
        ids = val_ids
    else:
        # Training requests after the first 2000, which the few-shot pool is drawn from.
        ids = train_ids[2000:]
    evals = [examples[i] for i in ids[args.start : args.start + args.n]]

    tag = f"{args.split}_{args.start}_{args.start + len(evals)}"
    out = Path(f"results/ouro_headroom/{args.model.split('/')[-1]}/{tag}")
    out.mkdir(parents=True, exist_ok=True)
    per_r: dict[int, dict[str, bool]] = {r: {} for r in args.steps}
    features: dict[str, list] = {
        "id": [],
        "pool1": [],
        "last1": [],
        "gate": [],
        "entropy": [],
        "margin": [],
        "prompt_tokens": [],
        "n_tools": [],
        "n_required": [],
        "n_calls": [],
    }
    start = time.perf_counter()
    # Example-major order: every finished example has all pass counts, so a partial run
    # (preemption, timeout) still yields complete rows.
    with open(out / "examples.jsonl", "w") as rows:
        for k, e in enumerate(evals, 1):
            prompt = build_prompt(shots, e)
            ids_ = tokenizer(prompt)["input_ids"]
            sig = prefill_signals(model, ids_, max(args.steps))
            features["id"].append(e.id)
            for key in ("pool1", "last1", "gate", "entropy", "margin"):
                features[key].append(sig[key])
            features["prompt_tokens"].append(len(ids_))
            features["n_tools"].append(len(e.functions))
            features["n_required"].append(
                sum(len((f.get("parameters") or {}).get("required") or []) for f in e.functions)
            )
            features["n_calls"].append(len(e.ground_truth))
            for r in args.steps:
                model.ut_steps = r
                text, n_tok, gates, secs = generate(model, tokenizer, prompt, args.max_tokens)
                ok, why = check_example(parse_calls(text), e.functions, e.ground_truth)
                per_r[r][e.id] = ok
                rows.write(
                    json.dumps(
                        {
                            "steps": r,
                            "id": e.id,
                            "correct": ok,
                            "reason": why,
                            "out_tokens": n_tok,
                            "seconds": round(secs, 3),
                            "gates": [round(g, 4) for g in gates],
                            "text": text[:400],
                        }
                    )
                    + "\n"
                )
            rows.flush()
            mx.clear_cache()
            if k % 50 == 0 or k == len(evals):
                accs = {r: round(sum(per_r[r].values()) / k, 3) for r in args.steps}
                print(
                    json.dumps(
                        {
                            "done": k,
                            "of": len(evals),
                            "accuracy": accs,
                            "elapsed_s": round(time.perf_counter() - start),
                        }
                    ),
                    flush=True,
                )
                np.savez_compressed(
                    out / "features.npz", **{key: np.array(v) for key, v in features.items()}
                )

    # Per-example oracle: smallest R that is correct (None if no R is).
    smallest = {}
    for e in evals:
        ok_rs = [r for r in sorted(args.steps) if per_r[r][e.id]]
        smallest[e.id] = ok_rs[0] if ok_rs else None
    acc = {r: sum(per_r[r].values()) / len(evals) for r in args.steps}
    best_r = max(args.steps, key=lambda r: (acc[r], -r))
    solved = [v for v in smallest.values() if v is not None]
    histogram = {str(r): sum(1 for v in solved if v == r) for r in sorted(args.steps)}
    summary = {
        "model": args.model,
        "revision": OURO_REVISIONS[args.model],
        "split": args.split,
        "start": args.start,
        "n": len(evals),
        "accuracy_by_steps": acc,
        "best_fixed_steps": best_r,
        "oracle_accuracy": len(solved) / len(evals),
        "oracle_mean_steps_on_solved": sum(solved) / max(len(solved), 1),
        "smallest_correct_steps_histogram": histogram,
        "unsolved": len(evals) - len(solved),
        "non_monotone_examples": sum(
            1
            for e in evals
            if any(
                per_r[a][e.id] and not per_r[b][e.id]
                for a in args.steps
                for b in args.steps
                if a < b
            )
        ),
        "shots": [s.id for s in shots],
        "args": vars(args),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
