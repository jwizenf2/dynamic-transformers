"""Batched serving throughput: per-request loop depth vs Ouro's per-token exit (MLX).

    python scripts/lab/ouro_serving_bench.py --batch 1 4 16 --context 1024 --new-tokens 64

Question (serving framing, docs/research/landscape_2026-10-15/README.md): adaptive depth saves
loops on paper, but in a batch every request decodes in lockstep. Do the savings survive?

Policies, each decoding ``--new-tokens`` tokens for a batch of B requests:
* ``fixed-R`` (R = 1..4): every request in the batch runs R passes per token. With the batch
  grouped by depth, this is exactly per-request depth: a server runs each depth group as its
  own batch, so per-request throughput for any depth mix follows from these timings
  (``summary.per_request_mix``).
* ``token-gate-q``: Ouro's Q-exit decided per token and per request inside the batch. A step
  stops after the pass at which *every* row has exited (lockstep), and each row emits the
  token from its own exit pass. KV caches of the passes not run at that step are filled with
  the deepest computed pass's K/V for the same layer ("best available", in the spirit of
  2610.09013), so later steps see complete caches. There is one host sync per pass to decide
  whether to continue, as any dynamic policy needs.

Prompts: real NESTful prompts (same builder as ``ouro_nestful.py``), left-cropped to
``--context`` tokens so a batch has equal lengths (no padding). Cropping removes part of the
first few-shot example; outputs are not graded here, only timed. Decode steps are forced to
``--new-tokens`` (no early stop). Prefill is timed separately and excluded from decode
throughput. KV memory grows with passes, so it is also reported.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from ouro_mlx.model import OuroConfig, OuroForCausalLM
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ouro_headroom as xh  # noqa: E402
import ouro_nestful as nf  # noqa: E402

from dht.toolcall.nestful import basic_function_names, load_nestful_parquet  # noqa: E402


def prompts(n: int) -> list[str]:
    snap = Path(
        snapshot_download(nf.NESTFUL_HF, repo_type="dataset", revision=nf.NESTFUL_HF_REVISION)
    )
    basic = basic_function_names(
        nf.checked(nf.VENDORED / "basic_functions.py", "basic_functions.py")
    )
    pool = [
        e
        for e in load_nestful_parquet(nf.checked(snap / nf.NESTFUL_PARQUET, nf.NESTFUL_PARQUET))
        if all(c["name"] in basic for c in e.calls)
    ]
    shots = json.loads(
        nf.checked(nf.VENDORED / "icl_examples.json", "icl_examples.json").read_text("utf-8")
    )
    return [nf.build_prompt(shots, e) for e in nf.ordered(pool, 0)[:n]]


def run_layers(model, h, mask, caches, step: int):
    n_layers = model.cfg.num_hidden_layers
    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask, caches[step * n_layers + i])
    return model.model.norm(h)


def prefill(model, ids: mx.array, caches, passes: int) -> mx.array:
    h = model.model.embed_tokens(ids)
    for step in range(passes):
        h = run_layers(model, h, "causal", caches, step)
    return model.lm_head(h[:, -1])


def decode_fixed(model, ids, passes: int, new_tokens: int) -> dict:
    model.ut_steps = passes
    caches = model.make_cache()
    t0 = time.perf_counter()
    logits = prefill(model, ids, caches, passes)
    mx.eval(logits)
    t1 = time.perf_counter()
    tok = mx.argmax(logits, axis=-1)[:, None]
    for _ in range(new_tokens):
        h = model.model.embed_tokens(tok)
        for step in range(passes):
            h = run_layers(model, h, None, caches, step)
        tok = mx.argmax(model.lm_head(h[:, -1]), axis=-1)[:, None]
        mx.eval(tok)
    t2 = time.perf_counter()
    return {"prefill_s": t1 - t0, "decode_s": t2 - t1, "passes_run": float(passes)}


def decode_token_gate(model, ids, q: float, new_tokens: int) -> dict:
    model.ut_steps = 4
    n_layers = model.cfg.num_hidden_layers
    caches = model.make_cache()
    t0 = time.perf_counter()
    logits = prefill(model, ids, caches, 4)
    mx.eval(logits)
    t1 = time.perf_counter()
    batch = ids.shape[0]
    tok = mx.argmax(logits, axis=-1)[:, None]
    lockstep, row_exit = [], []
    for _ in range(new_tokens):
        h = model.model.embed_tokens(tok)
        survive = mx.ones((batch,))
        total = mx.zeros((batch,))
        exited = mx.zeros((batch,), dtype=mx.bool_)
        exit_at = mx.full((batch,), 4)
        chosen = None
        last = 3
        for step in range(4):
            h = run_layers(model, h, None, caches, step)
            g = mx.sigmoid(model.model.early_exit_gate(h[:, -1]))[:, 0].astype(mx.float32)
            p = g * survive if step < 3 else survive
            total = total + p
            survive = survive * (1 - g)
            now = (total >= q) | (step == 3)
            newly = now & ~exited
            step_logits = model.lm_head(h[:, -1])
            chosen = (
                step_logits if chosen is None else mx.where(newly[:, None], step_logits, chosen)
            )
            exit_at = mx.where(newly, step + 1, exit_at)
            exited = exited | newly
            if bool(mx.all(exited).item()):  # host sync: decide whether to run another pass
                last = step
                break
        # Best-available KV fill for passes not run at this position.
        for step in range(last + 1, 4):
            for i in range(n_layers):
                src = caches[last * n_layers + i]
                off = src.offset
                caches[step * n_layers + i].update(
                    src.keys[..., off - 1 : off, :], src.values[..., off - 1 : off, :]
                )
        tok = mx.argmax(chosen, axis=-1)[:, None]
        # Evaluate the filled caches now; left lazy, each step's slice-assignment copies stay
        # alive until a later step reads them (the first run peaked at 62 vs 38 GB at b16).
        filled = [caches[s * n_layers + i] for s in range(last + 1, 4) for i in range(n_layers)]
        mx.eval(tok, exit_at, *[c.keys for c in filled], *[c.values for c in filled])
        lockstep.append(last + 1)
        row_exit.append(np.array(exit_at).mean())
    t2 = time.perf_counter()
    return {
        "prefill_s": t1 - t0,
        "decode_s": t2 - t1,
        "passes_run": float(np.mean(lockstep)),
        "mean_row_exit": float(np.mean(row_exit)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B")
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--q", type=float, nargs="+", default=[0.5, 0.8])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--weights", default=None, help="merged fine-tuned safetensors (gate included)"
    )
    parser.add_argument("--tag", default=None, help="output folder name")
    args = parser.parse_args()

    snap = Path(snapshot_download(args.model, revision=xh.OURO_REVISIONS[args.model]))
    model = OuroForCausalLM(OuroConfig.from_json(snap / "config.json"), ut_steps=4)
    model.load_weights(args.weights or str(snap / "model.safetensors"), strict=True)
    mx.eval(model.parameters())
    model.eval()
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(snap / "tokenizer.json"))
    texts = prompts(max(args.batch) * args.repeats)
    all_ids = [tokenizer(t)["input_ids"] for t in texts]
    if min(len(i) for i in all_ids) < args.context:
        raise ValueError("--context is longer than the shortest prompt")
    cropped = [i[-args.context :] for i in all_ids]

    tag = args.tag or (
        Path(args.weights).parent.name if args.weights else args.model.split("/")[-1]
    )
    out = Path(f"results/ouro_serving/{tag}/ctx{args.context}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    policies = [("fixed", r) for r in (1, 2, 3, 4)] + [("token-gate", q) for q in args.q]
    total = len(args.batch) * len(policies) * args.repeats
    done = 0
    # Repeat-major order interleaves policies, so drift from other jobs sharing the GPU
    # (the lab runs two jobs side by side) affects all policies alike.
    for rep in range(args.repeats):
        for b in args.batch:
            for kind, param in policies:
                ids = mx.array(cropped[rep * b : (rep + 1) * b])
                if kind == "fixed":
                    r = decode_fixed(model, ids, int(param), args.new_tokens)
                else:
                    r = decode_token_gate(model, ids, float(param), args.new_tokens)
                r.update(
                    {
                        "policy": f"{kind}-{param}",
                        "batch": b,
                        "repeat": rep,
                        "decode_tok_per_s": b * args.new_tokens / r["decode_s"],
                        "peak_mem_gb": mx.get_peak_memory() / 1e9,
                    }
                )
                rows.append(r)
                mx.clear_cache()
                mx.reset_peak_memory()
                done += 1
                print(f"::progress {done}/{total} runs", flush=True)
                print(
                    json.dumps(
                        {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}
                    ),
                    flush=True,
                )

    # Mean over repeats (first repeat includes warm-up; report both the mean and the last).
    summary: dict = {"args": vars(args), "runs": {}}
    for b in args.batch:
        for kind, param in policies:
            sel = [r for r in rows if r["batch"] == b and r["policy"] == f"{kind}-{param}"]
            summary["runs"][f"b{b}/{kind}-{param}"] = {
                "decode_tok_per_s_last": round(sel[-1]["decode_tok_per_s"], 2),
                "decode_tok_per_s_mean": round(
                    float(np.mean([s["decode_tok_per_s"] for s in sel])), 2
                ),
                "passes_run": round(float(np.mean([s["passes_run"] for s in sel])), 3),
                "mean_row_exit": round(
                    float(np.mean([s.get("mean_row_exit", s["passes_run"]) for s in sel])), 3
                ),
                "peak_mem_gb": round(max(s["peak_mem_gb"] for s in sel), 2),
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
