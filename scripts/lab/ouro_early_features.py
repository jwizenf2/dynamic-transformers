"""Early-decoding router features: the first K generated tokens at full depth, per pass.

    python scripts/lab/ouro_early_features.py --task xlam --split train --n 1000
    python scripts/lab/ouro_early_features.py --task xlam --split val --n 200
    python scripts/lab/ouro_early_features.py --task nestful --n 960
    python scripts/lab/ouro_early_features.py --hermes --adapters <run>/adapters.safetensors \
        --task xlam --split train --n 1000        # fine-tuned model, Hermes prompts
    python scripts/lab/ouro_early_features.py --hermes --adapters ... --task bfcl

``--hermes`` uses the prompts of ``ouro_sft_eval.py`` (Hermes system prompt, zero-shot, the same
request ids and order), so the features join its per-depth correctness rows by id; with
``--adapters`` the fine-tuned model is probed exactly (base + LoRA, as evaluated).

Why: the pass-1 prompt features (``ouro_headroom.py``) give a router that only matches fixed-3
at 78% on xLAM, and the routing literature reports that query-only features plateau
(2606.07587) while partial-trajectory signals help. This collects a "probe" from the start of
the answer: greedy decoding of the first ``--k`` tokens with all 4 passes, recording at every
token and pass

* Ouro's exit-gate value, the next-token entropy and top-2 log-prob margin,
* whether that pass's argmax token equals the pass-4 argmax,
* ‖h_p − h_{p−1}‖ / ‖h_{p−1}‖ at the token's position (relative change of the normed state).

A probe-then-commit policy would decode these K tokens at full depth and then run the rest of
the request at the chosen depth. The same request ids, prompts and orders as the per-pass
correctness runs are used (xLAM: ``ouro_headroom.py`` defaults, train = train_ids[2000:];
NESTful: ``ouro_nestful.py`` depth-balanced order), so features join the existing labels by id.
"""

from __future__ import annotations

import argparse
import json
import random
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

from dht.toolcall.data import load_xlam, split_ids  # noqa: E402
from dht.toolcall.nestful import basic_function_names, load_nestful_parquet  # noqa: E402


def xlam_prompts(split: str, start: int, n: int):
    root = Path(snapshot_download(xh.XLAM, repo_type="dataset", revision=xh.XLAM_REVISION))
    examples = {e.id: e for e in load_xlam(root / "xlam_function_calling_60k.json")}
    train_ids, val_ids = split_ids(list(examples), 2000, 0)
    pool = [examples[i] for i in train_ids[:2000] if len(examples[i].ground_truth) == 1]
    pool.sort(key=lambda e: len(json.dumps(e.functions)))
    shots = random.Random(0).sample(pool[:50], 2)
    ids = val_ids if split == "val" else train_ids[2000:]
    return [(i, xh.build_prompt(shots, examples[i])) for i in ids[start : start + n]]


def nestful_prompts(start: int, n: int):
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
    return [(e.id, nf.build_prompt(shots, e)) for e in nf.ordered(pool, 0)[start : start + n]]


def probe(model, ids: list[int], k: int) -> np.ndarray:
    """[k, 4, 5] = (gate, entropy, margin, agrees_with_pass4, rel_state_change) per token/pass.

    Rows after an early end of text are NaN.
    """
    n_layers = model.cfg.num_hidden_layers
    caches = model.make_cache()
    out = np.full((k, 4, 5), np.nan, dtype=np.float32)
    tokens = mx.array([ids])
    for t in range(k):
        h = model.model.embed_tokens(tokens)
        mask = "causal" if tokens.shape[1] > 1 else None
        prev = None
        stats, argmaxes = [], []
        for step in range(4):
            for i, layer in enumerate(model.model.layers):
                h = layer(h, mask, caches[step * n_layers + i])
            h = model.model.norm(h)
            last = h[0, -1].astype(mx.float32)
            z = model.lm_head(last)
            logp = z - mx.logsumexp(z)
            top2 = mx.topk(logp, 2)
            gate = mx.sigmoid(model.model.early_exit_gate(last))[0]
            ent = -(mx.exp(logp) * logp).sum()
            change = (
                mx.array(np.nan)
                if prev is None
                else mx.linalg.norm(last - prev) / (mx.linalg.norm(prev) + 1e-6)
            )
            stats.append(mx.stack([gate, ent, mx.abs(top2[1] - top2[0]), change]))
            argmaxes.append(mx.argmax(logp))
            prev = last
        stats = np.array(mx.stack(stats))
        am = [int(a.item()) for a in argmaxes]
        out[t, :, 0:3] = stats[:, 0:3]
        out[t, :, 3] = [float(a == am[3]) for a in am]
        out[t, :, 4] = stats[:, 3]
        if am[3] == 0:  # eos
            break
        tokens = mx.array([[am[3]]])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=["xlam", "nestful", "bfcl"], required=True)
    parser.add_argument("--hermes", action="store_true", help="ouro_sft_eval.py prompts")
    parser.add_argument("--adapters", default=None, help="fine-tuned run's adapters.safetensors")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    args = parser.parse_args()

    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    if args.hermes:
        from ouro_sft_eval import items as hermes_items
        from ouro_sft_eval import load_model

        model, tokenizer = load_model(args.model, None, args.adapters)
        spec = argparse.Namespace(task=args.task, n=args.n, xlam_split=args.split, start=args.start)
        # Token ids come ready-made (ChatML); ids are unique per task (BFCL ids include the
        # category).
        items = [(ex_id, ids) for _, ex_id, ids, _ in hermes_items(spec, tokenizer)]
        model_tag = Path(args.adapters).parent.name if args.adapters else "base-hermes"
    else:
        if args.task == "bfcl" or args.adapters:
            raise SystemExit("--task bfcl and --adapters need --hermes")
        snap = Path(snapshot_download(args.model, revision=xh.OURO_REVISIONS[args.model]))
        model = OuroForCausalLM(OuroConfig.from_json(snap / "config.json"), ut_steps=4)
        model.load_weights(str(snap / "model.safetensors"), strict=True)
        mx.eval(model.parameters())
        model.eval()
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(snap / "tokenizer.json"))
        items = (
            xlam_prompts(args.split, args.start, args.n)
            if args.task == "xlam"
            else nestful_prompts(args.start, args.n)
        )
        model_tag = args.model.split("/")[-1]
    name = args.split if args.task == "xlam" else args.task
    out = Path(
        f"results/ouro_early_features/{model_tag}/"
        f"{args.task}_{name}_{args.start}_{args.start + len(items)}"
    )
    out.mkdir(parents=True, exist_ok=True)
    ids_out, feats = [], []
    t0 = time.perf_counter()
    for j, (ex_id, prompt) in enumerate(items, 1):
        ids = prompt if isinstance(prompt, list) else tokenizer(prompt)["input_ids"]
        feats.append(probe(model, ids, args.k))
        ids_out.append(ex_id)
        mx.clear_cache()
        print(f"::progress {j}/{len(items)} requests", flush=True)
        if j % 100 == 0 or j == len(items):
            np.savez_compressed(out / "early.npz", id=np.array(ids_out), early=np.stack(feats))
            print(json.dumps({"done": j, "elapsed_s": round(time.perf_counter() - t0)}), flush=True)


if __name__ == "__main__":
    main()
