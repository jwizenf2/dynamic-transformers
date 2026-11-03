"""Loop-drift edits: repair Ouro beyond its trained depth by editing the state between loops.

    python scripts/lab/loop_edit.py collect --adapters A --n 150      # dev states, MathQA clean
    python scripts/lab/loop_edit.py directions                         # drift statistics + PCs
    python scripts/lab/loop_edit.py screen --adapters A                # teacher-forced proxy

Motivation: chain v3 is trained with 4 loops; NESTful accuracy is flat at 5–6 loops and collapses
at 8 (.50 → .20, corrupted JSON keys such as "asigns"; ``depth_extrapolation.py``). Following the
idea of locating a failure mode in one direction of the residual state and projecting it out
(directional ablation; e.g. the weight-space-programming report on Qwen3.5-0.8B looping, which we
treat as unverified), we test whether the collapse is a drift of the recurrent state along a few
directions that grow after loop 4, and whether removing that drift lets loops 5–8 run without
breaking the output.

Ouro's forward (``OuroForCausalLM.__call__``): h ← embed; for each loop t: h ← layers(h) with
loop t's KV caches, h ← norm(h) (the normed state is the next loop's input and the last loop's
readout). ``EditedOuro`` reproduces it exactly and, after the norm of every loop t > ``after``
(1-based; default 4), replaces h_t by an edit relative to the same positions' loop-4 state h_4:

* ``project``: h_t ← h_t − ((h_t − h_4) V^T) V, removing the drift along the rows of V
  ([k, H], orthonormal);
* ``damp``: h_t ← h_4 + β (h_t − h_4), a direction-free control (β = 0 freezes at loop 4).

Loops 1–4 are untouched, so depths <= 4 are unchanged by construction. The edit is an inference-
time intervention conditioned on the loop index; it is not a weight edit (loops share weights,
so a weight edit would also change loops 1–4).

``collect``: for the first ``--n`` clean MathQA-val tasks (development data, depth-balanced
order), the first prompt is prefilled at 8 loops and ``--k`` tokens are decoded greedily along
the loop-4 argmax (so every loop sees the same tokens), saving per token the normed state after
each loop ([k, 8, H] float16) and each loop's argmax ([k, 8]). ``directions``: drift
d_t = h_t − h_4 for t = 5..8 on the first 100 tasks: norms, cosine to h_4, share of tokens whose
argmax changes from loop 4's, by token type (structure / function name / number); PCA of the
drift; a contrast direction (mean drift at tokens whose loop-8 argmax differs minus where it
does not); saves ``directions.npz``. ``screen``: on the remaining tasks, teacher-forced 8-loop
pass over prompt + the loop-4 draft for each edit variant; per token type, the share of tokens
whose loop-6 and loop-8 argmax differs from loop 4's (structure changes are the collapse
signature; function-name and number changes are where extra loops could still help).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path("results/loop_edit")


class EditedOuro:
    """Drop-in for ``OuroForCausalLM`` (``__call__``, ``make_cache``, ``ut_steps``)."""

    def __init__(self, model, kind: str = "none", V=None, beta: float = 1.0, after: int = 4):
        self.base, self.kind, self.beta, self.after = model, kind, beta, after
        self.V = None if V is None else mx.array(np.asarray(V, np.float32))
        self.cfg, self.model, self.lm_head = model.cfg, model.model, model.lm_head

    @property
    def ut_steps(self) -> int:
        return self.base.ut_steps

    @ut_steps.setter
    def ut_steps(self, v: int) -> None:
        self.base.ut_steps = v

    def make_cache(self):
        return self.base.make_cache()

    def edit(self, h, h4):
        if self.kind == "project":
            hf, d = h.astype(mx.float32), (h - h4).astype(mx.float32)
            return (hf - (d @ self.V.T) @ self.V).astype(h.dtype)
        if self.kind == "damp":
            return (h4 + self.beta * (h - h4)).astype(h.dtype)
        return h

    def states(self, tokens: mx.array, caches) -> list[mx.array]:
        """Normed (and, after loop ``after``, edited) state after every loop."""
        h = self.model.embed_tokens(tokens)
        mask = "causal" if tokens.shape[1] > 1 else None
        n = self.cfg.num_hidden_layers
        out, h4 = [], None
        for step in range(self.ut_steps):
            for i, layer in enumerate(self.model.layers):
                h = layer(h, mask, caches[step * n + i])
            h = self.model.norm(h)
            if step + 1 == self.after:
                h4 = h
            elif step + 1 > self.after and self.kind != "none":
                h = self.edit(h, h4)
            out.append(h)
        return out

    def __call__(self, tokens: mx.array, caches) -> mx.array:
        return self.lm_head(self.states(tokens, caches)[-1])


def token_types(tok, ids: list[int]) -> list[str]:
    text, spans = "", []
    for t in ids:
        piece = tok.decode([t])
        spans.append((len(text), len(text) + len(piece)))
        text += piece
    names = [m.span(1) for m in re.finditer(r'"name":\s*"([^"]*)"', text)]
    nums = [m.span() for m in re.finditer(r"(?<=[:\s\[,])-?\d+(?:\.\d+)?", text)]
    out = []
    for a, b in spans:
        if any(a < e and b > s for s, e in names):
            out.append("function_name")
        elif any(a < e and b > s for s, e in nums):
            out.append("number")
        elif not text[a:b].strip() or re.fullmatch(r'[\s{}\[\]":,<>/_a-z]*', text[a:b]):
            out.append("structure")
        else:
            out.append("other")
    return out


def first_prompts(n: int, tok, tasks: str):
    from ouro_nestful_multiturn import requests
    from ouro_sft_eval import SYSTEM

    examples, _ = requests(n, tasks)
    for e in examples:
        system = SYSTEM.format(tools=json.dumps(e.tools, ensure_ascii=False))
        first = (
            f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{e.query}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        yield e.id, tok.encode(first, add_special_tokens=False)


def collect(args) -> None:
    from ouro_sft_eval import load_model

    model, tok = load_model(args.model, None, args.adapters)
    em = EditedOuro(model)
    em.ut_steps = 8
    stop = {0, tok.convert_tokens_to_ids("<|im_end|>")}
    (OUT / "states").mkdir(parents=True, exist_ok=True)
    meta = []
    for j, (tid, ids) in enumerate(first_prompts(args.n, tok, args.tasks), 1):
        caches = em.make_cache()
        hs = em.states(mx.array([ids]), caches)
        per_tok, arg, draft = [], [], []
        for _ in range(args.k):
            last = [h[0, -1] for h in hs]
            am = [int(mx.argmax(em.lm_head(x)).item()) for x in last]
            per_tok.append(np.stack([np.array(x.astype(mx.float32)) for x in last]))
            arg.append(am)
            if am[3] in stop:
                break
            draft.append(am[3])
            hs = em.states(mx.array([[am[3]]]), caches)
        np.save(OUT / "states" / f"{tid}.npy", np.stack(per_tok).astype(np.float16))
        meta.append({"id": tid, "prompt": ids, "draft": draft, "argmax": arg})
        mx.clear_cache()
        print(f"::progress {j}/{args.n} prompts", flush=True)
    (OUT / "meta.json").write_text(json.dumps(meta))


def directions(args) -> None:
    tok = tokenizer()
    meta = json.loads((OUT / "meta.json").read_text())
    dev = meta[: args.dev]
    drift, changed8, kinds = [], [], []
    stats = {t: {"norm": [], "cos": [], "changed": []} for t in range(5, 9)}
    for m in dev:
        s = np.load(OUT / "states" / f"{m['id']}.npy").astype(np.float32)  # [k, 8, H]
        a = np.array(m["argmax"])
        types = token_types(tok, m["draft"]) + ["structure"]
        for t in range(5, 9):
            d = s[:, t - 1] - s[:, 3]
            stats[t]["norm"] += list(np.linalg.norm(d, axis=1) / np.linalg.norm(s[:, 3], axis=1))
            cos = (s[:, t - 1] * s[:, 3]).sum(1) / (
                np.linalg.norm(s[:, t - 1], axis=1) * np.linalg.norm(s[:, 3], axis=1)
            )
            stats[t]["cos"] += list(cos)
            stats[t]["changed"] += list(a[:, t - 1] != a[:, 3])
            drift.append(d)
        changed8 += list(a[:, 7] != a[:, 3])
        kinds += types[: len(a)]
    D = np.concatenate(drift)  # rows: (token, loop) pairs for loops 5..8
    mu = D.mean(0)
    _, sv, vt = np.linalg.svd(D - mu, full_matrices=False)
    var = sv**2 / (sv**2).sum()
    d8 = np.concatenate(
        [
            np.load(OUT / "states" / f"{m['id']}.npy").astype(np.float32)[:, 7]
            - np.load(OUT / "states" / f"{m['id']}.npy").astype(np.float32)[:, 3]
            for m in dev
        ]
    )
    ch, kinds = np.array(changed8), np.array(kinds)
    contrast = d8[ch].mean(0) - d8[~ch].mean(0)
    contrast /= np.linalg.norm(contrast) + 1e-9
    mean_dir = mu / (np.linalg.norm(mu) + 1e-9)
    np.savez(OUT / "directions.npz", pcs=vt[:32], contrast=contrast, mean=mean_dir)
    report = {
        "n_dev_prompts": len(dev),
        "n_tokens": int(len(ch)),
        "by_loop": {
            t: {
                "rel_drift_from_loop4": float(np.mean(v["norm"])),
                "cos_to_loop4": float(np.mean(v["cos"])),
                "argmax_changed_vs_loop4": float(np.mean(v["changed"])),
            }
            for t, v in stats.items()
        },
        "loop8_changed_by_type": {
            k: float(ch[kinds == k].mean()) for k in sorted(set(kinds)) if (kinds == k).any()
        },
        "drift_pca_explained": [float(x) for x in var[:16]],
        "drift_pca_cumulative": {k: float(var[:k].sum()) for k in (1, 4, 16, 32)},
        "mean_drift_share_of_norm": float(np.linalg.norm(mu) / np.mean(np.linalg.norm(D, axis=1))),
    }
    (OUT / "directions.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def tokenizer():
    from transformers import PreTrainedTokenizerFast

    snap = next(Path.home().glob(".cache/huggingface/hub/models--ByteDance--Ouro-2.6B/snapshots/*"))
    return PreTrainedTokenizerFast(tokenizer_file=str(snap / "tokenizer.json"))


VARIANTS = (
    "none",
    "project_pc1",
    "project_pc4",
    "project_pc16",
    "project_contrast",
    "project_mean",
    "damp_0.5",
    "damp_0.25",
)


def edit_spec(name: str, dirs: str | Path = OUT / "directions.npz") -> tuple:
    """(kind, V, beta) for a variant name."""
    if name == "none":
        return "none", None, 1.0
    if name.startswith("damp_"):
        return "damp", None, float(name.split("_")[1])
    d = np.load(dirs)
    if name.startswith("project_pc"):
        return "project", d["pcs"][: int(name[len("project_pc") :])], 1.0
    return "project", d[name.split("_")[1]][None], 1.0


def variants() -> dict:
    return {name: edit_spec(name) for name in VARIANTS}


def screen(args) -> None:
    from ouro_sft_eval import load_model

    model, _ = load_model(args.model, None, args.adapters)
    tok = tokenizer()
    meta = json.loads((OUT / "meta.json").read_text())[args.dev :]
    report = {}
    for name, (kind, V, beta) in variants().items():
        em = EditedOuro(model, kind, V, beta)
        em.ut_steps = 8
        changed = {L: {} for L in (6, 8)}
        for m in meta:
            if not m["draft"]:
                continue
            ids = m["prompt"] + m["draft"]
            hs = em.states(mx.array([ids]), em.make_cache())
            p = len(m["prompt"])
            # position p - 1 + j predicts draft token j
            a4 = np.array(mx.argmax(em.lm_head(hs[3][0, p - 1 : len(ids)]), axis=-1))
            types = token_types(tok, m["draft"]) + ["structure"]
            for L in (6, 8):
                aL = np.array(mx.argmax(em.lm_head(hs[L - 1][0, p - 1 : len(ids)]), axis=-1))
                for k, c in zip(types, aL != a4, strict=False):
                    changed[L].setdefault(k, []).append(bool(c))
            mx.clear_cache()
        report[name] = {
            f"loop{L}_changed_vs_loop4": {k: float(np.mean(v)) for k, v in changed[L].items()}
            for L in (6, 8)
        }
        print(name, json.dumps(report[name]), flush=True)
    (OUT / "screen.json").write_text(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["collect", "directions", "screen"])
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B")
    parser.add_argument("--adapters", default=None)
    parser.add_argument("--tasks", default="results/mathqa/validation_clean_tasks.jsonl")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--dev", type=int, default=100, help="tasks used to find directions")
    args = parser.parse_args()
    {"collect": collect, "directions": directions, "screen": screen}[args.stage](args)


if __name__ == "__main__":
    main()
