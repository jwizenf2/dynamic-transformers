"""Router on a fine-tuned Ouro model: per-request depth from the early-decoding probe (path A).

    python scripts/lab/router_sft.py <eval dir> <features dir>

<eval dir> = ``results/ouro_sft_eval/<run>`` from ``ouro_sft_eval.py --adapters`` with
``xlam_train_0_1000/`` (router training labels, ``--xlam-split train``), ``bfcl/`` and
``xlam/`` (validation); rows carry per-depth correctness and ``out_tokens``.
<features dir> = ``results/ouro_early_features/<run>`` from ``ouro_early_features.py --hermes
--adapters`` with ``xlam_train_0_1000/``, ``bfcl_bfcl_0_*/`` and ``xlam_val_0_200/``.

Training never sees BFCL: the router is fitted on xLAM train-pool requests and tested on BFCL
(cross-benchmark) and xLAM val. Features: the 26 probe summaries of router v3
(``router_analysis_v3.early_summary``: per-pass gate, entropy, margin, agreement with pass 4,
state change over the first K = 8 tokens decoded at 4 passes, plus the mean Q-exit), z-scored
on the training pool. Choice rule and reporting as v3: four logistic predictors of
P(correct at R), CV over l2, pick argmax_R p_R − λ·R, sweep λ. Effective passes include the
probe: (min(K, T)·4 + max(T − K, 0)·R) / T with T the request's output length at 4 passes.
Baselines per λ: matched-histogram random allocation (2607.18553); also fixed depths and the
oracle (cheapest correct depth).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router_analysis import fit_logistic, predict  # noqa: E402
from router_analysis_v3 import curve, cv_l2, load_early  # noqa: E402


def labels(directory: Path, tokenizer=None) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Per request id: correctness at depths 1..4 and output length at depth 4.

    Rows written before ``out_tokens`` was logged (the 2026-10-17 BFCL/xLAM-val evals) get the
    length of their stored text, which was cut at 600 characters: long outputs are undercounted,
    so the probe's share of the cost is overstated (conservative for the router).
    """
    by: dict[str, dict[int, bool]] = {}
    toks: dict[str, int] = {}
    for line in (directory / "examples.jsonl").read_text().splitlines():
        r = json.loads(line)
        by.setdefault(str(r["id"]), {})[r["steps"]] = r["correct"]
        if r["steps"] == 4:
            if "out_tokens" in r:
                n = int(r["out_tokens"])
            elif tokenizer is not None:
                n = len(tokenizer.encode(r["text"], add_special_tokens=False))
            else:
                raise SystemExit(f"{directory}: rows have no out_tokens; pass a tokenizer")
            toks[str(r["id"])] = max(n, 1)
    full = {i: np.array([v[d] for d in (1, 2, 3, 4)], float) for i, v in by.items() if len(v) == 4}
    return full, toks


def joined(lab: dict, feat: dict, toks: dict):
    ids = [i for i in lab if i in feat and i in toks]
    return (
        ids,
        np.stack([lab[i] for i in ids]),
        np.stack([feat[i] for i in ids]),
        np.array([toks[i] for i in ids], float),
    )


def main() -> None:
    eval_dir, feat_dir = Path(sys.argv[1]), Path(sys.argv[2])
    # Feature folders are named <task>_<split>_<start>_<end>; group them by role.
    groups = {"train": "xlam_train_", "bfcl": "bfcl_", "val": "xlam_val_"}
    feat = {
        role: {
            i: v
            for p in sorted(feat_dir.glob(f"{prefix}*/early.npz"))
            for i, v in load_early(p).items()
        }
        for role, prefix in groups.items()
    }
    train_dir = next(eval_dir.glob("xlam_train_*"))
    from huggingface_hub import snapshot_download
    from ouro_headroom import OURO_REVISIONS
    from transformers import PreTrainedTokenizerFast

    model_id = "ByteDance/Ouro-2.6B"
    snap = Path(snapshot_download(model_id, revision=OURO_REVISIONS[model_id]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(snap / "tokenizer.json"))
    sets = {
        "train": labels(train_dir),
        "bfcl": labels(eval_dir / "bfcl", tokenizer),
        "val": labels(eval_dir / "xlam", tokenizer),
    }
    tr_ids, ytr, xtr, _ = joined(sets["train"][0], feat["train"], sets["train"][1])
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-8
    xtr = (xtr - mu) / sd
    l2 = cv_l2(xtr, ytr)
    models = [fit_logistic(xtr, ytr[:, k], l2) for k in range(4)]

    out: dict = {"n_train": len(tr_ids), "l2": l2}
    for role in ("bfcl", "val"):
        ids, y, x, t = joined(sets[role][0], feat[role], sets[role][1])
        if not ids:
            continue
        x = (x - mu) / sd
        p = np.stack([predict(m, x) for m in models], 1)
        out[role] = {
            "n": len(ids),
            "fixed": {d + 1: round(float(y[:, d].mean()), 4) for d in range(4)},
            "oracle": round(float(y.max(1).mean()), 4),
            "oracle_mean_depth": round(
                float(np.mean([(np.argmax(r) + 1) if r.any() else 1 for r in y])), 3
            ),
            "curve": curve(p, y, t, 8),
        }
    Path("results").mkdir(exist_ok=True)
    name = eval_dir.name
    Path(f"results/router_sft_{name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}))
    for role in ("bfcl", "val"):
        if role not in out:
            continue
        r = out[role]
        print(f"== {role} n={r['n']} fixed {r['fixed']} oracle {r['oracle']}")
        seen = set()
        for c in r["curve"]:
            key = (c["accuracy"], c["mean_R"])
            if key in seen:
                continue
            seen.add(key)
            print(
                f"  acc {c['accuracy']:.3f}  R {c['mean_R']:.2f}  eff {c['effective_passes']:.2f}"
                f"  random {c['random_same_histogram']:.3f} (p95 {c['random_p95']:.3f})"
            )


if __name__ == "__main__":
    main()
