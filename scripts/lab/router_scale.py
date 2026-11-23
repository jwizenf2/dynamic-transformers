"""Router scale-up analysis: hidden-state routers on 8,041 problems (pre-registered, offline).

    PYTHONPATH=scripts/lab python scripts/lab/router_scale.py

See docs/research/router_scale_2026-11-22.md. Data: chain v3 seed 0 greedy outcomes at 2 and 4
loops (``router_smoke``) and the 48-token probe's loop-1/2 hidden states (``router_probe2
--hidden``, [t, 2, H] per request) on NESTful [0, 1200), GSM8K test (800), MathQA clean val + test
pool (531) and GSM8K train (``run_stages.py --plan router-scale``, chunks of 1,000). Inputs as the
2026-11-18 follow-up: mean loop-2 state and mean h_2 − h_1 over the probe tokens.

Analysis 1: learning curve (PCA32/64-logistic, MLP; 5 seeds x 5 folds; benchmark one-hot) up to
the full fold. Analysis 2: the MLP (pre-committed) trained on GSM8K train + MathQA only, threshold
fixed from out-of-fold training scores at 0.85x fixed-4 cost, applied to GSM8K test (primary) and
NESTful [0, 1200); gain over the 1–4 hull of fixed depths, paired bootstrap 90% interval.
``router_nonlinear.py`` is left unchanged so its published numbers stay reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from phase_a_labels import auc
from router_diagnosis import predict, standardise
from router_nonlinear import MLP, train_net
from router_probe2_analysis import hull_interp

M = "ouro-chain-depth-v3"
SMOKE, PROBE = Path("results/router_smoke"), Path("results/router_probe2")
TRAIN_STARTS = range(0, 5510, 1000)
# (benchmark, split, router_smoke folder, hidden-probe folder)
PARTS = [
    ("mathqa", "train", M, f"{M}_hidden"),
    ("mathqa", "train", f"{M}_testpool", f"{M}_testpool_hidden"),
    ("nestful", "test", f"{M}_nestful", f"{M}_nestful_hid"),
    ("nestful", "test", f"{M}_nestful240", f"{M}_nestful240_hid"),
    ("nestful", "test", f"{M}_nestful720", f"{M}_nestful720_hid"),
    ("gsm8k", "test", f"{M}_gsm8k", f"{M}_gsm8k_hid"),
    *[("gsm8k", "train", f"{M}_gsm8ktrain{s}", f"{M}_gsm8ktrain{s}_hid") for s in TRAIN_STARTS],
]
BENCH = ("nestful", "gsm8k", "mathqa")
SIZES = (250, 500, 1000, 2000, 4000, None)  # None = the whole training fold
SEEDS, FOLDS, K, TARGET = 5, 5, 48, 0.85
torch.set_num_threads(8)


def load() -> list[dict]:
    """One row per request with a hidden-state file: label inputs, outcomes and fixed costs."""
    rows = []
    for bench, split, smoke, probe in PARTS:
        path = SMOKE / smoke / "examples.jsonl"
        if not path.exists():
            print(json.dumps({"missing": str(path)}), flush=True)
            continue
        for r in map(json.loads, path.open()):
            h = PROBE / probe / "hidden" / f"{r['id']}.npy"
            if not h.exists():
                continue
            runs = r["runs"]
            depths = sorted(int(k[1:].split("_")[0]) for k in runs if k.endswith("_greedy"))
            rows.append(
                {
                    "id": r["id"],
                    "bench": bench,
                    "split": split,
                    "hidden": h,
                    "y": {d: float(runs[f"d{d}_greedy"]["correct"]) for d in depths},
                    "t": {
                        d: float(
                            runs[f"d{d}_greedy"]["prompt_tokens"] + runs[f"d{d}_greedy"]["tokens"]
                        )
                        for d in depths
                    },
                }
            )
    return rows


def features(rows) -> np.ndarray:
    x = []
    for r in rows:
        h = np.load(r["hidden"]).astype(np.float32)
        x.append(np.concatenate([h[:, 1].mean(0), (h[:, 1] - h[:, 0]).mean(0)]))
    return np.array(x)


def onehot(rows) -> np.ndarray:
    return np.array([[float(r["bench"] == b) for b in BENCH] for r in rows])


def mlp(a, ya, b, seed) -> np.ndarray:
    return train_net(
        lambda d=a.shape[1]: MLP(d, 32),
        [torch.tensor(a, dtype=torch.float32)],
        ya,
        [torch.tensor(b, dtype=torch.float32)],
        seed,
        wd=1e-2,
    )


def curve(rows, x, y) -> dict:
    """Analysis 1: out-of-fold AUC by training size (protocol of ``hidden_curve_all``)."""
    x = np.hstack([x, onehot(rows)])
    n = len(rows)
    res: dict = {}
    full_scores: dict = {}
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        preds: dict = {}
        for f in range(FOLDS):
            te = order[f::FOLDS]
            pool = np.setdiff1d(order, te)
            for size in SIZES:
                tr = pool if size is None or size >= len(pool) else rng.choice(pool, size, False)
                a, b = standardise(x[tr], x[te])
                mu = a.mean(0)
                _, _, vt = np.linalg.svd(a - mu, full_matrices=False)
                key = size or "full"
                for k in (32, 64):
                    kk = min(k, len(tr) // 4)
                    preds.setdefault((key, f"pca{k}_logistic"), np.zeros(n))[te] = predict(
                        (a - mu) @ vt[:kk].T, y[tr], (b - mu) @ vt[:kk].T, l2=10.0
                    )
                preds.setdefault((key, "mlp"), np.zeros(n))[te] = mlp(a, y[tr], b, seed * 100 + f)
            print(json.dumps({"curve_seed": seed, "fold": f}), flush=True)
        for (size, model), p in preds.items():
            res.setdefault(model, {}).setdefault(str(size), []).append(auc(p, y))
            if size == "full":
                full_scores.setdefault(model, []).append(p)
    curves = {
        m: {s: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for s, v in by.items()}
        for m, by in res.items()
    }
    best = max(curves, key=lambda m: curves[m]["full"]["mean"])
    y2 = np.array([r["y"][2] for r in rows])
    by_bench, rescue = {}, []
    for p in full_scores[best]:
        fail = y2 == 0
        rescue.append(auc(p[fail], y[fail]))
        for b in BENCH:
            m = np.array([r["bench"] == b for r in rows])
            by_bench.setdefault(b, []).append(auc(p[m], y[m]))
    return {
        "n": n,
        "by_bench": {b: sum(r["bench"] == b for r in rows) for b in BENCH},
        "helps_rate": float(y.mean()),
        "curves": curves,
        "best": best,
        "auc_full": curves[best]["full"]["mean"],
        "reaches_0.72": bool(curves[best]["full"]["mean"] >= 0.72),
        "best_auc_full_by_bench": {b: float(np.mean(v)) for b, v in by_bench.items()},
        "best_rescue_auc_full": float(np.mean(rescue)),
    }


def costs(rows, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """(stay, escalate) loop-token costs: 2·T_2 and 4·T_4 + 2·k."""
    t2 = np.array([r["t"][2] for r in rows])
    t4 = np.array([r["t"][4] for r in rows])
    return 2 * t2, 4 * t4 + 2 * k


def outcome(rows, esc) -> tuple[np.ndarray, np.ndarray]:
    stay, up = costs(rows)
    y2 = np.array([r["y"][2] for r in rows])
    y4 = np.array([r["y"][4] for r in rows])
    return np.where(esc, y4, y2), np.where(esc, up, stay)


def fixed_points(rows, idx, depths) -> list[tuple[float, float]]:
    return [
        (
            float(np.mean([d * rows[i]["t"][d] for i in idx])),
            float(np.mean([rows[i]["y"][d] for i in idx])),
        )
        for d in depths
    ]


def gain(correct, cost, rows, idx, baseline: str) -> float:
    f4 = np.mean([4 * rows[i]["t"][4] for i in idx])
    c = float(cost[idx].mean())
    a4 = np.mean([rows[i]["y"][4] for i in idx])
    if c > f4:  # beyond the fixed range: compare with fixed-4
        return float(correct[idx].mean() - a4)
    depths = (1, 2, 3, 4) if baseline == "hull" else (2, 4)
    return float(correct[idx].mean() - hull_interp(fixed_points(rows, idx, depths), c))


def boot(stat, n: int) -> tuple[float, list[float]]:
    rng = np.random.default_rng(0)
    vals = [stat(rng.integers(0, n, n)) for _ in range(2000)]
    return stat(np.arange(n)), [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]


def evaluate(rows, scores, esc) -> dict:
    correct, cost = outcome(rows, esc)
    n = len(rows)
    f4 = np.mean([4 * r["t"][4] for r in rows])
    g, ci = boot(lambda i: gain(correct, cost, rows, i, "hull"), n)
    gc, ci_c = boot(lambda i: gain(correct, cost, rows, i, "chord"), n)
    return {
        "accuracy": float(correct.mean()),
        "cost_ratio_vs_fixed4": float(cost.mean() / f4),
        "escalated": float(esc.mean()),
        "gain_vs_hull_1_4": g,
        "ci90": ci,
        "verdict": "CONFIRMED" if ci[0] > 0 else "NOT CONFIRMED",
        "gain_vs_chord_2_4": gc,
        "ci90_chord": ci_c,
    }


def held_out(rows, x, y) -> dict:
    """Analysis 2: train on GSM8K train + MathQA, test on GSM8K test and NESTful [0, 1200)."""
    tr = [i for i, r in enumerate(rows) if r["split"] == "train"]
    R = [rows[i] for i in tr]
    xtr, ytr = x[tr], y[tr]
    oof = np.zeros(len(tr))
    order = np.random.default_rng(0).permutation(len(tr))
    for f in range(FOLDS):
        te = order[f::FOLDS]
        fit = np.setdiff1d(order, te)
        a, b = standardise(xtr[fit], xtr[te])
        oof[te] = mlp(a, ytr[fit], b, 1000 + f)
    stay, up = costs(R)
    f4 = np.mean([4 * r["t"][4] for r in R])
    cands = np.unique(oof)
    ratios = [np.where(oof > c, up, stay).mean() / f4 for c in cands]
    j = int(np.argmin(np.abs(np.array(ratios) - TARGET)))
    thr, share = float(cands[j]), float((oof > cands[j]).mean())
    out = {
        "train_n": len(tr),
        "train_oof_auc": auc(oof, ytr),
        "threshold": thr,
        "train_cost_ratio": float(ratios[j]),
        "train_escalated": share,
    }
    for bench in ("gsm8k", "nestful"):
        te = [i for i, r in enumerate(rows) if r["split"] == "test" and r["bench"] == bench]
        if not te:
            continue
        T = [rows[i] for i in te]
        a, b = standardise(xtr, x[te])
        s = mlp(a, ytr, b, 0)
        q = np.quantile(s, 1 - share)
        y2 = np.array([r["y"][2] for r in T])
        oracle = y[te] == 1
        out[bench] = {
            "n": len(te),
            "auc_4_helps": auc(s, y[te]),
            "rescue_auc": auc(s[y2 == 0], y[te][y2 == 0]),
            "fixed": {
                d: {"accuracy": acc, "cost_ratio": c / np.mean([4 * r["t"][4] for r in T])}
                for d, (c, acc) in zip(
                    (1, 2, 3, 4), fixed_points(T, range(len(T)), (1, 2, 3, 4)), strict=True
                )
            },
            "threshold_policy": evaluate(T, s, s > thr),
            "quantile_policy": evaluate(T, s, s >= q),
            "oracle": evaluate(T, s, oracle),
            "curve": [
                [float(c), float(acc)]
                for c, acc in (
                    (o[1].mean() / np.mean([4 * r["t"][4] for r in T]), o[0].mean())
                    for o in (
                        outcome(T, s >= np.quantile(s, 1 - p)) for p in np.linspace(0.02, 1, 50)
                    )
                )
            ],
        }
    out["primary"] = out.get("gsm8k", {}).get("threshold_policy")
    return out


def main() -> None:
    rows = load()
    x = features(rows)
    y = np.array([r["y"][2] == 0 and r["y"][4] == 1 for r in rows], float)
    report = {"analysis1_curve": curve(rows, x, y), "analysis2_held_out": held_out(rows, x, y)}
    out = Path("results/router_scale")
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=1, default=float)
    (out / "report.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
