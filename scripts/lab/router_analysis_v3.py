"""Router v3: per-request pass count from pass-1 features plus an early-decoding probe (xLAM).

    python scripts/lab/router_analysis_v3.py <labels dir> <early dir>

<labels dir> holds the per-pass correctness runs (val_0_200, train_0_1000, train_100_1000;
``ouro_headroom.py`` outputs). <early dir> holds ``ouro_early_features.py`` outputs
(xlam_train_0_1000/early.npz, xlam_val_0_200/early.npz).

Policy ("probe then commit"): decode the first K tokens at all 4 passes, read the probe, pick a
pass count R for the rest of the request. Labels are the per-pass correctness of plain R-pass
runs, an approximation: under the real policy the first K tokens come from 4 passes. Cost per
request = (min(K, T)·4 + max(T − K, 0)·R) / T passes per token, with T the request's 4-pass
output length. v2 features alone (no probe, cost R) are reported on the same split.

Choice rule as in v2: argmax_k p_k − λ·k over the four pass-count predictors (logistic, 5-fold
CV over l2). Also reported: the matched-histogram random baseline (2607.18553), fixed-3 and the
per-token gate (nominal accuracy from job 32cd8d78; its serving speedup is ×1.01–1.06 at
batch ≥ 4, see findings §7).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_analysis_v2 as ra  # noqa: E402
from router_analysis import concat, fit_logistic, load_run, predict  # noqa: E402

K_COST = 4  # passes used by the probe tokens


def early_summary(early: np.ndarray) -> np.ndarray:
    """[n, K, 4, 5] → [n, 26]: per pass nanmean/nanmax gate, nanmean entropy, margin,
    agreement with pass 4 and state change, plus the mean per-token Q-exit at q = 0.5."""
    gate = early[..., 0]
    parts = [
        np.nanmean(gate, 1),
        np.nanmax(gate, 1),
        np.nanmean(early[..., 1], 1),
        np.nanmean(early[..., 2], 1),
        np.nanmean(early[..., 3], 1)[:, :3],  # pass 4 agrees with itself
        np.nanmean(early[..., 4], 1)[:, 1:],  # undefined for pass 1
    ]
    # Ouro Q-exit at q = 0.5 per token, averaged over the probe tokens.
    survive = np.ones(gate.shape[:2])
    total = np.zeros(gate.shape[:2])
    exit_at = np.full(gate.shape[:2], 4.0)
    done = np.zeros(gate.shape[:2], bool)
    for k in range(4):
        total = total + (gate[..., k] * survive if k < 3 else survive)
        hit = (total >= 0.5) & ~done
        exit_at[hit] = k + 1
        done |= hit
        survive = survive * (1 - gate[..., k])
    exit_at[np.isnan(gate[..., 0])] = np.nan
    parts.append(np.nanmean(exit_at, 1)[:, None])
    out = np.concatenate(parts, 1)
    return np.nan_to_num(out, nan=0.0)


def load_early(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path)
    summ = early_summary(d["early"].astype(np.float64))
    return {str(i): summ[j] for j, i in enumerate(d["id"])}


def out_tokens(directory: Path) -> dict[str, int]:
    rows = (json.loads(x) for x in (directory / "examples.jsonl").read_text().splitlines())
    return {r["id"]: max(r["out_tokens"], 1) for r in rows if r["steps"] == 4}


def cv_l2(x: np.ndarray, y: np.ndarray, grid=(0.3, 1.0, 3.0, 10.0), folds=5) -> float:
    order = np.random.default_rng(0).permutation(len(y))
    chunks = np.array_split(order, folds)
    best, best_loss = grid[0], np.inf
    for l2 in grid:
        loss = []
        for f in range(folds):
            te = chunks[f]
            tr = np.concatenate([chunks[j] for j in range(folds) if j != f])
            for i in range(y.shape[1]):
                p = np.clip(predict(fit_logistic(x[tr], y[tr, i], l2), x[te]), 1e-6, 1 - 1e-6)
                t = y[te, i]
                loss.append(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))
        if np.mean(loss) < best_loss:
            best, best_loss = l2, float(np.mean(loss))
    return best


def curve(p: np.ndarray, y: np.ndarray, tokens: np.ndarray | None, k: int) -> list[dict]:
    rows, n = [], len(y)
    rng = np.random.default_rng(0)
    for lam in np.linspace(0, 0.3, 31):
        choice = np.argmax(p - lam * np.arange(1, 5), 1)
        acc = float(y[np.arange(n), choice].mean())
        passes = choice + 1.0
        if tokens is not None:
            probe = np.minimum(k, tokens)
            passes = (probe * K_COST + np.maximum(tokens - k, 0) * passes) / tokens
        rand = [y[np.arange(n), rng.permutation(choice)].mean() for _ in range(500)]
        rows.append(
            {
                "lambda": round(float(lam), 3),
                "accuracy": round(acc, 4),
                "mean_R": round(float(choice.mean() + 1), 3),
                "effective_passes": round(float(passes.mean()), 3),
                "random_same_histogram": round(float(np.mean(rand)), 4),
                "random_p95": round(float(np.percentile(rand, 95)), 4),
            }
        )
    return rows


def main() -> None:
    labels, early_dir = Path(sys.argv[1]), Path(sys.argv[2])
    tr_ids, ytr, ftr = concat(
        [load_run(labels / "train_0_1000"), load_run(labels / "train_100_1000")]
    )
    va_ids, yva, fva = load_run(labels / "val_0_200")
    early = load_early(early_dir / "xlam_train_0_1000" / "early.npz")
    early.update(load_early(early_dir / "xlam_val_0_200" / "early.npz"))
    keep_tr = [j for j, i in enumerate(tr_ids) if i in early]
    keep_va = [j for j, i in enumerate(va_ids) if i in early]
    tr_ids = [tr_ids[j] for j in keep_tr]
    va_ids = [va_ids[j] for j in keep_va]
    ytr, yva = ytr[keep_tr].astype(float), yva[keep_va].astype(float)
    ftr = ra.subset(ftr, np.array(keep_tr))
    fva = ra.subset(fva, np.array(keep_va))

    fm = ra.Features(ftr, np.zeros((len(tr_ids), 5)), 8)
    base_tr = fm(ftr, np.zeros((len(tr_ids), 5)), 1)
    base_va = fm(fva, np.zeros((len(va_ids), 5)), 1)
    etr = np.stack([early[i] for i in tr_ids])
    eva = np.stack([early[i] for i in va_ids])
    mu, sd = etr.mean(0), etr.std(0) + 1e-8
    x3_tr = np.c_[base_tr, (etr - mu) / sd]
    x3_va = np.c_[base_va, (eva - mu) / sd]
    toks = out_tokens(labels / "val_0_200")
    t_va = np.array([toks[i] for i in va_ids], dtype=float)

    out: dict = {
        "n_train": len(tr_ids),
        "n_val": len(va_ids),
        "fixed": {k + 1: round(float(yva[:, k].mean()), 4) for k in range(4)},
        "oracle": round(float(yva.max(1).mean()), 4),
        "mean_output_tokens_val": round(float(t_va.mean()), 1),
    }
    for name, (xtr, xva, toks_) in {
        "v2_pass1": (base_tr, base_va, None),
        "v3_pass1_plus_probe": (x3_tr, x3_va, t_va),
        "v3_probe_cost_ignored": (x3_tr, x3_va, None),
    }.items():
        l2 = cv_l2(xtr, ytr)
        p = np.stack([predict(fit_logistic(xtr, ytr[:, i], l2), xva) for i in range(4)], 1)
        out[name] = {"l2": l2, "curve": curve(p, yva, toks_, 8)}
    Path("results").mkdir(exist_ok=True)
    Path("results/router_analysis_v3.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict) or "curve" not in v}))
    for name in ("v2_pass1", "v3_pass1_plus_probe", "v3_probe_cost_ignored"):
        print(f"== {name} (l2 {out[name]['l2']})")
        seen = set()
        for r in out[name]["curve"]:
            key = (r["accuracy"], r["mean_R"])
            if key in seen:
                continue
            seen.add(key)
            print(
                f"  acc {r['accuracy']:.3f}  R {r['mean_R']:.2f}  eff {r['effective_passes']:.2f}"
                f"  random {r['random_same_histogram']:.3f} (p95 {r['random_p95']:.3f})"
            )


if __name__ == "__main__":
    main()
