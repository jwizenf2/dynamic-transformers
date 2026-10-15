"""Router analysis v2: sequential stopping, query-text features, CV tuning, seeds, bootstrap.

    python scripts/lab/router_analysis_v2.py

Same data and conventions as router_analysis.py (v1, kept unchanged for reproducibility).
Additions, following the v1 findings (learned one-shot router: about 37% of the oracle saving;
Ouro gate: 10%):

* Sequential routers. After pass k (k = 1..3) we know the gate, entropy and margin of passes
  1..k (logged during prefill), so a policy can decide to stop or continue. Ouro's gate works
  this way, so this is the fair comparison. Two learned stopping rules, each from
  L2-logistic models trained on train:
    - ``seq_threshold``: q_k = P(correct at k | features up to k); stop at the first k with
      q_k >= tau.
    - ``seq_gain``: also g_k = P(correct at 4 | features up to k); stop at the first k with
      g_k - q_k <= delta, i.e. when continuing is not expected to help.
* Query-text features (the request only, no answer key): word count, number of " and ",
  commas, question marks and numbers. These are proxies for requests that ask for several
  things; v1 found that multi-call requests need more passes.
* Pooled-state PCA size and L2 strength chosen by 5-fold CV on train (mean log loss).
* LinUCB: 5 stream orders x reward costs c x exploration settings; every configuration is a
  point on its curve.
* Bootstrap (1,000 resamples of val): for each method, the fewest mean passes within 1 point of
  the best fixed accuracy, as a median with a 90% interval.

Caveat, which applies equally to all methods: thresholds and configurations are swept on val
and the best point is reported, so absolute savings are mildly optimistic. Comparisons between
methods are fair.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router_analysis import (  # noqa: E402
    JOBS,
    STEPS,
    concat,
    evaluate,
    exit_cdf,
    first_true,
    fit_logistic,
    load_run,
    predict,
)

from dht.bandits.linear_ucb import LinearUCB  # noqa: E402
from dht.toolcall.data import load_xlam  # noqa: E402

XLAM = "Salesforce/xlam-function-calling-60k"
XLAM_REVISION = "26d14ebfe18b1f7b524bd39b404b50af5dc97866"
BASE = "results/ouro_headroom/Ouro-2.6B"


def query_features(queries: list[str]) -> np.ndarray:
    rows = []
    for q in queries:
        words = q.split()
        rows.append(
            [
                np.log1p(len(words)),
                len(re.findall(r"\band\b", q, re.I)),
                q.count(","),
                q.count("?"),
                len(re.findall(r"\d+(?:\.\d+)?", q)),
            ]
        )
    return np.array(rows, dtype=np.float64)


class Features:
    """Standardised features up to pass k, with PCA of the pass-1 pooled state (fit on train)."""

    def __init__(self, feats: dict, queries: np.ndarray, pca: int):
        pool = feats["pool1"].astype(np.float64)
        self.mu = pool.mean(0)
        _, _, vt = np.linalg.svd(pool - self.mu, full_matrices=False)
        self.components = vt[:pca]
        self.stats = {}
        for k in STEPS:
            raw = self.raw(feats, queries, k)
            self.stats[k] = (raw.mean(0), raw.std(0) + 1e-8)

    def raw(self, feats: dict, queries: np.ndarray, k: int) -> np.ndarray:
        parts = [
            feats["gate"][:, :k],
            feats["entropy"][:, :k],
            feats["margin"][:, :k],
            np.log(feats["prompt_tokens"].astype(np.float64))[:, None],
            feats["n_tools"].astype(np.float64)[:, None],
            feats["n_required"].astype(np.float64)[:, None],
            queries,
        ]
        if len(self.components):
            pool = feats["pool1"].astype(np.float64)
            parts.append((pool - self.mu) @ self.components.T)
        return np.concatenate([np.asarray(p, dtype=np.float64) for p in parts], 1)

    def __call__(self, feats: dict, queries: np.ndarray, k: int) -> np.ndarray:
        m, s = self.stats[k]
        return (self.raw(feats, queries, k) - m) / s


def subset(feats: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in feats.items()}


def cv_select(feats, queries, y, grid_pca, grid_l2, folds=5, seed=0) -> tuple[int, float]:
    """Pick (PCA size, L2) by k-fold mean log loss of the four one-shot (pass-1) predictors."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    chunks = np.array_split(order, folds)
    best, best_loss = None, np.inf
    for pca in grid_pca:
        for l2 in grid_l2:
            losses = []
            for f in range(folds):
                te = chunks[f]
                tr = np.concatenate([chunks[j] for j in range(folds) if j != f])
                fm = Features(subset(feats, tr), queries[tr], pca)
                xtr = fm(subset(feats, tr), queries[tr], 1)
                xte = fm(subset(feats, te), queries[te], 1)
                for i in range(4):
                    p = np.clip(predict(fit_logistic(xtr, y[tr, i], l2), xte), 1e-6, 1 - 1e-6)
                    t = y[te, i]
                    losses.append(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))
            loss = float(np.mean(losses))
            if loss < best_loss:
                best, best_loss = (pca, l2), loss
    return best


def passes_to_match(points, target):
    feasible = [c for a, c in points if a >= target - 1e-12]
    return min(feasible) if feasible else None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", nargs="*", default=None)
    parser.add_argument("--val", default=None)
    parser.add_argument("--no-text", action="store_true", help="skip query features (tests)")
    args = parser.parse_args()
    train_dirs = args.train or sorted(
        str(p) for p in JOBS.glob(f"dynamic-transformers-ouro-train*/{BASE}/train_*")
    )
    val_dir = args.val or next(JOBS.glob(f"dynamic-transformers-ouro-val/{BASE}/val_*"))
    tr_ids, ytr, ftr = concat([load_run(Path(d)) for d in train_dirs])
    va_ids, yva, fva = load_run(Path(val_dir))
    if len(set(tr_ids)) != len(tr_ids) or set(tr_ids) & set(va_ids):
        raise ValueError("train ids must be unique and disjoint from val")

    if args.no_text:
        qtr, qva = np.zeros((len(tr_ids), 5)), np.zeros((len(va_ids), 5))
    else:
        root = Path(snapshot_download(XLAM, repo_type="dataset", revision=XLAM_REVISION))
        records = load_xlam(root / "xlam_function_calling_60k.json")
        text = {e.id: e.messages[0]["content"] for e in records}
        qtr = query_features([text[i] for i in tr_ids])
        qva = query_features([text[i] for i in va_ids])

    pca, l2 = cv_select(ftr, qtr, ytr, grid_pca=(0, 8, 16, 32), grid_l2=(0.3, 1.0, 3.0, 10.0))
    fm = Features(ftr, qtr, pca)
    xtr = {k: fm(ftr, qtr, k) for k in STEPS}
    xva = {k: fm(fva, qva, k) for k in STEPS}
    out: dict = {"n_train": len(tr_ids), "n_val": len(va_ids), "cv": {"pca": pca, "l2": l2}}

    steps = np.array(STEPS, dtype=np.float64)
    curves: dict[str, list] = {}

    # Baselines (as v1).
    cdf = exit_cdf(fva["gate"].astype(np.float64))
    curves["ouro_gate"] = [
        (*evaluate(k := first_true(cdf >= t), k, yva), k) for t in np.linspace(0.05, 0.99, 40)
    ]
    margin = fva["margin"].astype(np.float64)
    curves["confidence_margin"] = [
        (*evaluate(k := first_true(margin >= t), k, yva), k)
        for t in np.quantile(margin, np.linspace(0, 1, 41))
    ]

    # One-shot learned (decide after pass 1), now with query features and CV.
    p1 = np.stack([predict(fit_logistic(xtr[1], ytr[:, i], l2), xva[1]) for i in range(4)], 1)
    curves["learned_one_shot"] = []
    for lam in np.linspace(0, 0.3, 61):
        choice = np.argmax(p1 - lam * steps, 1) + 1
        curves["learned_one_shot"].append((*evaluate(choice, choice, yva), choice))

    # Sequential learned stopping.
    q = np.zeros((len(yva), 3))
    g = np.zeros((len(yva), 3))
    for k in (1, 2, 3):
        q[:, k - 1] = predict(fit_logistic(xtr[k], ytr[:, k - 1], l2), xva[k])
        g[:, k - 1] = predict(fit_logistic(xtr[k], ytr[:, 3], l2), xva[k])
    curves["seq_threshold"] = []
    for tau in np.linspace(0.05, 0.99, 48):
        stop = np.c_[q >= tau, np.ones(len(yva), bool)]
        k = first_true(stop)
        curves["seq_threshold"].append((*evaluate(k, k, yva), k))
    curves["seq_gain"] = []
    for delta in np.linspace(-0.2, 0.5, 57):
        stop = np.c_[(g - q) <= delta, np.ones(len(yva), bool)]
        k = first_true(stop)
        curves["seq_gain"].append((*evaluate(k, k, yva), k))

    # LinUCB replay (one-shot, pass-1 features), several seeds and settings.
    ctx_tr = np.c_[np.ones(len(ytr)), xtr[1]]
    ctx_va = np.c_[np.ones(len(yva)), xva[1]]
    curves["linucb"] = []
    out["linucb"] = []
    for seed in range(5):
        order = np.random.default_rng(seed).permutation(len(ctx_tr))
        for c in (0.1, 0.2, 0.3, 0.4):
            for explore in (0.2, 0.5, 1.0):
                bandit = LinearUCB(4, ctx_tr.shape[1], ridge=1.0, exploration=explore)
                for i in order:
                    w = torch.tensor(ctx_tr[i])
                    arm = bandit.select(w.repeat(4, 1))
                    bandit.update(arm, w, float(ytr[i, arm]) - c * (arm + 1) / 4)
                choice = np.array(
                    [bandit.select(torch.tensor(v).repeat(4, 1), explore=False) + 1 for v in ctx_va]
                )
                acc, cost = evaluate(choice, choice, yva)
                curves["linucb"].append((acc, cost, choice))
                out["linucb"].append(
                    {
                        "seed": seed,
                        "c": c,
                        "exploration": explore,
                        "val_accuracy": acc,
                        "val_passes": cost,
                        "arm_counts": bandit.counts.tolist(),
                    }
                )

    # Point estimates and bootstrap intervals.
    fixed_acc = yva.mean(0)
    target = fixed_acc.max() - 0.01
    oracle = np.where(yva.any(1), first_true(yva), 1)
    out["fixed"] = {int(r): float(fixed_acc[i]) for i, r in enumerate(STEPS)}
    out["oracle"] = {"accuracy": float(yva.any(1).mean()), "passes": float(oracle.mean())}
    rng = np.random.default_rng(123)
    boots = [rng.integers(0, len(yva), len(yva)) for _ in range(1000)]
    out["methods"] = {}
    for name, pts in curves.items():
        need = passes_to_match([(a, c) for a, c, _ in pts], target)
        samples = []
        for b in boots:
            yb = yva[b]
            tb = yb.mean(0).max() - 0.01
            pb = [(yb[np.arange(len(b)), ch[b] - 1].mean(), ch[b].mean()) for _, _, ch in pts]
            samples.append(passes_to_match(pb, tb) or 4.0)
        lo, med, hi = np.percentile(samples, [5, 50, 95])
        saving_oracle = 4 - oracle.mean()
        out["methods"][name] = {
            "passes_within_1pt": need,
            "fraction_of_oracle_saving": None
            if need is None
            else round((4 - need) / saving_oracle, 3),
            "bootstrap_passes_median_5_95": [round(med, 3), round(lo, 3), round(hi, 3)],
        }

    Path("results").mkdir(exist_ok=True)
    Path("results/router_analysis_v2.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "linucb"}, indent=1))


if __name__ == "__main__":
    main()
