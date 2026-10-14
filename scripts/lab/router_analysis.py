"""Offline router analysis for Ouro loop counts (CPU only; reads ouro_headroom.py outputs).

    python scripts/lab/router_analysis.py [--train DIR ...] [--val DIR]

Step 1 (predictability) and step 2 (bandit replay) of the plan in
docs/research/findings_2026-10-14_routing_and_looped_models.md, section 7.

Data: each run directory holds ``examples.jsonl`` (one row per request x pass count, with
``correct``) and ``features.npz`` (router inputs computed from the prompt only). Only requests
present in ``features.npz`` are used, so partial runs are handled.

Router features (no answer-key information): pass-1 exit gate, entropy and top-2 margin;
log prompt length; number of tools; number of required arguments; and the top-k principal
components (fit on train only) of the pass-1 mean-pooled hidden state. ``n_calls`` comes from
the answer key, so it is NOT a router feature; it is used only for the dependency-depth
breakdown.

Policies evaluated on val (accuracy vs mean passes; cost = passes actually run):
* fixed R = 1..4, and the hindsight oracle (smallest correct R, else 1);
* Ouro gate: stop at the first pass whose cumulative exit probability >= tau (request-level
  proxy: the gate at the last prompt token), swept over tau;
* confidence stopping: stop at the first pass whose top-2 margin >= t (sweep);
* learned one-shot (DHT-style, decide after pass 1): per-pass-count logistic regressions give
  p(correct | R); choose argmax_R p - lambda * R, swept over lambda;
* LinUCB replay (``src/dht/bandits``): online over a shuffled train stream, seeing only the
  chosen arm's reward = correct - c * R / 4; then frozen and evaluated greedily on val.

Headline numbers: the fewest mean passes each policy needs to stay within 1 point of the best
fixed accuracy, and the fraction of the oracle's saving that this captures.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import torch

from dht.bandits.linear_ucb import LinearUCB

STEPS = (1, 2, 3, 4)
JOBS = Path.home() / "AI/jobs"


def load_run(directory: Path) -> tuple[list[str], np.ndarray, dict]:
    feats = dict(np.load(directory / "features.npz", allow_pickle=False))
    ids = [str(i) for i in feats["id"]]
    keep = set(ids)
    outcome: dict[str, dict[int, bool]] = {}
    for line in (directory / "examples.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["id"] in keep:
            outcome.setdefault(row["id"], {})[row["steps"]] = row["correct"]
    ok = [i for i in ids if all(r in outcome.get(i, {}) for r in STEPS)]
    index = [ids.index(i) for i in ok]
    y = np.array([[outcome[i][r] for r in STEPS] for i in ok], dtype=bool)
    sub = {k: v[index] for k, v in feats.items()}
    return ok, y, sub


def concat(runs):
    ids = sum((r[0] for r in runs), [])
    y = np.concatenate([r[1] for r in runs])
    feats = {k: np.concatenate([r[2][k] for r in runs]) for k in runs[0][2]}
    return ids, y, feats


class FeatureMap:
    """Standardised small features + PCA of the pass-1 pooled state, fit on train only."""

    def __init__(self, feats: dict, k: int):
        self.k = k
        pool = feats["pool1"].astype(np.float64)
        self.mu = pool.mean(0)
        _, _, vt = np.linalg.svd(pool - self.mu, full_matrices=False)
        self.components = vt[:k]
        raw = self._raw(feats)
        self.m, self.s = raw.mean(0), raw.std(0) + 1e-8

    def _raw(self, feats: dict) -> np.ndarray:
        pool = (feats["pool1"].astype(np.float64) - self.mu) @ self.components.T
        small = np.stack(
            [
                feats["gate"][:, 0],
                feats["entropy"][:, 0],
                feats["margin"][:, 0],
                np.log(feats["prompt_tokens"].astype(np.float64)),
                feats["n_tools"].astype(np.float64),
                feats["n_required"].astype(np.float64),
            ],
            1,
        )
        return np.concatenate([small, pool], 1)

    def __call__(self, feats: dict) -> np.ndarray:
        return (self._raw(feats) - self.m) / self.s


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    """L2-regularised logistic regression (with intercept) by L-BFGS in float64."""
    xt = torch.tensor(np.c_[np.ones(len(x)), x])
    yt = torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(xt.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = xt @ w
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt) + l2 * (
            w[1:] ** 2
        ).sum() / len(xt)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy()


def predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(np.c_[np.ones(len(x)), x] @ w)))


def evaluate(choice: np.ndarray, cost: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Accuracy of the chosen pass counts and mean passes actually run."""
    acc = y[np.arange(len(y)), choice - 1].mean()
    return float(acc), float(cost.mean())


def exit_cdf(gates: np.ndarray) -> np.ndarray:
    """Ouro's exit distribution from per-pass gate values; the last pass absorbs the rest."""
    survive = np.ones(len(gates))
    cdf = np.zeros_like(gates)
    total = np.zeros(len(gates))
    for k in range(gates.shape[1]):
        p = gates[:, k] * survive if k < gates.shape[1] - 1 else survive
        total = total + p
        cdf[:, k] = total
        survive = survive * (1 - gates[:, k])
    return cdf


def first_true(mask: np.ndarray) -> np.ndarray:
    """Index (1-based) of the first True per row; the last pass if none."""
    any_ = mask.any(1)
    return np.where(any_, mask.argmax(1) + 1, mask.shape[1])


def frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sorted unique (mean passes, accuracy) points."""
    return sorted({(round(c, 4), round(a, 4)) for a, c in points})


def passes_to_match(points, target: float) -> float | None:
    feasible = [c for a, c in points if a >= target]
    return min(feasible) if feasible else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", nargs="*", default=None)
    parser.add_argument("--val", default=None)
    parser.add_argument("--pca", type=int, default=16)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = "results/ouro_headroom/Ouro-2.6B"
    train_dirs = args.train or sorted(
        glob.glob(str(JOBS / "dynamic-transformers-ouro-train*" / base / "train_*"))
    )
    val_dir = args.val or glob.glob(str(JOBS / "dynamic-transformers-ouro-val" / base / "val_*"))[0]
    tr_ids, ytr, ftr = concat([load_run(Path(d)) for d in train_dirs])
    va_ids, yva, fva = load_run(Path(val_dir))
    if len(set(tr_ids)) != len(tr_ids) or set(tr_ids) & set(va_ids):
        raise ValueError("train ids must be unique and disjoint from val")
    print(f"train {len(tr_ids)} from {train_dirs}\nval {len(va_ids)} from {val_dir}")

    fmap = FeatureMap(ftr, args.pca)
    xtr, xva = fmap(ftr), fmap(fva)
    out: dict = {"n_train": len(tr_ids), "n_val": len(va_ids), "methods": {}}

    fixed = {r: evaluate(np.full(len(yva), r), np.full(len(yva), r), yva) for r in STEPS}
    out["fixed"] = {r: {"accuracy": a, "passes": c} for r, (a, c) in fixed.items()}
    best_acc = max(a for a, _ in fixed.values())
    target = best_acc - 0.01

    solved = yva.any(1)
    oracle_choice = np.where(solved, first_true(yva), 1)
    o_acc, o_cost = evaluate(oracle_choice, oracle_choice, yva)
    out["oracle"] = {"accuracy": o_acc, "passes": o_cost}

    curves: dict[str, list] = {}
    cdf = exit_cdf(fva["gate"].astype(np.float64))
    curves["ouro_gate"] = [
        evaluate(k := first_true(cdf >= tau), k, yva) for tau in np.linspace(0.05, 0.99, 40)
    ]
    margin = fva["margin"].astype(np.float64)
    curves["confidence_margin"] = [
        evaluate(k := first_true(margin >= t), k, yva)
        for t in np.quantile(margin, np.linspace(0, 1, 41))
    ]

    probs = np.stack([predict(fit_logistic(xtr, ytr[:, i], args.l2), xva) for i in range(4)], 1)
    steps = np.array(STEPS, dtype=np.float64)
    curves["learned_one_shot"] = []
    for lam in np.linspace(0, 0.3, 61):
        choice = np.argmax(probs - lam * steps, 1) + 1
        curves["learned_one_shot"].append(evaluate(choice, choice, yva))
    out["predictor_auc"] = {}
    for i, r in enumerate(STEPS):
        pos, neg = probs[yva[:, i], i], probs[~yva[:, i], i]
        auc = float((pos[:, None] > neg[None, :]).mean()) if len(pos) and len(neg) else None
        out["predictor_auc"][r] = auc

    # LinUCB replay: online over a shuffled train stream, bandit feedback only.
    rng = np.random.default_rng(args.seed)
    ctx_tr = np.c_[np.ones(len(xtr)), xtr]
    ctx_va = np.c_[np.ones(len(xva)), xva]
    out["linucb"] = {}
    for c in (0.1, 0.2, 0.4):
        bandit = LinearUCB(4, ctx_tr.shape[1], ridge=1.0, exploration=0.5)
        reward_sum = 0.0
        for i in rng.permutation(len(ctx_tr)):
            w = torch.tensor(ctx_tr[i])
            arm = bandit.select(w.repeat(4, 1))
            reward = float(ytr[i, arm]) - c * (arm + 1) / 4
            bandit.update(arm, w, reward)
            reward_sum += reward
        choice = np.array(
            [bandit.select(torch.tensor(v).repeat(4, 1), explore=False) + 1 for v in ctx_va]
        )
        acc, cost = evaluate(choice, choice, yva)
        curves.setdefault("linucb", []).append((acc, cost))
        out["linucb"][str(c)] = {
            "val_accuracy": acc,
            "val_passes": cost,
            "train_mean_reward": reward_sum / len(ctx_tr),
            "arm_counts": bandit.counts.tolist(),
        }

    saving_oracle = 4 - (passes_to_match([(o_acc, o_cost)], target) or 4)
    for name, pts in curves.items():
        need = passes_to_match(pts, target)
        out["methods"][name] = {
            "frontier": frontier(pts),
            "passes_within_1pt_of_best_fixed": need,
            "fraction_of_oracle_saving": None
            if need is None or saving_oracle <= 0
            else round((4 - need) / saving_oracle, 3),
        }

    # Dependency-depth first look: smallest correct passes by number of expected calls.
    allf = {k: np.concatenate([ftr[k], fva[k]]) for k in ("n_calls",)}
    ally = np.concatenate([ytr, yva])
    depth: dict = {}
    for n in sorted(set(allf["n_calls"].tolist())):
        rows = ally[allf["n_calls"] == n]
        s = rows.any(1)
        depth[int(n)] = {
            "n": int(len(rows)),
            "accuracy_by_steps": {r: float(rows[:, i].mean()) for i, r in enumerate(STEPS)},
            "mean_smallest_correct": float(first_true(rows[s]).mean()) if s.any() else math.nan,
        }
    out["by_n_calls"] = depth

    Path("results").mkdir(exist_ok=True)
    Path("results/router_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "methods"}, indent=1))
    for name, m in out["methods"].items():
        print(
            name,
            "passes within 1pt:",
            m["passes_within_1pt_of_best_fixed"],
            "fraction of oracle saving:",
            m["fraction_of_oracle_saving"],
        )


if __name__ == "__main__":
    main()
