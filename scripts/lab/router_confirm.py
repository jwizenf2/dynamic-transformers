"""Confirm the frozen escalation router once on NESTful (test set).

    python scripts/lab/router_confirm.py \
        --train-smoke results/router_smoke/ouro-chain-depth-v3 \
        --train-probe results/router_probe2/ouro-chain-depth-v3 \
        --test-smoke results/router_smoke/ouro-chain-depth-v3_nestful \
        --test-probe results/router_probe2/ouro-chain-depth-v3_nestful

Pre-registered 2026-10-27, before any NESTful router data. The router is the deployable
escalation design of ``router_probe2_analysis`` (start at 2 loops; 12 loop-1/2 features of the
first answer tokens; switch to 4 loops: cost 4·T_4 + 2·k, else 2·T_2). Everything is fixed on
the 292 clean MathQA-validation tasks:

* models: one L2 logistic regression per depth (2, 4) on all 292 tasks, features standardised
  with the training mean/sd;
* operating point: λ such that the out-of-fold policy on the training tasks spends 0.75x the
  fixed-4 cost (closest grid point), i.e. "save a quarter of the compute";
* policy: switch to 4 loops iff λ·4/4 − P_4 < λ·2/4 − P_2.

Then applied once to the 120 NESTful requests of ``ouro_nestful_multiturn`` (greedy labels at
depths 1, 2, 4 from ``router_smoke.py --tasks nestful --modes greedy``, probe from
``router_probe2.py --tasks nestful``). Primary: accuracy of the frozen policy minus the best
random mix of fixed depths on NESTful at the policy's realised NESTful cost (upper concave hull
of NESTful's fixed (mean loop-tokens, accuracy)), with a bootstrap 90% interval over requests
(choices fixed). CONFIRMED if the lower bound > 0, CONSISTENT if the point estimate > 0, else
NOT CONFIRMED. Secondary: the frozen models' full λ curve on NESTful, and the fixed-depth
accuracies (depth 4 should reproduce the earlier .475 of ``ouro_nestful_multiturn``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from router_probe2_analysis import features_escalate, hull_interp, policy_curve
from router_smoke_analysis import fit_logistic, oof_probs

DEPTHS = (1, 2, 4)
ESC = (2, 4)
TARGET = 0.75


def load(smoke_dir: str, probe_dir: str):
    smoke = [json.loads(x) for x in (Path(smoke_dir) / "examples.jsonl").read_text().splitlines()]
    probes = {
        r["id"]: r
        for r in map(json.loads, (Path(probe_dir) / "examples.jsonl").read_text().splitlines())
    }
    assert {r["id"] for r in smoke} == set(probes), f"{smoke_dir} and {probe_dir} differ"
    for r in probes.values():
        r["loop12_disagree"] = sum(t[0][3] != t[1][3] for t in r["probe"])
    x = np.stack([features_escalate(probes[r["id"]], r["query"]) for r in smoke])
    y = {d: np.array([r["runs"][f"d{d}_greedy"]["correct"] for r in smoke], float) for d in DEPTHS}
    t = {
        d: np.array(
            [
                r["runs"][f"d{d}_greedy"]["prompt_tokens"] + r["runs"][f"d{d}_greedy"]["tokens"]
                for r in smoke
            ],
            float,
        )
        for d in DEPTHS
    }
    k = np.array([len(probes[r["id"]]["probe"]) for r in smoke], float)
    fixed = {d: d * t[d] for d in DEPTHS}
    esc = {2: 2 * t[2], 4: 4 * t[4] + 2 * k}
    return x, y, fixed, esc


def choose(p: dict, lam: float) -> np.ndarray:
    return np.where(lam * 4 / 4 - p[4] < lam * 2 / 4 - p[2], 4, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-smoke", default="results/router_smoke/ouro-chain-depth-v3")
    parser.add_argument("--train-probe", default="results/router_probe2/ouro-chain-depth-v3")
    parser.add_argument("--test-smoke", default="results/router_smoke/ouro-chain-depth-v3_nestful")
    parser.add_argument("--test-probe", default="results/router_probe2/ouro-chain-depth-v3_nestful")
    parser.add_argument("--out", default="results/router_confirm")
    args = parser.parse_args()
    xtr, ytr, fixed_tr, esc_tr = load(args.train_smoke, args.train_probe)
    xte, yte, fixed_te, esc_te = load(args.test_smoke, args.test_probe)

    # operating point from out-of-fold probabilities on the training tasks
    oof = oof_probs(xtr, ytr)
    grid = np.linspace(0, 2, 401)
    ratios = []
    for lam in grid:
        c = choose(oof, lam)
        ratios.append(np.mean(np.where(c == 4, esc_tr[4], esc_tr[2])) / fixed_tr[4].mean())
    lam_star = float(grid[int(np.argmin(np.abs(np.array(ratios) - TARGET)))])

    # frozen models on all training tasks
    mu = np.nanmean(xtr, axis=0)
    sd = np.nanstd(xtr, axis=0) + 1e-6
    a, b = np.nan_to_num((xtr - mu) / sd), np.nan_to_num((xte - mu) / sd)
    p_te = {}
    for d in ESC:
        w = fit_logistic(a, ytr[d])
        p_te[d] = 1 / (1 + np.exp(-(np.hstack([b, np.ones((len(b), 1))]) @ w)))

    choice = choose(p_te, lam_star)
    n = len(choice)
    correct = np.where(choice == 4, yte[4], yte[2])
    cost = np.where(choice == 4, esc_te[4], esc_te[2])

    def gain(idx):
        fp = [(float(fixed_te[d][idx].mean()), float(yte[d][idx].mean())) for d in DEPTHS]
        return float(correct[idx].mean() - hull_interp(fp, float(cost[idx].mean())))

    rng = np.random.default_rng(0)
    boots = [gain(rng.integers(0, n, n)) for _ in range(2000)]
    g = gain(np.arange(n))
    lo, hi = float(np.percentile(boots, 5)), float(np.percentile(boots, 95))
    verdict = "CONFIRMED" if lo > 0 else ("CONSISTENT" if g > 0 else "NOT CONFIRMED")
    report = {
        "n_train": len(xtr),
        "n_test": n,
        "lambda": lam_star,
        "train_oof_cost_ratio": float(ratios[int(np.argmin(np.abs(np.array(ratios) - TARGET)))]),
        "nestful_fixed_accuracy": {d: float(yte[d].mean()) for d in DEPTHS},
        "nestful_fixed_cost_ratio": {
            d: float(fixed_te[d].mean() / fixed_te[4].mean()) for d in DEPTHS
        },
        "policy": {
            "accuracy": float(correct.mean()),
            "cost_ratio_vs_fixed4": float(cost.mean() / fixed_te[4].mean()),
            "escalated": float((choice == 4).mean()),
        },
        "primary_gain_vs_random_mix": g,
        "ci90": [lo, hi],
        "verdict": verdict,
        "secondary_curve": [
            (round(c / fixed_te[4].mean(), 3), round(acc, 3))
            for c, acc in policy_curve(p_te, yte, esc_te, np.arange(n), ESC)
        ],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=1, default=float)
    (out / "confirm.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
