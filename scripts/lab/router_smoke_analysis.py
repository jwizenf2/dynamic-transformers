"""Router smoke test, analysis: label stability and a cross-validated probe router vs random mixes.

    python scripts/lab/router_smoke_analysis.py results/router_smoke/chain-depth-v3

Reads ``examples.jsonl`` from ``router_smoke.py``. Pre-registered on 2026-10-25, before any data:

1. Stability. Per depth d: P(sampled run correct | greedy correct) and P(sampled correct |
   greedy wrong), and how often the cheapest solving depth (in ``DEPTHS``, "none" if unsolved)
   of the greedy runs equals that of the sampled runs. If depth need were noise, the two
   conditionals would be close and the cheapest-depth agreement near chance.
2. Router. Labels are the greedy runs. Features: per loop, the mean over the k probe tokens and
   the first token's value of the 5 probe statistics (40), plus question words and numbers (2);
   standardised inside each fold. One L2-regularised logistic regression per depth predicts
   "greedy run at d is correct"; 5-fold cross-validation repeated over 5 shuffles gives
   out-of-fold probabilities P_d. Policy: d = argmin_d (λ·d/4 − P_d), swept over λ, giving an
   accuracy vs mean loops curve. Baselines: the same policy on question features only, the best
   random mix of fixed depths at the same mean loops (linear interpolation of fixed accuracies)
   and the per-request oracle.
3. Gate (kill rule). PASS if either (i) the router's accuracy minus the random mix's at mean
   loops 2.5 (the curve linearly interpolated) is >= 0.03, or (ii) the router reaches fixed-4
   accuracy minus 0.01 with >= 15% fewer mean loops than the random mix needs for the same
   accuracy (amended 2026-10-25 before any real data: "fewer than 4" passed trivially when
   shallow depths already match depth 4). Reported with a bootstrap 90%
   interval over requests for (i). Loops count generation and prefill at the chosen depth; the
   probe's k full-depth tokens are not added (small next to a dialogue, reported separately).
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np

DEPTHS = (1, 2, 4)
STATS = ("gate", "entropy", "margin", "agree4", "change")


def features(row: dict, probe_only: bool = False, question_only: bool = False) -> np.ndarray:
    p = np.array(
        [[[np.nan if x is None else x for x in loop] for loop in t] for t in row["probe"]],
        dtype=float,
    )  # [k, 4, 5]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns (loop-1 change)
        mean = np.nanmean(p, axis=0).reshape(-1)  # [20]
    first = p[0].reshape(-1)  # [20]
    q = row["query"]
    surface = np.array([len(q.split()), len(re.findall(r"\d+(?:\.\d+)?", q))], dtype=float)
    if question_only:
        return surface
    f = np.concatenate([mean, first])
    return f if probe_only else np.concatenate([f, surface])


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """L2-regularised logistic regression by Newton's method; returns [d + 1] weights (bias
    last, not regularised)."""
    xb = np.hstack([x, np.ones((len(x), 1))])
    w = np.zeros(xb.shape[1])
    reg = np.full(xb.shape[1], l2)
    reg[-1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(xb @ w, -30, 30)))
        grad = xb.T @ (p - y) + reg * w
        hess = (xb * (p * (1 - p))[:, None]).T @ xb + np.diag(reg) + 1e-6 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def oof_probs(x: np.ndarray, y: dict, folds: int = 5, repeats: int = 5) -> dict:
    """Out-of-fold P(correct at d), averaged over shuffled repeats; NaN features → fold mean."""
    n = len(x)
    out = {d: np.zeros(n) for d in DEPTHS}
    for rep in range(repeats):
        order = np.random.default_rng(rep).permutation(n)
        for f in range(folds):
            test = order[f::folds]
            train = np.setdiff1d(order, test)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mu = np.nanmean(x[train], axis=0)
                sd = np.nanstd(x[train], axis=0) + 1e-6
            xt = np.nan_to_num((x[train] - mu) / sd)
            xv = np.nan_to_num((x[test] - mu) / sd)
            for d in DEPTHS:
                yt = y[d][train]
                if yt.min() == yt.max():
                    out[d][test] += yt[0] / repeats
                    continue
                w = fit_logistic(xt, yt)
                xvb = np.hstack([xv, np.ones((len(xv), 1))])
                out[d][test] += 1 / (1 + np.exp(-(xvb @ w))) / repeats
    return out


def curve(p: dict, y: dict, idx: np.ndarray | None = None) -> list[tuple[float, float]]:
    """(mean loops, accuracy) of the cost-aware argmax policy over a λ sweep."""
    idx = np.arange(len(y[DEPTHS[0]])) if idx is None else idx
    pts = []
    for lam in np.linspace(0, 2, 81):
        score = np.stack([lam * d / 4 - p[d][idx] for d in DEPTHS])  # [3, n]
        choice = np.array(DEPTHS)[score.argmin(axis=0)]
        acc = np.mean([y[c][i] for c, i in zip(choice, idx, strict=True)])
        pts.append((float(choice.mean()), float(acc)))
    return sorted(set(pts))


def at_loops(pts: list[tuple[float, float]], loops: float) -> float:
    """Accuracy of the upper envelope of ``pts`` (plus mixing) at ``loops``, by interpolation."""
    xs, ys = zip(*pts, strict=True)
    best = []  # running max of accuracy along increasing loops: mixing never hurts
    m = -1.0
    for a in ys:
        m = max(m, a)
        best.append(m)
    return float(np.interp(loops, xs, best))


def random_mix(acc: dict, loops: float) -> float:
    return float(np.interp(loops, list(DEPTHS), [acc[d] for d in DEPTHS]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="results/router_smoke/<model>")
    args = parser.parse_args()
    rows = [json.loads(x) for x in (Path(args.run) / "examples.jsonl").read_text().splitlines()]
    n = len(rows)
    g = {d: np.array([r["runs"][f"d{d}_greedy"]["correct"] for r in rows], float) for d in DEPTHS}
    s = {d: np.array([r["runs"][f"d{d}_sample"]["correct"] for r in rows], float) for d in DEPTHS}
    report: dict = {"n": n, "fixed_greedy": {d: g[d].mean() for d in DEPTHS}}
    report["fixed_sample"] = {d: s[d].mean() for d in DEPTHS}

    # 1. stability
    stab = {}
    for d in DEPTHS:
        gc, gw = g[d] == 1, g[d] == 0
        stab[d] = {
            "P(sample ok | greedy ok)": float(s[d][gc].mean()) if gc.any() else None,
            "P(sample ok | greedy wrong)": float(s[d][gw].mean()) if gw.any() else None,
        }

    def cheapest(lab):
        return [next((d for d in DEPTHS if lab[d][i]), 0) for i in range(n)]

    cg, cs = cheapest(g), cheapest(s)
    agree = float(np.mean([a == b for a, b in zip(cg, cs, strict=True)]))
    pg = {c: cg.count(c) / n for c in (0, *DEPTHS)}
    ps = {c: cs.count(c) / n for c in (0, *DEPTHS)}
    chance = sum(pg[c] * ps[c] for c in pg)
    report["stability"] = {
        "per_depth": stab,
        "cheapest_depth_agreement": agree,
        "chance_agreement": chance,
        "cheapest_greedy_hist": pg,
    }

    # 2. router
    x_full = np.stack([features(r) for r in rows])
    x_q = np.stack([features(r, question_only=True) for r in rows])
    p_full, p_q = oof_probs(x_full, g), oof_probs(x_q, g)
    c_full, c_q = curve(p_full, g), curve(p_q, g)
    acc = report["fixed_greedy"]
    oracle_acc = float(np.mean([max(g[d][i] for d in DEPTHS) for i in range(n)]))
    oracle_loops = float(np.mean([c if c else 1 for c in cg]))
    report["oracle"] = {"accuracy": oracle_acc, "mean_loops": oracle_loops}
    report["router_probe"] = {f"acc@{L}": at_loops(c_full, L) for L in (1.5, 2.0, 2.5, 3.0, 3.5)}
    report["router_question"] = {f"acc@{L}": at_loops(c_q, L) for L in (1.5, 2.0, 2.5, 3.0, 3.5)}
    report["random_mix"] = {f"acc@{L}": random_mix(acc, L) for L in (1.5, 2.0, 2.5, 3.0, 3.5)}

    # 3. gate
    gain = at_loops(c_full, 2.5) - random_mix(acc, 2.5)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(500):
        idx = rng.integers(0, n, n)
        acc_b = {d: g[d][idx].mean() for d in DEPTHS}
        boots.append(at_loops(curve(p_full, g, idx), 2.5) - random_mix(acc_b, 2.5))
    target = acc[4] - 0.01
    grid = np.linspace(1, 4, 301)
    reach = [L for L in grid if at_loops(c_full, L) >= target]
    reach_mix = [L for L in grid if random_mix(acc, L) >= target]
    loops_router = min(reach) if reach else 4.0
    loops_mix = min(reach_mix) if reach_mix else 4.0
    savings = (loops_mix - loops_router) / loops_mix
    passed = gain >= 0.03 or savings >= 0.15
    report["gate"] = {
        "gain_vs_random_at_2.5": gain,
        "gain_90ci": [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))],
        "loops_router_at_fixed4_minus_1pt": loops_router,
        "loops_random_mix_at_fixed4_minus_1pt": loops_mix,
        "loop_savings_vs_random_mix": savings,
        "verdict": "PASS" if passed else "FAIL",
        "probe_tokens_full_depth": len(rows[0]["probe"]),
    }
    text = json.dumps(report, indent=1, default=float)
    (Path(args.run) / "analysis.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
