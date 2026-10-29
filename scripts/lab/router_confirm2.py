"""Confirm the Phase A escalation router once on unseen NESTful requests [240, 720) (test set).

    python scripts/lab/phase_a_labels.py      # results/phase_a/labels.jsonl (soft labels)
    python scripts/lab/router_confirm2.py

Pre-registered 2026-10-29, before any NESTful [240, 720) data (collected by ``router_smoke.py
--tasks nestful --start 240 --n 480 --depths 1 2 3 4 --modes greedy --tag nestful240`` and
``router_probe2.py --tasks nestful --start 240 --n 480 --tag nestful240``). Phase A development
(``phase_a_router.py``, ``phase_a_trace_analysis.py``) used only the 531 MathQA tasks; NESTful
[0, 240) was used once, for the step-1 confirmation (``router_confirm.py``), and never to choose a
router. The slice [240, 720) has never been run.

Router (frozen, ``esc12_soft`` of ``phase_a_router``): start at 2 loops; the 12 loop-1/2 features
of the first answer tokens (``features_escalate``; NaN → 0, standardised with the training
mean/sd); one L2 logistic regression (``fit_logistic``, l2 = 1) per depth in {2, 4} on all 531
training tasks (292 clean MathQA-val + 239 MathQA-test pool), trained on soft labels (greedy + 4
samples, mean of 5); switch to 4 loops iff λ − P_4 < λ/2 − P_2. Cost: stay 2·T_2, escalate
4·T_4 + 2·k (k probe tokens; prompt loop-1/2 caches reused), T = prompt + generated tokens of the
greedy dialogue at that depth. Operating points: λ such that the out-of-fold policy (5 folds x 5
repeats on the 531, as ``phase_a_router``) spends the target share of fixed-4 cost (closest grid
point): 0.75x (primary; same rule as step 1) and 0.85x (secondary non-inferiority).

Primary (H1): accuracy of the frozen 0.75x policy on [240, 720) minus the best random mix of fixed
depths (upper concave hull of [240, 720)'s fixed (mean loop-tokens, accuracy), depths 1–4) at the
policy's realised cost; bootstrap 90% interval over requests (choices fixed, 2000 resamples, seed
0). CONFIRMED if the lower bound > 0, CONSISTENT if the point estimate > 0, else NOT CONFIRMED. If
the realised cost exceeds fixed-4's, the comparison is with fixed-4 (flagged). The baseline is
the hull over depths 1–4, so at ~0.75x it is effectively fixed 3 loops (development used the 2–4
chord; on [0, 240) fixed-3 sat .002 above it).

Power (corrected after review, before collection): on [0, 240) the gain's bootstrap SD was .0165
(n = 240), so ~.011 at n = 480, with a longer lower tail (the baseline is a max over mixes);
CONFIRMED needs a point estimate above ~.024, i.e. probability ~0.7 if the true gain is +.03.
But [240, 720) is shallower: the depth-balanced order runs out of long chains (dependency-depth
buckets [240, 480) 47/47/47/46/34/19 for 2/3/4/5/6/7+, [480, 720) 60/60/60/60/0/0; mean gold
calls 6.1 and 4.75 vs 7.8 on [0, 240)). On [0, 240) the step-1 router gained +.007 on tasks with
<= 5 gold calls and +.042 on longer ones (the 2→4 accuracy gap .047 vs .158), so the expected gain
here is ~+.02 and the power ~0.35–0.4. Every remaining unseen deep task is already in [240, 480)
(74 depth-6 and 59 depth-7+ tasks exist in total), so a larger n would add only short chains; the
slice is kept and the stratified results (S7) are pre-registered to show where any gain comes from.

Secondary (reported, no verdict changes the primary):
* S1 non-inferiority at the 0.85x operating point: accuracy − fixed-4 accuracy (paired), 90%
  interval; non-inferior at margin 3 points if the lower bound > −.03. Expected power ~0.4
  (development: −.013 at 0.85x; paired SD ~.012 at n = 480, so the estimate must exceed ~−.01).
* S2 paired against the step-1 frozen router (``router_confirm``: 292 clean tasks, greedy labels,
  0.75x). Step 1 pre-registered "paired accuracy difference at equal realised cost, one-sided
  bootstrap" on [240, 480); here on [240, 720) (plan of 2026-10-28, more power) and [240, 480).
  Equal cost: the new router's λ is set on the test requests so its realised cost is closest to
  the step-1 router's (cost matching, not selection; the new router's labels are not used); the
  one-sided 95% bound is the lower end of the 90% interval. Each router at its own λ is also shown.
* S3 paired against fixed 3 loops (accuracy difference and cost ratio).
* S4 LEV-style stopper (``phase_a_router`` A7, refitted on all 531): own threshold rule, accuracy,
  cost and gain over the mix.
* S5 pooled NESTful [0, 720) with the same frozen router (n = 720; [0, 240) was seen once).
* S6 the router's λ curve on [240, 720), AUC of P_4 − P_2 for "4 helps" (greedy wrong at 2, right
  at 4), the perfect 2-vs-4 router (oracle) and the fixed-depth accuracies and costs.
* S7 strata: the primary policy's accuracy, cost and gain over each stratum's own mix for gold
  calls <= 5 vs > 5, and for [240, 480) vs [480, 720).

The probe follows the loop-4 greedy draft (``router_probe2``) for train and test alike; a deployed
2-loop router would read its own draft, which differs where loop-2 and loop-4 argmax differ, so
the 2·k probe cost and deployability are approximations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from phase_a_labels import auc
from router_confirm import choose, load
from router_probe2_analysis import hull_interp, policy_curve
from router_smoke_analysis import fit_logistic, oof_probs

ESC = (2, 4)
TRAIN = (
    ("results/router_smoke/ouro-chain-depth-v3", "results/router_probe2/ouro-chain-depth-v3"),
    (
        "results/router_smoke/ouro-chain-depth-v3_testpool",
        "results/router_probe2/ouro-chain-depth-v3_testpool",
    ),
)
TEST = (
    "results/router_smoke/ouro-chain-depth-v3_nestful240",
    "results/router_probe2/ouro-chain-depth-v3_nestful240",
)
SEEN = (
    "results/router_smoke/ouro-chain-depth-v3_nestful",
    "results/router_probe2/ouro-chain-depth-v3_nestful",
)
LEV_COLS = [0, 1, 2, 4, 5, 6, 9]  # per loop 1–2: mean/max entropy, min margin; disagreements
LAMS = np.linspace(0, 2, 401)
FOLDS, REPEATS = 5, 5


def load_set(smoke_dir: str, probe_dir: str) -> dict:
    """``router_confirm.load`` plus ids, raw (NaN → 0) features and final answer values."""
    x, y, fixed, esc = load(smoke_dir, probe_dir)
    smoke = [json.loads(s) for s in (Path(smoke_dir) / "examples.jsonl").read_text().splitlines()]
    agree = np.array(
        [r["runs"]["d2_greedy"]["value"] == r["runs"]["d4_greedy"]["value"] for r in smoke], float
    )
    return {
        "ids": [r["id"] for r in smoke],
        "x_raw": x,
        "x": np.nan_to_num(x),
        "y": y,
        "fixed": fixed,
        "esc": esc,
        "agree": agree,
        "calls": np.array([r["gold_calls"] for r in smoke]),
    }


def subset(d: dict, mask: np.ndarray) -> dict:
    return {
        "ids": [i for i, m in zip(d["ids"], mask, strict=True) if m],
        **{k: d[k][mask] for k in ("x_raw", "x", "agree", "calls")},
        **{k: {j: v[mask] for j, v in d[k].items()} for k in ("y", "fixed", "esc")},
    }


def concat(a: dict, b: dict) -> dict:
    keys = set(a["y"]) & set(b["y"])
    return {
        "ids": a["ids"] + b["ids"],
        "x_raw": np.vstack([a["x_raw"], b["x_raw"]]),
        "x": np.vstack([a["x"], b["x"]]),
        "y": {d: np.concatenate([a["y"][d], b["y"][d]]) for d in keys},
        "fixed": {d: np.concatenate([a["fixed"][d], b["fixed"][d]]) for d in keys},
        "esc": {d: np.concatenate([a["esc"][d], b["esc"][d]]) for d in ESC},
        "agree": np.concatenate([a["agree"], b["agree"]]),
        "calls": np.concatenate([a["calls"], b["calls"]]),
    }


def logistic(xtr, ytr, xte) -> np.ndarray:
    w = fit_logistic(xtr, ytr)
    return 1 / (1 + np.exp(-(np.hstack([xte, np.ones((len(xte), 1))]) @ w)))


def standardise(a, b):
    mu, sd = a.mean(0), a.std(0) + 1e-6
    return (a - mu) / sd, (b - mu) / sd


def lam_for(p: dict, d: dict, target: float) -> tuple[float, float, float]:
    """(λ, realised cost ratio, escalation share) of the grid point closest to ``target``."""
    rows = []
    for lam in LAMS:
        c = choose(p, lam)
        ratio = np.where(c == 4, d["esc"][4], d["esc"][2]).mean() / d["fixed"][4].mean()
        rows.append((abs(ratio - target), float(lam), float(ratio), float((c == 4).mean())))
    return min(rows)[1:]


def gain_vs_mix(correct, cost, d, idx) -> float:
    fp = [(float(d["fixed"][k][idx].mean()), float(d["y"][k][idx].mean())) for k in d["fixed"]]
    c = float(cost[idx].mean())
    if c > d["fixed"][4][idx].mean():  # beyond the fixed range: compare with fixed-4
        return float(correct[idx].mean() - d["y"][4][idx].mean())
    return float(correct[idx].mean() - hull_interp(fp, c))


def boot(stat, n: int) -> tuple[float, list[float]]:
    rng = np.random.default_rng(0)
    vals = [stat(rng.integers(0, n, n)) for _ in range(2000)]
    return stat(np.arange(n)), [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]


def apply(choice: np.ndarray, d: dict) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.where(choice == 4, d["y"][4], d["y"][2]),
        np.where(choice == 4, d["esc"][4], d["esc"][2]),
    )


def summary(choice, d) -> dict:
    correct, cost = apply(choice, d)
    n = len(choice)
    g, ci = boot(lambda i: gain_vs_mix(correct, cost, d, i), n)
    return {
        "accuracy": float(correct.mean()),
        "cost_ratio_vs_fixed4": float(cost.mean() / d["fixed"][4].mean()),
        "escalated": float((choice == 4).mean()),
        "cost_above_fixed4_flag": bool(cost.mean() > d["fixed"][4].mean()),
        "gain_vs_random_mix": g,
        "ci90": ci,
    }


def paired(a_correct, b_correct, n) -> dict:
    diff, ci = boot(lambda i: float(a_correct[i].mean() - b_correct[i].mean()), n)
    return {
        "difference": diff,
        "ci90": ci,
        "wins": int((a_correct > b_correct).sum()),
        "losses": int((a_correct < b_correct).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", default="results/phase_a/labels.jsonl")
    parser.add_argument("--out", default="results/router_confirm2")
    args = parser.parse_args()

    clean, pool = (load_set(*p) for p in TRAIN)
    tr = concat(clean, pool)
    labels = {r["id"]: r for r in map(json.loads, Path(args.labels).read_text().splitlines())}
    assert [i for i in tr["ids"] if i not in labels] == [], "training tasks without soft labels"
    soft = {
        d: np.array([(labels[i][f"g{d}"] + sum(labels[i][f"s{d}"])) / 5 for i in tr["ids"]])
        for d in ESC
    }
    for d in ESC:  # the soft labels' greedy part is the same run as the smoke rows
        g = np.array([labels[i][f"g{d}"] for i in tr["ids"]], float)
        assert (g == tr["y"][d]).all(), f"greedy labels at depth {d} differ"
    te = load_set(*TEST)
    assert len(te["ids"]) == 480 and set(te["y"]) == {1, 2, 3, 4}, "test set incomplete"
    # row order = request order: router_probe2 asserted the complete smoke file equals requests
    # [start, start + n) in order before probing
    run = json.loads((Path(TEST[0]) / "args.json").read_text())
    assert (run["tasks"], run["start"], run["n"]) == ("nestful", 240, 480), run
    seen = load_set(*SEEN)
    assert not set(te["ids"]) & set(seen["ids"]) and not set(te["ids"]) & set(tr["ids"])
    n = len(te["ids"])

    # out-of-fold soft-label probabilities on the 531 → operating points
    oof = {d: np.zeros(len(tr["ids"])) for d in ESC}
    for rep in range(REPEATS):
        order = np.random.default_rng(rep).permutation(len(tr["ids"]))
        for f in range(FOLDS):
            v = order[f::FOLDS]
            t = np.setdiff1d(order, v)
            a, b = standardise(tr["x"][t], tr["x"][v])
            for d in ESC:
                oof[d][v] += logistic(a, soft[d][t], b) / REPEATS
    ops = {target: lam_for(oof, tr, target) for target in (0.75, 0.85)}

    # frozen models on all 531
    def frozen(x):
        a, b = standardise(tr["x"], x)
        return {d: logistic(a, soft[d], b) for d in ESC}

    p_te, p_seen = frozen(te["x"]), frozen(seen["x"])
    choice = {t: choose(p_te, lam) for t, (lam, _, _) in ops.items()}

    # step-1 frozen router (router_confirm: 292 clean, greedy labels, 0.75x)
    oof1 = oof_probs(clean["x_raw"], clean["y"])
    lam1 = lam_for(oof1, clean, 0.75)[0]
    mu, sd = np.nanmean(clean["x_raw"], 0), np.nanstd(clean["x_raw"], 0) + 1e-6
    a1, b1 = np.nan_to_num((clean["x_raw"] - mu) / sd), np.nan_to_num((te["x_raw"] - mu) / sd)
    choice1 = choose({d: logistic(a1, clean["y"][d], b1) for d in ESC}, lam1)

    # LEV-style stopper refitted on all 531 (agreement of the 2- and 4-loop answers)
    la, lb = standardise(tr["x"][:, LEV_COLS], te["x"][:, LEV_COLS])
    p_tr = logistic(la, tr["agree"], la)
    tau = next(
        (t for t in np.linspace(0, 1, 201) if np.where(p_tr < t, 1.0, tr["agree"]).mean() >= 0.98),
        1.0,
    )
    choice_lev = np.where(logistic(la, tr["agree"], lb) < tau, 4, 2)

    correct = {t: apply(c, te)[0] for t, c in choice.items()}
    c1, _ = apply(choice1, te)
    first = np.arange(240)
    cost1 = apply(choice1, te)[1].mean()
    lam_eq = min(LAMS, key=lambda lam: abs(apply(choose(p_te, lam), te)[1].mean() - cost1))
    c_eq = apply(choose(p_te, lam_eq), te)[0]
    report = {
        "n_train": len(tr["ids"]),
        "n_test": n,
        "operating_points": {
            str(t): {"lambda": lam, "train_oof_cost_ratio": r, "train_oof_escalated": e}
            for t, (lam, r, e) in ops.items()
        },
        "fixed_accuracy": {d: float(te["y"][d].mean()) for d in sorted(te["y"])},
        "fixed_cost_ratio": {
            d: float(te["fixed"][d].mean() / te["fixed"][4].mean()) for d in sorted(te["fixed"])
        },
        "oracle_2v4": summary(np.where((te["y"][2] == 0) & (te["y"][4] == 1), 4, 2), te),
        "primary_0.75x": summary(choice[0.75], te),
        "S1_noninferiority_0.85x": {
            **summary(choice[0.85], te),
            "vs_fixed4": paired(correct[0.85], te["y"][4], n),
        },
        "S2_vs_step1_router": {
            "step1": {**summary(choice1, te), "lambda": lam1},
            "equal_cost_lambda": float(lam_eq),
            "equal_cost_ratio": float(
                apply(choose(p_te, lam_eq), te)[1].mean() / te["fixed"][4].mean()
            ),
            "paired_equal_cost_240_720": paired(c_eq, c1, n),
            "paired_equal_cost_240_480": paired(c_eq[first], c1[first], 240),
            "paired_own_lambda_240_720": paired(correct[0.75], c1, n),
        },
        "S3_vs_fixed3": {
            "paired_0.75x": paired(correct[0.75], te["y"][3], n),
            "paired_0.85x": paired(correct[0.85], te["y"][3], n),
        },
        "S4_lev_style": {**summary(choice_lev, te), "tau": float(tau)},
        "S6_auc_4_helps": auc(p_te[4] - p_te[2], ((te["y"][2] == 0) & (te["y"][4] == 1)) * 1.0),
        "S6_curve": [
            (round(c / te["fixed"][4].mean(), 3), round(acc, 3))
            for c, acc in policy_curve(p_te, te["y"], te["esc"], np.arange(n), ESC)
        ],
    }
    report["primary_0.75x"]["verdict"] = (
        "CONFIRMED"
        if report["primary_0.75x"]["ci90"][0] > 0
        else (
            "CONSISTENT" if report["primary_0.75x"]["gain_vs_random_mix"] > 0 else "NOT CONFIRMED"
        )
    )
    report["S1_noninferiority_0.85x"]["non_inferior_margin_0.03"] = bool(
        report["S1_noninferiority_0.85x"]["vs_fixed4"]["ci90"][0] > -0.03
    )
    strata = {
        "calls_le5": te["calls"] <= 5,
        "calls_gt5": te["calls"] > 5,
        "requests_240_480": np.arange(n) < 240,
        "requests_480_720": np.arange(n) >= 240,
    }
    report["S7_strata"] = {
        k: {"n": int(m.sum()), **summary(choice[0.75][m], subset(te, m))} for k, m in strata.items()
    }
    both = concat(seen, te)
    report["S5_pooled_0_720"] = {
        str(t): summary(choose({d: np.concatenate([p_seen[d], p_te[d]]) for d in ESC}, lam), both)
        for t, (lam, _, _) in ops.items()
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=1, default=float)
    (out / "confirm.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
