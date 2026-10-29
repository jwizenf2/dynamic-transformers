"""Phase A router development on the 531 training tasks (A2–A4, A7; development data only).

    python scripts/lab/phase_a_labels.py      # first: results/phase_a/labels.jsonl
    python scripts/lab/phase_a_router.py

Decision: start at 2 loops, after the probe tokens either stay (cost 2·T_2) or switch to 4 loops
(cost 4·T_4 + 2·k), as ``router_probe2_analysis`` (``escalate`` accounting, no shared prefix).
Labels and costs come from the greedy runs; out-of-fold scores from 5-fold x 5 repeats. Methods:

* ``esc12``: the 12 loop-1/2 features of ``features_escalate``, logistic per depth on greedy
  outcomes (the frozen router's recipe);
* ``esc12_soft``: same features, trained on success rates over greedy + 4 samples (A2);
* ``hidden``: loop-1/2 hidden states at the probe tokens (A3, pre-registered): pooling in {mean
  loop-2, mean loop-2 − loop-1, mean loop-2 at tokens where loops 1/2 disagree (mean loop-2 if
  none)}, standardised, PCA fitted in the training fold, k in {16, 32, 64}, L2 strength in
  {0.1, 1, 10}; the configuration is chosen per outer fold by inner 4-fold CV log-loss (nested);
  trained on soft labels; plus the 12 features appended;
* ``lev``: an LEV-style stopper (Song et al. 2026, arXiv 2610.20951) adapted to generation: LEV's
  readout features available at depth 2 (per loop 1–2: mean/max entropy, min margin; loop-1/2
  answer disagreements) predicting *agreement* of the 2-loop and 4-loop final answers (no
  correctness labels), escalating when predicted agreement is below a threshold set, as in LEV,
  on unlabelled training requests as the smallest value giving >= 98% agreement with the deepest
  answer.

Reported per method: AUC of the predicted benefit of 4 loops for the greedy label "4 helps";
the cost-aware policy curve (argmin_d λ·d/4 − P_d over d in {2, 4}); its gain over the best random
mix of fixed depths at the mid budget; accuracy at 0.85x and 0.80x fixed-4 cost; the cost at
which it reaches fixed-4 accuracy − 1 point. LEV is reported at its own threshold rule and as a
curve over thresholds. Development numbers only; the confirmatory test is NESTful [240, 720).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from phase_a_labels import auc
from router_probe2_analysis import features_escalate, hull_interp
from router_smoke_analysis import fit_logistic

M = "ouro-chain-depth-v3"
FOLDS, REPEATS = 5, 5


def load():
    labels = [json.loads(x) for x in Path("results/phase_a/labels.jsonl").read_text().splitlines()]
    smoke = {}
    for folder in (M, f"{M}_testpool"):
        for r in map(json.loads, Path(f"results/router_smoke/{folder}/examples.jsonl").open()):
            smoke[r["id"]] = r
    probes, hidden_dir = {}, {}
    for folder in (M, f"{M}_testpool"):
        for r in map(json.loads, Path(f"results/router_probe2/{folder}/examples.jsonl").open()):
            probes[r["id"]] = r
    for folder in (f"{M}_hidden", f"{M}_testpool_hidden"):
        for p in Path(f"results/router_probe2/{folder}/hidden").glob("*.npy"):
            hidden_dir[p.stem] = p
    ids = [r["id"] for r in labels]
    assert all(i in smoke and i in probes and i in hidden_dir for i in ids)
    for i in ids:
        pr = probes[i]
        pr["loop12_disagree"] = sum(t[0][3] != t[1][3] for t in pr["probe"])
    return labels, smoke, probes, hidden_dir


def hidden_features(path: Path, probe: dict) -> np.ndarray:
    h = np.load(path).astype(np.float32)  # [t, 2, H]
    t = min(len(h), len(probe["probe"]))
    h = h[:t]
    dis = np.array([probe["probe"][j][0][3] != probe["probe"][j][1][3] for j in range(t)])
    mean2 = h[:, 1].mean(0)
    diff = (h[:, 1] - h[:, 0]).mean(0)
    at_dis = h[dis, 1].mean(0) if dis.any() else mean2
    return np.stack([mean2, diff, at_dis])  # [3, H]


def standardise(a, b):
    mu, sd = a.mean(0), a.std(0) + 1e-6
    return (a - mu) / sd, (b - mu) / sd


def logistic_probs(xtr, ytr, xte, l2=1.0):
    w = fit_logistic(xtr, ytr, l2=l2)
    return 1 / (1 + np.exp(-(np.hstack([xte, np.ones((len(xte), 1))]) @ w)))


def pca_fit(x, k):
    mu = x.mean(0)
    _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
    return mu, vt[:k]


def log_loss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main() -> None:
    labels, smoke, probes, hidden_dir = load()
    ids = [r["id"] for r in labels]
    n = len(ids)
    g = {d: np.array([r[f"g{d}"] for r in labels], float) for d in (2, 4)}
    soft = {d: np.array([(r[f"g{d}"] + sum(r[f"s{d}"])) / 5 for r in labels]) for d in (2, 4)}
    tok = {
        d: np.array(
            [
                smoke[i]["runs"][f"d{d}_greedy"]["prompt_tokens"]
                + smoke[i]["runs"][f"d{d}_greedy"]["tokens"]
                for i in ids
            ],
            float,
        )
        for d in (2, 4)
    }
    k = np.array([len(probes[i]["probe"]) for i in ids], float)
    cost = {2: 2 * tok[2], 4: 4 * tok[4] + 2 * k}
    c4 = (4 * tok[4]).mean()
    agree = np.array(
        [
            smoke[i]["runs"]["d2_greedy"]["value"] == smoke[i]["runs"]["d4_greedy"]["value"]
            for i in ids
        ],
        float,
    )
    x12 = np.nan_to_num(np.stack([features_escalate(probes[i], smoke[i]["query"]) for i in ids]))
    xlev = x12[
        :, [0, 1, 2, 4, 5, 6, 9]
    ]  # per loop 1–2: mean/max entropy, min margin; disagreements
    xh = np.stack([hidden_features(hidden_dir[i], probes[i]) for i in ids])  # [n, 3, H]

    oof = {m: {2: np.zeros(n), 4: np.zeros(n)} for m in ("esc12", "esc12_soft", "hidden")}
    lev_score, lev_thr = np.zeros(n), []
    for rep in range(REPEATS):
        order = np.random.default_rng(rep).permutation(n)
        for f in range(FOLDS):
            te = order[f::FOLDS]
            tr = np.setdiff1d(order, te)
            a, b = standardise(x12[tr], x12[te])
            for d in (2, 4):
                oof["esc12"][d][te] += logistic_probs(a, g[d][tr], b) / REPEATS
                oof["esc12_soft"][d][te] += logistic_probs(a, soft[d][tr], b) / REPEATS
            # hidden: nested choice of (pooling, k, l2) by inner CV log-loss on soft labels
            best, best_loss = None, np.inf
            inner = np.array_split(np.random.default_rng(100 + rep).permutation(tr), 4)
            for pool in range(3):
                for kk in (16, 32, 64):
                    for l2 in (0.1, 1.0, 10.0):
                        losses = []
                        for v in inner:
                            t2 = np.setdiff1d(tr, v)
                            ha, hb = standardise(xh[t2, pool], xh[v, pool])
                            mu, comp = pca_fit(ha, kk)
                            za, zb = (ha - mu) @ comp.T, (hb - mu) @ comp.T
                            ea, eb = standardise(x12[t2], x12[v])
                            za, zb = np.hstack([za, ea]), np.hstack([zb, eb])
                            for d in (2, 4):
                                losses.append(
                                    log_loss(logistic_probs(za, soft[d][t2], zb, l2), g[d][v])
                                )
                        if np.mean(losses) < best_loss:
                            best, best_loss = (pool, kk, l2), float(np.mean(losses))
            pool, kk, l2 = best
            ha, hb = standardise(xh[tr, pool], xh[te, pool])
            mu, comp = pca_fit(ha, kk)
            za = np.hstack([(ha - mu) @ comp.T, a])
            zb = np.hstack([(hb - mu) @ comp.T, b])
            for d in (2, 4):
                oof["hidden"][d][te] += logistic_probs(za, soft[d][tr], zb, l2) / REPEATS
            # LEV-style: predict 2-vs-4 answer agreement (no correctness labels)
            la, lb = standardise(xlev[tr], xlev[te])
            p_tr = logistic_probs(la, agree[tr], la)
            p_te = logistic_probs(la, agree[tr], lb)
            # threshold: smallest τ with >= 98% of final answers agreeing with the deepest answer
            # on the training requests (escalated requests count as agreeing)
            thr = 1.0
            for tau in np.linspace(0, 1, 201):
                fin_agree = np.where(p_tr < tau, 1.0, agree[tr]).mean()
                if fin_agree >= 0.98:
                    thr = tau
                    break
            lev_thr.append(thr)
            lev_score[te] += (1 - p_te) / REPEATS  # escalation score

    fixed_pts = [((d * tok[d]).mean(), g[d].mean()) for d in (2, 4)]  # the mix of the two options

    def curve_from_scores(score):
        pts = []
        for q in np.linspace(0, 1, 101):
            esc = score >= np.quantile(score, 1 - q) if q > 0 else np.zeros(n, bool)
            pts.append(
                (
                    float(np.where(esc, cost[4], cost[2]).mean()),
                    float(np.where(esc, g[4], g[2]).mean()),
                )
            )
        return sorted(pts)

    def summarize(pts):
        xs = np.array([p[0] for p in pts]) / c4
        ys = np.maximum.accumulate(np.array([p[1] for p in pts]))
        mid = (fixed_pts[0][0] + fixed_pts[1][0]) / 2
        gain_mid = float(np.interp(mid / c4, xs, ys) - hull_interp(fixed_pts, mid))
        target = g[4].mean() - 0.01
        reach = xs[ys >= target]
        return {
            "gain_vs_random_mix_mid_budget": gain_mid,
            "acc_at_0.80x": float(np.interp(0.80, xs, ys)),
            "acc_at_0.85x": float(np.interp(0.85, xs, ys)),
            "cost_to_reach_fixed4_minus_1pt": float(reach.min()) if len(reach) else None,
        }

    helps = ((g[2] == 0) & (g[4] == 1)).astype(float)
    report = {
        "n": n,
        "fixed": {
            "d2": [float(fixed_pts[0][0] / c4), float(g[2].mean())],
            "d4": [1.0, float(g[4].mean())],
        },
        "methods": {},
    }
    for m, p in oof.items():
        score = p[4] - p[2]
        report["methods"][m] = {
            "auc_4_helps": auc(score, helps),
            **summarize(curve_from_scores(score)),
        }
    report["methods"]["lev"] = {
        "auc_4_helps": auc(lev_score, helps),
        **summarize(curve_from_scores(lev_score)),
    }
    esc = (1 - lev_score) < np.mean(lev_thr)  # escalate if predicted agreement < τ
    report["methods"]["lev"]["own_threshold"] = {
        "tau_mean": float(np.mean(lev_thr)),
        "escalated": float(esc.mean()),
        "cost_ratio": float(np.where(esc, cost[4], cost[2]).mean() / c4),
        "accuracy": float(np.where(esc, g[4], g[2]).mean()),
    }
    out = Path("results/phase_a")
    text = json.dumps(report, indent=1)
    (out / "router_dev.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
