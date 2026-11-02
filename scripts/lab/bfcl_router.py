"""Short-horizon control: does early-signal depth routing work on single-turn BFCL? (analysis)

    PYTHONPATH=scripts/lab python scripts/lab/bfcl_router.py

Pre-registered 2026-11-02, before the BFCL probe run. Data: chain v3's BFCL v3 non-live
single-turn outcomes at 1–4 loops (``ouro_sft_eval.py --task bfcl --n 0``, 1,000 prompts,
``results/chains_v3_run/ouro_sft_eval/chain-depth-v3/bfcl``) joined by id with the 48-token
probe of ``bfcl_probe.py`` (``results/bfcl_probe/ouro-chain-depth-v3``).

Prediction from the router diagnosis (uncertainty is local, consequence builds up over long
chains): with one assistant turn, early signals should predict the benefit of depth better than
on NESTful (AUC .60 for "4 helps" with the same 12 loop-1/2 features, .64 on MathQA).
Primary: AUC of the out-of-fold score P_4 − P_2 (12 loop-1/2 features of ``features_escalate``,
query features set to 0 since BFCL prompts are not stored; logistic per depth; 5x5 CV) for
"4 helps" (wrong at 2, right at 4) on BFCL, with a bootstrap 90% interval over prompts;
SUPPORTED if the interval's lower bound exceeds .60 (the NESTful value). Power note: 4 loops help
only ~5–7% of BFCL prompts (chain v3 .846 → .879 from 2 to 4 loops), ~60 positives, so the
interval is roughly ±.07 and SUPPORTED needs an AUC of about .67 or more. Secondary: the rescue
AUC among 2-loop failures; the policy's gain over the 2–4 random mix at the mid budget (cost:
d·(prompt + generated tokens) per depth, escalation 4·T_4 + 2k as on NESTful); the same for 1 vs
2 loops with loop-1 features only (loop-1 entropy mean/max, min margin, mean gate); fork-free
share (requests whose 2- and 4-loop outputs are identical) per outcome pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from phase_a_labels import auc
from router_diagnosis import predict, standardise
from router_probe2_analysis import features_escalate, hull_interp

OUTCOMES = Path("results/chains_v3_run/ouro_sft_eval/chain-depth-v3/bfcl/examples.jsonl")
PROBE = Path("results/bfcl_probe/ouro-chain-depth-v3/examples.jsonl")


def cv_scores(x, ya, yb, repeats=5):
    n = len(x)
    s = np.zeros(n)
    for rep in range(repeats):
        order = np.random.default_rng(rep).permutation(n)
        for f in range(5):
            te = order[f::5]
            tr = np.setdiff1d(order, te)
            a, b = standardise(x[tr], x[te])
            s[te] += (predict(a, yb[tr], b) - predict(a, ya[tr], b)) / repeats
    return s


def boot_auc(score, y) -> list[float]:
    rng = np.random.default_rng(0)
    n = len(y)
    vals = []
    for _ in range(2000):
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n:
            vals.append(auc(score[i], y[i]))
    return [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]


def policy_gain(score, ya, yb, ca, cb) -> float:
    """Gain over the random mix at the mid budget for escalation by score quantile."""
    base = cb.mean()
    pts = []
    for q in np.linspace(0, 1, 101):
        esc = score >= np.quantile(score, 1 - q) if q > 0 else np.zeros(len(score), bool)
        pts.append(
            (float(np.where(esc, cb, ca).mean() / base), float(np.where(esc, yb, ya).mean()))
        )
    pts.sort()
    xs = np.array([p[0] for p in pts])
    ys = np.maximum.accumulate(np.array([p[1] for p in pts]))
    fixed = [(ca.mean() / base, ya.mean()), (1.0, yb.mean())]
    mid = (fixed[0][0] + 1.0) / 2
    return float(np.interp(mid, xs, ys) - hull_interp(fixed, mid))


def main() -> None:
    outcomes = {}
    for r in map(json.loads, open(OUTCOMES)):
        outcomes.setdefault(r["id"], {})[r["steps"]] = r
    probes = [json.loads(x) for x in open(PROBE)]
    for p in probes:
        p["loop12_disagree"] = sum(t[0][3] != t[1][3] for t in p["probe"])
    probes = [p for p in probes if all(d in outcomes.get(p["id"], {}) for d in (1, 2, 4))]
    n = len(probes)
    y = {d: np.array([outcomes[p["id"]][d]["correct"] for p in probes], float) for d in (1, 2, 4)}
    tok = {
        d: np.array(
            [p["prompt_tokens"] + outcomes[p["id"]][d]["out_tokens"] for p in probes], float
        )
        for d in (1, 2, 4)
    }
    k = np.array([len(p["probe"]) for p in probes], float)
    x12 = np.nan_to_num(np.stack([features_escalate(p, "") for p in probes]))
    x1 = x12[:, [0, 1, 2, 3]]  # loop-1 entropy mean/max, min margin, mean gate
    same = np.array(
        [outcomes[p["id"]][2]["text"] == outcomes[p["id"]][4]["text"] for p in probes], bool
    )
    report = {"n": n, "accuracy": {d: float(y[d].mean()) for d in (1, 2, 4)}}
    for a, b, x, name in ((2, 4, x12, "2v4"), (1, 2, x1, "1v2")):
        helps = ((y[a] == 0) & (y[b] == 1)).astype(float)
        s = cv_scores(x, y[a], y[b])
        m = y[a] == 0
        report[name] = {
            "helps_share": float(helps.mean()),
            "auc_helps": auc(s, helps),
            "auc_helps_ci90": boot_auc(s, helps),
            "rescue_auc_among_failures": auc(cv_scores(x[m], y[a][m], y[b][m]), y[b][m]),
            "gain_vs_mix_mid_budget": policy_gain(
                s, y[a], y[b], a * tok[a], b * tok[b] + (2 * k if b == 4 else 0)
            ),
        }
    patterns = {
        "helps": (y[2] == 0) & (y[4] == 1),
        "hurts": (y[2] == 1) & (y[4] == 0),
        "same_outcome": y[2] == y[4],
    }
    report["identical_2v4_output"] = {g: float(same[m].mean()) for g, m in patterns.items()}
    lo = report["2v4"]["auc_helps_ci90"][0]
    report["primary"] = {
        "auc_helps_2v4": report["2v4"]["auc_helps"],
        "ci90": report["2v4"]["auc_helps_ci90"],
        "nestful_reference": 0.60,
        "verdict": "SUPPORTED" if lo > 0.60 else "NOT SUPPORTED",
    }
    out = Path("results/bfcl_router")
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=1, default=float)
    (out / "report.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
