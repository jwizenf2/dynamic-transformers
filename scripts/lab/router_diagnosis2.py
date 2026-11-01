"""Router diagnosis, tier 1: rescuability and fork-point signals (offline, exploratory).

    PYTHONPATH=scripts/lab python scripts/lab/router_diagnosis2.py

Follows ``router_diagnosis.py`` (plan: docs/research/router_diagnosis_2026-11-01.md).

R1 Rescuability. Among requests 2 loops gets wrong, can any available signal tell those 4 loops
   fixes from those it does not? AUC, 5x5 CV within the subset (the subset is chosen with the
   true 2-loop outcome, i.e. as if a perfect verifier of 2-loop answers existed). Feature sets:
   MathQA 531: the 12 early loop-1/2 features (L0); + loop-1/2 hidden states at the probe tokens
   (mean loop-2 state and mean h_2 − h_1, PCA 32 fitted in the training fold); + whole 2-loop
   trace features (``phase_a_trace_analysis.point_features`` at "end": token-level loop-1/2
   signals over the full dialogue, tool-response surprisal, turns, finished, errors); all; each
   also trained on soft labels (4-loop success rate over greedy + 4 samples). NESTful 720: L0,
   + loop-3/4 readouts and realised revisions (``router_diagnosis`` L2); greedy and stable labels
   (stable: right at >= 2 of depths 4, 5, 6).
R2 Fork-point signals (MathQA traces, which reproduce the 2-loop greedy dialogues exactly). For
   each request whose 2- and 4-loop dialogues differ, the fork is the first differing assistant
   token. At the fork, the 2-loop trace's loop-1/2 signals (loop-2 entropy, top-2 margin,
   KL(p_2 ‖ p_1), loop-1/2 argmax disagreement, log p_1 of the emitted token) are ranked against
   every other token of the same dialogue (percentile; 1 = most uncertain). Reported: median fork
   percentile for rescued forks ("4 helps") and for neutral forks (same outcome at 2 and 4), AUC
   of the fork-token signal for rescued vs neutral forks, and how often the most uncertain token
   of the dialogue (by each signal) is the fork or within 3 tokens before it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from phase_a_labels import auc
from phase_a_trace_analysis import point_features
from router_diagnosis import (
    divergence,
    load,
    predict,
    standardise,
    token_types,
    tokenizer,
)
from router_probe2_analysis import features_escalate, features_v2

M = "ouro-chain-depth-v3"
SIGNALS = ("ent2", "mar2", "kl21", "disagree", "lp1")


def cv_auc(x, y_train, y_eval, pca: tuple | None = None, repeats: int = 5) -> float:
    """AUC of out-of-fold logistic scores; ``pca`` = (hidden array, k) appended after PCA."""
    n = len(x)
    s = np.zeros(n)
    for rep in range(repeats):
        order = np.random.default_rng(rep).permutation(n)
        for f in range(5):
            te = order[f::5]
            tr = np.setdiff1d(order, te)
            a, b = standardise(x[tr], x[te])
            if pca is not None:
                h, k = pca
                ha, hb = standardise(h[tr], h[te])
                _, _, vt = np.linalg.svd(ha - ha.mean(0), full_matrices=False)
                a = np.hstack([a, (ha - ha.mean(0)) @ vt[:k].T])
                b = np.hstack([b, (hb - ha.mean(0)) @ vt[:k].T])
            s[te] += predict(a, y_train[tr], b, l2=10.0 if pca else 1.0) / repeats
    return auc(s, y_eval)


def traces() -> dict:
    out = {}
    for folder in (f"{M}_clean", f"{M}_testpool"):
        for r in map(json.loads, open(f"results/router_trace/{folder}/examples.jsonl")):
            out[r["id"]] = r
    return out


def rescue_mathqa(rows, tr) -> dict:
    labels = {r["id"]: r for r in map(json.loads, open("results/phase_a/labels.jsonl"))}
    y2 = np.array([r["runs"]["d2_greedy"]["correct"] for r in rows], float)
    y4 = np.array([r["runs"]["d4_greedy"]["correct"] for r in rows], float)
    soft4 = np.array([(labels[r["id"]]["g4"] + sum(labels[r["id"]]["s4"])) / 5 for r in rows])
    m = y2 == 0
    x0 = np.nan_to_num(np.stack([features_escalate(r["probe2"], r["query"]) for r in rows]))
    hid = []
    for r in rows:
        h = np.load(r["hidden"]).astype(np.float32)
        hid.append(np.concatenate([h[:, 1].mean(0), (h[:, 1] - h[:, 0]).mean(0)]))
    hid = np.array(hid)
    xt = np.nan_to_num(np.stack([point_features(tr[r["id"]], "end")[0] for r in rows]))
    out = {"n_subset": int(m.sum()), "rescued": int(y4[m].sum())}
    for name, x, h in (
        ("L0", x0, None),
        ("L0+hidden", x0, hid),
        ("L0+trace", np.hstack([x0, xt]), None),
        ("all", np.hstack([x0, xt]), hid),
    ):
        pca = (h[m], 32) if h is not None else None
        out[name] = {
            "greedy_label": cv_auc(x[m], y4[m], y4[m], pca),
            "soft_label": cv_auc(x[m], soft4[m], y4[m], pca),
        }
    return out


def rescue_nestful(rows, tok) -> dict:
    deep = {}
    for t in ("nestful_deep", "nestful240_deep"):
        for r in map(json.loads, open(f"results/router_smoke/{M}_{t}/examples.jsonl")):
            deep[r["id"]] = r["runs"]
    y2 = np.array([r["runs"]["d2_greedy"]["correct"] for r in rows], float)
    y4 = np.array([r["runs"]["d4_greedy"]["correct"] for r in rows], float)
    k = y4 + np.array(
        [
            deep[r["id"]]["d5_greedy"]["correct"] + deep[r["id"]]["d6_greedy"]["correct"]
            for r in rows
        ]
    )
    stable = (k >= 2).astype(float)
    m = y2 == 0
    x0 = np.nan_to_num(np.stack([features_escalate(r["probe2"], r["query"]) for r in rows]))
    xv2 = np.nan_to_num(np.stack([features_v2(r["probe2"], r["query"]) for r in rows]))
    rev = []
    for r in rows:
        types = token_types(tok, r["probe2"])
        dis = [t[1][3] == 0 for t in r["probe2"]["probe"]]
        rev.append(
            [
                sum(dis),
                sum(d for d, t in zip(dis, types, strict=False) if t == "function_name"),
                sum(d for d, t in zip(dis, types, strict=False) if t == "number"),
            ]
        )
    x2 = np.hstack([x0, xv2[:, 8:16], xv2[:, 17:19], np.array(rev, float)])
    out = {
        "n_subset": int(m.sum()),
        "rescued": int(y4[m].sum()),
        "stable_rescued": int(stable[m].sum()),
    }
    for name, x in (("L0", x0), ("L2", x2)):
        out[name] = {
            "greedy_label": cv_auc(x[m], y4[m], y4[m]),
            "stable_label": cv_auc(x[m], stable[m], stable[m]),
        }
    return out


def fork_signals(rows, tr, tok) -> dict:
    y2 = {r["id"]: r["runs"]["d2_greedy"]["correct"] for r in rows}
    y4 = {r["id"]: r["runs"]["d4_greedy"]["correct"] for r in rows}
    pct = {s: {"rescued": [], "neutral": []} for s in SIGNALS}
    at_fork = {s: {"rescued": [], "neutral": []} for s in SIGNALS}
    top_hits = {s: {"rescued": [], "neutral": []} for s in SIGNALS}
    for r in rows:
        t = tr[r["id"]]
        a, b = r["runs"]["d2_greedy"]["texts"], r["runs"]["d4_greedy"]["texts"]
        pos, turn = divergence(tok, a, b)
        if pos >= 10**6 or turn >= len(t["turn_signals"]):
            continue
        group = (
            "rescued"
            if (not y2[r["id"]] and y4[r["id"]])
            else ("neutral" if y2[r["id"]] == y4[r["id"]] else None)
        )
        if group is None:
            continue
        before = sum(len(tok.encode(x)) for x in a[:turn])
        j = pos - before
        sig = t["turn_signals"]
        series = {
            "ent2": np.concatenate([s["ent2"] for s in sig]),
            "mar2": -np.concatenate([s["mar2"] for s in sig]),
            "kl21": np.concatenate([s["kl21"] for s in sig]),
            "disagree": 1 - np.concatenate([s["agree"] for s in sig]),
            "lp1": -np.concatenate([s["lp1"] for s in sig]),
        }
        offset = sum(len(s["ent2"]) for s in sig[:turn])
        idx = offset + j
        if j >= len(sig[turn]["ent2"]):
            continue
        for s, v in series.items():
            pct[s][group].append(float((v < v[idx]).mean() + 0.5 * (v == v[idx]).mean()))
            at_fork[s][group].append(float(v[idx]))
            top = int(np.argmax(v))
            top_hits[s][group].append(idx - 3 <= top <= idx)
    out = {}
    for s in SIGNALS:
        r_, n_ = np.array(at_fork[s]["rescued"]), np.array(at_fork[s]["neutral"])
        out[s] = {
            "n_rescued": len(r_),
            "n_neutral": len(n_),
            "median_percentile_rescued": float(np.median(pct[s]["rescued"])),
            "median_percentile_neutral": float(np.median(pct[s]["neutral"])),
            "auc_rescued_vs_neutral_fork": auc(
                np.concatenate([r_, n_]), np.r_[np.ones(len(r_)), np.zeros(len(n_))]
            ),
            "dialogue_max_at_or_just_before_fork_rescued": float(np.mean(top_hits[s]["rescued"])),
            "dialogue_max_at_or_just_before_fork_neutral": float(np.mean(top_hits[s]["neutral"])),
        }
    return out


def main() -> None:
    tok = tokenizer()
    mathqa, nestful = load("mathqa_531"), load("nestful_720")
    tr = traces()
    report = {
        "R1_rescue_auc": {
            "mathqa_531": rescue_mathqa(mathqa, tr),
            "nestful_720": rescue_nestful(nestful, tok),
        },
        "R2_fork_signals_mathqa": fork_signals(mathqa, tr, tok),
    }
    out = Path("results/router_diagnosis")
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=1, default=float)
    (out / "tier1.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
