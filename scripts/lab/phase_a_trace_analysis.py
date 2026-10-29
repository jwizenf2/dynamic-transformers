"""Late decision points for the escalation router (A8, pre-registered 2026-10-29; development data).

    python scripts/lab/phase_a_trace_analysis.py

Data: ``router_trace.py`` 2-loop traces (labels: each trace's own correctness), the existing 4-loop
greedy outcomes and token counts (``router_smoke`` rows), the early 12-feature probe
(``router_probe2`` rows), for the 292 clean MathQA-val tasks and the 239 MathQA-test pool tasks.

Decision points t: "early" (first ~45 tokens: the 12 escalate features), "tool1" (after the first
assistant turn and its tool result), "turn2" (after two turns and their tool results), "end" (the
whole 2-loop run: a cascade). A request that ends before t is decided at its end. Policy: stay at 2
loops (cost 2·T_2, T_2 = the trace's processed tokens) or restart at 4 loops from the prompt,
reusing
the prompt's loop-1/2 caches: 2·(work so far − P) + 4·T_4 (P = prompt tokens; "early" wastes the
2·k probe tokens). Features at t use only turns <= t; end-only fields (finished, turn-limit hit,
errors, truncated turns) only at "end". Model: L2 logistic per depth (P_2 on the trace outcome, P_4
on the 4-loop outcome) on standardised aggregated features + the 12 early features; score
P_4 − P_2; out-of-fold over 5x5 CV; curve over escalation shares. Pre-registered selection: the
decision point with the highest accuracy at 0.90x fixed-4 cost on the clean tasks, replication
checked on the pool tasks (5x5 CV within the pool, and trained on clean → applied to pool).
Also reported: gain over the 2-4 random mix at the mid budget, accuracy at 0.85x and 1.0x, and the
cascade's best accuracy at <= 1.0x.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from phase_a_labels import auc
from router_probe2_analysis import features_escalate, hull_interp
from router_smoke_analysis import fit_logistic

M = "ouro-chain-depth-v3"
POINTS = ("early", "tool1", "turn2", "end")
TOKEN_KEYS = ("ent2", "mar2", "gate2", "kl21", "agree", "lp1", "change")


def load(set_name: str):
    smoke_dir, probe_dir, trace_dir = {
        "clean": (M, M, f"{M}_clean"),
        "pool": (f"{M}_testpool", f"{M}_testpool", f"{M}_testpool"),
    }[set_name]
    smoke = {
        r["id"]: r
        for r in map(json.loads, open(f"results/router_smoke/{smoke_dir}/examples.jsonl"))
    }
    probes = {
        r["id"]: r
        for r in map(json.loads, open(f"results/router_probe2/{probe_dir}/examples.jsonl"))
    }
    traces = [json.loads(x) for x in open(f"results/router_trace/{trace_dir}/examples.jsonl")]
    for p in probes.values():
        p["loop12_disagree"] = sum(t[0][3] != t[1][3] for t in p["probe"])
    return smoke, probes, traces


def point_features(tr: dict, t: str) -> tuple[np.ndarray, float]:
    """(features using only turns <= t, tokens processed so far excluding the prompt)."""
    sig = tr["turn_signals"]
    n_turns = {"tool1": 1, "turn2": 2, "end": len(sig)}[t]
    use = sig[:n_turns]
    tools = [s["tool_before"] for s in sig[1 : n_turns + 1] if s["tool_before"]]
    if t == "end":
        tools = [s["tool_before"] for s in sig[1:] if s["tool_before"]]
    tok = {k: np.concatenate([np.array(s[k], float) for s in use]) for k in (*TOKEN_KEYS, "digit")}
    dig = tok["digit"] == 1

    def on(mask, key, fn, default):
        return float(fn(tok[key][mask])) if mask.any() else default

    f = [
        tok["kl21"].mean(),
        tok["kl21"].max(),
        1 - tok["agree"].mean(),
        on(dig, "agree", lambda a: 1 - a.mean(), 0.0),
        on(dig, "mar2", np.min, 20.0),
        on(dig, "lp1", np.mean, 0.0),
        on(dig, "lp1", np.min, 0.0),
        tok["ent2"].mean(),
        tok["ent2"].max(),
        tok["gate2"].mean(),
        tok["change"].mean(),
    ]
    sd = [np.array(x["surprisal"])[np.array(x["digit"], bool)] for x in tools]
    sd = np.concatenate(sd) if sd else np.zeros(0)
    f += [
        float(sd.mean()) if len(sd) else 0.0,
        float(sd.max()) if len(sd) else 0.0,
        float(any(x["error"] for x in tools)),
        float(len(use)),
        float(sum(s["n_tokens"] for s in use)),
    ]
    if t == "end":
        f += [
            float(tr["finished"]),
            float(tr["turns"] >= 8 and not tr["finished"]),
            float(tr["errors"]),
            float(any(s["n_tokens"] >= 256 for s in sig)),
        ]
    spent = sum(s["n_tokens"] for s in use) + sum(x["len"] for x in tools)
    return np.array(f, float), float(spent)


def build(set_name: str):
    smoke, probes, traces = load(set_name)
    ids = [tr["id"] for tr in traces]
    y2 = np.array([tr["correct"] for tr in traces], float)
    y4 = np.array([smoke[i]["runs"]["d4_greedy"]["correct"] for i in ids], float)
    t2 = np.array([tr["prompt_tokens"] + tr["tokens"] for tr in traces], float)
    t4 = np.array(
        [
            smoke[i]["runs"]["d4_greedy"]["prompt_tokens"] + smoke[i]["runs"]["d4_greedy"]["tokens"]
            for i in ids
        ],
        float,
    )
    k = np.array([len(probes[i]["probe"]) for i in ids], float)
    early = np.nan_to_num(np.stack([features_escalate(probes[i], smoke[i]["query"]) for i in ids]))
    feats, esc_cost = {"early": early}, {"early": 4 * t4 + 2 * k}
    for t in POINTS[1:]:
        rows = [point_features(tr, t) for tr in traces]
        feats[t] = np.hstack([np.stack([r[0] for r in rows]), early])
        esc_cost[t] = 4 * t4 + 2 * np.array([r[1] for r in rows])
    match = np.mean(
        [tr["correct"] == smoke[tr["id"]]["runs"]["d2_greedy"]["correct"] for tr in traces]
    )
    return {
        "y2": y2,
        "y4": y4,
        "c2": 2 * t2,
        "c4": 4 * t4,
        "feats": feats,
        "esc": esc_cost,
        "match_d2_greedy": float(match),
        "n": len(ids),
    }


def standardise(a, b):
    mu, sd = a.mean(0), a.std(0) + 1e-6
    return (a - mu) / sd, (b - mu) / sd


def scores_cv(d, t):
    n, x = d["n"], d["feats"][t]
    out = np.zeros(n)
    for rep in range(5):
        order = np.random.default_rng(rep).permutation(n)
        for f in range(5):
            te = order[f::5]
            tr = np.setdiff1d(order, te)
            a, b = standardise(x[tr], x[te])
            bb = np.hstack([b, np.ones((len(b), 1))])
            p = [1 / (1 + np.exp(-(bb @ fit_logistic(a, d[y][tr])))) for y in ("y2", "y4")]
            out[te] += (p[1] - p[0]) / 5
    return out


def scores_transfer(src, dst, t):
    a, b = standardise(src["feats"][t], dst["feats"][t])
    bb = np.hstack([b, np.ones((len(b), 1))])
    p = [1 / (1 + np.exp(-(bb @ fit_logistic(a, src[y])))) for y in ("y2", "y4")]
    return p[1] - p[0]


def evaluate(d, t, score):
    base = d["c4"].mean()
    pts = []
    for q in np.linspace(0, 1, 201):
        esc = score >= np.quantile(score, 1 - q) if q > 0 else np.zeros(d["n"], bool)
        pts.append(
            (
                float(np.where(esc, d["esc"][t], d["c2"]).mean() / base),
                float(np.where(esc, d["y4"], d["y2"]).mean()),
            )
        )
    pts.sort()
    xs = np.array([p[0] for p in pts])
    ys = np.maximum.accumulate(np.array([p[1] for p in pts]))
    fixed = [(d["c2"].mean() / base, d["y2"].mean()), (1.0, d["y4"].mean())]
    mid = (fixed[0][0] + 1.0) / 2

    def at(c):
        return float(np.interp(c, xs, ys)) if c >= xs[0] else float("nan")

    helps = ((d["y2"] == 0) & (d["y4"] == 1)).astype(float)
    return {
        "auc_4_helps": auc(score, helps),
        "acc_at_0.85x": at(0.85),
        "acc_at_0.90x": at(0.90),
        "acc_at_1.00x": at(1.00),
        "gain_vs_mix_mid": float(at(mid) - hull_interp(fixed, mid)),
        "min_cost": float(xs[0]),
    }


def main() -> None:
    data = {s: build(s) for s in ("clean", "pool")}
    report = {}
    for s, d in data.items():
        report[s] = {
            "n": d["n"],
            "match_trace_vs_d2_greedy": d["match_d2_greedy"],
            "fixed_d2": [float(d["c2"].mean() / d["c4"].mean()), float(d["y2"].mean())],
            "fixed_d4": [1.0, float(d["y4"].mean())],
            "points_cv": {t: evaluate(d, t, scores_cv(d, t)) for t in POINTS},
        }
    report["pool"]["points_trained_on_clean"] = {
        t: evaluate(data["pool"], t, scores_transfer(data["clean"], data["pool"], t))
        for t in POINTS
    }
    chosen = max(POINTS, key=lambda t: report["clean"]["points_cv"][t]["acc_at_0.90x"])
    report["selection"] = {"rule": "max clean CV accuracy at 0.90x", "chosen": chosen}
    out = Path("results/phase_a")
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=1)
    (out / "trace_points.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
