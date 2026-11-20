"""Pre-registered analysis of the late-switch collection (docs/research/late_switch_2026-11-20.md).

PYTHONPATH=scripts/lab python scripts/lab/late_switch_analysis.py --model ouro-chain-depth-v3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from phase_a_labels import auc
from router_diagnosis import divergence, tokenizer
from router_probe2_analysis import features_escalate
from router_window_hidden import cv_scores, window_features

SETS = {
    # set: (late_switch tags, router_smoke tags with the 4-loop runs, router_probe2 tags)
    "mathqa": (["mathqa_clean", "mathqa_pool"], ["", "_testpool"], ["", "_testpool"]),
    "nestful": (["nestful"], ["_nestful", "_nestful240"], ["_nestful", "_nestful240"]),
    "gsm8k": (["gsm8k"], ["_gsm8k"], ["_gsm8k"]),
}
POINTS = ("64", "128")


def load_set(model: str, name: str, base_model: str) -> list[dict]:
    ls_tags, smoke_tags, probe_tags = SETS[name]
    d4, probes = {}, {}
    for t in smoke_tags:
        for r in map(json.loads, open(f"results/router_smoke/{base_model}{t}/examples.jsonl")):
            d4[r["id"]] = (r["runs"]["d4_greedy"], r["query"])
    for t in probe_tags:
        for r in map(json.loads, open(f"results/router_probe2/{base_model}{t}/examples.jsonl")):
            r["loop12_disagree"] = sum(x[0][3] != x[1][3] for x in r["probe"])
            probes[r["id"]] = r
    rows = []
    for t in ls_tags:
        for r in map(json.loads, open(f"results/late_switch/{model}_{t}/examples.jsonl")):
            run4, query = d4[r["id"]]
            fork = fork_step(r, run4)
            rows.append(
                {
                    "id": r["id"],
                    "fork": fork,
                    "trace": r,
                    "x12": np.nan_to_num(features_escalate(probes[r["id"]], query)),
                    "y2": float(r["correct"]),
                    "y4": float(run4["correct"]),
                    "t2": float(r["prompt_tokens"] + r["tokens"]),
                    "t4": float(run4["prompt_tokens"] + run4["tokens"]),
                    "switch": r["switch"],
                }
            )
    return rows


_TOK = None


def fork_step(trace: dict, run4: dict) -> float:
    """Trace step of the first token where the 2-loop dialogue differs from the stored 4-loop one
    (inf if identical); stored 4-loop turn texts are cut at 300 characters, so a difference
    after that point within a turn shows up later (2% of turns are that long)."""
    global _TOK
    if _TOK is None:
        _TOK = tokenizer()
    pos, turn = divergence(_TOK, trace["texts"], run4["texts"])
    if pos >= 10**6:
        return float("inf")
    steps = [len(s["ent2"]) for s in trace["turn_signals"]]
    if turn >= len(steps):
        return float(sum(steps))
    return float(sum(steps[:turn]) + pos - sum(len(_TOK.encode(t)) for t in trace["texts"][:turn]))


def outcome_cost(r: dict, point: str) -> tuple[float, float]:
    s = r["switch"][point]
    if s.get("ended"):
        return r["y4"], 2 * r["t2"] + 4 * r["t4"]
    total = s["context_tokens"] + s["generated_after"] + s["prompt_tokens_after"]
    return float(s["correct"]), 4.0 * total


def policy(rows, point, score, idx=None) -> dict:
    idx = np.arange(len(rows)) if idx is None else idx
    R = [rows[i] for i in idx]
    sc = score[idx]
    base = 4 * np.mean([r["t4"] for r in R])
    c2 = 2 * np.mean([r["t2"] for r in R]) / base
    a2, a4 = np.mean([r["y2"] for r in R]), np.mean([r["y4"] for r in R])
    oc = [outcome_cost(r, point) for r in R]
    pts = []
    for q in np.linspace(0, 1, 101):
        esc = sc >= np.quantile(sc, 1 - q) if q > 0 else np.zeros(len(R), bool)
        acc = np.where(esc, [o[0] for o in oc], [r["y2"] for r in R]).mean()
        cost = np.where(esc, [o[1] for o in oc], [2 * r["t2"] for r in R]).mean() / base
        pts.append((float(cost), float(acc)))

    def best_under(cap):
        ok = [p for p in pts if p[0] <= cap]
        return max(ok, key=lambda p: p[1]) if ok else (np.nan, np.nan)

    c90, acc90 = best_under(0.90)
    mix90 = a2 + (a4 - a2) * (c90 - c2) / (1 - c2)
    c100, acc100 = best_under(1.0)
    return {
        "cost": c90,
        "acc": acc90,
        "gain_vs_mix": acc90 - mix90,
        "acc_at_1.0x": acc100,
        "fixed_d2": float(a2),
        "fixed_d4": float(a4),
        "fixed_d2_cost": float(c2),
    }


def analyse(rows) -> dict:
    out = {"n": len(rows)}
    y2 = np.array([r["y2"] for r in rows])
    y4 = np.array([r["y4"] for r in rows])
    for point in POINTS:
        ys = np.array([outcome_cost(r, point)[0] for r in rows])
        ended = np.array([bool(r["switch"][point].get("ended")) for r in rows])
        label = ((ys == 1) & (y2 == 0)).astype(float)
        feats = [window_features(r, int(point)) for r in rows]
        x = np.stack([f[0] for f in feats])
        timely = np.array([int(point) <= r["fork"] for r in rows]) & ~ended
        res = {
            "validation_switch_equals_d4_before_divergence": {
                "n": int(timely.sum()),
                "agreement": float((ys[timely] == y4[timely]).mean()) if timely.any() else None,
            },
            "switch_accuracy": float(ys.mean()),
            "ended_share": float(ended.mean()),
            "recovery": {
                "wrong_at_2_and_4_right": float(ys[(y2 == 0) & (y4 == 1)].mean()),
                "wrong_at_2_and_4_wrong": float(ys[(y2 == 0) & (y4 == 0)].mean()),
                "right_at_2_lost": float(1 - ys[y2 == 1].mean()),
            },
            "oracle": policy(rows, point, label + 1e-6 * np.arange(len(rows)) / len(rows)),
        }
        for model in ("logistic", "mlp"):
            s = cv_scores(x, label, model)
            res[model] = {"auc_switch_helps": auc(s, label), **policy(rows, point, s), "scores": s}
        out[point] = res
    return out


def bootstrap_gain(rows, point, scores) -> list[float]:
    rng = np.random.default_rng(0)
    n = len(rows)
    vals = [policy(rows, point, scores, rng.integers(0, n, n))["gain_vs_mix"] for _ in range(2000)]
    return [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="ouro-chain-depth-v3", help="late_switch folder prefix")
    ap.add_argument("--base-model", default="ouro-chain-depth-v3", help="router_smoke prefix")
    args = ap.parse_args()
    report = {}
    for name in SETS:
        try:
            rows = load_set(args.model, name, args.base_model)
        except FileNotFoundError as exc:
            report[name] = {"missing": str(exc)}
            continue
        report[name] = analyse(rows)
        report[name]["_rows"] = rows
    dev = report["mathqa"]
    choice = max(
        ((p, m) for p in POINTS for m in ("logistic", "mlp")),
        key=lambda pm: dev[pm[0]][pm[1]]["gain_vs_mix"],
    )
    report["selection"] = {"point": choice[0], "model": choice[1]}
    for name in ("nestful", "gsm8k"):
        if "missing" in report[name]:
            continue
        r = report[name][choice[0]][choice[1]]
        ci = bootstrap_gain(report[name]["_rows"], choice[0], r["scores"])
        report[name]["primary" if name == "nestful" else "secondary_test"] = {
            "point": choice[0],
            "model": choice[1],
            "gain_vs_mix": r["gain_vs_mix"],
            "ci90": ci,
            "verdict": "CONFIRMED" if ci[0] > 0 else "NOT CONFIRMED",
        }
    for name in SETS:
        if "_rows" in report[name]:
            del report[name]["_rows"]
            for p in POINTS:
                for m in ("logistic", "mlp"):
                    report[name][p][m].pop("scores", None)
    out = Path("results/late_switch")
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=1, default=float)
    (out / "analysis.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
