"""Formulation ablation: does giving the plan remove the benefit of extra loops? (analysis)

    PYTHONPATH=scripts/lab python scripts/lab/plan_ablation.py

Pre-registered 2026-11-02, before any plan-condition run. Data: chain v3 greedy multi-turn
NESTful dialogues at 2 and 4 loops on requests [0, 720) in three conditions, joined by id:
``none`` (existing rows, tags ``nestful``/``nestful240``), ``full`` and ``ops`` (queries from
``make_nestful_plans.py``; ``router_smoke.py --tasks results/nestful_plans/<c>_tasks.jsonl
--n 720 --depths 2 4 --modes greedy --tag plan_<c>``).

Hypothesis (from ``depth_anatomy.py``: 4 loops fix mostly formulation errors at 2 loops): with
the full plan given, the 2→4 accuracy gain shrinks. Primary: (gain_none − gain_full), gain =
acc_4 − acc_2 on the same requests, paired bootstrap 90% interval over requests (2000
resamples, seed 0): SUPPORTED if the lower bound > 0, NOT SUPPORTED if the interval covers 0,
REVERSED if the upper bound < 0. Secondary: the same for ``ops``; accuracy per condition and
depth; the gain by gold call count (2, 3, 4–5, 6–8, 9+) per condition, where the account
predicts the full-plan gain no longer grows with chain length; the error taxonomy of
``depth_anatomy.classify`` per condition and depth (the account predicts wrong-plan errors fall
most with the plan, and that the remaining 2→4 gain, if any, is in execution errors: copying,
order, stopping).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from depth_anatomy import CATEGORIES, classify, gold_values
from ouro_nestful_multiturn import load_tasks, requests

M = "ouro-chain-depth-v3"
SMOKE = Path("results/router_smoke")
BUCKETS = ((2, 2), (3, 3), (4, 5), (6, 8), (9, 99))


def rows_for(tags: list[str]) -> dict[str, dict]:
    out = {}
    for t in tags:
        for r in map(json.loads, open(SMOKE / f"{M}_{t}" / "examples.jsonl")):
            out[r["id"]] = r
    return out


def boot(f, n) -> list[float]:
    rng = np.random.default_rng(0)
    vals = [f(rng.integers(0, n, n)) for _ in range(2000)]
    return [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]


def main() -> None:
    examples, functions = requests(720)
    ids = [e.id for e in examples]
    conds = {
        "none": rows_for(["nestful", "nestful240"]),
        "full": rows_for(["plan_full"]),
        "ops": rows_for(["plan_ops"]),
    }
    plan_examples = {
        c: {e.id: e for e in load_tasks(Path(f"results/nestful_plans/{c}_tasks.jsonl"))}
        for c in ("full", "ops")
    }
    for c, rows in conds.items():
        assert all(i in rows for i in ids), f"{c}: missing requests"
    y = {
        c: {
            d: np.array([conds[c][i]["runs"][f"d{d}_greedy"]["correct"] for i in ids], float)
            for d in (2, 4)
        }
        for c in conds
    }
    gain = {c: y[c][4] - y[c][2] for c in conds}
    n = len(ids)
    calls = np.array([len(e.calls) for e in examples])
    report = {"n": n, "accuracy": {c: {d: float(y[c][d].mean()) for d in (2, 4)} for c in conds}}
    for c in ("full", "ops"):
        diff = gain["none"] - gain[c]
        ci = boot(lambda i, x=diff: x[i].mean(), n)
        report[f"gain_none_minus_gain_{c}"] = {
            "gain_none": float(gain["none"].mean()),
            f"gain_{c}": float(gain[c].mean()),
            "difference": float(diff.mean()),
            "ci90": ci,
            "verdict": "SUPPORTED" if ci[0] > 0 else ("REVERSED" if ci[1] < 0 else "NOT SUPPORTED"),
        }
    report["primary"] = report["gain_none_minus_gain_full"]
    report["gain_by_gold_calls"] = {
        c: {
            f"{lo}-{hi}": {
                "n": int(((calls >= lo) & (calls <= hi)).sum()),
                "gain": float(gain[c][(calls >= lo) & (calls <= hi)].mean()),
                "acc2": float(y[c][2][(calls >= lo) & (calls <= hi)].mean()),
                "acc4": float(y[c][4][(calls >= lo) & (calls <= hi)].mean()),
            }
            for lo, hi in BUCKETS
        }
        for c in conds
    }
    tax = {}
    for c in conds:
        tax[c] = {}
        for d in (2, 4):
            cnt = Counter()
            for e in examples:
                ex = plan_examples[c][e.id] if c != "none" else e
                cnt[
                    classify(
                        conds[c][e.id]["runs"][f"d{d}_greedy"],
                        ex,
                        functions,
                        gold_values(e, functions),
                    )[0]
                ] += 1
            tax[c][d] = {k: cnt[k] / n for k in ("correct", *CATEGORIES)}
    report["taxonomy_share"] = tax
    out = Path("results/plan_ablation")
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=1, default=float)
    (out / "report.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
