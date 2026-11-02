"""Do the Ouro depth findings replicate on Dory 1.0 4B? (analysis)

    PYTHONPATH=scripts/lab python scripts/lab/dory_replication.py

Pre-registered 2026-11-02, before the run. Data: Dory greedy (``--temp 0``), reasoning off,
multi-turn NESTful [0, 480) at 1–4 loops (``dory_nestful_multiturn.py --n 480 --loops 1 2 3 4
--temp 0 --no-thinking --tag greedy_nothink``), each dialogue's turns rendered in Hermes format
(``texts``) so ``depth_anatomy`` replays them unchanged; a turn whose raw answer
(``raw_texts``) attempts a call that does not parse counts as a ``format`` error. (n = 480, not
720: Dory recomputes the whole conversation each turn, ~150 s per request at 4 loop counts.)

Dory's main depth step is 1 → 2 loops (earlier greedy n = 60: .267/.350/.383/.350), Ouro's is
2 → 4; the predictions are stated for each model's main step (shallow s → deep t; for Dory
s = 1, t = 2, and 2 → 4 is also reported):
* P1 formulation: among requests fixed by the main step (wrong at s, right at t), formulation
  errors (``wrong_plan`` + ``wrong_operation``) are the majority error type at s. Ouro: 112/117.
* P2 compounding: the main-step gain is larger on 6–8-call chains than on 2–3-call chains
  (bootstrap 90% interval of the difference above 0). Ouro: +.19 vs +.06–.09.
* P3 stability: of the main-step gains, >= 60% are also right at both deeper loop counts
  available (Dory: 3 and 4). Ouro: 78% at 5 and 6.
Each is reported as SUPPORTED / NOT SUPPORTED. Also: accuracy per loop count, the taxonomy share
per loop count (with a ``no_tool_use`` count: final answers in prose without any call), the
per-step reliability fit, and the gain by gold call count. Early-signal routing is not tested
here (it needs a Dory loop-readout probe).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import depth_anatomy as da
import numpy as np
from ouro_nestful_multiturn import requests

PATH = Path("results/dory_multiturn/nestful/greedy_nothink/examples.jsonl")
LOOPS = (1, 2, 3, 4)
MAIN = (1, 2)


def load():
    examples, functions = requests(720)
    by = {}
    for r in map(json.loads, open(PATH)):
        by.setdefault(r["id"], {})[r["loops"]] = r
    rows = []
    for e in examples:
        runs = by.get(e.id, {})
        if not all(L in runs for L in LOOPS):
            continue
        rows.append(
            {
                "example": e,
                "gold_calls": len(e.calls),
                "runs": {f"d{L}_greedy": runs[L] for L in LOOPS},
            }
        )
    return rows, functions


def main() -> None:
    rows, functions = load()
    n = len(rows)
    y = {L: np.array([r["runs"][f"d{L}_greedy"]["correct"] for r in rows], bool) for L in LOOPS}
    cats = {}
    for i, r in enumerate(rows):
        gold = da.gold_values(r["example"], functions)
        for L in LOOPS:
            run = r["runs"][f"d{L}_greedy"]
            c = da.classify(run, r["example"], functions, gold)[0]
            raw = run.get("raw_texts") or []
            if c != "correct" and any(
                ("<tool_call>" in a or "<function=" in a) and "<tool_call>" not in t
                for a, t in zip(raw, run["texts"], strict=False)
            ):
                c = "format"  # a call was attempted but did not parse
            elif c != "correct" and not run["calls"]:
                c = "no_tool_use"
            cats[(i, L)] = c
    report = {
        "n": n,
        "accuracy": {L: float(y[L].mean()) for L in LOOPS},
        "taxonomy_share": {
            L: dict(Counter(cats[(i, L)] for i in range(n)).most_common()) for L in LOOPS
        },
    }
    for s, t in ((1, 2), (2, 4)):
        fixed = [i for i in range(n) if not y[s][i] and y[t][i]]
        types = Counter(cats[(i, s)] for i in fixed)
        deeper = [L for L in LOOPS if L > t]  # none for 2→4: stability not defined
        stable = [i for i in fixed if all(y[L][i] for L in deeper)] if deeper else []
        report[f"fixed_{s}to{t}"] = {
            "n": len(fixed),
            "error_type_at_shallow": dict(types.most_common()),
            "formulation_share": (types["wrong_plan"] + types["wrong_operation"])
            / max(1, len(fixed)),
            "also_right_at": deeper,
            "stable_share": len(stable) / max(1, len(fixed)) if deeper else None,
        }
    calls = np.array([r["gold_calls"] for r in rows])
    s, t = MAIN
    g = y[t].astype(float) - y[s].astype(float)
    short, long_ = (calls >= 2) & (calls <= 3), (calls >= 6) & (calls <= 8)
    idx_s, idx_l = np.where(short)[0], np.where(long_)[0]  # bootstrap within each bucket

    rng = np.random.default_rng(0)
    vals = [
        g[rng.choice(idx_l, len(idx_l))].mean() - g[rng.choice(idx_s, len(idx_s))].mean()
        for _ in range(2000)
    ]
    p2 = {
        "gain_2_3_calls": float(g[short].mean()),
        "gain_6_8_calls": float(g[long_].mean()),
        "difference_ci90": [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))],
    }
    da.DEPTHS = LOOPS
    report["chain_length"] = da.chain_length(rows)
    fx = report[f"fixed_{s}to{t}"]
    report["predictions"] = {
        "P1_formulation_majority": {
            "share": fx["formulation_share"],
            "verdict": "SUPPORTED" if fx["formulation_share"] > 0.5 else "NOT SUPPORTED",
        },
        "P2_compounding": {
            **p2,
            "verdict": "SUPPORTED" if p2["difference_ci90"][0] > 0 else "NOT SUPPORTED",
        },
        "P3_stability": {
            "share": fx["stable_share"],
            "verdict": "SUPPORTED" if fx["stable_share"] >= 0.6 else "NOT SUPPORTED",
        },
    }
    out = Path("results/dory_replication")
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=1, default=float)
    (out / "report.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
