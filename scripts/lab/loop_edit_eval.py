"""Pre-registered test of loop-drift edits at 8 loops on NESTful [0, 240) (analysis).

    python scripts/lab/loop_edit_eval.py

See docs/research/loop_edit_2026-11-03.md. Joins by id the edited 8-loop dialogues (tags
``nestful_edit_project_pc16``, ``nestful_edit_damp_0.5``) with the unedited 4- and 8-loop rows
(tags ``nestful`` and ``nestful_deep``). Primary: acc_8(project_pc16) − acc_8(unedited), paired
bootstrap 90% interval (2000 resamples, seed 0), REPAIRED if the lower bound > 0. Secondary:
each edit minus 4 loops; tool errors per dialogue and unfinished share per condition.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SMOKE = Path("results/router_smoke")
M = "ouro-chain-depth-v3"
EDITS = ("project_pc16", "damp_0.5")


def rows(tag: str) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, open(SMOKE / f"{M}_{tag}" / "examples.jsonl"))}


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    rng = np.random.default_rng(0)
    n = len(a)
    vals = [(a[i] - b[i]).mean() for i in (rng.integers(0, n, n) for _ in range(2000))]
    return {
        "difference": float((a - b).mean()),
        "ci90": [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))],
        "wins": int((a > b).sum()),
        "losses": int((a < b).sum()),
    }


def main() -> None:
    base, deep = rows("nestful"), rows("nestful_deep")
    ids = list(base)
    runs = {
        "d4": [base[i]["runs"]["d4_greedy"] for i in ids],
        "d8": [deep[i]["runs"]["d8_greedy"] for i in ids],
    }
    for e in EDITS:
        r = rows(f"nestful_edit_{e}")
        assert all(i in r for i in ids), f"{e}: missing requests"
        runs[f"d8_{e}"] = [r[i]["runs"]["d8_greedy"] for i in ids]
    y = {k: np.array([x["correct"] for x in v], float) for k, v in runs.items()}
    report = {
        "n": len(ids),
        "conditions": {
            k: {
                "accuracy": float(y[k].mean()),
                "errors_per_dialogue": float(np.mean([x["errors"] for x in v])),
                "unfinished": float(np.mean([not x["finished"] for x in v])),
            }
            for k, v in runs.items()
        },
    }
    for e in EDITS:
        report[f"{e}_minus_d8"] = paired(y[f"d8_{e}"], y["d8"])
        report[f"{e}_minus_d4"] = paired(y[f"d8_{e}"], y["d4"])
    p = report["project_pc16_minus_d8"]
    report["primary"] = {**p, "verdict": "REPAIRED" if p["ci90"][0] > 0 else "NOT REPAIRED"}
    out = Path("results/loop_edit")
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=1)
    (out / "eval.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
