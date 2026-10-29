"""Phase A label analysis (pre-registered as A2b in docs/research/plan_2026-10-28_router_paper.md).

    python scripts/lab/phase_a_labels.py

Joins, per task, the greedy runs and the 4 sampled runs (T 0.7) at depths 2 and 4 for the 531
router-training tasks (292 clean MathQA-val + 239 MathQA-test pool):

* clean: ``router_smoke/ouro-chain-depth-v3`` (greedy, sample s0) + ``..._clean_samples`` (s1–s3);
* pool: ``router_smoke/ouro-chain-depth-v3_testpool`` (greedy) + ``..._testpool_samples``
  (s0–s3).

Reliability ceiling of "4 loops helps": with disjoint sample halves A = {s0, s1}, B = {s2, s3},
Δ_X = P_4 − P_2 estimated on half X. Reported: corr(Δ_A, Δ_B) and its Spearman-Brown projection to
4 samples; AUC of Δ_A for (Δ_B > 0); AUC of the 4-sample Δ for the greedy label "4 helps"
(greedy wrong at 2, right at 4), the best any feature-based router can hope to rank greedy
outcomes if it learned per-task success rates perfectly from 4 samples. Writes
``results/phase_a/labels.jsonl`` (one row per task) and ``results/phase_a/ceiling.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("results/router_smoke")
M = "ouro-chain-depth-v3"


def rows(folder: str) -> dict[str, dict]:
    return {
        r["id"]: r
        for r in map(json.loads, (ROOT / folder / "examples.jsonl").read_text().splitlines())
    }


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos, neg = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main() -> None:
    sets = {
        "clean": (rows(M), rows(f"{M}_clean_samples")),
        "pool": (rows(f"{M}_testpool"), rows(f"{M}_testpool_samples")),
    }
    table = []
    for name, (base, extra) in sets.items():
        assert set(base) == set(extra), f"{name}: task sets differ"
        for i, b in base.items():
            runs = {**b["runs"], **extra[i]["runs"]}
            row = {"id": i, "set": name, "query": b["query"]}
            for d in (2, 4):
                row[f"g{d}"] = int(runs[f"d{d}_greedy"]["correct"])
                row[f"s{d}"] = [
                    int(runs[f"d{d}_sample" if s == 0 else f"d{d}_sample{s}"]["correct"])
                    for s in range(4)
                ]
            table.append(row)
    out = Path("results/phase_a")
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in table))

    s2 = np.array([r["s2"] for r in table], float)  # [n, 4]
    s4 = np.array([r["s4"] for r in table], float)
    g2 = np.array([r["g2"] for r in table], float)
    g4 = np.array([r["g4"] for r in table], float)
    da = s4[:, :2].mean(1) - s2[:, :2].mean(1)
    db = s4[:, 2:].mean(1) - s2[:, 2:].mean(1)
    d_all = s4.mean(1) - s2.mean(1)
    helps = ((g2 == 0) & (g4 == 1)).astype(float)
    r_half = float(np.corrcoef(da, db)[0, 1])
    report = {
        "n": len(table),
        "greedy_accuracy": {"d2": float(g2.mean()), "d4": float(g4.mean())},
        "sampled_accuracy": {"d2": float(s2.mean()), "d4": float(s4.mean())},
        "greedy_patterns_d2_d4": {
            f"{a}{b}": int(((g2 == a) & (g4 == b)).sum()) for a in (1, 0) for b in (1, 0)
        },
        "corr_delta_halves": r_half,
        "spearman_brown_4_samples": 2 * r_half / (1 + r_half),
        "auc_delta_A_for_delta_B_positive": auc(da, (db > 0).astype(float)),
        "auc_delta_4samples_for_greedy_4_helps": auc(d_all, helps),
        "auc_P4_4samples_for_greedy_d4_correct": auc(s4.mean(1), g4),
        "auc_P2_4samples_for_greedy_d2_correct": auc(s2.mean(1), g2),
        "share_tasks_delta_positive_4samples": float((d_all > 0).mean()),
        "share_tasks_delta_negative_4samples": float((d_all < 0).mean()),
    }
    text = json.dumps(report, indent=1)
    (out / "ceiling.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
