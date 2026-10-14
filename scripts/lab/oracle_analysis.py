"""Gate 1 headroom analysis from headroom.py outputs (CPU only; no model, no downloads).

    python scripts/lab/oracle_analysis.py [RESULT_DIR ...]

Default inputs are every ``~/AI/jobs/dynamic-transformers-headroom*/results/headroom/<model>`` on
the lab. For each model, using only the SINGLE-SKIP architectures (full + whole group /
attention-only / DeltaNet-only per group; greedy multi-group rows are excluded so every model
has the same menu):

* Static frontier: the best fixed architecture at each cost level.
* Per-request oracle: each example uses the cheapest architecture that answers it correctly.
  - ``oracle_keep``: examples nothing solves stay on the full model (cost 1).
  - Reported as accuracy and mean relative cost versus the full model.
  This is an UPPER BOUND on routing (it picks with hindsight from a 0/1 outcome over many
  architectures, so it includes "lucky" passes); the plan's held-out probe is the honest test.
* Lucky-pass check: examples the full model gets wrong but some skip gets right.
* Overlap: for the best skips, the 2x2 table (full correct?, skip correct?). A skip whose
  correct set is a subset of the full model's means routing can only save compute, never
  add accuracy; a mixed table means outcomes are partly noise.
* Savings-at-full-accuracy for a router restricted to the K best static skips (K = 1, 3), which
  shows how much of the oracle comes from a small menu a router could plausibly learn.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path


def load(result_dir: Path):
    summary = json.loads((result_dir / "summary.json").read_text())
    costs = {r["arch"]: r["relative_cost"] for r in summary["runs"]}
    outcome: dict[str, dict[str, bool]] = defaultdict(dict)
    for line in (result_dir / "examples.jsonl").read_text().splitlines():
        row = json.loads(line)
        outcome[row["arch"]][row["id"]] = row["correct"]
    singles = [a for a in outcome if a == "full" or not a.startswith("greedy")]
    return summary["meta"]["model"], costs, {a: outcome[a] for a in singles}


def oracle(ids, archs, costs, outcome):
    """Cheapest correct architecture per example; unsolved examples stay on full."""
    total_cost = solved = 0
    for i in ids:
        options = [costs[a] for a in archs if outcome[a][i]]
        solved += bool(options)
        total_cost += min(options) if options else 1.0
    return solved / len(ids), total_cost / len(ids)


def analyse(result_dir: Path) -> dict:
    model, costs, outcome = load(result_dir)
    ids = sorted(outcome["full"])
    full = outcome["full"]
    acc = {a: sum(o.values()) / len(ids) for a, o in outcome.items()}
    skips = sorted((a for a in outcome if a != "full"), key=lambda a: -acc[a])

    def overlap(a):
        table = {"both": 0, "full_only": 0, "skip_only": 0, "neither": 0}
        for i in ids:
            key = {(1, 1): "both", (1, 0): "full_only", (0, 1): "skip_only", (0, 0): "neither"}
            table[key[(int(full[i]), int(outcome[a][i]))]] += 1
        return table

    oracle_all = oracle(ids, list(outcome), costs, outcome)
    menus = {f"top{k}": ["full", *skips[:k]] for k in (1, 3)}
    menu_oracles = {name: oracle(ids, menu, costs, outcome) for name, menu in menus.items()}
    lucky = sum(1 for i in ids if not full[i] and any(outcome[a][i] for a in skips))
    # Savings while matching full accuracy exactly: only full-correct examples may move.
    keep_full_acc = sum(
        min([costs[a] for a in outcome if outcome[a][i]], default=1.0) if full[i] else 1.0
        for i in ids
    ) / len(ids)
    return {
        "model": model,
        "n": len(ids),
        "full_accuracy": acc["full"],
        "best_static": [(a, round(acc[a], 3), round(costs[a], 3)) for a in skips[:5]],
        "oracle_any_arch": {"accuracy": oracle_all[0], "mean_cost": oracle_all[1]},
        "oracle_menus": {k: {"accuracy": v[0], "mean_cost": v[1]} for k, v in menu_oracles.items()},
        "oracle_cost_at_full_accuracy": keep_full_acc,
        "lucky_examples": lucky,
        "overlap_best3": {a: overlap(a) for a in skips[:3]},
    }


def main() -> None:
    dirs = [Path(p) for p in sys.argv[1:]] or [
        Path(p)
        for p in sorted(
            glob.glob(
                str(Path.home() / "AI/jobs/dynamic-transformers-headroom*/results/headroom/*")
            )
        )
        if (Path(p) / "summary.json").exists()
    ]
    results = [analyse(d) for d in dirs]
    for r in results:
        print(json.dumps(r, indent=1), flush=True)
    out = Path("results/oracle_analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out} for {[r['model'] for r in results]}")


if __name__ == "__main__":
    main()
