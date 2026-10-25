"""v4 pilot decision: paired gains of the two continuation arms over chain v3 on clean MathQA-val.

    python scripts/lab/pilot_v4_analysis.py

Pre-registered 2026-10-25, before any data. Per request, the score is the mean correctness over
depths 2 and 4 (greedy, ``ouro_nestful_multiturn.py --variant plain`` protocol). v3's scores
come from the router smoke's greedy rows (``results/router_smoke/chain-depth-v3``), the arms'
from ``results/ouro_nestful_mt/pilot-{aug,epoch}/plain_validation_clean_tasks``; requests are
paired by id. Decision: run the full v4 only if the augmented arm's mean paired gain over v3
has a bootstrap (2,000 resamples of requests) 90% lower bound > 0 AND its point estimate is
above the second-pass arm's. Paired SE at n = 292 is about 0.027, so a true gain of ~5 points
is detected with reasonable power; smaller gains are not worth a 15+ hour run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEPTHS = (2, 4)


def scores_mt(path: Path) -> dict[str, float]:
    by: dict[str, dict[int, bool]] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        by.setdefault(r["id"], {})[r["steps"]] = r["correct"]
    return {
        i: float(np.mean([v[d] for d in DEPTHS])) for i, v in by.items() if set(DEPTHS) <= set(v)
    }


def scores_smoke(path: Path) -> dict[str, float]:
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out[r["id"]] = float(np.mean([r["runs"][f"d{d}_greedy"]["correct"] for d in DEPTHS]))
    return out


def paired(a: dict, b: dict, rng: np.random.Generator) -> dict:
    ids = sorted(set(a) & set(b))
    diff = np.array([a[i] - b[i] for i in ids])
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    return {
        "n": len(ids),
        "gain": float(diff.mean()),
        "ci90": [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", default="results/router_smoke/chain-depth-v3/examples.jsonl")
    parser.add_argument("--mt", default="results/ouro_nestful_mt")
    args = parser.parse_args()
    v3 = scores_smoke(Path(args.smoke))
    arms = {
        a: scores_mt(Path(args.mt) / f"pilot-{a}/plain_validation_clean_tasks/examples.jsonl")
        for a in ("aug", "epoch")
    }
    rng = np.random.default_rng(0)
    report = {
        "mean_d2_d4": {
            "v3": float(np.mean(list(v3.values()))),
            **{a: float(np.mean(list(s.values()))) for a, s in arms.items()},
        },
        "aug_vs_v3": paired(arms["aug"], v3, rng),
        "epoch_vs_v3": paired(arms["epoch"], v3, rng),
        "aug_vs_epoch": paired(arms["aug"], arms["epoch"], rng),
    }
    ok = report["aug_vs_v3"]["ci90"][0] > 0 and report["aug_vs_epoch"]["gain"] > 0
    report["verdict"] = "RUN FULL V4" if ok else "SKIP FULL V4"
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
