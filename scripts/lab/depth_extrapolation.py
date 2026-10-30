"""Depth extrapolation: chain v3 on NESTful at 5, 6 and 8 loops (trained with 4).

    python scripts/lab/depth_extrapolation.py

Pre-registered 2026-10-30, before any run above 4 loops. Data: greedy multi-turn NESTful dialogues
(``router_smoke.py --tasks nestful --depths 5 6 8 --modes greedy``, same protocol as depths 1–4:
plain prompt, every pass of every turn at the request's depth, incremental KV cache, executed and
graded) for requests [240, 720) (tag ``nestful240_deep``) and [0, 240) (tag ``nestful_deep``),
joined by id with the existing depth 1–4 rows (tags ``nestful240`` and ``nestful``).

Question: does accuracy keep rising past the trained depth? Primary: accuracy at depth 6 minus
depth 4 on the held-out requests [240, 720) (n = 480), paired bootstrap 90% interval over requests
(2000 resamples, seed 0): RISES if the lower bound > 0, FLAT if the interval covers 0, FALLS if the
upper bound < 0. Secondary: the same for depths 5 and 8 and on all 720 requests; accuracy and cost
(mean loop-tokens, d·(prompt + generated tokens), relative to depth 4) per depth 1–8; turns,
finished share and errors per depth; majority vote of the final answers at depths {4, 5, 6} and
{4, 6, 8} (ties → the deepest of the tied answers' depths) against depth 4 and against depth 6;
the 2-vs-4, 4-vs-6 and 4-vs-8 per-request oracles (headroom for a router between 4 and deeper).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("results/router_smoke")
M = "ouro-chain-depth-v3"
SLICES = {"240_720": ("nestful240", "nestful240_deep"), "0_240": ("nestful", "nestful_deep")}
DEPTHS = (1, 2, 3, 4, 5, 6, 8)


def rows(tag: str) -> list[dict]:
    return [
        json.loads(x) for x in (ROOT / f"{M}_{tag}" / "examples.jsonl").read_text().splitlines()
    ]


def load(base_tag: str, deep_tag: str) -> list[dict]:
    base, deep = rows(base_tag), {r["id"]: r for r in rows(deep_tag)}
    assert [r["id"] for r in base] == list(deep), f"{deep_tag}: ids or order differ from {base_tag}"
    for r in base:
        assert r["query"] == deep[r["id"]]["query"]
        r["runs"] = {**r["runs"], **deep[r["id"]]["runs"]}
        assert all(f"d{d}_greedy" in r["runs"] for d in DEPTHS), r["id"]
    return base


def paired_ci(a: np.ndarray, b: np.ndarray) -> dict:
    rng = np.random.default_rng(0)
    n = len(a)
    vals = [(a[i] - b[i]).mean() for i in (rng.integers(0, n, n) for _ in range(2000))]
    return {
        "difference": float((a - b).mean()),
        "ci90": [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))],
        "wins": int((a > b).sum()),
        "losses": int((a < b).sum()),
    }


def vote(r: dict, depths: tuple[int, ...]) -> bool:
    """Majority final answer over ``depths``; ties go to the deepest depth among the tied answers.
    Unfinished runs (no answer) do not vote; if none finished, the deepest run's outcome. Answers
    are grouped by equivalence: all correct answers form one group (the stored value is a
    truncated repr, so e.g. 5.0 and 4.999999999999982 would otherwise split a correct majority);
    wrong answers group by that repr."""
    runs = [(d, r["runs"][f"d{d}_greedy"]) for d in depths]
    votes = [(d, x) for d, x in runs if x["finished"]]
    if not votes:
        return bool(runs[-1][1]["correct"])
    key = lambda x: "CORRECT" if x["correct"] else x["value"]  # noqa: E731
    count = Counter(key(x) for _, x in votes)
    top = max(count.values())
    winner = max((d, x) for d, x in votes if count[key(x)] == top)[1]
    return bool(winner["correct"])


def report(rs: list[dict]) -> dict:
    y = {d: np.array([r["runs"][f"d{d}_greedy"]["correct"] for r in rs], float) for d in DEPTHS}
    cost = {
        d: np.mean(
            [
                d
                * (r["runs"][f"d{d}_greedy"]["prompt_tokens"] + r["runs"][f"d{d}_greedy"]["tokens"])
                for r in rs
            ]
        )
        for d in DEPTHS
    }
    per_depth = {
        d: {
            "accuracy": float(y[d].mean()),
            "cost_vs_d4": float(cost[d] / cost[4]),
            "turns": float(np.mean([r["runs"][f"d{d}_greedy"]["turns"] for r in rs])),
            "finished": float(np.mean([r["runs"][f"d{d}_greedy"]["finished"] for r in rs])),
            "errors": float(np.mean([r["runs"][f"d{d}_greedy"]["errors"] for r in rs])),
        }
        for d in DEPTHS
    }
    votes = {
        k: np.array([vote(r, ds) for r in rs], float)
        for k, ds in (("4,5,6", (4, 5, 6)), ("4,6,8", (4, 6, 8)))
    }
    return {
        "n": len(rs),
        "per_depth": per_depth,
        **{f"d{d}_minus_d4": paired_ci(y[d], y[4]) for d in (5, 6, 8)},
        "vote": {
            k: {
                "accuracy": float(v.mean()),
                "vs_d4": paired_ci(v, y[4]),
                "vs_d6": paired_ci(v, y[6]),
            }
            for k, v in votes.items()
        },
        "oracle": {
            f"{a}v{b}": float(np.maximum(y[a], y[b]).mean()) for a, b in ((2, 4), (4, 6), (4, 8))
        },
    }


def main() -> None:
    sets = {k: load(*tags) for k, tags in SLICES.items()}
    out = {k: report(v) for k, v in sets.items()}
    out["all_720"] = report(sets["0_240"] + sets["240_720"])
    p = out["240_720"]["d6_minus_d4"]
    out["primary_d6_minus_d4_240_720"] = {
        **p,
        "verdict": "RISES" if p["ci90"][0] > 0 else ("FALLS" if p["ci90"][1] < 0 else "FLAT"),
    }
    Path("results/depth_extrapolation").mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1)
    Path("results/depth_extrapolation/report.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
