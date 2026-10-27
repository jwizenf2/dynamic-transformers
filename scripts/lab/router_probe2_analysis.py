"""Router probe v2, analysis: decision-token probe router vs random mixes, probe cost included.

    python scripts/lab/router_probe2_analysis.py \
        --smoke results/router_smoke/ouro-chain-depth-v3 \
        --probe results/router_probe2/ouro-chain-depth-v3

Pre-registered 2026-10-26, before the probe v2 data. Labels: the router smoke's greedy runs at
depths 1, 2, 4 on the 292 clean MathQA-validation tasks (``router_smoke.py``). Features from
``router_probe2.py`` (up to 48 loop-4 greedy tokens of the first answer, per token and loop:
gate, entropy, top-2 margin, agreement with loop 4, relative state change):

* per loop p = 1..4: mean entropy, max entropy, min margin, mean gate over the probe tokens;
* per loop p = 2..4: mean relative state change;
* per loop p = 1..3: number of tokens whose argmax differs from loop 4's;
* question words and numbers (2). 24 features.

Router: as ``router_smoke_analysis`` (one L2 logistic regression per depth, 5-fold x 5 CV,
policy argmin_d λ·d/4 − P_d over a λ sweep). Compared on the same folds with the router smoke's
8-token features (v1) and question-only features.

Cost. A request at depth d processes T_d tokens (all prompt chunks + generated, recorded per
run) at d loops: d·T_d loop-tokens. The router additionally pays the probe: its prompt at the
loops above d ((4 − d)·P; loops 1..d of the prompt cache are the same computation and reusable)
and its k tokens at 4 loops (4·k). Two accountings:
* ``cached`` (primary): the system prompt (tool library, S tokens) is a shared prefix that a
  server computes once at 4 loops and caches, so it is removed from every request's cost:
  fixed d·(T_d − S), router d·(T_d − S) + (4 − d)·(P − S) + 4·k.
* ``uncached`` (conservative, reported): nothing is shared; the probe's 4-loop prompt prefill
  (~900 tokens, mostly tool specs) then costs nearly as much as running at depth 4.

Gate (stricter than the router smoke, as a second look at the same question). PASS only if both
(1) the router's gain over the best random mix at 2.5 mean loops without probe cost (comparable
to the router smoke's +0.034) is >= 0.03 with a bootstrap 90% lower bound > 0, and (2) under the
``cached`` accounting the router is not below the best random mix of fixed depths (upper concave
hull of (mean loop-tokens, accuracy)) at the budget halfway between fixed-2's and fixed-4's mean
cost (gain >= 0).

Amendment log: the first version (same day) charged only the ``uncached`` cost and gated on a
>= 0.03 cost-inclusive gain. A local smoke test on the first 60 of the 292 tasks showed the
uncached probe prefill dominating every request's cost (gain −0.25 for every feature set,
including question-only), i.e. a property of the cost model, not of the router. The gate was
changed to the above after seeing that smoke test (which also showed, on 60 tasks, v2 +0.05 and
v1 −0.02 at 2.5 loops without probe cost); this is disclosed here and in progress.md.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from router_smoke_analysis import DEPTHS, at_loops, features, oof_probs


def features_v2(probe: dict, query: str) -> np.ndarray:
    p = np.array(
        [[[np.nan if x is None else x for x in loop] for loop in t] for t in probe["probe"]],
        dtype=float,
    )  # [t, 4, 5]
    gate, ent, margin, agree, change = (p[..., j] for j in range(5))
    f = []
    for loop in range(4):
        f += [ent[:, loop].mean(), ent[:, loop].max(), margin[:, loop].min(), gate[:, loop].mean()]
    f += [np.nanmean(change[:, loop]) for loop in (1, 2, 3)]
    f += [float((agree[:, loop] == 0).sum()) for loop in (0, 1, 2)]
    f += [len(query.split()), len(re.findall(r"\d+(?:\.\d+)?", query))]
    return np.array(f, dtype=float)


def hull_interp(points: list[tuple[float, float]], x: float) -> float:
    """Upper concave hull of (cost, accuracy) points, non-decreasing, evaluated at ``x``."""
    pts = sorted(points)
    hull: list[tuple[float, float]] = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (y2 - y1) * (p[0] - x1) <= (p[1] - y1) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append(p)
    xs, ys = zip(*hull, strict=True)
    return float(np.interp(x, xs, np.maximum.accumulate(ys)))


def policy_curve(prob: dict, y: dict, cost: dict, idx: np.ndarray) -> list[tuple[float, float]]:
    """(mean cost, accuracy) of argmin_d λ·d/4 − P_d over a λ sweep, on requests ``idx``."""
    pts = []
    for lam in np.linspace(0, 2, 81):
        score = np.stack([lam * d / 4 - prob[d][idx] for d in DEPTHS])
        choice = np.array(DEPTHS)[score.argmin(axis=0)]
        acc = np.mean([y[c][i] for c, i in zip(choice, idx, strict=True)])
        c = np.mean([cost[c][i] for c, i in zip(choice, idx, strict=True)])
        pts.append((float(c), float(acc)))
    return sorted(set(pts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", default="results/router_smoke/ouro-chain-depth-v3")
    parser.add_argument("--probe", default="results/router_probe2/ouro-chain-depth-v3")
    args = parser.parse_args()
    smoke = {
        r["id"]: r
        for r in map(json.loads, (Path(args.smoke) / "examples.jsonl").read_text().splitlines())
    }
    probes = {
        r["id"]: r
        for r in map(json.loads, (Path(args.probe) / "examples.jsonl").read_text().splitlines())
    }
    ids = [i for i in smoke if i in probes]
    n = len(ids)
    rows = [smoke[i] for i in ids]
    y = {d: np.array([r["runs"][f"d{d}_greedy"]["correct"] for r in rows], float) for d in DEPTHS}
    tokens = {
        d: np.array(
            [
                r["runs"][f"d{d}_greedy"]["prompt_tokens"] + r["runs"][f"d{d}_greedy"]["tokens"]
                for r in rows
            ],
            float,
        )
        for d in DEPTHS
    }
    plen = np.array([probes[i]["prompt_tokens"] for i in ids], float)
    k = np.array([len(probes[i]["probe"]) for i in ids], float)
    sys_len = np.array([probes[i]["system_tokens"] for i in ids], float)
    costs = {}
    for name, shared in (("cached", sys_len), ("uncached", np.zeros(n))):
        fixed = {d: d * (tokens[d] - shared) for d in DEPTHS}
        router = {d: d * (tokens[d] - shared) + (4 - d) * (plen - shared) + 4 * k for d in DEPTHS}
        costs[name] = (fixed, router)

    x2 = np.stack([features_v2(probes[i], smoke[i]["query"]) for i in ids])
    x1 = np.stack([features(r) for r in rows])
    xq = np.stack([features(r, question_only=True) for r in rows])
    probs = {name: oof_probs(x, y) for name, x in (("v2", x2), ("v1", x1), ("question", xq))}
    all_idx = np.arange(n)

    def gain_cost(prob, idx, accounting="cached"):
        fixed_cost, router_cost = costs[accounting]
        fp = [(float(fixed_cost[d][idx].mean()), float(y[d][idx].mean())) for d in DEPTHS]
        budget = (fp[1][0] + fp[2][0]) / 2  # halfway between fixed-2 and fixed-4 mean cost
        return at_loops(policy_curve(prob, y, router_cost, idx), budget) - hull_interp(fp, budget)

    def gain_loops(prob, idx):
        loops = {d: np.full(n, float(d)) for d in DEPTHS}
        fp = [(float(d), float(y[d][idx].mean())) for d in DEPTHS]
        return at_loops(policy_curve(prob, y, loops, idx), 2.5) - hull_interp(fp, 2.5)

    rng = np.random.default_rng(0)
    boots = [gain_loops(probs["v2"], rng.integers(0, n, n)) for _ in range(500)]
    g_loops = gain_loops(probs["v2"], all_idx)
    g_cached = gain_cost(probs["v2"], all_idx, "cached")
    passed = g_loops >= 0.03 and np.percentile(boots, 5) > 0 and g_cached >= 0
    fixed_cached = costs["cached"][0]
    report = {
        "n": n,
        "fixed_accuracy": {d: float(y[d].mean()) for d in DEPTHS},
        "fixed_mean_loop_tokens_cached": {d: float(fixed_cached[d].mean()) for d in DEPTHS},
        "probe_mean_tokens": float(k.mean()),
        "probe_tokens_cost_vs_fixed4_cached": float((4 * k).mean() / fixed_cached[4].mean()),
        "gain_at_2.5_loops_no_probe_cost": {
            name: gain_loops(p, all_idx) for name, p in probs.items()
        },
        "gain_cost_cached": {name: gain_cost(p, all_idx, "cached") for name, p in probs.items()},
        "gain_cost_uncached": {
            name: gain_cost(p, all_idx, "uncached") for name, p in probs.items()
        },
        "gate": {
            "gain_v2_at_2.5_loops": g_loops,
            "ci90": [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))],
            "gain_v2_cost_cached": g_cached,
            "verdict": "PASS" if passed else "FAIL",
        },
    }
    text = json.dumps(report, indent=1, default=float)
    (Path(args.probe) / "analysis.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
