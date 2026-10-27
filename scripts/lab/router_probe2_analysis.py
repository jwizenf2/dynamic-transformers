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
run) at d loops: d·T_d loop-tokens (no shared prefix: every task has its own tool list, so its
~900-token system prompt is unique and cannot be prefix-cached across requests).
* ``escalate`` (primary, deployable): start every request at 2 loops; after the first k answer
  tokens, a router that sees only loop-1/2 signals of those tokens (computed by the 2-loop run
  itself) either continues at 2 loops (cost 2·T_2) or switches to 4 loops: the prompt's loops
  1–2 are reused (the server keeps the prompt's loop-2 output states), its loops 3–4 are
  computed, the k draft tokens are discarded and the answer is generated at 4 loops:
  2·P + 2·k + 2·P + 4·(T_4 − P) = 4·T_4 + 2·k (the review's "4·T_4 − 2·P + 2·k" counted the
  reuse twice; corrected before running). Features: per loop 1–2 mean/max entropy, min
  margin, mean gate; loop-2 state change; tokens where loop 1's argmax differs from loop 2's;
  question words and numbers (12). Options {2, 4}.
* ``full_probe`` (reported): the 24-feature router on all 4 loops pays the probe: prompt at the
  loops above d ((4 − d)·P) and k tokens at 4 loops: d·T_d + (4 − d)·P + 4·k.
* ``cached`` (reported, conditional): as ``full_probe`` but with the system prompt removed from
  every cost, as if a deployment used one fixed, prefix-cached tool library. Not the protocol
  the labels were collected under.

Gate (stricter than the router smoke, as a second look at the same question). PASS only if both
(1) predictability: the 24-feature router's gain over the best random mix at 2.5 mean loops
without probe cost (comparable to the router smoke's +0.034) is >= 0.03 with a bootstrap 90%
lower bound > 0, and (2) deployability: the ``escalate`` router is not below the best random mix
of fixed depths (upper concave hull of (mean loop-tokens, accuracy)) at the budget halfway
between fixed-2's and fixed-4's mean cost (gain >= 0; bootstrap interval reported).
Every metric is also reported on tasks 60..291 alone (not seen during the gate design).

Amendment log (2026-10-26, all before the lab run). v1 charged only the ``full_probe`` cost and
gated on a >= 0.03 cost-inclusive gain. A local smoke on the first 60 of the 292 tasks showed
the 4-loop prompt prefill dominating every request's cost (gain −0.25 for every feature set,
including question-only), so v2 gated on ``cached``. An independent review then showed
``cached`` assumes a shared tool library this protocol does not have (292 distinct tool lists),
so v3 (this) gates on the deployable ``escalate`` design. The smoke also showed, on 60 tasks, v2
+0.05 and v1 −0.02 at 2.5 loops without probe cost; those 60 tasks are inside the 292, hence the
separate report on tasks 60..291.
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


def features_escalate(probe: dict, query: str) -> np.ndarray:
    """Loop-1/2 signals only (what a 2-loop run computes itself): 12 features."""
    p = np.array(
        [[[np.nan if x is None else x for x in loop] for loop in t] for t in probe["probe"]],
        dtype=float,
    )[:, :2]  # [t, 2, 5] (loops 1, 2)
    gate, ent, margin, change = p[..., 0], p[..., 1], p[..., 2], p[..., 4]
    f = []
    for loop in range(2):
        f += [ent[:, loop].mean(), ent[:, loop].max(), margin[:, loop].min(), gate[:, loop].mean()]
    f += [float(np.nanmean(change[:, 1]))]
    f += [float(probe["loop12_disagree"])]
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


def policy_curve(
    prob: dict, y: dict, cost: dict, idx: np.ndarray, depths: tuple = DEPTHS
) -> list[tuple[float, float]]:
    """(mean cost, accuracy) of argmin_d λ·d/4 − P_d over ``depths`` and a λ sweep."""
    pts = []
    for lam in np.linspace(0, 2, 81):
        score = np.stack([lam * d / 4 - prob[d][idx] for d in depths])
        choice = np.array(depths)[score.argmin(axis=0)]
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
    ids = list(smoke)
    assert set(probes) == set(smoke), "probe rows and router-smoke rows cover different tasks"
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
    for i in ids:  # loop-1 vs loop-2 argmax disagreements along the probe
        a1 = [t[0][3] for t in probes[i]["probe"]]
        a2 = [t[1][3] for t in probes[i]["probe"]]
        probes[i]["loop12_disagree"] = sum(x != z for x, z in zip(a1, a2, strict=True))
    fixed = {d: d * tokens[d] for d in DEPTHS}
    costs = {
        "full_probe": {d: fixed[d] + (4 - d) * plen + 4 * k for d in DEPTHS},
        "cached": {
            d: d * (tokens[d] - sys_len) + (4 - d) * (plen - sys_len) + 4 * k for d in DEPTHS
        },
        "escalate": {1: fixed[1], 2: fixed[2], 4: 4 * tokens[4] + 2 * k},
    }
    fixed_for = {
        "full_probe": fixed,
        "escalate": fixed,
        "cached": {d: d * (tokens[d] - sys_len) for d in DEPTHS},
    }

    x2 = np.stack([features_v2(probes[i], smoke[i]["query"]) for i in ids])
    xe = np.stack([features_escalate(probes[i], smoke[i]["query"]) for i in ids])
    x1 = np.stack([features(r) for r in rows])
    xq = np.stack([features(r, question_only=True) for r in rows])
    probs = {
        name: oof_probs(x, y)
        for name, x in (("v2", x2), ("escalate", xe), ("v1", x1), ("question", xq))
    }

    def gain_cost(prob, idx, accounting, depths=DEPTHS):
        fc = fixed_for[accounting]
        fp = [(float(fc[d][idx].mean()), float(y[d][idx].mean())) for d in DEPTHS]
        budget = (fp[1][0] + fp[2][0]) / 2  # halfway between fixed-2 and fixed-4 mean cost
        curve_pts = policy_curve(prob, y, costs[accounting], idx, depths)
        return at_loops(curve_pts, budget) - hull_interp(fp, budget)

    def gain_loops(prob, idx):
        loops = {d: np.full(n, float(d)) for d in DEPTHS}
        fp = [(float(d), float(y[d][idx].mean())) for d in DEPTHS]
        return at_loops(policy_curve(prob, y, loops, idx), 2.5) - hull_interp(fp, 2.5)

    def gains(idx):
        return {
            "at_2.5_loops_no_probe_cost": {n_: gain_loops(p, idx) for n_, p in probs.items()},
            "escalate_deployable": gain_cost(probs["escalate"], idx, "escalate", (2, 4)),
            "full_probe_uncached": {
                n_: gain_cost(p, idx, "full_probe") for n_, p in probs.items() if n_ != "escalate"
            },
            "cached_conditional": {
                n_: gain_cost(p, idx, "cached") for n_, p in probs.items() if n_ != "escalate"
            },
        }

    def ci(f):
        rng = np.random.default_rng(0)
        b = [f(rng.integers(0, n, n)) for _ in range(500)]
        return [float(np.percentile(b, 5)), float(np.percentile(b, 95))]

    all_idx = np.arange(n)
    g_loops = gain_loops(probs["v2"], all_idx)
    g_esc = gain_cost(probs["escalate"], all_idx, "escalate", (2, 4))
    ci_loops = ci(lambda idx: gain_loops(probs["v2"], idx))
    ci_esc = ci(lambda idx: gain_cost(probs["escalate"], idx, "escalate", (2, 4)))
    passed = g_loops >= 0.03 and ci_loops[0] > 0 and g_esc >= 0
    mismatch = [
        probes[i]["text"][:60] != smoke[i]["runs"]["d4_greedy"]["texts"][0][:60] for i in ids
    ]
    report = {
        "n": n,
        "fixed_accuracy": {d: float(y[d].mean()) for d in DEPTHS},
        "fixed_mean_loop_tokens": {d: float(fixed[d].mean()) for d in DEPTHS},
        "probe_mean_tokens": float(k.mean()),
        "probe_vs_d4_first_turn_mismatch": float(np.mean(mismatch)),
        "gains_all_292": gains(all_idx),
        "gains_tasks_60_291_unseen": gains(np.arange(60, n)),
        "gate": {
            "1_predictability_gain_v2_at_2.5_loops": g_loops,
            "1_ci90": ci_loops,
            "2_deployable_escalate_gain": g_esc,
            "2_ci90": ci_esc,
            "verdict": "PASS" if passed else "FAIL",
        },
    }
    text = json.dumps(report, indent=1, default=float)
    (Path(args.probe) / "analysis.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
