"""Pre-registered replication metrics for one model and one NESTful slice (analysis).

    # unseen confirmation, chain v3 (router trained frozen on [0, 720))
    PYTHONPATH=scripts/lab python scripts/lab/replication_metrics.py --model ouro-chain-depth-v3 \
        --nestful-tag nestful720 --plan-tag plan_full720 --start 720 --n 480 \
        --rescue-train nestful nestful240
    # training seed / other model on [0, 720) (rescue AUC by cross-validation); --deep-tag adds
    # depth-6 rows run separately (P4); --tasks evaluates another task file (GSM8K)
    PYTHONPATH=scripts/lab python scripts/lab/replication_metrics.py --model chain-depth-v3-s1 \
        --nestful-tag nestful --plan-tag plan_full --start 0 --n 720 \
        --bfcl-outcomes results/ouro_sft_eval/chain-depth-v3-s1/bfcl/examples.jsonl \
        --bfcl-probe results/bfcl_probe/chain-depth-v3-s1/examples.jsonl

Pre-registered 2026-11-04 (docs/research/replication_2026-11-04.md), before any of the runs.
Inputs: greedy multi-turn NESTful dialogues (``router_smoke`` folder ``<model>_<nestful-tag>``,
depths 1–4, optionally 6), the 48-token probe (``router_probe2`` folder of the same name), the
full-plan condition at 2 and 4 loops (``<model>_<plan-tag>``), optionally BFCL outcomes at 1/2/4
loops and the BFCL probe. Each prediction is reported SUPPORTED / NOT SUPPORTED / N/A:

* P1 formulation: among requests fixed by 4 loops (wrong at 2, right at 4), the share whose
  2-loop error is ``wrong_plan`` or ``wrong_operation`` (``depth_anatomy.classify``) > .5.
* P2 compounding: 2→4 gain on 6–8 gold calls minus on 2–3 gold calls, bootstrap 90% lower bound
  > 0 (N/A if either bucket has < 30 requests).
* P3 plan ablation: gain_none − gain_full (gain = acc_4 − acc_2, same requests), 90% lower
  bound > 0.
* P4 stability (only if depth 6 was run): share of 4-loop fixes also right at 6 loops >= .6.
* P5 rescue limit: AUC of the early-feature router (12 loop-1/2 features, logistic per depth,
  score P_4 − P_2) for "4 loops right" among requests wrong at 2 loops <= .72. Router frozen on
  ``--rescue-train`` slices of the same model if given, else 5x5 cross-validation.
* P6 short horizon (only with BFCL): "4 helps" AUC on BFCL > the NESTful "4 helps" AUC of the same
  model (both cross-validated), and BFCL rescue AUC <= .72.
Also: accuracy by depth, gain by gold calls, error-type shares at 2 and 4 loops.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from depth_anatomy import classify, gold_values
from ouro_nestful_multiturn import requests
from phase_a_labels import auc
from router_diagnosis import predict, standardise
from router_probe2_analysis import features_escalate

SMOKE, PROBE = Path("results/router_smoke"), Path("results/router_probe2")
BUCKETS = ((2, 3), (4, 5), (6, 8), (9, 99))


def rows(folder: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, open(folder / "examples.jsonl"))}


def probes(model: str, tags: list[str]) -> dict[str, dict]:
    out = {}
    for t in tags:
        for r in rows(PROBE / f"{model}_{t}").values():
            r["loop12_disagree"] = sum(x[0][3] != x[1][3] for x in r["probe"])
            out[r["id"]] = r
    return out


def boot(f, n) -> list[float]:
    rng = np.random.default_rng(0)
    vals = [f(rng.integers(0, n, n)) for _ in range(2000)]
    return [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]


def cv_score(x, y2, y4) -> np.ndarray:
    n, s = len(x), np.zeros(len(x))
    for rep in range(5):
        order = np.random.default_rng(rep).permutation(n)
        for f in range(5):
            te = order[f::5]
            tr = np.setdiff1d(order, te)
            a, b = standardise(x[tr], x[te])
            s[te] += (predict(a, y4[tr], b) - predict(a, y2[tr], b)) / 5
    return s


def cv_prob(x, y) -> np.ndarray:
    n, s = len(x), np.zeros(len(x))
    for rep in range(5):
        order = np.random.default_rng(rep).permutation(n)
        for f in range(5):
            te = order[f::5]
            tr = np.setdiff1d(order, te)
            a, b = standardise(x[tr], x[te])
            s[te] += predict(a, y[tr], b) / 5
    return s


def verdict(ok: bool | None) -> str:
    return "N/A" if ok is None else ("SUPPORTED" if ok else "NOT SUPPORTED")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="router_smoke folder prefix")
    ap.add_argument("--nestful-tag", required=True)
    ap.add_argument("--plan-tag", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--rescue-train", nargs="*", default=[], help="tags to freeze the router on")
    ap.add_argument("--bfcl-outcomes", default=None)
    ap.add_argument("--bfcl-probe", default=None)
    ap.add_argument("--tasks", default=None, help="NESTful-format task file (default NESTful)")
    ap.add_argument("--deep-tag", default=None, help="extra folder with depth-6 rows, joined by id")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    examples, functions = requests(args.start + args.n, args.tasks)
    examples = examples[args.start :]
    base = rows(SMOKE / f"{args.model}_{args.nestful_tag}")
    if args.deep_tag:
        for i, r in rows(SMOKE / f"{args.model}_{args.deep_tag}").items():
            if i in base:
                base[i]["runs"].update(r["runs"])
    plan = rows(SMOKE / f"{args.model}_{args.plan_tag}")
    ids = [e.id for e in examples]
    assert all(i in base and i in plan for i in ids), "missing requests"
    depths = [d for d in (1, 2, 3, 4, 6) if f"d{d}_greedy" in base[ids[0]]["runs"]]
    y = {
        d: np.array([base[i]["runs"][f"d{d}_greedy"]["correct"] for i in ids], float)
        for d in depths
    }
    yp = {
        d: np.array([plan[i]["runs"][f"d{d}_greedy"]["correct"] for i in ids], float)
        for d in (2, 4)
    }
    calls = np.array([len(e.calls) for e in examples])
    n = len(ids)
    report = {"model": args.model, "slice": [args.start, args.start + args.n], "n": n}
    report["accuracy"] = {d: float(y[d].mean()) for d in depths}
    report["plan_accuracy"] = {d: float(yp[d].mean()) for d in (2, 4)}

    # P1 formulation and error types
    cats = {2: [], 4: []}
    for e in examples:
        g = gold_values(e, functions)
        for d in (2, 4):
            cats[d].append(classify(base[e.id]["runs"][f"d{d}_greedy"], e, functions, g)[0])
    fixed = [k for k in range(n) if y[2][k] == 0 and y[4][k] == 1]
    types = Counter(cats[2][k] for k in fixed)
    share = (types["wrong_plan"] + types["wrong_operation"]) / max(1, len(fixed))
    report["error_share"] = {d: dict(Counter(cats[d]).most_common()) for d in (2, 4)}
    report["P1_formulation"] = {
        "n_fixed": len(fixed),
        "types": dict(types.most_common()),
        "share": share,
        "verdict": verdict(share > 0.5 if fixed else None),
    }

    # P2 compounding
    gain = y[4] - y[2]
    report["gain_by_gold_calls"] = {
        f"{lo}-{hi}": {
            "n": int(((calls >= lo) & (calls <= hi)).sum()),
            "gain": float(gain[(calls >= lo) & (calls <= hi)].mean())
            if ((calls >= lo) & (calls <= hi)).any()
            else None,
        }
        for lo, hi in BUCKETS
    }
    short, long_ = np.where(calls <= 3)[0], np.where((calls >= 6) & (calls <= 8))[0]
    if len(short) >= 30 and len(long_) >= 30:
        rng = np.random.default_rng(0)
        vals = [
            gain[rng.choice(long_, len(long_))].mean() - gain[rng.choice(short, len(short))].mean()
            for _ in range(2000)
        ]
        ci = [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]
        report["P2_compounding"] = {
            "difference": float(gain[long_].mean() - gain[short].mean()),
            "ci90": ci,
            "verdict": verdict(ci[0] > 0),
        }
    else:
        report["P2_compounding"] = {"n_short": len(short), "n_long": len(long_), "verdict": "N/A"}

    # P3 plan ablation
    diff = gain - (yp[4] - yp[2])
    ci = boot(lambda i: diff[i].mean(), n)
    report["P3_plan_ablation"] = {
        "gain_none": float(gain.mean()),
        "gain_full_plan": float((yp[4] - yp[2]).mean()),
        "difference": float(diff.mean()),
        "ci90": ci,
        "verdict": verdict(ci[0] > 0),
    }

    # P4 stability
    if 6 in y:
        st = float(np.mean([y[6][k] for k in fixed])) if fixed else None
        report["P4_stability"] = {
            "share_right_at_6": st,
            "verdict": verdict(st >= 0.6 if st is not None else None),
        }
    else:
        report["P4_stability"] = {"verdict": "N/A"}

    # P5 rescue limit
    pr = probes(args.model, [args.nestful_tag])
    x = np.nan_to_num(np.stack([features_escalate(pr[i], base[i]["query"]) for i in ids]))
    helps = ((y[2] == 0) & (y[4] == 1)).astype(float)
    m = y[2] == 0
    if args.rescue_train:
        tr_rows, tr_pr = {}, probes(args.model, args.rescue_train)
        for t in args.rescue_train:
            tr_rows.update(rows(SMOKE / f"{args.model}_{t}"))
        tid = [i for i in tr_rows if i in tr_pr and i not in set(ids)]
        xt = np.nan_to_num(
            np.stack([features_escalate(tr_pr[i], tr_rows[i]["query"]) for i in tid])
        )
        t2 = np.array([tr_rows[i]["runs"]["d2_greedy"]["correct"] for i in tid], float)
        t4 = np.array([tr_rows[i]["runs"]["d4_greedy"]["correct"] for i in tid], float)
        a, b = standardise(xt, x)
        score = predict(a, t4, b) - predict(a, t2, b)
        mt = t2 == 0
        a2, b2 = standardise(xt[mt], x[m])
        rescue_score = predict(a2, t4[mt], b2)
        how = f"frozen on {len(tid)} requests ({', '.join(args.rescue_train)})"
    else:
        score = cv_score(x, y[2], y[4])
        rescue_score = cv_prob(x[m], y[4][m])
        how = "5x5 cross-validation"
    r_auc = auc(rescue_score, y[4][m])
    report["P5_rescue_limit"] = {
        "router": how,
        "auc_4_helps": auc(score, helps),
        "rescue_auc": r_auc,
        "n_failures_at_2": int(m.sum()),
        "verdict": verdict(r_auc <= 0.72),
    }

    # P6 short horizon (BFCL)
    if args.bfcl_outcomes and args.bfcl_probe:
        out = {}
        for r in map(json.loads, open(args.bfcl_outcomes)):
            out.setdefault(r["id"], {})[r["steps"]] = r["correct"]
        bp = [json.loads(z) for z in open(args.bfcl_probe)]
        for p in bp:
            p["loop12_disagree"] = sum(t[0][3] != t[1][3] for t in p["probe"])
        bp = [p for p in bp if all(d in out.get(p["id"], {}) for d in (2, 4))]
        b2 = np.array([out[p["id"]][2] for p in bp], float)
        b4 = np.array([out[p["id"]][4] for p in bp], float)
        bx = np.nan_to_num(np.stack([features_escalate(p, "") for p in bp]))
        bh = ((b2 == 0) & (b4 == 1)).astype(float)
        bm = b2 == 0
        b_auc = auc(cv_score(bx, b2, b4), bh)
        b_rescue = auc(cv_prob(bx[bm], b4[bm]), b4[bm])
        nest_auc = auc(cv_score(x, y[2], y[4]), helps)
        report["P6_short_horizon"] = {
            "bfcl_accuracy": {2: float(b2.mean()), 4: float(b4.mean())},
            "bfcl_auc_4_helps": b_auc,
            "nestful_auc_4_helps_cv": nest_auc,
            "bfcl_rescue_auc": b_rescue,
            "verdict": verdict(b_auc > nest_auc and b_rescue <= 0.72),
        }
    else:
        report["P6_short_horizon"] = {"verdict": "N/A"}

    path = Path(args.out or f"results/replication/{args.model}_{args.nestful_tag}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=1, default=float)
    path.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
