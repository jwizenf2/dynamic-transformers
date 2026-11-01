"""One-way depth switch: decode at 2 loops, switch to 4 loops when loops 1–2 are uncertain.

    PYTHONPATH=scripts/lab python scripts/lab/router_switch.py

Offline, exploratory, development data only (the 531 MathQA router-training tasks with
``router_trace.py`` traces, which reproduce the 2-loop greedy dialogues token for token).

Policy: decode greedily at 2 loops; before emitting each assistant token, if the loop-1/2 signal
of that step exceeds τ, switch the request to 4 loops for the rest of the dialogue. A switch
computes loops 3–4 over the whole context so far (prompt, tool responses and tokens already
written; loops 1–2 caches are reused, since step-s states depend only on steps <= s) and then
decodes at 4 loops. Until the first token where the 2- and 4-loop greedy dialogues differ (the
fork), both dialogues are identical, so a switch at or before the fork reproduces the 4-loop
dialogue exactly, and the request's compute is exactly that of the 4-loop run (4·T_4,
T = prompt + generated tokens): nothing is wasted. Without a trigger the request costs 2·T_2 and
gets the 2-loop outcome. A trigger after the fork continues at 4 loops from a 2-loop prefix
that already differs; its outcome is not in the data: bounds [min, max] of the 2- and 4-loop
outcomes, midpoint the 2-loop outcome, cost ≈ 4·T_2. Requests whose dialogues never differ get
the common outcome; cost 4·T_4 if triggered, else 2·T_2.

Signals (per decoding step of the 2-loop trace): loop-2 entropy, negative loop-2 top-2 margin,
KL(p_2 ‖ p_1), negative log p_1 of the emitted token, 1 − loop-2 top probability; on all tokens or
only tokens containing a digit. τ sweeps quantiles of the signal over all clean-set tokens.
Selection on the 292 clean tasks; the 239 pool tasks check replication. Reported: fixed 2 and 4
loops, the curve, accuracy (lower bound / midpoint) at 0.80, 0.85, 0.90x fixed-4 cost, the gain
over the 2–4 random mix at the same cost, the share of switches that land after the fork, and an
oracle trigger (switch exactly at the fork iff the 4-loop outcome is better).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from router_diagnosis import divergence, tokenizer

M = "ouro-chain-depth-v3"
SETS = {"clean": (M, f"{M}_clean"), "pool": (f"{M}_testpool", f"{M}_testpool")}
SIGNALS = ("ent2", "neg_mar2", "kl21", "neg_lp1", "one_minus_top2")
BUDGETS = (0.80, 0.85, 0.90)


def load(set_name: str, tok) -> list[dict]:
    smoke_dir, trace_dir = SETS[set_name]
    smoke = {
        r["id"]: r
        for r in map(json.loads, open(f"results/router_smoke/{smoke_dir}/examples.jsonl"))
    }
    rows = []
    for t in map(json.loads, open(f"results/router_trace/{trace_dir}/examples.jsonl")):
        s = smoke[t["id"]]
        d4 = s["runs"]["d4_greedy"]
        sig = t["turn_signals"]
        series = {
            "ent2": np.concatenate([x["ent2"] for x in sig]),
            "neg_mar2": -np.concatenate([x["mar2"] for x in sig]),
            "kl21": np.concatenate([x["kl21"] for x in sig]),
            "neg_lp1": -np.concatenate([x["lp1"] for x in sig]),
            "one_minus_top2": 1 - np.concatenate([x["top2"] for x in sig]),
        }
        digit = np.concatenate([x["digit"] for x in sig]).astype(bool)
        pos, turn = divergence(tok, t["texts"], d4["texts"])
        if pos >= 10**6:
            fork = None  # identical dialogues
        elif turn >= len(sig):
            fork = len(digit)  # the 2-loop dialogue ended first: any switch is before the fork
        else:
            fork = pos + turn  # one stop-step entry per earlier turn in the trace arrays
        rows.append(
            {
                "y2": float(t["correct"]),
                "y4": float(d4["correct"]),
                "t2": float(t["prompt_tokens"] + t["tokens"]),
                "t4": float(d4["prompt_tokens"] + d4["tokens"]),
                "series": series,
                "digit": digit,
                "fork": fork,
            }
        )
    return rows


def evaluate(rows, signal: str, mask: str, tau: float) -> tuple[float, float, float, float, float]:
    """(cost, accuracy low, mid, high, share of switches after the fork)."""
    cost = low = mid = high = 0.0
    late = switched = 0
    for r in rows:
        v = r["series"][signal]
        hit = v > tau if mask == "all" else (v > tau) & r["digit"]
        first = int(np.argmax(hit)) if hit.any() else None
        if first is None:
            c, lo, md, hi = 2 * r["t2"], r["y2"], r["y2"], r["y2"]
        elif r["fork"] is None or first <= r["fork"]:
            switched += 1
            c, lo, md, hi = 4 * r["t4"], r["y4"], r["y4"], r["y4"]
        else:
            switched += 1
            late += 1
            pair = (r["y2"], r["y4"])
            c, lo, md, hi = 4 * r["t2"], min(pair), r["y2"], max(pair)
        cost, low, mid, high = cost + c, low + lo, mid + md, high + hi
    n = len(rows)
    base = 4 * np.mean([r["t4"] for r in rows])
    return cost / n / base, low / n, mid / n, high / n, late / max(1, switched)


def curve(rows, signal, mask, taus):
    return sorted(evaluate(rows, signal, mask, t) for t in taus)


def at_budget(points, budget):
    """Best (low, mid) accuracy among points with cost <= budget."""
    ok = [p for p in points if p[0] <= budget]
    if not ok:
        return None
    best = max(ok, key=lambda p: p[1])
    return {
        "cost": best[0],
        "acc_low": best[1],
        "acc_mid": best[2],
        "acc_high": best[3],
        "late_switch_share": best[4],
    }


def summary(rows, signal, mask, taus) -> dict:
    pts = curve(rows, signal, mask, taus)
    base = 4 * np.mean([r["t4"] for r in rows])
    c2 = 2 * np.mean([r["t2"] for r in rows]) / base
    a2, a4 = np.mean([r["y2"] for r in rows]), np.mean([r["y4"] for r in rows])
    out = {}
    for b in BUDGETS:
        best = at_budget(pts, b)
        if best:
            mix = a2 + (a4 - a2) * (best["cost"] - c2) / (1 - c2)
            best["gain_low_vs_mix"] = best["acc_low"] - mix
            best["gain_mid_vs_mix"] = best["acc_mid"] - mix
        out[f"{b:.2f}x"] = best
    reach = [p[0] for p in pts if p[1] >= a4]
    out["cost_low_bound_reaches_fixed4"] = min(reach) if reach else None
    return out


def oracle(rows) -> dict:
    cost = acc = 0.0
    for r in rows:
        if r["fork"] is not None and r["y4"] > r["y2"]:
            cost, acc = cost + 4 * r["t4"], acc + 1
        else:
            cost, acc = cost + 2 * r["t2"], acc + r["y2"]
    base = 4 * np.mean([r["t4"] for r in rows])
    return {"cost": cost / len(rows) / base, "accuracy": acc / len(rows)}


def main() -> None:
    tok = tokenizer()
    data = {s: load(s, tok) for s in SETS}
    report = {}
    for s, rows in data.items():
        base = 4 * np.mean([r["t4"] for r in rows])
        report[s] = {
            "n": len(rows),
            "fixed_d2": [
                2 * np.mean([r["t2"] for r in rows]) / base,
                np.mean([r["y2"] for r in rows]),
            ],
            "fixed_d4": [1.0, np.mean([r["y4"] for r in rows])],
            "oracle_switch": oracle(rows),
        }
    clean_vals = {sig: np.concatenate([r["series"][sig] for r in data["clean"]]) for sig in SIGNALS}
    policies = {}
    for sig in SIGNALS:
        taus = np.quantile(clean_vals[sig], 1 - np.geomspace(0.5, 1e-5, 60))
        for mask in ("all", "digit"):
            policies[f"{sig}/{mask}"] = {
                s: summary(rows, sig, mask, taus) for s, rows in data.items()
            }
    report["policies"] = policies
    chosen = max(
        policies,
        key=lambda k: (policies[k]["clean"]["0.90x"] or {"acc_low": -1})["acc_low"],
    )
    report["selection"] = {
        "rule": "max clean accuracy lower bound at <= 0.90x",
        "chosen": chosen,
        "result": policies[chosen],
    }
    out = Path("results/router_switch")
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=1, default=float)
    (out / "report.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
