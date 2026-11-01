"""What extra loops buy in multi-step tool use (offline analysis of existing NESTful dialogues).

    python scripts/lab/depth_anatomy.py

Data: chain v3 greedy multi-turn dialogues on NESTful requests [0, 720) at 1–6 loops
(``router_smoke`` tags ``nestful``, ``nestful240`` for depths 1–4 and ``*_deep`` for 5, 6, 8) and
the 48-token loop-1–4 probe of the first answer (``router_probe2`` tags ``nestful``,
``nestful240``). Exploratory and descriptive (all 720 requests were seen in earlier tests); no
model is run.

1. Error taxonomy. Each dialogue is replayed from its stored assistant texts (``parse_hermes`` +
   ``run_call``, the functions are pure, so the replay equals the original execution; checked by
   comparing the replayed final value's correctness with the stored outcome). The gold chain is
   executed to get its intermediate values. A failed dialogue gets the first matching category:
   ``format`` (a ``<tool_call>`` block that does not parse, or an unknown function or argument
   name), ``exec_error`` (a call that fails to run: unresolved reference, out of range, ...),
   ``unfinished`` (turn limit), ``stopped_early`` (last value is a gold intermediate, not the
   answer), then the first predicted call whose value is not a gold intermediate value:
   ``wrong_operation`` (its function is not in the gold chain), ``invented_value`` (an argument
   is neither a number in the question, a small constant, nor an earlier result in the dialogue),
   ``argument_order`` (swapping its two arguments gives a gold intermediate value),
   ``wrong_plan`` (gold operations on available numbers, combined into the wrong quantities: a
   formulation error); otherwise
   ``other``. Reported per depth, and for the requests 4 loops fixes (wrong at 2, right at 4):
   the 2-loop error type and the relative position of the first divergent call in the chain.
2. Depth x chain length: accuracy per depth (1–6) by gold call count and by dependency depth,
   the 2→4 gain per bucket with a bootstrap 90% interval, and a per-step reliability fit
   acc_d(L) ≈ a_d · s_d^L (least squares on log accuracy by call-count bucket, weighted by n).
3. When tokens settle across loops: from the probe (per token and loop: whether the loop's argmax
   equals loop 4's), the settle loop of each token (first loop from which every later loop agrees
   with loop 4), by token type: ``structure`` (JSON/markup), ``function_name``, ``number``
   (argument values), ``other``. Shares settled by loops 1, 2, 3.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ouro_nestful_multiturn import requests  # noqa: E402
from ouro_sft_eval import TOOL_CALL, parse_hermes  # noqa: E402

from dht.toolcall.nestful import answers_match, run_call  # noqa: E402

SMOKE = Path("results/router_smoke")
PROBE = Path("results/router_probe2")
M = "ouro-chain-depth-v3"
SLICES = (("nestful", "nestful_deep", 0), ("nestful240", "nestful240_deep", 240))
DEPTHS = (1, 2, 3, 4, 5, 6)
CONSTANTS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 24, 60, 100, 1000, 360, 365, 3600}
NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
CATEGORIES = (
    "format",
    "exec_error",
    "unfinished",
    "stopped_early",
    "wrong_operation",
    "invented_value",
    "argument_order",
    "wrong_plan",
    "other",
)


def close(a, b, rel=1e-4) -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-9)


def load():
    examples, functions = requests(720)
    rows = []
    for base, deep, start in SLICES:
        extra = {
            r["id"]: r for r in map(json.loads, open(SMOKE / f"{M}_{deep}" / "examples.jsonl"))
        }
        probes = {
            r["id"]: r for r in map(json.loads, open(PROBE / f"{M}_{base}" / "examples.jsonl"))
        }
        for j, r in enumerate(map(json.loads, open(SMOKE / f"{M}_{base}" / "examples.jsonl"))):
            assert examples[start + j].id == r["id"]
            r["runs"].update(extra[r["id"]]["runs"])
            r["example"] = examples[start + j]
            r["probe2"] = probes[r["id"]]
            rows.append(r)
    return rows, functions


def gold_values(e, functions) -> list:
    results, values = {}, []
    for i, call in enumerate(e.calls):
        ok, v = run_call(call, i, e.tools, functions, results)
        values.append(v if ok else None)
    return values


def swapped_is_gold(call: dict, e, functions, gold: list) -> bool:
    """Whether the call with its two literal arguments swapped yields a gold intermediate value."""
    args = call.get("arguments") or {}
    keys = list(args)
    if len(keys) != 2 or any(isinstance(v, str) for v in args.values()):
        return False
    swapped = {**call, "arguments": {keys[0]: args[keys[1]], keys[1]: args[keys[0]]}}
    ok, v = run_call(swapped, 10**6, e.tools, functions, {})
    return ok and any(close(v, g) for g in gold if g is not None)


def classify(run: dict, e, functions, gold: list) -> tuple[str, float | None, bool]:
    """(category, relative position of the first divergent call or None, replay correct)."""
    results, n, available, preds = {}, 0, set(), []
    query_numbers = {float(x.replace(",", "")) for x in NUM.findall(e.query)}
    first_bad, category = None, None
    blocks = sum(len(TOOL_CALL.findall(t)) for t in run["texts"])
    parsed = 0
    for text in run["texts"]:
        calls = parse_hermes(text)
        parsed += len(calls)
        for c in calls:
            ok, v = run_call(c, n, e.tools, functions, results)
            n += 1
            preds.append((c, ok, v))
    replay_ok = bool(preds and preds[-1][1] and answers_match(preds[-1][2], e.answer))
    if run["correct"]:
        return "correct", None, replay_ok
    names = {c["name"] for c in e.calls}
    tool_names = {t["name"] for t in e.tools}
    for c, ok, v in preds:
        if not ok and (c["name"] not in tool_names or "unknown arguments" in str(v)):
            category = "format"
    if blocks > parsed:
        category = "format"
    if category is None and any(not ok for _, ok, _ in preds):
        category = "exec_error"
    if category is None and not run["finished"]:
        category = "unfinished"
    if (
        category is None
        and preds
        and any(close(preds[-1][2], g) for g in gold[:-1])
        and len(preds) < len(e.calls)
    ):
        category = "stopped_early"
    for k, (c, ok, v) in enumerate(preds):
        if ok and not any(close(v, g) for g in gold if g is not None):
            first_bad = k
            if category is None:
                args = [a for a in (c.get("arguments") or {}).values() if not isinstance(a, str)]
                known = query_numbers | CONSTANTS | available
                if c["name"] not in names:
                    category = "wrong_operation"
                elif any(not any(close(a, b, 1e-3) for b in known) for a in args):
                    category = "invented_value"
                elif swapped_is_gold(c, e, functions, gold):
                    category = "argument_order"
                else:
                    category = "wrong_plan"
            break
        if ok and isinstance(v, (int, float)) and abs(v) < 1e300:
            available.add(float(v))
    position = None if first_bad is None else first_bad / max(1, len(e.calls))
    return category or "other", position, replay_ok


def boot_ci(f, n: int) -> list[float]:
    rng = np.random.default_rng(0)
    vals = [f(rng.integers(0, n, n)) for _ in range(2000)]
    return [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]


def taxonomy(rows, functions) -> dict:
    cats = {d: Counter() for d in DEPTHS}
    replay_agree = []
    per_row = defaultdict(dict)
    for i, r in enumerate(rows):
        gold = gold_values(r["example"], functions)
        for d in DEPTHS:
            run = r["runs"][f"d{d}_greedy"]
            cat, pos, replay_ok = classify(run, r["example"], functions, gold)
            replay_agree.append(replay_ok == run["correct"])
            cats[d][cat] += 1
            per_row[i][d] = (cat, pos)
    n = len(rows)
    fixed = [i for i in per_row if per_row[i][2][0] != "correct" and per_row[i][4][0] == "correct"]
    broken = [i for i in per_row if per_row[i][2][0] == "correct" and per_row[i][4][0] != "correct"]
    pos = [per_row[i][2][1] for i in fixed if per_row[i][2][1] is not None]
    return {
        "replay_matches_stored_outcome": float(np.mean(replay_agree)),
        "share_of_requests_by_depth": {
            d: {c: cats[d][c] / n for c in ("correct", *CATEGORIES)} for d in DEPTHS
        },
        "fixed_by_4_loops": {
            "n": len(fixed),
            "error_type_at_2": dict(Counter(per_row[i][2][0] for i in fixed).most_common()),
            "first_divergence_position_at_2": {
                "first_call": float(np.mean([p == 0 for p in pos])) if pos else None,
                "mean_relative_position": float(np.mean(pos)) if pos else None,
                "n_with_position": len(pos),
            },
        },
        "broken_by_4_loops": {
            "n": len(broken),
            "error_type_at_4": dict(Counter(per_row[i][4][0] for i in broken).most_common()),
        },
    }


def chain_length(rows) -> dict:
    y = {d: np.array([r["runs"][f"d{d}_greedy"]["correct"] for r in rows], float) for d in DEPTHS}
    calls = np.array([r["gold_calls"] for r in rows])
    from dht.toolcall.nestful import dependency_depth

    dep = np.array([dependency_depth(r["example"].calls) for r in rows])
    out = {}
    for name, key, bins in (
        ("gold_calls", calls, ((2, 2), (3, 3), (4, 5), (6, 8), (9, 99))),
        ("dependency_depth", dep, ((1, 1), (2, 2), (3, 3), (4, 5), (6, 99))),
    ):
        table = {}
        for lo, hi in bins:
            m = np.where((key >= lo) & (key <= hi))[0]
            if not len(m):
                continue
            gain = y[4][m] - y[2][m]
            table[f"{lo}-{hi}"] = {
                "n": int(len(m)),
                "accuracy": {d: float(y[d][m].mean()) for d in DEPTHS},
                "gain_2_to_4": float(gain.mean()),
                "gain_ci90": boot_ci(lambda i, g=gain: g[i].mean(), len(m)),
            }
        out[name] = table
    # per-step reliability: log acc_d(L) = log a_d + L log s_d over call-count buckets
    fit = {}
    for d in DEPTHS:
        xs, ys, ws = [], [], []
        for L in sorted(set(calls)):
            m = calls == L
            if m.sum() >= 10 and y[d][m].mean() > 0:
                xs.append(L)
                ys.append(math.log(y[d][m].mean()))
                ws.append(m.sum())
        w = np.sqrt(np.array(ws, float))
        coef, *_ = np.linalg.lstsq(np.stack([w, w * np.array(xs)], 1), w * np.array(ys), rcond=None)
        fit[d] = {"a": float(math.exp(coef[0])), "per_step_s": float(math.exp(coef[1]))}
    out["per_step_fit"] = fit
    return out


def token_kind(text: str, names: list, numbers: list, a: int, b: int) -> str:
    """Type of the token at characters [a, b) of the probe text."""
    if any(a < e and b > s for s, e in names):
        return "function_name"
    if any(a < e and b > s for s, e in numbers):
        return "number"
    if not text[a:b].strip() or re.fullmatch(r'[\s{}\[\]":,<>/_a-z]*', text[a:b]):
        return "structure"
    return "other"


def token_settling(rows) -> dict:
    from transformers import PreTrainedTokenizerFast

    snap = next(Path.home().glob(".cache/huggingface/hub/models--ByteDance--Ouro-2.6B/snapshots/*"))
    tok = PreTrainedTokenizerFast(tokenizer_file=str(snap / "tokenizer.json"))
    settle = defaultdict(list)
    disagree2 = defaultdict(list)
    for r in rows:
        p = r["probe2"]
        ids = p["token_ids"]
        text, spans = "", []
        for t in ids[: len(p["probe"])]:
            piece = tok.decode([t])
            spans.append((len(text), len(text) + len(piece)))
            text += piece
        names = [m.span(1) for m in re.finditer(r'"name":\s*"([^"]*)"', text)]
        numbers = [m.span() for m in re.finditer(r"(?<=[:\s\[,])-?\d+(?:\.\d+)?", text)]

        for (a, b), loops in zip(spans, p["probe"], strict=False):
            agree = [bool(loops[k][3]) for k in range(4)]
            first = next(k for k in range(4) if all(agree[k:]))
            settle[token_kind(text, names, numbers, a, b)].append(first + 1)
            disagree2[token_kind(text, names, numbers, a, b)].append(not agree[1])
    return {
        k: {
            "n_tokens": len(v),
            "settled_by_loop": {j: float(np.mean(np.array(v) <= j)) for j in (1, 2, 3)},
            "loop2_differs_from_loop4": float(np.mean(disagree2[k])),
        }
        for k, v in settle.items()
    }


def main() -> None:
    rows, functions = load()
    report = {
        "n": len(rows),
        "taxonomy": taxonomy(rows, functions),
        "chain_length": chain_length(rows),
        "token_settling": token_settling(rows),
    }
    out = Path("results/depth_anatomy")
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=1, default=float)
    (out / "report.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
