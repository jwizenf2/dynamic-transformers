"""MathQA → NESTful-format tool-calling tasks: NESTful's own construction on MathQA train/val.

Why: NESTful's executable (math) portion, our multi-turn test set, was built from the **MathQA
test split** (NESTful paper, arXiv 2409.03797, §3: "we used the test set of MathQA where the
'problem' definition is the query", annotated formulas parsed into nested calls over 40 tools,
samples kept only if executing the calls reproduces the gold answer). Running the same
construction on MathQA **train/validation** (29,837 / 4,475 problems, Apache-2.0) gives training
and calibration tasks from the same distribution, disjoint from the test split by construction.
Our synthetic lookup chains (``chains.py``) were solved at every loop depth, so they carried no
signal about when loops matter; MathQA problems need arithmetic planning and NESTful accuracy
rises steeply with loops (multi-turn d1–d4 .000/.083/.175/.217).

Measured fidelity (2026-10-21, MathQA test vs NESTful's 1,415 math samples, matched by
normalised problem text):
* ``formula_calls`` reproduces NESTful's gold call sequence exactly for 1,407/1,415; the other 8
  are NESTful truncating decimals (22.5 → 22, 0.2778 → 2778); we keep the correct values.
* NESTful's query text is the original AQuA-RAT question (cased, untokenised) for ~93% of
  samples, MathQA's lowercased tokenised text otherwise; AQuA-RAT (Apache-2.0) recovers the
  original for 97.7% of MathQA train problems (``original_text``).
* NESTful's tool lists: the gold tools, plus add/subtract/multiply/divide/inverse (present in
  ≥99.7% of lists when not gold), plus random other math tools; 7–12 tools, mostly 9–10, shuffled
  (``tool_list``).
* NESTful has no single-call samples; we also require ``MIN_CALLS`` = 2. Resulting tasks
  (16,814 train, 2,517 validation) match NESTful's mean calls (5.11 / 5.04 vs 5.06) and mean
  dependency depth (3.62 / 3.59 vs 3.58).
* NESTful's answer filter could not be reproduced exactly: 268 MathQA test problems whose calls
  execute to exactly the correct option are not in NESTful. We keep a problem when its executed
  result is within ``ANSWER_REL_TOL`` (1%) of the correct option's number; on the test split this
  keeps 1,377/1,415 NESTful samples and 433 others. The gold answer of a task is the executed
  value (NESTful's gold answers equal their executed values for 1,416/1,416).

Formula grammar (MathQA ``annotated_formula``): ``op(arg, arg, ...)`` with arguments that are
nested calls, numbers (``36``, ``0.5``, ``2/3``) or constants ``const_<v>`` (``const_100`` = 100,
``const_0_25`` = 0.25, ``const_pi`` = π). Calls are emitted in post-order (arguments left to
right, then the call) with labels ``$var_1``, ``$var_2``, ... and positional arguments
``arg_0, arg_1``; a nested call becomes the reference ``"$var_k.result$"``. Shared
subexpressions are not merged (NESTful repeats them too).

Dialogue rendering (training data): the Hermes multi-turn protocol of
``scripts/lab/ouro_nestful_multiturn.py``. Calls are grouped into turns by dependency level
(level = 1 + max level of the calls it references; every call of a level goes in one assistant
turn), each turn's calls are executed with NESTful's functions and returned as
``<tool_response>`` blocks (``nestful.tool_responses``), and the last message states the final
result. ``args="literal"`` writes earlier results as their values (the convention of our chain
SFT data); ``args="ref"`` keeps ``"$var_k.result$"`` with call labels counted over the dialogue.
"""

from __future__ import annotations

import json
import math
import random
import re
from types import ModuleType
from typing import Any

from dht.toolcall.nestful import _refs, execute, run_call, tool_responses

ANSWER_REL_TOL = 0.01
MIN_CALLS = 2  # NESTful has no single-call samples (2–53 calls)
CORE_TOOLS = ("add", "subtract", "multiply", "divide", "inverse")
# Tool-list sizes of NESTful's 1,416 math samples (one outlier of 18 dropped).
LIST_SIZES = {7: 14, 8: 127, 9: 520, 10: 676, 11: 71, 12: 7}

_TOKEN = re.compile(r"\s*([A-Za-z_][A-Za-z_0-9]*|-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?|[(),])")


def normalize(text: str) -> str:
    """Lowercase alphanumerics only: the key that matches MathQA, AQuA-RAT and NESTful text."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _tokens(text: str) -> list[str]:
    out, pos, text = [], 0, text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise ValueError(f"unparseable formula at {text[pos : pos + 12]!r}")
        out.append(m.group(1))
        pos = m.end()
    return out


def parse_formula(text: str) -> Any:
    """``annotated_formula`` → nested ``(op, [args])`` tuples; leaves stay strings."""
    toks = _tokens(text)
    i = 0

    def expr() -> Any:
        nonlocal i
        tok = toks[i]
        i += 1
        if i < len(toks) and toks[i] == "(":
            i += 1
            args = []
            while toks[i] != ")":
                args.append(expr())
                if toks[i] == ",":
                    i += 1
            i += 1
            return (tok, args)
        return tok

    tree = expr()
    if i != len(toks):
        raise ValueError(f"trailing tokens in formula: {toks[i:]}")
    return tree


def _number(v: float) -> int | float:
    return int(v) if float(v).is_integer() else v


def leaf_value(tok: str) -> int | float:
    """A number or constant leaf: ``const_100`` → 100, ``const_0_25`` → 0.25, ``2/3`` → 0.666…"""
    if tok.startswith("const_"):
        name = tok[len("const_") :]
        if name == "pi":
            return math.pi
        return _number(float(name.replace("_", ".", 1)))
    if "/" in tok:
        a, b = tok.split("/")
        return float(a) / float(b)
    return _number(float(tok))


def formula_calls(text: str) -> list[dict[str, Any]]:
    """NESTful-format gold calls (post-order, ``arg_k``, ``$var_k`` labels) of a formula."""
    calls: list[dict[str, Any]] = []

    def walk(node: Any) -> Any:
        if isinstance(node, str):
            return leaf_value(node)
        name, args = node
        values = [walk(a) for a in args]
        calls.append(
            {
                "name": name,
                "label": f"$var_{len(calls) + 1}",
                "arguments": {f"arg_{k}": v for k, v in enumerate(values)},
            }
        )
        return f"$var_{len(calls)}.result$"

    tree = parse_formula(text)
    if not isinstance(tree, tuple):
        raise ValueError("formula is a bare number")
    walk(tree)
    return calls


def option_value(options: str, correct: str) -> float | None:
    """Number in MathQA's correct option (``"a ) rs . 400 , b ) …"``, ``"d ) - 49"``,
    ``"c ) 3 : 8"`` → 0.375); None when it has none."""
    parts = re.split(r"(?:^|,)\s*([a-e])\s*\)", options)
    by_letter = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}
    text = by_letter.get(correct.strip(), "")
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text).replace(" . ", ".")
    text = text.replace("- ", "-").replace("− ", "-").replace("−", "-")
    m = re.search(r"-?\d+(?:\.\d+)?(?:\s*[/:]\s*\d+(?:\.\d+)?)?", text)
    if not m:
        return None
    tok = m.group(0).replace(" ", "")
    for sep in "/:":
        if sep in tok:
            a, b = tok.split(sep)
            return float(a) / float(b) if float(b) else None
    return float(tok)


def tool_list(gold: set[str], catalog: dict[str, dict], rng: random.Random) -> list[dict[str, Any]]:
    """NESTful-style tool library: gold ∪ core tools ∪ random math tools, shuffled."""
    sizes, weights = zip(*LIST_SIZES.items(), strict=True)
    size = rng.choices(sizes, weights)[0]
    names = list(dict.fromkeys([*sorted(gold), *CORE_TOOLS]))
    rest = sorted(set(catalog) - set(names))
    names += rng.sample(rest, max(0, min(len(rest), size - len(names))))
    rng.shuffle(names)
    return [catalog[n] for n in names]


def levels(calls: list[dict[str, Any]]) -> list[int]:
    """Dependency level per call: 1 + max level of the calls it references (1 if none)."""
    level: dict[str, int] = {}
    out = []
    for i, call in enumerate(calls):
        refs = [r for v in call["arguments"].values() for r in _refs(v)]
        lv = 1 + max((level.get(r, 0) for r in refs), default=0)
        level[f"var_{i + 1}"] = lv
        out.append(lv)
    return out


def _fmt(value: Any) -> str:
    return json.dumps(value) if isinstance(value, float) else str(value)


def render_dialogue(
    query: str,
    calls: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    functions: ModuleType,
    system_template: str,
    args: str = "literal",
) -> list[dict[str, str]]:
    """Hermes conversation (``{"from", "value"}`` turns), one assistant turn per dependency level.

    Calls are re-ordered by level (stable within a level), so ``$var_k`` labels in ``ref`` mode
    are renumbered to the order in which the model writes them. Raises ``ValueError`` when a
    call fails to execute.
    """
    lv = levels(calls)
    order = sorted(range(len(calls)), key=lambda i: (lv[i], i))
    new_label = {f"var_{old + 1}": f"var_{new + 1}" for new, old in enumerate(order)}
    results: dict[str, dict[str, Any]] = {}
    values: dict[str, Any] = {}  # original label → executed value
    system = system_template.format(tools=json.dumps(tools, ensure_ascii=False))
    conv = [{"from": "system", "value": system}, {"from": "human", "value": query}]
    n = 0
    last: Any = None
    for level in sorted(set(lv)):
        blocks, outcomes = [], []
        for i in (j for j in order if lv[j] == level):
            call = calls[i]
            written = {}
            for k, v in call["arguments"].items():
                refs = _refs(v)
                if not refs:
                    written[k] = v
                elif args == "literal":
                    written[k] = values[refs[0]]
                else:
                    written[k] = f"${new_label[refs[0]]}.result$"
            ok, value = run_call(
                {"name": call["name"], "arguments": written}, n, tools, functions, results
            )
            if not ok:
                raise ValueError(f"call {call['name']} failed: {value}")
            n += 1
            values[f"var_{i + 1}"] = value
            last = value
            outcomes.append((call["name"], True, value))
            blocks.append(
                "<tool_call>\n"
                + json.dumps({"name": call["name"], "arguments": written}, ensure_ascii=False)
                + "\n</tool_call>"
            )
        conv.append({"from": "gpt", "value": "\n".join(blocks)})
        conv.append({"from": "tool", "value": tool_responses(outcomes)})
    conv.append({"from": "gpt", "value": f"The final result is {_fmt(last)}."})
    return conv


def build_task(
    row: dict[str, Any],
    catalog: dict[str, dict],
    functions: ModuleType,
    rng: random.Random,
    original_text: dict[str, str] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """One MathQA row → NESTful-format task ``{"id", "query", "tools", "calls", "answer"}``.

    Returns ``(task, "ok")`` or ``(None, reason)`` with reason in parse / single (fewer than
    ``MIN_CALLS`` calls) / op / exec / no_option / mismatch. ``original_text`` maps
    ``normalize(problem)`` to the AQuA-RAT question.
    """
    try:
        calls = formula_calls(row["annotated_formula"])
    except (ValueError, IndexError):
        return None, "parse"
    if len(calls) < MIN_CALLS:
        return None, "single"
    gold = {c["name"] for c in calls}
    if not gold <= set(catalog):
        return None, "op"
    tools = tool_list(gold, catalog, rng)
    ok, value = execute(calls, tools, functions)
    if not ok or isinstance(value, (tuple, list, bool)) or not math.isfinite(float(value)):
        return None, "exec"
    target = option_value(row["options"], row["correct"])
    if target is None:
        return None, "no_option"
    if abs(value - target) > ANSWER_REL_TOL * max(abs(target), 1e-9):
        return None, "mismatch"
    query = (original_text or {}).get(normalize(row["Problem"]), row["Problem"]).strip()
    return {
        "query": query,
        "tools": tools,
        "calls": calls,
        "answer": float(value),
        "category": row.get("category", ""),
    }, "ok"
