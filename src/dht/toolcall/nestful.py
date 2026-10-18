"""NESTful (IBM, Apache-2.0): chained tool calls, used as the dependency-depth axis.

Source: github.com/IBM/NESTFUL at a pinned commit (``data_v2/nestful_data.jsonl``, 1,861
sequences; paper arXiv 2409.03797). Each sample has a query, a tool library, a gold sequence of
calls in which later calls consume earlier results by reference (``"$var_1.result$"``), and an
executed gold answer.

What this module adds (plan section 7, step 3b):
* ``dependency_depth``: the longest chain of calls in the gold sequence, i.e. how many calls must
  finish before the last one can be written. Independent calls do not add depth; ``n_calls``
  is recorded separately so depth and length can be told apart.
* ``execute`` and ``answers_match``: run a predicted sequence with the benchmark's own functions
  and compare with the gold answer. This follows NESTful's "win rate" (src/scorer.py), with
  two stricter choices recorded here: arguments are bound to the function's parameters by the
  tool spec's order and names (NESTful binds by the order the model wrote them), and unknown
  argument names fail the sample. Float answers are rounded to the gold answer's decimals as in
  NESTful; exponent-format gold values use a relative tolerance of 1e-9 instead (NESTful's
  decimal count fails on them).
* ``exact_match``: the predicted sequence equals the gold sequence call by call, with labels
  renumbered by position (NESTful's "full match" is similar but per-call after alignment).

Only the math subset (every gold call is in ``basic_functions.py``, 40 pure functions using
``math``) is executed. The other samples need ~1,800 arbitrary Python files, some importing
os, torch or random, and are not run. Arguments are range-checked before execution so a model
cannot request e.g. ``factorial(10**9)``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

# A reference to an earlier call's output: "$var_3.result$", also "$var3.result" (both appear
# in model outputs; NESTful's scorer accepts both).
_REF = re.compile(r"^\$?(var_?\d+)\.(\w+)\$?$")
_LIMIT = 1e100  # largest |number| passed to any function (3**50 appears in gold)
_SMALL_INT_FUNCS = {"factorial": 1000, "permutation": 10_000, "choose": 10_000}


@dataclass(frozen=True)
class NestfulExample:
    id: str
    query: str
    tools: list[dict[str, Any]]
    calls: list[dict[str, Any]]  # gold: {"name", "label", "arguments"}
    answer: Any


def load_nestful(path: Path) -> list[NestfulExample]:
    """All samples of ``nestful_data.jsonl``, in file order."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        out.append(
            NestfulExample(
                id=row["sample_id"],
                query=row["input"],
                tools=row["tools"],
                calls=row["output"],
                answer=row["gold_answer"],
            )
        )
    return out


def load_nestful_parquet(path: Path) -> list[NestfulExample]:
    """The Hugging Face copy (``ibm-research/nestful``): same rows, JSON fields as strings.

    ``output`` and ``tools`` are JSON text. ``gold_answer`` is JSON text for numbers (all of
    the math subset); non-JSON answers stay strings. Numeric-looking string answers in non-math
    samples (e.g. "   30") decode as numbers here; those samples are not executed.
    """
    import pyarrow.parquet as pq

    def decode(text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    return [
        NestfulExample(
            id=row["sample_id"],
            query=row["input"],
            tools=json.loads(row["tools"]),
            calls=json.loads(row["output"]),
            answer=decode(row["gold_answer"]),
        )
        for row in pq.read_table(path).to_pylist()
    ]


def basic_function_names(functions_file: Path) -> set[str]:
    return set(re.findall(r"^def (\w+)\(", functions_file.read_text(encoding="utf-8"), re.M))


def load_functions(functions_file: Path) -> ModuleType:
    """Import ``basic_functions.py`` (a trusted, SHA256-pinned file: see the lab script)."""
    spec = importlib.util.spec_from_file_location("nestful_basic_functions", functions_file)
    if spec is None or spec.loader is None:
        raise ImportError(functions_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _label(call: dict[str, Any], index: int) -> str:
    """'var_3' for '$var_3'; '$var3' is normalized to 'var_3'. Missing labels are positional."""
    raw = str(call.get("label") or f"var_{index + 1}").strip("$")
    return re.sub(r"^var_?", "var_", raw)


def _refs(value: Any) -> list[str]:
    if isinstance(value, str):
        m = _REF.match(value)
        return [re.sub(r"^var_?", "var_", m.group(1))] if m else []
    if isinstance(value, list):
        return [r for v in value for r in _refs(v)]
    return []


def dependency_depth(calls: list[dict[str, Any]]) -> int:
    """Longest chain of calls (a call with no references has depth 1).

    depth(call) = 1 + max depth of the calls it references. The sequence's depth is the maximum
    over calls. Unknown references count as depth 0.
    """
    depth: dict[str, int] = {}
    for i, call in enumerate(calls):
        refs = [r for v in (call.get("arguments") or {}).values() for r in _refs(v)]
        depth[_label(call, i)] = 1 + max((depth.get(r, 0) for r in refs), default=0)
    return max(depth.values(), default=0)


def parse_sequence(text: str) -> list[dict[str, Any]]:
    """First JSON list of call objects in the model output; [] if none parses."""
    match = re.search(r"\[.*?\](?=\s*(?:\n|$))", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    calls = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            calls.append({"name": item["name"], "label": item.get("label"), "arguments": args})
    return calls


def _in_range(name: str, args: list[Any]) -> bool:
    def ok(v: Any) -> bool:
        if isinstance(v, bool):
            return False
        if isinstance(v, (int, float)):
            return math.isfinite(v) and abs(v) <= _LIMIT
        if isinstance(v, list):
            return len(v) <= 1000 and all(ok(x) for x in v)
        return False  # functions take numbers or lists of numbers

    if not all(ok(v) for v in args):
        return False
    if name in _SMALL_INT_FUNCS and any(
        isinstance(v, (int, float)) and abs(v) > _SMALL_INT_FUNCS[name] for v in args
    ):
        return False
    if name == "power" and len(args) == 2:
        base, exp = args
        if not isinstance(base, (int, float)) or not isinstance(exp, (int, float)):
            return False
        return abs(exp) <= 64 or abs(base) <= 1
    return True


def execute(
    calls: list[dict[str, Any]], tools: list[dict[str, Any]], functions: ModuleType
) -> tuple[bool, Any]:
    """Run a call sequence; (True, last call's result) or (False, reason).

    Each call's arguments are bound positionally in the order of the tool spec's parameters
    (``arg_0, arg_1`` or named); a reference resolves to the named output of an earlier call.
    Only functions present in both the request's tool list and ``functions`` can run.
    """
    if not calls:
        return False, "no calls"
    results: dict[str, dict[str, Any]] = {}
    value: Any = None
    for i, call in enumerate(calls):
        ok, value = run_call(call, i, tools, functions, results)
        if not ok:
            return False, value
    return True, value


def run_call(
    call: dict[str, Any],
    index: int,
    tools: list[dict[str, Any]],
    functions: ModuleType,
    results: dict[str, dict[str, Any]],
) -> tuple[bool, Any]:
    """Run one call; (True, result) or (False, reason). On success the result is stored in
    ``results`` under the call's label (positional ``var_{index + 1}`` when unlabelled), so
    later calls, also in later turns of a multi-turn dialogue, can reference it."""
    spec = {t["name"]: t for t in tools}
    name = call["name"]
    if name not in spec or not hasattr(functions, name):
        return False, f"unknown function {name}"
    params = list((spec[name].get("parameters") or {}).keys())
    given = call.get("arguments") or {}
    if set(given) - set(params):
        return False, f"{name}: unknown arguments {sorted(set(given) - set(params))}"
    args = []
    for p in params:
        if p not in given:
            break  # later parameters must take their defaults (e.g. log's base)
        v = given[p]
        if isinstance(v, str):
            m = _REF.match(v)
            if not m:
                return False, f"{name}: non-numeric argument {v!r}"
            src = re.sub(r"^var_?", "var_", m.group(1))
            if src not in results or m.group(2) not in results[src]:
                return False, f"{name}: unresolved reference {v!r}"
            v = results[src][m.group(2)]
        args.append(v)
    if len(args) < len([p for p in params if p in given]):
        return False, f"{name}: argument after a missing one"
    if not _in_range(name, args):
        return False, f"{name}: arguments out of range"
    try:
        value = getattr(functions, name)(*args)
    except Exception as exc:  # noqa: BLE001 (any failure of a predicted call is a miss)
        return False, f"{name}: {type(exc).__name__}"
    outputs = list((spec[name].get("output_parameters") or {"result": None}).keys())
    results[_label(call, index)] = {outputs[0]: value}
    return True, value


def tool_responses(outcomes: list[tuple[str, bool, Any]]) -> str:
    """Hermes ``<tool_response>`` blocks for one turn's calls: (name, ok, result or reason)."""
    blocks = []
    for name, ok, value in outcomes:
        content = (list(value) if isinstance(value, tuple) else value) if ok else {"error": value}
        body = json.dumps({"name": name, "content": content}, ensure_ascii=False)
        blocks.append(f"<tool_response>\n{body}\n</tool_response>")
    return "\n".join(blocks)


def run_dialogue(
    assistant: Callable[[str | None], list[dict[str, Any]]],
    tools: list[dict[str, Any]],
    functions: ModuleType,
    max_turns: int = 8,
) -> dict[str, Any]:
    """Multi-turn execution of a tool-calling dialogue (the Hermes protocol).

    ``assistant(feedback)`` returns the calls of the next assistant turn; ``feedback`` is the
    ``<tool_response>`` text for the previous turn's calls (None on the first turn). Every call
    is executed as it arrives; results persist across turns, so positional references
    (``$var_N.result$``, N counted over the whole dialogue) also resolve. The dialogue ends at
    the first turn without calls or after ``max_turns`` turns with calls. As in NESTful's
    scoring, the prediction is the result of the last call made.

    Returns ok/value of the last call, turns with calls, total calls, failed calls, and
    whether the model ended the dialogue itself (``finished``).
    """
    results: dict[str, dict[str, Any]] = {}
    last: tuple[bool, Any] = (False, "no calls")
    feedback: str | None = None
    turns = n_calls = errors = 0
    finished = False
    for _ in range(max_turns):
        calls = assistant(feedback)
        if not calls:
            finished = True
            break
        turns += 1
        outcomes = []
        for call in calls:
            ok, value = run_call(call, n_calls, tools, functions, results)
            n_calls += 1
            errors += not ok
            last = (ok, value)
            outcomes.append((call["name"], ok, value))
        feedback = tool_responses(outcomes)
    return {
        "ok": last[0],
        "value": last[1],
        "turns": turns,
        "calls": n_calls,
        "errors": errors,
        "finished": finished,
    }


def answers_match(pred: Any, gold: Any) -> bool:
    """NESTful's answer comparison (src/scorer.py ``calculate_win_score``)."""
    if isinstance(pred, bool):
        return False
    if isinstance(gold, float) and isinstance(pred, float):
        text = repr(gold)
        if "e" in text or not math.isfinite(gold):
            return math.isclose(pred, gold, rel_tol=1e-9)
        return round(pred, len(text.split(".")[1])) == gold
    if isinstance(pred, tuple) and isinstance(gold, list):
        pred = list(pred)
    return pred == gold


def _canonical(calls: list[dict[str, Any]]) -> list[tuple]:
    position = {_label(c, i): i for i, c in enumerate(calls)}

    def value(v: Any) -> Any:
        if isinstance(v, str) and (m := _REF.match(v)):
            return ("ref", position.get(re.sub(r"^var_?", "var_", m.group(1)), -1), m.group(2))
        if isinstance(v, list):
            return tuple(value(x) for x in v)
        return v

    return [
        (c["name"], tuple(sorted((k, value(v)) for k, v in (c.get("arguments") or {}).items())))
        for c in calls
    ]


def exact_match(pred: list[dict[str, Any]], gold: list[dict[str, Any]]) -> bool:
    """Same calls in the same order, with references compared by call position."""
    try:
        return _canonical(pred) == _canonical(gold)
    except TypeError:  # unhashable argument values in a prediction
        return False
