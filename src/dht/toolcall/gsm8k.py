"""GSM8K → NESTful-format tool-calling tasks (a second multi-step benchmark).

Why: every multi-step result so far is on NESTful's math subset, which was built from MathQA (the
fine-tunes also train on MathQA train). GSM8K (Cobbe et al. 2021, MIT) is a different source of
grade-school word problems whose reference solutions mark every calculation inline, e.g.
``Natalia sold 48/2 = <<48/2=24>>24 clips``, and end with ``#### <answer>``. Converting those
annotations into calls over NESTful's own tools gives executable multi-step tasks in the same
format and grading as NESTful, so the same models and analyses apply unchanged.

Construction (per problem):

* Each annotation ``<<expr=result>>`` is parsed with Python's ``ast`` (numbers, ``+ - * /``,
  ``**``, parentheses, unary minus; anything else rejects the problem). Binary operations become
  calls to ``add`` / ``subtract`` / ``multiply`` / ``divide`` / ``power`` (``arg_0``, ``arg_1``) in
  post-order, labelled ``$var_k`` over the whole solution; a compound expression gives several
  calls (``2*3+4`` → multiply, then add on its result).
* A number in an expression that is not in the question but equals the result of an earlier
  annotation (relative tolerance 1e-6) becomes a reference ``$var_k.result$`` to the call that
  produced it (the latest such annotation); numbers that appear in the question, and constants
  (``2``, ``100``, ``60`` ...), stay literals. An annotation that is a bare number (``<<5=5>>``)
  adds no call. This mirrors how NESTful chains MathQA's formulas: intermediate results flow by
  reference, inputs come from the question.
* The task is kept only if it has at least ``MIN_CALLS`` calls, every call but the last feeds a
  later one (``connected``: 18% of converted solutions have a dead intermediate result, mostly
  where the solution did part of the arithmetic without an annotation and the next annotation
  starts from an unannotated number), and executing the calls
  reproduces the ``####`` answer (relative tolerance 1e-4, absolute 1e-6); the gold answer is the
  ``####`` value (not the executed float, which can carry rounding noise). The tool list is
  NESTful-style (``mathqa.tool_list``: gold tools, core tools, random other math tools).
"""

from __future__ import annotations

import ast
import math
import random
import re
from types import ModuleType
from typing import Any

from dht.toolcall.mathqa import MIN_CALLS, tool_list
from dht.toolcall.nestful import execute

ANNOTATION = re.compile(r"<<([^<>=]+)=([^<>]*)>>")
QUESTION_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
OPS = {
    ast.Add: "add",
    ast.Sub: "subtract",
    ast.Mult: "multiply",
    ast.Div: "divide",
    ast.Pow: "power",
}
REL_TOL_REF = 1e-6
REL_TOL_ANSWER = 1e-4


def _number(v: float) -> int | float:
    return int(v) if float(v).is_integer() else float(v)


def question_numbers(text: str) -> set[float]:
    """Numbers written in digits in the question (thousands separators removed)."""
    out = set()
    for m in QUESTION_NUMBER.findall(text):
        try:
            out.add(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def final_answer(solution: str) -> float:
    """The value after ``####`` (thousands separators removed)."""
    tail = solution.rsplit("####", 1)
    if len(tail) != 2:
        raise ValueError("no #### answer")
    return float(tail[1].strip().replace(",", ""))


def _close(a: float, b: float, rel: float) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-9)


def solution_calls(solution: str, question: str) -> list[dict[str, Any]]:
    """NESTful-format calls for every annotated calculation of a GSM8K solution, in order.

    Raises ``ValueError`` for an unsupported expression."""
    given = question_numbers(question)
    calls: list[dict[str, Any]] = []
    produced: list[tuple[float, Any]] = []  # (value, literal or reference) per annotation

    def leaf(v: float) -> Any:
        if any(_close(v, q, REL_TOL_REF) for q in given):
            return _number(v)
        for value, source in reversed(produced):
            if _close(v, value, REL_TOL_REF):
                return source
        return _number(v)

    def walk(node: ast.AST) -> tuple[float, Any]:
        """(value, argument: literal or $var reference) of an expression node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            v = float(node.value)
            return v, leaf(v)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v, arg = walk(node.operand)
            if isinstance(arg, str):
                raise ValueError("negated reference")
            return -v, _number(-v)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            (a, x), (b, y) = walk(node.left), walk(node.right)
            name = OPS[type(node.op)]
            value = {"add": a + b, "subtract": a - b, "multiply": a * b}.get(name)
            if name == "divide":
                if b == 0:
                    raise ValueError("division by zero")
                value = a / b
            if name == "power":
                value = a**b
            calls.append(
                {
                    "name": name,
                    "label": f"$var_{len(calls) + 1}",
                    "arguments": {"arg_0": x, "arg_1": y},
                }
            )
            return float(value), f"$var_{len(calls)}.result$"
        raise ValueError(f"unsupported expression node {type(node).__name__}")

    for expr, _result in ANNOTATION.findall(solution):
        text = expr.replace(",", "").replace("x", "*").replace("×", "*").strip()
        try:
            tree = ast.parse(text, mode="eval").body
        except SyntaxError as exc:
            raise ValueError(f"unparseable annotation {expr!r}") from exc
        value, source = walk(tree)
        produced.append((value, source))
    return calls


def connected(calls: list[dict[str, Any]]) -> bool:
    """Every call but the last feeds a later call (no dead intermediate results)."""
    used = set(re.findall(r"\$var_(\d+)\.", repr([c["arguments"] for c in calls])))
    return all(str(k) in used for k in range(1, len(calls)))


def build_task(
    row: dict[str, Any], catalog: dict[str, dict], functions: ModuleType, rng: random.Random
) -> tuple[dict[str, Any] | None, str]:
    """One GSM8K row (``question``, ``answer``) → NESTful-format task, or (None, reason) with
    reason in parse / single / disconnected / op / exec / mismatch / no_answer."""
    try:
        target = final_answer(row["answer"])
    except ValueError:
        return None, "no_answer"
    try:
        calls = solution_calls(row["answer"], row["question"])
    except (ValueError, OverflowError, ZeroDivisionError):
        return None, "parse"
    if len(calls) < MIN_CALLS:
        return None, "single"
    if not connected(calls):
        return None, "disconnected"
    gold = {c["name"] for c in calls}
    if not gold <= set(catalog):
        return None, "op"
    tools = tool_list(gold, catalog, rng)
    ok, value = execute(calls, tools, functions)
    if not ok or isinstance(value, (tuple, list, bool)) or not math.isfinite(float(value)):
        return None, "exec"
    if not math.isclose(float(value), target, rel_tol=REL_TOL_ANSWER, abs_tol=1e-6):
        return None, "mismatch"
    return {
        "query": row["question"].strip(),
        "tools": tools,
        "calls": calls,
        # the #### answer, checked above to match the executed value: execution can carry float
        # noise (9/14*84 = 34.00000000000001) that answers_match would demand exactly
        "answer": float(target),
    }, "ok"
