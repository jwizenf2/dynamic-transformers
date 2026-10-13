"""BFCL-style abstract-syntax-tree (AST) checking of parsed tool calls, reimplemented independently.

Ground truth follows BFCL's ``possible_answer`` format: a list of calls, each
``{function_name: {param: [allowed values...]}}``. ``""`` in an allowed list means "may be
omitted". The rules below follow BFCL v3's documented single-turn AST checker; exact parity with
``bfcl-eval`` is NOT verified (its package was not imported, see the plan). Known choices:

* Strings are compared after BFCL's normalisation: lower-case, remove spaces and ``,./-_*^``,
  and treat single and double quotes alike.
* ``float`` parameters accept integers; ``tuple`` accepts lists; ``any`` skips the type check.
* Lists compare element-wise (strings normalised); dicts compare per key against nested allowed
  lists, and keys missing from the model's dict must allow ``""``.
* Parallel categories need a one-to-one match between calls and ground truth in any order.
  BFCL matches greedily in ground-truth order; we search for any perfect matching, which can
  only accept more. Recorded as a deliberate, documented difference.
* Irrelevance: correct iff the model makes no parsable call.

Every check returns ``(ok, reason)`` so failures can be aggregated by cause.
"""

from __future__ import annotations

import re
from typing import Any

from dht.toolcall.parser import ToolCall

_STRIP = re.compile(r"[ ,./\-_*^]")
PYTHON_TYPES = {
    "string": (str,),
    "integer": (int,),
    "float": (float, int),
    "boolean": (bool,),
    "array": (list,),
    "tuple": (list, tuple),
    "dict": (dict,),
}


def normalize_string(value: str) -> str:
    return _STRIP.sub("", value).lower().replace("'", '"')


def _type_ok(value: Any, bfcl_type: str | None) -> bool:
    if bfcl_type in (None, "any") or bfcl_type not in PYTHON_TYPES:
        return True
    if isinstance(value, bool) and bfcl_type in {"integer", "float"}:
        return False  # bool is a subclass of int in Python; BFCL treats them as distinct.
    return isinstance(value, PYTHON_TYPES[bfcl_type])


def _value_matches(value: Any, allowed: Any) -> bool:
    """Compare one model value with one allowed ground-truth value."""
    if isinstance(allowed, str) and isinstance(value, str):
        return normalize_string(value) == normalize_string(allowed)
    if isinstance(allowed, dict) and isinstance(value, dict):
        return _dict_matches(value, allowed)
    if isinstance(allowed, list | tuple) and isinstance(value, list | tuple):
        return len(value) == len(allowed) and all(
            _value_matches(v, a) for v, a in zip(value, allowed, strict=True)
        )
    if isinstance(value, bool) != isinstance(allowed, bool):
        return False
    return value == allowed


def _dict_matches(value: dict, allowed: dict) -> bool:
    """Nested dict whose ground-truth values are lists of allowed alternatives."""
    for key, item in value.items():
        if key not in allowed:
            return False
        options = allowed[key] if isinstance(allowed[key], list) else [allowed[key]]
        if not any(_value_matches(item, option) for option in options):
            return False
    return all("" in options for key, options in allowed.items() if key not in value)


def _any_match(value: Any, options: list[Any]) -> bool:
    return any(_value_matches(value, option) for option in options if option != "")


def check_call(
    call: ToolCall, function: dict[str, Any], expected: dict[str, dict[str, list[Any]]]
) -> tuple[bool, str]:
    """One parsed call against one ground-truth call ``{name: {param: [allowed]}}``."""
    ((expected_name, expected_args),) = expected.items()
    if call.name != expected_name:
        return False, "wrong_function"
    params = function.get("parameters") or {}
    properties = params.get("properties") or {}
    for name in params.get("required") or []:
        if name not in call.arguments:
            return False, "missing_required"
    for name, value in call.arguments.items():
        if name not in properties or name not in expected_args:
            return False, "unexpected_param"
        if not _type_ok(value, properties[name].get("type")):
            return False, "wrong_type"
        if not _any_match(value, expected_args[name]):
            return False, "wrong_value"
    for name, options in expected_args.items():
        if name not in call.arguments and "" not in options:
            return False, "missing_optional_value"
    return True, "ok"


def _perfect_matching(calls: list[ToolCall], ground_truth: list[dict], fns: dict) -> bool:
    """Backtracking search for a one-to-one assignment of calls to ground-truth calls."""

    def assign(gt_index: int, used: frozenset[int]) -> bool:
        if gt_index == len(ground_truth):
            return True
        (name,) = ground_truth[gt_index].keys()
        for i, call in enumerate(calls):
            if i in used or name not in fns:
                continue
            if check_call(call, fns[name], ground_truth[gt_index])[0] and assign(
                gt_index + 1, used | {i}
            ):
                return True
        return False

    return assign(0, frozenset())


def check_example(
    calls: list[ToolCall],
    functions: list[dict[str, Any]],
    ground_truth: list[dict[str, dict[str, list[Any]]]] | None,
) -> tuple[bool, str]:
    """Grade all calls for one request. ``ground_truth=None`` (or []) means irrelevance."""
    if not ground_truth:
        return (not calls), ("ok" if not calls else "called_irrelevant")
    if not calls:
        return False, "no_call"
    if len(calls) != len(ground_truth):
        return False, "wrong_call_count"
    fns = {f["name"]: f for f in functions}
    if any(call.name not in fns for call in calls):
        return False, "unknown_function"
    if len(ground_truth) == 1:
        return check_call(calls[0], fns[calls[0].name], ground_truth[0])
    return (True, "ok") if _perfect_matching(calls, ground_truth, fns) else (False, "no_matching")
