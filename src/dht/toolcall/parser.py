"""Parse Qwen3.5/3.8 tool calls (the ``qwen3_coder`` XML format) into typed Python values.

Observed on the lab (Qwen3.5-0.8B, enable_thinking=False)::

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Paris
    </parameter>
    </function>
    </tool_call>

Parameter values are raw text, so types come from the function schema (as serving parsers such as
vLLM's ``qwen3_coder`` do): integers/floats/booleans are converted, arrays/objects are parsed as
JSON (falling back to a Python literal), and strings stay strings. A value that fails conversion
is kept as text, which the checker will then reject on type, i.e. a malformed call is wrong,
never silently repaired. Text outside ``<tool_call>`` blocks is ignored.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)
FUNCTION_RE = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.DOTALL)
PARAMETER_RE = re.compile(
    r"<parameter=([^>\n]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.DOTALL
)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


def _strip_value(text: str) -> str:
    """Remove exactly the newline padding the template adds around a value."""
    if text.startswith("\n"):
        text = text[1:]
    if text.endswith("\n"):
        text = text[:-1]
    return text


def _literal(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def convert_value(text: str, bfcl_type: str | None) -> Any:
    """Convert one raw parameter string using its BFCL type name; unknown types try JSON."""
    stripped = text.strip()
    try:
        if bfcl_type == "string":
            return text
        if stripped.lower() in {"null", "none"}:
            return None
        if bfcl_type == "integer":
            return int(stripped)
        if bfcl_type == "float":
            return float(stripped)
        if bfcl_type == "boolean":
            if stripped.lower() in {"true", "false"}:
                return stripped.lower() == "true"
            return text
        if bfcl_type in {"array", "tuple", "dict"}:
            return _literal(stripped)
        return _literal(stripped)
    except (ValueError, SyntaxError, json.JSONDecodeError, TypeError, MemoryError, RecursionError):
        return text


def parameter_types(functions: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """``{function_name: {param: bfcl_type}}`` from BFCL-style descriptions."""
    out: dict[str, dict[str, str]] = {}
    for function in functions:
        props = (function.get("parameters") or {}).get("properties") or {}
        out[function["name"]] = {k: v.get("type", "any") for k, v in props.items()}
    return out


def parse_tool_calls(text: str, functions: list[dict[str, Any]]) -> list[ToolCall]:
    """All calls in generation order. Returns [] when the model made no call."""
    types = parameter_types(functions)
    calls: list[ToolCall] = []
    for block in TOOL_CALL_RE.findall(text):
        for name, body in FUNCTION_RE.findall(block):
            name = name.strip()
            fn_types = types.get(name, {})
            arguments = {
                key.strip(): convert_value(_strip_value(raw), fn_types.get(key.strip()))
                for key, raw in PARAMETER_RE.findall(body)
            }
            calls.append(ToolCall(name=name, arguments=arguments))
    return calls
