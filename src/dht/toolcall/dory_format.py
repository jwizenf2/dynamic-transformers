"""Tool-call format of NVIDIA Dory 1.0 (the ``qwen3_coder`` XML style of its chat template).

The chat template (``chat_template.jinja`` of ``nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16``)
lists tools from OpenAI-style function schemas and asks for calls as::

    <tool_call>
    <function=NAME>
    <parameter=ARG>
    VALUE
    </parameter>
    ...
    </function>
    </tool_call>

Generation starts inside ``<think>`` when thinking is enabled, so the visible answer is the text
after the first ``</think>``. Parameter values are text: we decode them as JSON when they parse
(numbers, lists) and keep the string otherwise, as vLLM's ``qwen3_coder`` parser does for
non-string schema types.
"""

from __future__ import annotations

import json
import re
from typing import Any

_CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
_PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
_TYPES = {"int or float": "number", "int": "integer", "float": "number", "number": "number"}


def openai_tool(spec: dict[str, Any]) -> dict[str, Any]:
    """NESTful tool spec (``parameters: {arg_0: {description, type}}``) → OpenAI function schema."""
    props = {}
    for name, p in (spec.get("parameters") or {}).items():
        t = str(p.get("type", "")).lower()
        if t.startswith("list") or t == "array":
            schema: dict[str, Any] = {"type": "array", "items": {"type": "number"}}
        else:
            schema = {"type": _TYPES.get(t, "string")}
        if p.get("description"):
            schema["description"] = p["description"]
        props[name] = schema
    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": spec.get("description", ""),
            "parameters": {"type": "object", "properties": props, "required": list(props)},
        },
    }


def split_thinking(text: str) -> tuple[str, str]:
    """(reasoning, answer). Without a closing ``</think>`` everything is reasoning."""
    if "</think>" not in text:
        return text, ""
    reasoning, answer = text.split("</think>", 1)
    return reasoning.replace("<think>", "").strip(), answer.strip()


def _value(raw: str) -> Any:
    s = raw.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def parse_calls(answer: str) -> list[dict[str, Any]]:
    """``[{"name", "arguments"}]`` from the visible answer text, in order."""
    calls = []
    for name, body in _CALL.findall(answer):
        args = {k: _value(v) for k, v in _PARAM.findall(body)}
        calls.append({"name": name, "arguments": args})
    return calls
