"""Convert BFCL and xLAM function descriptions into the OpenAI ``tools`` format Qwen expects.

BFCL uses Python-flavoured type names (``dict``, ``float``, ``tuple``, ``any``) inside an otherwise
JSON-schema-like object. Qwen's chat template renders whatever schema it is given, so we map those
names to JSON-schema names for the prompt, while the checker keeps the ORIGINAL BFCL type names
(BFCL's own grading semantics depend on them, e.g. ``float`` accepts integers).

xLAM stores parameters as a flat ``{name: {"type": "str, optional", "description", "default"}}``
mapping. A parameter is treated as required unless its type string contains ``optional`` or it
declares a default. This is our interpretation of an undocumented format.
"""

from __future__ import annotations

import copy
from typing import Any

BFCL_TO_JSON_SCHEMA = {
    "dict": "object",
    "float": "number",
    "tuple": "array",
    "integer": "integer",
    "string": "string",
    "boolean": "boolean",
    "array": "array",
}

XLAM_TO_BFCL = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "float",
    "number": "float",
    "bool": "boolean",
    "boolean": "boolean",
    "dict": "dict",
    "object": "dict",
    "list": "array",
    "array": "array",
    "tuple": "tuple",
}


def _json_schema(node: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(node)
    bfcl_type = out.pop("type", None)
    if bfcl_type in BFCL_TO_JSON_SCHEMA:
        out["type"] = BFCL_TO_JSON_SCHEMA[bfcl_type]
    # ``any`` (or a missing type) is rendered without a type constraint.
    if "properties" in out:
        out["properties"] = {k: _json_schema(v) for k, v in out["properties"].items()}
    if isinstance(out.get("items"), dict):
        out["items"] = _json_schema(out["items"])
    return out


def to_openai_tool(function: dict[str, Any]) -> dict[str, Any]:
    """BFCL-style function description -> ``{"type": "function", "function": {...}}``."""
    return {
        "type": "function",
        "function": {
            "name": function["name"],
            "description": function.get("description", ""),
            "parameters": _json_schema(function.get("parameters", {"type": "dict"})),
        },
    }


def _xlam_type(raw: str) -> str:
    """``"List[int], optional"`` -> ``"array"``; unknown names become ``any``."""
    base = raw.split(",")[0].strip()
    head = base.split("[")[0].strip().lower()
    return XLAM_TO_BFCL.get(head, "any")


def xlam_to_bfcl_function(tool: dict[str, Any]) -> dict[str, Any]:
    """xLAM tool -> BFCL-style function description (so one checker serves both datasets)."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, spec in (tool.get("parameters") or {}).items():
        raw_type = str(spec.get("type", ""))
        prop: dict[str, Any] = {"type": _xlam_type(raw_type)}
        if spec.get("description"):
            prop["description"] = spec["description"]
        if "default" in spec:
            prop["default"] = spec["default"]
        properties[name] = prop
        if "optional" not in raw_type.lower() and "default" not in spec:
            required.append(name)
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": {"type": "dict", "properties": properties, "required": required},
    }
