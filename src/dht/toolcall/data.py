"""Load BFCL (final test) and xLAM (search/train/validation) into one ``Example`` format.

Both datasets are read from the local Hugging Face cache; downloading is an explicit opt-in
(``scripts/lab/setup_check.py``). Revisions are pinned by the caller.

Splits (plan, "Datasets"): xLAM is shuffled with a fixed seed and cut into disjoint train and
validation ids; BFCL categories are only ever returned whole and are reserved for the final test.
xLAM ground truth is converted to BFCL's allowed-values format: each argument the reference
answer gives allows exactly that value (typed from the schema, nested dicts wrapped as BFCL
does); an optional parameter the answer omits may be omitted or set to its declared default.
This is our interpretation (xLAM ships single reference calls).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dht.toolcall.parser import convert_value
from dht.toolcall.schema import xlam_to_bfcl_function

BFCL_SINGLE_TURN = ("simple", "multiple", "parallel", "parallel_multiple", "irrelevance")


@dataclass(frozen=True)
class Example:
    id: str
    messages: list[dict[str, str]]
    functions: list[dict[str, Any]]  # BFCL-style descriptions (original type names)
    ground_truth: list[dict[str, dict[str, list[Any]]]]  # [] means "no call is correct"
    category: str


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_bfcl(root: Path, category: str) -> list[Example]:
    """One BFCL v3 single-turn category from a dataset snapshot directory."""
    questions = _jsonl(root / f"BFCL_v3_{category}.json")
    answer_path = root / "possible_answer" / f"BFCL_v3_{category}.json"
    answers = (
        {row["id"]: row["ground_truth"] for row in _jsonl(answer_path)}
        if (answer_path.exists())
        else {}
    )
    if category != "irrelevance" and not answers:
        raise FileNotFoundError(f"no possible answers for BFCL category {category!r}")
    examples = []
    for row in questions:
        turns = row["question"]
        if len(turns) != 1:
            raise ValueError(f"{row['id']}: expected a single-turn question")
        examples.append(
            Example(
                id=row["id"],
                messages=turns[0],
                functions=row["function"],
                ground_truth=answers.get(row["id"], []),
                category=category,
            )
        )
    return examples


def _typed(value: Any, bfcl_type: str | None) -> Any:
    """Give a reference value the type the parser would give the same text from the model.

    xLAM sometimes stores numbers/defaults as strings (``"default": "10"`` for an int). The
    model's ``<parameter=limit>10`` is parsed to ``10`` from the schema, so the reference must be
    converted the same way or a correct call would be graded wrong.
    """
    if isinstance(value, str) and bfcl_type not in (None, "string", "any"):
        return convert_value(value, bfcl_type)
    return value


def _allowed_format(value: Any) -> Any:
    """Raw nested value -> BFCL allowed-values format: dict values become one-option lists.

    BFCL writes nested dicts as ``{key: [alternatives...]}`` while list values stay raw lists
    (compared element-wise). xLAM gives raw values, so dicts (also inside lists) are wrapped.
    """
    if isinstance(value, dict):
        return {k: [_allowed_format(v)] for k, v in value.items()}
    if isinstance(value, list):
        return [_allowed_format(v) for v in value]
    return value


def _xlam_ground_truth(answers: list[dict], functions: dict[str, dict]) -> list[dict]:
    truth = []
    for answer in answers:
        name, args = answer["name"], answer.get("arguments") or {}
        params = functions.get(name, {}).get("parameters") or {}
        props = params.get("properties") or {}
        required = set(params.get("required") or [])
        allowed: dict[str, list[Any]] = {
            key: [_allowed_format(_typed(value, props.get(key, {}).get("type")))]
            for key, value in args.items()
        }
        for key, prop in props.items():
            if key not in allowed and key not in required:
                default = [_typed(prop["default"], prop.get("type"))] if "default" in prop else []
                allowed[key] = ["", *default]
        truth.append({name: allowed})
    return truth


def load_xlam(path: Path) -> list[Example]:
    """All 60k xLAM records. ``tools`` and ``answers`` are JSON strings in the release."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for row in rows:
        tools = row["tools"] if isinstance(row["tools"], list) else json.loads(row["tools"])
        answers = row["answers"] if isinstance(row["answers"], list) else json.loads(row["answers"])
        functions = [xlam_to_bfcl_function(tool) for tool in tools]
        by_name = {f["name"]: f for f in functions}
        examples.append(
            Example(
                id=f"xlam_{row['id']}",
                messages=[{"role": "user", "content": row["query"]}],
                functions=functions,
                ground_truth=_xlam_ground_truth(answers, by_name),
                category="xlam",
            )
        )
    return examples


def split_ids(ids: list[str], validation_size: int, seed: int) -> tuple[list[str], list[str]]:
    """Deterministic disjoint (train, validation) id lists; order is the shuffled order."""
    if not 0 < validation_size < len(ids):
        raise ValueError("validation size must be between 0 and the number of ids")
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be unique")
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    return shuffled[validation_size:], shuffled[:validation_size]
