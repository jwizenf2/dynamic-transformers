"""NESTful requests with the solution plan given in the request (formulation ablation).

    python scripts/lab/make_nestful_plans.py      # results/nestful_plans/{full,ops}_tasks.jsonl

Why: the depth anatomy (``depth_anatomy.py``) found that 4 loops fix mostly formulation errors
at 2 loops (which quantities to combine, which operation), not format, copying or arithmetic.
If so, telling the model the plan should remove most of the 2→4 gap. Two conditions on the same
requests (NESTful math subset, depth-balanced order, [0, 720), ids unchanged), each the original
query followed by a blank line and:

* ``full``: the gold chain as numbered steps with function and arguments, earlier results named
  by step ("1. divide(100, 2)", "3. subtract(result of step 1, result of step 2)"); the model
  only has to issue the calls in order, copy results into later calls and stop;
* ``ops``: only the function names in order ("Use these functions in this order: divide,
  multiply, subtract."), so choosing the operation is given but choosing the quantities is not.

The plan comes from the gold chain, which executes to the gold answer by construction. Rows are
NESTful-format tasks (``ouro_nestful_multiturn.load_tasks``) for ``router_smoke.py --tasks``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ouro_nestful_multiturn import requests  # noqa: E402

REF = re.compile(r"^\$var_?(\d+)\.\w+\$$")


def step_number(labels: dict[str, int], value) -> str:
    """'result of step k' for a reference to an earlier call, else the literal."""
    if isinstance(value, str):
        m = REF.match(value)
        if m and m.group(1) in labels:
            return f"result of step {labels[m.group(1)]}"
        return value
    return json.dumps(value)


def full_plan(calls: list[dict]) -> str:
    labels: dict[str, int] = {}
    lines = []
    for i, c in enumerate(calls, 1):
        args = ", ".join(step_number(labels, v) for v in (c.get("arguments") or {}).values())
        lines.append(f"{i}. {c['name']}({args})")
        digits = re.findall(r"\d+", c.get("label") or "")
        labels[digits[-1] if digits else str(i)] = i
    return "Solve it with these steps:\n" + "\n".join(lines)


def ops_plan(calls: list[dict]) -> str:
    return "Use these functions in this order: " + ", ".join(c["name"] for c in calls) + "."


def main() -> None:
    examples, _ = requests(720)
    out = Path("results/nestful_plans")
    out.mkdir(parents=True, exist_ok=True)
    for name, plan in (("full", full_plan), ("ops", ops_plan)):
        rows = [
            {
                "id": e.id,
                "query": f"{e.query}\n\n{plan(e.calls)}",
                "tools": e.tools,
                "calls": e.calls,
                "answer": e.answer,
            }
            for e in examples
        ]
        (out / f"{name}_tasks.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
    print(json.dumps({"n": len(examples), "out": str(out)}))


if __name__ == "__main__":
    main()
