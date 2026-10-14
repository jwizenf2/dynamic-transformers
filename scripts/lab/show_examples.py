"""Print real examples for the progress report (CPU only; reads cached data and saved results).

    python scripts/lab/show_examples.py

1. Three xLAM validation records (query, tools, reference answer) from the same seeded split the
   headroom probe uses, so the report shows the actual data the models were graded on.
2. For the 4B and 9B headroom runs: for a few validation ids, the model's output under the full
   model, a whole-group-2 skip and a DeltaNet-only group-2 skip, to illustrate the failure modes
   (texts are stored for failed rows only; correct rows print as "correct").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from headroom import validation_examples  # noqa: E402

RUNS = {
    "Qwen3.5-4B": "dynamic-transformers-headroom-4b",
    "Qwen3.5-9B": "dynamic-transformers-headroom-9b",
}
ARCHS = ["full", "group2", "deltanet2", "attn2"]


def main() -> None:
    examples = validation_examples(200, 2000, 0)
    for e in examples[:3]:
        print(
            json.dumps(
                {
                    "id": e.id,
                    "query": e.messages[0]["content"],
                    "functions": e.functions,
                    "ground_truth": e.ground_truth,
                },
                indent=1,
            ),
            flush=True,
        )

    for model, project in RUNS.items():
        path = Path.home() / "AI/jobs" / project / "results/headroom" / model / "examples.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        by = {(r["arch"], r["id"]): r for r in rows}
        shown = 0
        for e in examples:
            if shown == 4:
                break
            full = by.get(("full", e.id))
            dn = by.get(("deltanet2", e.id))
            if not (full and full["correct"] and dn and not dn["correct"]):
                continue
            shown += 1
            out = {"model": model, "id": e.id, "query": e.messages[0]["content"]}
            for a in ARCHS:
                r = by.get((a, e.id))
                if r:
                    out[a] = (
                        "correct"
                        if r["correct"]
                        else {"reason": r["reason"], "text": (r["text"] or "")[:600]}
                    )
            print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
