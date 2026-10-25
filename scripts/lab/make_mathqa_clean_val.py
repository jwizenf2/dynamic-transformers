"""MathQA-validation tasks that relate to MathQA-train the way NESTful test does ("clean").

    python scripts/lab/make_mathqa_clean_val.py      # results/mathqa/validation_clean_tasks.jsonl

``make_mathqa.py`` filters train and validation against MathQA test + NESTful only, not against
each other. Measured 2026-10-25 (review of the router/v4 gates): 94 of the first 120 validation
tasks have a train task at number-masked template similarity >= 0.3, i.e. they are mostly the
same problems as training tasks with other numbers (AQuA-RAT's rewrites), which the 0.3 filter
removed from the NESTful relation. Accuracy on them measures memorised templates, and
number-resampled training data (``make_mathqa_aug.py``) would help there without transfer to
NESTful.

This keeps a validation task only if, against all train tasks, its normalised text is new, its
template similarity is below ``NEAR_DUP_JACCARD`` and it has no same computation with >= 2
shared distinctive numbers: the three filters NESTful's relation to train already satisfies.
Order and content are otherwise unchanged; no downloads.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dht.toolcall.mathqa import (
    NEAR_DUP_JACCARD,
    NearDuplicateIndex,
    SameComputationIndex,
    normalize,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default="results/mathqa")
    args = parser.parse_args()
    d = Path(args.dir)

    def read(p: Path) -> list[dict]:
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]

    train, val = read(d / "train_tasks.jsonl"), read(d / "validation_tasks.jsonl")
    exact = {normalize(t["query"]) for t in train}
    near = NearDuplicateIndex([t["query"] for t in train])
    same = SameComputationIndex([t["calls"] for t in train])
    reasons: Counter = Counter()
    keep = []
    for t in val:
        if normalize(t["query"]) in exact:
            why = "exact"
        elif near.max_similarity(t["query"]) >= NEAR_DUP_JACCARD:
            why = "near_dup"
        elif same.matches(t["calls"]):
            why = "same_computation"
        else:
            why = "ok"
        reasons[why] += 1
        if why == "ok":
            keep.append(t)
    with open(d / "validation_clean_tasks.jsonl", "w") as f:
        for t in keep:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    stats = {"validation": len(val), "clean": len(keep), "filter": dict(reasons)}
    (d / "validation_clean_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
