"""Router-training pool: MathQA-test problems that are not NESTful and not close to any of them.

    python scripts/lab/make_mathqa_testpool.py     # results/mathqa/testpool_tasks.jsonl

Why: the router has 292 labelled clean MathQA-validation tasks. NESTful's math portion is a
subset of the MathQA test split (1,416 of 2,985 problems); ``make_mathqa.py`` already removed
from MathQA train every problem that matches *any* MathQA test problem (exact text, template
similarity >= 0.3, same computation), so the other test problems stand in the same relation to
the training data as NESTful does. They can label a router, provided none of them leaks NESTful:
a task is kept only if its normalised text is not a NESTful query and, against NESTful, its
template similarity is < ``NEAR_DUP_JACCARD`` and it has no same computation (>= 2 shared
distinctive numbers). Built with ``build_task`` (NESTful's construction, the train/val recipe).
Downloads: the pinned revisions of ``make_mathqa.py`` (explicit opt-in).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_mathqa as mm  # noqa: E402
import ouro_nestful as nf  # noqa: E402

from dht.toolcall.mathqa import (  # noqa: E402
    NEAR_DUP_JACCARD,
    NearDuplicateIndex,
    SameComputationIndex,
    build_task,
    normalize,
)
from dht.toolcall.nestful import basic_function_names, load_functions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="results/mathqa")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    specs_file = nf.VENDORED / "tool_specs.json"
    if hashlib.sha256(specs_file.read_bytes()).hexdigest() != mm.TOOL_SPECS_SHA256:
        raise ValueError(f"{specs_file} does not match its pinned SHA256")
    catalog = {t["name"]: t for t in json.loads(specs_file.read_text(encoding="utf-8"))}
    ffile = nf.checked(nf.VENDORED / "basic_functions.py", "basic_functions.py")
    functions = load_functions(ffile)
    rows, original, nestful = mm.load_sources(basic_function_names(ffile))
    nest_text = {normalize(e.query) for e in nestful}
    near = NearDuplicateIndex([e.query for e in nestful])
    same = SameComputationIndex([e.calls for e in nestful])

    rng = random.Random(f"{args.seed}-testpool")
    reasons: Counter = Counter()
    seen: set[str] = set()
    tasks = []
    for i, row in enumerate(rows["test"]):
        key = normalize(row["Problem"])
        if key in seen:
            reasons["duplicate"] += 1
            continue
        seen.add(key)
        task, why = build_task(row, catalog, functions, rng, original)
        if task is not None:
            if normalize(task["query"]) in nest_text or key in nest_text:
                task, why = None, "nestful"
            elif near.max_similarity(task["query"]) >= NEAR_DUP_JACCARD:
                task, why = None, "near_dup_nestful"
            elif same.matches(task["calls"]):
                task, why = None, "same_computation_nestful"
        reasons[why] += 1
        if task is not None:
            tasks.append({"id": f"mathqa-test-{i}", **task})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "testpool_tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    stats = {"test_rows": len(rows["test"]), "filter": dict(reasons), "tasks": len(tasks)}
    (out / "testpool_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
