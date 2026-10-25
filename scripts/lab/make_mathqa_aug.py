"""Number-resampled MathQA training tasks and SFT dialogues (augmentation for the v4 pilot).

    python scripts/lab/make_mathqa_aug.py --sft-n 6000     # needs results/mathqa/ first

Each MathQA-train task of ``make_mathqa.py`` (``results/mathqa/train_tasks.jsonl``) gets up to
``--variants`` copies with new numbers (``dht.toolcall.mathqa.resample_numbers``: the numbers
written in the problem change in the text and the gold calls together, the calls are re-executed
for the new answer). Only the train split is used.

Leakage guard: the source tasks already passed the three filters of ``make_mathqa.py`` against
MathQA test and NESTful. Resampling keeps the number-masked template and the call skeleton, so
the exact-text and near-duplicate filters are re-run only as a check, but the same-computation
filter uses numbers and can newly match; it is re-run against MathQA test + NESTful and also
against MathQA validation (the pilot's evaluation set), so augmentation adds no new overlap
with either. Downloads are the pinned revisions of ``make_mathqa.py`` (explicit opt-in).

Outputs (``--out``, default ``results/mathqa_aug``): ``train_aug_tasks.jsonl``,
``train_aug_sft.json`` (``--sft-n`` Hermes rows, as ``train_sft.json``) and ``stats.json``.
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
from ouro_sft_eval import SYSTEM  # noqa: E402

from dht.toolcall.mathqa import (  # noqa: E402
    NEAR_DUP_JACCARD,
    SameComputationIndex,
    levels,
    normalize,
    render_dialogue,
    resample_numbers,
)
from dht.toolcall.nestful import basic_function_names, load_functions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="results/mathqa")
    parser.add_argument("--out", default="results/mathqa_aug")
    parser.add_argument("--variants", type=int, default=1, help="resampled copies per task")
    parser.add_argument("--sft-n", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ffile = nf.checked(nf.VENDORED / "basic_functions.py", "basic_functions.py")
    functions = load_functions(ffile)
    rows, _, nestful = mm.load_sources(basic_function_names(ffile))
    banned, near, same_test = mm.leakage_guards(rows["test"], nestful)
    src = Path(args.source)
    read = lambda p: [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]  # noqa: E731
    train = read(src / "train_tasks.jsonl")
    val = read(src / "validation_tasks.jsonl")
    same_val = SameComputationIndex([t["calls"] for t in val])
    val_text = {normalize(t["query"]) for t in val}

    rng = random.Random(args.seed)
    reasons: Counter = Counter()
    seen = {normalize(t["query"]) for t in train}
    tasks = []
    for t in train:
        for v in range(args.variants):
            new = resample_numbers(t, functions, rng)
            if new is None:
                reasons["not_resampled"] += 1
                continue
            key = normalize(new["query"])
            if key in banned or key in val_text:
                why = "leak"
            elif key in seen:
                why = "duplicate"
            elif near.max_similarity(new["query"]) >= NEAR_DUP_JACCARD:
                why = "near_dup"
            elif same_test.matches(new["calls"]):
                why = "same_computation_test"
            elif same_val.matches(new["calls"]):
                why = "same_computation_val"
            else:
                why = "ok"
            reasons[why] += 1
            if why == "ok":
                seen.add(key)
                tasks.append({**new, "id": f"{t['id']}-aug{v}"})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train_aug_tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    pool = list(tasks)
    random.Random(args.seed).shuffle(pool)
    sft, failed = [], 0
    for t in pool:
        if len(sft) >= args.sft_n:
            break
        try:
            conv = render_dialogue(t["query"], t["calls"], t["tools"], functions, SYSTEM)
        except ValueError:
            failed += 1
            continue
        sft.append(
            {
                "id": t["id"],
                "conversations": conv,
                "n_calls": len(t["calls"]),
                "turns": len(set(levels(t["calls"]))),
            }
        )
    (out / "train_aug_sft.json").write_text(json.dumps(sft, ensure_ascii=False))
    stats = {
        "source_tasks": len(train),
        "variants": args.variants,
        "filter": dict(reasons),
        "tasks": len(tasks),
        "sft": len(sft),
        "render_failed": failed,
        "ids_sha256": hashlib.sha256("\n".join(r["id"] for r in sft).encode()).hexdigest(),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
