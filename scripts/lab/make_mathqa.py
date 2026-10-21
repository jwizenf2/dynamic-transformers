"""Build NESTful-format tasks and multi-turn SFT dialogues from MathQA train/validation.

    python scripts/lab/make_mathqa.py                     # results/mathqa/{train,validation}...

Downloads (explicit opt-in, pinned revisions): MathQA (``allenai/math_qa``, parquet conversion),
AQuA-RAT (``deepmind/aqua_rat``, original question text) and NESTful (``ibm-research/nestful``,
only to exclude its problems). Construction and measured fidelity: ``dht.toolcall.mathqa``.

Leakage guard: a train/validation problem is dropped when its normalised text equals any MathQA
test problem or any NESTful query (52 NESTful queries reappear in MathQA train, 6 in validation).

Outputs (``--out``, default ``results/mathqa``):
* ``<split>_tasks.jsonl``: ``{"id", "query", "tools", "calls", "answer", "category"}``, the
  layout of ``NestfulExample`` (read by ``ouro_nestful_multiturn.py --tasks``);
* ``train_sft.json``: Hermes rows ``{"id", "conversations", "n_calls", "turns"}`` for
  ``ouro_lora_sft.py --extra-data`` (``--sft-n`` rows, train split only);
* ``stats.json``: filter counts and call/depth histograms next to NESTful's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ouro_nestful as nf  # noqa: E402
from ouro_sft_eval import SYSTEM  # noqa: E402

from dht.toolcall.mathqa import build_task, levels, normalize, render_dialogue  # noqa: E402
from dht.toolcall.nestful import (  # noqa: E402
    basic_function_names,
    dependency_depth,
    load_functions,
    load_nestful_parquet,
)

MATHQA_HF = "allenai/math_qa"
MATHQA_REVISION = "fafb9f7ee5b9ec4da9499f9c4177a4c91389f2d6"  # refs/convert/parquet
AQUA_HF = "deepmind/aqua_rat"
AQUA_REVISION = "33301c6a050c96af81f63cad5562cb5363e88971"
TOOL_SPECS_SHA256 = "babf02c6e5b399e3aaf60c76bd7a73820eb4dcebdae3c56680210b79718b1eb0"


def _hist(values) -> dict:
    return dict(sorted(Counter(values).items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sft-n", type=int, default=8000, help="SFT dialogues (train split)")
    parser.add_argument("--args", choices=["literal", "ref"], default="literal")
    parser.add_argument("--out", default="results/mathqa")
    args = parser.parse_args()

    specs_file = nf.VENDORED / "tool_specs.json"
    if hashlib.sha256(specs_file.read_bytes()).hexdigest() != TOOL_SPECS_SHA256:
        raise ValueError(f"{specs_file} does not match its pinned SHA256")
    catalog = {t["name"]: t for t in json.loads(specs_file.read_text(encoding="utf-8"))}
    ffile = nf.checked(nf.VENDORED / "basic_functions.py", "basic_functions.py")
    functions = load_functions(ffile)
    basic = basic_function_names(ffile)

    mq = Path(snapshot_download(MATHQA_HF, repo_type="dataset", revision=MATHQA_REVISION))
    aq = Path(
        snapshot_download(
            AQUA_HF, repo_type="dataset", revision=AQUA_REVISION, allow_patterns=["raw/*"]
        )
    )
    ns = Path(
        snapshot_download(nf.NESTFUL_HF, repo_type="dataset", revision=nf.NESTFUL_HF_REVISION)
    )
    nestful = [
        e
        for e in load_nestful_parquet(nf.checked(ns / nf.NESTFUL_PARQUET, nf.NESTFUL_PARQUET))
        if all(c["name"] in basic for c in e.calls)
    ]

    original: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for r in pq.read_table(aq / f"raw/{split}-00000-of-00001.parquet").to_pylist():
            original.setdefault(normalize(r["question"]), r["question"])
    rows = {
        s: pq.read_table(mq / f"default/{s}/0000.parquet").to_pylist()
        for s in ("train", "validation", "test")
    }
    banned = {normalize(r["Problem"]) for r in rows["test"]} | {normalize(e.query) for e in nestful}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = {
        "nestful_math": {
            "n": len(nestful),
            "calls": _hist(len(e.calls) for e in nestful),
            "depth": _hist(dependency_depth(e.calls) for e in nestful),
        }
    }
    tasks_by_split = {}
    for split in ("train", "validation"):
        rng = random.Random(f"{args.seed}-{split}")
        reasons: Counter = Counter()
        seen: set[str] = set()
        tasks = []
        for i, row in enumerate(rows[split]):
            key = normalize(row["Problem"])
            if key in banned:
                reasons["leak"] += 1
                continue
            if key in seen:
                reasons["duplicate"] += 1
                continue
            seen.add(key)
            task, why = build_task(row, catalog, functions, rng, original)
            reasons[why] += 1
            if task is not None:
                tasks.append({"id": f"mathqa-{split}-{i}", **task})
        with open(out / f"{split}_tasks.jsonl", "w") as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        tasks_by_split[split] = tasks
        stats[split] = {
            "rows": len(rows[split]),
            "filter": dict(reasons),
            "tasks": len(tasks),
            "calls": _hist(len(t["calls"]) for t in tasks),
            "depth": _hist(dependency_depth(t["calls"]) for t in tasks),
            "original_text": sum(normalize(t["query"]) in original for t in tasks),
        }
        print(json.dumps({split: stats[split]["filter"], "tasks": len(tasks)}), flush=True)

    rng = random.Random(args.seed)
    pool = list(tasks_by_split["train"])
    rng.shuffle(pool)
    sft, failed = [], 0
    for t in pool:
        if len(sft) >= args.sft_n:
            break
        try:
            conv = render_dialogue(t["query"], t["calls"], t["tools"], functions, SYSTEM, args.args)
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
    (out / "train_sft.json").write_text(json.dumps(sft, ensure_ascii=False))
    stats["sft"] = {
        "n": len(sft),
        "render_failed": failed,
        "args": args.args,
        "turns": _hist(r["turns"] for r in sft),
        "ids_sha256": hashlib.sha256("\n".join(r["id"] for r in sft).encode()).hexdigest(),
    }
    stats["revisions"] = {
        "mathqa": MATHQA_REVISION,
        "aqua_rat": AQUA_REVISION,
        "nestful": nf.NESTFUL_HF_REVISION,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats["sft"]), flush=True)


if __name__ == "__main__":
    main()
