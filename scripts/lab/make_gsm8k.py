"""Write GSM8K test problems as NESTful-format tool-calling tasks (``dht.toolcall.gsm8k``).

    python scripts/lab/make_gsm8k.py        # results/gsm8k/test_tasks.jsonl

Downloads the GSM8K ``main`` test split (openai/gsm8k, MIT) at a pinned revision, converts each
problem's annotated solution into calls over NESTful's vendored tools, keeps the problems whose
calls execute to the gold answer (at least two calls), and writes tasks readable by
``ouro_nestful_multiturn.load_tasks`` (``router_smoke.py --tasks``). Prints the reasons for
dropped problems and the distribution of call counts and dependency depths. None of the models
was fine-tuned on GSM8K (chain v3: Hermes FC, synthetic chains, MathQA train); base-model
pretraining exposure cannot be ruled out and is stated as a limitation.
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
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ouro_nestful as nf  # noqa: E402
from make_mathqa import TOOL_SPECS_SHA256  # noqa: E402
from ouro_nestful import dependency_depth  # noqa: E402

from dht.toolcall.gsm8k import build_task  # noqa: E402

GSM8K = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="tool-list sampling")
    parser.add_argument("--out", default="results/gsm8k")
    args = parser.parse_args()

    specs = nf.VENDORED / "tool_specs.json"
    if hashlib.sha256(specs.read_bytes()).hexdigest() != TOOL_SPECS_SHA256:
        raise ValueError(f"{specs} does not match its pinned SHA256")
    catalog = {t["name"]: t for t in json.loads(specs.read_text(encoding="utf-8"))}
    functions = nf.load_functions(
        nf.checked(nf.VENDORED / "basic_functions.py", "basic_functions.py")
    )
    path = hf_hub_download(
        GSM8K, "main/test-00000-of-00001.parquet", repo_type="dataset", revision=GSM8K_REVISION
    )
    rows = pq.read_table(path).to_pylist()
    rng = random.Random(args.seed)
    tasks, reasons = [], Counter()
    for k, row in enumerate(rows):
        task, why = build_task(row, catalog, functions, rng)
        reasons[why] += 1
        if task:
            tasks.append({"id": f"gsm8k-test-{k}", **task})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_tasks.jsonl").write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks)
    )
    stats = {
        "source": f"{GSM8K}@{GSM8K_REVISION}",
        "problems": len(rows),
        "kept": len(tasks),
        "reasons": dict(reasons.most_common()),
        "calls": dict(sorted(Counter(len(t["calls"]) for t in tasks).items())),
        "dependency_depth": dict(
            sorted(Counter(dependency_depth(t["calls"]) for t in tasks).items())
        ),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
