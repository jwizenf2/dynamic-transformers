"""Write synthetic executed tool-call chains (dht.toolcall.chains) as Hermes-format JSON.

    python scripts/lab/make_chains.py --n 6000 --seed 0     # results/chains/chains_s0_n6000.json

The system prompt is the Hermes function-calling prompt used for evaluation
(``ouro_sft_eval.SYSTEM``), so training and evaluation share the tool-prompt format. Rows have
the ``{"id", "conversations": [{"from", "value"}]}`` layout of hermes-function-calling-v1 and are
read by ``ouro_lora_sft.py --extra-data``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ouro_sft_eval import SYSTEM  # noqa: E402

from dht.toolcall.chains import generate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--branch-p", type=float, default=0.2)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = generate(args.n, args.seed, SYSTEM, branch_p=args.branch_p)
    out = Path(args.out or f"results/chains/chains_s{args.seed}_n{args.n}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False))
    stats = {
        "n": len(rows),
        "calls": dict(sorted(Counter(r["n_calls"] for r in rows).items())),
        "turns": dict(sorted(Counter(r["turns"] for r in rows).items())),
        "out": str(out),
    }
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
