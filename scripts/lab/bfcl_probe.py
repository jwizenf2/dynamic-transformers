"""Early loop-1–4 probe on BFCL single-turn prompts (short-horizon routing control; collection).

    python scripts/lab/bfcl_probe.py \
        --adapters /Users/jeremywizenfeld/AI/keep/ouro-chain-depth-v3/adapters.safetensors

Why: on NESTful, per-request depth routing from early signals fails, and the diagnosis
(docs/research/router_diagnosis_2026-11-01.md) attributes it to consequences that build up over
many steps of a long chain. BFCL v3 non-live single-turn (simple, multiple, parallel,
parallel_multiple; 1,000 prompts) is the short-horizon control: one assistant turn, graded by
the AST checker. Chain v3's per-depth outcomes on these prompts already exist
(``ouro_sft_eval.py --task bfcl --n 0``, steps 1–4); this records, for the same prompts built by
``ouro_sft_eval.items``, the 48-token probe of ``router_probe2.probe`` (4-loop greedy draft,
per token and loop: exit gate, entropy, top-2 margin, agreement with loop 4, relative state
change), the probe tokens and the prompt length. Rows are appended to
``results/bfcl_probe/<model>/examples.jsonl``; a restart skips done ids.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ouro_sft_eval import items, load_model  # noqa: E402
from router_probe2 import probe  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B")
    parser.add_argument("--adapters", required=True)
    parser.add_argument("--k", type=int, default=48)
    parser.add_argument("--n", type=int, default=0, help="per category (0 = all)")
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    args = parser.parse_args()

    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    model, tok = load_model(args.model, None, args.adapters)
    stop_ids = {0, tok.convert_tokens_to_ids("<|im_end|>")}
    work = list(items(argparse.Namespace(task="bfcl", n=args.n), tok))
    out = Path(f"results/bfcl_probe/{Path(args.adapters).parent.name}")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "examples.jsonl"
    done, good = set(), []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                done.add(json.loads(line)["id"])
            except json.JSONDecodeError:
                break
            good.append(line)
        path.write_text("".join(x + "\n" for x in good))
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    t0 = time.perf_counter()
    with open(path, "a") as rows:
        for j, (cat, ex_id, ids, _grade) in enumerate(work, 1):
            if ex_id in done:
                continue
            stats, out_ids = probe(model, ids, args.k, stop_ids)
            row = {
                "id": ex_id,
                "cat": cat,
                "prompt_tokens": len(ids),
                "probe": [
                    [[None if math.isnan(x) else round(float(x), 5) for x in loop] for loop in t]
                    for t in stats.tolist()
                ],
                "token_ids": out_ids,
                "text": tok.decode(out_ids),
            }
            rows.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.flush()
            mx.clear_cache()
            print(f"::progress {j}/{len(work)} requests", flush=True)
            if j % 50 == 0:
                print(json.dumps({"done": j, "s": round(time.perf_counter() - t0)}), flush=True)


if __name__ == "__main__":
    main()
