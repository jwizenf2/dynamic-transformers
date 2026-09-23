# ruff: noqa: B023  (select() closure is only used inside its own loop iteration)
"""Validation-only: does WHICH heads run matter? Swap TS heads for random or lowest-scored."""

import random
import sys
from collections import Counter
from pathlib import Path

import torch

from dht.config import ExperimentConfig
from dht.data import load_split
from dht.models.pretrained import build_model
from dht.runtime import select_device
from dht.training.checkpointing import read_checkpoint
from dht.training.trainer import restore_controller


def macro_f1(pred, gold, k):
    f = []
    for c in range(k):
        tp = sum(p == c and g == c for p, g in zip(pred, gold, strict=True))
        fp = sum(p == c and g != c for p, g in zip(pred, gold, strict=True))
        fn = sum(p != c and g == c for p, g in zip(pred, gold, strict=True))
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f) / k


path = Path(sys.argv[1])
device = select_device("mps", allow_fallback=False)
state = read_checkpoint(path)
config = ExperimentConfig.model_validate(state["config"])
model = build_model(config, initialize=False).to(device)
model.load_state_dict(state["model"])
model.eval()
variant = config.model_copy(
    update={
        "dht": config.dht.model_copy(update={"execution": "batched", "context_pooling": "device"})
    }
)
tokens, labels = load_split(config, "validation")
gold = [int(x) for x in labels]
out = {}
allpreds = {}
alllog = {}
for strategy in ["ts", "random", "worst"]:
    ctl = restore_controller(variant, state)
    ts = ctl.ts
    orig = ts.select
    rng = random.Random(0)
    spreads, picks = [], Counter()

    def sel(features, count, *, explore=True, parameters=None, _o=orig):
        p = ts.probabilities(features)
        spreads.append(float(p.max() - p.min()))
        if strategy == "ts":
            s = _o(features, count, explore=explore, parameters=parameters)
        elif strategy == "random":
            s = tuple(sorted(rng.sample(range(ts.num_arms), count)))
        else:
            s = tuple(sorted(torch.argsort(p, stable=True)[:count].tolist()))
        if count < ts.num_arms:
            picks[s] += 1
        return s

    ts.select = sel
    preds = []
    with torch.inference_mode():
        for i in range(0, len(tokens), 64):
            logits, _ = ctl.predict(model, tokens[i : i + 64].to(device))
            preds += logits.argmax(-1).tolist()
            alllog.setdefault(strategy, []).append(logits.float().cpu())
    allpreds[strategy] = preds
    s = sorted(spreads)
    out[strategy] = {
        "macro_f1": round(
            macro_f1(
                preds, gold, config.data.num_classes if hasattr(config.data, "num_classes") else 4
            ),
            5,
        ),
        "accuracy": round(sum(p == g for p, g in zip(preds, gold, strict=True)) / len(gold), 5),
        "prob_spread_median": round(s[len(s) // 2], 4),
        "prob_spread_p95": round(s[int(0.95 * len(s))], 4),
        "distinct_subsets": len(picks),
        "top_subset_share": round(picks.most_common(1)[0][1] / max(1, sum(picks.values())), 3)
        if picks
        else None,
    }
    print(path.parent.name, strategy, out[strategy], flush=True)

L = {k: torch.cat(v) for k, v in alllog.items()}
for k in ["random", "worst"]:
    ch = sum(a != b for a, b in zip(allpreds["ts"], allpreds[k], strict=True))
    print(
        path.parent.name,
        k,
        "changed predictions vs ts:",
        ch,
        "max|dlogit|:",
        round(float((L[k] - L["ts"]).abs().max()), 4),
        "mean|dlogit|:",
        round(float((L[k] - L["ts"]).abs().mean()), 4),
    )
