"""Why early signals fail to route depth: router design, representation, or early-prediction limit.

    python scripts/lab/router_diagnosis.py

Offline and exploratory (no model run). Data, per set: greedy multi-turn dialogues at 2 and 4
loops (``router_smoke``) and the 48-token first-answer probe with all four loops' readouts
(``router_probe2``: per token and loop the exit gate, entropy, top-2 margin, whether the loop's
argmax equals loop 4's, relative state change), for NESTful [0, 720) and the 531 MathQA
router-training tasks (which also have loop-1/2 hidden states at the probe tokens).
"4 helps" = wrong at 2 loops, right at 4 (greedy).

D1 Where the 2- and 4-loop dialogues first differ: token index of the first differing assistant
   token (turns concatenated), by outcome pattern; the share of "4 helps" requests whose two
   dialogues are still identical after the probe window (45 tokens) and after the first turn.
   If the shallow and deep runs write the same first tokens, no readout of those tokens'
   *outputs* can separate them; only internal state could.
D2 Is loop 4's revision of a token predictable from loops 1–2? Token-level label: loop 2's argmax
   differs from loop 4's (along the probe). Predictors: loop-1/2 readouts (entropy, margin, gate,
   state change, loop-1 vs loop-2 disagreement); on MathQA also the loop-1/2 hidden states (PCA
   64 + logistic). Grouped 5-fold CV by request; AUC overall and on decision tokens (function
   names, numbers).
D3 Information ladder for "4 helps" (request level, 5x5 CV, AUC; each level adds information,
   ignoring what it would cost to obtain):
   L0 loop-1/2 readouts of the probe (deployable, the frozen router's 12 features);
   L1 + loop-3/4 readouts and convergence (state change at loops 3–4) on the probe tokens;
   L2 + the realised revisions: counts of tokens where loop 2's argmax differs from loop 4's,
       by token type;
   L3 + first-turn verification: whether the 2-loop first assistant turn differs from the 4-loop
       one (what a 4-loop pass over the drafted turn would reveal);
   L4 + where the full dialogues diverge (token index; needs the whole 4-loop run).
   Models: logistic on P_4 − P_2 (two per-depth models), logistic directly on the "4 helps" label
   (marginal benefit), and a quadratic-feature logistic (capacity check).
D4 Decomposition: "4 helps" = (2 wrong) AND (4 right). AUC of L0 features for "2 wrong" and for
   "4 right" separately, and for "4 right" among the requests 2 loops gets wrong (the escalation
   decision a perfect verifier of 2-loop answers would leave).
D5 (MathQA only, which has 4 sampled dialogues per depth, T 0.7) Output disagreement at 2 loops:
   the number of distinct first-turn calls and of distinct final answers among the greedy and 4
   sampled 2-loop runs, and the share of samples agreeing with the greedy answer. First-turn
   disagreement is cheap to deploy (samples share the prompt cache); final-answer disagreement
   needs four extra full 2-loop dialogues (~2x fixed-4 cost) and is an information bound only.
   AUC alone and added to L0 (5x5 CV, P_4 - P_2 scores).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from ouro_sft_eval import parse_hermes
from phase_a_labels import auc
from router_probe2_analysis import features_escalate, features_v2
from router_smoke_analysis import fit_logistic

M = "ouro-chain-depth-v3"
SETS = {
    "nestful_720": [("nestful", "nestful", None), ("nestful240", "nestful240", None)],
    "mathqa_531": [("", "", "_hidden"), ("_testpool", "_testpool", "_testpool_hidden")],
}
WINDOW = 45


def tag(t: str) -> str:
    return f"{M}_{t}" if t and not t.startswith("_") else f"{M}{t}"


def tokenizer():
    from transformers import PreTrainedTokenizerFast

    snap = next(Path.home().glob(".cache/huggingface/hub/models--ByteDance--Ouro-2.6B/snapshots/*"))
    return PreTrainedTokenizerFast(tokenizer_file=str(snap / "tokenizer.json"))


def load(name: str) -> list[dict]:
    rows = []
    for smoke_tag, probe_tag, hidden_tag in SETS[name]:
        probes = {
            r["id"]: r
            for r in map(json.loads, open(f"results/router_probe2/{tag(probe_tag)}/examples.jsonl"))
        }
        for r in map(json.loads, open(f"results/router_smoke/{tag(smoke_tag)}/examples.jsonl")):
            p = probes[r["id"]]
            p["loop12_disagree"] = sum(t[0][3] != t[1][3] for t in p["probe"])
            r["probe2"] = p
            r["hidden"] = (
                Path(f"results/router_probe2/{tag(hidden_tag)}/hidden/{r['id']}.npy")
                if hidden_tag
                else None
            )
            rows.append(r)
    return rows


def divergence(tok, a: list[str], b: list[str]) -> tuple[int, int]:
    """(token index of the first differing assistant token, turn index); identical → large."""
    pos = 0
    for k in range(max(len(a), len(b))):
        ta = tok.encode(a[k]) if k < len(a) else []
        tb = tok.encode(b[k]) if k < len(b) else []
        if ta == tb:
            pos += len(ta)
            continue
        j = next((i for i, (x, y) in enumerate(zip(ta, tb, strict=False)) if x != y), None)
        return pos + (j if j is not None else min(len(ta), len(tb))), k
    return 10**6, 10**6


def token_types(tok, p: dict) -> list[str]:
    text, spans = "", []
    for t in p["token_ids"][: len(p["probe"])]:
        piece = tok.decode([t])
        spans.append((len(text), len(text) + len(piece)))
        text += piece
    names = [m.span(1) for m in re.finditer(r'"name":\s*"([^"]*)"', text)]
    nums = [m.span() for m in re.finditer(r"(?<=[:\s\[,])-?\d+(?:\.\d+)?", text)]
    out = []
    for a, b in spans:
        if any(a < e and b > s for s, e in names):
            out.append("function_name")
        elif any(a < e and b > s for s, e in nums):
            out.append("number")
        else:
            out.append("other")
    return out + ["other"] * (len(p["probe"]) - len(out))  # the stop token has no id


def standardise(a, b):
    mu, sd = np.nanmean(a, 0), np.nanstd(a, 0) + 1e-6
    return np.nan_to_num((a - mu) / sd), np.nan_to_num((b - mu) / sd)


def predict(xa, ya, xb, l2=1.0):
    w = fit_logistic(xa, ya, l2=l2)
    return 1 / (1 + np.exp(-(np.hstack([xb, np.ones((len(xb), 1))]) @ w)))


def quad(x):
    i, j = np.triu_indices(x.shape[1])
    return np.hstack([x, x[:, i] * x[:, j]])


def cv_scores(x, y2, y4, helps, repeats=5):
    """Out-of-fold scores: P_4 − P_2, direct 'helps', quadratic direct."""
    n = len(x)
    out = {k: np.zeros(n) for k in ("p4_minus_p2", "direct", "direct_quadratic")}
    for rep in range(repeats):
        order = np.random.default_rng(rep).permutation(n)
        for f in range(5):
            te = order[f::5]
            tr = np.setdiff1d(order, te)
            a, b = standardise(x[tr], x[te])
            out["p4_minus_p2"][te] += (predict(a, y4[tr], b) - predict(a, y2[tr], b)) / repeats
            out["direct"][te] += predict(a, helps[tr], b) / repeats
            qa, qb = standardise(quad(a), quad(b))
            out["direct_quadratic"][te] += predict(qa, helps[tr], qb, l2=10.0) / repeats
    return out


def first_turn_signature(texts: list[str]) -> str:
    calls = parse_hermes(texts[0] if texts else "")
    return repr([(c["name"], sorted(c["arguments"].items())) for c in calls])


def sample_disagreement(rows: list[dict]) -> np.ndarray:
    """[n, 3]: distinct first turns, distinct final answers (greedy + 4 samples at 2 loops),
    share of the 4 samples whose final answer equals the greedy one."""
    extra = {}
    for folder in (f"{M}_clean_samples", f"{M}_testpool_samples"):
        for r in map(json.loads, open(f"results/router_smoke/{folder}/examples.jsonl")):
            extra[r["id"]] = r["runs"]
    out = []
    for r in rows:
        runs = {**r["runs"], **extra[r["id"]]}
        names = ["d2_greedy", "d2_sample", "d2_sample1", "d2_sample2", "d2_sample3"]
        rs = [runs[k] for k in names]
        out.append(
            [
                len({first_turn_signature(x["texts"]) for x in rs}),
                len({x["value"] for x in rs}),
                float(np.mean([x["value"] == rs[0]["value"] for x in rs[1:]])),
            ]
        )
    return np.array(out, float)


def analyse(name: str, tok) -> dict:
    rows = load(name)
    n = len(rows)
    y2 = np.array([r["runs"]["d2_greedy"]["correct"] for r in rows], float)
    y4 = np.array([r["runs"]["d4_greedy"]["correct"] for r in rows], float)
    helps = ((y2 == 0) & (y4 == 1)).astype(float)
    hurts = ((y2 == 1) & (y4 == 0)).astype(float)

    # D1 divergence
    div = [
        divergence(tok, r["runs"]["d2_greedy"]["texts"], r["runs"]["d4_greedy"]["texts"])
        for r in rows
    ]
    pos = np.array([d[0] for d in div])
    turn = np.array([d[1] for d in div])
    groups = {
        "helps": helps == 1,
        "hurts": hurts == 1,
        "both_right": (y2 == 1) & (y4 == 1),
        "both_wrong": (y2 == 0) & (y4 == 0),
    }
    d1 = {
        g: {
            "n": int(m.sum()),
            "identical_dialogues": float((pos[m] >= 10**6).mean()),
            "same_through_probe_window": float((pos[m] >= WINDOW).mean()),
            "same_first_turn": float((turn[m] >= 1).mean()),
            "median_divergence_token": float(np.median(pos[m][pos[m] < 10**6]))
            if (pos[m] < 10**6).any()
            else None,
        }
        for g, m in groups.items()
    }

    # D2 token-level revision predictability
    feats, labels, kinds, groups_tok, hid = [], [], [], [], []
    for i, r in enumerate(rows):
        p = np.array(
            [
                [[np.nan if x is None else x for x in loop] for loop in t]
                for t in r["probe2"]["probe"]
            ],
            float,
        )
        types = token_types(tok, r["probe2"])
        h = np.load(r["hidden"]).astype(np.float32) if r["hidden"] else None
        for t in range(len(p)):
            feats.append(
                [p[t, 0, 1], p[t, 1, 1], p[t, 0, 2], p[t, 1, 2], p[t, 0, 0], p[t, 1, 0], p[t, 1, 4]]
                + [float(p[t, 0, 3] != p[t, 1, 3])]
            )
            labels.append(float(p[t, 1, 3] == 0))
            kinds.append(types[t])
            groups_tok.append(i)
            if h is not None:
                hid.append(
                    np.concatenate(
                        [
                            h[min(t, len(h) - 1), 1],
                            h[min(t, len(h) - 1), 1] - h[min(t, len(h) - 1), 0],
                        ]
                    )
                )
    feats, labels = np.array(feats), np.array(labels)
    kinds, groups_tok = np.array(kinds), np.array(groups_tok)
    score = np.zeros(len(labels))
    score_h = np.zeros(len(labels)) if hid else None
    hid = np.array(hid) if hid else None
    folds = np.random.default_rng(0).permutation(n) % 5
    for f in range(5):
        te = folds[groups_tok] == f
        a, b = standardise(feats[~te], feats[te])
        score[te] = predict(a, labels[~te], b)
        if hid is not None:
            ha, hb = standardise(hid[~te], hid[te])
            mu = ha.mean(0)
            _, _, vt = np.linalg.svd(ha[:: max(1, len(ha) // 4000)] - mu, full_matrices=False)
            za, zb = (ha - mu) @ vt[:64].T, (hb - mu) @ vt[:64].T
            score_h[te] = predict(np.hstack([za, a]), labels[~te], np.hstack([zb, b]))
    d2 = {}
    for k, m in (
        ("all", np.ones(len(labels), bool)),
        ("function_name", kinds == "function_name"),
        ("number", kinds == "number"),
    ):
        d2[k] = {
            "n_tokens": int(m.sum()),
            "revised_share": float(labels[m].mean()),
            "auc_readouts": auc(score[m], labels[m]) if labels[m].any() else None,
            "auc_hidden_plus_readouts": auc(score_h[m], labels[m])
            if score_h is not None and labels[m].any()
            else None,
        }

    # D3 information ladder
    x0 = np.stack([features_escalate(r["probe2"], r["query"]) for r in rows])
    xv2 = np.stack([features_v2(r["probe2"], r["query"]) for r in rows])
    x1 = np.hstack([x0, xv2[:, 8:16], xv2[:, 17:19]])  # loops 3–4 readouts, change at loops 3–4
    rev = []
    for r in rows:
        types = token_types(tok, r["probe2"])
        dis = [t[1][3] == 0 for t in r["probe2"]["probe"]]
        rev.append(
            [
                sum(dis),
                sum(d for d, k in zip(dis, types, strict=False) if k == "function_name"),
                sum(d for d, k in zip(dis, types, strict=False) if k == "number"),
            ]
        )
    x2 = np.hstack([x1, np.array(rev, float)])
    x3 = np.hstack([x2, (turn < 1).astype(float)[:, None]])
    x4 = np.hstack([x3, np.log1p(np.minimum(pos, 5000))[:, None], (pos >= 10**6)[:, None]])
    ladder = {}
    for lvl, x in (
        ("L0_loop12", x0),
        ("L1_+loop34", x1),
        ("L2_+revisions", x2),
        ("L3_+first_turn_verification", x3),
        ("L4_+full_divergence", x4),
    ):
        s = cv_scores(x, y2, y4, helps)
        ladder[lvl] = {k: auc(v, helps) for k, v in s.items()}

    # D4 decomposition with the deployable features
    d4 = {}
    for lbl, y, m in (
        ("d2_wrong", 1 - y2, np.ones(n, bool)),
        ("d4_right", y4, np.ones(n, bool)),
        ("d4_right_given_d2_wrong", y4, y2 == 0),
    ):
        s = np.zeros(m.sum())
        xm, ym = x0[m], y[m]
        for rep in range(5):
            order = np.random.default_rng(rep).permutation(m.sum())
            for f in range(5):
                te = order[f::5]
                tr = np.setdiff1d(order, te)
                a, b = standardise(xm[tr], xm[te])
                s[te] += predict(a, ym[tr], b) / 5
        d4[lbl] = {"n": int(m.sum()), "auc": auc(s, ym)}
    d5 = None
    if name.startswith("mathqa"):
        xs = sample_disagreement(rows)
        d5 = {
            "auc_distinct_first_turns": auc(xs[:, 0], helps),
            "auc_distinct_final_answers": auc(xs[:, 1], helps),
            "auc_low_agreement_with_greedy": auc(-xs[:, 2], helps),
            "auc_L0_plus_first_turn": auc(
                cv_scores(np.hstack([x0, xs[:, :1]]), y2, y4, helps)["p4_minus_p2"], helps
            ),
            "auc_L0_plus_all": auc(
                cv_scores(np.hstack([x0, xs]), y2, y4, helps)["p4_minus_p2"], helps
            ),
            "auc_L0": ladder["L0_loop12"]["p4_minus_p2"],
            "mean_distinct_first_turns": {g: float(xs[m, 0].mean()) for g, m in groups.items()},
        }
    return {
        "n": n,
        "D5_sample_disagreement": d5,
        "rates": {
            "d2": float(y2.mean()),
            "d4": float(y4.mean()),
            "helps": float(helps.mean()),
            "hurts": float(hurts.mean()),
        },
        "D1_divergence": d1,
        "D2_token_revision": d2,
        "D3_ladder_auc_4_helps": ladder,
        "D4_decomposition_auc": d4,
    }


def main() -> None:
    tok = tokenizer()
    report = {name: analyse(name, tok) for name in SETS}
    out = Path("results/router_diagnosis")
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=1, default=float)
    (out / "report.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
