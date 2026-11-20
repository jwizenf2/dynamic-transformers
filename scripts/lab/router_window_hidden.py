"""Three offline router checks: decision window, attention over hidden states, cleaner labels.

    PYTHONPATH=scripts/lab python scripts/lab/router_window_hidden.py

Exploratory follow-ups to docs/research/router_nonlinear_2026-11-18.md (no model is run).

W  Decision window (MathQA 531, ``router_trace`` 2-loop traces): for a decision after T generated
   tokens of the 2-loop dialogue (T = 16 … 256, the end of turn 1, of turn 2, of the dialogue),
   features pooled over the trace's loop-1/2 per-token signals up to T (mean/max/min of entropy,
   top probability, margin, gate per loop, KL(p_2 ‖ p_1), loop-1/2 agreement, log p_1, state
   change; digit share; tool-response surprisal of the tool turns seen) plus the 12 early probe
   features; logistic and MLP, 5x5 CV. Reported per T: AUC for "4 helps"; timeliness = share of
   "4 helps" requests whose 2- and 4-loop dialogues have not yet diverged at T (a switch to 4
   loops there reproduces the 4-loop dialogue exactly, ``router_switch.py``); the switch policy
   (escalate the top-q scores at T): accuracy lower bound / midpoint at <= 0.90x fixed-4 cost and
   the gain over the 2–4 random mix. A switch after the divergence has an unknown outcome (bounds
   min/max of the 2- and 4-loop outcomes, midpoint the 2-loop one, cost 4·T_2); a dialogue shorter
   than T is decided at its end, and escalation then restarts at 4 loops (cost 2·T_2 + 4·T_4).
H  Attention over the 48 probe tokens' hidden states (2,531 problems with saved loop-1/2 states):
   per token [h_2, h_2 − h_1] → 64 → learned attention weights (with token-type embedding) →
   pooled vector + benchmark → MLP; vs the mean-pooled MLP of ``router_nonlinear``; same seeds,
   folds and full training size.
L  Cleaner labels: "stable helps" = wrong at 2, right at 4 AND at 6 loops (NESTful [0, 1200) and
   GSM8K have depth 6; MathQA keeps the greedy label). Mean-pooled hidden MLP trained on stable
   vs greedy labels, each scored on both labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from phase_a_labels import auc
from router_diagnosis import predict, standardise
from router_nonlinear import (
    MLP,
    SEEDS,
    TYPES,
    hidden_rows,
    load,
    source_onehot,
    train_net,
)
from router_probe2_analysis import features_escalate
from router_switch import load as load_switch

M = "ouro-chain-depth-v3"
SIG = (
    "ent1",
    "ent2",
    "top1",
    "top2",
    "mar1",
    "mar2",
    "gate1",
    "gate2",
    "kl21",
    "agree",
    "lp1",
    "change",
)
POINTS = (16, 32, 48, 64, 96, 128, 192, 256, "turn1", "turn2", "end")


# ---------------------------------------------------------------- W: decision window


def traces() -> list[dict]:
    """MathQA traces aligned with ``router_switch.load`` (forks, costs, outcomes)."""
    from router_diagnosis import tokenizer

    tok = tokenizer()
    out = []
    for set_name, smoke_dir, trace_dir, probe_dir in (
        ("clean", M, f"{M}_clean", M),
        ("pool", f"{M}_testpool", f"{M}_testpool", f"{M}_testpool"),
    ):
        sw = load_switch(set_name, tok)
        tr = [json.loads(x) for x in open(f"results/router_trace/{trace_dir}/examples.jsonl")]
        smoke = {
            r["id"]: r
            for r in map(json.loads, open(f"results/router_smoke/{smoke_dir}/examples.jsonl"))
        }
        probes = {
            r["id"]: r
            for r in map(json.loads, open(f"results/router_probe2/{probe_dir}/examples.jsonl"))
        }
        for t, s in zip(tr, sw, strict=True):
            p = probes[t["id"]]
            p["loop12_disagree"] = sum(x[0][3] != x[1][3] for x in p["probe"])
            out.append(
                {
                    **s,
                    "trace": t,
                    "x12": np.nan_to_num(features_escalate(p, smoke[t["id"]]["query"])),
                }
            )
    return out


def window_features(r: dict, point) -> tuple[np.ndarray, int]:
    """(features from the 2-loop trace up to the decision point, decision index in trace steps)."""
    sig = r["trace"]["turn_signals"]
    lens = [len(s["ent2"]) for s in sig]
    total = sum(lens)
    if point == "end":
        cut = total
    elif isinstance(point, str):
        k = int(point[-1])
        cut = sum(lens[:k]) if len(lens) > k else total
    else:
        cut = min(point, total)
    cols = {k: np.concatenate([s[k] for s in sig])[:cut] for k in (*SIG, "digit")}
    f = []
    for k in SIG:
        v = np.asarray(cols[k], float)
        f += [v.mean(), v.max(), v.min()] if len(v) else [0.0, 0.0, 0.0]
    f.append(float(np.mean(cols["digit"])) if cut else 0.0)
    # tool turns already returned before the cut
    ends = np.cumsum(lens)
    seen = [
        s["tool_before"]
        for i, s in enumerate(sig)
        if i > 0 and ends[i - 1] <= cut and s["tool_before"]
    ]
    sur = [x["surprisal_mean"] for x in seen]
    f += [float(np.mean(sur)) if sur else 0.0, float(len(seen)), float(cut)]
    return np.concatenate([np.array(f), r["x12"]]), cut


def cv_scores(x, y, model: str) -> np.ndarray:
    n, s = len(x), np.zeros(len(x))
    for rep in range(5):
        order = np.random.default_rng(rep).permutation(n)
        for f in range(5):
            te = order[f::5]
            tr = np.setdiff1d(order, te)
            a, b = standardise(x[tr], x[te])
            if model == "logistic":
                s[te] += predict(a, y[tr], b) / 5
            else:
                s[te] += (
                    train_net(
                        lambda d=a.shape[1]: MLP(d),
                        [torch.tensor(a, dtype=torch.float32)],
                        y[tr],
                        [torch.tensor(b, dtype=torch.float32)],
                        rep * 10 + f,
                    )
                    / 5
                )
    return s


def switch_eval(rows, score, cuts) -> dict:
    """Escalate the top-q scores at the decision point; accuracy bounds at <= 0.90x."""
    base = 4 * np.mean([r["t4"] for r in rows])
    c2 = 2 * np.mean([r["t2"] for r in rows]) / base
    a2, a4 = np.mean([r["y2"] for r in rows]), np.mean([r["y4"] for r in rows])
    pts = []
    for q in np.linspace(0, 1, 101):
        esc = score >= np.quantile(score, 1 - q) if q > 0 else np.zeros(len(rows), bool)
        cost = lo = mid = 0.0
        for r, e, cut in zip(rows, esc, cuts, strict=True):
            total = sum(len(s["ent2"]) for s in r["trace"]["turn_signals"])
            if not e:
                c, lo_r, mid_r = 2 * r["t2"], r["y2"], r["y2"]
            elif cut >= total:  # decided at the end: restart at 4 loops
                c, lo_r, mid_r = 2 * r["t2"] + 4 * r["t4"], r["y4"], r["y4"]
            elif r["fork"] is None or cut <= r["fork"]:  # exact switch
                c, lo_r, mid_r = 4 * r["t4"], r["y4"], r["y4"]
            else:  # switch after divergence: outcome unknown
                c, lo_r, mid_r = 4 * r["t2"], min(r["y2"], r["y4"]), r["y2"]
            cost, lo, mid = cost + c, lo + lo_r, mid + mid_r
        pts.append((cost / len(rows) / base, lo / len(rows), mid / len(rows)))
    ok = [p for p in pts if p[0] <= 0.90]
    best = max(ok, key=lambda p: p[1])
    mix = a2 + (a4 - a2) * (best[0] - c2) / (1 - c2)
    return {
        "cost": best[0],
        "acc_low": best[1],
        "acc_mid": best[2],
        "gain_low_vs_mix": best[1] - mix,
        "fixed_d4": a4,
    }


def window_curve() -> dict:
    rows = traces()
    helps = np.array([r["y2"] == 0 and r["y4"] == 1 for r in rows], float)
    out = {}
    for point in POINTS:
        feats = [window_features(r, point) for r in rows]
        x = np.stack([f[0] for f in feats])
        cuts = [f[1] for f in feats]
        timely = [
            (r["fork"] is None or cut <= r["fork"])
            and cut < sum(len(s["ent2"]) for s in r["trace"]["turn_signals"])
            for r, cut in zip(rows, cuts, strict=True)
        ]
        res = {
            "timely_share_of_helps": float(
                np.mean([t for t, h in zip(timely, helps, strict=True) if h])
            )
        }
        for model in ("logistic", "mlp"):
            s = cv_scores(x, helps, model)
            res[model] = {
                "auc_4_helps": auc(s, helps),
                "switch_at_0.90x": switch_eval(rows, s, cuts),
            }
        out[str(point)] = res
        print(
            json.dumps(
                {
                    "point": point,
                    "auc_logistic": res["logistic"]["auc_4_helps"],
                    "auc_mlp": res["mlp"]["auc_4_helps"],
                    "timely": res["timely_share_of_helps"],
                }
            ),
            flush=True,
        )
    return out


# ---------------------------------------------------------------- H and L: hidden states


class AttnPool(torch.nn.Module):
    def __init__(self, d_in: int, d_side: int, h: int = 64):
        super().__init__()
        self.proj = torch.nn.Linear(d_in, h)
        self.type_emb = torch.nn.Embedding(len(TYPES), h)
        self.score = torch.nn.Linear(h, 1)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(h + d_side, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x, types, mask, side):
        z = torch.nn.functional.gelu(self.proj(x) + self.type_emb(types))
        w = self.score(z).squeeze(-1).masked_fill(mask == 0, -1e4).softmax(-1)
        pooled = (w.unsqueeze(-1) * z).sum(1)
        return self.head(torch.cat([pooled, side], 1)).squeeze(-1)


def hidden_data(rows):
    sub = hidden_rows(rows)
    T = 48
    seqs = np.zeros((len(sub), T, 4096), np.float16)
    types = np.zeros((len(sub), T), np.int64)
    mask = np.zeros((len(sub), T), np.float32)
    for i, r in enumerate(sub):
        h = np.load(r["hidden"]).astype(np.float32)[:T]
        n = len(h)
        seqs[i, :n] = np.concatenate([h[:, 1], h[:, 1] - h[:, 0]], 1)
        types[i, :n] = [TYPES.index(t) for t in r["types"][:n]]
        mask[i, :n] = 1
    return sub, seqs, types, mask


def labels(sub):
    deep = {}
    for tag in ("nestful_deep", "nestful240_deep", "nestful720", "gsm8k"):
        for r in map(json.loads, open(f"results/router_smoke/{M}_{tag}/examples.jsonl")):
            if "d6_greedy" in r["runs"]:
                deep[r["id"]] = r["runs"]["d6_greedy"]["correct"]
    greedy = np.array([r["y2"] == 0 and r["y4"] == 1 for r in sub], float)
    stable = np.array(
        [r["y2"] == 0 and r["y4"] == 1 and deep.get(r["id"], True) for r in sub], float
    )
    return greedy, stable, float(np.mean([r["id"] in deep for r in sub]))


def hidden_checks(rows) -> dict:
    sub, seqs, types, mask = hidden_data(rows)
    greedy, stable, deep_share = labels(sub)
    mean_x = np.hstack(
        [
            (seqs.astype(np.float32) * mask[..., None]).sum(1) / mask.sum(1, keepdims=True),
            source_onehot(sub),
        ]
    )
    side_all = source_onehot(sub)
    n = len(sub)
    res: dict = {
        "n": n,
        "share_with_depth6": deep_share,
        "rate_greedy": float(greedy.mean()),
        "rate_stable": float(stable.mean()),
    }
    scores = {k: [] for k in ("mean_mlp_greedy", "mean_mlp_stable", "attention_greedy")}
    for seed in range(SEEDS):
        order = np.random.default_rng(seed).permutation(n)
        preds = {k: np.zeros(n) for k in scores}
        for f in range(5):
            te = order[f::5]
            tr = np.setdiff1d(order, te)
            a, b = standardise(mean_x[tr], mean_x[te])
            for lab, y in (("greedy", greedy), ("stable", stable)):
                preds[f"mean_mlp_{lab}"][te] = train_net(
                    lambda d=a.shape[1]: MLP(d, 32),
                    [torch.tensor(a, dtype=torch.float32)],
                    y[tr],
                    [torch.tensor(b, dtype=torch.float32)],
                    seed * 100 + f,
                    wd=1e-2,
                )
            mu = seqs[tr].astype(np.float32).reshape(-1, 4096)[mask[tr].reshape(-1) > 0].mean(0)
            sd = (
                seqs[tr].astype(np.float32).reshape(-1, 4096)[mask[tr].reshape(-1) > 0].std(0)
                + 1e-6
            )

            def tens(idx, mu=mu, sd=sd):
                x = torch.tensor((seqs[idx].astype(np.float32) - mu) / sd) * torch.tensor(
                    mask[idx]
                ).unsqueeze(-1)
                return [
                    x,
                    torch.tensor(types[idx]),
                    torch.tensor(mask[idx]),
                    torch.tensor(side_all[idx], dtype=torch.float32),
                ]

            preds["attention_greedy"][te] = train_net(
                lambda: AttnPool(4096, side_all.shape[1]),
                tens(tr),
                greedy[tr],
                tens(te),
                seed * 100 + f,
                wd=1e-2,
            )
            print(json.dumps({"hidden_seed": seed, "fold": f}), flush=True)
        for k, p in preds.items():
            scores[k].append({"auc_greedy": auc(p, greedy), "auc_stable": auc(p, stable)})
    for k, v in scores.items():
        res[k] = {
            m: {"mean": float(np.mean([s[m] for s in v])), "sd": float(np.std([s[m] for s in v]))}
            for m in ("auc_greedy", "auc_stable")
        }
    return res


def main() -> None:
    report = {"W_window": window_curve()}
    report["H_L_hidden"] = hidden_checks(load())
    out = Path("results/router_nonlinear")
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=1, default=float)
    (out / "window_hidden.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
