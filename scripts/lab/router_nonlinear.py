"""Learning curves for linear vs nonlinear early-signal routers (offline, pre-registered).

    PYTHONPATH=scripts/lab python scripts/lab/router_nonlinear.py

See docs/research/router_nonlinear_2026-11-18.md for the question, models and decision rule.
Data: chain v3 seed 0 greedy outcomes at 2 and 4 loops and the 48-token probe (``router_probe2``)
on NESTful [0, 1200), GSM8K test (800) and MathQA clean val + test pool (531); MathQA also has
loop-1/2 hidden states at the probe tokens. Label: "4 loops help" (wrong at 2, right at 4).
Per-token inputs use loops 1–2 only (exit gate, entropy, top-2 margin, loop-2 state change,
loop-1/2 argmax disagreement) plus the token type and the benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from phase_a_labels import auc
from router_diagnosis import predict, standardise, token_types, tokenizer
from router_probe2_analysis import features_escalate

M = "ouro-chain-depth-v3"
SOURCES = {
    "nestful": [("nestful", None), ("nestful240", None), ("nestful720", None)],
    "gsm8k": [("gsm8k", None)],
    "mathqa": [("", "_hidden"), ("_testpool", "_testpool_hidden")],
}
TYPES = ("function_name", "number", "other", "structure")
SIZES = (250, 500, 1000, None)  # None = the whole training fold
SEEDS, FOLDS, T = 5, 5, 48
torch.set_num_threads(8)


def folder(tag: str) -> str:
    return f"{M}_{tag}" if tag and not tag.startswith("_") else f"{M}{tag}"


def load():
    tok = tokenizer()
    rows = []
    for src, parts in SOURCES.items():
        for tag, hidden in parts:
            probes = {
                r["id"]: r
                for r in map(
                    json.loads, open(f"results/router_probe2/{folder(tag)}/examples.jsonl")
                )
            }
            for r in map(json.loads, open(f"results/router_smoke/{folder(tag)}/examples.jsonl")):
                p = probes[r["id"]]
                p["loop12_disagree"] = sum(t[0][3] != t[1][3] for t in p["probe"])
                a = np.array(
                    [
                        [[np.nan if x is None else x for x in loop] for loop in t]
                        for t in p["probe"]
                    ],
                    float,
                )  # [t, 4, 5]: gate, entropy, margin, agrees-with-loop-4, change
                seq = np.stack(
                    [
                        a[:, 0, 0],
                        a[:, 1, 0],
                        a[:, 0, 1],
                        a[:, 1, 1],
                        a[:, 0, 2],
                        a[:, 1, 2],
                        np.nan_to_num(a[:, 1, 4]),
                        (a[:, 0, 3] != a[:, 1, 3]).astype(float),
                    ],
                    1,
                )  # [t, 8]
                rows.append(
                    {
                        "src": src,
                        "y2": float(r["runs"]["d2_greedy"]["correct"]),
                        "y4": float(r["runs"]["d4_greedy"]["correct"]),
                        "x12": np.nan_to_num(features_escalate(p, r["query"])),
                        "seq": np.nan_to_num(seq),
                        "types": token_types(tok, p)[: len(seq)],
                        "hidden": Path(
                            f"results/router_probe2/{folder(hidden)}/hidden/{r['id']}.npy"
                        )
                        if hidden
                        else None,
                    }
                )
    return rows


def pooled(r) -> np.ndarray:
    s, ty = r["seq"], np.array(r["types"])
    f = [s.mean(0), s.max(0), s.min(0)]
    for k in ("function_name", "number"):
        m = ty == k
        f.append(s[m].mean(0) if m.any() else np.zeros(s.shape[1]))
    f.append(np.array([float((ty == k).sum()) for k in TYPES] + [len(s)]))
    return np.concatenate(f)


def source_onehot(rows) -> np.ndarray:
    return np.array([[float(r["src"] == k) for k in SOURCES] for r in rows])


def seq_tensor(rows, mu, sd):
    x = np.zeros((len(rows), T, 8 + len(TYPES)), np.float32)
    mask = np.zeros((len(rows), T), np.float32)
    for i, r in enumerate(rows):
        n = min(T, len(r["seq"]))
        x[i, :n, :8] = (r["seq"][:n] - mu) / sd
        for j, t in enumerate(r["types"][:n]):
            x[i, j, 8 + TYPES.index(t)] = 1.0
        mask[i, :n] = 1.0
    return torch.tensor(x), torch.tensor(mask)


class SeqNet(torch.nn.Module):
    def __init__(self, d_in: int, d_side: int, h: int = 32):
        super().__init__()
        self.emb = torch.nn.Linear(d_in, h)
        self.conv = torch.nn.Conv1d(h, h, 3, padding=1)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(2 * h + d_side, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x, mask, side):
        z = torch.nn.functional.gelu(self.emb(x))
        z = torch.nn.functional.gelu(self.conv(z.transpose(1, 2))).transpose(1, 2)
        m = mask.unsqueeze(-1)
        mean = (z * m).sum(1) / m.sum(1).clamp(min=1)
        mx = (z + (m - 1) * 1e4).max(1).values
        return self.head(torch.cat([mean, mx, side], 1)).squeeze(-1)


class MLP(torch.nn.Module):
    def __init__(self, d: int, h: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, h),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(h, h),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(h, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_net(make, inputs_tr, y_tr, inputs_te, seed, wd=1e-3, epochs=200):
    """Adam on BCE with early stopping on 10% of the training set; returns test scores."""
    g = torch.Generator().manual_seed(seed)
    n = len(y_tr)
    perm = torch.randperm(n, generator=g)
    nv = max(1, n // 10)
    val, tr = perm[:nv], perm[nv:]
    torch.manual_seed(seed)
    net = make()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    y = torch.tensor(y_tr, dtype=torch.float32)
    best, best_state, patience = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        for b in tr[torch.randperm(len(tr), generator=g)].split(64):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                net(*[x[b] for x in inputs_tr]), y[b]
            )
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    net(*[x[val] for x in inputs_tr]), y[val]
                )
            )
        if v < best - 1e-4:
            best, patience = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 15:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(*inputs_te).numpy()


def quad(x):
    i, j = np.triu_indices(x.shape[1])
    return np.hstack([x, x[:, i] * x[:, j]])


def fit_all(rows, tr, te, seed) -> dict[str, np.ndarray]:
    R = [rows[i] for i in tr]
    E = [rows[i] for i in te]
    y = np.array([r["y2"] == 0 and r["y4"] == 1 for r in R], float)
    x12tr, x12te = np.stack([r["x12"] for r in R]), np.stack([r["x12"] for r in E])
    s_tr, s_te = source_onehot(R), source_onehot(E)
    a, b = standardise(x12tr, x12te)
    a, b = np.hstack([a, s_tr]), np.hstack([b, s_te])
    out = {"linear": predict(a, y, b)}
    qa, qb = standardise(quad(a), quad(b))
    out["interactions"] = predict(qa, y, qb, l2=10.0)
    ptr = np.hstack([np.stack([pooled(r) for r in R]), x12tr])
    pte = np.hstack([np.stack([pooled(r) for r in E]), x12te])
    pa, pb = standardise(ptr, pte)
    pa, pb = np.hstack([pa, s_tr]), np.hstack([pb, s_te])
    out["mlp"] = train_net(
        lambda: MLP(pa.shape[1]),
        [torch.tensor(pa, dtype=torch.float32)],
        y,
        [torch.tensor(pb, dtype=torch.float32)],
        seed,
    )
    allseq = np.concatenate([r["seq"] for r in R])
    mu, sd = allseq.mean(0), allseq.std(0) + 1e-6
    xs_tr, m_tr = seq_tensor(R, mu, sd)
    xs_te, m_te = seq_tensor(E, mu, sd)
    side_tr = torch.tensor(a, dtype=torch.float32)  # 12 summary features + benchmark
    side_te = torch.tensor(b, dtype=torch.float32)
    out["sequence"] = train_net(
        lambda: SeqNet(xs_tr.shape[2], side_tr.shape[1]),
        [xs_tr, m_tr, side_tr],
        y,
        [xs_te, m_te, side_te],
        seed,
    )
    return out


def curves(rows) -> dict:
    n = len(rows)
    helps = np.array([r["y2"] == 0 and r["y4"] == 1 for r in rows], float)
    failed2 = np.array([r["y2"] == 0 for r in rows])
    y4 = np.array([r["y4"] for r in rows])
    res: dict = {}
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        preds: dict = {}
        for f in range(FOLDS):
            te = order[f::FOLDS]
            pool = np.setdiff1d(order, te)
            for size in SIZES:
                tr = pool if size is None or size >= len(pool) else rng.choice(pool, size, False)
                for model, p in fit_all(rows, tr, te, seed * 100 + f).items():
                    key = (size or "full", model)
                    preds.setdefault(key, np.zeros(n))[te] = p
            print(json.dumps({"seed": seed, "fold": f}), flush=True)
        for (size, model), p in preds.items():
            res.setdefault(model, {}).setdefault(str(size), []).append(auc(p, helps))
            if size == "full":
                res.setdefault(f"{model}_rescue", {}).setdefault("full", []).append(
                    auc(p[failed2], y4[failed2])
                )
    return {
        model: {
            size: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "seeds": v}
            for size, v in by.items()
        }
        for model, by in res.items()
    }


def hidden_curve(rows) -> dict:
    """MathQA only: PCA-logistic vs MLP on mean loop-2 state and mean h2 − h1."""
    sub = [r for r in rows if r["hidden"] is not None]
    x = []
    for r in sub:
        h = np.load(r["hidden"]).astype(np.float32)
        x.append(np.concatenate([h[:, 1].mean(0), (h[:, 1] - h[:, 0]).mean(0)]))
    x = np.array(x)
    y = np.array([r["y2"] == 0 and r["y4"] == 1 for r in sub], float)
    n = len(sub)
    res: dict = {}
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        preds: dict = {}
        for f in range(FOLDS):
            te = order[f::FOLDS]
            pool = np.setdiff1d(order, te)
            for size in (100, 200, None):
                tr = pool if size is None or size >= len(pool) else rng.choice(pool, size, False)
                a, b = standardise(x[tr], x[te])
                mu = a.mean(0)
                _, _, vt = np.linalg.svd(a - mu, full_matrices=False)
                k = min(32, len(tr) // 4)
                key = size or "full"
                preds.setdefault((key, "pca_logistic"), np.zeros(n))[te] = predict(
                    (a - mu) @ vt[:k].T, y[tr], (b - mu) @ vt[:k].T, l2=10.0
                )
                preds.setdefault((key, "mlp"), np.zeros(n))[te] = train_net(
                    lambda d=a.shape[1]: MLP(d, 32),
                    [torch.tensor(a, dtype=torch.float32)],
                    y[tr],
                    [torch.tensor(b, dtype=torch.float32)],
                    seed * 100 + f,
                    wd=1e-2,
                )
        for (size, model), p in preds.items():
            res.setdefault(model, {}).setdefault(str(size), []).append(auc(p, y))
    return {
        model: {size: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for size, v in by.items()}
        for model, by in res.items()
    }


def main() -> None:
    rows = load()
    helps = np.mean([r["y2"] == 0 and r["y4"] == 1 for r in rows])
    report = {
        "n": len(rows),
        "by_source": {k: sum(r["src"] == k for r in rows) for k in SOURCES},
        "helps_rate": float(helps),
        "curves_auc_4_helps": curves(rows),
        "mathqa_hidden_curves": hidden_curve(rows),
    }
    c = report["curves_auc_4_helps"]
    lin = c["linear"]["full"]["mean"]
    best = max(("interactions", "mlp", "sequence"), key=lambda m: c[m]["full"]["mean"])
    gain = c[best]["full"]["mean"] - lin
    slope = c[best]["full"]["mean"] - c[best]["1000"]["mean"]
    report["decision"] = {
        "best_nonlinear": best,
        "gain_over_linear_at_full": gain,
        "rise_1000_to_full": slope,
        "scale_up_worthwhile": bool(gain >= 0.03 and slope >= 0.01),
    }
    out = Path("results/router_nonlinear")
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=1, default=float)
    (out / "report.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
