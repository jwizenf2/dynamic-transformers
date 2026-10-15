"""Does the number of loop passes a request needs grow with its dependency depth?

    python scripts/lab/depth_analysis.py results/ouro_nestful/Ouro-2.6B/nestful_0_360 [...]

Reads one or more ``ouro_nestful.py`` result folders (``examples.jsonl`` rows plus
``features.npz``; partial runs are fine: only examples with all pass counts are used) and prints
JSON with:

* accuracy (executed answer correct) and exact match per pass count, and the per-request oracle;
* per depth bucket: accuracy per pass count, oracle, and the mean smallest correct pass count on
  solved requests;
* confound control. Depth rises with the number of calls and the prompt length, so each effect
  is estimated jointly:
  - logistic regression of "correct at pass k" on standardized depth, n_calls and
    log(prompt_tokens), for each k (IRLS, standard errors from the inverse Fisher information);
  - least squares of "smallest correct pass" on the same three, on solved requests only;
* the most common failure reasons per pass count.

Analysis only (CPU, no model). Small samples: read the standard errors, not just the signs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load(dirs: list[Path]):
    rows: dict[str, dict[int, dict]] = defaultdict(dict)
    feats: dict[str, dict] = {}
    for d in dirs:
        for line in (d / "examples.jsonl").read_text().splitlines():
            r = json.loads(line)
            rows[r["id"]][r["steps"]] = r
        f = np.load(d / "features.npz")
        for i, ex_id in enumerate(f["id"]):
            feats[str(ex_id)] = {
                "depth": int(f["depth"][i]),
                "n_calls": int(f["n_calls"][i]),
                "prompt_tokens": int(f["prompt_tokens"][i]),
            }
    steps = sorted({s for r in rows.values() for s in r})
    ids = [i for i, r in rows.items() if set(r) == set(steps)]
    for i in ids:  # features.npz is saved every 30 examples; rows carry depth as a fallback
        if i not in feats:
            feats[i] = None
    ids = [i for i in ids if feats[i] is not None]
    return ids, steps, rows, feats


def logistic(x: np.ndarray, y: np.ndarray, iters: int = 50, ridge: float = 1e-3):
    """IRLS logistic regression; returns (coefficients, standard errors). x includes a bias."""
    w = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-x @ w))
        h = x.T @ (x * (p * (1 - p))[:, None]) + ridge * np.eye(x.shape[1])
        step = np.linalg.solve(h, x.T @ (y - p) - ridge * w)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-x @ w))
    h = x.T @ (x * (p * (1 - p))[:, None]) + ridge * np.eye(x.shape[1])
    return w, np.sqrt(np.diag(np.linalg.inv(h)))


def ols(x: np.ndarray, y: np.ndarray):
    w, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ w
    dof = max(len(y) - x.shape[1], 1)
    cov = (resid @ resid / dof) * np.linalg.inv(x.T @ x)
    return w, np.sqrt(np.diag(cov))


def bucket(depth: int) -> str:
    return str(depth) if depth < 7 else "7+"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dirs", nargs="+", type=Path)
    args = parser.parse_args()
    ids, steps, rows, feats = load(args.dirs)
    y = np.array([[rows[i][s]["correct"] for s in steps] for i in ids], dtype=float)
    ex = np.array([[rows[i][s]["exact"] for s in steps] for i in ids], dtype=float)
    depth = np.array([feats[i]["depth"] for i in ids])
    calls = np.array([feats[i]["n_calls"] for i in ids])
    ptok = np.array([feats[i]["prompt_tokens"] for i in ids])
    solved = y.any(1)
    smallest = np.where(solved, y.argmax(1) + 1, 0)

    def table(mask: np.ndarray) -> dict:
        s = smallest[mask & solved]
        return {
            "n": int(mask.sum()),
            "accuracy_by_steps": {
                k: round(float(y[mask, j].mean()), 3) for j, k in enumerate(steps)
            },
            "oracle": round(float(solved[mask].mean()), 3),
            "solved": int(len(s)),
            "smallest_correct_histogram": {str(k): int((s == k).sum()) for k in steps},
            "mean_smallest_correct": round(float(s.mean()), 3) if len(s) else None,
        }

    out: dict = {"n": len(ids), "steps": steps}
    out["all"] = table(np.ones(len(ids), bool))
    out["exact_by_steps"] = {k: round(float(ex[:, j].mean()), 3) for j, k in enumerate(steps)}
    out["by_depth"] = {
        b: table(np.array([bucket(d) == b for d in depth]))
        for b in sorted({bucket(d) for d in depth}, key=lambda b: int(b.rstrip("+")))
    }
    out["corr_depth_ncalls"] = round(float(np.corrcoef(depth, calls)[0, 1]), 3)
    out["corr_depth_logprompt"] = round(float(np.corrcoef(depth, np.log(ptok))[0, 1]), 3)

    raw = np.c_[depth, calls, np.log(ptok)].astype(float)
    z = (raw - raw.mean(0)) / raw.std(0)
    x = np.c_[np.ones(len(ids)), z]
    names = ["depth", "n_calls", "log_prompt_tokens"]
    out["logistic_correct_at_k"] = {}
    for j, k in enumerate(steps):
        if 0 < y[:, j].sum() < len(ids):
            w, se = logistic(x, y[:, j])
            out["logistic_correct_at_k"][k] = {
                n: [round(float(w[i + 1]), 3), round(float(se[i + 1]), 3)]
                for i, n in enumerate(names)
            }
    if solved.sum() > 5:
        w, se = ols(x[solved], smallest[solved].astype(float))
        out["ols_smallest_correct_on_solved"] = {
            n: [round(float(w[i + 1]), 3), round(float(se[i + 1]), 3)] for i, n in enumerate(names)
        }
        out["ols_note"] = "coefficients per 1 SD of each predictor; [estimate, standard error]"
    out["failure_reasons"] = {
        k: Counter(
            str(rows[i][k]["reason"]).split(":")[-1].strip()[:40]
            for i in ids
            if not rows[i][k]["correct"]
        ).most_common(5)
        for k in steps
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
