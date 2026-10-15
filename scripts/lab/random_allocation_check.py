"""Does the one-shot xLAM router beat random allocation with the same pass histogram?

    python scripts/lab/random_allocation_check.py <dir with val_0_200, train_0_1000, train_100_1000>

Baseline from 2607.18553, where probe- and gate-based loop allocation on frozen Ouro did not
beat it. The router is the v2 one-shot router (pass-1 features, PCA 8, l2 10) without the
query-text features. For each compute penalty, its per-request choices are shuffled across
requests 2,000 times; the shuffled accuracy is the matched-histogram random baseline.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_analysis_v2 as ra  # noqa: E402
from router_analysis import concat, fit_logistic, load_run, predict  # noqa: E402


def main() -> None:
    root = Path(sys.argv[1])
    tr_ids, ytr, ftr = concat([load_run(root / "train_0_1000"), load_run(root / "train_100_1000")])
    va_ids, yva, fva = load_run(root / "val_0_200")
    qtr, qva = np.zeros((len(tr_ids), 5)), np.zeros((len(va_ids), 5))
    fm = ra.Features(ftr, qtr, 8)
    xtr, xva = fm(ftr, qtr, 1), fm(fva, qva, 1)
    p1 = np.stack([predict(fit_logistic(xtr, ytr[:, i], 10.0), xva) for i in range(4)], 1)
    steps = np.arange(1, 5)
    rng = np.random.default_rng(0)
    n = len(yva)
    rows = np.arange(n)
    print("fixed accuracy", yva.mean(0).round(3), "n", n)
    for lam in (0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2):
        choice = np.argmax(p1 - lam * steps, 1)  # 0-based pass index
        acc = yva[rows, choice].mean()
        rand = np.array([yva[rows, rng.permutation(choice)].mean() for _ in range(2000)])
        print(
            f"lam {lam:.2f}  passes {choice.mean() + 1:.2f}  "
            f"hist {np.bincount(choice, minlength=4)}  router {acc:.3f}  "
            f"random {rand.mean():.3f} (95th {np.percentile(rand, 95):.3f})  "
            f"p {(rand >= acc).mean():.4f}"
        )


if __name__ == "__main__":
    main()
