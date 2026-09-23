"""Shared loaders and plot styling for the analysis notebooks.

Everything here READS existing artifacts: raw CSVs under ``datasets/raw``,
per-sample exports from ``scripts/export_predictions.py`` under
``artifacts/analysis`` and run logs under ``artifacts/runs``. Nothing trains.
"""

import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
# Overridable only so notebooks can be developed against scratch copies; the
# committed notebooks are always executed against the real artifact directories.
RUNS = Path(os.environ.get("DHT_RUNS_DIR", ROOT / "artifacts" / "runs"))
ANALYSIS = Path(os.environ.get("DHT_ANALYSIS_DIR", ROOT / "artifacts" / "analysis"))

# Reference categorical palette, fixed order (dataviz skill, light mode).
# The model identity keeps its color in every chart: fixed=blue, T-DHT=orange,
# F-DHT=aqua. Classes use slots in label order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MODEL_COLORS = {
    "Fixed BERT": SERIES[0],
    "Full BERT": SERIES[0],
    "T-DHT": SERIES[1],
    "F-DHT": SERIES[2],
}
# Ordinal blue ramp for depth 3 -> 12 (light end clears 2:1 on the surface).
DEPTH_COLORS = {3: "#86b6ef", 6: "#3987e5", 9: "#1c5cab", 12: "#0d366b"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#fcfcfb",
)

AG_NEWS_CLASSES = ["World", "Sports", "Business", "Sci/Tech"]


def style() -> None:
    """Thin marks, recessive grid/axes, text in ink tokens rather than series colors."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "legend.frameon": False,
            "legend.labelcolor": INK2,
            "lines.linewidth": 2,
            "font.size": 10,
            "figure.dpi": 110,
            "patch.linewidth": 0,
        }
    )
    pd.set_option("display.max_colwidth", 140)
    pd.set_option("display.width", 160)


# --------------------------------------------------------------------------- datasets
# Zhang, Zhao & LeCun (2015) character-level CNN corpora; the paper's four datasets.
DATASETS = {
    "AG News": {
        "directory": "ag_news",
        "columns": ["label", "title", "description"],
        "classes": AG_NEWS_CLASSES,
        "paper_sequence_length": 128,
        "paper_train": 120_000,
        "paper_test": 7_600,
    },
    "DBpedia": {
        "directory": "dbpedia_csv",
        "columns": ["label", "title", "description"],
        "classes": None,  # read from classes.txt
        "paper_sequence_length": 128,
        "paper_train": 560_000,
        "paper_test": 70_000,
    },
    "Yelp Polarity": {
        "directory": "yelp_review_polarity_csv",
        "columns": ["label", "text"],
        "classes": ["Negative", "Positive"],  # 1-2★ vs 4-5★
        "paper_sequence_length": 300,
        "paper_train": 560_000,
        "paper_test": 38_000,
    },
    "Yelp Full": {
        "directory": "yelp_review_full_csv",
        "columns": ["label", "text"],
        "classes": ["1★", "2★", "3★", "4★", "5★"],
        "paper_sequence_length": 300,
        "paper_train": 650_000,
        "paper_test": 50_000,
    },
}


def class_names(name: str) -> list[str]:
    spec = DATASETS[name]
    if spec["classes"] is not None:
        return spec["classes"]
    return (RAW / spec["directory"] / "classes.txt").read_text().split()


def load_dataset(name: str, split: str) -> pd.DataFrame:
    """Read one raw CSV verbatim; ``text`` joins title + description like the repo.

    Labels become zero based. The CSVs contain literal ``\\n``/``\\"`` escape
    sequences and HTML entities; they are kept as-is (the training pipeline
    also keeps them, see ag_news.read_csv) so statistics match model inputs.
    """
    spec = DATASETS[name]
    path = RAW / spec["directory"] / f"{split}.csv"
    frame = pd.read_csv(
        path,
        header=None,
        names=spec["columns"],
        dtype={"label": np.int64},
        keep_default_na=False,
        quoting=csv.QUOTE_MINIMAL,
        engine="c",
    )
    frame["label"] -= 1
    if "title" in frame:
        frame["text"] = frame["title"] + " " + frame["description"]
    frame["class"] = pd.Categorical.from_codes(frame["label"], class_names(name))
    return frame


def tokenizer():
    """The same pinned BERT-base-uncased WordPiece tokenizer used for training."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(RAW / "ag_news" / "tokenizer.json"))


def wordpiece_lengths(texts: list[str], batch: int = 2048) -> np.ndarray:
    """Untruncated WordPiece length INCLUDING [CLS]/[SEP] (what hits the model)."""
    tok = tokenizer()
    tok.no_truncation()
    tok.no_padding()
    lengths = []
    for start in range(0, len(texts), batch):
        lengths.extend(len(e.ids) for e in tok.encode_batch(texts[start : start + batch]))
    return np.asarray(lengths)


# ---------------------------------------------------------------------- model outputs
EXPORTS = {
    "Fixed BERT": "ag-news-fixed-full-seed7",
    "T-DHT": "ag-news-tree-full-seed7-resume-d015",
    "F-DHT": "ag-news-fdht-full-seed7",
}


def available_models() -> list[str]:
    return [m for m, run in EXPORTS.items() if (ANALYSIS / f"{run}.test.jsonl").exists()]


def load_predictions(model: str) -> pd.DataFrame:
    """Per-sample test export joined with nothing else; one row per test sample."""
    rows = [json.loads(line) for line in (ANALYSIS / f"{EXPORTS[model]}.test.jsonl").open()]
    frame = pd.DataFrame(rows)
    frame["correct"] = frame["prediction"] == frame["label"]
    frame["confidence"] = frame["probabilities"].map(max)
    if frame["trace"].notna().all():
        frame["total_depth"] = frame["trace"].map(lambda t: t["total_depth"])
        frame["group_counts"] = frame["trace"].map(
            lambda t: tuple(g["head_count"] for g in t["groups"])
        )
        frame["group_heads"] = frame["trace"].map(
            lambda t: tuple(tuple(g["selected_heads"]) for g in t["groups"])
        )
    else:
        frame["total_depth"] = 12
        frame["group_counts"] = [(12, 12, 12)] * len(frame)
    return frame


def run_summary(model: str) -> dict:
    return json.loads((RUNS / EXPORTS[model] / "summary.json").read_text())


def run_json(model: str, name: str) -> dict:
    return json.loads((RUNS / EXPORTS[model] / name).read_text())


def training_selections(model: str, every: int = 500) -> pd.DataFrame:
    """Per-sample training-time architecture choices, binned by trained samples.

    The T-DHT run was resumed from its step-8,000 checkpoint (D015), so steps
    <=8,000 come from the failed attempt's log and later steps from the resume.
    Returns one row per trained sample: position, depth, per-group counts, loss.
    """
    run = EXPORTS[model]
    logs = [(RUNS / run / "metrics.jsonl", 0)]
    if model == "T-DHT":
        logs = [
            (RUNS / "ag-news-tree-full-seed7" / "metrics.jsonl", 8000),  # keep <= 8000
            (RUNS / run / "metrics.jsonl", None),
        ]
    records = []
    for path, keep_until in logs:
        with path.open() as stream:
            for line in stream:
                step = json.loads(line)
                if keep_until and step["step"] > keep_until:
                    break
                for sample in step.get("samples", []):
                    arch = sample["architecture"]
                    heads = arch["head_indices"]
                    groups = sorted(int(k) for k in heads)[::3]
                    records.append(
                        {
                            "step": step["step"],
                            "epoch": step["epoch"],
                            "depth": arch["total_depth"],
                            "counts": tuple(len(heads[str(g)]) for g in groups),
                            "loss": sample["loss"],
                        }
                    )
    frame = pd.DataFrame(records)
    frame["position"] = np.arange(len(frame))
    frame["bin"] = frame["position"] // every
    return frame


# --------------------------------------------------------------------------- compute
def relative_matmul_flops(depth: int, counts: tuple[int, ...], seq: int = 128) -> float:
    """Per-sample task-path matmul FLOPs relative to 12L/12H (same scope as flops.py).

    Per layer: QKV 3·2·s·d·(d·k/H) + scores/values 2·2·s²·(d·k/H) + output
    2·s·(d·k/H)·d + FFN 2·2·s·d·4d. Prefix layers (0-2) are always 12 heads.
    Classifier is negligible and ignored here; exact values live in summary.json.
    """
    d, heads = 768, 12

    def layer(k: int) -> float:
        width = d * k / heads
        return 6 * seq * d * width + 4 * seq * seq * width + 2 * seq * width * d + 16 * seq * d * d

    per_layer = [12, 12, 12] + [c for c in counts for _ in range(3)]
    return sum(layer(k) for k in per_layer[:depth]) / (12 * layer(12))
