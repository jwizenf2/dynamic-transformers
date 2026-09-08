"""Explicit, versioned AG News preparation; training itself never downloads data.

Paper §6.1 pp.445–446: AG News has 120000/7600 records, four labels, length128.
Tokenizer, validation split, text joining and shuffle are OUR documented protocol,
not established by DHT. Source rows are retained so disjointness is auditable.
The mirror's CSV format is label (1..4), title, description; csv.reader handles quotes.
"""

import csv
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import torch

AG_REVISION = "555590db4219b1243abb1918effd6a7425a2d75f"
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
AG_BASE = (
    f"https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/{AG_REVISION}/data/ag_news_csv"
)
BERT_BASE = f"https://huggingface.co/google-bert/bert-base-uncased/resolve/{BERT_REVISION}"
DATA_VERSION = "ag_news_wordpiece_v1"


def sha256(path: Path) -> str:
    """Stream hashes so a 440MB weight file does not require an extra RAM copy."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url: str, path: Path) -> dict:
    """Fetch an explicitly requested file once; leave completed existing files alone.

    URLs are immutable revisions, and byte counts/SHA256 are saved with preparation.
    A failed transfer leaves only a .partial file, never a usable final file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        partial.replace(path)
    return {"url": url, "sha256": sha256(path), "bytes": path.stat().st_size}


def read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    """Join title + one space + description as a single BERT segment.

    Preserve the CSV text verbatim beyond joining: no speculative HTML unescaping
    or replacement of literal backslash sequences. All segment/type IDs are zero.
    """
    texts, labels = [], []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) != 3 or row[0] not in {"1", "2", "3", "4"}:
                raise ValueError("invalid AG News CSV row")
            texts.append(row[1] + " " + row[2])
            labels.append(int(row[0]) - 1)
    return texts, np.asarray(labels, dtype=np.int64)


def split_indices(labels: np.ndarray, validation_size: int, seed: int) -> dict[str, np.ndarray]:
    """Hold out equal per-class validation counts from TRAIN only, then shuffle.

    This seed is fixed independently of the model/policy seed so repeated model
    runs use the same corpus split and presentation order. Prefix subsets for pilots
    come from the resulting shuffled arrays, never the first class-sorted raw rows.
    """
    if validation_size <= 0 or validation_size % 4 or set(labels.tolist()) != {0, 1, 2, 3}:
        raise ValueError("validation size must be a positive multiple of four")
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in range(4):
        indices = rng.permutation(np.flatnonzero(labels == label))
        count = validation_size // 4
        if len(indices) <= count:
            raise ValueError("validation leaves no training samples for a class")
        validation.extend(indices[:count])
        train.extend(indices[count:])
    return {"train": rng.permutation(train), "validation": rng.permutation(validation)}


def prepare(
    output: Path,
    raw: Path,
    *,
    sequence_length: int = 128,
    validation_size: int = 6000,
    split_seed: int = 42,
) -> dict:
    """Download ~31MB CSV/tokenizer assets, tokenize once, save non-pickle arrays.

    Optional tokenizers is the standard Rust WordPiece implementation; this uses
    its pinned serialized BERT normalizer/pretokenizer/postprocessor rather than
    an approximate homemade tokenizer. No model weights are downloaded here.
    """
    from tokenizers import Tokenizer

    if sequence_length < 2:
        raise ValueError("sequence length must include CLS/SEP")
    output.mkdir(parents=True, exist_ok=False)
    sources = {
        name: download(url, raw / name)
        for name, url in {
            "train.csv": f"{AG_BASE}/train.csv",
            "test.csv": f"{AG_BASE}/test.csv",
            "tokenizer.json": f"{BERT_BASE}/tokenizer.json",
        }.items()
    }
    tokenizer = Tokenizer.from_file(str(raw / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=sequence_length)
    tokenizer.enable_padding(length=sequence_length, pad_id=0, pad_token="[PAD]")
    train_texts, train_labels = read_csv(raw / "train.csv")
    test_texts, test_labels = read_csv(raw / "test.csv")
    if len(train_labels) != 120000 or len(test_labels) != 7600:
        raise ValueError("source sizes differ from the published AG News dataset")
    indices = split_indices(train_labels, validation_size, split_seed)
    indices["test"] = np.arange(len(test_labels))  # original public test order, untouched
    manifest = {
        "version": DATA_VERSION,
        "sources": sources,
        "sequence_length": sequence_length,
        "vocab_size": tokenizer.get_vocab_size(),
        "num_classes": 4,
        "split_seed": split_seed,
        "text_format": "title + single space + description; single segment; no HTML cleanup",
        "splits": {},
    }
    for split, rows in indices.items():
        texts, labels = (
            (test_texts, test_labels) if split == "test" else (train_texts, train_labels)
        )
        tokens = np.empty((len(rows), sequence_length), dtype=np.int32)
        for start in range(0, len(rows), 1024):
            encodings = tokenizer.encode_batch([texts[i] for i in rows[start : start + 1024]])
            tokens[start : start + len(encodings)] = [encoding.ids for encoding in encodings]
        path = output / f"{split}.npz"
        np.savez_compressed(path, tokens=tokens, labels=labels[rows], source_indices=rows)
        manifest["splits"][split] = {
            "rows": len(rows),
            "sha256": sha256(path),
            "source": "test.csv" if split == "test" else "train.csv",
            "class_counts": np.bincount(labels[rows], minlength=4).tolist(),
        }
        print(json.dumps({"prepared_split": split, "rows": len(rows)}), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_prepared(config, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Fail closed on wrong artifacts; no network, fitting, or test-dependent tuning.

    ExperimentConfig pins the MANIFEST hash; that manifest pins all split hashes.
    Resume therefore cannot silently change data while keeping the same config.
    """
    directory = Path(config.data.prepared_directory)
    if sha256(directory / "manifest.json") != config.data.manifest_sha256:
        raise ValueError("prepared dataset manifest hash mismatch")
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest["version"] != config.data.version
        or manifest["sequence_length"] != config.data.sequence_length
        or manifest["vocab_size"] != config.model.vocab_size
        or manifest["num_classes"] != config.model.num_classes
    ):
        raise ValueError("prepared data/model configuration mismatch")
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported split: {split}")
    path = directory / f"{split}.npz"
    if sha256(path) != manifest["splits"][split]["sha256"]:
        raise ValueError("prepared split hash mismatch")
    count = getattr(config.data, f"{split}_size")
    with np.load(path, allow_pickle=False) as arrays:
        tokens, labels = arrays["tokens"], arrays["labels"]
        if len(tokens) < count or len(labels) != len(tokens):
            raise ValueError("requested split size exceeds prepared data")
        if tokens.shape != (len(labels), config.data.sequence_length):
            raise ValueError("invalid prepared token shape")
        if (
            tokens.min() < 0
            or tokens.max() >= config.model.vocab_size
            or labels.min() < 0
            or labels.max() >= config.model.num_classes
        ):
            raise ValueError("invalid prepared token/label range")
        return torch.tensor(tokens[:count], dtype=torch.long), torch.tensor(
            labels[:count], dtype=torch.long
        )
