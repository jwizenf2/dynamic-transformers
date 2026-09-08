"""Explicit opt-in ~440MB pretrained download; does not launch training.

The DHT paper does not identify its checkpoint. This is our named initialization
choice, not a recovered author artifact. Loading only safetensors avoids importing
remote model code or deserializing an untrusted pickle checkpoint.
"""

import argparse
import json
from pathlib import Path

from dht.ag_news import BERT_BASE, BERT_REVISION, download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/pretrained/bert-base-uncased")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {
        name: download(f"{BERT_BASE}/{name}", args.output / name)
        for name in ("config.json", "model.safetensors")
    }
    record = {
        "model": "google-bert/bert-base-uncased",
        "revision": BERT_REVISION,
        "sources": sources,
    }
    (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
