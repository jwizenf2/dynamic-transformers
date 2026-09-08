"""Explicit preparation command; never invoked implicitly by training or tests."""

import argparse
import json
from pathlib import Path

from dht.ag_news import prepare, sha256

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/ag_news_wordpiece_v1"))
    parser.add_argument("--raw", type=Path, default=Path("datasets/raw/ag_news"))
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--validation-size", type=int, default=6000)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    prepare(
        args.output,
        args.raw,
        sequence_length=args.sequence_length,
        validation_size=args.validation_size,
        split_seed=args.split_seed,
    )
    print(json.dumps({"manifest_sha256": sha256(args.output / "manifest.json")}))
