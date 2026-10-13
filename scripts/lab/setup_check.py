"""Explicit opt-in: download tool-routing data and weights on ai-lab, then smoke-test MLX.

Run on the lab (downloads allowed):

    uv run --extra lab python scripts/lab/setup_check.py --models Qwen/Qwen3.5-0.8B \
        --generate Qwen/Qwen3.5-0.8B

Steps:
1. Hugging Face identity (needed for the gated xLAM dataset).
2. Datasets: Salesforce/xlam-function-calling-60k (train/validation pool) and the BFCL
   question/answer files (final test only; nothing here reads answers for tuning).
3. Model weights (bf16 safetensors) into the shared HF cache, recording the snapshot SHA.
4. Optional MLX generation: one tool-calling prompt with ``enable_thinking=False``.

Writes ``artifacts/lab/setup_manifest.json`` with revisions, file counts and record counts so
later experiments can pin exactly what was downloaded. Does not train anything.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

XLAM = "Salesforce/xlam-function-calling-60k"
BFCL = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
WEIGHT_PATTERNS = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "tokenizer*"]

# Tool-calling probe; expected call: get_weather(city="Paris", unit="celsius").
PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }
]
PROBE_MESSAGES = [{"role": "user", "content": "What's the weather in Paris in celsius?"}]


def count_json_records(path: Path) -> int:
    """Count records in a JSON array file or a JSON-lines file."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        return len(json.loads(text))
    return sum(1 for line in text.splitlines() if line.strip())


def fetch_datasets(manifest: dict) -> None:
    xlam_dir = Path(snapshot_download(XLAM, repo_type="dataset"))
    xlam_files = sorted(p for p in xlam_dir.rglob("*.json") if p.is_file())
    manifest["datasets"][XLAM] = {
        "revision": xlam_dir.name,
        "files": {p.name: count_json_records(p) for p in xlam_files},
    }

    bfcl_dir = Path(snapshot_download(BFCL, repo_type="dataset", allow_patterns=["*.json"]))
    bfcl_files = sorted(p for p in bfcl_dir.rglob("*.json") if p.is_file())
    counts = {}
    for p in bfcl_files:
        try:
            counts[str(p.relative_to(bfcl_dir))] = count_json_records(p)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            counts[str(p.relative_to(bfcl_dir))] = f"unparsed: {type(exc).__name__}"
    manifest["datasets"][BFCL] = {"revision": bfcl_dir.name, "files": counts}


def fetch_model(repo: str, manifest: dict) -> None:
    start = time.time()
    local = Path(snapshot_download(repo, allow_patterns=WEIGHT_PATTERNS))
    config = json.loads((local / "config.json").read_text())
    text_cfg = config.get("text_config", config)
    layer_types = text_cfg.get("layer_types") or []
    size_gb = sum(p.stat().st_size for p in local.rglob("*.safetensors")) / 1e9
    manifest["models"][repo] = {
        "revision": local.name,
        "safetensors_gb": round(size_gb, 2),
        "num_hidden_layers": text_cfg.get("num_hidden_layers"),
        "full_attention_layers": [i for i, t in enumerate(layer_types) if t == "full_attention"],
        "download_seconds": round(time.time() - start, 1),
    }


def generate_probe(repo: str, manifest: dict) -> None:
    from mlx_lm import generate, load

    start = time.time()
    model, tokenizer = load(repo)
    prompt = tokenizer.apply_chat_template(
        PROBE_MESSAGES,
        tools=PROBE_TOOLS,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    output = generate(model, tokenizer, prompt=prompt, max_tokens=128)
    manifest["generation"][repo] = {
        "prompt_tail": prompt[-400:],
        "output": output,
        "num_layers": len(model.layers),
        "seconds": round(time.time() - start, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--generate", nargs="*", default=[])
    parser.add_argument("--skip-datasets", action="store_true")
    parser.add_argument("--out", default="artifacts/lab/setup_manifest.json")
    args = parser.parse_args()

    import mlx.core as mx

    manifest: dict = {
        "platform": platform.platform(),
        "mlx_device": str(mx.default_device()),
        "hf_user": HfApi().whoami().get("name"),
        "datasets": {},
        "models": {},
        "generation": {},
    }
    print(f"HF user: {manifest['hf_user']}  MLX device: {manifest['mlx_device']}", flush=True)
    if not args.skip_datasets:
        fetch_datasets(manifest)
        print("datasets:", json.dumps(manifest["datasets"], indent=1), flush=True)
    for repo in args.models:
        fetch_model(repo, manifest)
        print(repo, json.dumps(manifest["models"][repo]), flush=True)
    for repo in args.generate:
        generate_probe(repo, manifest)
        print(repo, json.dumps(manifest["generation"][repo], indent=1), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
