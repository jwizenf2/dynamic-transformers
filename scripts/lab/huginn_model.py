"""Load Huginn-0125 from the vendored code (``third_party/huginn``) and decode at a fixed depth.

    from huginn_model import load_huginn, greedy
    model, tok = load_huginn()                     # bfloat16 on MPS (or CPU)
    ids = greedy(model, tok, prompt_ids, num_steps=32, max_tokens=256, stop_ids={...})

Weights: ``tomg-group-umd/huginn-0125`` at a pinned revision (safetensors, float32 on disk, cast to
bfloat16 as the card recommends); only weight, config and tokenizer files are downloaded, the
modeling code is the audited copy in ``third_party/huginn``.

``StepCache`` replaces the upstream per-token ``HuginnDynamicCache`` for fixed-depth decoding:
one preallocated key/value buffer per cache index (prelude blocks 0–1, recurrent block calls
2, 3, ... = 2 + 4·step + layer, coda blocks −1, −2), appended along the sequence axis. Every
token is decoded with the same ``num_steps`` within a dialogue, so every index has an entry for
every earlier position (asserted). The random initial state of the recurrence is made
reproducible by seeding torch before every forward call (``seed``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party"))
from huginn.raven_config_minimal import RavenConfig  # noqa: E402
from huginn.raven_modeling_minimal import RavenForCausalLM  # noqa: E402

# transformers >= 5 expects tied weights as {target: source}; the vendored class still uses the
# 4.x list form, which 5.17 rejects in ``tie_weights`` (patched here, the vendored file unchanged)
RavenForCausalLM._tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

REPO = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"


class StepCache:
    """Preallocated KV cache keyed by Huginn's block index (fixed number of steps per call).

    Each index gets one buffer [B, heads, capacity, head_dim] filled in place; capacity doubles
    when exceeded, so appending a token copies only that token (``torch.cat`` per token would copy
    the whole cache for each of the 4·num_steps + 4 indices on every decoding step)."""

    def __init__(self, capacity: int = 0) -> None:
        self.capacity = capacity
        self.k: dict[int, torch.Tensor] = {}
        self.v: dict[int, torch.Tensor] = {}
        self.n: dict[int, int] = {}
        self._seen = 0

    def update(self, key_states, value_states, step_idx_tensor, lookup_strategy=None):
        i = int(step_idx_tensor)
        if i == 0:
            self._seen += key_states.shape[-2]
        b, h, t, d = key_states.shape
        if i not in self.k:
            shape = (b, h, max(self.capacity, t + 512), d)  # prompt + room to decode
            self.k[i] = key_states.new_empty(shape)
            self.v[i] = value_states.new_empty(shape)
            self.n[i] = 0
        n = self.n[i]
        if n + t > self.k[i].shape[2]:
            grow = max(n + t, 2 * self.k[i].shape[2])
            for buf in (self.k, self.v):
                new = buf[i].new_empty((b, h, grow, d))
                new[:, :, :n] = buf[i][:, :, :n]
                buf[i] = new
        self.k[i][:, :, n : n + t] = key_states
        self.v[i][:, :, n : n + t] = value_states
        self.n[i] = n + t
        assert self.n[i] == self._seen, f"cache index {i} is missing positions"
        return self.k[i][:, :, : n + t], self.v[i][:, :, : n + t]

    def get_seq_length(self, step_idx: int = 0) -> int:
        return self._seen


def load_huginn(device: str | None = None, dtype=torch.bfloat16):
    from transformers import AutoTokenizer

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    snap = Path(
        snapshot_download(
            REPO,
            revision=REVISION,
            allow_patterns=["*.safetensors", "*.json"],
        )
    )
    import json

    cfg = RavenConfig(**json.loads((snap / "config.json").read_text()))
    with torch.device("meta"):
        model = RavenForCausalLM(cfg)
    state = {}
    for shard in sorted(snap.glob("model-*.safetensors")):
        state.update({k: v.to(dtype) for k, v in load_file(str(shard)).items()})
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    # tied embeddings: lm_head shares wte; freqs_cis is a persistent buffer in the checkpoint
    if "lm_head.weight" in missing:
        model.lm_head.weight = model.transformer.wte.weight
        missing = [m for m in missing if m != "lm_head.weight"]
    assert not missing and not unexpected, (missing, unexpected)
    model = model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(snap)
    return model, tok


@torch.no_grad()
def greedy(
    model,
    ids: list[int],
    cache: StepCache,
    num_steps: int,
    max_tokens: int,
    stop_ids: set[int],
    seed: int = 0,
) -> list[int]:
    """Feed ``ids`` after what ``cache`` holds, then greedy-decode until a stop token."""
    device = next(model.parameters()).device
    steps = int(num_steps)  # a plain int: the vendored forward calls len() on tensors
    pos = cache.get_seq_length()
    out: list[int] = []
    x = torch.tensor([ids], device=device)
    for t in range(max_tokens + 1):
        torch.manual_seed(seed + pos)
        cache_position = torch.arange(pos, pos + x.shape[1], device=device)
        logits = model(
            input_ids=x,
            num_steps=steps,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
            output_details={
                "return_logits": True,
                "return_latents": False,
                "return_head": False,
                "return_stats": False,
            },
        ).logits
        pos += x.shape[1]
        tid = int(logits[0, -1].argmax())
        if tid in stop_ids or t == max_tokens:
            break
        out.append(tid)
        x = torch.tensor([[tid]], device=device)
    return out
