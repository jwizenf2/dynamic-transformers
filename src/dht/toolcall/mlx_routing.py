"""Physically skip layers of an mlx-lm ``qwen3_5`` model (Qwen3.5 / Qwen3.8) for one architecture.

Layout (config ``full_attention_interval`` = 4): layer i is full attention iff (i + 1) % 4 == 0,
otherwise Gated DeltaNet. A *group* g is layers [4g, 4g + 3] = three DeltaNet layers then one
full-attention layer, each followed by its feed-forward block. Qwen3.8-27B has 16 groups.

Invariants (checked):
* Group 0 always runs. mlx-lm builds the DeltaNet mask from layer 0's cache and the attention
  mask from layer 3's cache (``ssm_idx`` / ``fa_idx``); skipping either would desynchronise the
  masks. Group 0 also produces the router features (plan: "Router input").
* Skipped layers are REMOVED from ``layers`` while the context is active, so their attention,
  feed-forward block and residual branch are not executed, their weights are not read, and
  ``make_cache`` builds no cache for them. (An earlier identity stand-in left empty caches that
  mlx-lm's ``BatchGenerator`` later failed to read on the lab, job 2d0c58b9.) Because group 0
  always runs, list positions 0 and ``fa_idx`` still hold the original layers 0 and 3, so the
  mask indices stay valid. The architecture must be fixed for the whole request (prefill and
  every decode step): the plan's "route once per request" rule.
* Layout helpers and costs always refer to ORIGINAL layer indices, also inside the context.

Architectures are given as the set of skipped LAYER indices, so both whole-group skipping and
the DeltaNet-vs-attention variants (sub-question 2) use one mechanism.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import mlx.nn as nn

# id(text model) -> its original layer list while ``skip_layers`` is active. Kept outside the
# module because mlx registers list attributes of a Module as child modules (parameters).
_ACTIVE: dict[int, list[nn.Module]] = {}


def text_model(model: nn.Module) -> nn.Module:
    """The inner ``Qwen3_5TextModel`` (holds ``layers``) of a loaded mlx-lm model."""
    inner = getattr(model, "language_model", model)
    return inner.model


def all_layers(model: nn.Module) -> list[nn.Module]:
    """Original decoder layers, even while ``skip_layers`` is active."""
    tm = text_model(model)
    return _ACTIVE.get(id(tm)) or list(tm.layers)


def interval(model: nn.Module) -> int:
    return text_model(model).fa_idx + 1


def num_groups(model: nn.Module) -> int:
    layers = len(all_layers(model))
    if layers % interval(model):
        raise ValueError("layer count is not a multiple of the full-attention interval")
    return layers // interval(model)


def group_layers(model: nn.Module, groups: Iterable[int]) -> set[int]:
    """Layer indices of whole groups."""
    k = interval(model)
    return {g * k + j for g in groups for j in range(k)}


def attention_layers(model: nn.Module, groups: Iterable[int]) -> set[int]:
    """Only the full-attention layer of each group."""
    k = interval(model)
    return {g * k + k - 1 for g in groups}


def deltanet_layers(model: nn.Module, groups: Iterable[int]) -> set[int]:
    """Only the three DeltaNet layers of each group."""
    k = interval(model)
    return {g * k + j for g in groups for j in range(k - 1)}


def validate_skips(model: nn.Module, skipped: Iterable[int]) -> frozenset[int]:
    skipped = frozenset(skipped)
    if any(not 0 <= i < len(all_layers(model)) for i in skipped):
        raise ValueError("skipped layer index out of range")
    if skipped & group_layers(model, [0]):
        raise ValueError("group 0 must run (mask caches and router features come from it)")
    return skipped


@contextmanager
def skip_layers(model: nn.Module, skipped: Iterable[int]) -> Iterator[frozenset[int]]:
    """Temporarily remove ``skipped`` layers; always restores the originals. Not reentrant."""
    skipped = validate_skips(model, skipped)
    tm = text_model(model)
    if id(tm) in _ACTIVE:
        raise RuntimeError("skip_layers is already active on this model")
    original = list(tm.layers)
    _ACTIVE[id(tm)] = original
    try:
        tm.layers = [layer for i, layer in enumerate(original) if i not in skipped]
        yield skipped
    finally:
        tm.layers = original
        del _ACTIVE[id(tm)]


def layer_parameter_counts(model: nn.Module) -> list[int]:
    """Weights per decoder layer: a per-token matmul-FLOP proxy (2 FLOPs per weight)."""
    from mlx.utils import tree_flatten

    return [sum(v.size for _, v in tree_flatten(layer.parameters())) for layer in all_layers(model)]


def relative_cost(model: nn.Module, skipped: Iterable[int]) -> float:
    """cost(a) / cost(full) over decoder layers, the plan's reward cost term.

    Counts decoder-layer weights only; embeddings, the final norm and the output projection are
    shared by every architecture and excluded. Attention-score and DeltaNet-state FLOPs (which
    depend on sequence length) are also excluded; this is a weight-read/matmul proxy, NOT
    latency, which is measured separately.
    """
    counts = layer_parameter_counts(model)
    skipped = validate_skips(model, skipped)
    return sum(c for i, c in enumerate(counts) if i not in skipped) / sum(counts)
