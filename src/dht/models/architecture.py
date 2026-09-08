"""Ordinary DHT architecture contract (§4.2, pp.442–443; decision D001)."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dht.config import ExperimentConfig, ModelConfig


@dataclass(frozen=True)
class ArchitectureSpec:
    """Total executed depth includes prefix; keys are zero-based layer indices.

    Omitted layers use all heads. One spec applies to a homogeneous batch.
    P-DHT's specialized frozen-layer setup is deliberately outside this contract.
    """

    total_depth: int
    head_indices: Mapping[int, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy and canonicalize head IDs so later caller mutation cannot alter a path."""
        if type(self.total_depth) is not int or self.total_depth < 1:
            raise ValueError("total_depth must be a positive integer")
        copied = {}
        for layer, heads in self.head_indices.items():
            if type(layer) is not int or layer < 0:
                raise ValueError("layer indices must be nonnegative integers")
            indices = tuple(heads)
            if not indices or any(type(h) is not int or h < 0 for h in indices):
                raise ValueError("head indices must be nonempty nonnegative integers")
            if len(indices) != len(set(indices)):
                raise ValueError("head indices must be unique")
            copied[layer] = tuple(sorted(indices))
        object.__setattr__(self, "head_indices", MappingProxyType(copied))

    def heads_for(self, layer: int, num_heads: int) -> tuple[int, ...]:
        """An omitted layer means all original heads, not an empty attention block."""
        return self.head_indices.get(layer, tuple(range(num_heads)))

    def validate(self, model: "ModelConfig") -> None:
        """Check depth bounds, full-prefix protection and one shared subset per group."""
        if not model.prefix_depth <= self.total_depth <= model.total_depth:
            raise ValueError("selected depth must lie between prefix and model depth")
        if (self.total_depth - model.prefix_depth) % model.group_size:
            raise ValueError("selected dynamic depth must contain complete groups")
        full = tuple(range(model.num_heads))
        for layer, heads in self.head_indices.items():
            if layer >= self.total_depth:
                raise ValueError("head specification targets an unexecuted layer")
            if any(h >= model.num_heads for h in heads):
                raise ValueError("head index exceeds num_heads")
            if layer < model.prefix_depth and heads != full:
                raise ValueError("ordinary DHT prefix must remain unpruned")
        for start in range(model.prefix_depth, self.total_depth, model.group_size):
            heads = self.heads_for(start, model.num_heads)
            if any(
                self.heads_for(layer, model.num_heads) != heads
                for layer in range(start + 1, start + model.group_size)
            ):
                raise ValueError("all layers in a dynamic group must use the same heads")

    def to_dict(self) -> dict:
        return {
            "total_depth": self.total_depth,
            "head_indices": {str(k): list(v) for k, v in self.head_indices.items()},
        }

    @classmethod
    def from_config(cls, config: "ExperimentConfig") -> "ArchitectureSpec":
        """Translate human-readable TOML keys to zero-based integer layer positions."""
        if config.architecture is None:
            return cls(config.model.total_depth)
        indices = {}
        for key, heads in config.architecture.head_indices.items():
            if not key.isdecimal() or str(int(key)) != key:
                raise ValueError("head_indices keys must be canonical decimal layer indices")
            indices[int(key)] = heads
        return cls(config.architecture.total_depth, indices)
