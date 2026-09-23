"""Contexts and train-only arm calibration: §§4.2,7.1–7.2, pp.442–443,449.

The paper's printed feature widths contradict its pooling bins. We compute widths
from explicit bins. Calibration executes each configured head count plus a full
path on a bounded training subset, averaging representations into arm descriptors.
This extra computation is real and is measured by the trainer (decision D006).

Normalization uses running moments while calibrating, then freezes them. This
documented stability choice avoids changing the coordinate system of accumulated
ridge statistics as each training batch arrives.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from dht.config import ExperimentConfig
from dht.models.transformer import TransformerClassifier


def pooled_features(
    hidden: Tensor, mask: Tensor, sequence_bins: tuple[int, ...], embedding_bins: tuple[int, ...]
) -> Tensor:
    """Pool one sample [sequence,hidden] into sum(seq_bins)*sum(emb_bins) features.

    Padding tokens are removed before pooling so they do not dilute the context.
    Adaptive average pooling supports short inputs; when bins exceed an axis length,
    regions overlap. That boundary behavior is our explicit interpretation.
    """
    if hidden.ndim != 2 or mask.shape != hidden.shape[:1] or mask.dtype != torch.bool:
        raise ValueError("pooling requires one [sequence,hidden] sample and boolean token mask")
    if not mask.any() or not sequence_bins or not embedding_bins:
        raise ValueError("pooling requires valid tokens and nonempty bin lists")
    if any(type(n) is not int or n < 1 for n in (*sequence_bins, *embedding_bins)):
        raise ValueError("pooling bin sizes must be positive integers")
    # Transfer in the ORIGINAL dtype first, then index/cast on CPU. On the tested
    # PyTorch 2.14/MPS stack, a combined .to(cpu, float64) produced nonfinite values
    # from finite activations. MPS itself does not support float64. Separating the
    # operations preserves the actual values; it is NOT NaN replacement or a
    # change to the bandit's float64 mathematics. See decision D011.
    values = hidden.detach().cpu()[mask.cpu()].double()[None, None]
    if not torch.isfinite(values).all():
        raise ValueError("context contains nonfinite activations")
    return torch.cat(
        [
            F.adaptive_avg_pool2d(values, (s, e)).flatten()
            for s in sequence_bins
            for e in embedding_bins
        ]
    )


def pooled_features_batch(
    hidden: Tensor, mask: Tensor, sequence_bins: tuple[int, ...], embedding_bins: tuple[int, ...]
) -> Tensor:
    """pooled_features() for a whole batch, ON THE MODEL'S DEVICE (D018).

    hidden [batch,sequence,width], mask [batch,sequence] with padding only at the end
    (the tokenizer pads right). Reproduces adaptive average pooling over each sample's
    valid tokens: sequence bin i covers [floor(i·n/s), ceil((i+1)·n/s)), computed from
    one prefix sum so every sample's own length is respected in parallel. Embedding
    bins then average equal contiguous chunks. Output [batch, Σs·Σe] in hidden's dtype.

    Runs in the device dtype (float32 on MPS, which has no float64), so values match
    the float64 CPU path to ~1e-6 relative, not bit-for-bit. Only this small result
    crosses to the CPU; the bandit math afterwards stays float64.
    """
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("batched pooling needs [batch,sequence,width] and a boolean mask")
    lengths = mask.sum(1)
    positions = torch.arange(mask.shape[1], device=mask.device)
    if (lengths < 1).any() or not torch.equal(mask, positions[None] < lengths[:, None]):
        raise ValueError("batched pooling requires right padding and one valid token per row")
    values = hidden.detach()
    batch, _, width = values.shape
    prefix = torch.cat([values.new_zeros(batch, 1, width), values.cumsum(1)], dim=1)
    pieces = []
    for s in sequence_bins:
        index = torch.arange(s, device=mask.device)
        start = (index[None] * lengths[:, None]) // s
        end = -((-(index[None] + 1) * lengths[:, None]) // s)
        upper = prefix.gather(1, end[..., None].expand(-1, -1, width))
        lower = prefix.gather(1, start[..., None].expand(-1, -1, width))
        segments = (upper - lower) / (end - start)[..., None].to(values.dtype)
        for e in embedding_bins:
            pooled = F.adaptive_avg_pool1d(segments.reshape(batch * s, 1, width), e)
            pieces.append(pooled.reshape(batch, s * e))
    return torch.cat(pieces, dim=1)


@dataclass
class FrozenStandardizer:
    """Train-calibrated mean/std; constant coordinates use scale one, never zero."""

    mean: Tensor
    scale: Tensor

    @classmethod
    def fit(cls, observations: list[Tensor], epsilon: float) -> "FrozenStandardizer":
        # Welford updates make the intended running-statistics calculation explicit.
        mean = torch.zeros_like(observations[0])
        m2 = torch.zeros_like(mean)
        for count, row in enumerate(observations, start=1):
            delta = row - mean
            mean += delta / count
            m2 += delta * (row - mean)
        variance = m2 / count  # Population variance; normalization is not an estimator test.
        scale = torch.where(variance > epsilon, variance.sqrt(), torch.ones_like(variance))
        return cls(mean, scale)

    def transform(self, value: Tensor) -> Tensor:
        """Apply a fixed coordinate system; evaluation does not update these moments."""
        return (value - self.mean) / self.scale

    def state_dict(self) -> dict:
        """Save calibration state so resumption never recalibrates a changed model."""
        return {"mean": self.mean.clone(), "scale": self.scale.clone()}


class FeatureBank:
    """Data-derived descriptors for depth, count, pair and positional head arms.

    Arm tensors are averages, not action one-hots. Head-count aggregates append a
    group one-hot as described in §4.2.1 p.443. TS concatenates a sample context
    with a positional head descriptor; its width is explicit rather than forced
    to the unexplained printed value 13.
    """

    def __init__(self, config: ExperimentConfig):
        if config.dht is None:
            raise ValueError("FeatureBank requires DHT configuration")
        self.config, self.options = config, config.dht
        self.depths, self.counts, self.pairs = config.dht.actions(config.model)
        self.groups = tuple(
            range(config.model.prefix_depth, config.model.total_depth, config.model.group_size)
        )
        self.normalizers: dict[str, FrozenStandardizer] = {}
        self.depth_arms: Tensor
        self.head_arms: Tensor
        self.pair_arms: Tensor
        self.ts_arms: Tensor

    def raw(self, hidden: Tensor, mask: Tensor, kind: str) -> Tensor:
        """Extract a detached context for one decision family."""
        if kind == "depth":
            seq, emb = self.options.sequence_bins, self.options.depth_embedding_bins
        elif kind == "head":
            seq, emb = self.options.sequence_bins, self.options.head_embedding_bins
        elif kind == "ts":
            seq, emb = self.options.ts_sequence_bins, self.options.ts_embedding_bins
        else:
            raise ValueError(f"unknown context kind: {kind}")
        return pooled_features(hidden[0], mask[0], seq, emb)

    @torch.no_grad()
    def calibrate(
        self,
        model: TransformerClassifier,
        tokens: Tensor,
        device: torch.device,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Build arm means from TRAIN tokens with dropout disabled, exactly once.

        Labels are not used. Each sample executes its full prefix once, then a
        separate path for each allowed head count (plus full heads). Random head
        identities use a separate fixed generator. This exhaustive count coverage
        replaces the paper's incompletely specified random/even count assignment.
        """
        if len(tokens) == 0:
            raise ValueError("calibration requires training samples")
        model_mode = model.training
        model.eval()
        depth_rows = {depth: [] for depth in self.depths}
        pair_rows = {pair: [] for pair in self.pairs}
        head_rows = {(g, h): [] for g in self.groups for h in self.counts}
        ts_rows = {(g, h): [] for g in self.groups for h in range(self.config.model.num_heads)}
        moments = {"depth": [], "head": [], "ts": []}
        generator = torch.Generator().manual_seed(self.config.bandit.seed)
        full_count = self.config.model.num_heads
        try:
            total = min(len(tokens), self.options.calibration_samples)
            for sample_index, sample in enumerate(tokens[:total], start=1):
                prefix, mask = model.prefix(sample[None].to(device))
                prefix_depth = self.config.model.prefix_depth
                prefix_raw = self.raw(prefix, mask, "depth")
                moments["depth"].append(prefix_raw)
                if prefix_depth in depth_rows:
                    depth_rows[prefix_depth].append(prefix_raw)
                if (prefix_depth, full_count) in pair_rows:
                    pair_rows[(prefix_depth, full_count)].append(prefix_raw)
                for count in sorted(set(self.counts) | {full_count}):
                    hidden = prefix
                    for start in self.groups:
                        moments["head"].append(self.raw(hidden, mask, "head"))
                        moments["ts"].append(self.raw(hidden, mask, "ts"))
                        heads = tuple(
                            sorted(torch.randperm(full_count, generator=generator)[:count].tolist())
                        )
                        end = start + self.config.model.group_size
                        for layer in model.layers[start:end]:
                            hidden = layer(hidden, mask, heads)
                        depth_raw = self.raw(hidden, mask, "depth")
                        moments["depth"].append(depth_raw)
                        if count == full_count and end in depth_rows:
                            depth_rows[end].append(depth_raw)
                        if (end, count) in pair_rows:
                            pair_rows[(end, count)].append(depth_raw)
                        head_raw = self.raw(hidden, mask, "head")
                        moments["head"].append(head_raw)
                        if count in self.counts:
                            head_rows[(start, count)].append(head_raw)
                        # Split by original hidden-position chunks as described on p.443.
                        # These are representation descriptors, NOT isolated head outputs.
                        for head, chunk in enumerate(hidden.chunk(full_count, dim=-1)):
                            raw = self.raw(chunk, mask, "ts")
                            ts_rows[(start, head)].append(raw)
                            moments["ts"].append(raw)
                if progress and (
                    sample_index == 1 or sample_index % 64 == 0 or sample_index == total
                ):
                    progress(sample_index, total)
        finally:
            model.train(model_mode)
        # Prefix-only networks never make head decisions; still supply valid empty banks.
        for kind in ("head", "ts"):
            if not moments[kind]:
                moments[kind].append(self.raw(prefix, mask, kind))
        self.normalizers = {
            kind: FrozenStandardizer.fit(rows, self.options.normalization_epsilon)
            for kind, rows in moments.items()
        }

        def average(rows: list[Tensor], kind: str) -> Tensor:
            if not rows:
                raise ValueError("calibration did not cover a configured action")
            return self.normalizers[kind].transform(torch.stack(rows).mean(0))

        self.depth_arms = torch.stack([average(depth_rows[d], "depth") for d in self.depths])
        self.pair_arms = torch.stack([average(pair_rows[p], "depth") for p in self.pairs])
        self.head_arms = (
            torch.stack(
                [
                    torch.stack([average(head_rows[(g, h)], "head") for h in self.counts])
                    for g in self.groups
                ]
            )
            if self.groups
            else torch.empty(0, len(self.counts), self.raw(prefix, mask, "head").numel())
        )
        self.ts_arms = (
            torch.stack(
                [
                    torch.stack([average(ts_rows[(g, h)], "ts") for h in range(full_count)])
                    for g in self.groups
                ]
            )
            if self.groups
            else torch.empty(0, full_count, self.raw(prefix, mask, "ts").numel())
        )

    def aggregate(self, hidden: Tensor, mask: Tensor, kind: str, group: int = 0) -> Tensor:
        """Return [candidate_arms,p] detached decision-time features."""
        family = "depth" if kind in ("depth", "pair") else kind
        context = self.normalizers[family].transform(self.raw(hidden, mask, family))
        if kind == "depth":
            arms = self.depth_arms
        elif kind == "pair":
            arms = self.pair_arms
        elif kind == "head":
            arms = self.head_arms[group]
        elif kind == "ts":
            arms = self.ts_arms[group]
        else:
            raise ValueError("unknown aggregate kind")
        result = torch.cat([context.expand(len(arms), -1), arms], dim=1)
        if kind == "head":
            marker = F.one_hot(torch.tensor(group), len(self.groups)).double()
            result = torch.cat([result, marker.expand(len(arms), -1)], dim=1)
        return result.clone()

    def aggregate_many(
        self, hidden: Tensor, mask: Tensor, kind: str, group: int = 0
    ) -> list[Tensor]:
        """Per-sample aggregate() for a batch; pooling location set by context_pooling."""
        if self.options.context_pooling == "device":
            return self._aggregate_many_device(hidden, mask, kind, group)
        return self._aggregate_many_cpu(hidden, mask, kind, group)

    def _aggregate_many_device(
        self, hidden: Tensor, mask: Tensor, kind: str, group: int
    ) -> list[Tensor]:
        """D018: pool on the device, move only [batch,features] to the CPU, then float64.

        The standardizer, arm descriptors and group markers are identical to aggregate().
        """
        family = "depth" if kind in ("depth", "pair") else kind
        bins = {
            "depth": (self.options.sequence_bins, self.options.depth_embedding_bins),
            "head": (self.options.sequence_bins, self.options.head_embedding_bins),
            "ts": (self.options.ts_sequence_bins, self.options.ts_embedding_bins),
        }[family]
        pooled = pooled_features_batch(hidden, mask, *bins).cpu().double()
        if not torch.isfinite(pooled).all():
            raise ValueError("context contains nonfinite activations")
        contexts = self.normalizers[family].transform(pooled)
        arms = {
            "depth": getattr(self, "depth_arms", None),
            "pair": getattr(self, "pair_arms", None),
            "head": self.head_arms[group] if kind == "head" else None,
            "ts": self.ts_arms[group] if kind == "ts" else None,
        }[kind]
        results = []
        for context in contexts:
            result = torch.cat([context.expand(len(arms), -1), arms], dim=1)
            if kind == "head":
                marker = F.one_hot(torch.tensor(group), len(self.groups)).double()
                result = torch.cat([result, marker.expand(len(arms), -1)], dim=1)
            results.append(result)
        return results

    def _aggregate_many_cpu(
        self, hidden: Tensor, mask: Tensor, kind: str, group: int
    ) -> list[Tensor]:
        """Per-sample aggregate() for a [batch,sequence,hidden] tensor.

        Batched DHT transfers the whole batch to the CPU ONCE (original dtype, D011)
        and then pools each sample there. Per-sample calls would force one
        device synchronization per sample per decision. Values are bit-identical
        to calling aggregate() on each sample separately.
        """
        host_hidden, host_mask = hidden.detach().cpu(), mask.cpu()
        return [
            self.aggregate(host_hidden[row : row + 1], host_mask[row : row + 1], kind, group)
            for row in range(len(host_hidden))
        ]

    def state_dict(self) -> dict:
        """Arm descriptors and frozen moments belong to the policy checkpoint."""
        return {
            "depth_arms": self.depth_arms.clone(),
            "pair_arms": self.pair_arms.clone(),
            "head_arms": self.head_arms.clone(),
            "ts_arms": self.ts_arms.clone(),
            "normalizers": {key: norm.state_dict() for key, norm in self.normalizers.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore an already calibrated bank; caller validates experiment identity."""
        for name in ("depth_arms", "pair_arms", "head_arms", "ts_arms"):
            value = state[name].cpu().double()
            if not torch.isfinite(value).all():
                raise ValueError("invalid calibration checkpoint")
            setattr(self, name, value.clone())
        self.normalizers = {
            key: FrozenStandardizer(value["mean"].clone(), value["scale"].clone())
            for key, value in state["normalizers"].items()
        }
