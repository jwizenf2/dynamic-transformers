"""Validated standalone TOML configurations; defaults are development choices."""

import math
import tomllib
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class Config(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, validate_default=True, allow_inf_nan=False
    )


class ModelConfig(Config):
    vocab_size: PositiveInt = 64
    total_depth: PositiveInt = 3
    prefix_depth: PositiveInt = 1
    group_size: PositiveInt = 1
    hidden_size: PositiveInt = 24
    num_heads: PositiveInt = 4
    intermediate_size: PositiveInt = 48
    max_sequence_length: PositiveInt = 16
    num_classes: Annotated[int, Field(strict=True, ge=2)] = 3
    dropout: Annotated[float, Field(ge=0, lt=1)] = 0.0
    layer_norm_eps: Annotated[float, Field(gt=0)] = 1e-5

    @model_validator(mode="after")
    def dimensions(self) -> Self:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.prefix_depth > self.total_depth:
            raise ValueError("prefix_depth cannot exceed total_depth")
        if (self.total_depth - self.prefix_depth) % self.group_size:
            raise ValueError("dynamic depth must contain complete groups")
        return self


class BanditConfig(Config):
    """Numerical controller settings; D005 documents the logistic approximation.

    UCB ridge regularization and exploration are different from reward scales.
    Feature width is derived from pooling, never padded to match a printed typo.
    """

    seed: NonNegativeInt = 17
    ucb_ridge: Annotated[float, Field(gt=0)] = 1.0
    ucb_exploration: Annotated[float, Field(ge=0)] = 1.0
    aggregate_feature_dim: PositiveInt | None = None
    ts_interpretation: Literal["unresolved", "diagonal_laplace"] = "diagonal_laplace"
    ts_prior_precision: Annotated[float, Field(gt=0)] = 1.0
    ts_max_iterations: PositiveInt = 30
    ts_tolerance: Annotated[float, Field(gt=0)] = 1e-7


class DHTConfig(Config):
    """Readable experiment choices for the ordinary tree/integrated DHT method.

    Empty action lists derive small-model defaults. Published lists are supplied
    explicitly in reference configs. Pooling defaults below are smaller than the
    paper's bins to keep CPU tests quick; every bin list is recorded with the run.
    """

    mode: Literal["tree", "integrated"] = "tree"
    depths: tuple[PositiveInt, ...] = ()
    head_counts: tuple[PositiveInt, ...] = ()
    pairs: tuple[tuple[PositiveInt, PositiveInt], ...] = ()
    calibration_samples: PositiveInt = 12
    calibration_sampling: Literal["first", "stratified"] = "first"
    calibration_seed: NonNegativeInt = 42
    sequence_bins: tuple[PositiveInt, ...] = (1, 2)
    depth_embedding_bins: tuple[PositiveInt, ...] = (1, 2, 4)
    head_embedding_bins: tuple[PositiveInt, ...] = (1, 2)
    ts_sequence_bins: tuple[PositiveInt, ...] = (1,)
    ts_embedding_bins: tuple[PositiveInt, ...] = (1, 2)
    normalization_epsilon: Annotated[float, Field(gt=0)] = 1e-6
    reward_epsilon: Annotated[float, Field(gt=0)] = 1e-8
    reward_scale: Annotated[float, Field(gt=0)] = 1.0
    reward_cap: Annotated[float, Field(gt=0)] | None = 100.0
    feedback_rule: Literal["paper_literal", "improvement"] = "paper_literal"
    # D017 engineering choices (no change to the research algorithm's decisions):
    # "batched" runs prefix/groups for a whole batch with per-sample head slices;
    # "per_sample" is the original readable reference path used by earlier runs.
    execution: Literal["per_sample", "batched"] = "per_sample"
    # Samples routed per no-grad prefix pass while queues fill. 1 reproduces the
    # original one-at-a-time routing; larger values route ahead with the SAME
    # (not yet updated) UCB state, i.e. slightly staler contexts, as §5 permits.
    routing_batch_size: PositiveInt = 1
    # D018: "device" pools bandit contexts on the model's device (float32 on MPS) and
    # scores UCB arms for a whole batch at once. Faster, but contexts are no longer
    # bit-identical to "cpu_float64", so rare near-tie decisions can differ.
    context_pooling: Literal["cpu_float64", "device"] = "cpu_float64"

    @model_validator(mode="after")
    def pooling_mode(self) -> Self:
        if self.context_pooling == "device" and self.execution != "batched":
            raise ValueError('context_pooling="device" requires execution="batched"')
        return self

    def actions(self, model: ModelConfig) -> tuple[tuple[int, ...], tuple[int, ...], tuple]:
        """Resolve total depths, counts and pairs in stable, checkpointed order."""
        depths = self.depths or tuple(
            range(model.prefix_depth, model.total_depth + 1, model.group_size)
        )
        counts = self.head_counts or tuple(sorted({max(1, model.num_heads // 2), model.num_heads}))
        pairs = self.pairs or tuple(
            (depth, count)
            for depth in depths
            for count in ((model.num_heads,) if depth == model.prefix_depth else counts)
        )
        return depths, counts, pairs


class DatasetConfig(Config):
    name: Literal["synthetic", "ag_news"] = "synthetic"
    version: Literal["token_modulo_v1", "ag_news_wordpiece_v1"] = "token_modulo_v1"
    prepared_directory: str | None = None
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    train_size: PositiveInt = 24
    validation_size: PositiveInt = 12
    test_size: PositiveInt = 12
    sequence_length: Annotated[int, Field(strict=True, ge=2)] = 8

    @model_validator(mode="after")
    def source(self) -> Self:
        if self.name == "ag_news":
            if (
                self.version != "ag_news_wordpiece_v1"
                or not self.prepared_directory
                or self.manifest_sha256 is None
            ):
                raise ValueError("AG News requires prepared data and a pinned manifest hash")
        elif (
            self.version != "token_modulo_v1"
            or self.prepared_directory is not None
            or self.manifest_sha256 is not None
        ):
            raise ValueError("synthetic data cannot specify AG News artifacts")
        return self


class ArchitectureConfig(Config):
    total_depth: PositiveInt
    # TOML table keys are strings. ArchitectureSpec converts validated decimal keys.
    head_indices: dict[str, tuple[NonNegativeInt, ...]] = Field(default_factory=dict)


class TrainingConfig(Config):
    """Opt-in long-run protocol, distinct from legacy step-limited smoke runs.

    The paper gives 3–6 epochs (§7.3 p.449), but does not specify this schedule or
    checkpoint policy. D013 records these as our reproducible engineering choices.
    LR progress uses TRAINED samples, not route counts or optimizer updates, because
    depth queues create variable-size batches. Validation occurs at every epoch end.
    """

    epochs: PositiveInt = 3
    schedule: Literal["linear_warmup_decay", "constant"] = "linear_warmup_decay"
    warmup_fraction: Annotated[float, Field(ge=0, lt=1)] = 0.1
    checkpoint_every_steps: PositiveInt = 1000
    selection_metric: Literal["macro_f1", "accuracy"] = "macro_f1"


class ExperimentConfig(Config):
    """Entire run protocol; adding [dht] enables learned architecture selection."""

    name: str = "cpu_sanity"
    seed: Annotated[int, Field(strict=True, ge=0, lt=2**32 - 3)] = 7
    device: Literal["auto", "cpu", "mps", "cuda"] = "cpu"
    allow_device_fallback: bool = True
    steps: PositiveInt = 3
    batch_size: PositiveInt = 4
    learning_rate: Annotated[float, Field(gt=0)] = 0.001
    weight_decay: Annotated[float, Field(ge=0)] = 0.0
    cpu_threads: PositiveInt = 1
    # Pilots inspect validation only. Test evaluation is an explicit protocol choice.
    evaluate_test: bool = True
    pretrained_weights: str | None = None
    pretrained_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    log_every: PositiveInt = 25
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DatasetConfig = Field(default_factory=DatasetConfig)
    bandit: BanditConfig = Field(default_factory=BanditConfig)
    architecture: ArchitectureConfig | None = None
    dht: DHTConfig | None = None
    training: TrainingConfig | None = None

    @model_validator(mode="after")
    def compatible(self) -> Self:
        from dht.models.architecture import ArchitectureSpec

        # Legacy checkpoints serialize the old default steps=3. Accept that default
        # with [training], but reject an explicit competing nondefault step budget.
        if self.training is not None and self.steps != 3:
            raise ValueError("[training] uses epochs; remove the legacy steps setting")

        if (self.pretrained_weights is None) != (self.pretrained_sha256 is None):
            raise ValueError("pretrained weights require an explicit SHA256")
        if self.pretrained_weights is not None and self.model.layer_norm_eps != 1e-12:
            raise ValueError("BERT-base weight import requires its LayerNorm epsilon 1e-12")
        if self.pretrained_weights is not None:
            # Tensor shapes cannot reveal the intended number of heads. A 768×768
            # matrix could load with the wrong partition and silently change BERT.
            # This importer is deliberately BERT-base-specific, not a generic hub.
            expected = {
                "vocab_size": 30522,
                "total_depth": 12,
                "hidden_size": 768,
                "num_heads": 12,
                "intermediate_size": 3072,
                "max_sequence_length": 512,
            }
            if any(getattr(self.model, key) != value for key, value in expected.items()):
                raise ValueError(
                    "pretrained importer requires the exact BERT-base encoder dimensions"
                )

        if self.data.sequence_length > self.model.max_sequence_length:
            raise ValueError("dataset sequence length exceeds model capacity")
        if self.data.name == "ag_news" and self.model.num_classes != 4:
            raise ValueError("AG News has four classes")
        if self.model.vocab_size < max(3, self.model.num_classes + 2):
            raise ValueError("vocab_size must cover synthetic token classes")
        if not self.name.strip() or not math.isfinite(self.learning_rate):
            raise ValueError("name and learning_rate must be valid")
        ArchitectureSpec.from_config(self).validate(self.model)
        if self.dht is not None:
            if self.architecture is not None:
                raise ValueError("choose either a fixed architecture or a learned DHT policy")
            if self.bandit.ts_interpretation != "diagonal_laplace":
                raise ValueError("DHT requires the documented diagonal_laplace TS interpretation")
            if self.bandit.aggregate_feature_dim is not None:
                raise ValueError(
                    "DHT derives feature widths from pooling; omit aggregate_feature_dim"
                )
            depths, counts, pairs = self.dht.actions(self.model)
            for options in (depths, counts, pairs):
                if len(set(options)) != len(options):
                    raise ValueError("DHT action lists must not contain duplicates")
            for depth in set(depths) | {pair[0] for pair in pairs}:
                ArchitectureSpec(depth).validate(self.model)
            if any(count > self.model.num_heads for count in counts):
                raise ValueError("DHT head count exceeds available heads")
            for depth, count in pairs:
                if depth not in depths or (
                    count not in counts and depth != self.model.prefix_depth
                ):
                    raise ValueError("pair actions must belong to configured depths/head counts")
                if count > self.model.num_heads or (
                    depth == self.model.prefix_depth and count != self.model.num_heads
                ):
                    raise ValueError("prefix-only pairs must use all heads")
            for bins in (
                self.dht.sequence_bins,
                self.dht.depth_embedding_bins,
                self.dht.head_embedding_bins,
                self.dht.ts_sequence_bins,
                self.dht.ts_embedding_bins,
            ):
                if not bins or len(set(bins)) != len(bins):
                    raise ValueError("pooling bins must be nonempty and unique")
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as stream:
        return ExperimentConfig.model_validate(tomllib.load(stream))
