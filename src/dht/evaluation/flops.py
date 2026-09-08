"""Dominant forward matmul FLOPs, NOT complete training or paper-equivalent FLOPs."""

from dht.config import ModelConfig
from dht.models.architecture import ArchitectureSpec


def forward_matmul_flops(
    model: ModelConfig, architecture: ArchitectureSpec, sequence_length: int
) -> int:
    """Two FLOPs per multiply-add; includes prefix, full FFN and first-token classifier.

    Excludes embeddings, normalization, nonlinearities, biases/additions, backward,
    optimizer, routing/calibration and bandits. See docs/reproduction/protocol.md.
    """
    architecture.validate(model)
    if not 1 <= sequence_length <= model.max_sequence_length:
        raise ValueError("unsupported sequence length")
    length, width = sequence_length, model.hidden_size
    total = 2 * width * model.num_classes
    for layer in range(architecture.total_depth):
        active_width = len(architecture.heads_for(layer, model.num_heads)) * (
            width // model.num_heads
        )
        total += 8 * length * width * active_width  # Q/K/V and output projections
        total += 4 * length * length * active_width  # QK and AV
        total += 4 * length * width * model.intermediate_size  # both FFN projections
    return total
