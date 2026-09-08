import pytest
import torch

from dht.config import ModelConfig
from dht.evaluation.flops import forward_matmul_flops
from dht.evaluation.metrics import classification_metrics
from dht.models.architecture import ArchitectureSpec


def test_metrics_with_missing_class():
    result = classification_metrics(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 1, 1]), 3)
    assert result["accuracy"] == result["micro_f1"] == 0.75
    assert result["macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 3)
    assert result["weighted_f1"] == pytest.approx((2 / 3 + 3 * 4 / 5) / 4)


def test_flops_hand_calculation_includes_prefix_ffn_and_classifier():
    model = ModelConfig(
        total_depth=2,
        prefix_depth=1,
        hidden_size=4,
        num_heads=2,
        intermediate_size=8,
        num_classes=2,
        max_sequence_length=4,
    )
    # sequence=2; prefix: 256 projections + 64 attention + 256 FFN = 576
    # half-head layer: 128 + 32 + 256 = 416; classifier 16
    assert forward_matmul_flops(model, ArchitectureSpec(2, {1: (1,)}), 2) == 1008
    assert forward_matmul_flops(model, ArchitectureSpec(2), 2) == 1168
    assert forward_matmul_flops(model, ArchitectureSpec(1), 2) == 592
