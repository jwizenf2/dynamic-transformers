"""These tests observe real operator dimensions, not just zero task gradients."""

import torch

from dht.evaluation.execution import verify_execution
from dht.models.architecture import ArchitectureSpec
from dht.models.transformer import TransformerClassifier


def test_profiler_confirms_physical_skipping(model_config):
    report = verify_execution(
        TransformerClassifier(model_config),
        torch.tensor([[1, 2, 3, 0], [1, 3, 4, 5]]),
        ArchitectureSpec(2, {1: (3, 1)}),
    )
    assert report["sliced_to_full_matmul_ratio"] < 1
    assert report["sliced_to_dense_masked_matmul_ratio"] < 1
    paths = report["paths"]
    # There are exactly two attention bmm calls per retained layer (QK and AV).
    # Batch=2 and heads=4/2 respectively => flattened bmm batches are 8/4.
    batches = [
        row["input_shapes"][0][0]
        for row in paths["sliced"]["operators"]
        if row["operator"] == "aten::bmm"
    ]
    assert batches == [8, 8, 4, 4]
    assert [row["layer"] for row in paths["sliced"]["layers"]] == [0, 1]
