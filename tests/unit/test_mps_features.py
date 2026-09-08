"""Optional accelerator regression: CPU-only tests never pretend MPS was verified."""

import pytest
import torch

from dht.bandits.features import pooled_features


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_transfer_then_cpu_float64_pooling_matches_cpu():
    hidden = torch.randn(128, 768)
    mask = torch.arange(128) < 71
    expected = pooled_features(hidden, mask, (1, 2), (4, 8, 16))
    actual = pooled_features(hidden.to("mps"), mask.to("mps"), (1, 2), (4, 8, 16))
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
