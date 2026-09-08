import random

import numpy as np
import pytest
import torch

from dht.runtime import seed_everything, select_device


def test_seeded_streams_repeat():
    def sample():
        streams = seed_everything(7, 13)
        return (
            random.random(),
            np.random.rand(),
            torch.rand(1).item(),
            streams.numpy.random(),
            torch.rand(1, generator=streams.policy).item(),
        )

    assert sample() == sample()


def test_fallback_is_explicit(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert select_device("auto").type == "cpu"
    with pytest.warns(UserWarning, match="falling back"):
        assert select_device("mps").type == "cpu"
    with pytest.raises(RuntimeError, match="unavailable"):
        select_device("cuda", allow_fallback=False)
