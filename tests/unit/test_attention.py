import pytest
import torch
from torch.nn import functional as F

from dht.models.attention import SelectableAttention


@pytest.mark.parametrize("dimension", [0, 1])
def test_unique_gather_derivative_matches_finite_differences(dimension):
    from dht.models.attention import _UniqueIndexSelect

    value = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    indices = torch.tensor([3, 1])
    assert torch.autograd.gradcheck(
        lambda x: _UniqueIndexSelect.apply(x, dimension, indices), (value,)
    )
    assert torch.autograd.gradgradcheck(
        lambda x: _UniqueIndexSelect.apply(x, dimension, indices), (value,)
    )


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_sliced_backward_matches_dense_reference_with_strict_determinism(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        attention = SelectableAttention(24, 4).to(device)
        hidden = torch.randn(2, 8, 24, device=device, requires_grad=True)
        mask = torch.ones(2, 8, device=device, dtype=torch.bool)
        output = attention(hidden, mask, (3, 1))
        output.square().sum().backward()
        gradients = {key: p.grad.cpu().clone() for key, p in attention.named_parameters()}
        input_gradient = hidden.grad.cpu().clone()
        attention.zero_grad(set_to_none=True)
        hidden.grad = None
        dense = attention(hidden, mask, (3, 1), dense_masked=True)
        dense.square().sum().backward()
        torch.testing.assert_close(output, dense, rtol=2e-5, atol=2e-5)
        for key, parameter in attention.named_parameters():
            torch.testing.assert_close(gradients[key], parameter.grad.cpu(), rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(input_gradient, hidden.grad.cpu(), rtol=2e-4, atol=2e-4)
    finally:
        torch.use_deterministic_algorithms(previous)


def test_sliced_matches_dense_mask_and_preserves_noncontiguous_positions():
    attention = SelectableAttention(12, 3).double()
    hidden = torch.randn(2, 5, 12, dtype=torch.float64)
    mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
    torch.testing.assert_close(
        attention(hidden, mask, (2, 0)), attention(hidden, mask, (2, 0), dense_masked=True)
    )
    # Independent per-head sum with the original output-column positions.
    expected = attention.output.bias.expand(2, 5, -1).clone()
    for head in (2, 0):
        head_slice = slice(head * 4, (head + 1) * 4)
        q, k, v = [
            F.linear(hidden, layer.weight[head_slice], layer.bias[head_slice])
            for layer in (attention.query, attention.key, attention.value)
        ]
        scores = (q @ k.transpose(-2, -1)) / 2
        scores = scores.masked_fill(~mask[:, None, :], -torch.inf)
        output = scores.softmax(-1) @ v
        expected += F.linear(output, attention.output.weight[:, head_slice])
    torch.testing.assert_close(attention(hidden, mask, (2, 0)), expected)


def test_active_gradients_and_inactive_parameter_perturbation():
    attention = SelectableAttention(12, 3).double()
    hidden = torch.randn(2, 5, 12, dtype=torch.float64)
    mask = torch.ones(2, 5, dtype=torch.bool)
    output = attention(hidden, mask, (0, 2))
    output.square().sum().backward()
    for layer in (attention.query, attention.key, attention.value):
        assert layer.weight.grad[4:8].count_nonzero() == 0
        assert layer.bias.grad[4:8].count_nonzero() == 0
        assert layer.weight.grad[:4].abs().sum() > 0
        assert layer.weight.grad[8:].abs().sum() > 0
    assert attention.output.weight.grad[:, 4:8].count_nonzero() == 0
    assert attention.output.weight.grad[:, :4].abs().sum() > 0
    with torch.no_grad():
        for layer in (attention.query, attention.key, attention.value):
            layer.weight[4:8].add_(100)
            layer.bias[4:8].add_(100)
        attention.output.weight[:, 4:8].add_(100)
    torch.testing.assert_close(attention(hidden, mask, (0, 2)), output)


def test_projection_matmuls_really_have_selected_width(monkeypatch):
    attention = SelectableAttention(12, 3)
    shapes = []
    original = F.linear

    def record(inputs, weight, bias=None):
        shapes.append(tuple(weight.shape))
        return original(inputs, weight, bias)

    monkeypatch.setattr(F, "linear", record)
    attention(torch.randn(1, 4, 12), torch.ones(1, 4, dtype=torch.bool), (2,))
    assert shapes == [(4, 12), (4, 12), (4, 12), (12, 4)]


@pytest.mark.parametrize("heads", [(), (0, 0), (-1,), (3,)])
def test_invalid_heads(heads):
    with pytest.raises(ValueError):
        SelectableAttention(12, 3)(torch.randn(1, 4, 12), torch.ones(1, 4, dtype=torch.bool), heads)


def test_fully_masked_input_rejected():
    with pytest.raises(ValueError, match="at least one"):
        SelectableAttention(12, 3)(torch.randn(1, 4, 12), torch.zeros(1, 4, dtype=torch.bool))
