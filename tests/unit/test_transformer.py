import pytest
import torch

from dht.models.architecture import ArchitectureSpec
from dht.models.transformer import TransformerClassifier


def test_full_encoder_matches_pytorch_reference(model_config):
    model = TransformerClassifier(model_config).double().eval()
    reference = (
        torch.nn.TransformerEncoderLayer(
            d_model=model_config.hidden_size,
            nhead=model_config.num_heads,
            dim_feedforward=model_config.intermediate_size,
            dropout=0.0,
            activation="gelu",
            layer_norm_eps=model_config.layer_norm_eps,
            batch_first=True,
            norm_first=False,
        )
        .double()
        .eval()
    )
    ours = model.layers[0]
    with torch.no_grad():
        reference.self_attn.in_proj_weight.copy_(
            torch.cat(
                [
                    layer.weight
                    for layer in (ours.attention.query, ours.attention.key, ours.attention.value)
                ]
            )
        )
        reference.self_attn.in_proj_bias.copy_(
            torch.cat(
                [
                    layer.bias
                    for layer in (ours.attention.query, ours.attention.key, ours.attention.value)
                ]
            )
        )
        reference.self_attn.out_proj.load_state_dict(ours.attention.output.state_dict())
        reference.linear1.load_state_dict(ours.ffn[0].state_dict())
        reference.linear2.load_state_dict(ours.ffn[2].state_dict())
        reference.norm1.load_state_dict(ours.attention_norm.state_dict())
        reference.norm2.load_state_dict(ours.output_norm.state_dict())
    hidden = torch.randn(2, 5, model_config.hidden_size, dtype=torch.float64)
    mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
    torch.testing.assert_close(
        ours(hidden, mask, tuple(range(model_config.num_heads))),
        reference(hidden, src_key_padding_mask=~mask),
        rtol=1e-10,
        atol=1e-10,
    )


def test_full_matches_fixed_and_dense(model_config):
    model = TransformerClassifier(model_config).eval()
    tokens = torch.tensor([[1, 3, 5, 0], [1, 2, 6, 7]])
    fixed = model(tokens)
    torch.testing.assert_close(
        fixed, model(tokens, architecture=ArchitectureSpec(3)), rtol=0, atol=0
    )
    torch.testing.assert_close(
        fixed, model(tokens, architecture=ArchitectureSpec(3), dense_masked=True)
    )


def test_depth_stops_and_prefix_is_full(model_config):
    model = TransformerClassifier(model_config)
    tokens = torch.tensor([[1, 2, 3, 4]])
    visited = []
    handles = [
        layer.register_forward_hook(lambda module, args, result, i=i: visited.append(i))
        for i, layer in enumerate(model.layers)
    ]
    output = model(tokens, architecture=ArchitectureSpec(2, {1: (1, 3)}))
    output.square().sum().backward()
    for handle in handles:
        handle.remove()
    assert visited == [0, 1]
    assert output.shape == (1, model_config.num_classes)
    assert all(p.grad is None for p in model.layers[2].parameters())
    grad = model.layers[0].attention.value.weight.grad
    assert all(part.abs().sum() > 0 for part in grad.chunk(model_config.num_heads))
    assert model.layers[1].ffn[0].weight.grad.abs().sum() > 0


def test_padding_cannot_affect_classification(model_config):
    model = TransformerClassifier(model_config).eval()
    mask = torch.tensor([[True, True, False, False]])
    torch.testing.assert_close(
        model(torch.tensor([[1, 2, 3, 4]]), mask),
        model(torch.tensor([[1, 2, 9, 8]]), mask),
    )


@pytest.mark.parametrize(
    "tokens",
    [torch.tensor([[0, 1]]), torch.tensor([[1, 64]]), torch.ones(1, 3)],
)
def test_invalid_input(tokens, model_config):
    with pytest.raises(ValueError):
        TransformerClassifier(model_config)(tokens)
