"""Layer skipping on a tiny random qwen3_5 model (no downloads; skipped where MLX is absent)."""

import pytest

mx = pytest.importorskip("mlx.core")
qwen3_5 = pytest.importorskip("mlx_lm.models.qwen3_5")

from dht.toolcall import mlx_routing  # noqa: E402

CONFIG = {
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 12,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 97,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "full_attention_interval": 4,
        "tie_word_embeddings": True,
    },
}


@pytest.fixture(scope="module")
def model():
    mx.random.seed(0)
    m = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(CONFIG))
    mx.eval(m.parameters())
    return m


def manual_forward(model, tokens, skipped):
    """Reference: the stock loop with skipped layers left out (no cache, causal masks)."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    tm = mlx_routing.text_model(model)
    h = tm.embed_tokens(tokens)
    fa_mask, ssm_mask = create_attention_mask(h, None), create_ssm_mask(h, None)
    for i, layer in enumerate(tm.layers):
        if i not in skipped:
            h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=None)
    return tm.embed_tokens.as_linear(tm.norm(h))


TOKENS = mx.array([[1, 5, 9, 2, 7, 3]])


def test_layout_helpers(model):
    assert mlx_routing.num_groups(model) == 3
    assert mlx_routing.group_layers(model, [1]) == {4, 5, 6, 7}
    assert mlx_routing.attention_layers(model, [1, 2]) == {7, 11}
    assert mlx_routing.deltanet_layers(model, [2]) == {8, 9, 10}


@pytest.mark.parametrize("skipped", [set(), {4, 5, 6, 7}, {7}, {8, 9, 10}, {5, 11}])
def test_skipping_matches_reference_and_restores(model, skipped):
    full = model(TOKENS)
    with mlx_routing.skip_layers(model, skipped):
        routed = model(TOKENS)
    assert mx.allclose(routed, manual_forward(model, TOKENS, skipped), atol=1e-5).item()
    assert mx.array_equal(model(TOKENS), full).item()  # originals restored
    if skipped:
        assert not mx.allclose(routed, full, atol=1e-4).item()


def test_skipping_with_cache_matches_full_recompute(model):
    """Incremental decoding (cache) under a fixed architecture equals recomputing the prefix."""
    skipped = mlx_routing.group_layers(model, [1])
    with mlx_routing.skip_layers(model, skipped):
        cache = model.make_cache()
        model(TOKENS[:, :-1], cache=cache)
        last = model(TOKENS[:, -1:], cache=cache)[:, -1]
        whole = model(TOKENS)[:, -1]
    assert mx.allclose(last, whole, atol=1e-4).item()


def test_group_zero_is_protected_and_restored_on_error(model):
    with pytest.raises(ValueError, match="group 0"):
        with mlx_routing.skip_layers(model, {3}):
            pass
    layers = list(mlx_routing.text_model(model).layers)
    with pytest.raises(RuntimeError):
        with mlx_routing.skip_layers(model, {4}):
            raise RuntimeError
    assert mlx_routing.text_model(model).layers == layers


def test_relative_cost(model):
    assert mlx_routing.relative_cost(model, set()) == 1.0
    one_group = mlx_routing.relative_cost(model, mlx_routing.group_layers(model, [1]))
    assert one_group == pytest.approx(2 / 3)
    attention_only = mlx_routing.relative_cost(model, mlx_routing.attention_layers(model, [1]))
    assert one_group < attention_only < 1


def greedy_tokens(model, prompt, steps):
    """Unbatched greedy reference with a fresh cache."""
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)[:, -1]
    out = []
    for _ in range(steps):
        token = mx.argmax(logits, axis=-1)
        out.append(token.item())
        logits = model(token[:, None], cache=cache)[:, -1]
    return out


@pytest.mark.parametrize("skipped", [set(), {4, 5, 6, 7}, {7, 11}, {8, 9, 10}])
def test_batch_generator_runs_under_skipping(model, skipped):
    """Regression for lab job 2d0c58b9: BatchGenerator read empty caches of skipped layers."""
    from mlx_lm.generate import BatchGenerator

    prompts = [[1, 5, 9, 2, 7, 3], [4, 4, 8], [11, 3, 6, 2]]
    with mlx_routing.skip_layers(model, skipped):
        assert mlx_routing.num_groups(model) == 3  # helpers keep original indexing
        gen = BatchGenerator(model)
        uids = gen.insert(prompts, [5] * len(prompts))
        tokens = {uid: [] for uid in uids}
        while responses := gen.next_generated():
            for r in responses:
                tokens[r.uid].append(r.token)
        gen.close()
        expected = [greedy_tokens(model, p, 5) for p in prompts]
    assert [tokens[u] for u in uids] == expected


def test_skip_layers_is_not_reentrant(model):
    with mlx_routing.skip_layers(model, {4}):
        with pytest.raises(RuntimeError, match="already active"):
            with mlx_routing.skip_layers(model, {5}):
                pass
    assert len(mlx_routing.text_model(model).layers) == 12
