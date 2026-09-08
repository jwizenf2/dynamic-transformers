"""Tiny local fixtures only; these tests never download news, tokenizers or weights."""

import json

import numpy as np
import pytest
import torch

from dht.ag_news import DATA_VERSION, load_prepared, read_csv, sha256, split_indices
from dht.config import ExperimentConfig
from dht.models.pretrained import bert_tensor_mapping, import_bert_encoder
from dht.models.transformer import TransformerClassifier


def test_csv_quoting_and_labels(tmp_path):
    path = tmp_path / "news.csv"
    path.write_text('1,"Quoted ""title"", here","Description"\n4,"Sport?","No, technology."\n')
    texts, labels = read_csv(path)
    assert texts == ['Quoted "title", here Description', "Sport? No, technology."]
    assert labels.tolist() == [0, 3]


def test_pretrained_config_cannot_silently_repartition_bert_heads():
    from dht.config import load_config

    original = load_config("configs/experiment/ag_news_fixed_pilot.toml").model_dump()
    original["model"]["num_heads"] = 16  # Same parameter shapes, WRONG pretrained head meaning.
    with pytest.raises(ValueError, match="BERT-base encoder dimensions"):
        ExperimentConfig.model_validate(original)


def test_stratified_train_validation_are_disjoint_and_reproducible():
    labels = np.repeat(np.arange(4), 20)
    first = split_indices(labels, 16, 42)
    second = split_indices(labels, 16, 42)
    assert not set(first["train"]) & set(first["validation"])
    assert sorted([*first["train"], *first["validation"]]) == list(range(80))
    assert np.bincount(labels[first["validation"]]).tolist() == [4] * 4
    for split in first:
        np.testing.assert_array_equal(first[split], second[split])


def prepared_config(tmp_path, *, dht=False):
    """Construct hash-pinned local data exercising the same path as a real corpus."""
    manifest = {
        "version": DATA_VERSION,
        "sequence_length": 8,
        "vocab_size": 64,
        "num_classes": 4,
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.npz"
        tokens = np.tile(np.array([1, 2, 3, 4, 5, 0, 0, 0]), (8, 1))
        labels = np.arange(8) % 4
        np.savez_compressed(path, tokens=tokens, labels=labels)
        manifest["splits"][split] = {"sha256": sha256(path)}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return ExperimentConfig.model_validate(
        {
            "steps": 2,
            "batch_size": 4,
            "evaluate_test": False,
            "model": {"num_classes": 4},
            "data": {
                "name": "ag_news",
                "version": DATA_VERSION,
                "train_size": 8,
                "validation_size": 4,
                "test_size": 4,
                "prepared_directory": str(tmp_path),
                "manifest_sha256": sha256(path),
            },
            **({"dht": {"calibration_samples": 2}} if dht else {}),
        }
    )


def test_manifest_integrity_and_bounded_loading(tmp_path):
    config = prepared_config(tmp_path)
    tokens, labels = load_prepared(config, "validation")
    assert tokens.shape == (4, 8) and labels.tolist() == [0, 1, 2, 3]
    assert tokens.dtype == torch.long
    (tmp_path / "validation.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="split hash"):
        load_prepared(config, "validation")
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest hash"):
        load_prepared(config, "train")


@pytest.mark.parametrize("dht", [False, True])
def test_prepared_data_trains_both_paths_without_touching_test(tmp_path, dht):
    from dht.cli import train

    config = prepared_config(tmp_path, dht=dht)
    # Deleting the test artifact proves train/validation do not require test access.
    (tmp_path / "test.npz").unlink()
    summary = train(config, tmp_path / "run")
    assert summary["test"] is None
    assert summary["completed_steps"] == 2
    assert "ag_news" in summary["scope"]
    assert len((tmp_path / "run/metrics.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("legacy_norm_names", [False, True])
def test_pretrained_mapping_is_complete_preserves_heads_and_rejects_missing(
    model_config, legacy_norm_names
):
    source = TransformerClassifier(model_config)
    target = TransformerClassifier(model_config)
    mapping = bert_tensor_mapping(model_config.total_depth)
    assert set(mapping) == set(source.state_dict()) - {"classifier.weight", "classifier.bias"}
    # Use distinct row values so accidentally transposing/reordering heads is visible.
    tensors = {remote: source.state_dict()[local].clone() for local, remote in mapping.items()}
    if legacy_norm_names:
        tensors = {
            key.replace(".LayerNorm.weight", ".LayerNorm.gamma").replace(
                ".LayerNorm.bias", ".LayerNorm.beta"
            ): value
            for key, value in tensors.items()
        }
    classifier = target.classifier.weight.detach().clone()
    import_bert_encoder(target, tensors)
    for local in mapping:
        torch.testing.assert_close(
            target.state_dict()[local], source.state_dict()[local], rtol=0, atol=0
        )
    torch.testing.assert_close(target.classifier.weight, classifier, rtol=0, atol=0)
    del tensors["bert.encoder.layer.1.attention.self.query.weight"]
    with pytest.raises(ValueError, match="missing/incompatible"):
        import_bert_encoder(target, tensors)
