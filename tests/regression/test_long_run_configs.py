"""Guard the predeclared main protocol; no datasets or pretrained models are loaded."""

from dht.config import load_config


def test_full_configs_keep_equal_exposure_and_literal_primary():
    fixed = load_config("configs/experiment/ag_news_fixed_full.toml")
    tree = load_config("configs/experiment/ag_news_tree_full.toml")
    for config in (fixed, tree):
        assert config.training.epochs == 3
        assert config.data.train_size == 114000
        assert config.data.validation_size == 6000
        assert config.data.test_size == 7600
        assert config.training.selection_metric == "macro_f1"
        assert config.training.schedule == "linear_warmup_decay"
        assert not config.evaluate_test
        assert not config.allow_device_fallback
    for key in (
        "model",
        "data",
        "training",
        "bandit",
        "seed",
        "pretrained_sha256",
        "learning_rate",
        "weight_decay",
        "batch_size",
    ):
        assert getattr(fixed, key) == getattr(tree, key)
    assert tree.dht.calibration_samples == 512
    assert tree.dht.calibration_sampling == "stratified"
    assert tree.dht.feedback_rule == "paper_literal"


def test_feedback_diagnostic_only_changes_rule_and_name():
    literal = load_config("configs/experiment/ag_news_readiness_literal.toml").model_dump()
    improved = load_config("configs/experiment/ag_news_readiness_improvement.toml").model_dump()
    assert literal.pop("name") != improved.pop("name")
    assert literal["dht"].pop("feedback_rule") == "paper_literal"
    assert improved["dht"].pop("feedback_rule") == "improvement"
    assert literal == improved
    assert literal["data"]["train_size"] == 1024
    assert literal["training"]["epochs"] == 2
