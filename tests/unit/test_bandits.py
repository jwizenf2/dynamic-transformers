"""Independent mathematical checks; these do not call a Transformer."""

import pytest
import torch

from dht.bandits.linear_ucb import LinearUCB
from dht.bandits.logistic_ts import (
    LogisticThompsonSampling,
    fit_logistic_map,
    logistic_objective,
)


def test_ucb_matches_direct_ridge_and_hand_computed_bonus():
    bandit = LinearUCB(2, 2, ridge=2.0, exploration=0.5)
    design = torch.tensor([[1.0, 2.0], [2.0, -1.0], [0.0, 1.0]], dtype=torch.float64)
    rewards = torch.tensor([3.0, -2.0, 1.0], dtype=torch.float64)
    for row, reward in zip(design, rewards, strict=True):
        bandit.update(0, row, float(reward))
    expected_a = 2 * torch.eye(2, dtype=torch.float64) + design.T @ design
    expected_b = design.T @ rewards
    torch.testing.assert_close(bandit.A[0], expected_a)
    torch.testing.assert_close(bandit.b[0], expected_b)
    features = torch.tensor([[1.0, -1.0], [2.0, 0.0]], dtype=torch.float64)
    mean = features[0] @ torch.linalg.solve(expected_a, expected_b)
    bonus = 0.5 * (features[0] @ torch.linalg.solve(expected_a, features[0])).sqrt()
    assert bandit.scores(features)[0] == pytest.approx(float(mean + bonus))
    assert bandit.scores(features, explore=False)[0] == pytest.approx(float(mean))
    torch.testing.assert_close(bandit.A[1], 2 * torch.eye(2, dtype=torch.float64))
    assert bandit.counts.tolist() == [3, 0]


def test_ucb_explores_unobserved_arm_and_checkpoints():
    bandit = LinearUCB(2, 1)
    w = torch.ones(2, 1)
    assert bandit.select(w) == 0
    bandit.update(0, w[0], 0)
    assert bandit.select(w) == 1
    restored = LinearUCB(2, 1)
    restored.load_state_dict(bandit.state_dict())
    torch.testing.assert_close(restored.scores(w), bandit.scores(w))


def test_ucb_learns_context_dependent_choices():
    bandit = LinearUCB(2, 2, exploration=0.3)
    generator = torch.Generator().manual_seed(3)
    # Opposing linear reward slopes require the selected arm to depend on context.
    for _ in range(100):
        x = float(torch.rand((), generator=generator)) * 2 - 1
        features = torch.tensor([[1.0, x], [1.0, x]], dtype=torch.float64)
        arm = bandit.select(features)
        bandit.update(arm, features[arm], x if arm == 0 else -x)
    assert bandit.select(torch.tensor([[1.0, 1.0], [1.0, 1.0]]), explore=False) == 0
    assert bandit.select(torch.tensor([[1.0, -1.0], [1.0, -1.0]]), explore=False) == 1


@pytest.mark.parametrize("ridge,exploration", [(0, 1), (-1, 1), (1, -1), (float("nan"), 1)])
def test_ucb_rejects_invalid_hyperparameters(ridge, exploration):
    with pytest.raises(ValueError):
        LinearUCB(2, 2, ridge=ridge, exploration=exploration)


def test_bandits_reject_invalid_feedback_or_features():
    bandit = LinearUCB(2, 2)
    with pytest.raises(ValueError):
        bandit.update(2, torch.ones(2), 1)
    with pytest.raises(ValueError):
        bandit.select(torch.ones(2, 3))
    with pytest.raises(ValueError):
        bandit.update(0, torch.ones(2), float("inf"))
    ts = LogisticThompsonSampling(2, 2)
    for feedback in (-1, 2, 0.5, True):
        with pytest.raises(ValueError):
            ts.update((0,), torch.ones(2, 2), feedback)


def test_logistic_objective_gradient_and_extreme_logits():
    theta = torch.tensor([0.4, -0.2], dtype=torch.float64, requires_grad=True)
    features = torch.tensor([[1.0, 2.0], [-2.0, 1.0]], dtype=torch.float64)
    labels = torch.tensor([1.0, 0.0], dtype=torch.float64)
    prior = torch.zeros(2, dtype=torch.float64)
    precision = torch.tensor([2.0, 3.0], dtype=torch.float64)
    objective = logistic_objective(theta, features, labels, prior, precision)
    actual = torch.autograd.grad(objective, theta)[0]
    expected = precision * theta + features.T @ ((features @ theta).sigmoid() - labels)
    torch.testing.assert_close(actual, expected)
    extreme = torch.tensor([1000.0, -1000.0], dtype=torch.float64)
    assert torch.isfinite(logistic_objective(extreme, features, labels, prior, precision))


def test_logistic_map_reduces_objective_and_uses_both_labels():
    prior = torch.zeros(2, dtype=torch.float64)
    precision = torch.ones(2, dtype=torch.float64)
    features = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    estimates = []
    for label in (0.0, 1.0):
        labels = torch.tensor([label], dtype=torch.float64)
        theta, diagnostics = fit_logistic_map(features, labels, prior, precision)
        assert diagnostics["converged"]
        assert logistic_objective(theta, features, labels, prior, precision) < (
            logistic_objective(prior, features, labels, prior, precision)
        )
        estimates.append(theta)
    assert estimates[0].sum() < 0 < estimates[1].sum()


def test_ts_selected_only_precision_and_rng_roundtrip():
    ts = LogisticThompsonSampling(3, 2, seed=5)
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    before = ts.state_dict()
    ts.update((2,), features, 0)
    torch.testing.assert_close(ts.mean[:2], before["mean"][:2])
    probability = (features[2] @ ts.mean[2]).sigmoid()
    torch.testing.assert_close(
        ts.precision[2],
        before["precision"][2] + probability * (1 - probability) * features[2].square(),
    )
    assert ts.counts.tolist() == [0, 0, 1]
    restored = LogisticThompsonSampling(3, 2, seed=999)
    restored.load_state_dict(ts.state_dict())
    for _ in range(5):
        assert ts.select(features, 2) == restored.select(features, 2)
    rng = ts.generator.get_state().clone()
    assert len(ts.select(features, 2, explore=False)) == 2
    assert torch.equal(rng, ts.generator.get_state())


def test_ts_gaussian_moments_and_logistic_learning():
    ts = LogisticThompsonSampling(2, 1, prior_precision=4, seed=11)
    draws = torch.stack([ts.sample_parameters() for _ in range(3000)])
    torch.testing.assert_close(
        draws.mean(0), torch.ones(2, 1, dtype=torch.float64), atol=0.04, rtol=0
    )
    torch.testing.assert_close(
        draws.var(0), torch.full((2, 1), 0.25, dtype=torch.float64), atol=0.03, rtol=0
    )
    # Controlled binary evidence: one head reliably succeeds, another fails.
    features = torch.ones(2, 1, dtype=torch.float64)
    for _ in range(30):
        ts.update((0,), features, 0)
        ts.update((1,), features, 1)
    assert ts.select(features, 1, explore=False) == (1,)
    assert ts.mean[0] < 0 < ts.mean[1]


@pytest.mark.parametrize("count", [0, 4, -1, True])
def test_ts_top_k_validation(count):
    with pytest.raises(ValueError):
        LogisticThompsonSampling(3, 2).select(torch.ones(3, 2), count)
