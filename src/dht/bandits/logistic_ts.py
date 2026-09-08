"""Gaussian logistic Thompson semi-bandit: §3.2, printed pp.441–442.

The paper specifies parameter sampling, sigmoid scores and top-K head selection,
but prints a singular prior and an inconsistent optimization objective. Decision
D005 repairs these with an online diagonal Laplace approximation to binary
logistic regression. This is NOT Beta-Bernoulli Thompson sampling.

A selected subset receives one observed binary label. We assign that label only
to selected heads; this does not pretend to observe independent head effects.
"""

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from dht.bandits.linear_ucb import feature_tensor


def logistic_objective(
    theta: Tensor, features: Tensor, labels: Tensor, prior_mean: Tensor, prior_precision: Tensor
) -> Tensor:
    """Negative log posterior: Gaussian penalty plus stable Bernoulli loss.

    theta, prior_mean, prior_precision: [p]; features: [observations,p];
    labels: [observations], values 0 or 1. softplus(z)-y*z is binary cross
    entropy without forming log(sigmoid(z)), which is unstable at extreme z.
    """
    logits = features @ theta
    return (
        0.5 * (prior_precision * (theta - prior_mean).square()).sum()
        + (F.softplus(logits) - labels * logits).sum()
    )


def fit_logistic_map(
    features: Tensor,
    labels: Tensor,
    prior_mean: Tensor,
    prior_precision: Tensor,
    max_iterations: int = 30,
    tolerance: float = 1e-7,
) -> tuple[Tensor, dict]:
    """Find the MAP parameters using damped Newton steps and backtracking.

    The MAP Hessian may be full even though our saved Gaussian approximation is
    diagonal. A positive prior makes this small convex problem strictly convex.
    History is already represented by the prior: features contains NEW feedback
    only, avoiding double-counting old observations on every update.
    """
    theta = prior_mean.clone()
    converged = False
    gradient_norm = float("inf")
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        probability = (features @ theta).sigmoid()
        gradient = prior_precision * (theta - prior_mean) + features.T @ (probability - labels)
        gradient_norm = float(gradient.abs().max())
        if gradient_norm <= tolerance:
            converged = True
            break
        curvature = probability * (1 - probability)
        hessian = torch.diag(prior_precision) + features.T @ (curvature[:, None] * features)
        direction = torch.linalg.solve(hessian, gradient)
        before = logistic_objective(theta, features, labels, prior_mean, prior_precision)
        step_size = 1.0
        for _ in range(30):
            candidate = theta - step_size * direction
            after = logistic_objective(candidate, features, labels, prior_mean, prior_precision)
            if after <= before - 1e-4 * step_size * torch.dot(gradient, direction):
                theta = candidate
                break
            step_size *= 0.5
        else:
            # Close to the optimum objective differences can be below float precision.
            if gradient_norm <= 10 * tolerance:
                converged = True
                break
            raise RuntimeError("logistic MAP line search failed")
    # Check the final accepted step too (including one taken on the last iteration).
    gradient = prior_precision * (theta - prior_mean) + features.T @ (
        (features @ theta).sigmoid() - labels
    )
    gradient_norm = float(gradient.abs().max())
    converged = converged or gradient_norm <= 10 * tolerance
    diagnostics = {
        "iterations": iterations,
        "gradient_max": gradient_norm,
        "converged": converged,
        "objective": float(
            logistic_objective(theta, features, labels, prior_mean, prior_precision)
        ),
    }
    if not converged:
        raise RuntimeError(f"logistic MAP did not converge: {diagnostics}")
    return theta, diagnostics


class LogisticThompsonSampling:
    """Independent diagonal Gaussian parameter posterior per head.

    mean and precision have shape [num_heads, feature_dim]. The default ones
    mean follows the paper's rectangular I notation; positive precision is an
    explicit repair. Sampling uses a dedicated CPU generator, saved in checkpoints.
    """

    def __init__(
        self,
        num_arms: int,
        feature_dim: int,
        prior_precision: float = 1,
        seed: int = 17,
        max_iterations: int = 30,
        tolerance: float = 1e-7,
    ):
        if (
            type(num_arms) is not int
            or type(feature_dim) is not int
            or min(num_arms, feature_dim) < 1
        ):
            raise ValueError("positive integer arm count/feature width required")
        if not math.isfinite(prior_precision) or prior_precision <= 0:
            raise ValueError("prior precision must be finite and positive")
        if max_iterations < 1 or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("invalid logistic optimization settings")
        self.num_arms, self.feature_dim = num_arms, feature_dim
        self.prior_precision = prior_precision
        self.max_iterations, self.tolerance = max_iterations, tolerance
        self.mean = torch.ones(num_arms, feature_dim, dtype=torch.float64)
        self.precision = torch.full_like(self.mean, prior_precision)
        self.counts = torch.zeros(num_arms, dtype=torch.long)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.last_fit: dict[int, dict] = {}

    def sample_parameters(self) -> Tensor:
        """θ_h ~ N(mean_h, diag(1/precision_h)); no model gradients are involved."""
        noise = torch.randn(self.mean.shape, generator=self.generator, dtype=torch.float64)
        return self.mean + noise / self.precision.sqrt()

    def select(self, features: Tensor, count: int, *, explore: bool = True) -> tuple[int, ...]:
        """Return a sorted set of original head positions, not compacted positions.

        At evaluation use posterior means, so neither RNG nor posterior changes.
        Stable sorting makes ties reproducible. All unrestricted K-subsets are
        possible; no unsupported 70/28 combination restriction is invented.
        """
        w = feature_tensor(features, tuple(self.mean.shape))
        if type(count) is not int or not 1 <= count <= self.num_arms:
            raise ValueError("head count must lie in [1,num_arms]")
        theta = self.sample_parameters() if explore else self.mean
        probability = (w * theta).sum(-1).sigmoid()
        selected = torch.argsort(probability, descending=True, stable=True)[:count].tolist()
        return tuple(sorted(selected))

    def update(self, selected: tuple[int, ...], features: Tensor, feedback: int) -> None:
        """Observe the executed superarm's 0/1 label; unselected heads stay untouched."""
        if type(feedback) is not int or feedback not in (0, 1):
            raise ValueError("feedback must be binary integer 0 or 1")
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(type(arm) is not int or not 0 <= arm < self.num_arms for arm in selected)
        ):
            raise ValueError("selected arms must be nonempty, unique and valid")
        w = feature_tensor(features, tuple(self.mean.shape))
        pending = {}
        # Calculate first and commit only if ALL selected-head fits succeed.
        for arm in selected:
            theta, diagnostics = fit_logistic_map(
                w[arm : arm + 1],
                torch.tensor([feedback], dtype=torch.float64),
                self.mean[arm],
                self.precision[arm],
                self.max_iterations,
                self.tolerance,
            )
            probability = torch.sigmoid(w[arm] @ theta)
            new_precision = self.precision[arm] + probability * (1 - probability) * w[arm].square()
            pending[arm] = theta, new_precision, diagnostics
        for arm, (theta, precision, diagnostics) in pending.items():
            self.mean[arm], self.precision[arm] = theta, precision
            self.counts[arm] += 1
            self.last_fit[arm] = diagnostics

    def state_dict(self) -> dict:
        """Save posterior, solver settings, observation counts and sampling stream."""
        return {
            "shape": tuple(self.mean.shape),
            "prior_precision": self.prior_precision,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "mean": self.mean.clone(),
            "precision": self.precision.clone(),
            "counts": self.counts.clone(),
            "rng": self.generator.get_state().clone(),
            "last_fit": {arm: dict(info) for arm, info in self.last_fit.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        """Load only a compatible, finite, strictly positive posterior."""
        if (
            tuple(state["shape"]) != tuple(self.mean.shape)
            or state["prior_precision"] != self.prior_precision
            or state["max_iterations"] != self.max_iterations
            or state["tolerance"] != self.tolerance
        ):
            raise ValueError("incompatible logistic TS checkpoint")
        precision = feature_tensor(state["precision"], tuple(self.mean.shape))
        if (precision <= 0).any():
            raise ValueError("TS precision must be positive")
        counts = state["counts"]
        if counts.shape != self.counts.shape or counts.dtype != torch.long or (counts < 0).any():
            raise ValueError("invalid TS observation counts")
        self.mean = feature_tensor(state["mean"], tuple(self.mean.shape)).clone()
        self.precision = precision.clone()
        self.counts = counts.cpu().clone()
        self.generator.set_state(state["rng"])
        self.last_fit = {arm: dict(info) for arm, info in state["last_fit"].items()}
