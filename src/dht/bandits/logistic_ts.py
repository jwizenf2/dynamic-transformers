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
    #
    # Absolute gradient magnitude alone is not a scale-invariant stopping rule. As
    # online observations accumulate, the prior precision can become large enough
    # that a residual gradient around 1e-6 implies a vanishingly small Newton step.
    # Treat that case as converged when the next local Newton update is smaller than
    # tolerance relative to the current parameter scale. This does not change the
    # MAP objective or accept a materially movable solution; it prevents harmless
    # floating-point stationarity floors from aborting a long experiment (D015).
    gradient = prior_precision * (theta - prior_mean) + features.T @ (
        (features @ theta).sigmoid() - labels
    )
    gradient_norm = float(gradient.abs().max())
    probability = (features @ theta).sigmoid()
    curvature = probability * (1 - probability)
    hessian = torch.diag(prior_precision) + features.T @ (curvature[:, None] * features)
    newton_step = torch.linalg.solve(hessian, gradient)
    relative_newton_step = float(newton_step.abs().max() / (1.0 + theta.abs().max()))
    converged = converged or gradient_norm <= 10 * tolerance or relative_newton_step <= tolerance
    diagnostics = {
        "iterations": iterations,
        "gradient_max": gradient_norm,
        "relative_newton_step_max": relative_newton_step,
        "converged": converged,
        "objective": float(
            logistic_objective(theta, features, labels, prior_mean, prior_precision)
        ),
    }
    if not converged:
        raise RuntimeError(f"logistic MAP did not converge: {diagnostics}")
    return theta, diagnostics


def fit_logistic_map_many(
    features: Tensor,
    label: float,
    prior_mean: Tensor,
    prior_precision: Tensor,
    max_iterations: int = 30,
    tolerance: float = 1e-7,
) -> tuple[Tensor, list[dict]]:
    """fit_logistic_map for K INDEPENDENT heads at once, one observation each (D017).

    features, prior_mean, prior_precision: [K,p]; every head shares the superarm's
    binary ``label``. Each head runs exactly the per-head algorithm above (damped
    Newton, Armijo backtracking, same absolute-gradient and D015 relative-Newton
    stopping rules); converged heads are frozen while others keep iterating.
    Vectorizing only removes the Python loop over heads: the objective, step rule
    and acceptance tests are unchanged. Tests compare it with the looped solver.
    """
    count = len(features)
    labels = torch.full((count,), float(label), dtype=torch.float64)

    def objective(theta: Tensor) -> Tensor:  # per-head negative log posterior, [K]
        logits = (features * theta).sum(-1)
        return 0.5 * (prior_precision * (theta - prior_mean).square()).sum(-1) + (
            F.softplus(logits) - labels * logits
        )

    def gradient_and_hessian(theta: Tensor) -> tuple[Tensor, Tensor]:
        probability = (features * theta).sum(-1).sigmoid()
        gradient = (
            prior_precision * (theta - prior_mean) + features * (probability - labels)[:, None]
        )
        curvature = probability * (1 - probability)
        hessian = torch.diag_embed(prior_precision) + curvature[:, None, None] * (
            features[:, :, None] * features[:, None, :]
        )
        return gradient, hessian

    theta = prior_mean.clone()
    active = torch.ones(count, dtype=torch.bool)
    converged = torch.zeros(count, dtype=torch.bool)
    iterations = torch.zeros(count, dtype=torch.long)
    for _ in range(max_iterations):
        if not active.any():
            break
        iterations += active
        gradient, hessian = gradient_and_hessian(theta)
        gradient_norm = gradient.abs().amax(-1)
        done = active & (gradient_norm <= tolerance)
        converged |= done
        active &= ~done
        if not active.any():
            break
        direction = torch.linalg.solve(hessian, gradient)
        before = objective(theta)
        slope = (gradient * direction).sum(-1)
        step = torch.ones(count, dtype=torch.float64)
        searching = active.clone()
        for _ in range(30):
            candidate = theta - step[:, None] * direction
            accept = searching & (objective(candidate) <= before - 1e-4 * step * slope)
            theta = torch.where(accept[:, None], candidate, theta)
            searching &= ~accept
            if not searching.any():
                break
            step = torch.where(searching, step * 0.5, step)
        if searching.any():
            # Same fallback as the scalar solver: tiny gradients count as converged.
            floor = searching & (gradient_norm <= 10 * tolerance)
            converged |= floor
            active &= ~floor
            if (searching & ~floor).any():
                raise RuntimeError("logistic MAP line search failed")
    gradient, hessian = gradient_and_hessian(theta)
    gradient_norm = gradient.abs().amax(-1)
    newton = torch.linalg.solve(hessian, gradient)
    relative = newton.abs().amax(-1) / (1.0 + theta.abs().amax(-1))
    converged |= (gradient_norm <= 10 * tolerance) | (relative <= tolerance)
    final = objective(theta)
    diagnostics = [
        {
            "iterations": int(iterations[k]),
            "gradient_max": float(gradient_norm[k]),
            "relative_newton_step_max": float(relative[k]),
            "converged": bool(converged[k]),
            "objective": float(final[k]),
        }
        for k in range(count)
    ]
    if not converged.all():
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

    def probabilities(self, features: Tensor) -> Tensor:
        """Posterior-mean σ(θ̄_hᵀw_h) per head: the frozen-policy ranking score."""
        w = feature_tensor(features, tuple(self.mean.shape))
        return (w * self.mean).sum(-1).sigmoid()

    def select(
        self,
        features: Tensor,
        count: int,
        *,
        explore: bool = True,
        parameters: Tensor | None = None,
    ) -> tuple[int, ...]:
        """Return a sorted set of original head positions, not compacted positions.

        At evaluation use posterior means, so neither RNG nor posterior changes.
        Stable sorting makes ties reproducible. All unrestricted K-subsets are
        possible; no unsupported 70/28 combination restriction is invented.
        """
        w = feature_tensor(features, tuple(self.mean.shape))
        if type(count) is not int or not 1 <= count <= self.num_arms:
            raise ValueError("head count must lie in [1,num_arms]")
        # Batched DHT pre-draws parameters in the per-sample path's RNG order and
        # passes them here, so both execution modes consume identical random draws.
        if parameters is not None:
            theta = parameters
        else:
            theta = self.sample_parameters() if explore else self.mean
        probability = (w * theta).sum(-1).sigmoid()
        selected = torch.argsort(probability, descending=True, stable=True)[:count].tolist()
        return tuple(sorted(selected))

    def update(
        self,
        selected: tuple[int, ...],
        features: Tensor,
        feedback: int,
        *,
        vectorized: bool = False,
    ) -> None:
        """Observe the executed superarm's 0/1 label; unselected heads stay untouched.

        vectorized=True fits all selected heads in one fit_logistic_map_many call
        (batched execution, D017); False keeps the original per-head loop.
        """
        if type(feedback) is not int or feedback not in (0, 1):
            raise ValueError("feedback must be binary integer 0 or 1")
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(type(arm) is not int or not 0 <= arm < self.num_arms for arm in selected)
        ):
            raise ValueError("selected arms must be nonempty, unique and valid")
        w = feature_tensor(features, tuple(self.mean.shape))
        if vectorized:
            arms = list(selected)
            thetas, infos = fit_logistic_map_many(
                w[arms],
                feedback,
                self.mean[arms],
                self.precision[arms],
                self.max_iterations,
                self.tolerance,
            )
            probability = (w[arms] * thetas).sum(-1).sigmoid()
            precisions = (
                self.precision[arms] + (probability * (1 - probability))[:, None] * w[arms].square()
            )
            for row, arm in enumerate(arms):  # Commit only after ALL fits succeeded.
                self.mean[arm], self.precision[arm] = thetas[row], precisions[row]
                self.counts[arm] += 1
                self.last_fit[arm] = infos[row]
            return
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
