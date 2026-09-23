"""Disjoint linear UCB: §3.1, Algorithm 1, printed p.441.

Each arm has its OWN linear reward model. If w concatenates sample and arm
features, p=len(w), A has shape [arms,p,p], and b has shape [arms,p].
This repairs the paper's inconsistent d/2d dimensions (decision D005).
No explicit matrix inverse and no neural controller are needed.
"""

import math

import torch
from torch import Tensor


def feature_tensor(value: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Detach controller input from autograd, copy to CPU double, validate dimensions.

    The controller observes features; task gradients must not flow through its
    discrete action. Callers retain decision-time copies for delayed rewards.
    """
    result = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
    if result.shape != shape or not torch.isfinite(result).all():
        raise ValueError(f"expected finite features of shape {shape}, got {tuple(result.shape)}")
    return result


class LinearUCB:
    """Per-arm ridge regression with an upper confidence exploration bonus.

    Arms are integer positions in a caller-owned ordered action list. Ties go to
    the first arm, so action ordering is part of experiment/checkpoint provenance.
    """

    def __init__(self, num_arms: int, feature_dim: int, ridge: float = 1, exploration: float = 1):
        if type(num_arms) is not int or type(feature_dim) is not int:
            raise ValueError("arm count and feature width must be integers")
        if num_arms < 1 or feature_dim < 1:
            raise ValueError("arm count and feature width must be positive")
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        if not math.isfinite(exploration) or exploration < 0:
            raise ValueError("exploration must be finite and nonnegative")
        self.num_arms, self.feature_dim = num_arms, feature_dim
        self.ridge, self.exploration = ridge, exploration
        self.A = ridge * torch.eye(feature_dim, dtype=torch.float64).repeat(num_arms, 1, 1)
        self.b = torch.zeros(num_arms, feature_dim, dtype=torch.float64)
        self.counts = torch.zeros(num_arms, dtype=torch.long)

    def scores(self, features: Tensor, *, explore: bool = True) -> Tensor:
        """Return one score per arm for features shaped [num_arms, feature_dim].

        A = λI + Σwwᵀ, b = Σreward*w; solve(A,b) is the ridge estimate.
        sqrt(wᵀ A⁻¹ w) is large in poorly observed feature directions.
        Evaluation sets explore=False and uses estimated mean rewards only.
        """
        w = feature_tensor(features, (self.num_arms, self.feature_dim))
        theta = torch.linalg.solve(self.A, self.b.unsqueeze(-1)).squeeze(-1)
        mean = (w * theta).sum(-1)
        if not explore or self.exploration == 0:
            return mean
        inverse_times_w = torch.linalg.solve(self.A, w.unsqueeze(-1)).squeeze(-1)
        variance = (w * inverse_times_w).sum(-1)
        if torch.any(variance < -1e-10):
            raise FloatingPointError("UCB covariance lost positive definiteness")
        # A tiny negative quadratic form can arise from floating-point roundoff.
        return mean + self.exploration * variance.clamp_min(0).sqrt()

    def select_many(self, features: Tensor, *, explore: bool = True) -> list[int]:
        """scores()+argmax for N samples at once: features [N, num_arms, feature_dim].

        One solve per arm with N right-hand sides replaces N separate solves (D018).
        Same float64 formulas; only the solver's batching differs, so results agree
        with select() to rounding, and ties (first arm) are resolved identically.
        """
        w = torch.as_tensor(features).detach().to(device="cpu", dtype=torch.float64)
        if w.ndim != 3 or w.shape[1:] != (self.num_arms, self.feature_dim):
            raise ValueError("expected features of shape [N, num_arms, feature_dim]")
        if not torch.isfinite(w).all():
            raise ValueError("features must be finite")
        theta = torch.linalg.solve(self.A, self.b.unsqueeze(-1)).squeeze(-1)
        scores = (w * theta).sum(-1)
        if explore and self.exploration:
            right = w.permute(1, 2, 0)  # [arms, p, N]
            variance = (right * torch.linalg.solve(self.A, right)).sum(1).T  # [N, arms]
            if torch.any(variance < -1e-10):
                raise FloatingPointError("UCB covariance lost positive definiteness")
            scores = scores + self.exploration * variance.clamp_min(0).sqrt()
        return scores.argmax(-1).tolist()

    def select(self, features: Tensor, *, explore: bool = True) -> int:
        """Choose an arm without updating any statistics."""
        return int(self.scores(features, explore=explore).argmax())

    def update(self, arm: int, features: Tensor, reward: float) -> None:
        """Update ONLY the executed arm using its stored decision-time context."""
        if type(arm) is not int or not 0 <= arm < self.num_arms:
            raise ValueError("invalid arm index")
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        w = feature_tensor(features, (self.feature_dim,))
        self.A[arm] += torch.outer(w, w)
        self.b[arm] += reward * w
        self.counts[arm] += 1

    def state_dict(self) -> dict:
        """Independent snapshot: later updates do not mutate the saved state."""
        return {
            "shape": (self.num_arms, self.feature_dim),
            "ridge": self.ridge,
            "exploration": self.exploration,
            "A": self.A.clone(),
            "b": self.b.clone(),
            "counts": self.counts.clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Reject incompatible priors/action counts instead of silently resetting."""
        if (
            tuple(state["shape"]) != (self.num_arms, self.feature_dim)
            or state["ridge"] != self.ridge
            or state["exploration"] != self.exploration
        ):
            raise ValueError("incompatible UCB checkpoint")
        matrix = feature_tensor(state["A"], tuple(self.A.shape))
        if not torch.allclose(matrix, matrix.transpose(-2, -1)):
            raise ValueError("UCB checkpoint matrix must be symmetric")
        torch.linalg.cholesky(matrix)  # Fails explicitly if not positive definite.
        self.A = matrix.clone()
        self.b = feature_tensor(state["b"], tuple(self.b.shape)).clone()
        counts = state["counts"]
        if counts.shape != self.counts.shape or counts.dtype != torch.long or (counts < 0).any():
            raise ValueError("invalid UCB observation counts")
        self.counts = counts.cpu().clone()
