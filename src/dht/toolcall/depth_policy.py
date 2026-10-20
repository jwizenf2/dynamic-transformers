"""Per-call loop-depth policy for looped LMs in multi-turn tool use (path B), pure NumPy.

Decision problem (an episodic MDP with one decision per assistant turn):

* state s_t: features available before the turn is generated (turn index, calls so far, whether
  the last call failed, prompt length, and per-loop statistics of the last prompt token: exit
  gate, entropy, top-2 margin, agreement with loop 4);
* action a_t ∈ {1, 2, 3, 4}: the number of loops for every token of that turn;
* reward at the end: R = 1[final answer correct] − λ · (L̄ − 1) / 3, where L̄ is the mean loops
  per generated token over the dialogue (so the cost term is in [0, λ]).

Policy: softmax over a linear score, π(a | s) = softmax(W s + b), with s standardised by running
statistics. Training: REINFORCE with a batch-mean baseline and an entropy bonus,

    θ ← θ + η · mean_episodes[(R − b) Σ_t ∇ log π(a_t | s_t)] + η·β·∇ H(π(· | s_t)).

The gradient of log softmax for a linear score is (onehot(a) − π) ⊗ s for W and (onehot(a) − π)
for b; the entropy gradient for logits z is −π ⊙ (log π + H) (per state), propagated the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEPTHS = (1, 2, 3, 4)


def reward(correct: bool, loops_per_token: float, lam: float) -> float:
    """1[correct] − λ·(L̄ − 1)/3; L̄ ∈ [1, 4]."""
    return float(correct) - lam * (loops_per_token - 1.0) / 3.0


@dataclass
class RunningNorm:
    """Running mean/variance (Welford) for feature standardisation."""

    dim: int
    n: int = 0
    mean: np.ndarray = field(default=None)  # type: ignore[assignment]
    m2: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.dim) if self.mean is None else np.asarray(self.mean, float)
        self.m2 = np.zeros(self.dim) if self.m2 is None else np.asarray(self.m2, float)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        d = x - self.mean
        self.mean = self.mean + d / self.n
        self.m2 = self.m2 + d * (x - self.mean)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.n < 2:
            return x - self.mean
        sd = np.sqrt(self.m2 / (self.n - 1)) + 1e-6
        return (x - self.mean) / sd


@dataclass
class DepthPolicy:
    """Linear softmax policy over the four loop counts."""

    dim: int
    seed: int = 0
    W: np.ndarray = field(default=None)  # type: ignore[assignment]
    b: np.ndarray = field(default=None)  # type: ignore[assignment]
    norm: RunningNorm = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        if self.W is None:
            self.W = rng.normal(0, 0.01, (len(DEPTHS), self.dim))
        if self.b is None:
            self.b = np.zeros(len(DEPTHS))
        if self.norm is None:
            self.norm = RunningNorm(self.dim)

    def probs(self, s: np.ndarray) -> np.ndarray:
        z = self.W @ self.norm(s) + self.b
        z = z - z.max()
        p = np.exp(z)
        return p / p.sum()

    def act(self, s: np.ndarray, rng: np.random.Generator | None = None) -> int:
        """Sample a depth (training) or take the argmax (``rng=None``, evaluation)."""
        p = self.probs(s)
        k = int(rng.choice(len(DEPTHS), p=p)) if rng is not None else int(np.argmax(p))
        return DEPTHS[k]

    def update(
        self, episodes: list[tuple[list[np.ndarray], list[int], float]], lr: float, beta: float
    ) -> dict:
        """One REINFORCE step on a batch of (states, depths, reward) episodes."""
        rewards = np.array([r for _, _, r in episodes])
        base = rewards.mean()
        gW = np.zeros_like(self.W)
        gb = np.zeros_like(self.b)
        ent = []
        for states, depths, r in episodes:
            adv = r - base
            for s, d in zip(states, depths, strict=True):
                x = self.norm(s)
                p = self.probs(s)
                onehot = np.zeros(len(DEPTHS))
                onehot[DEPTHS.index(d)] = 1.0
                g_logit = adv * (onehot - p)
                h = -(p * np.log(p + 1e-12)).sum()
                ent.append(h)
                g_logit += beta * (-p * (np.log(p + 1e-12) + h))
                gW += np.outer(g_logit, x)
                gb += g_logit
        n = max(1, len(episodes))
        self.W += lr * gW / n
        self.b += lr * gb / n
        return {
            "mean_reward": float(base),
            "entropy": float(np.mean(ent)) if ent else 0.0,
        }

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "W": self.W.tolist(),
            "b": self.b.tolist(),
            "norm": {
                "n": self.norm.n,
                "mean": self.norm.mean.tolist(),
                "m2": self.norm.m2.tolist(),
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> DepthPolicy:
        norm = RunningNorm(
            d["dim"], d["norm"]["n"], np.array(d["norm"]["mean"]), np.array(d["norm"]["m2"])
        )
        return cls(d["dim"], W=np.array(d["W"]), b=np.array(d["b"]), norm=norm)


def loop_stats(logits_per_loop: list[np.ndarray], gates: list[float]) -> np.ndarray:
    """Per-loop features of one position: gate, entropy, top-2 margin, argmax == loop-4 argmax.

    ``logits_per_loop``: four vocabulary logit vectors (loops 1..4); returns 16 numbers.
    """
    out = []
    last = int(np.argmax(logits_per_loop[-1]))
    for z, g in zip(logits_per_loop, gates, strict=True):
        z = z - z.max()
        logp = z - np.log(np.exp(z).sum())
        p = np.exp(logp)
        top2 = np.sort(logp)[-2:]
        out += [g, float(-(p * logp).sum()), float(top2[1] - top2[0]), float(np.argmax(z) == last)]
    return np.array(out)


def context_features(turn: int, n_calls: int, last_error: bool, prompt_tokens: int, n_tools: int):
    """Dialogue-level features (turn index capped at 5, one-hot)."""
    onehot = np.zeros(6)
    onehot[min(turn, 5)] = 1.0
    return np.concatenate(
        [onehot, [n_calls, float(last_error), np.log1p(prompt_tokens), float(n_tools)]]
    )
