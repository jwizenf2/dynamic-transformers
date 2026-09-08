"""Previous-batch loss/cost ratios from §4.3, printed pp.443–444.

Depth product: loss*total_depth (explicit).
Head product: loss*head_count; pair product: loss*depth*head_count (interpretations).
Each key compares to its previous OBSERVED batch mean, never another sample in
the same current batch. An unseen key initializes history and produces no update.
"""

import math
from collections import defaultdict


class RewardHistory:
    """Small stateful reward calculator; history must be checkpointed with queues."""

    def __init__(self, scale: float = 1, epsilon: float = 1e-8, cap: float | None = 100):
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("reward scale/epsilon must be finite and positive")
        if cap is not None and (not math.isfinite(cap) or cap <= 0):
            raise ValueError("reward cap must be positive or None")
        self.scale, self.epsilon, self.cap = scale, epsilon, cap
        self.previous: dict[str, float] = {}
        self.clipped = 0

    def observe_batch(self, products: list[dict[str, float]]) -> list[dict[str, float | None]]:
        """Compute all ratios against OLD history, then commit current batch means.

        Missing group keys are ignored rather than assigned hypothetical rewards.
        Histories for absent groups retain their most recent observed batch mean.
        cap limits extreme ratios for engineering stability; None disables it.
        """
        grouped = defaultdict(list)
        for sample in products:
            for key, value in sample.items():
                if not math.isfinite(value) or value < 0:
                    raise ValueError("loss/cost products must be finite and nonnegative")
                grouped[key].append(value)
        rewards = []
        for sample in products:
            current = {}
            for key, value in sample.items():
                if key not in self.previous:
                    current[key] = None
                    continue
                ratio = self.scale * self.previous[key] / max(value, self.epsilon)
                if self.cap is not None and ratio > self.cap:
                    ratio = self.cap
                    self.clipped += 1
                if not math.isfinite(ratio):
                    raise FloatingPointError("reward overflow; use a finite reward_cap")
                current[key] = ratio
            rewards.append(current)
        for key, values in grouped.items():
            self.previous[key] = sum(values) / len(values)
        return rewards

    def binary_feedback(self, reward: float, rule: str) -> int:
        """Keep the printed contradictory threshold visible and configurable.

        paper_literal: 1 when reward<=1, as printed p.444.
        improvement: 1 when reward>1, a prose-consistent correction/ablation.
        This function does not claim to resolve the authors' intended direction.
        """
        if not math.isfinite(reward):
            raise ValueError("binary feedback requires a finite reward")
        if rule == "paper_literal":
            return int(reward <= 1)
        if rule == "improvement":
            return int(reward > 1)
        raise ValueError("unknown feedback rule")

    def state_dict(self) -> dict:
        """Preserve both historical products and the number of capped rewards."""
        return {"previous": dict(self.previous), "clipped": self.clipped}

    def load_state_dict(self, state: dict) -> None:
        """Restore prior completed-batch observations."""
        if any(not math.isfinite(v) or v < 0 for v in state["previous"].values()):
            raise ValueError("invalid reward history")
        self.previous, self.clipped = dict(state["previous"]), int(state["clipped"])
