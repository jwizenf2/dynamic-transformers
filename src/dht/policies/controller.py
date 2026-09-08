"""Tree and integrated DHT execution: §4.2, Algorithm 2, Figure 1.

Read route() and execute() in order. route() observes a no-gradient prefix and
selects depth (or depth/count); execute() reruns the prefix with task gradients
and selects each group's heads using its current intermediate representation.
The training loop owns queues and optimizer steps; this class owns bandit state.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from dht.bandits.features import FeatureBank
from dht.bandits.linear_ucb import LinearUCB
from dht.bandits.logistic_ts import LogisticThompsonSampling
from dht.config import ExperimentConfig
from dht.models.architecture import ArchitectureSpec
from dht.models.transformer import TransformerClassifier
from dht.policies.rewards import RewardHistory


@dataclass
class RouteDecision:
    """Queue payload: original sample ID plus immutable decision-time evidence.

    The context can become stale while the queue fills. That is expected: updates
    must use the context that produced the action, not a newly computed substitute.
    """

    sample_id: int
    total_depth: int
    route_arm: int
    route_features: Tensor
    head_count: int | None = None


@dataclass
class GroupDecision:
    """Observed count/subset choices for one executed c-layer group."""

    group_index: int
    start_layer: int
    count_arm: int | None
    count_features: Tensor | None
    head_indices: tuple[int, ...]
    ts_features: Tensor


@dataclass
class Execution:
    """One sample's task logits and decisions; only logits retain the autograd graph."""

    logits: Tensor
    architecture: ArchitectureSpec
    route: RouteDecision
    groups: list[GroupDecision]


class DHTController:
    """Two readable branches implement tree and integrated search in one workflow."""

    def __init__(self, config: ExperimentConfig, features: FeatureBank):
        if config.dht is None:
            raise ValueError("DHT configuration required")
        self.config, self.options, self.features = config, config.dht, features
        self.depths, self.counts, self.pairs = self.options.actions(config.model)
        raw_depth = features.depth_arms.shape[-1]
        raw_head = features.head_arms.shape[-1]
        raw_ts = features.ts_arms.shape[-1]
        route_count = len(self.depths) if self.options.mode == "tree" else len(self.pairs)
        self.route_ucb = LinearUCB(
            route_count, 2 * raw_depth, config.bandit.ucb_ridge, config.bandit.ucb_exploration
        )
        self.head_ucb = (
            LinearUCB(
                len(self.counts),
                2 * raw_head + len(features.groups),
                config.bandit.ucb_ridge,
                config.bandit.ucb_exploration,
            )
            if self.options.mode == "tree" and features.groups
            else None
        )
        self.ts = LogisticThompsonSampling(
            config.model.num_heads,
            2 * raw_ts,
            config.bandit.ts_prior_precision,
            config.bandit.seed,
            config.bandit.ts_max_iterations,
            config.bandit.ts_tolerance,
        )
        self.rewards = RewardHistory(
            self.options.reward_scale, self.options.reward_epsilon, self.options.reward_cap
        )

    @torch.no_grad()
    def route(
        self, model: TransformerClassifier, tokens: Tensor, sample_id: int, *, explore: bool = True
    ) -> RouteDecision:
        """Stage 1: full-head prefix without dropout/task gradients, then route UCB.

        No optimizer or bandit update occurs here. Training-mode restoration is
        essential: otherwise routing would accidentally disable dropout afterward.

        Paper evidence: §4.2.1 Step 1 (p.443), §5 (p.444), Algorithm 2 lines
        2–6 (p.444), and Figure 1 Stage 1 (p.445). "Frozen" means no update
        during this pass; these are the SAME trainable prefix parameters used by
        execute(), not a frozen copy. @no_grad prevents building the task graph;
        eval() separately disables dropout (our D008 choice, not a paper detail).
        """
        if len(tokens) != 1:
            raise ValueError("routing expects one sample")
        previous_mode = model.training
        model.eval()
        try:
            hidden, mask = model.prefix(tokens)
            kind = "depth" if self.options.mode == "tree" else "pair"
            w = self.features.aggregate(hidden, mask, kind)
            arm = self.route_ucb.select(w, explore=explore)
            depth, count = (self.depths[arm], None) if kind == "depth" else self.pairs[arm]
            return RouteDecision(sample_id, depth, arm, w[arm].clone(), count)
        finally:
            model.train(previous_mode)

    def execute(
        self,
        model: TransformerClassifier,
        tokens: Tensor,
        route: RouteDecision,
        *,
        explore: bool = True,
    ) -> Execution:
        """Stage 2: rerun prefix, choose/execute each group, classify at selected depth.

        Task gradients flow through the selected Transformer path only. Contexts
        are detached by FeatureBank; discrete choices are never differentiated.
        Per-sample execution is intentional: different samples in a depth batch
        may retain different heads. It favors correctness/readability over speed.

        §4.2.1 Step 2 / §4.2.2 (p.443), Algorithm 2 line 7 (p.444): start
        from tokens again, including embeddings and all b prefix layers. Never
        reuse the detached Stage 1 hidden state. The caller enables training;
        DHTTrainer.step() backpropagates the final task loss through this fresh
        prefix graph, then optimizer.step() updates weights. No update happens
        halfway through the forward pass. Queue delay can make routing context
        older than these weights; retain it only for the original UCB observation.
        """
        hidden, mask = model.prefix(tokens)
        head_indices = {}
        decisions = []
        for group_index, start in enumerate(self.features.groups):
            if start >= route.total_depth:
                break
            count_arm, count_features = None, None
            if self.options.mode == "tree":
                w_count = self.features.aggregate(hidden, mask, "head", group_index)
                count_arm = self.head_ucb.select(w_count, explore=explore)
                count = self.counts[count_arm]
                count_features = w_count[count_arm].clone()
            else:
                count = route.head_count  # The integrated count stays fixed over groups.
            w_ts = self.features.aggregate(hidden, mask, "ts", group_index)
            selected = self.ts.select(w_ts, count, explore=explore)
            decisions.append(
                GroupDecision(group_index, start, count_arm, count_features, selected, w_ts)
            )
            for layer_index in range(start, start + self.config.model.group_size):
                head_indices[layer_index] = selected
                hidden = model.layers[layer_index](hidden, mask, selected)
        architecture = ArchitectureSpec(route.total_depth, head_indices)
        architecture.validate(self.config.model)
        return Execution(model.classify(hidden), architecture, route, decisions)

    def update(self, executions: list[Execution], losses: list[float]) -> list[dict]:
        """After the optimizer step, update only decisions with observed rewards.

        Losses came from the executed forward BEFORE the optimizer step. Using
        hypothetical post-update losses would require uncounted extra forwards.
        Prefix-only samples have no head events and never update head bandits.
        """
        if len(executions) != len(losses):
            raise ValueError("one observed loss is required per executed sample")
        products = []
        route_key = "depth" if self.options.mode == "tree" else "pair"
        for execution, loss in zip(executions, losses, strict=True):
            depth = execution.route.total_depth
            route_cost = depth if route_key == "depth" else depth * execution.route.head_count
            sample = {route_key: loss * route_cost}
            if route_key == "depth":
                sample.update(
                    {
                        f"head:{event.start_layer}": loss * len(event.head_indices)
                        for event in execution.groups
                    }
                )
            products.append(sample)
        rewards = self.rewards.observe_batch(products)
        for execution, sample in zip(executions, rewards, strict=True):
            route_reward = sample[route_key]
            if route_reward is not None:
                self.route_ucb.update(
                    execution.route.route_arm, execution.route.route_features, route_reward
                )
            for event in execution.groups:
                relevant = (
                    sample[f"head:{event.start_layer}"] if route_key == "depth" else route_reward
                )
                if relevant is None:
                    continue  # No previous observed batch for this reward key yet.
                if event.count_arm is not None:
                    self.head_ucb.update(event.count_arm, event.count_features, relevant)
                self.ts.update(
                    event.head_indices,
                    event.ts_features,
                    self.rewards.binary_feedback(relevant, self.options.feedback_rule),
                )
        return rewards

    @torch.no_grad()
    def predict(
        self, model: TransformerClassifier, tokens: Tensor
    ) -> tuple[Tensor, list[ArchitectureSpec]]:
        """Frozen mean-policy inference: no labels, queues, reward updates or RNG draws.

        Both routing and execution are included. The routing prefix is rerun here
        too, matching training's two-stage structure; its extra cost is reported.
        """
        old_mode = model.training
        model.eval()
        try:
            outputs, specs = [], []
            for index, sample in enumerate(tokens):
                route = self.route(model, sample[None], index, explore=False)
                execution = self.execute(model, sample[None], route, explore=False)
                outputs.append(execution.logits)
                specs.append(execution.architecture)
            return torch.cat(outputs), specs
        finally:
            model.train(old_mode)

    def state_dict(self) -> dict:
        """Persist controller, calibration, reward history, and TS sampling state."""
        return {
            "features": self.features.state_dict(),
            "route_ucb": self.route_ucb.state_dict(),
            "head_ucb": self.head_ucb.state_dict() if self.head_ucb else None,
            "ts": self.ts.state_dict(),
            "rewards": self.rewards.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore this same configured policy without recalibration."""
        self.features.load_state_dict(state["features"])
        self.route_ucb.load_state_dict(state["route_ucb"])
        if self.head_ucb:
            self.head_ucb.load_state_dict(state["head_ucb"])
        self.ts.load_state_dict(state["ts"])
        self.rewards.load_state_dict(state["rewards"])
