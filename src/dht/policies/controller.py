"""Tree and integrated DHT execution: §4.2, Algorithm 2, Figure 1.

Read route() and execute() in order. route() observes a no-gradient prefix and
selects depth (or depth/count); execute() reruns the prefix with task gradients
and selects each group's heads using its current intermediate representation.
The training loop owns queues and optimizer steps; this class owns bandit state.
"""

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor

from dht.bandits.features import FeatureBank
from dht.bandits.linear_ucb import LinearUCB
from dht.bandits.logistic_ts import LogisticThompsonSampling
from dht.config import ExperimentConfig
from dht.models.architecture import ArchitectureSpec
from dht.models.attention import unique_index_select
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
        self,
        model: TransformerClassifier,
        tokens: Tensor,
        sample_id: int,
        *,
        explore: bool = True,
        trace: dict | None = None,
    ) -> RouteDecision:
        """Stage 1: full-head prefix without dropout/task gradients, then route UCB.

        No optimizer or bandit update occurs here. Training-mode restoration is
        essential: otherwise routing would accidentally disable dropout afterward.

        Paper evidence: §4.2.1 Step 1 (p.443), §5 (p.444), Algorithm 2 lines
        2–6 (p.444), and Figure 1 Stage 1 (p.445). "Frozen" means no update
        during this pass; these are the SAME trainable prefix parameters used by
        execute(), not a frozen copy. @no_grad prevents building the task graph;
        eval() separately disables dropout (our D008 choice, not a paper detail).

        ``trace`` (optional, read-only) receives the per-arm UCB scores that led
        to the choice. It never changes a decision or consumes RNG; see explain().
        """
        if len(tokens) != 1:
            raise ValueError("routing expects one sample")
        previous_mode = model.training
        model.eval()
        try:
            hidden, mask = model.prefix(tokens)
            kind = "depth" if self.options.mode == "tree" else "pair"
            w = self.features.aggregate(hidden, mask, kind)
            if trace is not None:
                trace["route_kind"] = kind
                trace["route_actions"] = list(self.depths if kind == "depth" else self.pairs)
                trace["route_scores"] = self.route_ucb.scores(w, explore=explore).tolist()
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
        trace: dict | None = None,
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

        ``trace`` (optional) records each group's count-UCB scores (tree only) and
        per-head TS probabilities. With explore=False these are exactly the
        posterior-mean quantities that determined the selection.
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
            if trace is not None and not explore:
                # Posterior-mean head probabilities; exploring TS would consume RNG.
                trace.setdefault("groups", []).append(
                    {
                        "start_layer": start,
                        "count_scores": (
                            self.head_ucb.scores(w_count, explore=False).tolist()
                            if count_arm is not None
                            else None
                        ),
                        "head_count": count,
                        "head_probabilities": self.ts.probabilities(w_ts).tolist(),
                    }
                )
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
                    vectorized=self.options.execution == "batched",
                )
        return rewards

    # ------------------------------------------------------------------ batched path
    # route_batch()/execute_batch() are the SAME algorithm as route()/execute(), laid
    # out for parallel hardware (decision D017). Per-sample decisions depend only on
    # that sample's own hidden state, so batching changes no decision logic:
    #   * one prefix pass for the whole batch instead of one per sample;
    #   * one GPU->CPU context transfer per group per batch (aggregate_many);
    #   * each c-layer group runs once per HEAD-COUNT partition (≤3 here), with every
    #     sample keeping its own head identities inside one batched matmul;
    #   * TS parameters are pre-drawn in per-sample (sample-major) order, so the
    #     random stream and therefore exploring choices match execute() exactly.
    # Only floating-point summation order can differ (batched kernels).

    def _select_many(self, ucb: LinearUCB, contexts: list[Tensor], explore: bool) -> list[int]:
        """Per-sample UCB choices; D018 device mode scores the batch in one solve."""
        if self.options.context_pooling == "device":
            return ucb.select_many(torch.stack(contexts), explore=explore)
        return [ucb.select(w, explore=explore) for w in contexts]

    @torch.no_grad()
    def route_batch(
        self,
        model: TransformerClassifier,
        tokens: Tensor,
        sample_ids: list[int],
        *,
        explore: bool = True,
        keep_prefix: dict | None = None,
    ) -> list[RouteDecision]:
        """Stage 1 for many samples: ONE no-grad prefix pass, then per-sample UCB.

        Identical to calling route() per sample: the UCB is not updated between
        these selections, exactly as it is not updated while queues fill.
        ``keep_prefix`` (inference only) receives the no-grad prefix output so
        execute_batch() can reuse it instead of recomputing the same three layers.
        """
        if len(tokens) != len(sample_ids):
            raise ValueError("one sample ID is required per routed sample")
        previous_mode = model.training
        model.eval()
        try:
            hidden, mask = model.prefix(tokens)
            if keep_prefix is not None:
                keep_prefix.update(hidden=hidden, mask=mask)
            kind = "depth" if self.options.mode == "tree" else "pair"
            decisions = []
            contexts = self.features.aggregate_many(hidden, mask, kind)
            arms = self._select_many(self.route_ucb, contexts, explore)
            for sample_id, w, arm in zip(sample_ids, contexts, arms, strict=True):
                depth, count = (self.depths[arm], None) if kind == "depth" else self.pairs[arm]
                decisions.append(RouteDecision(sample_id, depth, arm, w[arm].clone(), count))
            return decisions
        finally:
            model.train(previous_mode)

    def execute_batch(
        self,
        model: TransformerClassifier,
        tokens: Tensor,
        routes: list[RouteDecision],
        *,
        explore: bool = True,
        prefix: dict | None = None,
    ) -> list[Execution]:
        """Stage 2 for a batch (mixed depths allowed): one fresh prefix, group by group.

        ``rows`` tracks which original batch position each current hidden row holds.
        Samples whose selected depth is reached are classified and leave the batch;
        the rest are split by head count, run through the group's c layers, and
        rejoined. Returned executions are in the ORIGINAL order, each holding its
        own logits row; gradients flow exactly as in per-sample execute().

        ``prefix`` may carry route_batch()'s prefix output ONLY for frozen
        inference (eval mode, no grad), where it equals a rerun exactly. Training
        must always rerun the prefix with gradients (Algorithm 2 line 7).
        """
        if prefix is not None and (model.training or torch.is_grad_enabled()):
            raise ValueError("prefix reuse is only valid for no-grad eval-mode inference")
        if len(tokens) != len(routes) or not routes:
            raise ValueError("one route decision is required per executed sample")
        groups = self.features.groups
        draws = [
            [
                self.ts.sample_parameters() if explore else None
                for start in groups
                if start < route.total_depth
            ]
            for route in routes
        ]
        hidden, mask = (prefix["hidden"], prefix["mask"]) if prefix else model.prefix(tokens)
        rows = list(range(len(routes)))
        decisions: list[list[GroupDecision]] = [[] for _ in routes]
        head_indices: list[dict[int, tuple[int, ...]]] = [{} for _ in routes]
        pieces, positions = [], []

        def keep(indices: list[int]) -> None:
            nonlocal hidden, mask, rows
            if not indices:
                rows = []  # Every remaining sample exited; nothing left to execute.
            elif indices != list(range(len(rows))):
                hidden = unique_index_select(hidden, 0, indices)
                mask = mask[torch.tensor(indices, dtype=torch.long, device=mask.device)]
                rows = [rows[i] for i in indices]

        for group_index, start in enumerate(groups):
            exiting = [i for i, r in enumerate(rows) if routes[r].total_depth <= start]
            if exiting:
                pieces.append(model.classify(unique_index_select(hidden, 0, exiting)))
                positions.extend(rows[i] for i in exiting)
                keep([i for i, r in enumerate(rows) if routes[r].total_depth > start])
            if not rows:
                break
            w_ts = self.features.aggregate_many(hidden, mask, "ts", group_index)
            w_count = (
                self.features.aggregate_many(hidden, mask, "head", group_index)
                if self.options.mode == "tree"
                else None
            )
            chosen = []
            count_arms = (
                self._select_many(self.head_ucb, w_count, explore) if w_count is not None else None
            )
            for i, r in enumerate(rows):
                count_arm, count_features = None, None
                if w_count is not None:
                    count_arm = count_arms[i]
                    count = self.counts[count_arm]
                    count_features = w_count[i][count_arm].clone()
                else:
                    count = routes[r].head_count
                selected = self.ts.select(
                    w_ts[i], count, explore=explore, parameters=draws[r][group_index]
                )
                decisions[r].append(
                    GroupDecision(group_index, start, count_arm, count_features, selected, w_ts[i])
                )
                chosen.append(selected)
            partitions = defaultdict(list)
            for i, selected in enumerate(chosen):
                partitions[len(selected)].append(i)
            order = [i for count in sorted(partitions) for i in partitions[count]]
            keep(order)  # Contiguous head-count partitions (no-op if already ordered).
            chosen = [chosen[i] for i in order]
            outputs, offset = [], 0
            for count in sorted(partitions):
                size = len(partitions[count])
                part_rows = list(range(offset, offset + size))
                part = hidden if size == len(rows) else unique_index_select(hidden, 0, part_rows)
                part_mask = mask[offset : offset + size]
                heads = chosen[offset : offset + size]
                for layer_index in range(start, start + self.config.model.group_size):
                    part = model.layers[layer_index](part, part_mask, heads)
                    for row, selected in zip(rows[offset : offset + size], heads, strict=True):
                        head_indices[row][layer_index] = selected
                outputs.append(part)
                offset += size
            hidden = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
        if rows:
            pieces.append(model.classify(hidden))
            positions.extend(rows)
        logits = torch.cat(pieces) if len(pieces) > 1 else pieces[0]
        inverse = sorted(range(len(positions)), key=positions.__getitem__)
        logits = unique_index_select(logits, 0, inverse)
        executions = []
        for index, route in enumerate(routes):
            architecture = ArchitectureSpec(route.total_depth, head_indices[index])
            architecture.validate(self.config.model)
            executions.append(
                Execution(logits[index : index + 1], architecture, route, decisions[index])
            )
        return executions

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
            if self.options.execution == "batched":
                # One prefix pass serves routing AND execution at inference (D017):
                # with dropout off and no gradients the rerun would be identical.
                shared: dict = {}
                routes = self.route_batch(
                    model, tokens, list(range(len(tokens))), explore=False, keep_prefix=shared
                )
                executions = self.execute_batch(model, tokens, routes, explore=False, prefix=shared)
                return (
                    torch.cat([e.logits for e in executions]),
                    [e.architecture for e in executions],
                )
            outputs, specs = [], []
            for index, sample in enumerate(tokens):
                route = self.route(model, sample[None], index, explore=False)
                execution = self.execute(model, sample[None], route, explore=False)
                outputs.append(execution.logits)
                specs.append(execution.architecture)
            return torch.cat(outputs), specs
        finally:
            model.train(old_mode)

    @torch.no_grad()
    def explain(self, model: TransformerClassifier, tokens: Tensor) -> tuple[Tensor, dict]:
        """Frozen single-sample decision trace for analysis notebooks.

        Runs the same route()/execute() code as predict() with explore=False and
        returns logits plus the scores behind every choice: route UCB means per
        arm (depth arms for tree, (depth,count) pairs for integrated), per-group
        count-UCB means (tree only) and TS posterior-mean head probabilities.
        Nothing is updated and no RNG is consumed, so it is safe after training.
        """
        old_mode = model.training
        model.eval()
        try:
            trace: dict = {"mode": self.options.mode}
            route = self.route(model, tokens, 0, explore=False, trace=trace)
            execution = self.execute(model, tokens, route, explore=False, trace=trace)
            trace["total_depth"] = route.total_depth
            trace["route_arm"] = route.route_arm
            trace.setdefault("groups", [])
            for record, group in zip(trace["groups"], execution.groups, strict=True):
                record["selected_heads"] = list(group.head_indices)
            return execution.logits, trace
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
