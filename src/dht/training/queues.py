"""Depth FIFO queues: §5 p.444, Algorithm 3 p.445.

Queue entries contain sample IDs and decision-time features, never autograd graphs.
At epoch end, partial queues are flushed in depth order. No sample is dropped.
"""

from dataclasses import asdict

from dht.policies.controller import RouteDecision


class DepthQueues:
    """One FIFO per total depth, releasing capacity-sized homogeneous depth batches."""

    def __init__(self, depths: tuple[int, ...], capacity: int):
        if capacity < 1 or not depths or len(set(depths)) != len(depths):
            raise ValueError("queues need unique depths and positive capacity")
        self.capacity = capacity
        self.pending: dict[int, list[RouteDecision]] = {depth: [] for depth in sorted(depths)}

    def push(self, decision: RouteDecision) -> None:
        """Retain the ORIGINAL routing context while model weights may change."""
        if decision.total_depth not in self.pending:
            raise ValueError("unknown queue depth")
        # Chunked routing (routing_batch_size > 1) may briefly overfill a queue;
        # FIFO order is kept and pop() releases capacity-sized batches in order.
        self.pending[decision.total_depth].append(decision)

    def pop(self, *, flush: bool = False) -> list[RouteDecision] | None:
        """Release the first full queue, or a partial queue when flushing an epoch."""
        for depth, queue in self.pending.items():
            if len(queue) >= self.capacity or (flush and queue):
                result = queue[: self.capacity]
                self.pending[depth] = queue[self.capacity :]
                return result
        return None

    def state_dict(self) -> dict:
        """Serialize dataclasses into plain records and detached CPU tensor copies."""
        return {
            "capacity": self.capacity,
            "pending": {
                depth: [asdict(record) for record in records]
                for depth, records in self.pending.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore FIFO order and delayed decision evidence at a step boundary."""
        if state["capacity"] != self.capacity or set(state["pending"]) != set(self.pending):
            raise ValueError("incompatible queue checkpoint")
        restored = {}
        for depth, records in state["pending"].items():
            restored[depth] = [RouteDecision(**record) for record in records]
            if any(record.total_depth != depth for record in restored[depth]):
                raise ValueError("queue record depth mismatch")
        self.pending = restored
