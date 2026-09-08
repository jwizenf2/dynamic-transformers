"""Head selection with preserved output positions (§4.1 p.442; decision D003)."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _UniqueIndexSelect(torch.autograd.Function):
    """Gather unique channels; backward assigns instead of accumulating duplicates.

    torch.index_select's general backward must support duplicate indices, so it
    accumulates. PyTorch2.14 MPS rejects that scatter with strict determinism.
    Our validated heads/channels are UNIQUE: the Jacobian simply writes each
    incoming gradient to its original position. index_copy_ expresses that exact
    derivative without atomic accumulation. All other positions are zero.

    This still allocates a dense gradient buffer; it does not eliminate dense
    optimizer work. The private function is only called after head validation.
    CPU gradcheck and CPU/MPS sliced-vs-dense gradients test this implementation.
    """

    @staticmethod
    def forward(ctx, value: Tensor, dim: int, indices: Tensor) -> Tensor:
        ctx.save_for_backward(indices)
        ctx.input_shape, ctx.dim = value.shape, dim
        return value.index_select(dim, indices)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None, None]:
        (indices,) = ctx.saved_tensors
        restored = gradient.new_zeros(ctx.input_shape)
        restored.index_copy_(ctx.dim, indices, gradient)
        return restored, None, None


class SelectableAttention(nn.Module):
    """Slice projection rows/columns before matmul; dense masking is a reference."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor,
        head_indices: Sequence[int] | None = None,
        *,
        dense_masked: bool = False,
    ) -> Tensor:
        """Compute [batch,sequence,hidden] attention with the requested original heads.

        With dense_masked=False, only selected projection rows are multiplied.
        The dense_masked=True path computes every head then zeroes inactive outputs;
        it is a semantic reference and must never be counted as physical savings.
        Output projection columns follow the same original head IDs as Q/K/V rows.
        """
        selected = tuple(range(self.num_heads)) if head_indices is None else tuple(head_indices)
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(type(h) is not int or not 0 <= h < self.num_heads for h in selected)
        ):
            raise ValueError("selected heads must be unique, nonempty and in range")
        if (
            attention_mask.dtype != torch.bool
            or attention_mask.shape != hidden.shape[:2]
            or not attention_mask.any(dim=1).all()
        ):
            raise ValueError("attention_mask must be boolean with at least one valid token per row")
        executed = tuple(range(self.num_heads)) if dense_masked else selected
        # A full, canonically ordered path should use the original matrices directly.
        # Gathering every row would add a copy and an autograd scatter for no saving.
        # A permutation of all heads still needs gathering to preserve its positions.
        full_order = executed == tuple(range(self.num_heads))
        columns = torch.tensor(
            [h * self.head_dim + j for h in executed for j in range(self.head_dim)],
            dtype=torch.long,
            device=hidden.device,
        )
        batch, length, _ = hidden.shape

        def project(projection: nn.Linear) -> Tensor:
            """Project only active rows, then reshape into [batch,heads,sequence,head_dim]."""
            result = F.linear(
                hidden,
                projection.weight
                if full_order
                else _UniqueIndexSelect.apply(projection.weight, 0, columns),
                projection.bias
                if full_order
                else _UniqueIndexSelect.apply(projection.bias, 0, columns),
            )
            return result.view(batch, length, len(executed), self.head_dim).transpose(1, 2)

        query, key, value = (project(layer) for layer in (self.query, self.key, self.value))
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~attention_mask[:, None, None, :], -torch.inf)
        heads = self.dropout(scores.softmax(dim=-1)) @ value
        if dense_masked:
            mask = hidden.new_tensor([h in selected for h in executed])
            heads = heads * mask[None, :, None, None]
        joined = heads.transpose(1, 2).reshape(batch, length, -1)
        output_weight = (
            self.output.weight
            if full_order
            else _UniqueIndexSelect.apply(self.output.weight, 1, columns)
        )
        return F.linear(joined, output_weight, self.output.bias)
