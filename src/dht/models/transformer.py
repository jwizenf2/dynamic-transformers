"""BERT-style selectable encoder; pretrained.py optionally imports base weights.

Direct construction is randomly initialized. Importing a pretrained encoder does
not claim equivalence to a particular paper initialization or classifier pooler.
"""

import torch
from torch import Tensor, nn

from dht.config import ModelConfig
from dht.models.architecture import ArchitectureSpec
from dht.models.attention import SelectableAttention


class EncoderLayer(nn.Module):
    """Post-norm attention and full-width FFN; selecting heads never shrinks residuals."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = SelectableAttention(config.hidden_size, config.num_heads, config.dropout)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, hidden: Tensor, mask: Tensor, heads: tuple[int, ...], *, dense_masked: bool = False
    ) -> Tensor:
        """Execute selected attention heads, then the unchanged FFN for this layer."""
        attended = self.attention(hidden, mask, heads, dense_masked=dense_masked)
        hidden = self.attention_norm(hidden + self.dropout(attended))
        return self.output_norm(hidden + self.dropout(self.ffn(hidden)))


class TransformerClassifier(nn.Module):
    """Fixed forward plus small primitives used by sequential DHT decisions.

    DHT calls embed(), executes the prefix, then chooses heads using intermediate
    hidden states. It cannot choose every group in advance: context changes as
    selected layers execute. The ordinary forward still accepts a complete spec.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.positions = nn.Embedding(config.max_sequence_length, config.hidden_size)
        self.token_types = nn.Embedding(2, config.hidden_size)
        self.embedding_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.total_depth)])
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def embed(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Validate tokens and produce [batch,sequence,hidden] embeddings and mask.

        Token 0 is padding; token position 0 must be valid because the classifier
        uses that position. DHT uses this same method in BOTH routing and training.
        """
        return self._embed_validated(input_ids, attention_mask)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        architecture: ArchitectureSpec | None = None,
        *,
        dense_masked: bool = False,
    ) -> Tensor:
        spec = architecture or ArchitectureSpec(self.config.total_depth)
        spec.validate(self.config)
        hidden, mask = self.embed(input_ids, attention_mask)
        for layer_index in range(spec.total_depth):
            hidden = self.layers[layer_index](
                hidden,
                mask,
                spec.heads_for(layer_index, self.config.num_heads),
                dense_masked=dense_masked,
            )
        return self.classify(hidden)

    def classify(self, hidden: Tensor) -> Tensor:
        """Shared first-token classifier at every possible exit depth (D002)."""
        return self.classifier(hidden[:, 0])

    def prefix(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Execute the unpruned prefix; caller chooses grad/no-grad and train/eval mode."""
        hidden, mask = self.embed(input_ids)
        full_heads = tuple(range(self.config.num_heads))
        for layer in self.layers[: self.config.prefix_depth]:
            hidden = layer(hidden, mask, full_heads)
        return hidden, mask

    def _embed_validated(
        self, input_ids: Tensor, attention_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Shared input checks prevent routing and ordinary forward from diverging."""
        if (
            input_ids.ndim != 2
            or input_ids.dtype != torch.long
            or input_ids.shape[0] == 0
            or not 1 <= input_ids.shape[1] <= self.config.max_sequence_length
        ):
            raise ValueError("input_ids must be nonempty int64 [batch, supported sequence length]")
        if torch.any((input_ids < 0) | (input_ids >= self.config.vocab_size)):
            raise ValueError("input token is outside the vocabulary")
        mask = input_ids.ne(0) if attention_mask is None else attention_mask
        if (
            mask.shape != input_ids.shape
            or mask.dtype != torch.bool
            or mask.device != input_ids.device
            or not mask[:, 0].all()
        ):
            raise ValueError(
                "boolean mask must match tokens and keep the first classification token"
            )
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = (
            self.tokens(input_ids)
            + self.positions(position_ids)[None]
            + self.token_types(torch.zeros_like(input_ids))
        )
        return self.dropout(self.embedding_norm(hidden)), mask
