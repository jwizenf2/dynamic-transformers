"""Explicit BERT tensor import into OUR selectable encoder; no author DHT code.

We do not run a dense Hugging Face backbone then mask its output. We copy the
standard pretrained embedding/encoder tensors into the same independently
implemented modules used by fixed and sliced execution. The MLM/NSP heads and
pooler are not used; Figure1 p.445 motivates our shared first-token linear head,
but neither this pooling detail nor this initialization is established by DHT.
"""

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

from dht.ag_news import sha256
from dht.config import ExperimentConfig
from dht.models.transformer import TransformerClassifier


def bert_tensor_mapping(total_depth: int) -> dict[str, str]:
    """Our parameter name -> standard Hugging Face BertForMaskedLM parameter name.

    No shape transposition or head reordering: BERT packs head0, then head1, etc.
    This exact layout is crucial when selecting QKV rows/output columns later.
    """
    mapping = {
        "tokens.weight": "bert.embeddings.word_embeddings.weight",
        "positions.weight": "bert.embeddings.position_embeddings.weight",
        "token_types.weight": "bert.embeddings.token_type_embeddings.weight",
        "embedding_norm.weight": "bert.embeddings.LayerNorm.weight",
        "embedding_norm.bias": "bert.embeddings.LayerNorm.bias",
    }
    components = {
        "attention.query": "attention.self.query",
        "attention.key": "attention.self.key",
        "attention.value": "attention.self.value",
        "attention.output": "attention.output.dense",
        "attention_norm": "attention.output.LayerNorm",
        "ffn.0": "intermediate.dense",
        "ffn.2": "output.dense",
        "output_norm": "output.LayerNorm",
    }
    for index in range(total_depth):
        for local, source in components.items():
            for kind in ("weight", "bias"):
                mapping[f"layers.{index}.{local}.{kind}"] = (
                    f"bert.encoder.layer.{index}.{source}.{kind}"
                )
    return mapping


def import_bert_encoder(model: TransformerClassifier, tensors: Mapping[str, Tensor]) -> None:
    """Require every encoder tensor with an exact shape; never silently partially load.

    Validate the entire mapping before modifying weights. We retain the randomly
    initialized task classifier, rather than importing a head trained on AG News.
    """
    own = model.state_dict()
    mapping = bert_tensor_mapping(model.config.total_depth)
    # The pinned safetensors asset retains original TensorFlow-style LayerNorm
    # gamma/beta names. They are scale/offset, exactly PyTorch weight/bias; this
    # is a name conversion only, not a numerical adjustment or missing-weight fill.
    resolved = dict(tensors)
    for source, tensor in tensors.items():
        canonical = source.replace(".LayerNorm.gamma", ".LayerNorm.weight").replace(
            ".LayerNorm.beta", ".LayerNorm.bias"
        )
        if canonical != source:
            if canonical in resolved:
                raise ValueError(f"ambiguous duplicate pretrained tensor: {canonical}")
            resolved[canonical] = tensor
    for local, source in mapping.items():
        if source not in resolved or resolved[source].shape != own[local].shape:
            raise ValueError(f"missing/incompatible pretrained tensor: {source}")
        if not torch.isfinite(resolved[source]).all():
            raise ValueError(f"nonfinite pretrained tensor: {source}")
    own.update({local: resolved[source] for local, source in mapping.items()})
    model.load_state_dict(own, strict=True)


def build_model(config: ExperimentConfig, *, initialize: bool = True) -> TransformerClassifier:
    """Build on CPU before device transfer; resume/evaluate skip pretrained reloading.

    SHA256 is part of the validated experiment config/checkpoint. Loading is local
    safetensors only, with no automatic model or code downloads.
    """
    model = TransformerClassifier(config.model)
    if config.pretrained_weights is not None and initialize:
        from safetensors.torch import load_file

        path = Path(config.pretrained_weights)
        if sha256(path) != config.pretrained_sha256:
            raise ValueError("pretrained weight hash mismatch")
        import_bert_encoder(model, load_file(str(path), device="cpu"))
        # Match BERT's usual small task-head scale, equally for fixed and DHT.
        torch.nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(model.classifier.bias)
    return model
