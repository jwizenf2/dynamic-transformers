"""Per-sample "anatomy" helpers: what one input looks like to full BERT vs DHT.

Everything runs frozen on CPU from trusted local checkpoints: tokenization,
full-BERT vs DHT predictions, the exact layers/heads DHT executed, what DHT
would have predicted at other exits, leave-one-word-out importance, the
pretraining-style masked-word demo, and one training sample's history.
"""

import html
import json
import re
from functools import cache

import matplotlib.pyplot as plt
import nb_utils as u
import numpy as np
import torch
from matplotlib.patches import Rectangle

from dht.config import ExperimentConfig
from dht.models.architecture import ArchitectureSpec
from dht.models.pretrained import build_model
from dht.training.checkpointing import read_checkpoint
from dht.training.trainer import restore_controller

CLASSES = u.AG_NEWS_CLASSES
LENGTH = 128


# ------------------------------------------------------------------ loading
@cache
def tokenizer():
    tok = u.tokenizer()
    tok.enable_truncation(max_length=LENGTH)
    tok.enable_padding(length=LENGTH, pad_id=0, pad_token="[PAD]")
    return tok


@cache
def load(model_name: str):
    """(config, model, controller|None) for a validation-selected checkpoint."""
    state = read_checkpoint(u.RUNS / u.EXPORTS[model_name] / "best.pt")  # trusted local
    config = ExperimentConfig.model_validate(state["config"])
    model = build_model(config, initialize=False)
    model.load_state_dict(state["model"])
    model.eval()
    return config, model, (restore_controller(config, state) if config.dht else None)


def encode(text: str) -> torch.Tensor:
    return torch.tensor([tokenizer().encode(text).ids])


def token_view(text: str) -> dict:
    """WordPiece tokens exactly as the model receives them (first 128 positions)."""
    encoding = tokenizer().encode(text)
    tokens = encoding.tokens
    real = sum(1 for t in tokens if t != "[PAD]")
    return {"tokens": tokens[:real], "real": real, "pad": LENGTH - real}


# --------------------------------------------------------------- predictions
@torch.no_grad()
def full_bert(text: str) -> torch.Tensor:
    _, model, _ = load("Fixed BERT")
    return model(encode(text), architecture=ArchitectureSpec(12)).softmax(-1)[0]


@torch.no_grad()
def dht(text: str, name: str = "T-DHT") -> tuple[torch.Tensor, dict]:
    _, model, controller = load(name)
    logits, trace = controller.explain(model, encode(text))
    return logits.softmax(-1)[0], trace


@torch.no_grad()
def dht_at_every_exit(text: str, name: str = "T-DHT") -> dict[int, torch.Tensor]:
    """What the SAME DHT network predicts if forced to exit at 3/6/9/12 (all heads).

    DHT shares one classifier across exits and trains it at every depth, so each
    exit is a meaningful prediction. This shows whether the router chose well.
    """
    _, model, _ = load(name)
    ids = encode(text)
    return {d: model(ids, architecture=ArchitectureSpec(d)).softmax(-1)[0] for d in (3, 6, 9, 12)}


def heads_by_layer(trace: dict) -> dict[int, tuple[int, ...]]:
    layers = {layer: tuple(range(12)) for layer in range(3)}
    for group in trace["groups"]:
        for layer in range(group["start_layer"], group["start_layer"] + 3):
            layers[layer] = tuple(group["selected_heads"])
    return layers


# ------------------------------------------------------------------- figures
def draw_grid(ax, layers: dict[int, tuple[int, ...]], color: str, title: str) -> None:
    """12 layers × 12 heads: filled = computed, pale = skipped head, gray = layer not run."""
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(11.5, -0.5)
    for layer in range(12):
        for head in range(12):
            if layer not in layers:
                face, edge = "#f0efec", "#e1e0d9"
            elif head in layers[layer]:
                face, edge = color, u.SURFACE
            else:
                face, edge = "#ffffff", color
            ax.add_patch(
                Rectangle(
                    (head - 0.45, layer - 0.45),
                    0.9,
                    0.9,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=0.8,
                )
            )
    ax.axhline(2.5, color=u.INK, lw=1, ls=":")
    ax.set_xticks(range(12), [str(h) for h in range(12)], fontsize=7)
    ax.set_yticks(range(12), [f"L{layer}" for layer in range(12)], fontsize=7)
    ax.set_xlabel("attention head")
    ax.grid(False)
    ax.set_title(title, fontsize=10)


def anatomy_figure(text: str, label: int | None = None, name: str = "T-DHT"):
    fixed_p = full_bert(text)
    dht_p, trace = dht(text, name)
    layers = heads_by_layer(trace)
    counts = tuple(g["head_count"] for g in trace["groups"])
    compute = u.relative_matmul_flops(trace["total_depth"], counts)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    y = np.arange(4)
    ax.barh(y - 0.2, fixed_p.numpy(), 0.38, color=u.MODEL_COLORS["Fixed BERT"], label="Full BERT")
    ax.barh(y + 0.2, dht_p.numpy(), 0.38, color=u.MODEL_COLORS[name], label=name)
    for i in range(4):
        ax.text(
            fixed_p[i] + 0.01, i - 0.2, f"{fixed_p[i]:.2f}", va="center", fontsize=8, color=u.INK2
        )
        ax.text(dht_p[i] + 0.01, i + 0.2, f"{dht_p[i]:.2f}", va="center", fontsize=8, color=u.INK2)
    names = [c + ("  ← true" if label == i else "") for i, c in enumerate(CLASSES)]
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("predicted probability")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8)
    ax.set_title("class probabilities", pad=22)
    draw_grid(
        axes[1],
        {layer: tuple(range(12)) for layer in range(12)},
        u.MODEL_COLORS["Fixed BERT"],
        "Full BERT: 12 layers × 12 heads (1.00x)",
    )
    draw_grid(
        axes[2],
        layers,
        u.MODEL_COLORS[name],
        f"{name}: {trace['total_depth']} layers, heads/group {counts or '—'} ({compute:.2f}x)",
    )
    fig.tight_layout()
    return fig, fixed_p, dht_p, trace


# ------------------------------------------------------------- word importance
WORD = re.compile(r"\S+")


@torch.no_grad()
def word_importance(text: str, predictor) -> tuple[list[str], np.ndarray, int]:
    """Leave-one-word-out: drop in the prediction's log-odds MARGIN when a word is deleted.

    margin = log P(predicted class) - max log P(other class) (= logit difference).
    Unlike the probability itself it does not saturate at 0.999, so confident
    predictions still show which words carry the evidence. Positive = the word
    supported the prediction; negative = it argued against it. For DHT the router
    re-decides on every edited input, exactly as it would in deployment.
    """
    words = WORD.findall(text)
    base = predictor(text)
    target = int(base.argmax())

    def margin(probabilities: torch.Tensor) -> float:
        logs = probabilities.clamp_min(1e-12).log()
        others = torch.cat([logs[:target], logs[target + 1 :]])
        return float(logs[target] - others.max())

    reference = margin(base)
    scores = []
    for i in range(len(words)):
        edited = " ".join(words[:i] + words[i + 1 :]) or "[UNK]"
        scores.append(reference - margin(predictor(edited)))
    return words, np.asarray(scores), target


def highlight(words: list[str], scores: np.ndarray, title: str) -> str:
    """HTML: orange = supports the prediction, blue = against it; darker = stronger."""
    scale = max(1e-6, float(np.abs(scores).max()))
    spans = []
    for word, score in zip(words, scores, strict=True):
        alpha = min(1.0, abs(score) / scale)
        rgb = "235,104,52" if score > 0 else "42,120,214"
        spans.append(
            f'<span title="{score:+.3f}" style="background:rgba({rgb},{0.85 * alpha:.2f});'
            f'padding:1px 2px;border-radius:3px;margin:1px">{html.escape(word)}</span>'
        )
    return f'<div style="margin:4px 0"><b>{html.escape(title)}</b><br>{" ".join(spans)}</div>'


# ------------------------------------------------------- pretraining demo (MLM)
@cache
def pretrained_mlm():
    """Original pretrained BERT-base + its masked-LM head (before any AG News training)."""
    from safetensors import safe_open

    config, _, _ = load("Fixed BERT")
    # Config paths are repo-relative; notebooks run from notebooks/.
    config = config.model_copy(
        update={"pretrained_weights": str(u.ROOT / config.pretrained_weights)}
    )
    model = build_model(config, initialize=True)  # imports the pinned, hash-checked encoder
    model.eval()
    path = u.ROOT / "artifacts/pretrained/bert-base-uncased/model.safetensors"
    with safe_open(str(path), "pt") as f:
        head = {k: f.get_tensor(k) for k in f.keys() if k.startswith("cls.predictions")}
    return model, head


@torch.no_grad()
def fill_mask(text_with_mask: str, top: int = 5) -> list[tuple[str, float]]:
    model, head = pretrained_mlm()
    tok = tokenizer()
    ids = encode(text_with_mask)
    position = tok.encode(text_with_mask).tokens.index("[MASK]")
    hidden, mask = model.embed(ids)
    for layer in model.layers:
        hidden = layer(hidden, mask, tuple(range(12)))
    h = hidden[0, position]
    h = torch.nn.functional.gelu(
        h @ head["cls.predictions.transform.dense.weight"].T
        + head["cls.predictions.transform.dense.bias"]
    )
    h = torch.nn.functional.layer_norm(
        h,
        (768,),
        head["cls.predictions.transform.LayerNorm.gamma"],
        head["cls.predictions.transform.LayerNorm.beta"],
        1e-12,
    )
    probs = (h @ model.tokens.weight.T + head["cls.predictions.bias"]).softmax(-1)
    values, indices = probs.topk(top)
    return [(tok.id_to_token(int(i)), float(v)) for v, i in zip(values, indices, strict=True)]


# ---------------------------------------------------------- training history
def train_text(sample_id: int) -> tuple[str, int]:
    """Prepared-train index -> raw AG News text and label (same join as training)."""
    with np.load(u.ROOT / "datasets/ag_news_wordpiece_v1/train.npz") as arrays:
        source = int(arrays["source_indices"][sample_id])
        label = int(arrays["labels"][sample_id])
    raw = u.load_dataset("AG News", "train")
    return raw.text.iloc[source], label


def training_history(sample_ids: set[int], model_name: str = "T-DHT") -> dict[int, list[dict]]:
    """Every time each training sample was trained on (once per epoch).

    Reads the streamed per-step logs; for the resumed T-DHT run, steps <= 8,000
    come from the first attempt and later steps from the resume (D015).
    """
    run = u.RUNS / u.EXPORTS[model_name]
    logs = [(run / "metrics.jsonl", None)]
    if model_name == "T-DHT":
        logs = [
            (u.RUNS / "ag-news-tree-full-seed7" / "metrics.jsonl", 8000),
            (run / "metrics.jsonl", None),
        ]
    found: dict[int, list[dict]] = {i: [] for i in sample_ids}
    for path, until in logs:
        with path.open() as stream:
            for line in stream:
                if not any(f'"sample_id": {i},' in line for i in sample_ids):
                    continue
                step = json.loads(line)
                if until and step["step"] > until:
                    break
                for sample in step["samples"]:
                    if sample["sample_id"] in found:
                        heads = sample["architecture"]["head_indices"]
                        found[sample["sample_id"]].append(
                            {
                                "epoch": step["epoch"],
                                "step": step["step"],
                                "depth": sample["architecture"]["total_depth"],
                                "heads/group": "/".join(
                                    str(len(heads[k])) for k in sorted(heads, key=int)[::3]
                                )
                                or "—",
                                "loss": sample["loss"],
                                "rewards": {
                                    k: round(v, 3)
                                    for k, v in (sample.get("rewards") or {}).items()
                                    if v is not None
                                },
                            }
                        )
    return found


def compare_figure(text: str, label: int | None = None, names=("T-DHT", "F-DHT")):
    """Full BERT plus several DHT variants on ONE sentence: probabilities and grids."""
    probs = {"Full BERT": full_bert(text)}
    traces = {}
    for name in names:
        probs[name], traces[name] = dht(text, name)
    fig, axes = plt.subplots(
        1,
        2 + len(names),
        figsize=(5 + 3.6 * (1 + len(names)), 4.1),
        gridspec_kw={"width_ratios": [1.4] + [1] * (1 + len(names))},
    )
    ax = axes[0]
    width = 0.8 / len(probs)
    for k, (name, p) in enumerate(probs.items()):
        offset = (k - (len(probs) - 1) / 2) * width
        ax.barh(
            np.arange(4) + offset, p.numpy(), width * 0.92, color=u.MODEL_COLORS[name], label=name
        )
    ax.set_yticks(range(4), [c + ("  ← true" if label == i else "") for i, c in enumerate(CLASSES)])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("predicted probability")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8)
    ax.set_title("class probabilities", pad=22)
    draw_grid(
        axes[1],
        {layer: tuple(range(12)) for layer in range(12)},
        u.MODEL_COLORS["Full BERT"],
        "Full BERT (1.00x)",
    )
    for ax, name in zip(axes[2:], names, strict=True):
        trace = traces[name]
        counts = tuple(g["head_count"] for g in trace["groups"])
        compute = u.relative_matmul_flops(trace["total_depth"], counts)
        verdict = CLASSES[int(probs[name].argmax())]
        draw_grid(
            ax,
            heads_by_layer(trace),
            u.MODEL_COLORS[name],
            f"{name}: {trace['total_depth']}L {counts or ''} → {verdict} ({compute:.2f}x)",
        )
    fig.tight_layout()
    return fig, probs, traces
