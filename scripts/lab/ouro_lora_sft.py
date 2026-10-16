"""LoRA SFT of Ouro on Hermes function calling, in MLX: 2608.18171's recipe or depth-aware.

    python scripts/lab/ouro_lora_sft.py --loss ouro  --name ouro26-hermes-ouroloss
    python scripts/lab/ouro_lora_sft.py --loss depth --name ouro26-hermes-depth
    python scripts/lab/ouro_lora_sft.py --loss depth --max-steps 30 --name timing   # timing run

Reproduction target: "Looped Language Models Improve Compositional Tool Calling" (2608.18171,
§4.1, App. A–B): released Ouro, NousResearch hermes-function-calling-v1 (ChatML, Hermes
``<tool_call>`` JSON), 90/10 split, max length 4,096, 2 epochs, AdamW, LR 2e-5 cosine with 3%
warmup, effective batch 4, LoRA rank 32, bf16, and Ouro's Stage-I objective

    L = Σ_t p_t · CE_t + β · KL(p ‖ Uniform),  β = 0.1,

where p_t is the per-token exit distribution from the gate (λ_t = σ(gate(h_t)),
p_t = λ_t ∏_{j<t} (1 − λ_j), last step takes the remainder) and CE_t the token cross-entropy of
loop t's logits. The gate head is trained in this mode.

Depth-aware mode (ours, ``--loss depth``): per-request serving runs a whole answer at one depth R,
and loops 1..R compute the same thing whatever the maximum is, so one 4-loop forward gives the
logits of every fixed-depth model. The loss is their average, so every depth is trained to be
good on its own,

    L = (1/4) Σ_t CE_t + α · (1/3) Σ_{t<4} KL(stopgrad(softmax(z_4)) ‖ softmax(z_t)),

with optional self-distillation from loop 4 (α = ``--distill``, default 0). The gate is frozen.

Choices 2608.18171 leaves unspecified, recorded here: Hermes files func-calling.json (multi-turn;
func-calling-singleturn.json is its first-turn prefix, so it is not added) and
glaive-function-calling-5k.json (JSON-mode files excluded); conversations over 4,096 tokens are
dropped; loss on assistant turns only (content plus ``<|im_end|>``); LoRA on q/k/v/o and
gate/up/down projections of every layer (shared across loops), alpha 64, no dropout; seed 0;
AdamW weight decay 0; loss averaged per sequence, then over the 4 sequences of a step (not
token-weighted); the trailing newline after ``<|im_end|>`` is also a target. Note that HF's own
``modeling_ouro.py`` loss uses CE of Σ_t p_t·logits_t; we follow the paper's Σ_t p_t·CE_t.
Resume: ``--resume`` continues from ``state/`` (adapters, optimizer state, data order and
position, saved atomically every ``--save-every`` steps). On SIGUSR1 (the lab broker asking the
job to yield, with 60 s to checkpoint) the step in progress finishes, ``state/`` is saved and the
process re-raises SIGUSR1, so the lab still records "preempted"; resubmitting the same command
with ``--resume`` continues from that exact step. Logging: validation CE at step 0 and every
``--eval-every`` steps; train loss also as a moving average over the last 50 steps, because a
single step is only 4 conversations and swings a lot.
Framework: MLX on the ouro-mlx port (the official HF modeling code does not load with
transformers 5.x), with per-layer gradient checkpointing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from collections import deque
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from ouro_mlx.model import OuroConfig, OuroDecoderLayer, OuroForCausalLM
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ouro_headroom import OURO_REVISIONS  # noqa: E402

HERMES = "NousResearch/hermes-function-calling-v1"
HERMES_REVISION = "dae3e1d28cfbcf4b915c04ea1e072030529b4bda"
HERMES_FILES = ("func-calling.json", "glaive-function-calling-5k.json")
ROLES = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}
LORA_TARGETS = {
    "self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) · B A dropout(x); W frozen, A Gaussian, B zero at start."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        out_dim, in_dim = base.weight.shape
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = mx.random.normal((in_dim, rank)) * (1 / math.sqrt(in_dim))
        self.lora_b = mx.zeros((rank, out_dim))

    def __call__(self, x):
        y = self.base(x)
        z = (self.dropout(x) @ self.lora_a.astype(x.dtype)) @ self.lora_b.astype(x.dtype)
        return y + self.scale * z

    def merged_weight(self) -> mx.array:
        delta = (self.lora_a @ self.lora_b).T * self.scale  # [out, in]
        return (self.base.weight.astype(mx.float32) + delta).astype(self.base.weight.dtype)


def add_lora(model: OuroForCausalLM, rank: int, alpha: float, dropout: float) -> None:
    for layer in model.model.layers:
        for block, names in LORA_TARGETS.items():
            mod = getattr(layer, block)
            for name in names:
                setattr(mod, name, LoRALinear(getattr(mod, name), rank, alpha, dropout))


def checkpoint_layers() -> None:
    """Recompute each decoder layer in the backward pass (mlx-lm's grad_checkpoint pattern)."""
    fn = OuroDecoderLayer.__call__

    def checkpointed(layer, x, mask, cache):
        def inner(params, x):
            layer.update(params)
            return fn(layer, x, mask, cache)

        return mx.checkpoint(inner)(layer.trainable_parameters(), x)

    OuroDecoderLayer.__call__ = checkpointed


def conversations(tokenizer, max_len: int) -> tuple[list, int]:
    """Token ids and assistant-target masks per conversation; drops ones over max_len."""
    out, dropped = [], 0
    for fname in HERMES_FILES:
        path = hf_hub_download(HERMES, fname, repo_type="dataset", revision=HERMES_REVISION)
        for row in json.loads(Path(path).read_text(encoding="utf-8")):
            ids, mask = [], []
            for m in row["conversations"]:
                role = ROLES[m["from"]]
                head = tokenizer.encode(f"<|im_start|>{role}\n", add_special_tokens=False)
                body = tokenizer.encode(m["value"] + "<|im_end|>\n", add_special_tokens=False)
                ids += head + body
                mask += [0] * len(head) + [int(role == "assistant")] * len(body)
            if len(ids) > max_len:
                dropped += 1
                continue
            out.append((f"{fname}:{row['id']}", ids, mask))
    return out, dropped


def exit_pdf(gates: list[mx.array]) -> mx.array:
    """[B, L, T] exit distribution from per-loop gate logits ([B, L] each)."""
    remaining = mx.ones_like(gates[0])
    pdf = []
    for t, g in enumerate(gates):
        lam = mx.sigmoid(g)
        if t < len(gates) - 1:
            pdf.append(lam * remaining)
            remaining = remaining * (1 - lam)
        else:
            pdf.append(remaining)
    return mx.stack(pdf, -1)


CE_CHUNK = 1024  # tokens per vocabulary-projection chunk


def token_ce(model, h: mx.array, targets: mx.array) -> mx.array:
    """[1, L] cross-entropy of lm_head(h), in checkpointed sequence chunks.

    The full-vocabulary fp32 logits (L x 49,152 per loop) dominated memory at L = 4,096; each
    chunk's logits are recomputed in the backward pass instead of kept (lm_head is frozen).
    """

    def chunk(tc):
        # Targets are closed over: mx.checkpoint differentiates every positional argument.
        return mx.checkpoint(
            lambda hc: nn.losses.cross_entropy(
                model.lm_head(hc).astype(mx.float32), tc, reduction="none"
            )
        )

    parts = [
        chunk(targets[:, i : i + CE_CHUNK])(h[:, i : i + CE_CHUNK])
        for i in range(0, h.shape[1], CE_CHUNK)
    ]
    return mx.concatenate(parts, 1)


def loss_fn(model, ids, mask, args):
    """Returns (loss, per-loop masked CE [T]) for one sequence ([1, L] ids and mask)."""
    inputs, targets = ids[:, :-1], ids[:, 1:]
    w = mask[:, 1:].astype(mx.float32)
    denom = mx.maximum(w.sum(), 1.0)
    h = model.model.embed_tokens(inputs)
    ces, gates, logits_list = [], [], []
    for _ in range(4):
        for layer in model.model.layers:
            h = layer(h, "causal", None)
        h = model.model.norm(h)
        ces.append(token_ce(model, h, targets))  # [1, L]
        gates.append(model.model.early_exit_gate(h.astype(mx.float32))[..., 0])
        if args.loss == "depth" and args.distill > 0:  # full logits only when distilling
            logits_list.append(model.lm_head(h).astype(mx.float32))
    ce = mx.stack(ces, -1)  # [1, L, T]
    per_loop = (ce * w[..., None]).sum((0, 1)) / denom
    if args.loss == "ouro":
        p = exit_pdf(gates)
        token = (p * ce).sum(-1) + args.beta * (p * mx.log(p + 1e-9)).sum(-1)
        loss = (token * w).sum() / denom
    else:
        loss = per_loop.mean()
        if args.distill > 0:
            teacher = mx.stop_gradient(mx.softmax(logits_list[-1], -1))
            kl = []
            for z in logits_list[:-1]:
                logq = z - mx.logsumexp(z, -1, keepdims=True)
                kl.append(((teacher * (mx.log(teacher + 1e-9) - logq)).sum(-1) * w).sum() / denom)
            loss = loss + args.distill * mx.stack(kl).mean()
    return loss, per_loop


def export_merged(model: OuroForCausalLM, snap: Path, path: Path) -> None:
    """Base weights with LoRA merged in, under the HF / ouro-mlx parameter names."""
    weights = dict(mx.load(str(snap / "model.safetensors")))
    for i, layer in enumerate(model.model.layers):
        for block, names in LORA_TARGETS.items():
            for name in names:
                weights[f"model.layers.{i}.{block}.{name}.weight"] = getattr(
                    getattr(layer, block), name
                ).merged_weight()
    dtype = weights["model.early_exit_gate.weight"].dtype  # gate trains in fp32 (see main)
    weights["model.early_exit_gate.weight"] = model.model.early_exit_gate.weight.astype(dtype)
    weights["model.early_exit_gate.bias"] = model.model.early_exit_gate.bias.astype(dtype)
    mx.save_safetensors(str(path), weights)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ByteDance/Ouro-2.6B")
    parser.add_argument("--name", required=True)
    parser.add_argument("--loss", choices=["ouro", "depth"], required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--distill", type=float, default=0.0)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=64.0)
    # 0: with per-layer checkpointing a dropout mask is redrawn in the backward pass, which
    # makes gradients inconsistent (code review 2026-10-16); 2608.18171 gives no dropout.
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=None, help="optimizer steps (timing)")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=50, help="resume checkpoint period")
    parser.add_argument("--resume", action="store_true", help="continue from state/ in --name")
    parser.add_argument("--cache-limit-gb", type=float, default=4.0)
    args = parser.parse_args()
    if args.dropout > 0:
        raise SystemExit("--dropout > 0 is incompatible with gradient checkpointing here")

    # MLX keeps freed buffers in a reuse cache that otherwise grows toward all free memory
    # (the first timing run hit 96 GB); cap it and clear it every optimizer step.
    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    mx.random.seed(args.seed)
    snap = Path(snapshot_download(args.model, revision=OURO_REVISIONS[args.model]))
    model = OuroForCausalLM(OuroConfig.from_json(snap / "config.json"), ut_steps=4)
    model.load_weights(str(snap / "model.safetensors"), strict=True)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(snap / "tokenizer.json"))
    model.freeze()
    add_lora(model, args.rank, args.alpha, args.dropout)
    if args.loss == "ouro":
        # bf16 AdamW updates at LR 2e-5 are below half an ulp of the gate's weights, so the
        # gate would never move; keep it in fp32 while training (exported back as bf16).
        gate = model.model.early_exit_gate
        gate.weight = gate.weight.astype(mx.float32)
        gate.bias = gate.bias.astype(mx.float32)
        gate.unfreeze()
    checkpoint_layers()
    mx.eval(model.parameters())
    n_train_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))

    data, dropped = conversations(tokenizer, args.max_len)
    random.Random(args.seed).shuffle(data)
    n_val = len(data) // 10
    val, train = data[:n_val], data[n_val:]
    steps_per_epoch = len(train) // args.grad_accum
    total = args.max_steps or int(steps_per_epoch * args.epochs)
    warmup = max(1, int(0.03 * total))
    schedule = optim.join_schedules(
        [optim.linear_schedule(0.0, args.lr, warmup), optim.cosine_decay(args.lr, total - warmup)],
        [warmup],
    )
    opt = optim.AdamW(learning_rate=schedule, weight_decay=0.0)

    out = Path(f"results/ouro_sft/{args.name}")
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        **vars(args),
        "hermes_revision": HERMES_REVISION,
        "hermes_files": HERMES_FILES,
        "n_train": len(train),
        "n_val": len(val),
        "dropped_over_max_len": dropped,
        "trainable_params": n_train_params,
        "total_steps": total,
        "warmup_steps": warmup,
        "val_ids": [v[0] for v in val],
    }
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "val_ids"}), flush=True)

    value_and_grad = nn.value_and_grad(model, lambda m, i, k: loss_fn(m, i, k, args))

    def evaluate() -> list[float]:
        model.eval()
        sums = np.zeros(4)
        for _, ids, mask in val[: args.eval_n]:
            _, per_loop = loss_fn(model, mx.array([ids]), mx.array([mask]), args)
            sums += np.array(per_loop)
        model.train()
        return (sums / min(args.eval_n, len(val))).round(4).tolist()

    state_dir = out / "state"
    order = list(range(len(train)))
    rng = random.Random(args.seed + 1)
    cursor, tokens, start = len(order), 0, 1
    if args.resume and (state_dir / "progress.json").exists():
        prog = json.loads((state_dir / "progress.json").read_text())
        model.load_weights(str(state_dir / "adapters.safetensors"), strict=False)
        opt.init(model.trainable_parameters())
        opt.state = tree_unflatten(list(mx.load(str(state_dir / "optimizer.safetensors")).items()))
        order, cursor, tokens = prog["order"], prog["cursor"], prog["tokens"]
        rng.setstate(tuple(tuple(x) if isinstance(x, list) else x for x in prog["rng"]))
        start = prog["step"] + 1
        print(json.dumps({"resumed_from_step": prog["step"]}), flush=True)

    def save_state(step: int) -> None:
        """Adapters, optimizer state and data position, written atomically (tmp + rename)."""
        state_dir.mkdir(exist_ok=True)
        tmp = state_dir / "tmp"
        tmp.mkdir(exist_ok=True)
        mx.save_safetensors(
            str(tmp / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters()))
        )
        mx.save_safetensors(str(tmp / "optimizer.safetensors"), dict(tree_flatten(opt.state)))
        (tmp / "progress.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "order": order,
                    "cursor": cursor,
                    "tokens": tokens,
                    "rng": rng.getstate(),
                }
            )
        )
        for name in ("adapters.safetensors", "optimizer.safetensors", "progress.json"):
            (tmp / name).replace(state_dir / name)

    yield_requested = False

    def on_yield(signum, frame) -> None:
        nonlocal yield_requested
        yield_requested = True
        print(json.dumps({"yield_requested": True}), flush=True)

    signal.signal(signal.SIGUSR1, on_yield)

    log = open(out / "log.jsonl", "a")
    if start == 1:
        rec0 = {"step": 0, "val_ce_by_loop": evaluate()}
        log.write(json.dumps(rec0) + "\n")
        log.flush()
        print(json.dumps(rec0), flush=True)
    model.train()
    t0 = time.perf_counter()
    tokens0 = tokens
    recent_loss, recent_loops = deque(maxlen=50), deque(maxlen=50)
    for step in range(start, total + 1):
        acc_grads, acc_loss, acc_loops = None, 0.0, np.zeros(4)
        for _ in range(args.grad_accum):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            _, ids, mask = train[order[cursor]]
            cursor += 1
            tokens += len(ids)
            (loss, per_loop), grads = value_and_grad(model, mx.array([ids]), mx.array([mask]))
            acc_grads = grads if acc_grads is None else tree_map(mx.add, acc_grads, grads)
            mx.eval(loss, per_loop, acc_grads)
            acc_loss += loss.item()
            acc_loops += np.array(per_loop)
        lr_used = float(schedule(opt.state["step"]) if "step" in opt.state else schedule(0))
        opt.update(model, tree_map(lambda g: g / args.grad_accum, acc_grads))
        mx.eval(model.trainable_parameters(), opt.state)
        del acc_grads
        mx.clear_cache()
        recent_loss.append(acc_loss / args.grad_accum)
        recent_loops.append(acc_loops / args.grad_accum)
        rec = {
            "step": step,
            "loss": round(acc_loss / args.grad_accum, 4),
            "ce_by_loop": (acc_loops / args.grad_accum).round(4).tolist(),
            "loss_avg50": round(float(np.mean(recent_loss)), 4),
            "ce_by_loop_avg50": np.mean(recent_loops, 0).round(4).tolist(),
            "lr": lr_used,
            "tok_per_s": round((tokens - tokens0) / (time.perf_counter() - t0), 1),
            "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
        }
        # Skip validation when asked to yield: the 60 s grace period is for saving.
        if (step % args.eval_every == 0 or step == total) and not yield_requested:
            rec["val_ce_by_loop"] = evaluate()
            mx.save_safetensors(
                str(out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters()))
            )
        if step % args.save_every == 0 or step == total or yield_requested:
            save_state(step)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if yield_requested and step < total:
            print(json.dumps({"yielded_at_step": step}), flush=True)
            log.close()
            signal.signal(signal.SIGUSR1, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGUSR1)
        print(f"::progress {step}/{total} steps", flush=True)
        if step % 10 == 0 or "val_ce_by_loop" in rec:
            print(json.dumps(rec), flush=True)

    export_merged(model, snap, out / "model.safetensors")
    print(json.dumps({"done": total, "elapsed_s": round(time.perf_counter() - t0)}), flush=True)


if __name__ == "__main__":
    main()
