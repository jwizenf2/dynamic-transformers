"""LoRA tool-calling fine-tune of Huginn-0125 with randomised recurrence depth (PyTorch).

    python scripts/lab/huginn_sft.py --name huginn-chain \
        --extra-data results/chains/chains_s1_n8000_p0.2_s0.1.json results/mathqa/train_sft.json

Why: Huginn is a base model; zero-shot it writes tool-call syntax only at >= 16 steps and never the
right call (``huginn_check.py``). To compare it with chain v3 it gets the same training mix as chain
v3's stage 2 (Hermes FC v1 subsample + synthetic chains v2 + MathQA train dialogues, the files
``sft_queue.py --plan chains-v3`` prepares), rendered in Huginn's own chat template
(``<|begin_header|>{role}<|end_header|>\\n\\n{text}<|end_turn|>``, the assistant role is
"Huginn", BOS before the first turn), loss on the assistant turns only (their text and
``<|end_turn|>``).

Depth: as in Huginn's pretraining (``randomized_iteration_sampler``), every optimizer micro-step
draws a total step count p ~ Poisson(LogNormal(log m − σ²/2, σ)) + 1 with σ = .5 and mean
``--mean-steps`` m, runs max(p − s, 0) steps without gradient and min(s, p) with gradient
(``--backprop-steps`` s), so the adapters are trained for the whole range of depths the evaluation
uses. Choices (recorded, not from the paper): m = 16, s = 4 (pretraining used 32 and 8; halved for
compute), max length 2,048 tokens, 6,000 training conversations sampled from the mix
(``--max-train``), one epoch, AdamW lr 1e-4 with 3% warmup and cosine decay, effective batch 4.

LoRA (rank 32, alpha 64, no dropout) on every linear layer of the prelude, the recurrent block,
the coda (attention ``Wqkv``/``proj``, MLP ``fc``/``proj``) and the injection ``adapter``; base
weights frozen in bfloat16, adapters in float32. Validation: assistant-token cross-entropy on 100
held-out Hermes conversations at 4, 16 and 32 steps (state seeded, ``--eval-every``).

Resume: ``--resume`` continues from ``results/huginn_sft/<name>/state.pt`` (adapters, optimizer,
data order, step), saved every ``--save-every`` steps and on SIGUSR1 (the lab asking the job to
yield), after which SIGUSR1 is re-raised so the lab records a stop. Output: ``adapters.safetensors``
(LoRA tensors), ``config.json`` and ``log.jsonl``; ``done`` marks a finished run.
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
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from huginn_model import load_huginn  # noqa: E402
from ouro_lora_sft import HERMES, HERMES_FILES, HERMES_REVISION, ROLES  # noqa: E402

TARGETS = {"attn": ("Wqkv", "proj"), "mlp": ("fc", "proj")}


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        dev = base.weight.device
        self.lora_a = torch.nn.Parameter(torch.empty(rank, base.in_features, device=dev))
        self.lora_b = torch.nn.Parameter(torch.zeros(base.out_features, rank, device=dev))
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        delta = (x.float() @ self.lora_a.T @ self.lora_b.T) * self.scale
        return self.base(x) + delta.to(x.dtype)


def add_lora(model, rank: int, alpha: float) -> dict[str, LoRALinear]:
    out = {}
    t = model.transformer
    for part in ("prelude", "core_block", "coda"):
        for i, block in enumerate(t[part]):
            for sub, names in TARGETS.items():
                mod = getattr(block, sub)
                for name in names:
                    lora = LoRALinear(getattr(mod, name), rank, alpha)
                    setattr(mod, name, lora)
                    out[f"{part}.{i}.{sub}.{name}"] = lora
    t.adapter = LoRALinear(t.adapter, rank, alpha)
    out["adapter"] = t.adapter
    return out


def lora_state(loras: dict[str, LoRALinear]) -> dict[str, torch.Tensor]:
    return {
        f"{k}.{p}": getattr(m, p).detach().cpu().contiguous()
        for k, m in loras.items()
        for p in ("lora_a", "lora_b")
    }


def encode(tok, row: dict, max_len: int):
    """(ids, assistant-target mask) in Huginn's chat template, or None if over max_len."""
    ids, mask = [tok.bos_token_id], [0]
    for m in row["conversations"]:
        role = ROLES[m["from"]]
        head = tok.encode(
            f"<|begin_header|>{'Huginn' if role == 'assistant' else role}<|end_header|>\n\n",
            add_special_tokens=False,
        )
        body = tok.encode(m["value"].strip() + "<|end_turn|>", add_special_tokens=False)
        ids += head + body
        mask += [0] * len(head) + [int(role == "assistant")] * len(body)
    return (ids, mask) if len(ids) <= max_len else None


def sample_steps(mean: int, backprop: int) -> tuple[int, int]:
    """(steps without gradient, steps with gradient), Huginn's pretraining sampler."""
    sigma = 0.5
    mu = math.log(mean) - sigma**2 / 2
    rate = float(torch.zeros(1).log_normal_(mean=mu, std=sigma))
    p = int(torch.poisson(torch.tensor([rate])).item()) + 1
    return max(p - backprop, 0), min(backprop, p)


def loss_fn(model, ids, mask, steps, device) -> torch.Tensor:
    x = torch.tensor([ids], device=device)
    tgt = torch.tensor(ids[1:] + [-100], device=device)
    keep = torch.tensor(mask[1:] + [0], device=device).bool()
    tgt = torch.where(keep, tgt, torch.full_like(tgt, -100))
    out = model(
        input_ids=x,
        num_steps=steps,
        output_details={
            "return_logits": True,
            "return_latents": False,
            "return_head": False,
            "return_stats": False,
        },
    )
    return torch.nn.functional.cross_entropy(out.logits[0].float(), tgt, ignore_index=-100)


@torch.no_grad()
def validate(model, val, device, steps_list=(4, 16, 32)) -> dict:
    res = {}
    for r in steps_list:
        tot = 0.0
        for k, (ids, mask) in enumerate(val):
            torch.manual_seed(k)
            tot += float(loss_fn(model, ids, mask, (r, 0), device))
        res[r] = round(tot / max(1, len(val)), 4)
    return res


stop = {"requested": False}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--extra-data", nargs="*", default=[])
    ap.add_argument("--hermes-train-n", type=int, default=3000)
    ap.add_argument("--max-train", type=int, default=6000)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--mean-steps", type=int, default=16)
    ap.add_argument("--backprop-steps", type=int, default=4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(f"results/huginn_sft/{args.name}")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done").exists():
        print(json.dumps({"skip": "finished", "name": args.name}), flush=True)
        return
    signal.signal(signal.SIGUSR1, lambda s, f: stop.update(requested=True))
    model, tok = load_huginn()
    device = next(model.parameters()).device
    for p in model.parameters():
        p.requires_grad_(False)
    loras = add_lora(model, args.rank, args.alpha)
    model.gradient_checkpointing = True
    params = [p for m in loras.values() for p in (m.lora_a, m.lora_b)]

    hermes = []
    for fname in HERMES_FILES:
        path = hf_hub_download(HERMES, fname, repo_type="dataset", revision=HERMES_REVISION)
        hermes += json.loads(Path(path).read_text(encoding="utf-8"))
    enc_h = [e for e in (encode(tok, r, args.max_len) for r in hermes) if e]
    random.Random(0).shuffle(enc_h)
    n_val = len(enc_h) // 10
    val, train_h = enc_h[: args.eval_n], enc_h[n_val:]
    pool = random.Random(args.seed + 2).sample(train_h, min(args.hermes_train_n, len(train_h)))
    for path in args.extra_data:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        pool += [e for e in (encode(tok, r, args.max_len) for r in rows) if e]
    rng = random.Random(args.seed)
    train = rng.sample(pool, min(args.max_train, len(pool)))
    total = args.max_steps or int(len(train) * args.epochs) // args.grad_accum
    warmup = max(1, int(0.03 * total))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            (s + 1) / warmup
            if s < warmup
            else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup)))
        ),
    )
    step, order = 0, list(range(len(train)))
    rng.shuffle(order)
    if args.resume and (out / "state.pt").exists():
        st = torch.load(out / "state.pt", map_location="cpu", weights_only=True)
        for k, m in loras.items():
            m.lora_a.data.copy_(st["lora"][f"{k}.lora_a"])
            m.lora_b.data.copy_(st["lora"][f"{k}.lora_b"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, order = st["step"], st["order"]
        torch.set_rng_state(st["torch_rng"])
    meta = {
        **vars(args),
        "n_train": len(train),
        "n_val": len(val),
        "total_steps": total,
        "hermes_revision": HERMES_REVISION,
        "trainable_params": sum(p.numel() for p in params),
    }
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    if step == 0:
        print(json.dumps({"step": 0, "val_ce_by_steps": validate(model, val, device)}), flush=True)

    def save_state() -> None:
        tmp = out / "state.pt.tmp"
        torch.save(
            {
                "lora": lora_state(loras),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "step": step,
                "order": order,
                "torch_rng": torch.get_rng_state(),
            },
            tmp,
        )
        os.replace(tmp, out / "state.pt")

    log = open(out / "log.jsonl", "a")
    t0, avg = time.time(), []
    while step < total:
        opt.zero_grad(set_to_none=True)
        for j in range(args.grad_accum):
            ids, mask = train[order[(step * args.grad_accum + j) % len(order)]]
            steps = sample_steps(args.mean_steps, args.backprop_steps)
            loss = loss_fn(model, ids, mask, steps, device) / args.grad_accum
            loss.backward()
            avg.append(float(loss.detach()) * args.grad_accum)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        print(f"::progress {step}/{total} steps", flush=True)
        if step % 10 == 0:
            rec = {
                "step": step,
                "loss_avg": round(sum(avg[-40:]) / len(avg[-40:]), 4),
                "lr": sched.get_last_lr()[0],
                "s": round(time.time() - t0),
            }
            if step % args.eval_every == 0:
                rec["val_ce_by_steps"] = validate(model, val, device)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
        if step % args.save_every == 0 or stop["requested"]:
            save_state()
        if stop["requested"]:
            print(json.dumps({"yield": step}), flush=True)
            signal.signal(signal.SIGUSR1, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGUSR1)
    save_file(lora_state(loras), str(out / "adapters.safetensors"))
    final = {"step": step, "val_ce_by_steps": validate(model, val, device)}
    log.write(json.dumps(final) + "\n")
    print(json.dumps(final), flush=True)
    (out / "done").write_text("ok\n")


if __name__ == "__main__":
    main()
