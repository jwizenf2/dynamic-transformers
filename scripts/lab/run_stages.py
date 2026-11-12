"""Run a fixed list of experiment stages in order, as one resumable lab job.

    python scripts/lab/run_stages.py --plan replication

The lab has no job dependencies, so a multi-day queue runs as one job (as ``sft_queue.py``):

* Stages run one at a time (one GPU user). Each stage's command resumes by itself (the
  collection scripts skip done request ids, ``sft_queue.py`` skips trained models and finished
  evals), so resubmitting the same command in the same project folder continues where the last
  attempt stopped; finished stages only reload a model and exit.
* A stage runs only if every path in its ``needs`` exists (e.g. the adapters a fine-tune
  stage should have written); otherwise it is skipped and reported. A failing stage does not stop
  the stages after it, except through their ``needs``.
* SIGUSR1 (the lab broker asking the job to yield) or SIGTERM (cancel) is forwarded to the running
  stage, which saves its state; the driver waits for it and re-raises the signal so the lab
  records a stop.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
KEEP = "/Users/jeremywizenfeld/AI/keep"
V3 = f"{KEEP}/ouro-chain-depth-v3/adapters.safetensors"
O14 = ["--model", "ByteDance/Ouro-1.4B"]


def seed_or_model_stages(plan: str, name: str, model: list[str]) -> list[dict]:
    """Fine-tune ``name`` with ``sft_queue --plan``, keep its adapters, evaluate on [0, 720)."""
    a = f"results/ouro_sft/{name}/adapters.safetensors"
    done = f"results/ouro_sft/{name}/model.safetensors"  # written only when training finishes
    smoke = f"results/router_smoke/{name}_nestful/examples.jsonl"
    return [
        {"cmd": [PY, "scripts/lab/sft_queue.py", "--plan", plan]},
        {"cmd": ["mkdir", "-p", f"{KEEP}/{name}"], "needs": [done]},
        {
            "cmd": ["cp", a, f"results/ouro_sft/{name}/config.json", f"{KEEP}/{name}/"],
            "needs": [done],
        },
        {"cmd": [PY, "scripts/lab/make_nestful_plans.py"]},
        {
            "cmd": [
                PY,
                "scripts/lab/router_smoke.py",
                *model,
                "--adapters",
                a,
                "--tasks",
                "nestful",
            ]
            + [
                "--n",
                "720",
                "--depths",
                "1",
                "2",
                "3",
                "4",
                "--modes",
                "greedy",
                "--tag",
                "nestful",
            ],
            "needs": [done],
        },
        {
            "cmd": [
                PY,
                "scripts/lab/router_probe2.py",
                *model,
                "--adapters",
                a,
                "--tasks",
                "nestful",
            ]
            + ["--n", "720", "--tag", "nestful", "--smoke", smoke],
            "needs": [done, smoke],
        },
        {
            "cmd": [PY, "scripts/lab/router_smoke.py", *model, "--adapters", a]
            + ["--tasks", "results/nestful_plans/full_tasks.jsonl", "--n", "720"]
            + ["--depths", "2", "4", "--modes", "greedy", "--tag", "plan_full"],
            "needs": [done, "results/nestful_plans/full_tasks.jsonl"],
        },
        {"cmd": [PY, "scripts/lab/bfcl_probe.py", *model, "--adapters", a], "needs": [done]},
    ]


GSM8K = "results/gsm8k/test_tasks.jsonl"
GSM8K_PLAN = "results/gsm8k_plans/full_tasks.jsonl"
GSM8K_SMOKE = "results/router_smoke/ouro-chain-depth-v3_gsm8k/examples.jsonl"


def depth6_stage(name: str) -> dict:
    """Depth-6 dialogues on NESTful [0, 720) for a kept model (P4 stability)."""
    a = f"{KEEP}/{name}/adapters.safetensors"
    return {
        "cmd": [PY, "scripts/lab/router_smoke.py", "--adapters", a, "--tasks", "nestful"]
        + ["--n", "720", "--depths", "6", "--modes", "greedy", "--tag", "nestful_d6"],
        "needs": [a],
    }


PLANS = {
    # docs/research/replication_2026-11-04.md: unseen confirmation, seed 1, Ouro-1.4B, seed 2
    "replication": [
        {
            "cmd": [PY, "scripts/lab/make_nestful_plans.py", "--start", "720", "--n", "480"]
            + ["--tag", "720"]
        },
        {
            "cmd": [PY, "scripts/lab/router_smoke.py", "--adapters", V3, "--tasks", "nestful"]
            + ["--start", "720", "--n", "480", "--depths", "1", "2", "3", "4", "6"]
            + ["--modes", "greedy", "--tag", "nestful720"],
        },
        {
            "cmd": [PY, "scripts/lab/router_probe2.py", "--adapters", V3, "--tasks", "nestful"]
            + ["--start", "720", "--n", "480", "--tag", "nestful720", "--smoke"]
            + ["results/router_smoke/ouro-chain-depth-v3_nestful720/examples.jsonl"],
            "needs": ["results/router_smoke/ouro-chain-depth-v3_nestful720/examples.jsonl"],
        },
        {
            "cmd": [PY, "scripts/lab/router_smoke.py", "--adapters", V3]
            + ["--tasks", "results/nestful_plans/full720_tasks.jsonl", "--n", "480"]
            + ["--depths", "2", "4", "--modes", "greedy", "--tag", "plan_full720"],
            "needs": ["results/nestful_plans/full720_tasks.jsonl"],
        },
        *seed_or_model_stages("chains-v3-s1", "chain-depth-v3-s1", []),
        *seed_or_model_stages("ouro14", "ouro14-chain-depth-v3", O14),
        *seed_or_model_stages("chains-v3-s2", "chain-depth-v3-s2", []),
    ],
    # docs/research/replication_2026-11-04.md, "Second benchmark": GSM8K test as tool chains with
    # chain v3 seed 0 (1–4 and 6 loops, probe, full plan), then depth 6 for seeds 1 and 2 (P4)
    "gsm8k-p4": [
        {"cmd": [PY, "scripts/lab/make_gsm8k.py"]},
        {
            "cmd": [PY, "scripts/lab/make_nestful_plans.py", "--tasks", GSM8K, "--n", "800"]
            + ["--out", "results/gsm8k_plans"],
            "needs": [GSM8K],
        },
        {
            "cmd": [PY, "scripts/lab/router_smoke.py", "--adapters", V3, "--tasks", GSM8K]
            + ["--n", "800", "--depths", "1", "2", "3", "4", "6", "--modes", "greedy"]
            + ["--tag", "gsm8k"],
            "needs": [GSM8K],
        },
        {
            "cmd": [PY, "scripts/lab/router_probe2.py", "--adapters", V3, "--tasks", GSM8K]
            + ["--n", "800", "--tag", "gsm8k", "--smoke", GSM8K_SMOKE],
            "needs": [GSM8K_SMOKE],
        },
        {
            "cmd": [PY, "scripts/lab/router_smoke.py", "--adapters", V3, "--tasks", GSM8K_PLAN]
            + ["--n", "800", "--depths", "2", "4", "--modes", "greedy", "--tag", "gsm8k_plan_full"],
            "needs": [GSM8K_PLAN],
        },
        depth6_stage("chain-depth-v3-s1"),
        depth6_stage("chain-depth-v3-s2"),
    ],
}

current: list[subprocess.Popen] = []
stop = {"signal": None}


def on_signal(signum, frame) -> None:
    stop["signal"] = signum
    print(json.dumps({"stages": "stop requested", "signal": signum}), flush=True)
    for p in current:
        if p.poll() is None:
            p.send_signal(signal.SIGUSR1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", choices=sorted(PLANS), required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGUSR1, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    t0, log = time.time(), []
    stages = PLANS[args.plan]
    for k, st in enumerate(stages, 1):
        missing = [p for p in st.get("needs", []) if not Path(p).exists()]
        if missing:
            log.append({"stage": k, "skipped": True, "missing": missing})
            print(json.dumps(log[-1]), flush=True)
            continue
        print(json.dumps({"stage": k, "of": len(stages), "cmd": " ".join(st["cmd"])}), flush=True)
        proc = subprocess.Popen(st["cmd"])
        current.append(proc)
        code = proc.wait()
        current.remove(proc)
        log.append({"stage": k, "exit": code})
        if stop["signal"] is not None:
            print(json.dumps({"stages": "stopped", "at": k}), flush=True)
            signal.signal(stop["signal"], signal.SIG_DFL)
            os.kill(os.getpid(), stop["signal"])
        if code != 0:
            print(json.dumps({"stage": k, "failed": code}), flush=True)
    hours = round((time.time() - t0) / 3600, 2)
    print(json.dumps({"stages": "done", "hours": hours, "log": log}), flush=True)


if __name__ == "__main__":
    main()
