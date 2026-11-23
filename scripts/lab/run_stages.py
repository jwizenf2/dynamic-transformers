"""Run a fixed list of experiment stages in order, as one resumable lab job.

    python scripts/lab/run_stages.py --plan replication

The lab has no job dependencies, so a multi-day queue runs as one job (as ``sft_queue.py``):

* Stages run one at a time (one GPU user). Each stage's command resumes by itself (the
  collection scripts skip done request ids, ``sft_queue.py`` skips trained models and finished
  evals), so resubmitting the same command in the same project folder continues where the last
  attempt stopped; finished stages only reload a model and exit.
* A stage with ``skip_if`` is skipped when that path exists (data preparation).
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


CHAINS_V2 = "results/chains/chains_s1_n8000_p0.2_s0.1.json"
MATHQA = "results/mathqa/train_sft.json"
HUGINN = "results/huginn_sft/huginn-chain/adapters.safetensors"
HUGINN_DONE = "results/huginn_sft/huginn-chain/done"
HUGINN_KEEP_STATE = f"{KEEP}/huginn-chain/state.pt"
PLAN480 = "results/nestful_plans/full480_tasks.jsonl"
GSM8K = "results/gsm8k/test_tasks.jsonl"
GSM8K_PLAN = "results/gsm8k_plans/full_tasks.jsonl"
GSM8K_SMOKE = "results/router_smoke/ouro-chain-depth-v3_gsm8k/examples.jsonl"
GSM8K_TRAIN = "results/gsm8k/train_tasks.jsonl"
GSM8K_TRAIN_N, GSM8K_CHUNK = 5510, 1000


def depth6_stage(name: str) -> dict:
    """Depth-6 dialogues on NESTful [0, 720) for a kept model (P4 stability)."""
    a = f"{KEEP}/{name}/adapters.safetensors"
    return {
        "cmd": [PY, "scripts/lab/router_smoke.py", "--adapters", a, "--tasks", "nestful"]
        + ["--n", "720", "--depths", "6", "--modes", "greedy", "--tag", "nestful_d6"],
        "needs": [a],
    }


def late_switch_stage(tasks: str, n: int, tag: str, smoke: list[str]) -> dict:
    return {
        "cmd": [PY, "scripts/lab/late_switch.py", "--adapters", V3, "--tasks", tasks]
        + ["--n", str(n), "--tag", tag, "--points", "64", "128", "--smoke", *smoke],
        "needs": ([] if tasks == "nestful" else [tasks]) + smoke,
    }


def hidden_probe_stage(tasks: str, start: int, n: int, smoke: str, tag: str = "nestful") -> dict:
    """``router_probe2 --hidden`` for an already-labelled slice (rows matched to ``smoke``)."""
    return {
        "cmd": [PY, "scripts/lab/router_probe2.py", "--adapters", V3, "--tasks", tasks]
        + ["--start", str(start), "--n", str(n), "--hidden", "--smoke", smoke]
        + ["--tag", f"{tag}_hid"],
        "needs": [smoke] + ([] if tasks == "nestful" else [tasks]),
    }


def gsm8k_seed_stages(name: str) -> list[dict]:
    """GSM8K test (800) for a kept training seed: 2, 4, 6 loops, the probe, the full plan."""
    a = f"{KEEP}/{name}/adapters.safetensors"
    smoke = f"results/router_smoke/{name}_gsm8k/examples.jsonl"
    return [
        {
            "cmd": [PY, "scripts/lab/router_smoke.py", "--adapters", a, "--tasks", GSM8K]
            + ["--n", "800", "--depths", "2", "4", "6", "--modes", "greedy", "--tag", "gsm8k"],
            "needs": [a, GSM8K],
        },
        {
            "cmd": [PY, "scripts/lab/router_probe2.py", "--adapters", a, "--tasks", GSM8K]
            + ["--n", "800", "--tag", "gsm8k", "--smoke", smoke],
            "needs": [smoke],
        },
        {
            "cmd": [PY, "scripts/lab/router_smoke.py", "--adapters", a, "--tasks", GSM8K_PLAN]
            + ["--n", "800", "--depths", "2", "4", "--modes", "greedy", "--tag", "gsm8k_plan_full"],
            "needs": [a, GSM8K_PLAN],
        },
    ]


def gsm8k_train_stages(start: int) -> list[dict]:
    """Router labels on a GSM8K-train chunk: 2- and 4-loop greedy dialogues, then the 48-token
    probe with loop-1/2 hidden states on the same requests (chain v3 seed 0)."""
    n = min(GSM8K_CHUNK, GSM8K_TRAIN_N - start)
    tag = f"gsm8ktrain{start}"
    smoke = f"results/router_smoke/ouro-chain-depth-v3_{tag}/examples.jsonl"
    return [
        {
            "cmd": [PY, "scripts/lab/router_smoke.py", "--adapters", V3, "--tasks", GSM8K_TRAIN]
            + ["--start", str(start), "--n", str(n), "--depths", "2", "4"]
            + ["--modes", "greedy", "--tag", tag],
            "needs": [GSM8K_TRAIN],
        },
        hidden_probe_stage(GSM8K_TRAIN, start, n, smoke, tag),
    ]


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
    # docs/research/late_switch_2026-11-20.md: 2-loop traces with a switch to 4 loops at trace
    # steps 64 and 128 (MathQA development first, then the NESTful and GSM8K test sets)
    "late-switch": [
        {"cmd": [PY, "scripts/lab/make_gsm8k.py"], "skip_if": GSM8K},
        late_switch_stage(
            f"{KEEP}/mathqa_v3/validation_clean_tasks.jsonl",
            292,
            "mathqa_clean",
            [f"{KEEP}/router_smoke_v3/examples.jsonl"],
        ),
        late_switch_stage(
            f"{KEEP}/mathqa_v3/testpool_tasks.jsonl",
            239,
            "mathqa_pool",
            [f"{KEEP}/router_smoke_v3_testpool/examples.jsonl"],
        ),
        late_switch_stage(
            "nestful",
            720,
            "nestful",
            [
                f"{KEEP}/router_smoke_nestful/examples.jsonl",
                f"{KEEP}/router_confirm2_nestful240/router_smoke/ouro-chain-depth-v3_nestful240/"
                "examples.jsonl",
            ],
        ),
        late_switch_stage(
            GSM8K,
            800,
            "gsm8k",
            [f"{KEEP}/gsm8k/router_smoke/ouro-chain-depth-v3_gsm8k/examples.jsonl"],
        ),
    ],
    # docs/research/router_nonlinear_2026-11-18.md, follow-up: the 48-token probe again with
    # loop-1/2 hidden states saved, on the already-labelled NESTful [0, 1200) and GSM8K problems
    # (no new dialogues; the probe asserts the same requests in the same order as the labels)
    "hidden-probes": [
        {"cmd": [PY, "scripts/lab/make_gsm8k.py"], "skip_if": GSM8K},
        hidden_probe_stage("nestful", 0, 240, f"{KEEP}/router_smoke_nestful/examples.jsonl"),
        hidden_probe_stage(
            "nestful",
            240,
            480,
            f"{KEEP}/router_confirm2_nestful240/router_smoke/ouro-chain-depth-v3_nestful240/"
            "examples.jsonl",
            "nestful240",
        ),
        hidden_probe_stage(
            "nestful",
            720,
            480,
            f"{KEEP}/replication/router_smoke/ouro-chain-depth-v3_nestful720/examples.jsonl",
            "nestful720",
        ),
        hidden_probe_stage(
            GSM8K,
            0,
            800,
            f"{KEEP}/gsm8k/router_smoke/ouro-chain-depth-v3_gsm8k/examples.jsonl",
            "gsm8k",
        ),
    ],
    # docs/research/router_scale_2026-11-22.md: GSM8K test for training seeds 1 and 2 (the second
    # benchmark on all three seeds), then router labels with hidden states on GSM8K train in
    # chunks of 1,000 (each chunk usable on its own if the run is stopped)
    "router-scale": [
        {"cmd": [PY, "scripts/lab/make_gsm8k.py"], "skip_if": GSM8K},
        {"cmd": [PY, "scripts/lab/make_gsm8k.py", "--split", "train"], "skip_if": GSM8K_TRAIN},
        {
            "cmd": [PY, "scripts/lab/make_nestful_plans.py", "--tasks", GSM8K, "--n", "800"]
            + ["--out", "results/gsm8k_plans"],
            "needs": [GSM8K],
            "skip_if": GSM8K_PLAN,
        },
        *gsm8k_seed_stages("chain-depth-v3-s1"),
        *gsm8k_seed_stages("chain-depth-v3-s2"),
        *[st for s in range(0, GSM8K_TRAIN_N, GSM8K_CHUNK) for st in gsm8k_train_stages(s)],
    ],
    # docs/research/huginn_2026-11-14.md: Huginn-0125 LoRA tool fine-tune with randomised depth on
    # chain v3's stage-2 mix, then NESTful [0, 480) at 4, 8, 32, 64 steps and the full plan at
    # 8 and 32
    "huginn": [
        {
            "cmd": [PY, "scripts/lab/make_chains.py", "--n", "8000", "--seed", "1"]
            + ["--parallel-p", "0.2", "--single-p", "0.1"],
            "skip_if": CHAINS_V2,
        },
        {"cmd": [PY, "scripts/lab/make_mathqa.py"], "skip_if": MATHQA},
        # restore the kept checkpoint into a fresh project folder (a new commit replaces it)
        {"cmd": ["mkdir", "-p", "results/huginn_sft/huginn-chain"], "needs": [HUGINN_KEEP_STATE]},
        {
            "cmd": ["cp", HUGINN_KEEP_STATE, "results/huginn_sft/huginn-chain/state.pt"],
            "needs": [HUGINN_KEEP_STATE],
            "skip_if": "results/huginn_sft/huginn-chain/state.pt",
        },
        {
            "cmd": [PY, "scripts/lab/huginn_sft.py", "--name", "huginn-chain", "--resume"]
            + ["--extra-data", CHAINS_V2, MATHQA],
            "needs": [CHAINS_V2, MATHQA],
        },
        {
            "cmd": [PY, "scripts/lab/huginn_nestful.py", "--adapters", HUGINN, "--tasks", "nestful"]
            + ["--n", "480", "--steps", "4", "8", "32", "64", "--tag", "nestful"],
            "needs": [HUGINN_DONE],
        },
        {
            "cmd": [PY, "scripts/lab/make_nestful_plans.py", "--n", "480", "--tag", "480"],
            "skip_if": PLAN480,
        },
        {
            "cmd": [PY, "scripts/lab/huginn_nestful.py", "--adapters", HUGINN, "--tasks", PLAN480]
            + ["--n", "480", "--steps", "8", "32", "--tag", "plan_full"],
            "needs": [HUGINN_DONE, PLAN480],
        },
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
        if st.get("skip_if") and Path(st["skip_if"]).exists():
            log.append({"stage": k, "skipped": True, "exists": st["skip_if"]})
            print(json.dumps(log[-1]), flush=True)
            continue
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
