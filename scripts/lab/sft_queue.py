"""Run the SFT stages of one plan in order, as one lab job that can be resubmitted after a stop.

    python scripts/lab/sft_queue.py --plan full
    python scripts/lab/sft_queue.py --plan pretest --self-yield-after 240   # preemption test
    python scripts/lab/sft_queue.py --plan seeds        # training seeds 1 and 2 for both losses

The lab runs at most two jobs side by side and has no job dependencies, so the whole queue
(train with the Ouro loss, train with the depth loss, evaluate each) is one job:

* Stages are idempotent. A training stage is done when its ``model.safetensors`` exists, else it
  runs with ``--resume`` (continuing from ``state/`` if a previous attempt saved one). An eval
  stage is done when its ``summary.json`` exists. Resubmitting the same command in the same
  project folder therefore continues where the last attempt stopped.
* The evaluation of a finished model runs in the background while the next model trains
  (about 10 GB next to the trainer's 44 GB), so the GPU never waits on it.
* SIGUSR1 (the lab broker asking the job to yield) is forwarded to every running stage: the
  trainer saves ``state/`` at the end of its current step and exits, evals just stop (on the
  next attempt they keep the rows already graded and continue). Then this driver re-raises
  SIGUSR1 so the lab records a stop.
* ``--self-yield-after S`` sends that SIGUSR1 to the driver after S seconds (for the test).
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
SFT = "scripts/lab/ouro_lora_sft.py"
EVAL = "scripts/lab/ouro_sft_eval.py"
MULTITURN = "scripts/lab/ouro_nestful_multiturn.py"
# Stage-1 depth-aware model on the lab (job 03ce6df5); the chain run warm-starts from it.
DEPTH_FULL = (
    "/Users/jeremywizenfeld/AI/jobs/dynamic-transformers-sft-full/results/ouro_sft/"
    "depth-full/adapters.safetensors"
)
CHAINS = "results/chains/chains_s0_n6000.json"
CHAINS_V2 = "results/chains/chains_s1_n8000_p0.2_s0.1.json"

# name, loss, extra trainer args
PLANS = {
    "full": {
        "train": [
            ("ouro-full", "ouro", []),
            ("depth-full", "depth", []),
        ],
        # task, extra eval args
        "eval": [
            ("bfcl", ["--n", "0"]),
            ("xlam", ["--n", "200"]),
            ("nestful", ["--n", "120"]),
        ],
    },
    "pretest": {
        "train": [
            (
                "pretest",
                "ouro",
                ["--max-steps", "60", "--max-len", "1024", "--save-every", "5"]
                + ["--eval-every", "60", "--eval-n", "4"],
            ),
        ],
        "eval": [("bfcl", ["--n", "2", "--steps", "2", "4"])],
    },
    # Training seeds for the loss comparison: same recipe and the same train/val split
    # (--split-seed 0, as in ouro-full/depth-full); the seed changes LoRA init and data order.
    # BFCL only, because xLAM/NESTful made the full evals slow.
    # Path B stage 2: depth-aware loss on 6,000 synthetic executed chains plus 2,000 Hermes
    # conversations (replay against forgetting single calls), 1 epoch, warm-started from the
    # stage-1 depth-aware model. Evaluated on BFCL (regression check), single-turn NESTful and
    # multi-turn NESTful without the sequential hint (``plain``).
    "chains": {
        "prepare": [(CHAINS, [PY, "scripts/lab/make_chains.py", "--n", "6000", "--seed", "0"])],
        "train": [
            (
                "chain-depth",
                "depth",
                ["--init-adapters", DEPTH_FULL, "--extra-data", CHAINS]
                + ["--hermes-train-n", "2000", "--epochs", "1"],
            ),
        ],
        "eval": [
            ("bfcl", ["--n", "0"]),
            ("nestful", ["--n", "120"]),
            ("nestful-mt", ["--n", "120", "--variant", "plain"]),
        ],
    },
    # Path B v2: as "chains" but the synthetic mix adds parallel conversations (20%, several
    # independent calls in one turn, against the BFCL parallel regression) and single-call
    # conversations (10%, teach stopping after one result), and the Hermes replay grows to 3,000.
    "chains-v2": {
        "prepare": [
            (
                CHAINS_V2,
                [PY, "scripts/lab/make_chains.py", "--n", "8000", "--seed", "1"]
                + ["--parallel-p", "0.2", "--single-p", "0.1"],
            )
        ],
        "train": [
            (
                "chain-depth-v2",
                "depth",
                ["--init-adapters", DEPTH_FULL, "--extra-data", CHAINS_V2]
                + ["--hermes-train-n", "3000", "--epochs", "1"],
            ),
        ],
        "eval": [
            ("bfcl", ["--n", "0"]),
            ("nestful-mt", ["--n", "120", "--variant", "plain"]),
        ],
    },
    "seeds": {
        "train": [
            ("ouro-s1", "ouro", ["--seed", "1"]),
            ("depth-s1", "depth", ["--seed", "1"]),
            ("ouro-s2", "ouro", ["--seed", "2"]),
            ("depth-s2", "depth", ["--seed", "2"]),
        ],
        "eval": [("bfcl", ["--n", "0"])],
    },
}

running: list[subprocess.Popen] = []
yield_requested = False
stop_signal = signal.SIGUSR1


def on_yield(signum, frame) -> None:
    """SIGUSR1 (broker yield) or SIGTERM (lab cancel): children get SIGUSR1 and save state."""
    global yield_requested, stop_signal
    yield_requested, stop_signal = True, signum
    print(json.dumps({"queue": "yield requested", "forwarding_to": len(running)}), flush=True)
    for proc in running:
        if proc.poll() is None:
            proc.send_signal(signal.SIGUSR1)


def start(cmd: list[str], log: Path | None = None) -> subprocess.Popen:
    print(json.dumps({"queue": "start", "cmd": " ".join(cmd)}), flush=True)
    out = open(log, "a") if log else None
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT if out else None)
    running.append(proc)
    return proc


def wait(proc: subprocess.Popen) -> int:
    code = proc.wait()
    running.remove(proc)
    return code


def stop_now() -> None:
    for proc in list(running):
        wait(proc)
    print(json.dumps({"queue": "stopped for yield"}), flush=True)
    signal.signal(stop_signal, signal.SIG_DFL)
    os.kill(os.getpid(), stop_signal)


def eval_chain(name: str, tasks: list) -> list[list[str]]:
    adapters = f"results/ouro_sft/{name}/adapters.safetensors"
    cmds = []
    for task, extra in tasks:
        if task == "nestful-mt":
            variant = extra[extra.index("--variant") + 1] if "--variant" in extra else "sequential"
            done = Path(f"results/ouro_nestful_mt/{name}/{variant}/summary.json").exists()
            cmd = [PY, MULTITURN, "--adapters", adapters, *extra]
        else:
            done = Path(f"results/ouro_sft_eval/{name}/{task}/summary.json").exists()
            cmd = [PY, EVAL, "--adapters", adapters, "--task", task, *extra]
        if not done:
            cmds.append(cmd)
    return cmds


def run_evals_background(name: str, tasks: list) -> subprocess.Popen | None:
    """One background shell that runs the model's pending eval tasks one after another."""
    cmds = eval_chain(name, tasks)
    if not cmds:
        return None
    script = " && ".join(" ".join(c) for c in cmds)
    log = Path(f"results/ouro_sft_eval/{name}/eval.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    # exec per command via a small Python runner so SIGUSR1 reaches the eval process itself.
    runner = (
        "import subprocess,signal,sys,json\n"
        "cmds=json.loads(sys.argv[1]);cur=[None]\n"
        "signal.signal(signal.SIGUSR1,lambda s,f:cur[0] and cur[0].terminate())\n"
        "for c in cmds:\n"
        "    cur[0]=subprocess.Popen(c)\n"
        "    if cur[0].wait()!=0: sys.exit(1)\n"
    )
    print(json.dumps({"queue": "eval in background", "model": name, "script": script}), flush=True)
    return start([PY, "-c", runner, json.dumps(cmds)], log)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", choices=sorted(PLANS), required=True)
    parser.add_argument("--self-yield-after", type=float, default=None)
    args = parser.parse_args()
    plan = PLANS[args.plan]
    signal.signal(signal.SIGUSR1, on_yield)
    signal.signal(signal.SIGTERM, on_yield)
    if args.self_yield_after:
        signal.signal(signal.SIGALRM, lambda s, f: os.kill(os.getpid(), signal.SIGUSR1))
        signal.alarm(int(args.self_yield_after))

    t0 = time.time()
    for path, cmd in plan.get("prepare", []):
        if not Path(path).exists():
            if wait(start(cmd)) != 0:
                raise SystemExit(f"prepare step failed: {' '.join(cmd)}")
    background: list[tuple[str, subprocess.Popen]] = []
    for name, loss, extra in plan["train"]:
        if Path(f"results/ouro_sft/{name}/model.safetensors").exists():
            print(json.dumps({"queue": "skip trained", "model": name}), flush=True)
        else:
            if yield_requested:
                stop_now()
            code = wait(start([PY, SFT, "--loss", loss, "--name", name, "--resume", *extra]))
            if yield_requested:
                stop_now()
            if code != 0:
                raise SystemExit(f"training {name} failed with exit code {code}")
        proc = run_evals_background(name, plan["eval"])
        if proc:
            background.append((name, proc))

    failed = []
    for name, proc in background:
        code = wait(proc)
        if yield_requested:
            stop_now()
        if code != 0:
            failed.append(name)
    summary = {}
    for name, _, _ in plan["train"]:
        for task, _ in plan["eval"]:
            path = Path(f"results/ouro_sft_eval/{name}/{task}/summary.json")
            if path.exists():
                summary[f"{name}/{task}"] = json.loads(path.read_text())[
                    "overall_mean_of_categories"
                ]
    print(json.dumps({"queue": "done", "hours": round((time.time() - t0) / 3600, 2)}), flush=True)
    print(json.dumps(summary, indent=1), flush=True)
    if failed:
        raise SystemExit(f"evaluation failed for {failed}; see results/ouro_sft_eval/*/eval.log")


if __name__ == "__main__":
    main()
