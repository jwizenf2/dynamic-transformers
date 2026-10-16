"""Run the SFT stages of one plan in order, as one lab job that can be resubmitted after a stop.

    python scripts/lab/sft_queue.py --plan full
    python scripts/lab/sft_queue.py --plan pretest --self-yield-after 240   # preemption test

The lab runs at most two jobs side by side and has no job dependencies, so the whole queue
(train with the Ouro loss, train with the depth loss, evaluate each) is one job:

* Stages are idempotent. A training stage is done when its ``model.safetensors`` exists, else it
  runs with ``--resume`` (continuing from ``state/`` if a previous attempt saved one). An eval
  stage is done when its ``summary.json`` exists. Resubmitting the same command in the same
  project folder therefore continues where the last attempt stopped.
* The evaluation of a finished model runs in the background while the next model trains
  (about 10 GB next to the trainer's 44 GB), so the GPU never waits on it.
* SIGUSR1 (the lab broker asking the job to yield) is forwarded to every running stage: the
  trainer saves ``state/`` at the end of its current step and exits, evals just stop (they redo
  their task on the next attempt). Then this driver re-raises SIGUSR1 so the lab records a stop.
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
}

running: list[subprocess.Popen] = []
yield_requested = False


def on_yield(signum, frame) -> None:
    global yield_requested
    yield_requested = True
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
    signal.signal(signal.SIGUSR1, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGUSR1)


def eval_chain(name: str, tasks: list) -> list[list[str]]:
    adapters = f"results/ouro_sft/{name}/adapters.safetensors"
    cmds = []
    for task, extra in tasks:
        if not Path(f"results/ouro_sft_eval/{name}/{task}/summary.json").exists():
            cmds.append([PY, EVAL, "--adapters", adapters, "--task", task, *extra])
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
    if args.self_yield_after:
        signal.signal(signal.SIGALRM, lambda s, f: os.kill(os.getpid(), signal.SIGUSR1))
        signal.alarm(int(args.self_yield_after))

    t0 = time.time()
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
