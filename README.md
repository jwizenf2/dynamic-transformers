# Dynamic Transformers with Reinforcement Learning

**Improving Sample-Based Architecture Selection** — a semester research project
and independent implementation of Meng et al., *Sample-Based Dynamic Hierarchical
Transformer with Layer and Head Flexibility via Contextual Bandit*, Proceedings on
Engineering Sciences 6(2), 439–452 (2024), DOI
[10.24874/PES06.02.001](https://doi.org/10.24874/PES06.02.001).

The first academic checkpoint is **Select & Present a Research Paper** and
**Related Work Implementation**. We aim to reproduce the core contextual-bandit
method independently, then evaluate an original extension after the reproduction
is stable. No authors' code was imported. Candidate extension ideas are not novelty
claims or approved research contributions.

## Current status

Independent linear UCB, logistic-Gaussian Thompson Sampling, tree/integrated DHT,
depth queues, and full controller checkpoint/resume run on synthetic or explicitly
prepared AG News data. A pinned BERT-base import supports matched 12-layer/12-head
real-data pilots. **Full paper-number reproduction remains unverified.** No pretrained
model or real dataset is downloaded implicitly. Start with the
[guided code walkthrough](docs/code_walkthrough.md).

The [long-run readiness checks](reports/reproduction/long_run_readiness.md) are
complete:512-record balanced calibration, an epoch-based full-data protocol,
periodic/latest and validation-best checkpoints, and a controlled feedback
diagnostic. **110 tests pass with MPS available.** CPU replay is exact; MPS
restoration is exact but continued training shows small numerical differences,
with matching choices/scores in the measured replay. No full-data runs have begun.
See [main-run commands and recovery](docs/reproduction/long_runs.md).

The scaffold slices Q/K/V projection rows and output projection columns before
matrix multiplication. A dense masked reference path verifies head semantics.
The hidden dimension and residual/FFN widths stay fixed. DHT routes each sample
and executes its own head subsets inside a depth-homogeneous training batch. Selected total depth includes the full-head prefix. See
[decisions](docs/project/decisions.md) for this interpretation and its evidence.

**Update 2026-09-22:** integrated search (F-DHT) now has a full matched AG News run
(test macro F1 .9409 vs T-DHT .9400, fixed .9490), an opt-in batched execution engine
(`[dht] execution = "batched"`, D017; plus `context_pooling = "device"`, D018) under which
T-DHT trains 1.6× and infers ~1.1× faster than fixed BERT on MPS, analysis notebooks in `notebooks/`, and a verified research-topic list in
`docs/research/next_topics.md`. See
[the report](reports/reproduction/ag_news_fdht_and_batched_execution.md). 130 tests pass.
Notebooks need `uv sync --locked --extra nlp --extra analysis`.

## Setup and verification

Python 3.12.12 and [uv](https://docs.astral.sh/uv/) are used. Install uv separately
if it is not available. No GPU, dataset download or pretrained checkpoint is needed.

```sh
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run dht train --config configs/experiment/smoke.toml --output artifacts/runs/fixed-smoke
uv run dht train --config configs/experiment/smoke_dynamic.toml --output artifacts/runs/dynamic-smoke
uv run dht evaluate --checkpoint artifacts/runs/dynamic-smoke/checkpoint.pt --split validation
uv run dht benchmark --checkpoint artifacts/runs/dynamic-smoke/checkpoint.pt --output artifacts/runs/dynamic-smoke/benchmark.json
```

Run directories must be new; commands refuse to overwrite existing runs. Omit
`--output` on training for a timestamped directory. The package CLI replaces thin
`scripts/train.py` wrappers. Configurations are standalone TOML with validated
defaults; every effective setting is saved to `config.json` and `metadata.json`.

CPU is the smoke default. Set `device = "auto"`, `"mps"`, or `"cuda"` in a new
configuration to select another backend. Auto chooses CUDA, then MPS, then CPU;
an unavailable explicit device warns and falls back unless
`allow_device_fallback = false`. Evaluation/benchmark accept `--device` separately.
Accelerator correctness and cross-device reproducibility require additional checks.

## Learned DHT commands

```sh
uv run dht train --config configs/experiment/dht_tree_smoke.toml --output artifacts/runs/my-tree
uv run dht train --config configs/experiment/dht_integrated_smoke.toml --output artifacts/runs/my-integrated
uv run dht evaluate --checkpoint artifacts/runs/my-tree/checkpoint.pt --split validation
uv run dht benchmark --checkpoint artifacts/runs/my-tree/checkpoint.pt
```

Adding `[dht]` enables the learned policy. `mode = "tree"` selects depth and then
head counts; `mode = "integrated"` chooses a depth/count pair. Both use Gaussian
logistic TS for specific heads. The defaults preserve the printed binary-feedback
inequality; `feedback_rule = "improvement"` is a named correction, not a silent fix.
See [adopted decisions D005–D008](docs/project/decisions.md) for the complete nuances.

## Checkpoint and resume

```sh
uv run dht train --config configs/experiment/smoke.toml --stop-after 1 --output artifacts/runs/partial
uv run dht train --config configs/experiment/smoke.toml --resume artifacts/runs/partial/checkpoint.pt --output artifacts/runs/resumed
```

Resume uses the same configuration and continues to its configured total steps.
Checkpoints contain model, optimizer, step/data cursor, configuration, and Python,
NumPy, PyTorch and independent policy RNG states. CPU interrupted/resumed training
is tested with dropout for fixed, tree, and integrated paths. DHT checkpoints also
persist calibration, bandit statistics, TS sampling state, reward history, pending
queues and epoch/data cursor. Only load trusted local checkpoints. The same resume
commands work with the DHT configuration files.

Each run writes local JSON/CSV losses, validation/test accuracy and macro/micro/
weighted F1, architecture information, a partial forward FLOP estimate, elapsed
training time, a checkpoint, and code/dependency/device provenance. Benchmark
measures synchronized forward batch latency and throughput after warmup. These
tiny measurements do not predict BERT-scale performance. DHT reports calibration,
routing, task-forward/head-policy, backward/optimizer and controller-update time
separately. The execution audit verifies supported CPU matrix operations; complete
training-operation and peak-memory accounting remain future work.

## Real AG News and physical compute skipping

The [completed AG News pilot report](reports/reproduction/ag_news_pilot.md) records
the matched pretrained 12-layer runs and all 6,000 held-out validation results:
macro F1 .9011 fixed versus .8233 T-DHT. Physical skipping is verified, but T-DHT
inference was about 2× slower. These are pilot validation results, not paper test
reproduction. That earlier session passed88 tests; the latest suite passes110.

See [the AG News protocol and commands](docs/reproduction/ag_news_protocol.md) for
explicit dataset/weight downloads, the train-only validation split, hash-pinned
configuration, and bounded fixed/T-DHT pilots. Install `uv sync --locked --extra nlp`
for these workflows; ordinary offline tests do not need downloads or that extra.
Live per-step metrics are flushed to each run's `metrics.jsonl`.

[Physical execution explained](docs/reproduction/physical_execution.md) follows
head IDs through smaller Q/K/V matrices and matching output columns. The production
path does not zero-mask computed heads. `scripts/verify_execution.py` profiles
actual matrix dimensions and layer calls, including a learned policy when passed
its checkpoint. Dense masking is a reference, not a compute optimization. Dense
optimizer state, FFNs and routing still cost work even when attention is sliced.

[The two prefix passes](docs/reproduction/two_pass_prefix.md) explains the paper
and code: layers0–2 run without gradients for routing, then run again with task
gradients during training. They are not permanently frozen. CPU/MPS tests verify
actual layer calls, gradients and weight updates, not just configuration flags.

## Research sources and fidelity

The [monitored training audit](reports/reproduction/monitored_training_audit.md)
records twelve 300-step synthetic runs. Numerics were stable, but DHT favored
shallow paths, underperformed the full fixed model, and ran slower in the current
per-sample CPU implementation. The paper's quality/efficiency result is not yet
reproduced. Repeat the live-log check with:

```sh
uv run python scripts/audit_training.py --output artifacts/runs/my-audit
```

The [supplied PDF](dynamic_transformers_reference_paper.pdf) is the source of truth.
The audit records section/page/formula evidence and distinguishes **explicitly
specified**, **reasonable interpretation**, and **our modification**. Start with:

- [Paper summary and reconstructed equations](docs/reproduction/paper_summary.md)
- [Notation glossary](docs/reproduction/notation.md) and [algorithm mapping](docs/reproduction/algorithm_mapping.md)
- [Reproducibility issues and alternatives](docs/reproduction/ambiguities.md)
- [Experimental protocol](docs/reproduction/protocol.md)
- [Implemented UCB/TSB specification](docs/reproduction/bandit_implementation_plan.md)
- [Code walkthrough and nuances](docs/code_walkthrough.md)
- [Authors' Table 1/2 data](references/paper_results.csv) and [published settings](references/paper_settings.json)
- [Checkpoint presentation outline](reports/checkpoint_1/presentation.md)

The 64 reference-table records are authors' reported values, not our measurements.
The paper's “up to 74%” claim corresponds to an estimated .26× FLOP ratio and
does not establish .26× runtime. Our current `forward_matmul_flops` includes active
attention, full FFNs, the executed prefix and classifier, but excludes non-matmul
operations, backward, optimizer and routing/controller overhead. It must not be
presented as a complete paper-equivalent cost or a training speedup.

Material open issues include inconsistent depth/features, singular TS initialization,
an invalid printed logistic objective, conflicting binary feedback, incomplete head
reward definitions, unknown head-combination restrictions, and missing training
hyperparameters. Our executable conventions are recorded beside the mathematics
and in the decision log. Adopting a convention does not resolve the source's gaps.

## Repository layout

```text
dynamic-transformers/
├── AGENTS.md, README.md, pyproject.toml, uv.lock, .python-version
├── dynamic_transformers_reference_paper.pdf
├── configs/experiment/          # CPU fixed/dynamic sanity configurations
├── src/dht/
│   ├── config.py, cli.py, data.py, runtime.py
│   ├── models/                 # architecture, selectable attention, encoder
│   ├── bandits/                # linear UCB, logistic TS, calibrated contexts
│   ├── policies/               # tree/integrated controller and rewards
│   ├── training/               # DHT loop, depth queues, checkpoint/resume
│   └── evaluation/             # F1 metrics, forward matmul estimate
├── tests/{unit,integration,regression}/
├── docs/{project,reproduction,research}/
├── references/                 # bibliography and author reference data
├── reports/checkpoint_1/        # presentation outline
├── experiments/00_sanity/       # curated observed verification
└── artifacts/                  # ignored generated runs/checkpoints
```

Empty future module directories are intentionally deferred. Large datasets, weights,
caches and generated runs stay outside version control; curated experiment
manifests/summaries remain tracked.

## Staged roadmap

1. Audit, repository foundation, deterministic encoder correctness — implemented.
2. Independent UCB/logistic TS and mathematical tests — implemented.
3. Contexts, rewards, tree/integrated policies, queues, tiny integration — implemented.
4. AG News fixed baseline versus T-DHT; checkpoint presentation and observed results.
5. Published S/F action configurations, multiple seeds/datasets, P-DHT interpretation and table reproduction.
6. Literature-informed original extension with baseline and ablations.

See the [concrete plan](docs/project/roadmap.md), [progress](docs/project/progress.md),
and [research log](docs/research/experiment_log.md). Before the next research run,
record the code/configuration, dataset version, seeds, validation protocol and budget.
No full-scale training, multi-gigabyte model download or paid compute has been launched.
