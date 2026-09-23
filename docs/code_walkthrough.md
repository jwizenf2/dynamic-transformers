# Reading and extending the DHT implementation

## New: batched execution (D017) and integrated search on real data (D016)

Two execution layouts implement the SAME algorithm. Read the per-sample reference
first (`DHTController.route` → `execute` → `update`), then its batched twin:

1. `SelectableAttention._per_sample_forward`: one head tuple per sample, per-sample
   weight gathers stacked into `[batch, K·head_dim, hidden]`, one `torch.bmm`.
2. `DHTController.route_batch`: one no-grad prefix for a routing chunk.
3. `DHTController.execute_batch`: fresh prefix once per batch, then per group:
   classify samples whose depth is reached, pool contexts after one CPU transfer
   (`FeatureBank.aggregate_many`), pick counts/heads per sample (TS parameters
   pre-drawn in sample-major order), split rows by head count, run the c layers
   once per partition, rejoin. `rows` maps hidden rows to original positions.
4. `LogisticThompsonSampling.update(..., vectorized=True)` →
   `fit_logistic_map_many`: all selected heads' independent MAP fits in one solve.
5. `predict` (batched) passes the routing prefix into `execute_batch` because at
   frozen inference the rerun is identical; training always reruns it.

Enable with `[dht] execution = "batched"` (and optionally `routing_batch_size`).
Add `context_pooling = "device"` (D018) to pool bandit contexts on the GPU with
`pooled_features_batch` and score UCB arms per batch with `LinearUCB.select_many`;
bandit statistics stay float64 on the CPU.
`tests/integration/test_batched_dht.py` is the equivalence contract; read it to see
what "same algorithm" means precisely. `scripts/benchmark_training.py` measures
end-to-end training throughput; `bench` numbers live in the reproduction report.

`DHTController.explain` returns the frozen decision trace (route UCB scores per
arm, per-group count scores, TS head probabilities) through the per-sample code.
`scripts/export_predictions.py` writes per-sample JSONL used by `notebooks/`.
Integrated search on AG News: `configs/experiment/ag_news_fdht_full.toml` (D016).

## New: the long-run protocol

`training/calibration.py` selects training-only row IDs with a private RNG. The new
stratified mode refuses insufficient class coverage; it never silently falls back
to four samples. `DHTTrainer` saves the resulting IDs/counts and restores the feature
bank directly on resume. The feature mathematics and frozen-coordinate convention
are unchanged.

`training/epochs.py` is the opt-in `[training]` runner shared by fixed and DHT.
Read `scheduled_learning_rate`, `train_epochs`, and `_publish_best` in that order.
The schedule depends on trained-sample count; each epoch includes all queue tails.
Streaming logs avoid retaining all sample traces in RAM. Epoch-end validation,
earliest-tie selection, atomic latest saves and immutable best candidates are
deliberately separate from the underlying controller algorithm.

`checkpoint.pt` is for recovery; `best.pt` is for final evaluation. They may refer
to different epochs. A resumed run carries its earlier best and selection history
forward. The immutable candidate files protect that reference if a crash occurs
while publishing a later best. Schema3 adds run-level counters/selection state
without discarding older checkpoint readers. Tests simulate that crash and verify
exact interrupted/resumed fixed, tree and integrated CPU training with dropout.
Real MPS restoration is exact before computation, but continued training has small
measured numerical drift (D014); the readiness report does not claim bitwise replay.

`scripts/check_long_run_readiness.py` runs only bounded, named diagnostic configs.
The disputed feedback direction remains explicit: an alternative result cannot
silently change `ag_news_tree_full.toml`. See `reproduction/long_runs.md` for commands,
sampling and LR endpoints, logging scope, checkpoint retention and test-set rules.

For the two prefix passes, read `DHTController.route`, then `execute`, then
`DHTTrainer.step`. [This guide](reproduction/two_pass_prefix.md) links each step to
the PDF and new instrumented tests. The first call is no-grad; the second builds
a fresh task graph through the same trainable layers, not cached routing activations.

## New: real-data and physical-execution reading path

`ag_news.py`: pinned source URLs → CSV parsing → training-only stratified validation
split → serialized WordPiece tokenization → hashed non-pickle arrays. `data.load_split`
dispatches to this offline source or the original synthetic task. Experiments pin
the manifest hash; changing data requires a new explicit config, including on resume.

`models/pretrained.py`: every local embedding/encoder parameter maps to a standard
BERT name. Shape/finite checks precede import; gamma/beta map to weight/bias exactly.
The classifier remains ours. `build_model` skips pretrained import on checkpoint
resume/evaluation because the checkpoint already contains the learned weights.

`evaluation/execution.py`: profile actual layer calls and CPU mm/addmm/bmm events,
compare sliced and dense-mask outputs, and require observed matrix FLOPs to match
the same-scope analytic formula. `scripts/verify_execution.py` also checks a learned
controller's returned spec against its actual routing/task call trace. See
`reproduction/physical_execution.md` for why this still is not zero-cost training
of inactive parameters or proof of wall-clock speedup.

`configs/experiment/ag_news_{fixed,tree}_pilot.toml` specify the matched real-data
pilot; `reproduction/ag_news_protocol.md` records every non-paper protocol choice.
The core `cli.py`/`training/trainer.py` paths are reused, not separate experiment
copies. Both now flush per-step JSONL for monitoring. `evaluate_test=false` prevents
automatic public-test evaluation after a pilot; explicit evaluate commands remain
available for a predeclared final evaluation, never repeated test-based tuning.

This is the main reader's guide. The goal is an independent, understandable
implementation of the core mechanism, with explicit choices where the PDF is
incomplete. It is not an exact numerical reproduction claim.

## Start here

| Order | File | What to look for |
| --- | --- | --- |
| 1 | [config.py](../src/dht/config.py) | Dimensions, action lists, pooling and reward choices |
| 2 | [architecture.py](../src/dht/models/architecture.py) | Total depth, zero-based IDs, prefix/group invariants |
| 3 | [attention.py](../src/dht/models/attention.py) | Selected Q/K/V rows and original output columns |
| 4 | [linear_ucb.py](../src/dht/bandits/linear_ucb.py) | Ridge statistics, predicted reward and uncertainty |
| 5 | [logistic_ts.py](../src/dht/bandits/logistic_ts.py) | Gaussian sampling, logistic MAP and precision updates |
| 6 | [features.py](../src/dht/bandits/features.py) | Activations become contexts and arm descriptors |
| 7 | [controller.py](../src/dht/policies/controller.py) | route → execute → update; tree/integrated branches |
| 8 | [rewards.py](../src/dht/policies/rewards.py) | Previous-batch ratios and conflicting binary threshold |
| 9 | [queues.py](../src/dht/training/queues.py) | Depth grouping and delayed decision evidence |
| 10 | [trainer.py](../src/dht/training/trainer.py) | Training loop, checkpoint state, metrics and cost scopes |

For the shortest end-to-end read, start with DHTTrainer.step(), then follow
next_batch(), controller.route(), controller.execute(), and controller.update().
The CLI handles files/devices; the algorithm lives in importable modules.

## Follow one sample

~~~mermaid
flowchart TD
    A[Original token sequence] --> B[No-grad full-head prefix]
    B --> C[UCB chooses depth or depth/count pair]
    C --> D[Queue sample ID and decision-time context by depth]
    D --> E[Release full batch or epoch tail]
    E --> F[Rerun prefix with task gradients]
    F --> G[Tree: UCB head count / Integrated: retain chosen count]
    G --> H[Logistic TS chooses original head positions]
    H --> I[Execute next c layers with that subset]
    I -->|More selected layers| G
    I -->|Selected depth reached| J[Classifier and per-sample cross entropy]
    J --> K[One mean-loss optimizer step for released batch]
    K --> L[Previous-batch rewards and executed-action bandit updates]
~~~

Routing and training prefixes are different passes. Routing disables dropout and
gradients; training reruns the current model with gradients and configured dropout.
We do not retain stale autograd graphs while samples wait. Queues store sample IDs
and detached routing evidence.

Samples in one depth batch can have different head masks. They execute individually,
then their mean loss makes ONE optimizer step. This is straightforward but can be
slow at BERT scale. Later grouping identical masks can optimize execution without
changing the policy.

## Depth and original head positions

With total_depth=9, prefix_depth=3, group_size=3, execute layers 0–2 with all heads,
choose one subset for 3–5 and another for 6–8, then stop before layer 9. A prefix-only
choice makes no head decisions and cannot generate head-bandit updates. D001
explains the paper's contradictory b+L notation.

In a 12-head, width-768 model, each head spans 64 projected coordinates. Head 7
uses Q/K/V rows 448–511 and output columns 448–511. It must not be reassigned to
head 0's output columns. Residuals and FFNs stay width 768. Dense masking computes
every head before zeroing outputs; selected execution slices weights BEFORE matmul.

Inactive slices have zero task gradients through skipped operations. Momentum or
weight decay can still modify them in a dense optimizer. Zero task gradient does
not mean a parameter can never change.

## Linear UCB

For p aggregate features, each arm has its own A matrix [p,p] initialized to λI
and b vector [p] initialized to zero. Solve Aθ=b, predict wᵀθ, and add
exploration×sqrt(wᵀsolve(A,w)). Poorly observed feature directions have larger bonuses.
An observed action adds wwᵀ and reward×w to that arm alone.

The paper's inconsistent d/2d dimensions are repaired by using actual p. Action IDs
are positions in ordered config lists. The generic bandit does not know whether an
arm means depth 6 or count 8. Fixed ordering makes ties reproducible. Evaluation
disables the bonus and chooses from estimated mean rewards.

## Logistic TS

TS maintains a logistic parameter posterior per head, samples a Gaussian parameter
vector, computes sigmoid(wᵀθ), and retains the K largest probabilities. K comes from
the head-count decision.

The printed zero precision cannot be inverted and its 0/1 objective is inconsistent.
Our executable interpretation uses positive diagonal precision s and mean g.
On NEW binary feedback, minimize:

~~~text
0.5 * Σ_j s_old[j] * (θ[j] − g_old[j])²
  + Σ_i (softplus(w_iᵀθ) − y_i * w_iᵀθ)
~~~

Damped Newton finds the new mean using a full temporary Hessian. The STORED Gaussian
is still diagonal: precision adds p_i(1−p_i)w_i², and sampling uses standard deviation
1/sqrt(precision). The prior already summarizes history; reusing all past labels
would double-count them.

A subset has one observed label, shared only among selected heads. We do not invent
individual head credit or feedback for unused heads. The dedicated generator and
fit diagnostics are saved. Nonconvergent fits raise explicitly. D005 labels these
mathematical repairs rather than claiming they reproduce unknown author code.

## Contexts and arm descriptors

FeatureBank.calibrate uses a configured TRAIN subset, no labels or evaluation data.
For each sample it runs a prefix and separate paths for each configured head count
plus full heads. Prefix activations are reused within calibration. UCB arms average
corresponding depth/count/group representations; TS descriptors average original
embedding-position chunks. These chunks are descriptors, not isolated head outputs.

Contexts remove padding then apply sequence/embedding mean-pooling grids. Widths
come from the bin sums. Paper bins yield raw widths 84/42/12; aggregate widths are
168 for depth/pair, 87 for head count with three group markers, and 24 for TS.
The printed TS width 13 is unexplained; no padding/truncation forces it.

Normalization fits running moments during calibration then freezes them, keeping
a stable ridge coordinate system. Frozen moments, bounded calibration subset and
exhaustive count coverage are our documented choices. Calibration costs are measured.

## Rewards and traps

Products are loss×depth, loss×head_count and loss×depth×head_count. Only the first
is completely explicit in print. Reward is scale×previous_mean/current_product.

All samples in a batch use the SAME old history for a key; only afterward is its
mean replaced. Each observed head group has a key. Missing groups do not create zero
rewards. A key's first batch initializes history and skips its bandit updates.
epsilon protects division; a configurable cap bounds extremes and counts clips.

| Configuration | TS label 1 means | Fidelity |
| --- | --- | --- |
| paper_literal | continuous reward ≤ 1 | Printed p.444 inequality; default |
| improvement | continuous reward > 1 | Prose-consistent correction/ablation |

The source disagrees with itself here. Both options are named so a correction is
never hidden. D007 documents the product/history/scaling assumptions.

## Training, evaluation and resumption

steps counts optimizer updates. A step can train an epoch-tail partial batch.
The cursor counts routed samples, so other queues may still be waiting. A step-budget
stop preserves them. At epoch completion tails are flushed without drops.

Schema 2 includes model, optimizer, RNGs, calibration, bandits, TS generator, rewards,
queues and epoch/cursor. Restore global RNGs AFTER constructing the model/trainer,
since construction consumes random numbers. Resume never recalibrates a changed
model. Tests compare model/optimizer/posterior/history/queues against uninterrupted
CPU training with dropout. Timing counters naturally need not match.

Evaluation is read-only: dropout off, UCB bonuses off, TS means, frozen normalization,
no labels for selection, no sampling or updates. It bypasses queues and includes
both prefix passes.

## Costs and logs

task_forward_matmul_flops counts selected attention, full FFNs, prefix and classifier.
Routing/calibration are separate so duplicated work stays visible. Estimates exclude
non-matmul neural operations, controller math and backward/optimizer FLOPs. No total
training ratio or paper-equivalent speedup is inferred.

Training times routing, forward including head decisions, backward/optimizer and
controller updates separately. Evaluation/benchmark includes routing and policy.
The execution audit measures supported CPU matrix operations; complete training
FLOPs/memory remain future work. Null means unmeasured, not zero.

~~~sh
uv run dht train --config configs/experiment/dht_tree_smoke.toml
uv run dht train --config configs/experiment/dht_integrated_smoke.toml
uv run pytest tests/unit/test_bandits.py tests/unit/test_features_rewards.py tests/integration/test_dht.py
~~~

Open config.json for effective settings, metrics.json for sample architectures/rewards,
metrics.csv for timings/loss, summary.json for distributions/evaluation, and
metadata.json for source/lock hashes. Copy a TOML file to study a nuance; record its
hypothesis and compare with an unchanged baseline. Prepared AG News now uses the
same loop; full multi-epoch reproduction remains next. Synthetic runs demonstrate
mechanics, not NLP quality.
