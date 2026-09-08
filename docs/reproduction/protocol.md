# Current reproduction protocol

AG News now uses these same fixed/DHT paths with explicitly prepared, hash-pinned
data and optional BERT-base weight import. See [the real-data protocol](ag_news_protocol.md)
for settings distinct from the synthetic defaults below and the bounded pilot scope.
The opt-in epoch protocol is now specified in [long_runs.md](long_runs.md). It adds
balanced512 calibration, complete corpus exposure, a sample-based LR schedule,
epoch-end validation selection, and periodic schema3 checkpoints. It does not
reinterpret the controller equations or launch full-data training automatically.

## Executable scope

Fixed and learned tree/integrated DHT run on a versioned synthetic token task.
The second token determines the label; a first-token classifier must use attention.
Independent split RNG streams provide train/validation/test examples. This is
software validation, not evidence of reproduced NLP accuracy.

Standalone TOML uses validated model/data/bandit/DHT objects. [dht] enables learned
selection; fixed ArchitectureSpec cannot be supplied simultaneously. Effective
configuration, source/config/lock hashes, versions, device and seed are saved.

D005–D008 specify positive diagonal logistic TS, repaired binary objective, explicit
loss-cost products, literal feedback default, numerical cap, bounded train-only
calibration and frozen normalization. Source gaps remain documented.

## Data and training order

Legacy calibration uses the first min(calibration_samples,train_size) TRAIN examples.
New main configs select512 class-balanced training records; labels are used only
for stratification, not as bandit context. Descriptors use no dropout, gradients or
evaluation data. Each head count plus full heads gets a path; the prefix is reused.
Calibration remains fixed and checkpointed with selected IDs/counts.

Training cycles in fixed dataset order. Routing is no-grad with dropout off.
Samples enter depth FIFOs with original ID and decision-time features. Ready queues
train immediately. Epoch tails flush in depth order at their actual sizes.
Step-budget stops save waiting entries rather than pretending an epoch completed.

Each depth batch executes per-sample heads; mean cross entropy makes one optimizer
step. Detached pre-update losses yield previous-batch ratios. Only observed actions
update. A reward key's first observation initializes history; missing groups retain
their most recent observed batch mean.

## Evaluation/resumption

Inference disables UCB bonuses and uses TS means; no sampling, normalizer fitting,
bandit update or optimizer update occurs. It bypasses queues and includes routing
plus prefix re-execution. Report accuracy and macro/micro/weighted F1; missing-class
F1 is zero.

Schema 2 includes model, optimizer, config, global/independent RNGs, calibration,
bandits, reward history, queues and epoch/cursor. Resume requires the same effective
config and never recalibrates. Exact CPU resumption is tested for both modes with
dropout. Schema 3 adds epoch-run progress and best-selection history. MPS checkpoint
restoration is exact before computation, but real-backbone continuation has measured
floating drift (D014); matching bounded-run decisions/validation does not certify
bitwise accelerator replay. Cross-device identity is not promised. MPS gradients and context
transfers have separate accelerator regressions (D011/D012); CUDA remains unverified.

## Computation/logging

Logs include per-sample architectures/rewards, depth/count distributions, head
frequencies and separate stage times: routing, forward including head policies,
backward/optimizer, controller update. Calibration is separate. Resumed-run training
distributions/times refer to that segment; explicitly cumulative routing/calibration
counters retain earlier work. The new epoch runner additionally preserves cumulative
training sample/architecture counters across segments; per-step files remain segment logs.

Matmul estimates use two FLOPs per multiply-add: active Q/K/V/output, QK/AV, full
FFNs and classifier. Prefix/calibration exclude classification when unexecuted.
Task/routing/calibration estimates are separate and exclude controller,
backward/optimizer and non-matmul operations. No total training ratio is claimed.

Evaluation/benchmark reports synchronized actual time including routing/policy.
Per-sample Python execution prioritizes correctness/readability. An independent CPU
operator profiler now verifies forward matrix dimensions/counts and skipped layers;
complete training FLOPs/memory remain unmeasured. Authors' tables stay separate.

## Before a full AG News reproduction

Pinned data/tokenizer preprocessing, a training-only validation split and explicit
pretrained initialization are implemented. The earlier pilot is not the main-run
protocol: D013 now specifies optimizer/schedule/batch size/epochs/seeds and checkpoint
selection in `long_runs.md`. Bounded readiness checks are complete with the explicit
MPS qualification. Full-data execution still awaits runtime approval.

Compare matched fixed/T-DHT first. Tune on validation and evaluate selected checkpoints
on test. Record seeds/uncertainty, architecture choices and actual overhead. Then
extend to published S/F, other datasets and separately interpreted P-DHT.
The explicitly approved pretrained model and dataset downloads are local. No
full-scale multi-epoch training has been launched.
