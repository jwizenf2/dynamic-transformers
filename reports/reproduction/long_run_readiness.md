# Long-run readiness — 2026-09-08

**Ready for the initial matched full-data fixed-BERT/T-DHT runs**, with the explicit
MPS replay qualification below. This is engineering readiness for a documented
independent reproduction, not a guarantee of matching the authors' scores.
No full-data training or public-test evaluation was launched in this session.

## Three targeted improvements completed

1. **Representative calibration:** main configs use 512 distinct, seeded,
   class-balanced training records (128/class), rather than the pilot's four.
   Selected IDs, normalization and descriptors are checkpointed and never refit
   on resume/evaluation. Full 114,000-record loading and balanced selection passed
   a read-only preflight. All-data/frozen-normalization deviations remain open.
2. **Complete experiment/recovery protocol:** three full epochs, equal 342,000
   exposures, no dropped/duplicated tails; explicit 10% warmup/linear decay by
   trained-sample exposure; 6,000-example validation evaluation each epoch and macroF1-best
   selection. Atomic latest saves every 1,000 updates, at epochs and explicit stops.
   Immutable best candidates protect crash recovery. JSONL/CSV streams to disk.
3. **Isolated feedback diagnostic:** literal/improvement variants differ only in
   name and disputed binary-feedback direction. Main T-DHT remains `paper_literal`;
   diagnostic scores did not redefine the reproduced method.

See [main-run protocol and commands](../../docs/reproduction/long_runs.md) and
D013 for choices, source evidence, qualifications and recovery instructions.

## Actually executed

Pinned pretrained 12-layer/12-head/768-width BERT, b=c=3, length 128, seed 7/policy 17,
Apple M2 Max MPS. Each readiness run trained the same 1,024 AG News records for two
complete epochs and evaluated the same 512 held-out validation records. This is
a subset experiment, not the full 114,000/6,000/7,600 protocol.

| Run | Updates | Training exposures | Selected validation macro F1 | Clipped rewards |
| --- | ---: | ---: | ---: | ---: |
| Fixed full BERT |256|2048|.922362|N/A|
| T-DHT, paper-literal feedback |261|2048|.870696|5|
| T-DHT, named improvement feedback |259|2048|.872349|7|

All selected checkpoints were from epoch 2. Each training record appears exactly
once per epoch, DHT queues finish empty, and logged losses are finite. Final TS
states are finite, precisions positive (minimum 1.7582/1.9184), and recorded latest
per-head logistic fits converged. This does not assert every historical fit's
convergence or statistical convergence of the research experiment.

Observed TS group-feedback counts: literal 441 zeros/329 ones; alternative 413
zeros/473 ones. Some continuous ratios hit the existing cap 100, shown above;
this modification was neither hidden nor retuned. Keep monitoring it. The F1
difference of .001652 is not evidence that the ambiguity is resolved or a significant
improvement from a one-seed, 512-example validation comparison. The fixed baseline
still outperforms both dynamic variants substantially.

Task-only validation matrix-operation ratios were .27633 literal/.26813 alternative;
routing adds approximately .25 of a full forward in both cases. These exclude other
overhead and are NOT total training FLOPs or runtime ratios. Accumulated training
step times: 40.60s fixed/56.74s literal/56.59s alternative, excluding calibration,
validation and checkpoint I/O. No new latency speedup claim. Authors' AG News
F1 .944/.945 are test reference results, not directly comparable to these short
subset-validation runs; they remain in a separate reference file.

## Recovery: what passed and what did not

The literal step 64 checkpoint was retained and replayed through step 261. The
original harness's **bitwise state check failed**. We preserved the failure and
ran a separate inspection; no tolerance was adjusted to manufacture an exact pass.

- Live MPS restoration before computation exactly matches saved model, optimizer,
  controller and MPS RNG state. CPU fixed/tree/integrated exact replay tests pass.
- All 197 replayed updates preserve discrete sample/head actions and RNG states.
  Both epoch validation results and best-step selection match exactly, excluding
  timing. First logged floating difference appears at step 67, not restoration.
- Final model maximum absolute difference: 2.0741e-5; optimizer: 1.3113e-6;
  accumulated UCB reward-vector difference: .004394. Maximum individual logged
  task-loss difference: 1.6689e-6. Continuous controller statistics are NOT identical.
- A fresh 32-example MPS validation probe has identical architectures/predictions
  and maximum absolute logit difference 1.4305e-6.

This supports operational recovery, **not bitwise MPS continuation**. The precise
backend numerical cause is not established; tiny discrepancies could change an
action near a boundary in a longer run. Keep strict deterministic mode, record
restart boundaries, and do not promise exact accelerated replay. For the user's
reliable-but-not-perfect target, qualified recovery is sufficient to start the
main experiments; it remains an open engineering limitation.

## Additional verification: two prefix passes

Confirmed in §4.2.1/§4.2.2 p.443, §5/Algorithm 2 p.444, Figure 1 p.445. Routing
uses the first three full-head layers without a gradient graph or weight update.
Training re-embeds tokens and recomputes those SAME three trainable layers, then
final task-loss backpropagation/optimizer updates them.

New instrumented tests pass on CPU AND MPS, for tree/integrated and selected
depth 3/depth 6. They verify actual call counts, separate activations, no routing
updates, nonzero task gradients and changed weights in every prefix layer, full
prefix heads, and untouched skipped layers. See [the paper-to-code explanation](../../docs/reproduction/two_pass_prefix.md).

## Verification and provenance

- CPU: **104 passed, 6 MPS skips**, 6.51s.
- With MPS: **110 passed**, 10.97s, including all eight prefix cases.
- Ruff format: 79 files unchanged; Ruff lint passed. Tests require no downloads.
- Raw logs/checkpoints: `artifacts/runs/long-run-readiness-001/`.
- Partial `readiness.json` preserves completed run results; separate `inspection.json`
  records the failed bitwise comparison and successful checks.
- [Curated configs, results, source/lock/data/checkpoint hashes](../../experiments/03_ablations/ag_news_readiness_2026-09-08.json).

Next, after runtime approval: execute `ag_news_fixed_full.toml`, then
`ag_news_tree_full.toml` sequentially. Monitor loss, validation F1, queue/sample
conservation, architecture collapse, reward clipping and actual overhead. Later,
use validation-selected `best.pt` for the once-only public-test comparison. No
speculative method changes are needed before that first controlled long comparison.
