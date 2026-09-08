# AG News baseline, T-DHT pilot and physical execution audit

Completed September 7–8, 2026 on Apple M2 Max, 32 GiB RAM, MPS,
macOS 26.6.2, Python 3.12.12, PyTorch 2.14.0. Strict deterministic algorithms stayed
enabled. No cloud resources were used. This is a **real-data pilot, not a completed
numerical reproduction** of the paper's Tables 1/2.

## Matched protocol and results

Both models start from the same pinned BERT-base-uncased encoder and identically
seeded classifier: 12 layers, 12 heads, hidden size 768, FFN 3072, sequence length
128, four classes. T-DHT uses the published b=c=3, depths {3,6,9,12}, head counts
{8,10,12}. See [protocol](../../docs/reproduction/ag_news_protocol.md) and decisions
D010–D012 for initialization, data and implementation nuances absent from the paper.

Each pilot used 128 AdamW updates, batch 8, learning rate 2e-5, weight decay .01,
and a shuffled 1,024-record training subset. Fixed trained all 1,024; T-DHT trained
1,018, with six depth-9 records still queued and checkpointed. No samples were
silently dropped. This is a matched update budget, not exactly equal sample exposure.

The initial 128-record validation results were optimistic. Both unchanged final
checkpoints were then evaluated on **all 6,000 held-out validation records** from
the original training set, with no further training, calibration or checkpoint
selection. The public 7,600-record test was not evaluated.

| Our measured full-validation result | Fixed BERT | Tree DHT |
| --- | ---: | ---: |
| Accuracy / micro F1 | .901167 | .822500 |
| Macro / weighted F1 | .901090 | .823288 |
| Average executed task depth | 12 | 4.7855 |
| Task-forward matrix FLOPs / fixed | 1.0000 | .3893 |
| Task + extra routing-prefix matrix FLOPs / fixed | 1.0000 | .6393 |

Validation is class-balanced, so macro and weighted F1 coincide. These are one-seed
results, not means across independent runs. The paper's unspecified F1 averaging
and our validation-versus-test distinction prevent a direct reproduction-gap claim.

Training exercised all published choices: depth counts 445/389/88/96 for 3/6/9/12,
dynamic-group head-count counts 430/313/110 for 8/10/12 heads. Full-validation depth
counts were 2478/3492/11/19. Head counts were 329/3240/2 over executed dynamic groups.
Thus the policy is neither a constant architecture nor a controller that never
exercises head selection, but it strongly favors shallow paths after this short run.

## Comparison with the authors' results

The source is the supplied PDF, printed p.447, Tables 1/2. Original values are kept
separately in [paper_results.csv](../../references/paper_results.csv).

| Authors' AG News public-test results | F1 | Training FLOP ratio | Inference FLOP ratio |
| --- | ---: | ---: | ---: |
| Full BERT / 12-VaBERT | .944 | 1.00 | 1.00 |
| T-DHT | .945 | .42 | .81 |

Our pilot does **not** reproduce the authors' near-baseline T-DHT quality. It also
does not establish their FLOP ratios: our counts cover specified forward matrix
operations; the paper's complete accounting is unclear. Do not compare the pilot's
.389 task-only ratio to .81 while omitting routing, or label it an improvement.
The approximately one-subset-pass pilot is much shorter than their stated 3–6
epochs. Pretrained initialization, small calibration, classifier/pooling and all
previously documented bandit interpretations remain explicit fidelity limitations.

## Physical skipping: measured, not inferred from zeros

The CPU operator profiler ran the actual model functions on real validation tokens.
With the same weights/input, a 9-layer candidate with eight noncontiguous original
heads in layers 3–8 produced these executed mm/addmm/bmm counts:

| Task path | Supported forward matrix FLOPs / sample |
| --- | ---: |
| Full 12-layer / 12-head | 22,347,257,856 |
| Nine layers, dense heads then zero-mask | 16,760,444,928 |
| Nine layers, physically sliced heads | 15,451,822,080 |

Slicing used .69144 of full-model matrix work and .92192 of the nine-layer dense
reference. Sliced and dense-reference logits matched exactly in the recorded check.
Layer hooks verified that layers 9–11 were never called. Q/K/V projections and the
two attention products had only the selected head width/count. Original head
positions were preserved in output projection. The first three layers remained full.
Learned-policy traces independently check actual routing/task calls against the
controller's returned architecture, including prefix re-execution.

Concrete learned-policy example: validation record index 4 selected total depth 6
and excluded original heads **2 and 7** (zero-based) from layers **3,4,5**. The
observed layer trace was `[0,1,2,0,1,2,3,4,5]`: routing prefix, repeated task prefix,
then the dynamic group. Layers 6–11 were never called. Dynamic Q/K/V width was
**640 instead of 768**, and output projection consumed that same 640 active width.
The script's `--find-pruned` option locates an actual frozen-policy example without
changing its decisions; search forwards are outside the profiler and not timed.

Backpropagation was also verified: CPU finite-difference first/second derivative
tests, CPU/MPS sliced-versus-dense input/parameter gradient comparisons, inactive
head gradient checks and skipped-layer `grad=None` checks pass. The private unique
gather writes gradients to unique original channels without atomic accumulation.
It still allocates dense gradient buffers; it is not a sparse optimizer.

The [physical execution guide](../../docs/reproduction/physical_execution.md)
explains the remaining FFN, padding, dense AdamW, gather, routing and controller costs.

## Actual timing (separate from the profiler)

Sequential MPS inference benchmarks, batch 8, length 128, five warmups, 20 repeats,
synchronized before/after each measurement:

| Measured local snapshot | Fixed BERT | Tree DHT |
| --- | ---: | ---: |
| Mean inference batch latency | 55.688 ms | 113.912 ms |
| Throughput | 143.66 samples/s | 70.23 samples/s |
| 128-update training segment | 34.428 s | 33.463 s |

Tree inference was **2.046× slower**, despite less neural matrix work. The baseline
batches eight samples; the current DHT path executes samples separately and includes
CPU policies/transfers/routing. This scope difference is real end-to-end behavior,
not evidence that a zero mask was used. We have not causally isolated every overhead.

T-DHT also spent 1.417 s in calibration before its training segment, making
calibration + segment about 34.881 s. Its segment contained 8.215 s routing,
12.524 s forward/head decisions, 10.905 s backward/optimizer and 1.642 s controller
updates. Timings exclude setup, data preparation and checkpoint I/O. These short,
single-run snapshots—on different dates, with slightly different sample exposures—
do not demonstrate a training speedup. Full-training FLOPs and peak memory remain
unmeasured, not zero.

## Failures caught and verification

- Fixed launch `ag-news-fixed-pilot-001`: stopped before training because the
  pinned weights retain LayerNorm gamma/beta names. Added strict name conversion;
  no tensors were randomly filled to bypass the failure.
- Tree launch `ag-news-tree-pilot-001`: stopped during calibration. Finite MPS
  activations became nonfinite in the combined CPU/float64 transfer on this stack.
  Transferring original dtype first, then converting on CPU, exactly matches the
  CPU context reference. No NaN replacement was used.
- Tree launch `ag-news-tree-pilot-002`: stopped after a prefix-only update because
  MPS rejected general accumulating gather backward under strict determinism.
  Unique-index assignment implements the same derivative without duplicate-index
  accumulation. Strict determinism was not disabled.
- Successful runs: `artifacts/runs/ag-news-fixed-pilot-002/` and
  `artifacts/runs/ag-news-tree-pilot-003/`. Live losses agree with final metrics;
  first-to-last 16-step mean losses decreased 1.310→.260 and 1.424→.452.
- Post-run checks inspected 796 fixed and 822 DHT saved tensors for finiteness,
  positive TS precision, converged saved fits, valid logged architectures and queue
  conservation. These are post-run checks, not continuous inspection of every gradient.
- Final full test suite with MPS available: **88 passed in 2.23 s**. Sandboxed CPU
  suite: **86 passed, 2 MPS tests skipped in 1.86 s**. Ruff formatting/lint passed.

## Artifacts, commands and next milestone

[Curated machine-readable record](../../experiments/02_related_work/ag_news_pilot_2026-09-08.json)
contains configs, source/weight/data hashes, full validation, benchmarks, checkpoint
checks and profiler evidence. Raw arrays, weights, checkpoints and logs are ignored
by Git. No author DHT implementation was downloaded or used.

Commands are in the protocol. Additional read-only checks:

```sh
uv run --extra nlp python scripts/evaluate_full_validation.py --checkpoint artifacts/runs/ag-news-tree-pilot-003/checkpoint.pt --output artifacts/runs/new-full-validation.json
uv run --extra nlp python scripts/verify_execution.py --checkpoint artifacts/runs/ag-news-tree-pilot-003/checkpoint.pt --find-pruned --output artifacts/runs/new-execution-check.json
```

Next: establish a full-split, matched epoch/sample-exposure protocol with scheduled
validation, resumable periodic checkpoints and final validation-based selection.
Choose calibration size/schedule and optimization schedule explicitly, run both
baselines for the same corpus exposure, then evaluate selected checkpoints once on
the full public test. A rough linear extrapolation from the fixed pilot suggests
about 3.2 hours for three passes over 114,000 records at batch 8, excluding evaluation
and I/O; DHT architecture distribution can change that estimate substantially. No
multi-hour run was launched. Execution batching and altered rewards belong in
separate documented experiments, not silent changes to this reproduced core.
