# Monitored training audit and comparison with the paper

Observed 2026-09-07. Conclusion: the implementation is numerically stable in these
bounded runs, but the desired accuracy/computation tradeoff has NOT been reproduced.
DHT exits early, loses synthetic classification quality relative to the full
baseline, and is slower in the current per-sample CPU implementation.

## Protocol and reproducibility

Twelve runs: fixed full encoder, tree DHT, integrated DHT and a diagnostic fixed
prefix-only control, each with seeds 7/11/19 and 300 optimizer steps (3600 total).
The one-layer control was added after observing early exits on training/validation
logs; it is a diagnostic, not tuning or a claimed paper baseline.

All use CPU, one PyTorch thread, random 3-layer/4-head/width-24 backbone, prefix=1,
group=1, sequence length 8, dropout .1, AdamW lr=.001, zero weight decay, batch size 8,
and synthetic data with 256 train / 128 validation / 128 test samples and 3 classes.
Seeds affect model initialization AND synthetic data; the reported sample standard
deviation captures both, unlike repeated model seeds on a fixed real corpus.
Matched modes use the same data and initialization seed. Per-sample execution and
batch reordering mean they do not share an identical dropout trajectory.

Tree/integrated actions use depths {1,2,3} and counts {2,4}; calibration uses 32
training examples. Literal printed feedback and existing D005–D008 conventions
are unchanged. No real dataset/pretrained model was downloaded.

Fixed modes saw 2400 samples/run; tree saw 2298/2320/2312 and integrated
2312/2320/2328 because depth-queue epoch tails make some batches smaller.
This is an equal-optimizer-step diagnostic, not an equal-data/compute-budget claim.

Reproduce:
~~~sh
uv run python scripts/audit_training.py --output artifacts/runs/new-audit
uv run python scripts/audit_training.py --output artifacts/runs/new-prefix-control --modes prefix
~~~

The monitor prints/flushes JSON every 25 steps and checks every optimizer update.
Its hooks call original functions unchanged, inspect detached values and do not
alter actions/rewards/optimizer settings/RNGs. Instrumented training times contain
diagnostic overhead and must not be used as speed benchmarks.

Raw logs, audits, configurations and checkpoints are in
artifacts/runs/monitored-audit-001/ and artifacts/runs/monitored-prefix-control-001/.
Machine-readable curated records, source hashes and paper references:
[monitored_training_2026-09-07.json](../../experiments/00_sanity/monitored_training_2026-09-07.json).
Core numerical algorithms were not changed in this audit.

## Checks passed

- All 12 runs completed 300 steps; all observed losses, gradients and weights were finite.
- All DHT UCB statistics and logistic means/precisions remained finite; precisions positive.
- Every inspected TS fit converged; no reward clipping occurred.
- All expected bandits received observations, including all three tree decision levels.
- Routed sample counts equaled trained samples plus pending queue entries at every step.
- Every run's mean loss over its last 25 updates was below its first 25 updates.
- Existing mathematical/integration/regression suite: **74 passed in 2.37 seconds**.

These checks establish bounded numerical/behavioral sanity, not universal correctness,
convergence or reproduction. Test metrics were evaluated after training and were not
used to select hyperparameters or checkpoints.

## Our synthetic results

Mean ± sample standard deviation across three seeds. Loss columns average each
run's first/last 25 step losses (not weighted equally per sample when batches differ).

| Mode | Validation macro F1 | Test macro F1 | Test accuracy | Training loss |
| --- | ---: | ---: | ---: | ---: |
| fixed | 0.644 ± 0.081 | 0.699 ± 0.131 | 73.2% ± 9.5% | 1.153 → 0.684 |
| prefix | 0.557 ± 0.096 | 0.579 ± 0.094 | 61.2% ± 5.7% | 1.133 → 0.882 |
| tree | 0.526 ± 0.152 | 0.524 ± 0.162 | 58.3% ± 10.6% | 1.125 → 0.896 |
| integrated | 0.505 ± 0.025 | 0.511 ± 0.033 | 54.4% ± 5.0% | 1.142 → 0.921 |

The fixed baseline learns the task best under this budget. Tree DHT's validation
mean depth was essentially 1; its test mean depth was 1.005. In the first 25 updates,
only 60–72% of trained tree samples chose depth 1; in the final 25, about 93–95%
did. Very few later examples train the deeper layers.

The fixed one-layer control is also weaker than the full baseline and is close
to tree DHT. This supports shallow-path capacity/learning budget as PART of the
explanation, not proof of a particular routing bug. DHT's additional gap can also
reflect architecture reordering, head masking, dropout trajectory and different
sample exposure. The small/high-variance experiment does not isolate those causes.

Integrated DHT retained a mean test depth of 1.565, usually using two heads when
dynamic groups were executed, but did not recover fixed-baseline quality. Finite
loss and optimizer convergence alone therefore do not establish a useful policy.

## Actual execution time and accounting

Separate uninstrumented CLI benchmarks, seed 7, same first validation batch of 8,
sequence length 8, 5 warmups and 50 repeats. Runs were measured sequentially.
DHT includes routing and policy execution; fixed uses its normal batched path.

| Mode | Mean batch latency (ms) | Samples/s | Runtime/full |
| --- | ---: | ---: | ---: |
| fixed | 0.578 | 13852 | 1.00× |
| tree | 5.708 | 1401 | 9.88× |
| integrated | 6.284 | 1273 | 10.88× |

These are measurements of the implementation as it currently executes, not controlled
kernel-only comparisons. The fixed encoder batches samples; DHT loops over samples,
does controller/CPU feature work and reruns the prefix. Those implementation choices
are consistent with the measured slowdown; no profiler breakdown isolates each cause.

Across three seeds, test task-forward matmul ratios were 0.335 (tree) and 0.471
(integrated). Counting the additional routing prefix raises these to approximately
0.668 and 0.804. Neither includes non-matmul/controller operations. Calibration adds
training overhead; backward/optimizer FLOPs remain unestimated. This is why reporting
only selected-path FLOPs would materially overstate total savings.

## What can be compared with the paper

Source: supplied PDF, Section 6.2, Tables 1 and 2, printed p.447
(PDF page 9); [reference data](../../references/paper_results.csv).

| Authors' AG News result | Reported F1 | Training FLOPs | Inference FLOPs |
| --- | ---: | ---: | ---: |
| Full BERT | .944 | 1.00× | 1.00× |
| T-DHT | .945 | .42× | .81× |
| P-DHT | .935 | .59× | .39× |
| S-DHT | .941 | .38× | .37× |
| F-DHT | .943 | .59× | .35× |

The authors report near-baseline AG News F1 with lower estimated operations.
Our synthetic audit has NOT demonstrated that combination: DHT loses quality and
the actual CPU implementation is slower. We cannot compute a valid numerical
reproduction gap against .945: data/task, backbone, initialization, training budget,
feature/reward conventions and F1 averaging differ. Nor can our partial matmul ratio
be treated as matching the paper's estimated FLOPs.

Qualitatively, cheaper architecture choices emerge, but this is not enough. Section
7.4 p.449 also describes T-DHT eventually favoring 12 layers in supplementary plots;
our tiny tree runs favor the prefix. Missing supplementary figures and different
tasks prevent a direct trajectory reproduction claim.

## Next evidence needed

1. Investigate the early-exit/quality tradeoff with validation-only controls before
   scaling. Longer matched-budget or fixed-depth/depth-only diagnostics can separate
   insufficient training from policy behavior; none is claimed solved here.
2. Add pinned AG News preprocessing/tokenization and a documented initialization/
   train-validation-test protocol, then compare matched fixed/T-DHT backbones.
3. Optimize grouping/batching and profile controller/routing overhead if wall-clock
   savings are a requirement. Preserve the current readable implementation as the
   correctness reference.
4. Keep literal rewards as the reference setting; any feedback-direction or reward
   change must be a separate named experiment, not a silent repair to improve scores.
