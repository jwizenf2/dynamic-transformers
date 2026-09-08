# Initial CPU software sanity — 2026-09-07

Later in the session, learned tree/integrated DHT was added and each completed eight
optimizer updates. See `dht_observed_2026-09-07.json` for separate measured results
and exact code/configuration hashes. The expanded suite passed 74 tests. The original
fixed/supplied-path observations below remain preserved as historical evidence.

These are observed synthetic checks, not paper reproduction results. The curated
`observed_2026-09-07.json` contains configurations, dependency versions, source/lock
hashes, checkpoint locations and actual metrics. Raw runs remain ignored under
`artifacts/runs/`. Code was uncommitted on top of revision
`12b88cdb467e9be775902cd736d95fc206d5371c`; hashes identify the implementation.

Both configurations used seed 7, width 24, 4 available heads, CPU with one PyTorch
thread, and three AdamW steps on 24 synthetic training examples. Validation and
test contain 12 examples each. The fixed path executed three layers; the supplied
dynamic path executed two layers, with heads (1,3) in its second layer.

| Observed value | Fixed path | Supplied dynamic path |
| --- | ---: | ---: |
| Validation accuracy / micro F1 | 0.416667 | 0.416667 |
| Validation macro F1 | 0.196078 | 0.196078 |
| Test accuracy / micro F1 | 0.166667 | 0.166667 |
| Forward matmul FLOPs/sample | 239760 | 138384 |
| Relative forward matmul FLOPs | 1.000000 | 0.577177 |

Three steps do not establish task learning or accuracy preservation. Ratios count
only the explicitly documented forward matrix products, not total training cost.

The dynamic path's separate CPU timing check used three warmup batches and ten
measured batches (batch size 4, sequence length 8), with mean 0.355817 ms/batch
and approximately 11241.74 samples/s. No other task command was run concurrently
with this final timing check. It is a tiny forward-only software check; there is
no paired fixed-path latency measurement, no speedup claim, no controller/routing,
and no peak-memory measurement. A preliminary timing run overlapped other CLI
checks and is retained only in raw artifacts, not used in this curated result.

Verification: 45 tests passed in 1.27 s; Ruff lint and format check passed; locked
dependency sync and `git diff --check` passed. Tests cover comparison with PyTorch's
standard encoder layer, dense mask equivalence, projection positions and actual
selected widths, prefix/group constraints, gradients, and exact CPU interrupted
resume with dropout. The train/evaluate/benchmark and partial/resume commands in
the root README were executed successfully. MPS/CUDA were unavailable to this
process, so accelerator execution has not been validated.
