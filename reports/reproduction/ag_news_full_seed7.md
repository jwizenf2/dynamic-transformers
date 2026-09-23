# AG News fixed BERT versus T-DHT, seed 7

Date: 2026-09-09. These are our measured results, not the authors' table. Both
models used the pinned BERT-base encoder and AG News WordPiece corpus, seed 7,
batch size 8, and three complete 114,000-sample epochs. Validation selected the
checkpoint before one public-test evaluation. See D010, D013, and D015.

## Main result

| Model | Our validation macro F1 | Our test macro F1 | Paper test F1 | Task-path FLOPs | Paper ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed BERT | .94801 | .94903 | .944 | 1.000x | 1.00x |
| T-DHT | .94383 | .93995 | .945 | .39781x train / .38576x test | .42x train / .81x inference |

Our T-DHT is 0.00908 absolute test macro F1 below our fixed baseline and 0.00505
below the paper's T-DHT value. It preserves 99.04% of our fixed F1 while its
selected task path uses 38.58% of fixed-model test matmul FLOPs. It reproduces the
accuracy/computation tradeoff, but not the paper's small T-DHT accuracy improvement.
The paper does not provide enough protocol detail to expect exact numerical equality.

The task-path ratio excludes DHT's first routing pass. Adding its full-head three-
layer prefix makes the test neural-forward ratio about `.63576x`; controller math
is still excluded. Training task paths use `.39781x`; adding per-sample routing and
once-only calibration gives about `.65135x` forward matmul work. These scopes are
not necessarily the authors' accounting, so the apparent closeness of `.39781x`
to their `.42x` must not be overstated.

## Architecture behavior

On 7,600 test samples, selected depths were 3: 4,759 (62.62%), 6: 1,514
(19.92%), 9: 1,066 (14.03%), and 12: 261 (3.43%). Average total depth was 4.748.
Across dynamic-group decisions, head counts were 8: 1,579 (35.65%), 10: 1,995
(45.04%), and 12: 855 (19.30%). Average heads over all executed layers was 11.143
because the three prefix layers are always full-head. Every published tree action
was exercised; the policy did not collapse to one architecture. It exits much more
aggressively than the paper's `.81x` T-DHT inference ratio suggests.

Validation macro F1 increased each epoch: `.93733`, `.94101`, `.94383`. Fixed BERT
also increased: `.94444`, `.94604`, `.94801`. Both final epochs were selected.
Each run trained exactly 342,000 samples; T-DHT routed exactly 342,000 and ended
with all four queues empty.

## Runtime and limitations

Fixed training-step time was 6,765.09 seconds; T-DHT was 10,669.60 seconds, or
1.58x slower, excluding validation/checkpoint I/O and with calibration recorded
separately. A matched MPS batch-8 benchmark measured 174.06 samples/s fixed versus
68.13 T-DHT, making the readable per-sample DHT path 2.55x slower. Physical matrix
skipping is real, but heterogeneous per-sample execution, two prefix passes,
transfers, and controller overhead prevent a wall-clock speedup. Grouping identical
architectures is a future execution optimization, not part of this reproduction.

A post-training semantic audit used depth 9 with eight heads in both dynamic groups:
observed matrix FLOPs exactly matched its same-scope analytic count, and sliced and
dense-mask reference logits had maximum difference 0.0. A separate learned-controller
trace selected depth 6 and eight heads. It called the three full-head routing-prefix
layers, reran those prefix layers for the task, called task layers 3–5 with eight-head
projection shapes, and never called layers 6–11. These single-sample audits are
structural evidence, not the aggregate test FLOP ratio. See `execution.json`.

The first T-DHT attempt stopped at step 8,721 when an online logistic MAP fit was
numerically stationary but narrowly missed its absolute-gradient cutoff. D015 adds
a relative Newton-step stopping criterion without changing the objective or policy.
The run resumed from its clean atomic step-8,000 checkpoint. MPS restoration is
exact before computation but continued execution is not promised bitwise identical
(D014). The failed run remains preserved. The final code passed 111 tests on MPS.

Reward ratios hit the configured cap 81,574 times across all route/group events.
This is finite and expected with very small cross-entropy denominators, but frequent
clipping may influence learning and deserves a predeclared ablation later. It is
not silently tuned after seeing this result. One seed establishes feasibility, not
variance or statistical significance.

## Artifacts

- Fixed run: `artifacts/runs/ag-news-fixed-full-seed7/`
- T-DHT failed attempt: `artifacts/runs/ag-news-tree-full-seed7/`
- T-DHT clean resume/final: `artifacts/runs/ag-news-tree-full-seed7-resume-d015/`
- Test outputs: each final run's `final_test.json`
- Matched latency: each final run's `benchmark.json`
- Curated record: `experiments/02_related_work/ag_news_full_seed7_2026-09-09.json`

Raw artifact directories are Git-ignored. Checkpoints are trusted local pickle-based
files and must not be loaded from an untrusted source.
