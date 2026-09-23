# Findings

## Tree vs integrated search and batched execution (2026-09-22)

On one seed F-DHT (.94093) and T-DHT (.93995) tie on test; both trail fixed BERT (.94903).
They spend compute differently: T-DHT exits 63% of samples at 3 layers; F-DHT sends 63% to
6 layers, mostly with 8 heads. They agree on 97.7% of predictions but 50.8% of depths.
Both reverse the paper's 3-layer training trend; the paper's F-DHT < T-DHT inference-FLOP
ordering is not reproduced. Batched execution (D017) makes FLOP savings real for T-DHT
training (1.55× fixed-BERT throughput); GPU inference remains ~10–16% slower than fixed,
dominated by CPU-side context pooling.


## Full AG News fixed versus T-DHT reproduction (2026-09-09)

The seed-7 three-epoch runs completed with one documented T-DHT resume (D015).
One-time test macro F1 was .94903 fixed and .93995 T-DHT, versus paper values
.944/.945. T-DHT selected-path test matmul FLOPs were .38576x fixed and .63576x
after adding the routing prefix. It selected average depth 4.748 and preserved
99.04% of our fixed F1, a credible accuracy/computation tradeoff but not the paper's
slight accuracy gain. Actual MPS batch-8 throughput was 174.06 fixed versus 68.13
samples/s T-DHT: physical skipping does not overcome the current per-sample routing
and execution overhead. Frequent reward clipping is a future predeclared ablation,
not a post-hoc change. See `../../reports/reproduction/ag_news_full_seed7.md`.

## Readiness diagnostic (2026-09-08)

The two feedback directions produce similar subset-validation macro F1 in one
bounded run (.870696 literal / .872349 alternative); both trail fixed BERT (.922362).
This does not resolve the paper's conflicting inequality or establish a meaningful
improvement. Main feedback remains literal. Larger calibration, epoch exposure,
validation selection and recovery are now implemented, not claimed research novelty.
MPS numerical replay is qualified rather than bitwise; CPU replay is exact in tests.
The paper's two prefix passes are directly verified, including second-pass weight
updates. See `../../reports/reproduction/long_run_readiness.md` for evidence.

## Real-data pilot (2026-09-08)

Physical head/layer skipping is verified by actual matrix dimensions, layer traces,
sliced/dense output equivalence and CPU/MPS gradient tests. It does not by itself
imply runtime savings. On the matched AG News pilot, T-DHT full-validation macroF1
was .823288 versus fixed BERT .901090; task+routing matrix work was about .639×,
but measured MPS inference was2.046× slower. This is a one-seed short pilot, not a
full paper reproduction, and no public test was evaluated. See the detailed
`../../reports/reproduction/ag_news_pilot.md`; no novelty or causal diagnosis is claimed.

The paper audit identified specification defects recorded in
`../reproduction/ambiguities.md`. Independent learned tree/integrated DHT now executes
all decision levels on synthetic CPU data. Mathematical and integration tests cover
the adopted conventions. No NLP reproduction result, generalization claim, or
original research finding exists yet. Synthetic outcomes belong to the experiment
log, not the authors' results table.

The subsequent 300-step, three-seed synthetic audit found stable numerics and
decreasing losses, but early shallow routing, lower F1 than a fixed full encoder,
and slower actual per-sample DHT execution. A fixed-prefix control also underperformed
the full encoder, so this does not isolate a controller bug. This is a limitation
to investigate, not a paper reproduction or original research contribution. See
`../../reports/reproduction/monitored_training_audit.md` for evidence and scope.
