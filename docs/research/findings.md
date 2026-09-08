# Findings

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
