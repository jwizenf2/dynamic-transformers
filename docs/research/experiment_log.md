# Experiment log

## 2026-09-22 — F-DHT full run and batched-execution benchmarks

F-DHT (D016) seed 7, three epochs, 342,000 exposures: validation macro F1 .93757/.94523/
.94010, selected epoch 2, one test evaluation .94093 (task FLOPs .432x, mean depth 5.35).
Per-sample exports of fixed/T-DHT/F-DHT reproduce each recorded test F1 exactly. D017
benchmarks on idle MPS: see `../../reports/reproduction/ag_news_fdht_and_batched_execution.md`;
curated record `../../experiments/02_related_work/ag_news_fdht_batched_2026-09-22.json`.


## 2026-09-09 — Full AG News fixed/T-DHT seed-7 reproduction

Both matched three-epoch runs completed all 342,000 exposures. Fixed validation/test
macro F1: .94801/.94903. T-DHT: .94383/.93995, selected-path train/test matmul
ratios .39781/.38576; test task+routing ratio .63576. Matched MPS throughput:
174.06 fixed / 68.13 T-DHT samples/s. Test was evaluated once per validation-selected
checkpoint. T-DHT resumed from clean step 8,000 after D015's numerical stopping fix;
the failed attempt is retained. Report: `../../reports/reproduction/ag_news_full_seed7.md`;
curated record: `../../experiments/02_related_work/ag_news_full_seed7_2026-09-09.json`.

## 2026-09-08 — Targeted long-run readiness (D013–D014)

Matched two-epoch readiness runs: 1,024 AG News training records, 512 held-out
validation records, seed 7, pinned 12-layer BERT/MPS; fixed, literal T-DHT and
separately named improvement-feedback T-DHT. Each trained 2,048 exposures exactly.
Macro F1 .922362/.870696/.872349; clipped ratios 5 literal / 7 alternative.
Main method stays literal regardless of scores. No public test or full-data run.

MPS periodic-checkpoint replay restored state exactly before computation but failed
bitwise continuation. Discrete actions/RNG/validation match in this bounded check;
floating differences are quantified, not suppressed. CPU exact replay tests pass.
Two-prefix-pass behavior was verified directly in the PDF and by CPU/MPS hooks.

Raw: `artifacts/runs/long-run-readiness-001/`; report:
`../../reports/reproduction/long_run_readiness.md`; curated configs/source/lock/data/
checkpoint provenance: `../../experiments/03_ablations/ag_news_readiness_2026-09-08.json`.
Tests: 110 passed with MPS available. This diagnostic belongs to our ablations,
not the authors' reference table or a claim of improved paper reproduction accuracy.

## 2026-09-07–08: AG News paper-backbone pilot and execution audit

Hypothesis/check: the real-data core can train and physically avoid unselected
head/layer operations; paper-equivalent quality/speed is not assumed. BERT-base
pretrained initialization and all non-paper protocol choices are D010. No reward
or exploration hyperparameter was tuned in response to validation outcomes.

Runs: fixed `artifacts/runs/ag-news-fixed-pilot-002`, tree
`artifacts/runs/ag-news-tree-pilot-003`. Prior failed launch directories are preserved;
LayerNorm naming and MPS context/backward compatibility repairs are D011/D012.
128 updates, same1024 training subset, M2Max MPS. DHT trained1018 and saved6 pending.
Unchanged checkpoints evaluated on all6000 held-out validation records; public test
untouched. MacroF1 .901090 fixed/.823288 tree, mean tree depth4.7855. Tree task-only
matrix ratio .38928 (.63928 with routing), but measured MPS batch inference2.046×
slower. No paper-equivalent numerical reproduction or speedup established.

Report: `../../reports/reproduction/ag_news_pilot.md`. Curated provenance/measurements:
`../../experiments/02_related_work/ag_news_pilot_2026-09-08.json`. Tests with MPS88passed.

## 2026-09-07 — Monitored 300-step training audit

Twelve synthetic CPU runs (fixed, tree, integrated, diagnostic fixed-prefix control;
seeds 7/11/19) completed. Finite-state/gradient and queue checks passed; all losses
decreased. DHT sacrificed quality and was slower than the full batched model in
separate inference measurements. No reward/controller changes were made to improve
these scores. Full protocol, caveats and comparison to authors' Tables 1/2:
`../../reports/reproduction/monitored_training_audit.md`. Curated measured data and
exact provenance: `../../experiments/00_sanity/monitored_training_2026-09-07.json`.

## 2026-09-07 — Independent DHT integration

Tree and integrated policies each completed 8 synthetic CPU optimizer updates.
All three tree decision levels received observed feedback; integrated pair/TS
updates also executed. Exact resumed/uninterrupted state equality is tested for
both modes with dropout. Curated configs, metrics, source hashes and checkpoints:
`../../experiments/00_sanity/dht_observed_2026-09-07.json`. These are independent
algorithm mechanics checks, not AG News results. See D005–D008 for adopted nuances.

Run records must link configuration, code revision and file hashes, dependency lock,
seed, checkpoint, dataset version and actual metrics. Raw outputs live in ignored
`artifacts/`; curated observed summaries live in `experiments/`.
Initial verification results are recorded in `../project/progress.md`.

## 2026-09-07 — Initial offline sanity

Fixed and supplied dynamic architectures each completed three CPU training steps.
Evaluation, benchmark, and interrupted/resumed CLI runs completed. Curated results,
exact configurations, dependency versions, source hashes and checkpoint locations:
`../../experiments/00_sanity/observed_2026-09-07.json`. Interpretation and limitations:
`../../experiments/00_sanity/README.md`. These are software sanity results only;
neither configuration uses learned architecture selection.
