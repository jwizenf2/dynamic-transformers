# Project progress

## 2026-10-15 — NESTful dependency-depth result (Ouro-2.6B)

- Serving framing (afternoon): real per-token gate on xLAM (32cd8d78) gives 81.5% at 3.11
  passes and 77.5% at 2.36, beating our per-request router on paper; C2 withdrawn. Its
  per-request mean exit is nearly constant (SD 0.08) and independent of call count, so its
  saving is token-level and should not survive lockstep batching; serving bench 3ff76c7d
  queued. Landscape survey: `docs/research/landscape_2026-10-15/`.
- Evening: serving bench (3ff76c7d) shows per-token gate savings vanish in lockstep batches
  (passes run 3.22 → 3.88 from b1 to b16; no throughput gain at b ≥ 4) while per-request depth
  keeps its saving. But the router at 78% equals fixed-3; it only beats fixed mixtures at low
  budgets. NESTful gate (8989781a): 19.2% at 3.12 passes, identical depth for every dependency
  depth. Bench ran alongside other jobs: rerun on an idle GPU before quoting numbers.
- Clean rerun (d3b2bc00, alone, memory leak fixed): gate speedup over fixed-4 ×1.20 / 1.06 /
  1.03 / 1.01 at batch 1/4/8/16; per-request fixed-3 ×1.33 at every batch.

## 2026-10-16 — Router v3 (early-decoding probe) and fine-tuning pipeline

- Router v3 on xLAM val: 82.0% at 3.67 effective passes, 81.0% at 3.27, 80.5% at 3.19 (fixed-3
  78% at 3.0); ≈×1.22 batched speedup at ~81% vs ×1.01–1.06 for the per-token gate. Beats
  matched-histogram random by ~6 points. Approximate labels; real hybrid run pending.
- `scripts/lab/ouro_lora_sft.py` (MLX LoRA SFT: 2608.18171 recipe and depth-aware loss) and
  `scripts/lab/ouro_sft_eval.py` (Hermes-format eval at fixed depth). Code review and an
  end-to-end smoke test before the full runs.
- SFT timing run d72d8aa6 (max length 4096: n_train 6,378, n_val 708, 16 dropped) peaked at
  96 GB in its first step and was cancelled by the user (swap). Causes: no MLX cache limit in
  the trainer, and full-vocabulary fp32 logits for 4 loops at 4,096 tokens. Fix (cache cap,
  chunked/checkpointed loss) and a measured memory request before any full run.

- Job f19ecb14 (commit a595031): 360 depth-balanced NESTful math requests at 1–4 passes.
  Accuracy 0.3/3.1/11.7/18.9%, oracle 22.5%. Mean smallest correct passes on solved requests
  rises with depth: 2.82 (depth 2), 3.30 (3), 3.62 (4), 3.5–3.8 (5–6), vs 2.00 on xLAM (depth
  1). Depth keeps its effect when n_calls and prompt length are controlled (+0.46 per SD,
  SE 0.19); caveats (collinearity 0.79, survivorship, low accuracy) in the findings doc §7.
- Data source fix: lab jobs reach Hugging Face but not raw GitHub, so NESTful now loads from
  the pinned HF parquet (math subset verified identical) with `basic_functions.py` and the ICL
  examples vendored in `third_party/nestful` (Apache-2.0). `scripts/lab/depth_analysis.py`
  added. Queued: 2.6B NESTful 360–960 (8cd5243f), Ouro-1.4B xLAM val (ccc11c05) and NESTful
  (37599673), 2.6B NESTful at 5–6 passes (cc487f32).
- Ouro-1.4B xLAM val (ccc11c05): 18/43.5/58/66%, oracle 70.5%. At matched compute (1.4B at
  2k passes = 2.6B at k) 1.4B wins: 43.5 vs 26% (48 layers), 66 vs 61% (96 layers, n.s.).
- 05:41: memory-broker preemption killed 8cd5243f (218/600) and 37599673 (283/360); the lab
  refused to resume them. Partial rows analysed (findings §7); pooled 2.6B (570) keeps the
  depth effect (+0.29 SD, SE 0.15). Not resubmitted: lab automation was paused. The probe
  needs a SIGUSR1 handler / `--start` resume to survive preemption.
- Passes 5–6 (cc487f32): 22.8% at 5 vs 18.9% at 4 (+37/−23 requests); gains concentrate at
  depth ≥ 3 (depth 6: 6.7 → 15.0%); pass 6 is worse than 5 everywhere. Oracle over 1–6 passes
  33.6%. Shallow requests degrade past 4 (xLAM), deep chains improve: per-request control.

## 2026-10-14 — Ouro loop-count routing: router analysis v2 and NESTful adapter

- Direction and gate results: `docs/research/findings_2026-10-14_routing_and_looped_models.md`
  (section 7). Router analysis v2 (job 13176da7, commit 894161f, `router_analysis_v2.py`):
  CV picked PCA 8, l2 10; query-text features did not help. Passes needed to stay within
  1 point of fixed-4 (82%) on 200 xLAM val: gate 3.78, confidence 3.72, learned one-shot 3.22
  (37% of the oracle saving; bootstrap median 3.27, 5–95% [2.78, 4.00]), sequential
  threshold 3.58, sequential gain 3.65. LinUCB never reaches 81% in any of 60 seed/setting
  runs (best 80.0% at 3.0 passes); v1's "3.35 passes" for LinUCB is withdrawn.
- `src/dht/toolcall/nestful.py`: NESTful loader, dependency depth, executor (math subset
  only, range-checked), NESTful answer matching, exact sequence match. The executor
  reproduces 1,415/1,416 math-subset gold answers locally (1,416/1,416 after raising the
  argument limit for 3**50). `scripts/lab/ouro_nestful.py` runs Ouro at 1–4 passes on a
  depth-balanced order (~60 per depth 2..6, 7+). 184 tests pass.

## 2026-10-13 — Tool-calling routing: lab setup and Phase 0 evaluation code

- New direction (plan: `docs/research/plan_tool_calling_routing.md`, D019): frozen
  Qwen3.5/Qwen3.8 hybrid DeltaNet/attention models, per-request layer-group skipping,
  BFCL-graded; UCT as offline teacher. Model ladder Qwen3.5-0.8B/4B/9B/27B + Qwen3.8-27B.
- ai-lab (Mac Studio, M5 Max 128 GB) jobs run from `wizenfeld-lab/dynamic-transformers`.
  `scripts/lab/setup_check.py` downloaded xLAM-60k (rev 26d14eb), BFCL v3 (rev 61fc060)
  and the models with pinned revisions; MLX produced a correct Qwen tool call (0.8B).
- `src/dht/toolcall/`: Qwen `qwen3_coder` call parser, independent BFCL-style AST checker
  (not verified against `bfcl-eval`), BFCL/xLAM loaders with a seeded split, and physical
  layer skipping for mlx-lm `qwen3_5` models. `scripts/lab/headroom.py` is the probe.
- mlx-lm 0.31.3 corrupted batched generation for left-padded prompts (caught by
  `test_mlx_routing.py`); pinned 0.32.0. The first headroom run (0.8B, "70% full") used
  the buggy batching and is DISCARDED. No headroom result exists yet.
- Code review fixes: xLAM nested/typed ground truth, greedy termination, pinned model
  revisions, exact generated-token counts. 167 tests pass.

## 2026-09-22 — Dataset EDA, F-DHT (integrated search), batched execution, next topics

- User-approved downloads of DBpedia / Yelp Polarity / Yelp Full (Zhang et al. CSVs,
  hashes in the curated record) for EDA only; sizes match the paper exactly.
- `notebooks/01_dataset_eda_and_dht_examples.ipynb`: four-dataset EDA (balance, lengths
  vs 128/300 limits, vocabulary, quality, Naive Bayes difficulty) and fixed BERT vs T-DHT
  per-sample analysis, including Part 3 "one sample at a time" anatomy cards.
- D016: full-data F-DHT run completed (342,000 samples, per-sample engine). Selected
  epoch 2; test macro F1 .94093 (T-DHT .93995, fixed .94903); task FLOPs .432x.
- D017: batched execution engine (opt-in) plus `DHTController.explain`,
  `fit_logistic_map_many`, `scripts/{export_predictions,benchmark_inference,benchmark_training}.py`.
  MPS: batched T-DHT trains 1.55× faster than fixed BERT (was 0.61×); inference
  0.84–0.90× fixed at batch 32 (was 0.20–0.38×); decisions identical to per-sample.
- `notebooks/02_tree_vs_integrated_search.ipynb`: concepts, code diff, results, per-sample
  traces, head usage, dynamics and speed. `docs/research/next_topics.md`: verified list.
- Verification: 126 tests pass; ruff clean. Report:
  `../../reports/reproduction/ag_news_fdht_and_batched_execution.md`.
- D018 device context pooling (opt-in): batched T-DHT inference 1.12× and training 1.60×
  fixed BERT; F-DHT 1.11× / 0.96×. 6,000/6,000 validation decisions unchanged. 130 tests.
- Next: ≥3 seeds per mode, S-DHT, a cheaper head-sliced backward, a full batched retrain.


## 2026-09-09 — Full-run monitoring and T-DHT numerical recovery

- Fixed AG News seed-7 completed all three epochs and 342,000 sample exposures.
  Validation macro F1 progressed `.94444`, `.94604`, `.94801`; public test remains
  untouched. The final epoch is the validation-selected best checkpoint.
- The initial paper-literal T-DHT attempt stopped in epoch 1 at optimizer step 8,721
  after 69,768 trained samples: a logistic MAP fit reached gradient `1.0217e-6`
  after 30 iterations and narrowly missed its absolute convergence cutoff. Its
  atomic step-8,000 checkpoint and failed logs are retained.
- Added D015's relative Newton-step convergence criterion and a high-precision
  regression. This changes only numerical stopping, not the posterior objective,
  actions, reward formulation, or paper-facing configuration.
- Focused bandit verification: `17 passed`. Full verification and resumed T-DHT
  completion subsequently passed 111 MPS tests. The resumed run completed all
  three epochs; validation/test macro F1 was `.94383`/`.93995`, versus fixed
  `.94801`/`.94903`. Each selected checkpoint was evaluated on public test exactly
  once. T-DHT test task-path matmul ratio was `.38576x` (`.63576x` with routing),
  but matched MPS throughput was 68.13 versus fixed 174.06 samples/s. Full report:
  `../../reports/reproduction/ag_news_full_seed7.md`.

## Latest session: 2026-09-08 — Long-run readiness and two-pass verification

Objective: implement the three approved readiness fixes and explicitly verify the
paper's no-update routing prefix / trainable second prefix pass. Ready for the
initial matched long runs with documented MPS numerical-replay limitations; no
full-data training, new download, public-test evaluation or cloud use occurred.

Completed: typed epoch/schedule configuration; seeded balanced training-only
calibration and saved selection metadata; shared epoch runner with complete sample
exposure, queue-tail flush, sample-based warmup/decay, epoch validation and best
selection; atomic periodic/latest and immutable best checkpoints (schema 3);
streamed JSONL/CSV. Added two full-run and three bounded-readiness configs, scripts
for readiness and read-only replay inspection, and calibration/epoch/config tests.
New prefix tests instrument real CPU/MPS routing, queues, forwards and optimizer
updates, with b=c=3, tree/integrated and selected depths 3/6. Controller docstrings
and a dedicated paper-to-code guide explain the mechanisms and dropout nuance.

Actual experiments: fixed/literal/improvement, each 2 epochs over 1,024 training
records, 512 validation records, same pinned BERT-base/AG News and seed 7. Larger
DHT calibration uses 512 distinct training records, 128/class. Updates 256/261/259;
all trained 2,048 examples and flushed queues. Selected validation macro F1:
.922362/.870696/.872349. Literal/alternative clipped rewards: 5/7 at existing cap
100; no reward hyperparameter was changed. Main feedback remains paper_literal.
Full-data read-only preflight loaded 114,000 records and selected 128/class.

The MPS step-64-to-261 replay initially FAILED bitwise weight equality; that result
is preserved. Follow-up confirms exact model/optimizer/controller restoration
before computation, all 197 replayed discrete sample/head choices and RNG states,
epoch validation scores and best-step choice. Maximum final weight difference
2.0741e-5; 32-example validation-probe logit difference 1.4305e-6, identical labels
and architectures. Continuous UCB statistics also differ. Cause is not proven;
MPS is operationally resumable, not certified bitwise reproducible. CPU exact
fixed/tree/integrated recovery and simulated best/latest crash tests pass.

Verification actually run: CPU 104 passed / 6 MPS skips (6.51s); full MPS suite
110 passed (10.97s). All eight new two-pass tests pass with MPS. Ruff format/lint
pass (final formatting check: 79 Python files); `git diff --check` passed. Both
prefix passes execute; first has no graph or updates,
second gives every prefix layer task gradients and updates. Skipped layers stay
unexecuted/unchanged. PDF §§4.2.1–4.2.2 p.443, §5/Algorithm2 p.444, Fig.1 p.445
directly support this mechanism; permanently freezing the prefix would be wrong.

Changed files: config/CLI, calibration/features/trainer/checkpointing, new epochs
module and five TOMLs; readiness/inspection scripts; calibration/epochs/config/
prefix tests; README/AGENTS/walkthrough, D013–D014 and reproduction docs; new
readiness report and curated experiment JSON. Earlier research artifacts preserved.

Raw outputs: `artifacts/runs/long-run-readiness-001/`. Detailed outcome:
`../../reports/reproduction/long_run_readiness.md`; curated config/source/lock/data/
checkpoint hashes: `../../experiments/03_ablations/ag_news_readiness_2026-09-08.json`.
Results are subset validation, NOT reproduced paper test metrics. Remaining gaps:
unknown author initialization/settings, TS/context/reward ambiguities, frozen
calibration approximation, MPS bitwise continuation, runtime/complete FLOP accounting.

Next concrete step: after runtime approval, run `ag_news_fixed_full.toml`, then
`ag_news_tree_full.toml` sequentially (3 epochs, 114,000 train / 6,000 validation;
342,000 exposures each). Follow `docs/reproduction/long_runs.md`, monitor logs and
selection/clipping, select validation-best, and evaluate public test only later.
No speculative methodology changes are needed first. Earlier entries are historical.

## Latest session: 2026-09-07–08 — Real AG News pilots and verified physical skipping

Objective: compare against a paper baseline, run a real-data experiment, and verify
that selected-out heads/layers avoid computation. Completed bounded pretrained
12-layer BERT and T-DHT pilots; full multi-epoch reproduction remains outstanding.

Implemented pinned AG News CSV/WordPiece preparation (`ag_news.py`, preparation
script), hash-validated offline loading, optional tokenizers/safetensors dependency
extra and lock update, strict local BERT encoder import, matched paper-backbone
pilot TOMLs, validation-only pilot control and flushed per-step JSONL in both
training paths. Prepared 114000/6000/7600 train/validation/test arrays. Downloads of
~31MB data/tokenizer and ~440MB weights were explicitly approved; no cloud use.

Added CPU operator profiler and layer traces (`evaluation/execution.py`, verification
script), data/import/gradient tests, a read-only CPU/MPS diagnostic, full-validation
evaluation and post-run collector. Heavy mathematical/implementation docstrings,
AG News protocol and physical-execution guide explain every choice and remaining
cost. README, AGENTS, walkthrough, mapping and ambiguity/decision logs updated.

Detected and repaired two genuine MPS integration issues without changing bandit
math: combined CPU/float64 transfer corrupted finite contexts (D011), and general
index-select backward accumulation lacked strict deterministic MPS support (D012).
Transfer first then cast on CPU; use a mathematically equivalent unique-channel
gather derivative that assigns rather than accumulates. Also handled the downloaded
weights' legacy LayerNorm gamma/beta names explicitly. Failed launches are preserved
and described in the report; no failure is counted as completed training.

Successful runs: `artifacts/runs/ag-news-fixed-pilot-002/` and
`artifacts/runs/ag-news-tree-pilot-003/`, 128 updates each, seed7, Apple M2 Max 32GiB
MPS with strict determinism. Same 1024-example training subset and initialization;
DHT trained1018 with6 pending, fixed1024. All depth/count actions were exercised.
Both losses declined; finite saved weights/optimizer/controller, converged TS fits,
positive precision, zero clipped rewards, valid architectures and queue conservation
verified. Public test NOT evaluated. Full6000-validation macroF1: fixed .901090,
T-DHT .823288. The earlier128-validation scores (.941069/.843610) were optimistic.

Full-validation T-DHT mean task depth4.7855, task matrix ratio .38928, task+routing
ratio .63928. Paper-sized forced9-layer/8-head profile: 15,451,822,080 matrix FLOPs
versus22,347,257,856 full,16,760,444,928 same-depth dense-zero-mask. Exact sliced/
dense-reference outputs in the recorded check. Layer/operator dimensions and learned
policy traces verify real skipping. Tests verify inactive and skipped gradients.
Dense gradient buffers/AdamW/FFNs/routing still cost work; no total-training savings.
An actual frozen-policy trace at validation index4 selected depth6 and excluded
heads2/7 in zero-based layers3–5: QKV width640, visited layers
`[0,1,2,0,1,2,3,4,5]`, no execution of layers6–11. Both prefix-only and pruned-head
traces are preserved in the curated record/raw outputs.

Sequential MPS inference benchmark (batch8,length128,5warmups,20repeats): fixed55.688ms,
tree113.912ms, thus tree2.046× slower. Training segments34.428s/33.463s, with another
1.417s T-DHT calibration. These snapshots do not establish a training speedup.

Verification: final CPU suite86 passed,2 MPS skips (1.86s); full suite with MPS
available **88 passed (2.23s)**. Ruff formatting/lint passed. Full detailed report:
`../../reports/reproduction/ag_news_pilot.md`; curated results/configs/source/data/
checkpoint hashes: `../../experiments/02_related_work/ag_news_pilot_2026-09-08.json`.

Unresolved: paper initialization/classifier/reward/context gaps remain; our pilot
does not match near-baseline T-DHT quality or demonstrate runtime savings. Next:
add epoch/sample-exposure budgeting, periodic resumable checkpoints and scheduled
validation-based selection; predeclare full-data optimizer/calibration protocol and
multi-hour runtime budget; then run matched fixed/T-DHT and final public test once.
No full-scale run or speculative research modification has been launched.

Earlier entries below are historical.

## Latest session: 2026-09-07 — Monitored training and paper comparison

Objective: run the code, watch logs, assess behavior and compare honestly with the
paper. Completed a bounded synthetic audit; did not launch full-scale NLP training.

Added `scripts/audit_training.py` with live flushed JSON logs and read-only observation
hooks, plus `configs/experiment/audit_{fixed,tree,integrated,prefix}.toml`. Core DHT
algorithms/reward conventions were unchanged. Ran seeds 7/11/19, 300 steps each for
fixed/tree/integrated and a later fixed-one-layer diagnostic control: 12 runs,
3600 optimizer steps. All completed. Every step checked finite losses, task gradients,
weights, DHT statistics and positive TS precision; DHT checks also confirmed converged
fits and exact routed=trained+pending sample conservation. No reward clipping occurred.

Every run reduced its first-to-last 25-step mean loss, but DHT underperformed the
full fixed baseline. Test macro F1 mean±sample SD: fixed .699±.131, prefix-only
.579±.094, tree .524±.162, integrated .511±.033. Tree rapidly favored the prefix
(test mean depth 1.005); integrated test mean depth 1.565. The prefix control suggests
shallow-path capacity/budget explains part, not all, of the difference. This is a
learning concern, not proof of a numerical implementation bug or a novel finding.

Separate sequential seed-7 CPU benchmarks used batch 8, length 8, 5 warmups and
50 repeats. Mean batch latency: full fixed .578 ms, tree 5.708 ms, integrated
6.284 ms. DHT is roughly 9.9×/10.9× slower in this current per-sample implementation.
Routing/calibration costs remain separately visible; task-only matmul estimates
must not be presented as total savings. Instrumented training times contain audit
overhead and are not benchmarks.

Verification: `uv run pytest` passed **74 tests in 2.37 s**; audit harness formatting/
lint and `git diff --check` passed. Logs, metrics and checkpoints:
`artifacts/runs/monitored-audit-001/`, `artifacts/runs/monitored-prefix-control-001/`.
Curated run/source/config provenance:
`../../experiments/00_sanity/monitored_training_2026-09-07.json`.
Detailed outcome and authors' AG News comparison:
`../../reports/reproduction/monitored_training_audit.md`.

The paper's near-baseline F1 plus lower estimated FLOPs is NOT reproduced by this
audit. Our synthetic data, small model and explicit F1 convention prevent a numerical
score-gap comparison with the authors' AG News .945 T-DHT result. Next: investigate
early routing/quality with validation-only controls, add real AG News preprocessing
and a matched protocol, then address batching/overhead before claiming speedups.
All earlier entries below are historical.

## Latest session: 2026-09-07 — Independent DHT and reader documentation

### Current objective and outcome

The user prioritized a reliable independent code base and heavily documented,
readable implementation over perfect matching of an incomplete specification.
Implemented and verified both tree and integrated DHT end to end on synthetic
CPU data, with explicit adopted interpretations. Real NLP experiments remain next.

### Completed work / changed files

- Added `src/dht/bandits/{linear_ucb,logistic_ts,features}.py`: per-arm ridge UCB,
  Gaussian logistic TS, stable MAP fitting, train-only arm calibration, explicit
  pooling dimensions and frozen normalization. Math, shapes, and nuances documented.
- Added `src/dht/policies/{controller,rewards}.py`: tree/integrated route/execute/update,
  original decision-time contexts, loss-cost ratios, literal/corrected binary feedback.
- Added `src/dht/training/{queues,trainer}.py`: depth FIFOs, partial-tail flush,
  per-sample heads within one mean-loss optimizer step, observed-only policy updates,
  full state resumption, architecture distributions and separate cost/timing scopes.
- Extended model primitives for intermediate decisions, config validation, checkpoint
  schema 2, and all CLI commands. Existing fixed-path workflows remain available.
- Added small tree/integrated TOML configs and an unrun paper-backbone development
  template. No dataset/model download or large training was launched.
- Added mathematical/feature/reward/queue tests and full DHT integration tests.
- Added `docs/code_walkthrough.md`, expanded code docstrings/comments, and updated
  README, AGENTS, algorithm mapping, protocol, decisions D005–D008, ambiguity status,
  presentation and research notes. No ambiguity is claimed to be resolved by authors.

### Verification actually observed

- Existing 45 tests passed after integration. Expanded suite: 72 passed, then
  **74 passed in 2.51 s** after multi-layer group checks were added.
- Ruff formatting/lint passed; `git diff --check` passed.
- Tests verify direct ridge calculations, confidence exploration and contextual
  learning; logistic objective/gradient/MAP, both binary labels, Gaussian moments,
  selected-only precision updates and RNG round trips; pooling/padding; bootstrap,
  reward caps and threshold direction; queue ordering and epoch tails.
- Tree and integrated interrupted/resumed CPU training match uninterrupted weights,
  optimizer, bandits, calibration, reward history and queue state exactly with dropout.
- Tests cover frozen inference, no-gradient routing, no evaluation calibration,
  heterogeneous heads in one depth batch, and shared heads across c-layer groups.
- CLI tree and integrated smoke runs each completed 8 optimizer updates. Tree
  evaluation and benchmark commands loaded the full policy checkpoint successfully.

### Observed results and provenance

Raw runs: `artifacts/runs/tree-dht-v1/` and `artifacts/runs/integrated-dht-v1/`.
Curated records including exact run source/config/lock hashes:
`../../experiments/00_sanity/dht_observed_2026-09-07.json`.

Tree trained 22 samples in 8 queue releases, choosing depths 1/2/3 for 13/3/6
samples; 2 other routed samples remained queued and checkpointed. Head-count UCB
received 9 observations, and TS received selected-head feedback. Integrated trained
24 samples, choosing depths 1/2/3 for 9/6/9 samples, and updated its pair UCB/TS.
Both synthetic validation accuracies were 0.416667 after only eight updates.
These are mechanics checks, not evidence of NLP quality or reproduced paper scores.

Task-forward matmul ratios were 0.5323 (tree) and 0.6332 (integrated), excluding
separately recorded routing/calibration and unestimated backward/controller costs.
No computation-saving conclusion follows from these short runs. Tree CLI benchmark
included routing/policy and measured about 2.176 ms per three-sample CPU batch;
this is not a BERT-scale speedup comparison. Accelerators remain untested.

### Remaining work / next step

No code blocker for the documented synthetic baseline. Add pinned real-corpus
preprocessing/tokenization and an AG News train/validation/test protocol, choose
random/pretrained initialization, then measure matched fixed/T-DHT runs within a
profiled hardware budget. Published S/F experiments, P-DHT freeze behavior,
efficient batched execution and detailed memory/profiler accounting remain future
work. All changes remain uncommitted; no speculative extension was implemented.

The earlier entry below is historical, including its then-pending milestones.

## Session: 2026-09-07 — Phase 0/1 foundation

### Current objective

Establish a maintainable independent DHT research repository, audit the supplied
paper, and validate a small deterministic Transformer before bandit integration.
The immediate first-execution assignment is complete. Related Work Implementation
as an academic deliverable still requires learned bandits and a measured comparison.

### Completed work and files changed

- Inspected the initial repository (README, Python ignore template, supplied PDF)
  and Apple Silicon environment. Kept the original PDF in place and recorded SHA256.
- Read the complete 14-page PDF text, including references and final header-only
  page; inspected rendered pages 3–11 for mathematics, algorithms, tables and figures.
- Added detailed `docs/reproduction/{paper_summary,notation,algorithm_mapping,
  ambiguities,protocol,bandit_implementation_plan}.md`; logged 22 reproducibility
  issues with alternatives. Adopted total-depth boundary before scaffold coding.
- Preserved all 64 Table 1/2 records in `references/paper_results.csv`, exact variant
  settings including P-DHT in `references/paper_settings.json`, and bibliography.
- Added `AGENTS.md`, professional README, roadmap/decisions, research question/log/
  findings documents, and a checkpoint presentation outline. No novelty claim made.
- Added `pyproject.toml`, `uv.lock`, Python 3.12.12 pin and research ignore rules.
  Dependencies installed into local `.venv`; a network escalation was approved for
  dependency installation. No dataset/pretrained model/cloud resource was acquired.
- Implemented `src/dht/config.py`, architecture/attention/encoder modules, synthetic
  data, runtime provenance/device/RNG support, trusted checkpointing, F1 metrics,
  partial matmul accounting, and train/evaluate/benchmark CLI.
- Added standalone fixed/dynamic CPU TOML configurations and unit/integration/
  regression tests. Added curated observed sanity results in `experiments/00_sanity/`.

### Verification actually run

- `uv sync --locked`: passed (42 resolved packages; 22 installed/audited on this host).
- `uv run ruff format .`: applied formatting; final `--check` reported 38 files
  already formatted. `uv run ruff check .`: all checks passed.
- `uv run pytest`: initial 44 tests passed; after adding independent PyTorch encoder
  comparison, final **45 passed in 1.27 seconds**.
- `git diff --check`: passed.
- README CLI fixed and dynamic training: both completed three CPU optimizer steps.
- CLI evaluation: matched saved validation metrics.
- CLI partial training and resumption: completed steps 1 then 2–3. Integration test
  additionally proved exact CPU model/optimizer equality versus uninterrupted
  training with dropout, and verified all persisted RNG streams.
- CLI benchmark: final separate timing check completed 3 warmups and 10 measured
  batches; raw result `artifacts/runs/dynamic-smoke/benchmark-isolated.json`.

### Results actually observed

Python 3.12.12, PyTorch 2.14.0, NumPy 2.5.3, Pydantic 2.13.5, pytest 9.1.1,
Ruff 0.16.6; arm64 macOS, CPU. MPS reported unavailable to this process; CUDA
unavailable. Hardware-specific GPU execution is untested.

Both tiny paths produced validation accuracy 0.416667 and macro F1 0.196078;
test accuracy 0.166667 and macro F1 0.095238. These are sanity outcomes after
three steps, not a reproduction, performance target, or evidence of equal quality.
Fixed/dynamic dominant forward matmul counts were 239760/138384 per sample.
The dynamic-only timing check averaged 0.355817 ms per four-sample batch; it does
not establish a speedup. Full configs, hashes and observed measurements are in
`../../experiments/00_sanity/observed_2026-09-07.json`.

### Open issues and scope limits

No UCB, logistic TS, reward policy, feature calibration/normalization, depth queues,
sample-specific learned routing, real dataset loader or pretrained BERT exists yet.
Bandit/queue/history checkpoint state is consequently deferred. Current checkpoints
cover the deterministic scaffold only. Full DHT cost, routing/controller overhead,
profiler counts, memory and repeated-seed NLP comparisons remain future work.

Research fidelity remains unresolved for TS initialization/objective/precision,
feedback direction, head/pair cost products, feature dimensions, calibration,
head-combination restrictions, P-DHT and the training/evaluation protocol. These
do not block generic UCB. No author code or supplement was verified; a limited
publisher metadata check is logged with URLs. No author code was imported.

All repository changes remain uncommitted; the supplied PDF was already untracked.
No existing user file was removed and no full-scale training was launched.

### Exact next steps

1. Implement `src/dht/bandits/linear_ucb.py` and its tiny direct-ridge tests using
   actual aggregate width p, per-arm λI statistics, solve-based estimates, confidence
   bonus and observed-arm-only updates. No pooling choice is required yet.
2. Review the concrete proposed diagonal logistic-Laplace repair in
   `../reproduction/bandit_implementation_plan.md` (A03–A05) before implementing TSB.
   Recommend positive precision, stable penalized binary cross entropy and selected
   subset feedback only; preserve the printed binary threshold for a literal
   reproduction configuration and name the opposite direction as a correction.
3. Resolve reward/context conventions before integrating tree routing and queues;
   then validate tiny DHT end to end before defining the AG News comparison budget.
