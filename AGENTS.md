# Working in this research repository

Current phase: independent UCB/logistic TS and tree/integrated DHT, with synthetic
tests and pinned pretrained BERT/AG News pilots. Full paper reproduction is unverified.
Read `docs/reproduction/ag_news_protocol.md` before launching real-data training.
For the epoch-based main runs, also read `docs/reproduction/long_runs.md` (D013).

- Read `docs/project/progress.md` and `docs/project/decisions.md` before changing code.
- Source of truth: `dynamic_transformers_reference_paper.pdf` (printed pp. 439–452).
  Read relevant rendered equations as well as extracted text. Research notes live in
  `docs/reproduction/`; the algorithm mapping distinguishes implemented and planned work.
- Never silently resolve paper ambiguities. Record section/page/equation, alternatives,
  chosen interpretation, and fidelity classification in the ambiguity/decision logs.
- Use Python 3.12, uv, PyTorch, typed TOML configuration, and the `src/dht` package.
  Keep reusable algorithms separate from experiment settings. No author code was imported.
- Layer indices are zero based; `total_depth` includes the unpruned prefix. Preserve
  original head positions in output projection. Masking alone is not compute avoidance.
- Run `uv sync --locked`, `uv run ruff format --check .`, `uv run ruff check .`, and
  `uv run pytest`. CPU sanity: `uv run dht train --config configs/experiment/smoke.toml`.
- Tests must not download datasets or pretrained weights. Do not launch large training,
  paid resources, or speculative extensions without explicit authorization.
- Keep authors' reference results separate from measured results. Never fabricate metrics,
  claim tests ran when they did not, or equate FLOP estimates with latency.
- Preserve user changes. Update progress, decisions, and experiment provenance after
  substantial work. Ignore raw datasets, caches, checkpoints, and generated artifacts.
- Checkpoints are trusted local files; never load untrusted pickle-based checkpoints.
- The user prioritizes a reliable, readable foundation over exact numerical matching.
  Make reasonable documented interpretations without repeatedly requesting approval.
  See D005–D008 for adopted bandit/reward/context conventions; source gaps stay open.
- Heavily document mathematical code: shapes, paper section/page, update order,
  invariants and deviations. Update `docs/code_walkthrough.md` when workflows change.
- DHT sanity: `uv run dht train --config configs/experiment/dht_tree_smoke.toml`.
  `[dht]` enables learned routing; fixed architectures cannot also be supplied.
- Real data needs `uv sync --locked --extra nlp`; preparation/download scripts are
  explicit opt-ins. Config pins the data manifest and pretrained file SHA256.
  `evaluate_test=false` keeps pilots off the public test; do not bypass it for tuning.
- Physical execution: `scripts/verify_execution.py` plus `test_execution.py` verify
  actual CPU mm/addmm/bmm shapes and layer calls. Forward savings are NOT total
  training savings: dense gradients/AdamW, FFNs, routing and CPU controllers remain.
- MPS contexts: transfer to CPU in original dtype BEFORE converting to float64.
  D011 records the observed combined-transfer failure. Never replace NaNs to hide it.
- `[training]` opts into epoch budgets, trained-sample LR scheduling, epoch-end
  validation selection and schema3 periodic checkpoints. Do not extend the old
  pilot steps setting and call it a full experiment. Main calibration is 512 balanced
  training-only records; its selected IDs are saved and never refit on resume.
- `checkpoint.pt` is latest; `best.pt` is validation-selected. Keep immutable
  best-step candidate files with the run for crash recovery; they are hard-linked,
  not duplicate weights. Arbitrary interruption recovers the last atomic checkpoint.
- Main T-DHT feedback remains paper_literal. The improvement readiness configuration
  is an explicitly separate diagnostic, not a replacement chosen after seeing scores.
- Readiness: `reports/reproduction/long_run_readiness.md`; 110 tests passed with
  MPS. No full-data long runs launched yet. CPU replay is exact; MPS restoration
  is exact before computation but continued training is NOT bitwise identical.
  Preserve this qualification and failed strict check (D014); do not hide drift.
- `docs/reproduction/two_pass_prefix.md` and `test_prefix_passes.py` verify both
  real prefix passes. Routing uses no_grad; task execution re-embeds/recomputes
  the same trainable prefix. eval() alone does not disable gradients.
