# Implementation plan

Current status: Phases 0–2 and synthetic Phase 3 integration are implemented.
Both tree and integrated search pass end-to-end tests. The next milestone is
real-corpus preprocessing and an AG News fixed/T-DHT comparison protocol. The
staged acceptance criteria below remain the roadmap; see progress.md for evidence.

1. **Phase 0 / first session:** audit all PDF pages; record source hash, equations,
   tables and ambiguity alternatives; initialize locked local tooling and typed
   configurations. Acceptance: documented CPU smoke and independent tests.
2. **Phase 1 / first session:** fixed small encoder; immutable validated total-depth
   architecture; dense masking reference and projection-sliced execution. Acceptance:
   fixed equivalence, stopping, prefix/group checks, gradient and projection tests.
3. **Phase 2 / next:** implement dimension-agnostic per-arm linear UCB independently,
   CPU float64 statistics, local sampling generator and versioned state. Validate
   against direct ridge regression, hand-computed scores/bonuses, observed-arm-only
   updates, exploration, seeds and checkpoint round trips. Then implement logistic
   Gaussian TS after recording A03–A05 choices: stable penalized likelihood,
   positive precision, sampling, top-K, selected-only binary feedback, optimizer
   convergence, covariance updates and synthetic learning behavior.
4. **Phase 3:** settle feature/reward conventions; add train-only calibration,
   pooling/normalization, tree policy and depth queues. Keep routing no-grad separate
   from rerun prefix; save decision-time contexts. Train tiny synthetic data first.
   Verify delayed updates, tail queues, no evaluation updates and resume equivalence
   including policies/queues/history/RNGs. Add integrated S/F policies afterward.
5. **Phase 4:** define AG News train/validation/test protocol, initialization/tokenizer
   and budget; run fixed baseline and T-DHT; prepare checkpoint presentation,
   observed F1/cost/action distributions and honest deviation list. No numeric-match
   prerequisite for presenting a working independent method.
6. **Phase 5:** repeated seeds, S/F and additional datasets; separately interpret
   P-DHT and its freeze schedule; reproduce tables as closely as evidence allows.
7. **Phase 6:** only after stable reproduction, review current related work and
   select a measurable extension, baseline and ablations. No novelty claim yet.

No full training or dataset/model download belongs to the first session. Each later
run must have an experiment manifest and compute budget before launch.
