# Plan: a robust router paper (2026-10-28, 8 weeks)

User's order of priorities: (A) make the router effect clearly significant first, then (B) a second
benchmark, (C) other models, (D) measured throughput, (E) more baselines and the remaining items.
Each phase gates the next. Pre-register every confirmatory test; develop only on non-test data.

## Where we are

- Escalation router (start at 2 loops, loop-1/2 decision-token signals, switch to 4) frozen on 292
  clean MathQA-val tasks; NESTful [0, 240): .471 at 0.764x fixed-4 cost vs fixed-4 .492; +.030 over
  the best random mix (90% CI −.005 to +.048): CONSISTENT, not significant.
- Ceiling analysis (2026-10-28): the perfect per-request router's advantage comes mostly from
  requests unsolvable at every depth (clean val 60%, NESTful 48%). A perfect "give up" detector
  (unsolvable → 1 loop, else 4) matches fixed-4 accuracy at 0.54x (clean) / 0.62x (NESTful) cost;
  a perfect 2-vs-4 choice saves only 12–17%. Solvability is more predictable than depth benefit
  (cross-validated AUC .76 vs .66 with loop-1/2 features; question-only .56 / .48). Outcome noise is
  large: deciding from an independent sampled run gives .267 as a give-up rule, so the router needs
  calibrated probabilities. Realistic target: fixed-4 accuracy (−1 point) at <= 0.80x cost.

## Phase A: a router that matches fixed-4 accuracy at less compute (first)

A1. Three-way decision at the 2-loop start: give up (continue at 1 loop, keeping the probe tokens:
    2·(P+k) + (T_1 − P − k)), stay at 2 (2·T_2), escalate to 4 (4·T_4 + 2·k). Variant: start at 1
    loop with loop-1-only signals.
A2. Denoised labels: 3–4 sampled dialogues per depth (1, 2, 4) on the 531 training tasks (292 clean
    val + 239 MathQA-test pool) → per-task success-probability estimates; train on soft targets.
A3. Features: hidden states at the decision tokens (loops 1–2, pooled; PCA/ridge), token-type
    features (function-name token, digit tokens), probe length 16/32/48/64.
A4. Models/objective: logistic, ridge on PCA, small MLP, gradient boosting; direct expected-reward
    policy (argmax_d p_d − λ·cost_d) instead of independent per-depth classifiers.
A5. Protocol: develop by cross-validation on the 531; pre-register the final router and target
    (within 1 point of fixed-4 at <= 0.80x cost, plus gain over the random mix); test once on
    NESTful [240, 720) (480 unseen requests); report pooled with [0, 240) as secondary.
A6. Optional: mid-dialogue escalation after the first tool result (needs new runs); bandit
    (online, bandit feedback) learning the same decision, offline replay with regret curves.

## Phase B: a second benchmark

Executable, multi-step, verifiable tool use outside the MathQA distribution. Candidates: BFCL v3
multi-turn, NESTful's non-math portion if executable, other executable tool suites. 1–2 day
feasibility check, then SFT data / zero-shot protocol, depth labels, router transfer.

## Phase C: other models

Ouro-1.4B with the chain-v3 recipe (scale axis); Dory 1.0 4B zero-shot (another lab; depth saturates
at ~2 loops, so allocation should matter less); optionally Dory fine-tuned (MLX trainer to build).

## Phase D: measured throughput

End-to-end serving benchmark (MLX) with mixed single-call / multi-step traffic, batch 1–16:
router vs fixed depths vs Ouro's exit gate; tokens/s and latency, not only loop-tokens. A simple
LoopSpec-style draft-and-verify decoder (drafts at 1 loop, verified by loops 2–4) to show the router
and speculative decoding stack (LoopSpec reduces batch-1 latency, not compute; the router reduces
compute and keeps its savings under batching).

## Phase E: baselines and remaining items

- Baselines: fixed depths, random mix, task-type rule (single-call → 2, multi-step → 4), Ouro's exit
  gate (q sweep), question-only router, 8-token probe router, LoopSpec, perfect router (ceiling).
- Training seeds (2 more) for chain v3 / depth-aware vs Ouro loss claims.
- Ablations: features, probe length, operating points, {1,2,4} vs {1,2,3,4}.
- Related work: LoopSpec, WaveFront, DAS, LoopAhead (speculative decoding for looped LMs); LoopOPD,
  LoopFormer (cross-loop distillation); 2607.20519 (fixed-prior depth supervision); Dr.LLM, LEDE,
  LayerSkip, CALM (early exit); Ouro, Dory, Huginn/Raven (looped LMs).
- Writing (weeks 7–8): analysis-paper framing (compute allocation in looped LMs for tool use; per-step
  depth = search; per-request headroom = mostly unsolvable requests; deployable router; batching;
  depth vs width; MathQA leakage audit), independent review, code/data release.
