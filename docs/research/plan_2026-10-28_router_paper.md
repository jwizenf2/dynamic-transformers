# Plan: a robust router paper (2026-10-28, 8 weeks; revised the same day)

User's order of priorities: (A) make the router effect clearly significant first, then (B) a second
benchmark, (C) other models, (D) measured throughput, (E) more baselines and the remaining items.
Each phase gates the next. Pre-register every confirmatory test; develop only on non-test data.

## Venue (decided 2026-10-28)

ACL 2027 via ACL Rolling Review, submission deadline 4 January 2027 (AoE; Kyoto, Aug 17–22,
2027): long paper, 8 pages + references + appendix, Limitations section required; Findings is a
possible outcome. Fallback if Phases B–D slip: COLM 2027 (deadline expected late March 2027, not
yet announced). ICML 2027 (expected late January) fits less well: the paper is an empirical study
of language models in tool use. Draft: `paper/` (ACL template, style files pinned).
Astra (GPT-6) is not cited: its looped architecture rests on one anonymous report (The
Information, via Fortune), with no OpenAI technical disclosure.

## Positioning (what is ours, what is not)

- **The router is per request**: decided once, after the first ~45 tokens of the first tool call;
  the whole request then runs at 2 or at 4 loops (escalation restarts the first call at 4 loops,
  reusing the prompt's loop-1/2 work). Per-tool-call depth was tested and is search (per-step
  oracle .308 vs same-budget sampling .325 on Ouro; .633 vs .617 on Dory).
- **Closest prior work: LEV** (Song, Yan, Zhang, Sun; arXiv 2610.20951, 14 Oct 2026). Per-question
  depth on Ouro-1.4B/2.6B and Nanbeige4.2-3B; a logistic stopper on signals from consecutive loops
  (answer change, KL, margin, entropy, top probability) predicting agreement with the deepest
  answer; threshold set on unlabelled requests. 76–87% of deepest compute at −0.3 to −0.9 points;
  +0.2 to +1.0 over the fixed-mix frontier; confidence thresholds do not beat the frontier on Ouro.
  Multiple choice scored in one forward pass (no generation, no tool calls); latency at batch 1
  only; batched serving left to future work.
- **Also prior**: early-exit and cascade confidence signals (CALM, Gupta et al. 2024 cascades),
  stopping when successive loops agree (Geiping et al. 2025), trajectory readouts (Popescu et al.
  2607.20519), whether loops help tool calling (Popescu et al. 2608.18171: single-turn, fixed depth
  or Ouro's gate).
- **Ours**: depth allocation for *generative, multi-step, executed* tool use: (1) per-step
  allocation is search, per request works; (2) the signal lives in the decision tokens
  (function name, arguments), not in the boilerplate where loops always agree, so a generative
  setting needs the probe placed there; (3) escalation mid-generation with prompt-cache reuse;
  (4) behaviour under batching (measured); (5) the MathQA/NESTful overlap quantified (credit: Amini
  et al. 2019 noted AQuA-RAT near-duplicates; Calc-X deduplicated AQuA-RAT train vs test; nobody
  we found measured MathQA's own random split or its consequence for NESTful).
- **MathQA numbers for the paper (strict thresholds)**: of 1,293 validation tasks, 40 identical in
  text to a training task, 914 (71%) at number-masked template similarity >= 0.7, 666 (52%) at
  >= 0.9; before filtering, 1,074 of NESTful's 1,416 problems had a training match at >= 0.7.
  Wording: "we quantify", not "we discover".
- **Abstract/introduction**: to be rewritten with this framing (contributions in order: per-step vs
  per-request allocation in generative tool use; decision-token signal and mid-generation
  escalation; batching; MathQA/NESTful overlap and clean splits) and LEV, Calc-X and Amini et al.
  added to related work with the explicit differences above.

## What makes it publishable (bar agreed 2026-10-28)

"Same accuracy at ~75% compute" is competitive (LEV 76–87% at −0.3 to −0.9; Mishra et al. −30%
FLOPs per token on Ouro) if it is backed by:
1. **Statistics on held-out requests** (NESTful [240, 720)): non-inferiority, router − fixed-4
   accuracy with a 90% lower bound > −1 point; and gain over the equal-compute random mix with a
   lower bound > 0.
2. **Baselines beyond fixed depths and random mixes**: an LEV-style stopper (same features as LEV,
   same threshold rule), simple confidence thresholds, KL halting (Geiping et al.), Ouro's gate,
   the task-type rule.
3. **Generality**: a second model (Ouro-1.4B or Dory) or a second benchmark; both is much stronger.
4. **Measured speed** under batching (tokens/s, latency), not only loop counts.
5. **The analysis story** as the paper's differentiation from LEV.

| Have | Realistic outcome |
|---|---|
| Phase A passes, NESTful only, one model | workshop |
| + second model or benchmark, + measured throughput, + LEV-style and confidence baselines | ACL Findings (likely) |
| + both generality axes, clearly beating the LEV-style stopper, + analysis | ACL main (competitive) |

Main risk: an LEV-style stopper matching our router. Then the method contribution shrinks to "it
also works for generative tool use" and the paper rests on the analysis. Run that baseline as soon
as the Phase A labels land (A7); it decides which paper we are writing.

## Where we are

- Escalation router frozen on 292 clean MathQA-val tasks; NESTful [0, 240): .471 at 0.764x fixed-4
  cost vs fixed-4 .492; +.030 over the best random mix (90% CI −.005 to +.048): CONSISTENT.
- Ceiling: perfect 2-vs-4 router .521 at 0.83x (NESTful). Per request: 85 correct at both depths,
  33 only at 4, 7 only at 2, 115 at neither. "4 helps" AUC ~.66 today; simulation says ~.75 matches
  or beats fixed-4 at 0.76–0.80x. Requests failed at every depth (48%) are a later, separate topic.
- Phase A data job f15ee917 running (commit 53540bd): hidden states at the probe tokens, 3 extra
  sampled dialogues per depth (2, 4) on the 531 training tasks.

## Phase A: a router that matches fixed-4 accuracy at less compute (first)

A1. Focus: the 2-vs-4 decision (match fixed-4 accuracy at less compute by escalating exactly the
    requests that need 4 loops). "Give up" on requests failed at every depth is deferred and, if
    used, framed as selective prediction (flag likely failures).
A2. Denoised labels: 3 extra sampled dialogues per depth (2, 4) on the 531 training tasks (292 clean
    val + 239 MathQA-test pool) → per-task success rates; train on the expected benefit of 4 loops
    (greedy + samples), evaluate on greedy outcomes only. CV folds keep all samples of a task
    together.
A2b. Pre-registered label analysis: per task, success rates at 2 and 4 loops from greedy + 4
    samples (T 0.7); Δ = P_4 − P_2. Reliability ceiling: disjoint sample halves A = {s0, s1},
    B = {s2, s3}; report corr(Δ_A, Δ_B), AUC of Δ_A for (Δ_B > 0) and for the greedy "4 helps"
    label (Spearman-Brown to 4 samples); no feature router can beat the sample→greedy AUC.
    Sample-trained models are a label-smoothing variant compared against greedy-only training.
A3. Features: hidden states at the decision tokens (pre-registered: <= 3 poolings — mean loop-2
    state; mean loop-2 − loop-1; mean at tokens where loops 1/2 disagree → standardise → PCA inside
    each training fold, k in {16, 32, 64} → L2 logistic/ridge, λ from a 3-point grid by nested CV;
    paired bootstrap against the 12-feature router); token-type features (function-name token,
    digit tokens); probe length 16/32/48/64.
A4. Models/objective: logistic, ridge on PCA, small MLP, gradient boosting; direct expected-reward
    policy (argmax_d p_d − λ·cost_d) instead of independent per-depth classifiers.
A5. Protocol: develop by cross-validation on the 531 (held-out 20% dev split or nested CV for the
    final choice, short pre-listed model set); pre-register the final router and both tests
    (non-inferiority vs fixed-4 at <= 0.85x cost, stretch <= 0.80x; gain over the random mix);
    test once on NESTful [240, 720) (480 unseen requests); pooled with [0, 240) as secondary.
A6. Bandit (after A5): online learning of the same decision from bandit feedback, offline replay
    with regret curves (exact, since outcomes at both depths are known); note the deployment reward
    problem (correctness is not observed online). MCTS is not used: the per-request decision has
    2–3 actions with known outcomes, and per-step depth is search.
A8. Late decision points (pre-registered 2026-10-29, before trace data; `router_trace.py`
    records loop-1/2 signals on every token of the actual 2-loop dialogue, tool-response
    surprisal per token, per-turn hidden states). Decision points: {after ~45 tokens (current),
    after the first tool result, after turn 2, end of the 2-loop run (cascade)}. Policy at each
    point: stay at 2 or restart the request at 4 loops from the prompt (the only policy the recorded
    4-loop outcomes support; "continue at 4 from turn t" is not evaluable offline). Cost on
    escalation: 2-loop work so far + 4·T_4, with the prompt's loop-1/2 caches reused (stated as an
    assumption; no latency claim, escalation is sequential). Features at point t use only turns
    <= t (row-level turns/finished/value only at the end). Labels: the trace's own 2-loop
    correctness vs the existing d4 greedy outcome. One model class (L2 logistic on aggregated
    turn features, plus the 12 early features), 5x5 CV. Selection rule: highest accuracy at a fixed
    cost cap of 0.90x fixed-4 on the 292 clean tasks; check that the chosen point replicates on the
    239 pool tasks; then freeze for NESTful [240, 720) (needs d2 traces and d4 greedy on 480 new
    requests). The cascade cannot save compute (even its oracle costs ~0.94x on clean with cache
    reuse, 1.27x without); its only possible value is accuracy above fixed-4 at ~1x, reported as
    such.
A7. LEV-style baseline (as soon as A2 labels land): LEV's features from consecutive loops (answer
    change, KL, margin, entropy, top probability) read at the same probe tokens, logistic stopper
    predicting agreement with the deepest answer, threshold set on unlabelled requests; compared
    on the same folds and, frozen, on NESTful [240, 720).

## Phase B: a second benchmark

Executable, multi-step, verifiable tool use outside the MathQA distribution. Candidates: BFCL v3
multi-turn, NESTful's non-math portion if executable, other executable tool suites. 1–2 day
feasibility check, then SFT data / zero-shot protocol, depth labels, router transfer.

## Phase C: other models

Ouro-1.4B with the chain-v3 recipe (scale axis); Dory 1.0 4B zero-shot (another lab; depth
saturates at ~2 loops, so allocation should matter less); optionally Dory fine-tuned (MLX trainer
to build).

Candidate: Nanbeige4.2-3B (Nanbeige Lab, arXiv 2607.22083; weights at
huggingface.co/Nanbeige/Nanbeige4.2-3B, weight license not stated in the paper): the whole stack run
twice (2 passes, fixed in training), strong agentic results (MCP-Atlas 57.8, SWE-Bench Verified
63.6). Only a 1-vs-2 choice exists, and only if 1-pass output is usable; LEV uses it at 2 steps,
so check 1 vs 2 passes on BFCL / NESTful before adopting.

## Phase D: measured throughput

End-to-end serving benchmark (MLX) with mixed single-call / multi-step traffic, batch 1–16:
router vs fixed depths vs Ouro's exit gate; tokens/s and latency, not only loop-tokens. A simple
LoopSpec-style draft-and-verify decoder (drafts at 1 loop, verified by loops 2–4) to show the
router and speculative decoding stack (LoopSpec reduces batch-1 latency at a fixed depth; the
router reduces compute and keeps its savings under batching).

## Phase E: baselines and remaining items

- Baselines: fixed depths, random mix, task-type rule (single-call → 2, multi-step → 4), Ouro's
  exit gate (q sweep), KL halting (Geiping et al. 2025), LEV-style stopper (A7), confidence
  thresholds, question-only router, 8-token probe router, LoopSpec, perfect router (ceiling).
- Training seeds (2 more) for chain v3 / depth-aware vs Ouro loss claims.
- Ablations: features, probe length, operating points, {1,2,4} vs {1,2,3,4}.
- Related work (in `paper/sections/related.tex`; to add: LEV, Calc-X, Amini et al.'s duplicate
  note): looped LMs (Ouro, Dory, Huginn/Raven, Nanbeige4.2, Universal Transformers); adaptive depth
  (ACT, PonderNet, Ouro gate, 2607.20519, 2610.18127, 2610.09013, CALM, LayerSkip, MoD,
  Router-Tuning, FlexiDepth, Dr.LLM, CoLa, LEDE, Mixture-of-Recursions, LEV); speculative decoding
  with loops (LoopSpec, WaveFront, DAS, LoopAhead) and cross-loop distillation (LoopOPD);
  test-time compute allocation (Snell et al., Damani et al., self-consistency, FrugalGPT,
  RouteLLM, cascades); tool use (Toolformer, Gorilla, APIGen, BFCL, NESTful, MathQA, AQuA-RAT,
  2608.18171).

## Timeline (8 weeks, to the 4 January 2027 ARR deadline)

| Week | Experiments | Writing |
|---|---|---|
| 1 | Phase A data lands; A2b ceiling; A7 LEV-style baseline | rewrite abstract/intro with the positioning above; related work additions |
| 2 | A3–A5 router development; pre-register; NESTful [240, 720) labels + probe | setup section (models, data, protocol, compute, leakage audit) |
| 3 | A5 confirmatory test; Phase B feasibility | per-step and loop-count findings |
| 4–5 | Phase B and/or C; A6 bandit replay | router section; generality |
| 5–6 | Phase D throughput; Phase E baselines, seeds, ablations | serving section; full draft by end of week 6 |
| 7 | gap-filling from an independent full-draft review | limitations, appendix |
| 8 | — | polish; code and clean-split release; submit |

Writing order rule: abstract and conclusion last; red placeholders (`\pending{}`) for every number
not yet measured.
