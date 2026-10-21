# Plan: MCTS teacher, contextual bandit at inference (2026-10-21)

Professor's suggestion (2026-10-21), agreed by the user: use MCTS to judge intermediate states of
multi-step tool calling, and a bandit (greedy or exploring) to choose the loop count at inference.
This replaces the end-of-dialogue REINFORCE policy (failed: synthetic tasks solvable at every
depth; one sparse reward per dialogue; 40 updates).

## Gate 0 (running): is per-step depth a lever at all?

Per-call oracle v2 (job 1f2f3c8b): same-budget sampling oracle, best-of-4 sampled dialogues,
escalation rules, graded-turn and prefill-inclusive costs. If the sampling oracle matches the
depth oracle, stop here: per-step depth is not the lever (see
`review_2026-10-21_independent.md`).

## Stage 1: MCTS teacher (offline, training problems only)

- Tree: node = dialogue state after a turn; action = loop count 1–4 for the next turn; leaf =
  finished dialogue. Reward = 1[final answer correct] − λ · cost, cost counted as loop-tokens
  including prefill (Ouro) or block-tokens (Dory).
- UCT selection; rollouts greedy at depth 4 (as the v1 oracle) or by the current policy; the
  turn cache of `ouro_percall_oracle.py` makes repeated prefixes free.
- Cheap value prior for intermediate states: share of the gold intermediate values (MathQA
  tasks carry gold calls) computed so far. Guidance only, since valid alternative orders exist.
- Output: Q(s, d) estimates for visited states. Budget: ~30–50 simulations per task, a few
  hundred depth-sensitive (solvable-at-some-depth) MathQA-train tasks.
- Control: the same simulation budget spent on sampling instead of depth; MCTS is only a useful
  teacher if depth search beats it.

## Stage 2: contextual bandit at inference (DHT's formulation, per tool-call step)

- Context: turn features (turn index, calls so far, last call failed, per-loop
  confidence/agreement signals, optionally hidden-state probes); arms: loop counts 1–4.
- Warm start from the MCTS Q(s, d) values (supervised fit of the bandit's reward model).
- Inference modes: greedy (argmax predicted reward − λ·cost, no exploration) and LinUCB /
  logistic Thompson sampling (reusing `dht.bandits`), learning online from immediate signals
  (tool-call errors, compute) and, where a verifier exists, final correctness. Report regret for
  the exploring mode.
- Must be contextual: a non-contextual multi-armed bandit converges to one fixed depth.
- DAgger rounds: label states the deployed policy visits with fresh MCTS values.

## Evaluation (NESTful once, MathQA-val for tuning)

The policy alone, no search, against fixed depths 1–4, Ouro's gate (q sweep), post-hoc
confidence readouts (2607.20519), the escalation rule (greedy training-free baseline), a matched
random allocation, and the MCTS teacher as a ceiling. Accuracy vs compute including prefill;
≥500 validation tasks for headline comparisons; measured throughput with depth-bucketed batches
and the planned cache (full per-loop KV for new text, last-loop KV for generated text). Ouro and,
if its calibration passes, Dory.

## Open question for the professor

"Greedy at inference": greedy choice from learned values (assumed), or the greedy escalation rule
(start cheap, escalate on a failed tool call), which is already a baseline.
