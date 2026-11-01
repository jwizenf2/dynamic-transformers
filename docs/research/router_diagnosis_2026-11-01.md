# Why early signals fail to route depth (2026-11-01)

Question: chain v3 (Ouro-2.6B, LoRA) needs 4 loops instead of 2 on ~16% of NESTful requests, and
which requests is largely stable under greedy decoding (78% of the 2→4 gains are also solved at
5 and 6 loops). Yet routers built on loop-1/2 signals of the first ~45 answer tokens rank "4
helps" at AUC .57–.65 and did not beat a random mix of fixed depths on held-out NESTful
[240, 720). Hypothesis: extra depth improves problem formulation (`depth_anatomy.py`: 99 of 117
fixed requests were formulation errors at 2 loops), and early states do not reveal when the
shallow model formulates wrongly. Three explanations to separate:

* **Router design**: the information is in the signals; the model, target or aggregation is wrong.
* **Representation**: the information exists only after deeper computation; loops 1–2 do not
  encode it.
* **Early-prediction limit**: the outcome is decided later in the trajectory; no signal from the
  first answer tokens, at any depth, determines it.

## Diagnostics already run (offline, no GPU)

`scripts/lab/router_diagnosis.py`, `results/router_diagnosis/report.json`. Two sets: NESTful
[0, 720) (exploratory; all seen before) and the 531 MathQA router-training tasks. Data: greedy
dialogues at 2 and 4 loops, the 48-token probe with all four loops' readouts, loop-1/2 hidden
states (MathQA), 4 sampled 2-loop dialogues (MathQA).

**D1, where the 2- and 4-loop dialogues first differ.** For requests 4 loops fixes, the two
dialogues are token-identical through the 45-token probe window in 55% (NESTful) / 57% (MathQA)
of cases and through the whole first turn in 34% / 41%; median first difference at token 51 /
49. When the outputs are the same through the window, no readout of those outputs can separate
the requests; any early signal has to come from internal state that has not yet changed the
output.

**D2, can loops 1–2 foresee loop 4's revisions?** Token level, along the probe: loop 4 changes
loop 2's argmax on 14.2% of function-name tokens and 3.8% of number tokens. Loop-1/2 readouts
(entropy, margin, gate, state change, loop-1/2 disagreement) predict which tokens loop 4 will
revise with AUC .87 (function names) / .96 (numbers); hidden states add nothing (.83 / .94 on
MathQA). The early states *do* represent where deeper loops will change the output.

**D3, information ladder for "4 helps"** (request level, AUC, 5x5 CV; each level adds
information regardless of its cost; three model variants):

| Information | NESTful 720 | MathQA 531 |
|---|---|---|
| L0 loop-1/2 readouts of the probe (deployable) | .599 | .643 |
| L1 + loop-3/4 readouts and convergence on the same tokens | .608 | .639 |
| L2 + the realised loop-2→4 revisions on the probe | .609 | .660 |
| L3 + first-turn verification (2-loop first turn ≠ 4-loop) | .637 | .656 |
| L4 + where the full dialogues diverge (needs the full 4-loop run) | .701 | .757 |

Best of the three models per row (P_4 − P_2, direct "helps", quadratic): direct training on the
benefit label and quadratic features do not help (quadratic is worse; L4 direct .736 / .812 is
partly mechanical, since identical dialogues cannot differ in outcome). Hidden states with
nested PCA gave .664 on MathQA earlier (`phase_a_router.py`).

**D4, decomposition.** "4 helps" = 2 loops wrong AND 4 loops right. The deployable features
predict "2 loops wrong" with AUC .79 / .82 and "4 loops right" with .77 / .79, but "4 loops right
among requests 2 loops gets wrong" with only .66 / .68. The bottleneck is telling rescuable
failures from hopeless ones (334 of the 451 NESTful 2-loop failures are wrong at 4 too), not
detecting failure.

**D5, output disagreement at 2 loops (MathQA).** Distinct first turns among greedy + 4 samples:
AUC .51 (useless). Distinct final answers .54; low agreement of samples with the greedy answer
.57; adding all to L0 raises AUC to .714, but final-answer disagreement needs four extra full
2-loop dialogues (~2x fixed-4 cost): it is a correctness detector, not a cheap router signal.

## Reading

* **Not primarily router design.** Changing the target (direct benefit), capacity (quadratic,
  hidden-state PCA) or features (loop 3–4 readouts) moves AUC by at most ±.03 on the same
  information.
* **Not a representation limitation in the usual sense.** Loops 1–2 already encode which tokens
  deeper loops will revise (AUC .87–.96), and adding what loops 3–4 actually compute on the probe
  tokens (L1, L2) barely helps. Even the deep model's own view of the first answer tokens does
  not determine whether the request will be rescued.
* **Mostly an early-prediction limit.** In over half of the rescued requests the shallow and deep
  runs write identical tokens through the window; the outcome separates later (median ~50 tokens,
  often in later turns), and only full-trajectory information (L4) reaches AUC .70–.76. This fits
  the anatomy results: depth raises per-step reliability (s .85 → .91), and the payoff
  accumulates over the chain, so small early differences compound into an outcome difference that
  is not yet visible.
* The specific missing piece is **rescuability**: early signals tell that a request is hard
  (likely to fail at 2) but not whether 4 loops will fix it or fail too.

Caveats: greedy single-run labels (the stable-label check gave the same AUC, .574 vs .583);
NESTful [0, 720) is no longer a clean test, so these are descriptive; the probe follows the
4-loop draft, not the 2-loop one, for the first 48 tokens.

## Related work and what it implies

* LEV (Song et al. 2026, 2610.20951): per-question depth for looped models with a small stopper;
  saves 15–24% compute for <1 point, "matching or slightly beating a fixed mix of depths" on its
  own (multiple-choice) benchmark. The same ceiling we see, in an easier setting.
* Popescu et al. 2026 (2607.20519), diagnosing halting in looped transformers incl. Ouro: simple
  post-hoc confidence readouts often match or beat learned linear/MLP gates; Ouro's pretrained
  ponder gates are competitive but not Pareto-optimal. Consistent with D3: more router capacity
  does not help.
* Token-level halting and cascades: CALM (Schuster et al. 2022), KL-based halting for recurrent
  depth (Geiping et al. 2025), token-level uncertainty for cascade deferral (Gupta et al. 2024).
  These decide per token, where D2 shows the signal is strong; the request-level outcome is a
  different quantity.
* Routing between models by predicted quality gap: RouteLLM (Ong et al. 2025), FrugalGPT (Chen
  et al. 2024), input-adaptive allocation (Damani et al. 2024, which learns marginal-reward
  predictors for best-of-k and routing in math/code/dialog). D4 says our gap is not learnable from
  early signals because it requires rescuability, not difficulty.
* Self-speculative decoding for looped models (LoopSpec 2609.17184, WaveFront 2609.23033,
  depth-asynchronous self-speculation 2609.34538, LoopAhead 2610.21594): a cheap-depth draft
  verified by full depth gives exactly the deep output faster. Any router that relies on deep
  verification of the shallow draft competes with this, which already reaches 4-loop accuracy;
  verification-based routing only makes sense if it is cheaper than speculative verification.
* Per-loop KV caches (Mishra et al. 2026, 2610.09013) make mid-sequence depth switches pay for
  the missing deep KV; Ouro's loop-1/2 caches are reusable, loops 3–4 must be computed over the
  context.

## Feasibility with the current architecture

| Signal | Available now | Deployable cost | Status |
|---|---|---|---|
| Loop-1/2 readouts, hidden states on early tokens | yes | free in a 2-loop run | tested: AUC .60–.66 |
| Hidden-state convergence ‖h_2 − h_1‖ | yes | free | in L0, adds nothing |
| Loop-3/4 readouts on early tokens | yes (probe) | 4-loop pass over ~48 tokens + loops 3–4 over the prompt | tested (L1): +.01 |
| Marginal-benefit (direct) target | yes | — | tested: no gain |
| Output disagreement among 2-loop samples | yes (MathQA) | first-turn samples cheap; full samples ~2x fixed-4 | first turn useless; full answers only as correctness detector |
| Verification of the drafted first turn by 4 loops | approximated from greedy texts | loops 3–4 over prompt + turn | tested binary (L3): +.03 |
| Late / online signals along the 2-loop dialogue | yes (`router_trace.py`, MathQA) | free, but restart wastes the 2-loop work | tested (A8): AUC up to .69, no policy gain |
| Mid-dialogue escalation (continue at 4 from turn t) | needs new runs | loops 3–4 over the context so far | untested |

## Proposed experiments (cheapest first)

**Tier 1, offline, hours of analysis, no GPU.**
1. Rescuability, not difficulty. Train the deployable features, hidden states and trace signals
   on the D4 sub-task (among predicted 2-loop failures, rescued vs hopeless) on MathQA, with
   nested CV. If no feature set exceeds AUC ~.70 there, early routing is closed for this model.
2. Continuous verification from existing traces: on MathQA, does the first turn at which the
   2-loop trace's loop-1/2 disagreement or tool-response surprisal spikes coincide with the
   divergence point from the 4-loop dialogue (D1)? If signals peak at the divergence, a late
   decision at that turn is the right design; if not, late signals are also blind.
3. Stable labels on NESTful (depths 4–6 agree) as the target for D3/D4, to bound how much label
   noise depresses the ladder (expected small: .574 vs .583).

**Tier 2, GPU, few hours, no architecture change.**
4. Mid-dialogue escalation on the 292 clean MathQA tasks: at each turn boundary of the 2-loop
   dialogue, compute loops 3–4 over the context (reusing loop-1/2 caches) and continue at 4
   loops; record accuracy and cost per switch turn (~292 x 3.6 continuations, ~4 h). This tells
   whether a late signal plus continuation (no wasted restart) can beat the fixed-depth hull,
   which A8's restart-only accounting could not test. Only worth running if Tier-1 item 2 finds
   that signals peak near the divergence.
5. Deep verification along the 2-loop trajectory: one teacher-forced 4-loop pass over each
   2-loop dialogue (prefill only, no generation) to get per-token KL(p_4 ‖ p_2) and rejected
   tokens at every turn; AUC for "4 helps" by turn. An information bound for verification-based
   routing (compare with self-speculative decoding's cost before proposing it as a method).

**Tier 3, training, no architecture change (only if Tier 1–2 find signal).**
6. A rescue head trained on more labels (2-loop and 4-loop dialogues on ~3k MathQA train tasks,
   ~25 h), targeting the D4 sub-task with the best Tier-1 features.

**Tier 4, architecture (not recommended now).** Joint training of a request-level halting head
with a depth-benefit objective, or token-level depth (Mixture-of-Recursions style) with shared
KV. The diagnostics point to an early-prediction limit, which these do not remove for
request-level routing; token-level depth is where the signal is (D2), but its savings vanish
under batching (`progress.md`, serving bench).

## Decision rule

* Tier 1 item 1 finds a rescue AUC well above .70 → router design problem on the rescue sub-task;
  build that router (Tier 3) and re-test on unseen NESTful [720, 1416).
* Item 1 fails but item 2 shows signals peaking at the divergence turn → early limit, but late
  routing with continuation may work; run Tier 2 item 4.
* Both fail → report the early-prediction limit as the finding: the need for depth is a stable
  property of the request that the model only reveals by computing it.

## Tier 1 results (2026-11-01, `router_diagnosis2.py`, `router_switch.py`)

**Rescuability (item 1).** Among requests 2 loops gets wrong (as if a perfect 2-loop verifier
existed), AUC for "4 loops fixes it": MathQA early features .68, + loop-1/2 hidden states (PCA 32)
.72–.73, + whole 2-loop trace .69, all .72; NESTful early features .66, + loop-3/4 readouts and
revisions .66; stable labels (depths 4–6) .63–.64. Below the bar of "well above .70": a better
router on the same signals is not the fix.

**Fork-point signals (item 2).** At the first token where the 2- and 4-loop dialogues differ, the
2-loop trace is highly uncertain: the fork ranks at the 97th–99th percentile of the dialogue's
tokens by loop-2 entropy, margin, KL(p_2 ‖ p_1) and log p_1. But equally so for forks that
change the outcome and forks that do not (AUC rescued vs neutral forks .44–.55). The model
knows where its trajectory can branch, not which branch matters.

**One-way switch (new, evaluable offline).** Decode at 2 loops; at the first token whose loop-1/2
signal exceeds τ, compute loops 3–4 over the context and continue at 4 loops. A switch at or
before the fork reproduces the 4-loop dialogue exactly at exactly the 4-loop cost (no wasted
work), so restart waste, the reason late decision points failed in A8, is removed. Result
(MathQA, selection on clean, check on pool): best signal loop-2 entropy, at <= 0.90x fixed-4
cost accuracy .325 vs mix .334 on clean (−.009, lower bound) and .506 vs .497 on pool (+.009);
fixed-4 accuracy is reached only at 0.95–0.99x. Only 5–8% of switches land after the fork, so
timing is not the problem; selectivity is: nearly every dialogue has some uncertain token.
Tier-2 item 4 (mid-dialogue escalation runs) is therefore not needed: the switch evaluation
already covers continuation exactly wherever it matters.

**Decision.** Rescue AUC is not well above .70, and fork-time signals do not separate consequential
from inconsequential forks; an online switch without restart waste still equals the random mix.
Verification of the drafted call (Tier-2 item 5) faces the same problem: at a fork, a deep pass
shows that the deep model would write something else, which every fork has by definition, not
whether that change leads to success. Conclusion: an early-prediction limit. Loops 1–2 encode
where computation is uncertain (token level, AUC .87–.96 for deep revisions; forks at the 97th+
percentile), but whether extra depth changes the final outcome is decided by how a fork
propagates through later steps, which no single-step signal shows. For the paper: uncertainty is
local, consequence is global.
