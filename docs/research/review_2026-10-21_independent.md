<!-- Independent review by a Fable subagent (max effort), 2026-10-21. Read-only review; [V]/[J] marks are the reviewer's. Spot-checked: per-call-only depth sequences and stop-turn depths match results/ouro_percall_oracle/chain-depth/examples.jsonl. -->

# Independent review: loop-depth control for looped tool-calling LMs (2026-10-21)

Legend: **[V]** = verified by reading repo files/result rows or the arXiv/HF page today; **[J]** = my judgment. All result-row analyses below were run read-only on `results/` (scratchpad scripts only; nothing in the repo was modified).

---

## 1. Verdict on the current path

**Chain v3 (MathQA SFT): sound as capability work, but it does not test the thesis.** It is needed because the model cannot do the chains (80% of MathQA-val tasks unsolved at every depth [V]). But before spending ~15 h on it, resolve one discrepancy that could change the whole picture (gap G1 below): your Hermes-SFT Ouro-2.6B scores .11 on single-turn NESTful while 2608.18171 reports .371 with the same recipe, and your *base* model with NESTful's few-shot prompt scores .189 — i.e., your SFT model under your prompt is *worse than base few-shot*. That smells like an eval-prompt/format artifact, not model capacity.

**Per-turn bandit/RL as the headline: likely a waste in its current form.** When I decomposed the per-call oracle rows (`results/ouro_percall_oracle/chain-depth/examples.jsonl`), the headroom that motivates it mostly dissolves:

- **The 6 "only per-call" wins are search, not allocation [V].** Their depth sequences are `[4,4,4,1,1]`, `[4,4,4,1,2]`, `[4,3,2,1]`, `[1,2,1]`, `[4,4,4,2,1]`, `[4,4,4,4,4,1]`. In 5/6 the oracle was *forced* to depth 4 on early turns (no shallower depth worked), and the win came from a late shallow turn that perturbed a failing fixed-4 trajectory (fixed-4 hit the 9-turn limit on one, stopped after 2–4 turns on others). This is best-of-4-decoders-per-turn with hindsight, ~20 rollouts per request vs 4 for the per-request oracle. The sign test (6 vs 0, two-sided p=.03) is nominally significant but the comparison is not budget-matched.
- **"Late turns favor depth 1" is largely the ungraded stop turn [V].** The dialogue ends when a turn emits no call; grading uses the last *executed* call (`ouro_percall_oracle.py:100`, `correct()`), so the final answer text is never graded. 16/22 per-call-correct dialogues end with a depth-1 "stop" turn; in all 25 correct d4 multi-turn dialogues the last turn is plain text ("The gain% or loss% is 33.33%."). Among the 16 requests solved by both, dropping the stop turn moves the depth histogram from {1:32, 2:31, 3:18, 4:2} to {1:22, 2:29, 3:15, 4:2} and loops/turn from 1.95 to 2.07. First-turn depth among those 16 is {2:7, 1:5, 3:4}, i.e. the "plan deep, execute shallow" story in the talk track is **not** in the data; the deep-early sequences are the forced-to-4 cases.
- **The cost metric hides the dominant cost [V].** In the oracle and policy environment every turn re-prefills the whole conversation at 4 loops (`gen_turn`, `Env._prefill`), and only decode loops are counted (`loops_per_token`). With ~50 decoded tokens per turn against a 1–3k-token context, per-turn decode depth is a second-order FLOP saving; under KV reuse you inherit the ragged-KV accuracy problem of 2610.09013, which you have not measured for mixed-depth histories.
- **Labels are noisy [V].** Correctness vs depth is non-monotone in 6/24 solved MathQA-val tasks and 2/16 NESTful; chain v1 vs v2 agree on d4 correctness with κ=0.58 (both 16, v1-only 9, v2-only 7). A single greedy rollout per (turn, depth) is a noisy label; Ares (2603.07915) uses K=3 trials with majority for exactly this reason.
- **Label cost [V/J].** The oracle run took ~4.9 h for 60 requests (job 09:36→14:29), ~5 min/task. Per-turn counterfactual labels on all 8,772 train tasks ≈ 700 h; even on the ~20% solvable subset ≈ 150 h. You can label a few hundred tasks, not thousands, and only ~24/120 carry any depth signal.

Net: the "steps need different computation" thesis is currently supported by an optimistic oracle whose two most quotable facts (+10 points; late turns shallow) are explained by search budget and by an ungraded turn. It may still be true, but it has to be re-measured (Section 4, move 1) before anything is built on it.

---

## 2. Gaps and flaws

### Methodological
- **G1. Unreconciled 3× gap with 2608.18171 on NESTful [V].** Their SFT Ouro-2.6B: win rate .371 (single-turn full-sequence generation with `$var` references; 500-example subset for depth sweeps; executed to check the answer). Yours: .108–.117 single-turn at d4 (Hermes zero-shot prompt), base few-shot .189. Until explained (ICL examples in the prompt? `$var` syntax? subset? win-rate definition?), "NESTful ceiling ~20%" and "model capacity is the limit" are not established. Cheap to test: run your SFT adapters with NESTful's 3-ICL prompt in single-turn.
- **G2. Oracle optimism** (above). Required control: best-of-N temperature sampling at fixed depth 4 with the same rollout budget as the per-call oracle, and a "sampling-perturbation oracle" (per turn, try N samples at depth 4) vs "depth-perturbation oracle". If they match, depth is a noise source, not an allocation lever.
- **G3. Grade the stop turn.** Either grade the final text answer (it is the real output) or fix the stop turn's depth to 1 for every method and report depth statistics excluding it.
- **G4. Cost accounting.** Report total FLOPs including prefill per turn, and measured latency in both serving models (recompute vs KV reuse with best-available fill), with accuracy under the KV-reuse model, which is what a deployed per-turn policy would use.
- **G5. Statistics.** d4 accuracy n=120: .208 [.142, .283]; n=60: .267 [.167, .383] (bootstrap). Any per-turn policy effect under ~8 points is invisible. You have 1,293 decontaminated MathQA-val tasks; use ≥500 for every headline comparison, paired per task, and 2–3 training seeds for the chain SFT (the parallel_multiple −6.5 at d4 is outside your ~2-point seed band; the NESTful v1→v2 difference is inside the noise).
- **G6. Depth-aware loss is a two-factor change, unablated [V].** Ouro loss = Σ p_φ(t)·CE_t + β·KL(p‖U) with the gate *trained*; yours = mean_t CE_t with the gate *frozen* (`ouro_lora_sft.py:211–219`). Reviewers will ask for (Ouro weights, frozen gate) and (uniform weights, trained gate). Also the frozen base gate exiting at loop 2 for 96% of tokens on the depth-aware model is a calibration accident, not a designed property; say so.
- **G7. Comparability and distribution leakage.** Chain v3 learns NESTful's tool-list rule, its formula→calls construction and MathQA's question style; instance-level decontamination (Jaccard ≥ .3 masked templates, [V]) is well done, but a reviewer will still read "trained on the construction of the test benchmark". Keep the Hermes-only model as the 2608.18171-comparable reference, report both NESTful protocols, and report a held-out *non-MathQA* chain benchmark (your synthetic chains, or ToolHop 2501.02506 / FuncBenchGen 2509.26553 once licenses are checked).
- **G8. Multi-turn failure mode is partly protocol [V].** 58% of d4 failures made <60% of gold calls; your MathQA dialogues end with "The final result is X." after the last tool response, which teaches "answer after a tool response". Measure "stopped early" separately from "wrong plan" and consider a stop-token/answer-format that is graded.
- **G9. Serving baselines are out of date [V].** LoopSpec (2609.17184): training-free, lossless, 2.89–3.36× on Ouro-2.6B at batch 1 (RTX PRO 6000, SGLang; single-request setting only). LoopCD (2610.02185, Oct 1): training-free contrastive decoding; claims halving loops matches full-depth unguided decoding (22–48% fewer FLOPs). Your "per-request depth beats the gate" story must now beat lossless self-speculation at low batch and LoopCD at matched loops; the regime where per-request depth uniquely helps is the compute-bound batched regime you already measured (b8–16).

### Framing / novelty (checked on arXiv, 2026-10-21)
- **Per-step compute allocation in agents is published**: Ares 2603.07915 (per-step effort router, minimum-sufficient labels, GRPO; TAU-bench), TAB 2604.05164 (per-turn thinking budgets, BFCL v4 multi-turn), BAAR 2602.21227, Handoff Curves 2610.15993 (later handoff adds nothing over initial selection on BFCL; monotone fits +0.2–4.1 pp). "Per-turn + loop count + tool use" specifically: my searches found nothing [V, not exhaustive]. So a positive result would be novel in combination only; the machinery (labels, bandit/RL) is standard.
- **Learned loop policies on Ouro exist at token level**: LoopRPT 2603.19714; LatentHarness 2609.39740 (counterfactual policy distillation — branch every action per state, cost-charged gain; this is your planned per-turn counterfactual labeling, at token level, already published); RecurTrace 2609.03379 (sequence-level halting head from oracle labels, *on MathQA*); TaH2 weights now public (`nics-efc/TaH2-1.7B-max{2,4,8}`, Oct 2026, trained on tool data) [V].
- **Negative precedent**: 2607.18553 — per-task loop allocation from probes/gate on frozen Ouro = matched-histogram random (GSM8K). Your own xLAM router beats matched random only at low budgets [V].
- **New substrate, 5 days old [V]**: NVIDIA Nemotron-Labs **Dory 1.0 4B** (HF, GA 2026-10-16): looped (8/20/8 layers, input re-injected each loop), 1–4 loops set via config (`n_recurrent_loops`), SFT+RL for agentic/coding, tool calling supported (vLLM `qwen3_coder` parser), OpenMDW-1.1 license (commercial OK), `transformers` support, no MLX port; `last_loop` KV mode. Its table reads (my reading of column order, unverified) Terminal-Bench 2.1 12.5/18.1/22.2 and τ³-bench Banking 11.6/12.9/14.7 at 1/2/4 loops; card says most gain comes from loop 2 and looping reduces output tokens. The ELR/Koi-4B paper (2610.12582) is by NVIDIA authors but neither page cross-references the other; treat the Dory=Koi link as unverified. Implication: "looped agentic model with elastic loops" is now a product; the field will ask "why Ouro, and why fine-tune at all?"

---

## 3. New angles (not yet tried)

**A. Falsification first: sampling-control oracle (cost ~1 lab day, reuse `ouro_percall_oracle.py` with temperature).** Per turn, try N=4 temperature samples at depth 4 (plus depth-4 rollout) instead of 4 depths. Compare accuracy and loops with the per-call oracle on the same 60–120 requests. Novelty: none; value: decides whether anything per-turn is worth building.

**B. Verifier-triggered depth escalation (training-free per-turn policy; ~1 lab day).** Run each turn at d_low (2); if an executed call errors (unknown tool, bad args, range), regenerate that turn at depth 4; optionally escalate for the rest of the dialogue. Errors/call fall .41/.20/.07/.04 across depths [V], so the environment is a free, exact critic for exactly the turns where depth matters. Compare against the fixed-depth frontier at matched *total* loops (including the wasted low-depth attempt). Novelty: tool-execution feedback as the escalation signal for latent depth — I found no prior instance [V, not exhaustive]; it is cascade/deferral logic (Jitkrittum 2307.02764) applied to loops. If this beats fixed depth, it is the cleanest possible existence proof for per-turn allocation with a signal that exists; if not, learned per-turn policies are unlikely to either.

**C. Deep prefill, shallow decode (~2 h on BFCL).** The oracle protocol already does d4 prefill + d-loop decode. Measure d4/d2 vs d2/d2 (.888) vs d3/d3 (.911) on BFCL with the depth-aware model. Prefill is compute-bound and batches well; decode is memory-bound, so this is a serving configuration no one has reported for Ouro [J]. Also measure mixed-depth KV-reuse accuracy (G4) with the same run.

**D. Reproduce 2608.18171's NESTful protocol with your adapters (~2 h).** Closes G1 and gives you a credible comparison row.

**E. Backbone spike on Dory 1.0 4B (2 days, PyTorch-MPS, no LoRA).** Per-request loop oracle (1/2/3/4) on BFCL multi-turn and your MathQA-val/NESTful multi-turn protocol. A model trained with mixed loop counts has no "one-loop tax", so the question becomes purely allocation, and multi-turn BFCL (800 stateful cases) becomes usable. Kill if MPS port fails in 2 days or the per-request oracle gap at equal accuracy is < 20% loops.

**F. If depth-aware is a headline: the missing ablations and competitors (~2 × 11 h + evals).** (Ouro weights, frozen gate), (uniform, trained gate), plus a cross-loop distillation baseline (LoopOPD/D-LoopOPD 2610.10623; FlexLoop 2609.34488 adjacent-depth distillation), and LoopCD at halved loops. Your `--distill` flag already exists (α=0) [V].

Not recommended: Huginn (base-only, slow, no tool ability), retrofits (single-digit NESTful in 2608.18171), per-turn RL on Ouro weights (GRPO collapse, RLTT), more synthetic-chain RL.

---

## 4. Ranked next moves

| # | Move | Payoff | Cost | Kill criterion |
|---|---|---|---|---|
| 1 | Oracle kill test: sampling control (A) + graded/fixed stop turn (G3) + prefill-inclusive cost (G4) on the existing 60 + 60 more | Decides the headline | ~1 lab day | If the sampling oracle matches the per-call oracle within ±2 requests, or the loops advantage of per-call over per-request on commonly solved requests drops below 10% once stop turn + prefill are counted → drop learned per-turn allocation as a headline |
| 2 | Reproduce 2608.18171's NESTful protocol (D) with ouro-full/depth-full | Validates the whole NESTful line | ~2 h | If you reach ~.3–.37, your "ceiling" was the prompt; re-plan chain v3 as an increment, not a rescue |
| 3 | Verifier-triggered escalation (B) on 300 MathQA-val tasks with chain v1/v2 (no new training) | Cheapest positive per-turn result | ~1 lab day | Not above the fixed-depth frontier at matched total loops on ≥300 tasks → per-turn is dead for this model |
| 4 | Chain v3 SFT, but evaluated on ≥500 val tasks, both NESTful protocols, with the Hermes-only reference kept, and a second seed later | Capability | ~15 h + eval | NESTful multi-turn d4 not above .30 (≈ the base few-shot/2608.18171 level) → stop adding MathQA data; the limit is elsewhere |
| 5 | Headline-dependent: either the anytime paper (F + serving with LoopSpec/LoopCD baselines, 3 seeds, Ouro-1.4B) or Dory spike (E) | Paper | 4–5 lab days / 2 days | — |

**Stop doing:** REINFORCE on synthetic chains (no signal, verified); loops/token as the cost metric; per-request routers on single-turn BFCL (correctly parked); writing "late steps need little compute" until the stop-turn artifact is removed.

---

## 5. Sharper framing I would defend

**Pitch (if move 1 kills per-turn, which I expect [J]):** "Train once, serve at any depth. A looped tool-calling LM should be an *anytime* model: one fine-tune, every loop count a valid operating point, depth chosen per request by the server. We show that a uniform per-loop objective closes the shallow-depth gap of the published recipe (+6–8 BFCL points at one loop across seeds, no loss at four, more monotone), that Ouro's per-token exit loses its savings under batched serving while per-request depth keeps 1.3–2× throughput at matched accuracy, and that on multi-step chains the headroom for per-step loop allocation is bounded: an outcome oracle over per-turn depths beats the per-request oracle only through search, and a verifier-triggered escalation rule recovers [X]% of it without training." Honest, measurable, and each claim already has most of its evidence.

**Pitch (if move 1 and 3 survive):** "Tool execution as the critic for latent depth: per-turn loop escalation in looped agents" — with B as the training-free method and a learned predictor as the refinement, framed against Ares/TAB (tokens) and LoopRPT/LatentHarness (per token, weights changed).

**Mandatory baselines either way:** fixed depths 1–5; 2608.18171 Ouro loss; matched-histogram random allocation (2607.18553); Ouro gate q-sweep; confidence/margin/hidden-cosine readouts (2607.20519); kNN router (2505.12601); LoopCD at halved loops; LoopSpec (batch 1, lossless) for serving; best-of-N sampling at fixed depth for every oracle; prefill-inclusive FLOPs and measured tok/s at batch 1–16.

**Novelty answer for the professor [J]:** per-request/per-turn loop count for tool use is still an open combination, but its two neighbours (Ares/TAB per-step budgets; LoopRPT/LatentHarness/TaH2 per-token loop policies) are close and Dory 1.0 shows industry shipping elastic-loop agentic models. A positive per-turn result needs the controls above to be believed; the anytime-training + serving story is less novel in method but far more defensible with what you have measured.

Key files: `docs/project/progress.md` (2026-10-17 → 10-21), `scripts/lab/ouro_percall_oracle.py`, `scripts/lab/ouro_depth_policy.py`, `src/dht/toolcall/depth_policy.py`, `src/dht/toolcall/mathqa.py` (`render_dialogue`), `scripts/lab/ouro_lora_sft.py` (`loss_fn`), `results/ouro_percall_oracle/chain-depth/examples.jsonl`, `results/mathqa_calib/.../examples.jsonl`, `results/depth_policy/chain-depth/eval.jsonl`.

Sources checked today: arXiv 2609.17184 (LoopSpec), 2610.02185 (LoopCD), 2608.18171, 2610.12582 (ELR), 2609.39740 (LatentHarness), 2603.07915 (Ares), 2610.15993 (Handoff Curves), 2609.35748 + HF `nics-efc/TaH2-*` (TaH2 weights), HF `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-{BF16,GGUF}`.
