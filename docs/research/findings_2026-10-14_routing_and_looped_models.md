# What we learned: layer routing on hybrid LLMs, the 2026 literature, and looped models

Written 2026-10-14. Covers the work since the 2026-09-23 progress meeting. Results are measured
on the ai-lab Mac Studio (MLX, pinned code/data/model revisions; see `scripts/lab/`).
Literature statements come from searches on 2026-10-13/14 and should be rechecked before
writing. Presentation version: `reports/progress_2026-10-14/progress_update_2026-10-14.html`.

## 1. Setup that now exists

- Models: Qwen3.5-0.8B/4B/9B/27B and Qwen3.8-27B. All use the `qwen3_5` hybrid layout:
  repeating groups of [3 Gated DeltaNet + 1 gated full-attention] layers, each with an FFN
  (24/32/32/64/64 layers = 6/8/8/16/16 groups). The 4B and 9B have the same depth; the 9B is wider.
- Data: xLAM function-calling 60k (rev 26d14eb; seeded train/val split) and BFCL v3 (rev 61fc060),
  which is held out as the final test.
- `src/dht/toolcall/`: Qwen `qwen3_coder` tool-call parser, an independent BFCL-style AST checker
  (it reproduces BFCL answer keys on 835 of 840 cases; the other 5 are errors in BFCL's keys),
  and data loaders. `mlx_routing.py` removes layers physically.
- xLAM grading is our design (D019): the reference value is the only accepted value, and optional
  parameters may be omitted or set to their default. It is stricter than BFCL and not comparable
  to leaderboard scores.
- Gotcha: mlx-lm 0.31.3 corrupts batched generation for left-padded prompts. Use ≥ 0.32.0;
  `test_mlx_routing.py` guards this.

## 2. Layer-removal results (frozen models, xLAM validation, greedy decoding)

For every group except group 0, we removed (a) the whole group, (b) only its attention layer,
and (c) only its 3 DeltaNet layers. Accuracy is the raw % of requests correct.

| Model | Full | Best single removal | Best whole group | Two groups (greedy) | n |
|---|---|---|---|---|---|
| 0.8B | 65 | 26 (attn g2) | 16 (g2) | 0 | 100 |
| 4B | 83.5 | 78.5 (attn g1) | 58.5 (g5) | 18.5 | 200 |
| 9B | 84 | 81.5 (attn g5) | 68 (g2) | 16 | 200 |
| 3.8-27B | 83 | ~86 (noise) | ~84 | not run | 100 |

- **27B:** most single-group removals stay at 76–86%. The exceptions are group 8 (60), group 12
  (42) and the last group, 15 (0 / 0 / 11). Verified on separate prompts: skipping really runs
  60 or 63 of 64 layers; the logits change but the output text often does not.
- **Failure mode:** when a removal hurts, the model mostly stops calling tools and repeats text
  until it hits the token limit. Wrong argument values are rare.
- **Anomaly ("removing more hurts less"):** in group 2, 4B scores 56.5 with the whole group
  removed but 8.5 with only the DeltaNet layers removed; 9B scores 68 vs 6. The hypothesis is
  "orphaned attention": the attention layer depends on its DeltaNet feeders (which write
  pre-attention spikes and gate inputs) to abstain. Closest prior work is Pre-Attention Spikes
  (2608.12149). This is untested, and the FFN-vs-mixer decomposition was not run. The effect
  does not appear in the 27B.
- **Oracle (perfect hindsight per request):** matching full accuracy needs ~90% of compute for
  4B and 9B (a 10% saving) and ~96% for 0.8B. With only the best single removal on the menu,
  the saving is about 2–3%. Of 200 requests, 11–12 were solved only by a cut-down version,
  which is consistent with luck. Gate 1 fails: per-request depth routing on frozen hybrids is
  not worth building.
- **Cost facts (Qwen3.8-27B, estimated from config):** FFN ≈64% of weights, DeltaNet ≈21%,
  all 16 attention layers ≈6%, embedding+head ≈9%. One group is 6.25% of decoder compute;
  one attention layer is ≈1.5%.

## 3. Conclusions about the original direction

- DHT's head selection fits BERT (attention ≈26% of weights) but not hybrids (≈6%).
- DHT trains router and network jointly, so the network learns to work with parts missing.
  Released LLMs are frozen and never learned this, so removing layers breaks them except at
  27B scale. A pruned 9B (68) is worse than an unpruned 4B (83.5).
- Speculative decoding with a layer-skipped Qwen3.5 draft is not viable: 2605.01106 reports
  acceptance of 0.038.

## 4. Literature status (2026)

| Direction | Status | Key papers |
|---|---|---|
| Per-request layer routing, frozen LLM | Taken | Dr.LLM (ICLR'26) |
| Bandit routing between models | Taken | RouteLLM, PILOT, MixLLM, BaRP |
| Per-step agent model routing | Taken | AgentRouter (ICML'26), BAAR 2602.21227, DialRouter 2604.12385 (MCTS teacher); Handoff Curves 2610.15993 finds no switching gain on BFCL |
| Per-turn thinking budgets | Taken | TAB 2604.05164 (BFCL multi-turn), Ares 2603.07915 |
| Speculative decoding for tool calls | Taken | ToolSpec 2604.13519, OoO-Spec 2608.00814, SchemaFill 2610.07086; BanditSpec, Learning to Draft |
| Speculative tool execution | Taken | Speculative Actions 2510.04371, PASTE 2603.18897 |
| Why hybrids break under edits | Mostly covered | Pre-Attention Spikes 2608.12149; Qwen3.5 ablation 2603.22473; MiniMax M2 postmortem (HF blog, 2025-10-30; tech report 2605.26494) |
| RL halting for looped LLMs | Partly covered | LoopRPT 2603.19714; RL-Halting 2606.29983 (synthetic only, no compute term) |
| **Bandit / learned per-request loop control at inference, tool use, latency** | **Open (as of 2026-10-14)** | none found among the 167 papers citing Ouro |

MiniMax note: the multi-hop deficit came from lightning hybrids trained from scratch; the
retrofit (CPT to SWA) failure was a separate experiment. They say sinks are not the cause.

## 5. Looped models (the chosen direction)

- **What they are:** one block of layers reused R times, with R chosen at run time. Open
  models: Ouro 1.4B/2.6B (R ≤ 4, learned exit gate, Apache-2.0, MLX port
  GeorgiyDemo/ouro-mlx without early exit); Huginn 3.5B (core looped many times, KL-style
  stopping, no MLX port). Others: Parcae, Mixture-of-Recursions, LoopUS, retrofitted
  recurrence, Koi-4B (ELR 2610.12582; weights not yet checked).
- **LoopRPT (closest):** on Omni-Math text it RL-trains Ouro's exit gate and backbone. The
  per-pass reward is the next-token log-probability gain over an EMA teacher, minus a time
  penalty. It uses 8 noisy latent rollouts and GRPO-style advantages. On Ouro-2.6B hard tokens,
  average passes fall 3.51 → 2.28 and accuracy rises 34.4 → 37.2. Per-token, training-time,
  no tool use. Hyperparameters: lr 5e-6, ρ = 0.2, σ = 0.1, λ_base = 0.02.
- **Looped tool calling (2608.18171):** LoRA SFT (rank 32, Hermes FC v1, 2 epochs, 1×A100).
  Ouro-2.6B scores 86.4 on BFCL non-live single-turn vs 56.7 for Qwen3-4B under the same SFT.
  Gains come mostly by loop 3. Adaptive mode is Ouro's untrained gate with thresholds. There is
  no per-example best-depth analysis and no latency measurement, and multi-turn is left as
  future work.
- **Enabling Dynamic Computation (2610.09013):** Ouro's early exit saves nothing in practice
  because of per-loop KV caches. A "best-available" KV strategy gives −30% FLOPs and KV memory
  at full accuracy, and an objective fix makes exits depend on difficulty. Needed for any real
  latency claim.
- **Other adjacent work:** Monitoring latent reasoning (2610.18127) finds depth use does not
  track difficulty. Diagnosing halting gates (2607.20519) finds confidence readouts are strong
  baselines. T-LoopFormer (token routers), FlexLoop (2609.34488) and Looped Actor (2609.37432)
  are looped RL agents with heuristic stopping; both name learned stopping as future work.

## 6. Proposed project and first test

**Proposed project:** a contextual bandit picks Ouro's loop count (1–4) per request from
first-pass features. The reward is tool-call correctness minus a per-loop cost. Baselines:
Ouro's gate, confidence thresholds, LoopRPT-style RL, and fixed depths at matched compute.
Evaluate on xLAM val, then BFCL single- and multi-turn, with measured MLX latency using the
2610.09013 caching.

**Gate test:** run Ouro-2.6B (probably after LoRA SFT on function calling) at R = 1, 2, 3, 4 on
200 xLAM val examples. Keep the idea only if the best R varies by example and a per-request
oracle beats the best fixed R at matched compute.

## 7. Plan update (2026-10-14, evening)

**Primary direction (unchanged):** a per-request bandit picks the loop count for Ouro.

**Gate result.** Ouro-2.6B, few-shot, 200 xLAM val requests. Accuracy at 1/2/3/4 passes is
26/61/78/82%. A per-request oracle reaches 85.5% at about 1.9 mean passes, versus 82% at a
fixed 4. Smallest-correct passes: 52/75/36/8, with 29 unsolved. Beyond 4 passes accuracy
falls (5: 80, 6: 79, 8: 73 on 100 val), so the action set is {1, 2, 3, 4}. Router training
data covers 1,000 train requests (jobs 875302ad for the first 100, 0cdf6a52 for 100–999).

**Next steps, in order:**
1. Offline predictability. Train correctness predictors for each pass count on train features
   (pass-1 states, gate, entropy and margin per pass, prompt facts). On val, plot accuracy vs.
   mean passes against fixed counts, Ouro's gate, confidence stopping and the oracle.
2. Bandit replay. Run LinUCB and Thompson sampling from `src/dht/bandits` on the logged
   outcomes. Each step reveals only the chosen arm's reward.
3. Dependency-depth axis, the main evaluation structure. Hypothesis: the passes a request
   needs grow with its dependency depth (how many calls must happen before the last one can
   be written), so a per-request controller saves the most on mixed workloads. Run it in
   three stages:
   a. Free first look: compare smallest-correct passes for single-call vs multi-call xLAM
      requests (`n_calls` is already logged). xLAM multi-call requests are mostly parallel,
      so this is only a partial test.
   b. Add chained calls (NESTful, API-Bank higher levels) and measure needed passes by depth.
      Control for prompt length and tool count. Nested output may need light Ouro LoRA SFT
      or tailored few-shot examples (2608.18171 used LoRA SFT on Hermes FC).
   c. BFCL multi-turn with executed simulated tools, which is where the claims are strongest.
4. Latency. Implement early exit with best-available KV caching (2610.09013) and measure
   MLX wall-clock, not just passes.

**Router analysis v2 (job 13176da7).** Passes needed to stay within 1 point of fixed-4:
Ouro gate 3.78, confidence margin 3.72, learned one-shot 3.22 (37% of the oracle saving),
sequential threshold 3.58, sequential gain 3.65; oracle 1.86. Query-text features and CV did
not improve on v1. LinUCB with bandit feedback reaches 74–80% accuracy at 2.6–3.4 passes but
never 81%, so v1's LinUCB figure is withdrawn. Bootstrap 5–95% intervals are wide (learned
one-shot [2.78, 4.00]); 200 val requests cannot separate the methods firmly. The large gap to
the oracle says the pass-1 features miss most of what decides the needed depth.

**NESTful set-up (step 3b).** Math subset only (1,416 of 1,861; 40 pure functions), depth
2–12 (322/492/294/175/74/59 at 2/3/4/5/6/7+). Graded by executing the predicted calls
(NESTful win rate) plus exact match. Prompt: NESTful's 3 ICL examples, few-shot.

**NESTful depth result (job f19ecb14, commit a595031, Ouro-2.6B, 360 requests, ~60 per
depth; analysis `scripts/lab/depth_analysis.py`).** Accuracy at 1/2/3/4 passes is
0.3/3.1/11.7/18.9%; oracle 22.5%; exact sequence match 0/1.1/1.1/4.7%. Much lower than xLAM
(26/61/78/82) because the untuned base model must also do multi-step math. Failures: at pass 1
almost all outputs contain no parsable call (321/360); from pass 2 on the format is right and
most failures are wrong arithmetic plans ("wrong answer", 243/360 at pass 4).

| Depth | n | Acc. at 1/2/3/4 passes (%) | Oracle (%) | Solved | Smallest correct 1/2/3/4 | Mean smallest correct |
|---|---|---|---|---|---|---|
| 1 (xLAM, ref.) | 200 | 26/61/78/82 | 85.5 | 171 | 52/75/36/8 | 2.00 |
| 2 | 61 | 1.6/14.8/27.9/36.1 | 45.9 | 28 | 1/9/12/6 | 2.82 |
| 3 | 60 | 0/3.3/18.3/31.7 | 33.3 | 20 | 0/2/10/8 | 3.30 |
| 4 | 60 | 0/0/8.3/20.0 | 21.7 | 13 | 0/0/5/8 | 3.62 |
| 5 | 60 | 0/0/10.0/13.3 | 20.0 | 12 | 0/0/6/6 | 3.50 |
| 6 | 60 | 0/0/1.7/6.7 | 8.3 | 5 | 0/0/1/4 | 3.80 |
| 7+ | 59 | 0/0/3.4/5.1 | 5.1 | 3 | 0/0/2/1 | 3.33 |

- **Needed passes rise with depth.** Depth 1 (xLAM) 2.00 (1.74 single-call, 2.26 two-call), depth 2 2.82, depth 3 3.30, depth
  4–6 about 3.5–3.8. On solved requests, least squares of smallest-correct passes on
  standardized depth, n_calls and log prompt length gives depth +0.46 per SD (SE 0.19),
  n_calls −0.11 (0.23), prompt length −0.07 (0.08). Logistic regressions of correctness at
  passes 3 and 4 give depth −1.03 (0.42) and −0.80 (0.32) with n_calls near 0. So depth, not
  the number of calls or prompt length, carries the effect, within the limits below.
- **Caveats.** Depth and n_calls correlate at 0.79, so their split is uncertain. Only 81 of
  360 requests are solved at any pass count, and solved deep requests are the easy ones, so
  the smallest-correct means are biased low at depth ≥ 4 and capped at 4 passes. Depth 7+ has
  3 solved requests. NESTful is math, xLAM is not, so the depth-1 vs depth-2 jump also
  contains a domain change; the within-NESTful trend (depth 2 → 4) does not.
- **What it means for the router.** On this set the oracle needs 3.25 passes on solved
  requests, versus 2.00 on xLAM, so a per-request controller has most to gain on mixed
  workloads (single calls and chains together), as hypothesized. Within NESTful alone the room
  is small (fixed 4 gets 18.9 of the oracle's 22.5). Accuracy this low limits the claims;
  LoRA SFT of Ouro on function calling (as in 2608.18171) is the obvious next step to lift
  it, and passes above 4 may matter for deep chains (being tested).

**Ouro-1.4B on xLAM val (job ccc11c05, commit a5d90d0, same probe and 200 requests).**
Ouro-1.4B has the same layer shape as 2.6B (hidden 2048, same FFN) but 24 instead of 48 layers
per pass, so 1.4B at 2k passes costs the same compute per token as 2.6B at k passes.

| Model | 1 / 2 / 3 / 4 passes (%) | Oracle (%) | Smallest correct 1/2/3/4 | Mean on solved (passes / layers) |
|---|---|---|---|---|
| 1.4B | 18 / 43.5 / 58 / 66 | 70.5 | 36/62/28/15 | 2.16 / 52 |
| 2.6B | 26 / 61 / 78 / 82 | 85.5 | 52/75/36/8 | 2.00 / 96 |

- **At matched compute the smaller, more-looped model is as good or better.** 1.4B at 2 passes
  vs 2.6B at 1 (48 layers each): 43.5 vs 26%; per request 48 solved only by 1.4B, 13 only by
  2.6B (sign test p < 1e-5). 1.4B at 4 vs 2.6B at 2 (96 layers): 66 vs 61%; 23 vs 13
  discordant (p ≈ 0.13, not significant). 2.6B's extra layers pay off only beyond 96 layers.
- **Per-request choice across both models:** 88% of requests are solved by some (model,
  passes) pair; the cheapest correct option averages 67 layers on solved requests, against 96
  for the 2.6B oracle alone. Only 5 requests are solved by 1.4B and never by 2.6B. A router over
  (model, passes) is a natural extension of the action set, but model routing is crowded
  prior work (section 4), so it stays secondary to loop-count routing.

**Overnight partial runs (2026-10-15).** At 05:41 the lab's memory broker preempted jobs
8cd5243f and 37599673 (SIGUSR1); the probe has no yield handler, and the lab then refused to
resume them ("code changed after it was approved"). Their saved rows are analysed below; the
remaining ranges (2.6B 570–960, 1.4B 270–360) were not resubmitted because lab automation was
paused at the time.

*2.6B NESTful pooled, 0–360 plus 360–570 (570 requests; 360–960 is meant as router training
data).* Accuracy 0.2/3.3/11.4/19.6%, oracle 23.7%. Mean smallest correct on solved by depth
2/3/4/5/6/7+: 3.04/3.18/3.71/3.52/3.67/3.33 (51/33/21/21/6/3 solved). Controlling for n_calls
and prompt length: least squares on solved, depth +0.29 per SD (SE 0.15), n_calls −0.02
(0.18); logistic at pass 3 and 4, depth −0.94 (0.36) and −0.65 (0.25), n_calls ≈ 0. The depth
trend holds with more data but is weaker than on the first 360 (+0.46).

*Ouro-1.4B NESTful, first 270 of the same order (job 37599673, partial).* Accuracy 0/1.1/3.3/
8.1%, oracle 10.0%, 27 solved (smallest correct 0/3/9/15, mean 3.44). By depth 2/3/4/5/6/7+:
pass-4 accuracy 20/20/4.4/0/2.2/2.2%. Too few solved requests for the depth regression (all
standard errors larger than the estimates). On the 283 requests both models ran, matched
compute favours 1.4B again: 1.4B at 4 passes vs 2.6B at 2 (96 layers each) is 24 vs 10 correct
(21 vs 7 discordant), but 2.6B at 4 passes (192 layers) reaches 56. Unlike xLAM, NESTful's
multi-step math needs the larger model's full depth.

**Passes 5 and 6 on NESTful (job cc487f32, commit b4847a4, same 360 requests as f19ecb14,
merged per request).** Ouro was trained with at most 4 passes; on xLAM accuracy fell beyond 4
(80/79/73 at 5/6/8 on 100 val). On chained calls it does not:

| Depth | n | Acc. at 1/2/3/4/5/6 passes (%) | Solved within 1–4 | Within 1–6 | 4→5 gained/lost | Mean smallest correct (1–6) |
|---|---|---|---|---|---|---|
| 2 | 61 | 1.6/14.8/27.9/36.1/39.3/34.4 | 28 | 31 | +5/−3 | 3.03 |
| 3 | 60 | 0/3.3/18.3/31.7/31.7/28.3 | 20 | 30 | +9/−9 | 3.93 |
| 4 | 60 | 0/0/8.3/20.0/25.0/20.0 | 13 | 20 | +6/−3 | 4.15 |
| 5 | 60 | 0/0/10.0/13.3/16.7/8.3 | 12 | 20 | +7/−5 | 4.20 |
| 6 | 60 | 0/0/1.7/6.7/15.0/11.7 | 5 | 13 | +6/−1 | 4.69 |
| 7+ | 59 | 0/0/3.4/5.1/8.5/1.7 | 3 | 7 | +4/−2 | 4.29 |
| all | 360 | 0.3/3.1/11.7/18.9/22.8/17.5 | 81 | 121 | +37/−23 | 3.88 |

- **A 5th pass helps, mostly for deep chains.** Overall 18.9 → 22.8% (37 requests gained, 23
  lost; sign test p ≈ 0.09). Depth 6 more than doubles (6.7 → 15.0%); depth 2 barely moves.
  Of the 40 requests first solved at pass 5 or 6, 37 have depth ≥ 3. Pass 6 is worse than 5 at
  every depth (12 gained, 31 lost).
- **The per-request oracle over 1–6 passes reaches 33.6%** (vs 22.5% over 1–4 and 22.8% for the
  best fixed count, 5). Mean smallest correct pass count rises almost monotonically with depth,
  3.0 at depth 2 to 4.3–4.7 at depth 6–7+, and passes beyond the trained maximum are where deep
  requests are won. The deep-request gains are small counts; the trend, not each cell, is the
  evidence.
- **For the router:** shallow requests (xLAM, NESTful depth 2) lose accuracy past 4 passes while
  deep chains gain, so no fixed count is right for a mixed workload, and the action set should
  include 5 for chained requests. This is the clearest support so far for a per-request loop
  controller conditioned on the dependency structure of the request.

**Supporting experiment (secondary, requested by the professor): dense vs hybrid.** Run the
same frozen layer-removal map and oracle on dense Qwen3-4B and Qwen3-8B (Qwen3-32B and
Granite 4.2-3B/8B optional), with attention-only vs FFN-only removal as the dense analogue
of the layer-type split. It needs a general layer-skip module for standard decoder layers and
a JSON `<tool_call>` parser. The closest prior work is 2609.17515 (Qwen3-4B vs Qwen3.5-4B/9B
depth pruning on smart-home tool calling, with SFT healing; it found the hybrid more robust).
Our frozen, per-type, per-request version would extend that comparison but is not a
standalone contribution. Per-request skipping on dense LLMs is already covered by FlexiDepth,
Dr.LLM, LayerRoute (2609.13682) and DiffSkip, none of which test tool calling.
