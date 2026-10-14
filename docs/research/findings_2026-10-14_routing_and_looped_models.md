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
