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

## 2026-10-17 — Full LoRA SFT: Ouro loss vs depth-aware loss (job 03ce6df5)

- Job 03ce6df5 (commit f76d0b3, plan `full`): Ouro-2.6B LoRA SFT on Hermes FC v1 (6,378 train /
  708 val conversations, max length 4,096, 2 epochs, 3,188 steps, seed 0), Ouro loss
  (2608.18171) then depth-aware loss, each evaluated with `--adapters` at fixed depth 1–4.
  35.9 h wall time; ouro-full trained in 11.3 h; one preemption at 05:14 (depth-full step 1,498)
  resumed by the lab at the saved step; peak 52.5 GB, no swap.
- Validation CE by loop (100 val conversations), step 0 → 3,188: Ouro loss 0.793/0.525/0.473/0.465
  → 0.221/0.157/0.151/0.150; depth-aware 0.793/0.525/0.473/0.465 → 0.181/0.154/0.150/0.149.
  Depth-aware lowers loop-1 CE (0.181 vs 0.221) with loops 2–4 equal; both curves flatten after
  ~2,000 steps, no rise (no overfitting signal).
- BFCL v3 non-live AST (our checker, not `bfcl-eval`; 400/200/200/200), d1/d2/d3/d4:

  | | Ouro loss | Depth-aware |
  |---|---|---|
  | simple | .728/.920/.938/.938 | .778/.905/.925/.930 |
  | multiple | .665/.895/.940/.915 | .735/.880/.915/.910 |
  | parallel | .590/.870/.885/.900 | .620/.880/.890/.905 |
  | parallel_multiple | .395/.855/.905/.915 | .490/.870/.900/.910 |
  | mean of categories | .594/.885/.917/.917 | .656/.884/.908/.914 |
  | request-weighted | .621/.892/.921/.921 | .680/.888/.911/.917 |

  Base Ouro produced no Hermes-format tool calls.
- xLAM val (200): Ouro loss .570/.755/.830/.815; depth-aware .660/.775/.830/.840 (base, few-shot
  prompt, not the same format: d3 .78, d4 .82).
- NESTful math (first 120 of the depth-balanced order): Ouro loss .000/.025/.067/.117;
  depth-aware .008/.042/.108/.108 (base few-shot on 360: .003/.031/.117/.189; not comparable
  prompts). Hermes data has no nested calls.
- Oracle per-request depth (cheapest correct depth; a ceiling, not a policy): BFCL Ouro loss
  .951 at 1.38 mean loops, depth-aware .940 at 1.30; xLAM .855 at 1.40 / .855 at 1.28; NESTful
  .167 at 1.39 / .175 at 1.34.
- Single training seed, no intervals yet: depth-aware vs Ouro loss differences (+3 to +10 at
  d1, within ~1 point at d2–d4 on BFCL) need the seed runs and a paired bootstrap.
- Models kept on the lab for testing: `~/AI/jobs/dynamic-transformers-sft-full/results/ouro_sft/
  {ouro-full,depth-full}/` (`model.safetensors` merged bf16 for `--weights`,
  `adapters.safetensors` exact for `--adapters`, `config.json`, `log.jsonl`). Local copies of
  adapters, logs and eval rows: `results/ouro_sft_full/` (not committed). Lab cleanup deleted
  resume state and the e2e/pretest leftovers (24.7 → 10.5 GB).
- Analysis of the eval rows (paired over the same requests, 2,000 bootstrap resamples):
  depth-aware minus Ouro loss at d1 BFCL +0.059 [+0.038, +0.080], xLAM +0.090 [+0.050,
  +0.135]; d2–d4 BFCL −0.004/−0.010/−0.004, all CIs include 0 (d3 [−0.021, +0.001]). Depth-aware
  is more monotonic in depth: BFCL d3→d4 lost/gained 3/9 requests vs 14/14; right at a lower
  depth but wrong at d4 2.3% vs 3.0% (xLAM 1.5% vs 4.0%). BFCL errors are wrong arguments, not
  missing calls (no call ≤1.2% at d1, ≤0.3% at d2–d4). NESTful: 0/120 d4 outputs (Ouro loss)
  use `"label"`, the model writes all calls in one turn with self-computed intermediate values;
  multi-turn evaluation is added to the plan before nested SFT data. Training-seed variance
  is not in these intervals.
- 2026-10-18 00:08: seed queue job 635c8cb3 (commit c00b9b5, project dynamic-transformers-sft-seeds)
  started; step-0 val CE equals ouro-full's, confirming the same split.
- Multi-turn NESTful eval (`scripts/lab/ouro_nestful_multiturn.py`, commit 0d7bffb): calls are
  executed per turn and results fed back as Hermes `<tool_response>` turns. Correction to the
  note above: missing labels are not fatal (labels default to positions); the issue is that the
  model writes intermediate values itself. Smoke 59b75235 (depth-full, 3 requests, d4): the
  protocol runs, but in all 3 dialogues the model answers in text after the first tool response
  instead of making the next call (1 turn each; 1/3 correct). Full run 23852b7d: both models,
  sequential variant, first 60 requests, d1–4.
- Multi-turn NESTful result (job 23852b7d, 60 requests, same ids as single-turn), d1–4:
  depth-aware multi-turn .000/.050/.117/.133 vs single-turn .017/.050/.133/.133; Ouro loss
  .000/.017/.167/.150 vs .000/.017/.083/.133. Feeding results back does not change accuracy:
  at d2–d4 98–100% of dialogues end after one turn (the model answers in text after the first
  tool response), with 3.4–5.0 calls vs 7.7 gold calls per request. Ouro loss at d1 loops
  instead (mean 3.1 turns, 30% hit the 8-turn limit, 16.8 calls). d4 accuracy by gold chain
  depth is flat-to-falling (e.g. depth-aware 1/10, 4/10, 0/10, 2/10, 1/10, 0/10 for depths
  2–7+). Conclusion: the Hermes SFT models do not do multi-step chains under either protocol;
  path B needs multi-step training data (the nested-SFT variant), not only a different eval.
  Results: `results/ouro_nestful_mt/` (not committed). Both seed and multi-turn jobs were
  preempted again at ~05:35 (the daily early-morning memory event; 05:14 the day before) and
  resumed by the lab.
- First router on a fine-tuned model (depth-full; job ba545a1c data; `router_sft.py`): trained on
  1,000 xLAM train-pool requests (probe features, 4 logistic predictors, CV l2 = 10), tested
  on BFCL (1,000) and xLAM val (200). Effective passes include the 8-token probe at 4 passes;
  BFCL/xLAM-val output lengths come from the stored text (cut at 600 chars, conservative).
  BFCL: fixed d1–4 .680/.888/.911/.917 (oracle .940 at 1.30); router .911 at 3.38 effective
  passes, .909 at 3.00, .897 at 2.52, .888 at 2.40, i.e. on or below the fixed-depth points
  (fixed-3 .911 at 3, fixed-2 .888 at 2); above matched random by 0–3 points. xLAM val: fixed
  .66/.775/.83/.84; router .840 at 3.42 (matches fixed-4 with ~15% fewer passes), .785 at 2.27,
  .770 at 1.99, roughly on the fixed frontier. Reading: on the fine-tuned model, fixed depth 3
  (or 2) already captures most of the gain, and the probe features do not identify which
  requests are solved at depth 1 (oracle headroom .940 at 1.30 is not captured). Path A as a
  router-vs-fixed-depth result is weak with these features; next: per-feature predictability
  (AUC for correct-at-d1), richer signals, and the Ouro-loss model when its data lands.
  Predictability (depth-full, probe features, trained on xLAM train): AUC for correct-at-d1
  .837 on xLAM val (in-sample .850) but .685 on BFCL (in-sample .720), correct-at-d2 .754 /
  .594. The probe signal transfers poorly across benchmarks and is limited on BFCL even
  in-sample.
- 2026-10-18 evening: routers parked (user decision); focus on the most probable headline.
  Router-data job ba545a1c paused (cancelled; depth-full data complete, ouro-full xLAM labels
  308/4000 rows kept for a later resume). Token-gate eval for fine-tuned models
  (`ouro_token_gate.py --hermes --adapters`, commit 508d228) submitted as job 4c77f09f (BFCL,
  q 0.5/0.9, ouro-full then depth-full).
- Fine-tuned gate, ouro-full (trained gate), first 939 BFCL requests: q 0.5 .931 at 2.64
  nominal passes/token (exits: 0% pass 1, 35.9% pass 2, 63.9% pass 3, 0.3% pass 4); q 0.9 .922
  at 4.00; fixed depth on the same requests .651/.899/.922/.922. Unlike the base model, the
  SFT gate exits early and is accurate. Under lockstep batching most steps should still need
  pass 3 (64% of tokens), so its realised cost is expected near fixed-3; to be measured with
  the serving bench on the merged ouro-full weights (`--weights`, commit 45ff461).
- Seed check: depth-s1 (depth-aware, seed 1) training near done (step 3,136/3,188 at 08:49 Oct
  19, after a preemption at step 2,848).
- Seed 1, Ouro loss (ouro-s1, all 1,000 BFCL): d1–4 .598/.872/.913/.924 vs seed 0
  .621/.892/.921/.921 (seed spread ≤2.3 points). Depth-aware seed 0 minus Ouro seed 1:
  d1 +.082 [+.060, +.105], d2 +.016 [+.002, +.030], d3 −.002, d4 −.007 (both CIs include 0).
  d1 by category, Ouro s0 / s1 / depth-aware s0: simple .728/.715/.777, multiple
  .665/.610/.735, parallel .590/.575/.620, parallel_multiple .395/.375/.490. depth-s1 trained;
  its BFCL eval and ouro-s2 training running.
- 2026-10-19 09:10: user asked for a cooldown and the pivot to chaining. Seed job 635c8cb3
  paused (ouro-s1 trained + evaluated, depth-s1 trained, its eval 24/4000 rows; ouro-s2 just
  started, no state saved). Seed 2 for both losses deferred to paper-writing time. Gate job
  4c77f09f paused after ouro-full finished (depth-full not started).
- Fine-tuned gate, ouro-full, all 1,000 BFCL: q 0.5 .929 at 2.65 nominal passes/token (exits
  0/35.1/64.6/0.3% at passes 1–4); q 0.9 .921 at 4.00 (all at pass 4); fixed .621/.892/.921/.921.
- Chain training data survey: `docs/research/chain_training_data_2026-10-19.md` (Hermes v1 has 0
  call-after-response turns; recommended mix: own executed chains + APIGen-MT + BUTTON).
- User decision: no APIGen-MT (non-commercial); train on our own executed chains so the release
  stays permissive. `dht.toolcall.chains` (27 tools in text/list/date/unit/fictional-lookup
  domains, none named like NESTful's 40; chains of 2–6 executed dependent calls, 20% start with a
  two-call branch) + `scripts/lab/make_chains.py`; 6,000 conversations, median ~1,100 tokens,
  none over 4,096. Trainer: `--init-adapters`, `--extra-data`, `--hermes-train-n`; queue plan
  `chains` (commit 38f1bbc).
- 2026-10-19 10:49: chain job 3b941faa started (depth loss, warm start from depth-full,
  6,000 chains + 2,000 Hermes, 1 epoch = 2,000 steps; step-0 val CE equals depth-full's final
  .1813/.1536/.1499/.1493, confirming the warm start). Serving bench with the ouro-full gate:
  job 9bea1457 at batch 1/4/8 (batch 16 needs ~40 GB, which does not fit next to the training);
  tok/s shared with the training.
- Serving bench with the fine-tuned (ouro-full) gate, job 9bea1457 (NESTful prompts cropped to
  1,024 tokens, 64 forced decode steps, 2 repeats; GPU shared with the chain training, so tok/s
  are indicative, passes_run exact). Gate q0.5 lockstep passes per step: 3.05 (b1), 3.34 (b4),
  3.48 (b8) while each row's own mean exit stays ~3.0–3.05; q0.9 runs 4.00. Decode tok/s at b8:
  fixed-1/2/3/4 173.8/92.6/64.7/46.4, gate q0.5 51.3, q0.9 44.9; at b4: 108.0/56.6/33.7/25.6, gate
  32.3. At b8 the gate is 1.11× fixed-4 and 0.79× fixed-3, and lockstep passes rise with batch
  size toward 4. With BFCL accuracy gate q0.5 .929 vs fixed-3 .921 / fixed-2 .892: in batched
  serving fixed-3 is ~1.26× faster than the gate for −0.8 points, fixed-2 ~1.8× for −3.7 points.
  Caveats: exit statistics come from NESTful-style prompts with forced decoding (per-token mean
  exit 3.05 here vs 2.65 on BFCL answers); clean (unshared) timing and batch 16 still to run.
- depth-s1 BFCL eval submitted (job in project dynamic-transformers-seed-eval).
- 2×2 seeds on all 1,000 BFCL (job 6ab74f6e for depth-s1), d1–4:
  Ouro loss s0 .621/.892/.921/.921, s1 .598/.872/.913/.924 (mean .609±.012/.882±.010/.917±.004/
  .923±.002); depth-aware s0 .680/.888/.911/.917, s1 .683/.891/.913/.917 (mean .681±.002/.889±.002/
  .912±.001/.917±.000). Depth-aware minus Ouro per seed pairing: d1 +.059 [+.038,+.081] and +.085
  [+.060,+.109]; d2 −.004 [−.016,+.008] and +.019 [+.005,+.035]; d3 −.010 and .000; d4 −.004 and
  −.007 (all d3/d4 CIs include 0). The d1 gain holds across seeds and the depth-aware runs are
  also more stable across seeds (half-range ≤.002 vs up to .012). Gate job 4c77f09f resumed for
  depth-full.
- Chain run 3b941faa: training done (2,000 steps); Hermes val CE by loop step 0 → 2,000
  .1813/.1536/.1499/.1493 → .1818/.1540/.1504/.1498 (peak .195 at step 200, recovered); evals
  running (BFCL 778/1,000 requests at 21:55).
- Chain run 3b941faa results (chain-depth = depth-full + 6,000 synthetic chains + 2,000 Hermes):
  * BFCL (mean of categories) .649/.873/.886/.891 vs depth-full .656/.884/.908/.914: simple
    .803/.890/.920/.930 (d4 =), multiple .770/.925/.930/.925 (d4 +1.5), parallel .535/.840/.860/
    .865 (d4 −4.0), parallel_multiple .490/.835/.835/.845 (d4 −6.5). Parallel single-turn calls
    regress (the chain data teaches one dependent call per turn).
  * Single-turn NESTful .008/.042/.058/.042 (vs .008/.042/.108/.108): single-turn scoring sees
    only the first call, and the model now calls one step at a time.
  * Multi-turn NESTful, plain prompt (no sequential hint), n=120: .000/.083/.167/.208; on the first
    60 ids .000/.117/.200/.250 vs depth-full with the sequential hint .000/.050/.117/.133. The
    model now chains: mean turns 3.5–4.2 (was ~1), 13–22% end after one turn (was 98–100%),
    ~4 calls vs 7.5 gold. Accuracy rises with loops at every step and invalid calls fall with
    loops (errors per call .41/.20/.07/.04 at d1–4); 13–30% of dialogues hit the 8-turn limit
    (more at low depth). d4 by gold chain depth 2..7+: 8/20, 7/20, 3/20, 2/20, 3/20, 2/20.
  * Hermes val CE recovered to the stage-1 level (above). Model kept on the lab:
    `~/AI/jobs/dynamic-transformers-chains/results/ouro_sft/chain-depth/` (state/ deleted); local
    results `results/chains_run/` (not committed).
- Depth-full gate (job 4c77f09f, all 1,000 BFCL): q0.5 .912 at 2.04 nominal passes/token (exits
  0/96.4/3.6/0% at passes 1–4); q0.9 .917 at 4.00; fixed .680/.888/.911/.917. With the depth-aware
  loss the gate exits at pass 2 for 96% of tokens and matches fixed-3 accuracy; the Ouro-loss model's
  gate (trained gate) exits at pass 3 for 65% (.929 at 2.65). Under lockstep batching a 96% pass-2
  exit rate should keep far more steps at 2 passes (to be measured with the serving bench on
  depth-full merged weights).
- 2026-10-20 ~04:50, queued (user approved all three): serving bench b0af962b (depth-full gate,
  batch 1–16, clean while alone; plus ouro-full gate at batch 16); chain v2 807a0063 (synthetic mix
  with 20% parallel and 10% single-call conversations, 3,000 Hermes replay, warm start from
  depth-full; commit 6fffdd0); per-call depth oracle 0321c446 (`ouro_percall_oracle.py`, commit
  853eeda; chain-depth v1, 60 NESTful requests; fixed depths, per-request oracle, outcome-greedy
  per-call oracle) as Gate B for the RL policy.
- 2026-10-20 09:20–09:36: the serving bench (peak 60.7 GB at batch 16) and chain v2 (60 GB) were
  admitted together; swap grew 7.4 GB in 10 min and the broker preempted the bench (8/15 runs,
  no resume support). On the user's request chain v2 was stopped (lab cancel = SIGTERM; the
  trainer only saved on SIGUSR1, so its 42 steps were lost) and the per-call oracle runs alone.
  Fix (commit 02dc192): trainer and driver save state on SIGTERM too (stub test: stop at step 7,
  resume from 7). From now on one heavy job at a time: oracle → serving bench (asks 70 GB so nothing
  runs beside it) → chain v2.
- Per-call oracle, first request: every fixed depth fails (d1 stops after 1 turn, d2–d4 loop to
  the 9-turn limit) while the per-call oracle succeeds with depths [4, 4, 4, 1, 1] (5 turns).
- Per-call oracle (job 0321c446, chain-depth v1, first 60 multi-turn NESTful requests, plain
  prompt, 8-call limit): fixed d1–4 .000/.117/.200/.267 at 1/2/3/4 loops/token; per-request
  oracle (cheapest correct fixed depth) .267 at 3.65; per-call outcome-greedy oracle .367 at 3.36
  loops/token (3.32 per turn). 6 requests are solved per call but by no fixed depth; none the
  other way. On the 16 requests both solve, per-call uses 1.99 vs 2.69 loops/token (−26%). Depth
  varies within the dialogue in 20/60; late turns lean shallow (turn ≥5 of correct dialogues:
  depth 1 in 15/30 turns, depth 4 in 1), e.g. [4,4,4,1,1], [2,1,2,1], [4,3,2,1].
  Gate B: passes as an upper bound (higher accuracy at lower cost than the per-request oracle).
  Caveat: the per-call oracle sees future outcomes and tries more trajectories (up to 4 per turn
  with a depth-4 rollout each), so part of the +10 points is search/selection, not depth
  allocation; the cost reduction on commonly solved requests is the cleaner signal. A learned
  policy has no lookahead; its accuracy will be lower than this ceiling.
- Serving bench resubmitted alone (70 GB) at 15:05.
- RL depth policy (path B method): `dht.toolcall.depth_policy` (linear softmax over depths 1–4,
  REINFORCE with batch-mean baseline and entropy bonus; reward 1[correct] − λ(L̄−1)/3; unit test:
  learns shallow-on-easy/deep-on-hard on a toy chain) and `scripts/lab/ouro_depth_policy.py`
  (environment = frozen chain model, per-turn probe features from the 4-loop prefill of the last
  prompt position, cached turns; trains on synthetic chains seed 7; evaluates on 120 NESTful
  requests vs fixed 1–4 and schedules deep-early / decreasing). Commit a300e8c; not yet run on
  the lab before this submission.
- 2026-10-20 ~15:15, user away: queued directly on the lab (no monitor running): chain v2
  63bcd748 (held until the serving bench frees memory), RL policy dd255e86 (`sleep 900` first so
  the bench finishes alone; chain-depth v1, train-n 120, 5 epochs, λ 0.2/0.5, eval-n 120,
  timeout 48h). Serving bench 77731ae3 finishing (47/72 runs at 15:08).

## 2026-10-21 — RL depth policy, chain v2 and clean serving bench

- RL depth policy dd255e86 (commit a300e8c, chain-depth v1, 10.6 h): NESTful multi-turn, plain,
  n=120, accuracy at loops/token: fixed 1–4 .000/.083/.175/.217 at 1/2/3/4; deep-early .158 at
  3.18; decreasing .117 at 2.77; policy λ0.2 .108 at 2.83; policy λ0.5 .117 at 2.74. Both
  policies fall below the fixed-2/fixed-3 line (≈.157 at 2.8) and solve no request that no fixed
  depth solves (deep-early and decreasing each solve 1). No headroom from the per-call oracle
  (.367 on the first 60) is recovered.
- Diagnosis: synthetic training chains are solved at essentially every depth. Mean reward starts
  at .899 (λ0.2) and .748 (λ0.5), exactly 1 − λ·0.5 for 100% accuracy under uniform depths, so
  the only training signal is cost; correctness carries none. Learning was also slow: entropy
  1.386 → 1.345 (λ0.2) and 1.204 (λ0.5) after 40 steps. The learned depth mix on NESTful
  (λ0.2: 52% depth 2, 40% depth 4; λ0.5: 47% depth 3) is not tied to difficulty. Synthetic
  chains are too easy to teach when loops matter; the environment needs tasks where shallow
  depth fails (NESTful-like, disjoint from the test split) or a reward from per-call oracle
  labels.
- Chain v2 63bcd748 (commit 02dc192, 11k rows incl. parallel/single chains + 3k Hermes,
  2,750 steps, 2.9 h train; final val CE by loop .181/.154/.151/.150). BFCL mean of categories
  d1–4 .731/.876/.891/.896 (v1 .649/.873/.886/.891; depth-full .656/.884/.908/.914; earlier text
  quoted depth-full's request-weighted .680/.888/.911/.917 here by mistake); parallel
  .675/.850/.860/.865, parallel_multiple .630/.860/.875/.875: smaller regression than v1, still
  ~2 points under depth-full at d4, best d1 of all runs. Multi-turn NESTful plain d1–4
  .000/.058/.167/.192 (v1 .000/.083/.167/.208): no gain. Turn-limit rate 11/26/12/14% (v1
  30/26/16/13%), errors/call .17/.12/.06/.06 (v1 .41/.20/.07/.04). Lab state/ deleted;
  model.safetensors kept.
- Clean serving bench 77731ae3 (depth-full, alone, ctx 1024, 64 new tokens), tok/s fixed-1/2/3/4
  vs gate q0.5 (passes run / mean row exit): b1 80/41/28/21 vs 33 (2.38/2.38); b4 260/132/88/66
  vs 85 (2.93/2.40); b8 411/207/138/104 vs 121 (3.25/2.44); b16 432/218/146/110 vs 125
  (3.37/2.42). With BFCL (gate .912, fixed-3 .911) the gate is ×1.20 over fixed-3 at batch 1
  and slower at b ≥ 4 (×0.97/0.88/0.86). ouro-full b16: gate 114 tok/s (3.71/3.04) vs fixed-3
  145. Per-request fixed depth keeps its saving under batching; the per-token gate does not.
- MathQA train/val as NESTful-distribution tasks (`dht.toolcall.mathqa`,
  `scripts/lab/make_mathqa.py`): NESTful's math portion is the MathQA test split (paper §3), so
  running its construction on MathQA train/validation (Apache-2.0; original text from AQuA-RAT,
  Apache-2.0) gives training/RL tasks from the test distribution, disjoint by construction.
  Checks: our formula → call conversion equals NESTful's gold for 1,407/1,415 test samples (the
  8 others are NESTful truncating decimals); tool lists follow NESTful's rule (gold + add/
  subtract/multiply/divide/inverse + random math tools, 7–12); NESTful's exact answer filter is
  not reproducible (268 exactly-correct test problems are absent), we keep results within 1% of
  the correct option and ≥2 calls.
  Leakage (user question): exact-text dedup removed 131 + 17 problems, but AQuA-RAT's train set
  holds crowd-sourced rewrites with other numbers: 1,074/1,416 NESTful problems had a train
  problem at number-masked template similarity ≥ 0.7 (798 at ≥ 0.9). Added two filters: template
  similarity ≥ 0.3 to any MathQA-test/NESTful problem (7,947 + 1,207 dropped) and same
  computation skeleton with ≥ 2 shared distinctive numbers (95 + 17). Brute-force recheck: no
  NESTful problem has a task at ≥ 0.4. Result 8,772 train / 1,293 val tasks; mean calls
  5.53 / 5.46 (NESTful 5.06), mean dependency depth 3.74 / 3.75 (3.58). 8,000 multi-turn SFT dialogues (one turn per dependency
  level, literal values). `ouro_nestful_multiturn.py --tasks` evaluates on them. Fixed a crash
  in `nestful._in_range` on integers above float range (now a failed call). Nothing run on the
  lab yet; next is a fixed-depth calibration of chain-depth on 120 validation tasks.
- Calibration c4908a53 (commit d1a29a4; task build on the lab reproduced the local ids hash):
  chain-depth v1, multi-turn plain, first 120 decontaminated MathQA validation tasks
  (depth-balanced), d1–4 .008/.092/.133/.150 (NESTful same model .000/.083/.175/.217). Solved at
  some depth 24/120 (NESTful 29/120); of those, 23 depend on depth (only 1 solved at every depth;
  synthetic chains: ~all at every depth). Most common pattern 0011 (7) as on NESTful (12);
  cheapest correct depth 1/2/3/4: 1/10/8/5. Non-monotone patterns (e.g. 0100, 0110) 6/24.
  Errors/call .35/.06/.04/.04; turn-limit 23/26/13/19%. d4 by gold depth 2..6+: 5/20, 6/20,
  3/20, 2/20, 2/40. Verdict: the environment carries a depth signal, but 80% of tasks are
  unsolved at every depth and give no correctness signal; train the policy on tasks solvable at
  some depth, or raise solvability first (MathQA chain SFT).
- Independent review (`docs/research/review_2026-10-21_independent.md`). Correction to the per-call
  oracle reading above (checked on the rows): the 6 per-call-only wins are [4,4,4,1,1], [4,4,4,1,2],
  [4,3,2,1], [1,2,1], [4,4,4,2,1], [4,4,4,4,4,1], mostly forced depth 4 early plus a late shallow
  turn that changes a failing trajectory, i.e. largely search with ~5x the rollout budget of the
  per-request oracle. "Late turns lean shallow" is mostly the final no-call turn (16/22 correct
  per-call dialogues end with a depth-1 stop turn, which is never graded); first-turn depth on the
  16 commonly solved requests is 2/1/3 for 7/5/4, so "plan deep, execute shallow" is not supported.
  Open gap: single-turn NESTful .11 for our SFT model vs .371 reported by 2608.18171 and .189 for
  our base few-shot prompt; check the protocol before attributing the ceiling to capacity.
- Queued 12:56 (commit 4b6ee90, both ~10–12 GB, side by side): `ouro_nestful_protocols.py`
  (job 778fd9b1; ouro-full then depth-full; single-turn NESTful, first 240, depths 2 and 4;
  prompts hermes / hermes-icl / raw-icl) and per-call oracle v2 (job 1f2f3c8b; chain-depth v1,
  NESTful 120; fixed 1–4, per-call depth oracle, same-budget sampling oracle, best-of-4 sampled
  dialogues, escalation at 1/2/3 and sticky 2; costs with and without the final no-call turn,
  recompute and incremental prefill). The update page and talk track no longer claim that late
  turns need fewer loops.
- Dory 1.0 4B spike (user decision; NVIDIA looped agentic model, released 2026-10-16, OpenMDW-1.1,
  1–4 loops set at run time, no adaptive mechanism; runtime support only via an unmerged llama.cpp
  PR or vLLM PR; `transformers` 5.17 has no `dory`). Route: community MLX port
  (internet-projects/dory-mlx @ 0184a45, Apache-2.0, audited, vendored in `third_party/dory_mlx`;
  upstream parity vs an fp32 PyTorch reference top-1 100%, KL ≤ .003) on NVIDIA's BF16 weights;
  loop count set per call. `dht.toolcall.dory_format` (Dory's XML tool-call format; tests) and
  `scripts/lab/dory_nestful_multiturn.py` (fixed loops, Dory chat template with tools, thinking on,
  NVIDIA sampling, NESTful execution and grading; compute as loop-tokens incl. prefill). Local smoke
  with a tiny random Dory + the real tokenizer/template: loop switching and the tool round trip work.
  Job 14d18849 replaced by bbae7a8c (commit 66eb9da, adds block-token compute: Dory's 8 input and
  8 output blocks run once, so 1/2/3/4 loops = 36/56/76/96 blocks per token): reasoning off first (NESTful 60, MathQA-val 60),
  then reasoning on (same), loops 1–4; reasoning off is the Ouro-comparable setting. Kill criterion:
  4-loop NESTful < ~.35 or no depth dependence.
- Prior art for the depth-aware loss (read 2026-10-21): Popescu, Sáez de Ocáriz Borde & Liò,
  arXiv 2607.20519, "fixed-prior depth supervision": uniform prior π_t = 1/T is our depth-aware
  objective; on synthetic MANO/parity, fixed-prior + post-hoc confidence readouts (margin, entropy,
  top-1) beat learned linear/MLP gates; geometric prior (λ = .3) best; on released Ouro checkpoints
  (not retrained) post-hoc readouts match the gate; latency 1.24–1.58x for the gate on Ouro-1.4B
  (per example, batching not tested). Our contribution narrows to the first test on a pretrained
  looped LLM fine-tune plus serving and per-step control. Update page and talk track now cite it.
  Candidate follow-ups: geometric-prior SFT; post-hoc confidence readouts as controller baselines.
- Planned serving design for per-step depth (user agreed 2026-10-21): new text (prompt, tool
  responses, the previous turn's output) is prefilled with full per-loop KV at 4 loops; generated
  tokens of a turn run at the chosen depth and are cached last-loop only. Basis: Ouro paper §5.4.2
  (Table 14): last-step KV reuse during decoding GSM8K 78.85 vs 78.92 full, MATH-500 80.40 vs 82.40,
  4x less memory; reuse during prefilling costs >10 points on GSM8K. Dory exposes the same idea as
  `recurrent_kv_cache_mode="last_loop"`. Neither was tested with depth changing between turns.
  Experiment (after the oracle controls): mixed per-turn depths with this cache vs full-cache
  recompute, accuracy and measured throughput with depth-bucketed batches, Ouro and Dory.
- Professor's suggestion (2026-10-21, user agreed): MCTS as an offline teacher for intermediate
  states, contextual bandit (greedy or LinUCB/TS, DHT's formulation per tool-call step) at
  inference, escalation rule as the greedy training-free baseline. Plan:
  `docs/research/plan_2026-10-21_mcts_bandit.md`; starts only if oracle v2 shows depth is a lever.

## 2026-10-22 — Oracle v2 (gate 0 fails on Ouro) and Dory reasoning-off calibration

- Per-call oracle v2 (job 1f2f3c8b, chain-depth v1, NESTful 120, 8-turn limit): accuracy at decode
  loops/token: fixed 1–4 .000/.083/.175/.225; per-request depth oracle .242 at 3.72; per-call depth
  oracle .308 at 3.45; **same-budget per-call sampling oracle (3 samples at T 0.7 + greedy, depth 4)
  .325**; best of 4 sampled depth-4 dialogues .308. Depth-only vs sample-only wins 6 vs 8 (31
  both); problems solved by no fixed depth: depth oracle 8, sampling oracle 10, best-of-4 8.
  Escalation (redo the turn at 4 when a call fails) adds nothing: escalate-1/2/3 .017/.083/.175 at
  1.26/2.06/3.02 (= fixed 1/2/3); escalations occur in 41/17/9 of 120 dialogues. Cost: with
  per-turn recompute, decode is 3% of fixed-4 loop-tokens (799 of 23,336); the depth oracle's
  recompute total (24,217) is not below fixed-4's. Verdict (review's kill criterion): the per-call
  oracle's gain is search, not depth allocation; per-step loop allocation on this Ouro model is
  dropped as a headline. The MCTS + bandit plan is not started on Ouro.
- Dory 1.0 4B, reasoning off (job bbae7a8c, paused by the user during the slow reasoning-on stage;
  rows kept, resumable), NVIDIA sampling T 1.0 / top-p .95, one sample per (problem, loops):
  NESTful first 60 multi-turn: loops 1–4 .267/.400/.383/.383 (errors/call .087/.027/.017/.028,
  turn limit 3–5%); MathQA-val first 60: .250/.267/.333/.400. Block-tokens per problem (incl.
  prefill) NESTful 253k/445k/557k/720k. Dory at 1 loop matches the fine-tuned Ouro at 4 loops; on
  NESTful the gain is mostly 1 → 2 loops; on MathQA it keeps rising to 4. Single sampled run, n=60:
  intervals are wide (±~12 points).
- Queued Dory oracle control (job dcc301b2, commit a6b9809, `dory_percall_oracle.py`, NESTful 60,
  reasoning off, greedy base, 3 samples at T 0.7 for the sampling oracle): fixed 1–4, per-call depth
  oracle, same-budget sampling oracle, best of 4 sampled dialogues; block-token costs. Decides
  whether the MCTS + bandit plan moves to Dory. Open Ouro angle noted (not started): depth vs
  width (loops vs majority-voted samples) at matched compute; needs a novelty check.
- NESTful prompt-protocol check (job 778fd9b1, single-turn, first 240, depths 2/4): accuracy at
  d2 / d4 — ouro-full: hermes .013/.158, hermes-icl .046/.204, raw-icl .067/.229; depth-full:
  hermes .029/.138, hermes-icl .021/.217, raw-icl .075/.212. NESTful's three ICL examples add
  ~5–9 points at depth 4 (and switch the model to `$var` references) but no prompt reaches the .371
  reported by 2608.18171; the low NESTful ceiling is mostly the model (or an unreported protocol
  difference), not our prompt. Use the ICL prompt for future single-turn NESTful numbers.
- Dory oracle control (job dcc301b2, NESTful 60, reasoning off, greedy): fixed 1–4
  .267/.350/.383/.350; per-request depth oracle .583 at 2.73; per-call depth oracle .633 at 2.55;
  same-budget sampling oracle .617 at 4.0; best of 4 sampled dialogues .700. Depth-only vs
  sample-only wins 8 vs 7 (30 both); solved by no fixed depth: depth 7, sample 8, best-of-4 10.
  6 problems are solved only at 1 loop. Generation is 4–7% of block-tokens under per-turn
  recompute. Verdict: as on Ouro, per-step depth acts as a source of diversity, not an allocation
  lever; per-step depth control is dropped on both models.
- Free check of width (Dory calibration rows, sampled T 1.0, NESTful 60): majority vote over the
  final answers at loops 1–4 .467 (2–4: .467) vs best single loop count .400, at ~2.5x generation
  cost; oracle any-loop .700. Next candidate: depth vs width (loops x voted samples) at matched
  compute on Ouro and Dory.
- 2026-10-23, Shabbos queue (user away ~30 h, lab idle): one sequential job in project
  `dynamic-transformers-dory` (keeps the paused reasoning-on rows and `results/mathqa/`):
  (1) `depth_width.py --backend dory --n 60` (k=4 sampled dialogues at each of loops 1–4, T .7,
  majority vote at matched block-tokens); (2) the same on Ouro chain v1, NESTful 120;
  (3) `sft_queue.py --plan chains-v3` (chain v2 mix + 8,000 MathQA-train dialogues, depth-aware,
  warm start from depth-full; BFCL, multi-turn NESTful plain, multi-turn MathQA val);
  (4) Dory reasoning-on NESTful 60 (resumes). Steps are separated so a failure moves on. Smoke
  tests: Dory and Ouro, 1 request each; Dory sampling varies with seed (4–6 distinct of 6).
- Depth vs width results (job 65b8f28d, `results/depth_width/`, T .7, majority vote, ties to the
  earliest attempt). Compute = prefill at 4 loops + generation at d, so it is dominated by the
  number of attempts; depth is nearly free under it, not under decode-only cost.
  * Dory (reasoning off, NESTful 60), single sample d1–4 .317/.550/.483/.450. Voting at one loop
    count: d2 x4 .433, d3 x4 .517, d4 x4 .550 (pass@4 .52/.67/.65/.72). A vote over one sample
    at each loop count 1–4 (mixed) .617 at 2.64M block-tokens, below d4 x4's 2.90M: paired
    wins/losses vs d4 x4 7/3 (sign test ~.34, not significant), vs d3 x4 8/2, vs d2 x4 13/2.
    Loop count looks like a useful diversity axis for voting on Dory; needs more requests.
  * Ouro chain v1 (NESTful 120): single d1–4 .017/.042/.100/.150; d4 x3 .208 (= x4), d3 x4 .175;
    mixed .092 (losses 14/0 vs d4 x4): shallow Ouro answers are noise, so depth dominates and
    voting only helps at depth 4 (+6 points for 3x attempts).
- Chain v3 training (same job): 4,746 steps, 13.1 h (one preemption, resumed at step 750); final
  Hermes val CE by loop .181/.154/.150/.149 (v2 .181/.154/.151/.150).
- Chain v3 evals (job finished 2026-10-25 09:27, 42.6 h total; `results/chains_v3_run/`):
  multi-turn NESTful plain d1–4 .158/.383/.400/.475 (v2 .000/.058/.167/.192, v1
  .000/.083/.167/.208); multi-turn MathQA val .167/.367/.417/.458 (chain v1 .008/.092/.133/.150).
  BFCL mean of categories .686/.840/.862/.873 (v2 .731/.876/.891/.896; depth-full
  .656/.884/.908/.914): a 2–4 point single-call regression at d2–d4.
  * Overlap check of the 120 NESTful requests against the 8,772 MathQA-train tasks: max template
    similarity < .3 for all (solved median .100, max .293; unsolved .083/.286); same computation
    with ≥2 shared numbers 0/0. Solved requests more often share a computation skeleton (operation
    tree, numbers ignored) with some training task (28/57 vs 14/63). That is in-distribution
    learning of MathQA problem types (NESTful's math part is the MathQA test split), not
    duplicates. Published NESTful numbers are zero/few-shot, so .475 is not comparable to the
    .371 of 2608.18171 or to zero-shot Dory (.383) as a "beats" claim.
- Dory reasoning on (NESTful 60, sampled T 1.0, max 3,072 tokens per turn): .183/.133/.150/.150,
  ~2,800 reasoning tokens per dialogue and 40–45 truncated turns per loop count. Not a valid
  reasoning-on measurement: the budget truncates the reasoning. Reasoning off stays the
  comparison (.267/.400/.383/.383).
- 2026-10-25, two cheap gates before ~40 GPU hours (plan: per-request router + v4), reviewed by an
  independent Fable agent before queueing:
  * Gate 1 router smoke (`router_smoke.py` + `router_smoke_analysis.py`, pre-registered): depth
    1/2/4 x {greedy, sample} dialogues + an 8-token full-depth probe per request; stability of
    depth need; 5-fold CV logistic router on probe + question features vs question-only vs the
    best random mix (concave hull of fixed accuracies) vs oracle. PASS if gain over the mix at
    2.5 mean loops >= .03 or >= 15% fewer loops than the mix at fixed-4 − .01. Synthetic check:
    signal → PASS (+.125), noise → FAIL; reviewer's null simulation 0/60 false passes.
  * Gate 2 v4 pilot (`sft_queue.py --plan pilot-v4`, `pilot_v4_analysis.py`): continue v3 600
    steps on number-resampled MathQA (`resample_numbers`, `make_mathqa_aug.py`: 6,110 tasks;
    guard dropped 25 exact copies of val/test problems and 15 same-computation matches) vs a
    partial second pass over v3's MathQA data (Hermes share 1/7 in both); full v4 only if the
    aug arm's paired gain (mean of depths 2, 4) has a 90% bootstrap lower bound > 0 and beats the
    second-pass arm.
  * Review blocker, fixed: MathQA-val was filtered against test/NESTful but not against train;
    1,001 of 1,293 val tasks are exact copies, near-duplicates (template similarity >= .3) or
    same computations of train tasks. Only 292 are "clean" (`make_mathqa_clean_val.py`). v3 on
    the first 120 val tasks at d1–4: clean (26) .000/.231/.192/.346 vs other (94)
    .213/.404/.479/.489, so the earlier MathQA-val numbers (.167/.367/.417/.458) are inflated by
    memorised templates. NESTful (filtered against train) is unaffected. Both gates now use the
    292 clean tasks; earlier MathQA-val calibration numbers should not be cited as held-out.
  * Other review fixes: trainer always writes the final adapters; random-mix baseline uses the
    concave hull; percent words in resampling; queue summary includes multi-turn results.
  * Lab: a new commit replaces the project folder (2026-10-17 note; it also happened to
    `dynamic-transformers-dory` in the Shabbos job). v3 adapters + config copied to
    `~/AI/keep/ouro-chain-depth-v3/` on the lab (sha256 2c89ba25…60dc) and locally to
    `results/chains_v3_run/ouro_sft/chain-depth-v3/`; the gates run in a new project folder.
- Gate results (job 6b4cc637, commit 80b8498, project dynamic-transformers-gates; results
  `results/router_smoke/`, `results/pilot_v4/`):
  * Router smoke, 292 clean MathQA-val tasks, v3 greedy d1/2/4 .096/.233/.360 (sampled
    .068/.202/.298). Depth need is a stable property: P(sample ok | greedy ok) .39/.63/.69 vs
    P(sample ok | greedy wrong) .03/.07/.08; cheapest-depth agreement .68 vs chance .43.
    Oracle .404 at 1.62 mean loops. CV probe router at mean loops 1.5/2/2.5/3/3.5:
    .192/.259/.299/.308/.339; question-only .192/.232/.263/.299/.314; random mix
    .164/.233/.265/.296/.328. Gate: gain at 2.5 = +.034 (90% CI +.008 to +.063), PASS on
    criterion (i) by .004; criterion (ii) fails (router needs 4.0 loops to reach fixed-4 − .01,
    mix 3.85). Reading: a real but weak signal, about a quarter of the per-request headroom at
    mid budgets, no savings at the high-accuracy end; question features add nothing.
  * v4 pilot, paired on the same 292 tasks, mean of d2 and d4: v3 .296, aug arm .283 (d2/d4
    .240/.325), partial second pass .318 (.281/.356). aug − v3 −.014 (90% CI −.038 to +.012);
    second pass − v3 +.022 (−.003 to +.048); aug − second pass −.036 (−.062 to −.012). Verdict
    SKIP FULL V4: number-resampled MathQA does not help on clean problems (and is worse than
    more steps on the original data); the remaining errors are unfamiliar problem types, not
    templates. Pilot final val CE by loop: aug .192/.164/.159/.158 vs v3 .181/.154/.150/.149.
- 2026-10-26, mixed-traffic check (offline, v3 BFCL 1,000 + clean multi-step 292): the best
  task-type rule (single-call → 2 loops, multi-step → 4) at 50% single-call traffic gives .610 at
  3.0 loops vs .622 at fixed 4 and .584 for the random mix (+.026; +.016 at 80% single-call); the
  per-request oracle is .654 at 1.42 loops. Task type explains little: single calls also gain
  from 1 → 2 loops (BFCL .702/.846/.879), and most headroom is within task type.
- Probe flaw found: the router smoke's 8 probe tokens were always `<tool_call>\n{"name": "` and
  loops 1–3 matched loop 4 on 100% of them, so the probe never saw the decision tokens.
  Loop speculative decoding (LoopSpec 2609.17184 and others) and cross-loop distillation (LoopOPD
  2610.10623) were checked and are prior work; not pursued.
- Router probe v2 (`router_probe2.py`, `router_probe2_analysis.py`): 48-token probe on the same
  292 tasks, labels reused. Gate: (1) 24-feature router +.03 over the random mix at 2.5 loops
  (no probe cost) with 90% lower bound > 0, and (2) a deployable escalation router (start at 2
  loops, loop-1/2 signals of the first 48 tokens, switch to 4: cost 4·T_4 + 2k) not below the
  random mix at the budget halfway between fixed-2 and fixed-4. Amendments before the lab run,
  disclosed: (a) after a local smoke on the first 60 of the 292 tasks showed the uncached 4-loop
  probe prefill dominating cost (−.25 for all feature sets), the gate moved to a prefix-cached
  accounting; (b) the review showed every task has its own tool list (no shared prefix), so the
  gate moved to the escalation design; (c) the review's escalation cost double-counted the reused
  prompt loops and was corrected. The 60-task smoke showed v2 +.05 / v1 −.02 at 2.5 loops; every
  metric is therefore also reported on tasks 60–291. Labels kept on the lab at
  `~/AI/keep/router_smoke_v3/` (sha256 a2219190…0c42).
- Router probe v2 result (job df8f262b, commit 9e699e1; `results/router_probe2/`): PASS.
  (1) 24-feature decision-token router +.062 over the random mix at 2.5 loops, no probe cost
  (90% CI +.035 to +.087; tasks 60–291 +.055); 8-token v1 +.034, question-only −.002. (2)
  Deployable escalation router (start at 2 loops, loop-1/2 signals of the first ~45 tokens,
  switch to 4) +.029 over the random mix at the mid budget (CI +.010 to +.050; tasks 60–291
  +.017). Its curve: .339 at 0.79x fixed-4 cost (fixed-4 .360; random mix .307 at that cost),
  .325 at 0.74x, .291 at 0.61x; it never reaches fixed-4 accuracy with savings. The full 4-loop
  probe is not deployable without a shared prefix (−.18 uncached); with a shared cached tool
  library +.012. Probe trajectory differs from the labelled d4 first turn on 3.4% of tasks.
  Reading: decision-token signals predict depth need about twice as well as the 8-token probe;
  a deployable escalation policy gains ~3 points at equal compute (~20–25% compute saved for
  2–3.5 points of accuracy), a modest positive result.
- 2026-10-27 pre-registration (before any NESTful router data; `router_confirm.py` docstring):
  the escalation router frozen on the 292 clean tasks (λ from training out-of-fold probabilities
  at 0.75x fixed-4 cost) is applied once to NESTful requests [0, 240) (greedy labels at depths
  1–4; the first 120 reproduce all earlier NESTful numbers). Primary: gain over the best random
  mix of fixed depths at the realised cost, bootstrap 90% CI; CONFIRMED / CONSISTENT / NOT
  CONFIRMED. n raised from 120 to 240 after the review's power estimate (P(CONFIRMED) ~0.44 at
  120, ~0.7 at 240 if the clean-set effect transfers). Step 2 (stronger router, developed only on
  the clean val + the 239-task MathQA-test pool from `make_mathqa_testpool.py`) is tested once on
  the unseen NESTful requests [240, 480), paired against the frozen router.
- NESTful confirmation (job 52705b16, commit 0557977; `results/router_confirm/confirm.json`):
  CONSISTENT. Fixed depths on NESTful [0, 240) d1–4 .171/.383/.442/.492 (first 120 reproduce
  .158/.383/.400/.475 exactly). Frozen escalation router (λ .22): .471 at 0.764x fixed-4 cost,
  escalating 49.6% (training out-of-fold 47.3%); gain over the best random mix at that cost
  +.030 (90% CI −.005 to +.048; clean-set estimate was +.029). Versus fixed 3 loops at the same
  cost (0.769x): +.029 paired (CI −.008 to +.067; 18 wins / 11 losses). Halves: first 120 router
  .475 = fixed-4 at 0.76x cost; requests 120–239 router .467 vs fixed-3 .483 / fixed-4 .508.
  Reading: the effect size transfers almost exactly but n = 240 cannot exclude zero at 90%.
  Test-pool labels for step 2 collected (239 tasks, greedy d1/2/4 + probe).
- Phase A data (job f15ee917, commit 53540bd): 3 extra sampled dialogues per depth (2, 4) and
  loop-1/2 hidden states at the probe tokens for the 531 router-training tasks (292 clean val + 239
  MathQA-test pool). `scripts/lab/phase_a_labels.py`, `scripts/lab/phase_a_router.py`,
  `results/phase_a/`.
  * Label reliability (A2b, pre-registered): greedy d2/d4 .305/.437 (patterns 11/10/01/00 =
    149/13/83/286). Per-depth success is very predictable from 4 samples (AUC .93 at d4, .92 at
    d2), but the benefit of 4 loops is noisy: corr of Δ = P_4 − P_2 between disjoint sample halves
    .25 (Spearman-Brown to 4 samples .40); the 4-sample Δ ranks the greedy "4 helps" label with AUC
    .77, an approximate ceiling for any router (the simulation said ~.75 is needed to match fixed-4
    at 0.76–0.80x).
  * Router development (5x5 CV, development data only), AUC for "4 helps" / gain over the 2-4
    random mix at the mid budget / accuracy at 0.85x / cost to reach fixed-4 − 1 point:
    12 loop-1/2 features .643 / +.026 / .417 / 0.874x; same trained on soft labels (greedy + 4
    samples) .652 / +.034 / .424 / 0.859x; hidden states (nested PCA/logistic, soft labels)
    .664 / +.026 / .422 / 0.872x; fixed-4 .437.
  * LEV-style stopper (agreement of 2- and 4-loop answers, no correctness labels, threshold for
    98% agreement on unlabelled requests): AUC .46, −.020 vs the random mix; at its own threshold it
    escalates 87% for .422 at 0.95x. In generative tool use with ~54% of requests failed at both
    depths, 2- and 4-loop answers disagree on unsolvable requests too, so agreement-based stopping
    escalates them uselessly; correctness supervision is needed.
  * Reading: soft labels help a little; hidden states add ranking (.664) but not policy gain; all
    methods are well below the .77 ceiling and reach fixed-4 − 1 point only at ~0.86x cost.
- Late decision points (A8; job 82abf1ad, commit 4f4cdfa; `scripts/lab/phase_a_trace_analysis.py`,
  `results/phase_a/trace_points.json`): traces reproduce the labelled 2-loop outcomes on 531/531.
  Clean 292 CV, AUC "4 helps" / accuracy at 0.90x / gain over the 2-4 mix: early .644 / .339 /
  +.029; after first tool result .622 / .336 / +.022; after turn 2 .658 / .342 / +.022; end
  (cascade) .694 / .353 / +.022; fixed-4 .360. Pool (trained on clean): early +.031, later points
  +.006 to +.010. Later signals raise ranking slightly but never reach the ~.77 ceiling and the wasted
  2-loop work cancels it; the cascade never beats fixed-4. The pre-registered rule (max accuracy at
  0.90x) picks "end", which fails the objective (below fixed-4, smallest gain); deviation recorded:
  no later point improves on the early router, so the early router (denoised labels) goes to the
  NESTful [240, 720) confirmation. Reading: per-request routing on this task is capped around +3
  points at equal compute / ~14% compute at equal accuracy, because the benefit of extra depth for a
  given request is mostly run-to-run noise.
- 2026-10-29 pre-registration of the step-2 confirmation (`scripts/lab/router_confirm2.py`
  docstring, committed before any NESTful [240, 720) data): frozen early router (12 loop-1/2
  features, soft labels, all 531 MathQA tasks); primary = gain over the depth 1–4 random-mix hull at
  the 0.75x operating point (training out-of-fold rule), bootstrap 90% CI; secondaries:
  non-inferiority vs fixed-4 at 0.85x (margin .03), step-1 router at equal realised cost, fixed-3,
  LEV-style, pooled [0, 720), strata. Fable review: code faithful; corrected the power claim
  (bootstrap SD .0165 at n = 240, not .013) and flagged that [240, 720) is shallower (mean gold
  calls 6.1/4.75 vs 7.8; the depth-balanced order runs out of long chains), where the step-1 gain
  on [0, 240) was +.007 on <= 5-call vs +.042 on longer tasks, so power is ~0.35–0.4; slice kept
  (all remaining deep tasks are in it), strata pre-registered. Smoke run on the seen [0, 240):
  step-1 router reproduces .471 at 0.764x; the new router .467 at 0.74x (+.031 vs mix) and .488 at
  0.85x (fixed-4 .492; paired −.004, CI −.025 to +.017). [0, 240) is not a clean test.
- Step-2 confirmation (job 5d8bb524, commit b9aaf91; `results/router_confirm2/confirm.json`, rows
  kept in `~/AI/keep/router_confirm2_nestful240` on the lab): NOT REPLICATED in substance. Fixed
  d1–4 on NESTful [240, 720) .167/.369/.452/.504 (cost .25/.50/.76/1.0x); perfect 2-vs-4 router
  .544 at 0.59x. Primary (0.75x): router .448 at 0.741x, gain over the mix +.002 (90% CI −.025 to
  +.023); verdict label CONSISTENT only because the point estimate is above zero, which is effectively
  zero. S1: at 0.85x .483 vs fixed-4 .504, paired −.021 (CI −.037 to −.006; 5 wins / 15 losses), so
  not non-inferior and significantly worse than fixed-4. S2: step-1 router +.005 vs mix; new vs
  step-1 at equal cost −.004 (CI −.017 to +.008). S3: vs fixed-3 at 0.75x −.004 (29/31). S4
  LEV-style +.001 (escalates 92%). AUC "4 helps" .58 (development .65). S7: <= 5 calls +.005, > 5
  calls −.003, [240, 480) −.010, [480, 720) +.014, so the [0, 240) long-chain effect did not recur.
  S5 pooled [0, 720) (includes the once-seen slice): +.012 at 0.75x (CI −.010 to +.029), +.016 at
  0.85x (CI +.000 to +.029).
  Reading: early-token routing between 2 and 4 loops does not reliably beat mixing fixed depths
  on held-out NESTful (at ~0.75x it equals fixed 3 loops), despite +.146 oracle headroom. Together
  with the label ceiling (benefit of depth corr .25 between sample halves), the evidence is that
  which request benefits from depth is mostly run-to-run noise, not a property early signals can
  read. Per the pre-registration this closes router tuning on NESTful.

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
- E2E smoke 6150424d (commit a02ac5c, 18 min, exit 0). (1) Depth loss at max length 4,096:
  peak 44.4 GB (lab peak 42.9 GB), 372–376 tok/s, ~14.5 s per optimizer step; 3,188 steps
  ≈ 13 h per full run plus validation. (2) Ouro loss, LR 2e-4, 60 steps, max length 2,048:
  peak 27.8 GB; per-loop train CE swings with the 4-conversation batch (loop 4: 0.18, 0.70,
  0.46, 0.21, 0.36, 0.47 at steps 10–60, the rise at the end with LR ≈ 0 is batch noise);
  val CE at step 60 by loop 0.289/0.213/0.207/0.208 (20 conversations, no step-0 value).
  (3) Merged weights load (strict); Hermes `<tool_call>` outputs appear (base Ouro produced
  none). BFCL, 5 per category, mean over categories by depth 1–4: 0.60/0.95/0.90/0.85.
  NESTful, 6 requests: 0/0/0/1. Pipeline check only, not results (n tiny, 60 steps).
- Trainer follow-ups (after the smoke): validation CE at step 0, 50-step moving averages of
  train loss, and SIGUSR1 yield handling (finish step, save `state/`, re-raise so the lab
  records a preemption; `--resume` continues from that step).
- `scripts/lab/sft_queue.py` runs a whole plan as one lab job (the lab runs at most two jobs and
  has no dependencies): train Ouro loss, train depth loss, each model's eval in the background
  while the next trains; stages skip when their outputs exist. `ouro_sft_eval.py --adapters`
  evaluates the unmerged trained model exactly. Logged Ouro loss now includes β·log 4 so it
  equals Σ p·CE + β·KL(p‖U) (gradients unchanged; earlier logs are lower by 0.139).
- Preemption test a0614797 (commit d5ff9a1): the driver yielded itself twice (steps 29, 59);
  each time the lab recorded "preempted", restarted the job in the same folder a minute later,
  and the trainer resumed at the saved step with the LR schedule continuing; then trained to
  step 60 and the `--adapters` eval ran (BFCL 2 per category, depths 2 and 4). The 05:41
  "code changed after it was approved" refusal did not recur.
- Full queue submitted (plan `full`): ouro-full, then depth-full, 2 epochs each at max length
  4,096; eval per model: BFCL non-live (all 1,000), xLAM val 200, NESTful 120, depths 1–4.
- Plan for the paper paths (serving router vs RL over loops on chains vs both), training seeds
  and questions for the professor: `docs/research/plan_2026-10-17_paper_paths.md`.
- Lab behaviour found 2026-10-17 (job dd6a839b, cancelled): submitting a new commit into an
  existing project replaces the folder, deleting its `results/` (the pretest model was gone,
  so the eval-resume test started a full training run instead). The lab's own auto-resume
  after a preemption keeps the folder. Rule: new commits go to new project folders, and
  results are copied out before any manual resubmit. Eval resume (`load_done`) is tested
  locally only.
- Seed runs (plan `seeds`: ouro-s1, depth-s1, ouro-s2, depth-s2; `--split-seed 0` keeps the
  ouro-full/depth-full split; BFCL eval only) queued to start ~1 h after job 03ce6df5 ends.

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
