# Looped language models: research landscape (2026-10-15)

Consolidated from six survey slices in this folder (written by research sub-agents, 2026-10-15),
plus the earlier novelty checks in `../findings_2026-10-14_routing_and_looped_models.md`.
About 230 distinct papers were triaged, including 190 papers citing Ouro (arXiv 2510.25741;
a lower bound, Semantic Scholar plus an arXiv full-text supplement). Each slice marks what was
read in full, from the abstract, or only from search snippets. **Recheck any number before
citing it.**

| File | Slice |
|---|---|
| `01_ouro_citations.md` | All 190 papers citing Ouro, categorized; threats; dead ends |
| `02_architectures_models.md` | Looped/recurrent-depth architectures and open models; which we can run |
| `03_halting_adaptive_depth.md` | Halting, early exit, adaptive depth; baselines a controller must beat |
| `04_theory_interpretability.md` | Expressivity, depth/length generalization, overthinking, interpretability |
| `05_agents_tools_rl.md` | Looped models for tools/agents/RL; tool benchmarks with dependency depth; 2608.18171 recipe |
| `06_compute_allocation.md` | Per-request compute allocation outside looped models; methods to borrow |

Our claims, as tested so far (frozen few-shot Ouro, xLAM and NESTful):
C1 needed passes per request grow with dependency depth; C2 Ouro's gate does not track it;
C3 beyond the trained 4 passes shallow requests lose, deep chains gain; C4 a per-request,
structure-aware loop-count controller. Side result: Ouro-1.4B at 2k passes ≥ Ouro-2.6B at k.

## 1. Verdict per direction

| Direction | Status | Decisive papers | What is left |
|---|---|---|---|
| Needed loops grow with chain/hop depth (C1) | **Largely shown** in synthetic or adapted settings | SanSi 2610.07730 (LoRA Ouro; settle loop and gate depth grow with chain depth); 2609.36585 (pointer chains on frozen and LoRA Ouro); 2604.07822 (from scratch, hops); 2608.18171 (tool calling, aggregate curves) | Real tool-call DAGs on frozen generative Ouro, per request, controlled for #calls/length; curve shape (log vs linear) on real data |
| Ouro's gate ignores structure (C2) | **Contested; probably false as stated** | SanSi: frozen gate tracks chain depth (3.6/6.5/7.7 loops) and saves 44%; 2608.18171: gate matches depth-4 on NESTful with fewer iterations; 2607.20519: gate competitive, not Pareto-optimal; 2610.18127: gate fires 3–4 loops late | Only "leaves most per-request oracle headroom on generative tool calls", and only after testing the real per-token gate (our 10% used a one-shot prompt-end proxy) |
| Extra loops past the trained max: hurt shallow, help deep (C3) | **Contested** | Against: SanSi (16 loops = 8), 2609.36585 (frozen Ouro plateaus after loop 3; 12 loops hurt all lengths), 2609.36636 (short chains gain more), Ouro §5.3, STARS, 2610.18127. For: 2610.14456 (theory: an extrapolation window), 2604.07822 (from scratch), 2609.36585 (only after LoRA) | Our pass-5 result is weak (+37/−23, p ≈ 0.09); needs replication and fine-tuned models |
| Per-request loop controller (C4) | **Open combination, but with a direct negative precedent** | 2607.18553: probe- and gate-based per-task loop allocation on frozen Ouro does not beat matched-histogram random allocation; RecurTrace 2609.03379 (sequence-level halting head, retrofitted Qwen3); LoopRPT 2603.19714 (RL exit, per token, fine-tuned); LatentHarness 2609.39740 (per-token policy on Ouro); Dr.LLM, TAB, Ares, BAAR, HAB (non-looped analogues) | Structure-aware features + tool calling + oracle-gap metric + beating matched-histogram random allocation |
| Looped vs dense at matched compute | **Mostly negative for looping** | TaH2 2609.35748 (looped less accurate than dense post-trained baseline at matched compute); 2604.21106 (a recurrence worth ~0.46 of a unique layer); 2605.09165 (only looped-MoE scales better) | Our 1.4B@2k vs 2.6B@k comparison is new but is looped-vs-looped |
| RL on looped LMs | **Known failure mode** | Plain GRPO fails (RLTT 2602.10520); fixes: RLTT, LoopRPT, LoopOPD | Avoid RL on Ouro weights; a separate frozen-model controller sidesteps it |
| Early exit → wall-clock savings | **Known pitfall** | 2610.09013 (per-depth KV caches erase savings; fix loses at batch 32); SanSi (saving 44% → ~25% under static batching); LayerRoute/FlexiDepth (≈1.0x) | Per-request depth avoids ragged per-token KV; must measure latency |

## 2. Key papers (closest to our work)

| arXiv | Title (short) | Setting | Key result | Bears on |
|---|---|---|---|---|
| 2610.07730 | SanSi: looped typed decision model | Ouro-1.4B/2.6B + LoRA (frozen backbone), 8 loops, one-pass classification | Settle loop and frozen gate depth grow with chain depth; gate keeps loop-8 accuracy at 4.5 loops; 16 loops = 8 | C1 (covers), C2 (contradicts), C3 (contradicts) |
| 2609.36585 | Transformers stop thinking too early; tiny LoRA fixes it | 13 base LMs incl. frozen/LoRA Ouro, Huginn; pointer chains | Frozen Ouro reach plateaus at ~2.2 lines after loop 3; LoRA rank 8 extends to ≥160 lines at 8 loops; 12 loops hurt | C1, C3 |
| 2608.18171 | Looped LMs improve compositional tool calling | LoRA SFT Ouro (rank 32, Hermes FC v1), retrofits; BFCL, NESTful, API-Bank | Ouro-2.6B NESTful win .190 base → .371 SFT; gate matches depth 4 with fewer iterations | C1 (aggregate), C2 (tension), C4 (gate only) |
| 2607.18553 | Operational proto-introspection in looped LMs | Frozen Ouro-RLTT/Thinking 2.6B; GSM8K | Per-task loop allocation from probes or gate = random with matched depth histogram | C4 (negative precedent) |
| 2610.15218 | Running fewer loops removes refusal | Frozen Ouro; safety | Gate spends equal loops on harmful/harmless; per-prompt oracle large; label-level router recovers almost none | C2 (supports), C4 (caution) |
| 2609.03379 | RecurTrace | Retrofitted Qwen3-1.7B loop; MathQA | Sequence-level halting head from oracle labels: 56.9% at ~2.0 loops vs best fixed 54.7%; ACT/PonderNet collapse to 1 loop | C4 (closest method) |
| 2603.19714 | LoopRPT | Ouro-2.6B, RL on exit + backbone, per token | Hard split 37.24 at 2.28 steps vs 34.35 at 3.51; −39.5% latency | C4 (strongest trained baseline) |
| 2609.39740 | LatentHarness | Ouro-Thinking, per-token THINK/RECALL/EXIT policy | 55% of full-depth compute, +3.9/+4.1 pts | C4 (per token) |
| 2607.20519 | Diagnosing learned halting gates | Frozen Ouro on MC benchmarks; synthetic MANO | Post-hoc readouts beat gate in 5–8 of 12 settings; 1.24–1.58x measured | C2, baselines |
| 2610.09013 | Enabling dynamic computation in looped LMs | Ouro KV caching | Up to −30% FLOPs/KV; throughput below baseline at batch 32 | Latency |
| 2610.18127 | Monitoring latent reasoning in looped LMs | 8 looped LMs, interpretability | Depth use does not track MATH difficulty; gate fires 3.2–4.1 loops late; 6 loops < 4 | C2, C3 |
| 2610.14456 | Learning to extrapolate in depth with looped transformers | Theory + synthetic graphs | Extrapolation window: too few loops fail, too many create false connections; log growth in hardness | C1 shape, C3 (theory) |
| 2604.07822 | Loop, think & generalize | From scratch, KG hops | Halting iterations rise with hops; more test loops reach deeper hops; overthinking | C1, C3 (synthetic) |
| 2609.36636 | What makes recurrence effective in looped LMs | From scratch, CLUTRR/ProofWriter | Beyond horizon: reasoning +2.97, knowledge −10.7; short chains gain most | C3 (contradicts in part) |
| 2609.35748 | TaH2 | Qwen3 + token-level iteration decider; BFCL v3 | Ouro-style looping less accurate than dense at matched compute | Looped vs dense |
| 2502.17416 | Reasoning with latent thoughts (Saunshi) | Theory + from scratch | T CoT steps ≈ T loops; p-hop needs ⌊log2 p⌋+2 loops | C1 shape |
| 2509.26553 | FuncBenchGen | Tool-call benchmark generator | Dependency depth set independently of #calls and distractors | Controlled-depth benchmark |
| 2606.07587 | Routing plateau | 21 routers incl. kNN | Query-only routers cluster far below oracle | C4 (expect plateau) |
| 2604.05164 / 2606.01168 | TAB / HAB | Non-looped compute allocation | Plan-conditioned budgets (TAB All-SubQ); predict depth then allocate (HAB) | C4 (structure-aware analogues) |

The full tables (about 200 rows) are in the slice files.

## 3. Known dead ends (do not repeat)

1. Per-task loop allocation from frozen hidden-state probes or Ouro's gate does not beat
   random allocation with the same depth histogram (2607.18553). **Any router result must
   include this baseline.**
2. ACT/PonderNet halting added post hoc to a frozen looped backbone collapses to one loop
   (2609.03379). A learned linear per-token router gave no reduction where a training-free
   convergence rule saved 38% (2607.14427).
3. Plain GRPO/DAPO on looped LMs fails (credit only at the final latent state; RLTT 2602.10520).
4. Running past the trained depth on average degrades accuracy, especially knowledge
   (Ouro §5.3, 2609.36636, STARS, 2610.18127); frozen Ouro gains little past loop 3 on pointer
   chains (2609.36585).
5. Early exit rarely gives wall-clock gains without KV changes, and even then can lose at
   batch 32 (2610.09013, SanSi, FlexiDepth, LayerRoute).
6. Query-only routers plateau far below the oracle; kNN often matches learned routers
   (2606.07587, 2505.12601). Failure is predictable; which extra compute fixes it is not.
7. RL routers collapse to one mode without an accuracy gap (2608.20256).
8. Coarse-label routers fail where per-prompt oracles succeed (2610.15218).
9. Upcycling dense → looped is worse than training looped; sharing KV across loops loses gains
   (2607.22083).

## 4. What this means for our project

1. **C1 alone is not a contribution.** The depth → needed-loops relation is already shown on
   (LoRA-adapted) Ouro with controlled synthetic chains (SanSi, 2609.36585). Our real-tool-DAG
   version is a confirmation, not a headline.
2. **C2 must be re-tested before it is claimed.** Our gate number used Ouro's gate value at the
   last prompt token as a request-level proxy. SanSi and 2608.18171 report the real per-token
   gate tracking depth and saving compute. Next: run the real gate during generation on
   xLAM/NESTful and compare at matched accuracy.
3. **C3 is fragile.** One p ≈ 0.09 result against several papers reporting the opposite.
4. **C4 is the only clearly open piece, and it has a known failure mode.** It has to beat
   matched-histogram random allocation (2607.18553), Ouro's real per-token gate, the best
   post-hoc readout (2607.20519), DART-style draft agreement, kNN, and LoopRPT/RecurTrace-style
   learned halting, and report two oracle-gap metrics and latency.
5. **Checked 2026-10-15: our xLAM router does beat matched-histogram random allocation, except
   near full accuracy.** One-shot router (pass-1 features, PCA 8, l2 10, no text features) on 200
   val requests, 2,000 random permutations of its own choices:

   | Mean passes | Router acc. | Random (same histogram) mean / 95th pct | p |
   |---|---|---|---|
   | 3.82 | 81.5 | 81.1 / 82.5 | 0.43 |
   | 3.56 | 80.0 | 79.0 / 81.0 | 0.27 |
   | 2.94 | 78.0 | 74.2 / 76.5 | 0.01 |
   | 2.67 | 77.0 | 71.6 / 74.0 | 0.001 |
   | 2.36 | 73.0 | 66.3 / 69.0 | < 0.001 |

   So, unlike 2607.18553 on GSM8K, the per-request signal is real at moderate budgets (+4 to +7
   points over random at 2.4–2.9 passes), but it does not help where the target is matching
   fixed-4 accuracy, which is the regime of our "passes within 1 point" metric. Report both.
6. **Models for generality:** Ouro-Thinking, Huginn-0125 (PyTorch-MPS), McLeish retrofits
   (`smcleish/…-train-recurrence-{4..32}`), LoopUS Qwen3-1.7B. Koi-4B has no public weights.
