# Papers citing Ouro (arXiv 2510.25741): systematic crawl

Crawl date: 2026-10-15. Compiled by a survey sub-agent. Numbers come from abstracts unless the row says "(full text)". Where a full text was read, the relevant section was read through arxiv.org/html/<id>.

## Sources and coverage

| Source | What it gave |
|---|---|
| Semantic Scholar Graph API, `/paper/arXiv:2510.25741/citations?limit=1000` (retrieved after 429 retries) | **175 citing records** in one page with no `next` cursor. 168 have arXiv ids. 7 have none: "Think-at-Hard: Selective Latent Iterations…" (an earlier title of 2511.08577), "T-LoopFormer" (duplicate of 2609.15160), "Beyond Test-Time Training…", "Towards efficient and reliable AI through neuromorphic principles", "LpCoder∞", "Language Models Need Sleep" (probably an earlier version of 2605.26099; not verified), and "WhiteMatter". |
| S2 reported `citationCount` | **Not obtained.** Every call to the paper-metadata endpoint returned HTTP 429. I therefore cannot compare the 175 retrieved records with S2's own total. |
| Supplement: arXiv API keyword search, plus a full-text check of each hit | I ran 6 arXiv queries on looped, recurrent-depth and early-exit terms and filtered to papers submitted on or after 2025-10-30. That gave 290 candidates that S2 did not list. For each one I fetched arxiv.org/html/<id> (or the PDF through pdftotext) and searched for "2510.25741", the Ouro title, or the word "Ouro". **15 confirmed additional citing papers**, which S2 is missing. Among them are LoopRPT, SanSi, Closing the Loop and the CoT-monitorability paper. 264 candidates did not match and 11 PDF-only ones also did not match. |
| **Total triaged** | **190 papers** (175 from S2 and 15 from the supplement). Coverage is a lower bound. S2 clearly lags arXiv by days to weeks: 15 confirmed misses out of 290 candidates. Citing papers that use none of my search keywords would be missed by both sources. Google Scholar was not tried; it blocks scripted access. |

### Counts per category (each paper assigned one primary category)

| Category | S2 list | Supplement | Total |
|---|---|---|---|
| (a) new looped or recurrent architecture, or pretrained model | 57 | 6 | 63 |
| (b) adaptive depth, halting, early exit, efficiency (KV, quantization, decoding) | 37 | 0 | 37 |
| (c) theory or interpretability of looping | 25 | 6 | 31 |
| (d) looped models for reasoning, agents, tool use or RL | 15 | 2 | 17 |
| (e) per-request (sequence-level) compute allocation | 2 | 0 | 2 |
| (f) cited only in passing, or unrelated | 39 | 1 | 40 |
| **Total** | **175** | **15** | **190** |

The primary category is my judgement. Several papers in (b), (c) and (d) contain per-request allocation experiments; these are flagged in the overlap column and ranked in "Threats to novelty". Category (e) is kept strict: the paper's main method must pick one depth per sequence or request.

---

## Threats to novelty (ranked by closeness to C1–C4)

1. **2610.07730 SanSi: A Looped Typed Decision Model for "System 1.5" Thinking** (2026-10-06; S2 misses it; full text read). This is the **closest threat to C1 and the strongest counter-evidence to C2.**
   - Setup: built on Ouro-1.4B and 2.6B with a frozen backbone, rank-64 LoRA and per-loop readouts, trained with 8 loops. Inputs are one-pass typed decisions, not generated text.
   - Depth-controlled tasks (liar chains, object swaps):
     - The loop at which the answer settles grows with chain depth: on liar chains, from 1.0 to 2.3 to 4.2 for increasing depth.
     - **Ouro's frozen exit gate also tracks depth:** on liar chains it stops after 3.6, 6.5 and 7.7 loops at increasing depths; on object swaps after 2.8, 4.8 and 6.2.
     - On unseen chain depths 9–16, one loop gives 49.9% (chance) and eight loops give 72.6%.
   - Mixed suite: the gate keeps loop-8 accuracy (72.0%) with 4.5 loops on average, 44% fewer loops. It runs longer on harder items: 4.2 loops on items all three baselines solve vs 4.7, 4.9 and 4.8 on harder ones. 11.7% of items are right at some loop but wrong at loop 8.
   - **C3 contrast:** running 16 loops, twice the trained 8, gives the same accuracy as 8 on unseen depths ("loops beyond the trained ones do not extend the depth").
   - Differences from us: LoRA-adapted, not frozen few-shot; one-pass classification, not generation; synthetic depth tasks, not tool DAGs; no learned router. Its gate result directly contradicts the framing "Ouro's gate does not track difficulty" unless C2 is scoped to frozen generative tool calling.
2. **2609.36585 Transformers Stop Thinking Too Early, and a Tiny LoRA Fixes It** (2026-09-29; full text read). This is the closest to **C1 and C3 on frozen Ouro.** The task is in-context pointer chains; chain length is literally dependency depth.
   - Frozen Ouro-1.4B: three-chain "reach" is 0, 1.6, 2.3, 2.2 and 2.2 lines after 1–5 loops. Frozen loops add little past loop 3.
   - Two-hop fictional facts on frozen Ouro: 45%, 64%, 78% and 82% over loops 1–4.
   - With a rank-8 LoRA (32,769 parameters), reach is 1.9, 7.1 and 17 lines after 1–3 loops, then about 60 at 4 loops, about 146 at 6 and at least 160 at 8. So **loops beyond the trained 4 extend deep chains**, a C3-like result, but only after a LoRA.
   - At 12 loops accuracy falls at every length (91% at 4 lines, 68% at 64).
   - There is no per-request controller and no tool calling.
3. **2608.18171 Looped LMs Improve Compositional Tool Calling** (already known to us; full text re-read).
   - Same benchmarks as ours (BFCL, NESTful, API-Bank), with LoRA SFT on Hermes FC.
   - Accuracy rises with depth on multi-call and nested tasks.
   - **It claims Ouro's per-token exit gate gives a better compute–performance trade-off on NESTful**: with Ouro-2.6B the gate matches the depth-4 Win Rate with fewer iterations. This is in tension with C2. The difference is likely the metric (token-level average depth vs our per-request oracle-saving share) and the setting (fine-tuned vs frozen few-shot). Our paper must reconcile the two explicitly.
   - It does not stratify by dependency depth, does not go beyond 4 loops, and has no per-request router.
4. **2607.18553 Operational Proto-Introspection in Looped LMs** (2026-07-20; full text read). This is a **direct negative precedent for C4.**
   - Setup: frozen Ouro-RLTT and Ouro-Thinking 2.6B on GSM8K and Horizon Logic.
   - Per-task loop allocation using a hidden-state probe ("tap") and using Ouro's native gate were compared at exactly matched compute. **Neither beats the 95th percentile of 100 random allocations with the same depth histogram.**
   - This null holds even though the pool is depth-sensitive (RLTT fixed-depth success 0.38, 0.84, 0.86, 0.80 at depths 1–4).
   - The paper does not stratify by structural features. A structure-aware router is the obvious way to differentiate from this result.
5. **2610.15218 Running Fewer Loops Removes Refusal Behaviour** (2026-10-10; full text read). This is a per-request depth negative result in the safety domain.
   - Ouro's exit rules spend nearly equal loops on harmful and harmless prompts: 2.439 vs 2.397 mean loops, a difference of at most 0.058.
   - A per-prompt oracle "picks the lowest depth that still produces the right behaviour" and shows a large gain in principle.
   - A router that knows only the harmful/harmless label recovers almost none of it: "the useful information is … which particular prompt needs how much depth".
   - This supports the C2 framing (the gate ignores the property that matters) and warns that coarse-label routers fail.
6. **2609.36636 What Makes Recurrence Effective in Looped LMs?** (2026-09-29; full text read). This **partly contradicts C3.**
   - Models are trained from scratch (Llama-3.1-1B and Qwen3-0.6B shapes; Ouro-style variants). Running beyond the training horizon improves reasoning (28.52 to 31.49) but degrades knowledge (62.80 to 52.11).
   - On CLUTRR, **short relation chains gain most (+31.6 at length 2, +23.4 at length 3), while long chains (4–10) gain less or decline.** On ProofWriter, deeper proofs (3–5) gain about 7.5 pp.
   - Their conclusion: "harder instances do not consistently benefit more." This is the opposite direction from our pass-5 result on one benchmark. C3 should be scoped to frozen Ouro on tool DAGs.
7. **2609.03379 RecurTrace** (2026-09-03; full text read). This is the **closest method to C4.**
   - It uses a sequence-level halting head (one depth per example) distilled from an oracle that asks "does a deeper loop still lower loss".
   - Results on MathQA with a retrofitted Qwen3-1.7B loop: 56.9% at about 2.0 loops vs the best fixed depth at 54.7%.
   - Dead ends on a frozen backbone: ACT and PonderNet halting collapse to one loop across all regularisation settings. CALM gets 54.1% at 5.6 loops.
   - It is not Ouro, not tool use, and does not use structural features.
8. **2609.35748 TaH2: Improving Test-Time Scaling with Adaptive Looped Transformers** (2026-09-28; full text read).
   - A token-level iteration decider trained with online "does iterating help" labels. Evaluation includes BFCL v3 tool use (Qwen3 post-training).
   - It reports that Ouro-style fixed-depth and gated looping have steeper scaling slopes but "remain less accurate" than the non-looped baseline at matched compute. "Ouro's adaptive exit gate improves compute efficiency, but not enough to surpass Standard."
   - On AIME the slope is 2.74 vs 1.79.
9. **2609.39740 LatentHarness** (2026-09-30; full text read). A learned latent-action policy (THINK, RECALL or EXIT per position) on **Ouro-1.4B/2.6B-Thinking**, trained by counterfactual policy distillation. It uses 55% of full-depth compute and adds 3.9 / 4.1 points on a six-suite average over the strongest baseline. It is per-token rather than per-request, and there is no tool use.
10. **2607.20519 Diagnosing learned halting gates** (already known) and **2610.09013 Enabling dynamic computation in looped LMs** (already known). Both bear on C2. 2610.09013 says Ouro's early-exit prior makes "static lower-depth like processing of all tokens regardless of difficulty" and gets up to 30% FLOP and KV savings with best-available KV caching. 2607.20519 finds Ouro's pretrained gates "competitive but not uniformly Pareto-optimal".
11. **2607.14427 Per-Token Fixed-Point Convergence** (2026-07-15; full text read). This is a C4 caution. A training-free "halt when output stabilises" rule matches depth-8 quality at 4.94 average loops (−38%). **A linear router trained on convergence labels from the same model needs about 7.99 loops**, i.e. it gives no reduction. Results are at 135M scale, per token, from scratch.
12. **2602.08864 ANIRA: Understanding Dynamic Compute Allocation in Recurrent Transformers** (full text partly read). This is C1-adjacent and from scratch. Token-level compute allocation "aligned with task complexity can emerge without explicit difficulty supervision" on CLRS, BREVO and other tasks. Early-decision deciders rely on static structural cues (e.g. out-degree, DFS depth); online halting tracks execution state. Allocation does not give extrapolation.
13. Weaker C1-type evidence in other domains:
    - 2609.37432 Looped Actor (already known): on Boxoban, loops increase with remaining pushes.
    - 2607.20594 Convergence selection: halting rule T* = ceil(n/v) on group word problems; more test-time loops rescue late positions.
    - 2602.07845 RD-VLA: some tasks go from 0% at 1 iteration to over 90% at 4, while simple tasks saturate quickly.
    - 2605.21488 Equilibrium Reasoners: easy Sudoku converges in 1–5 steps, hard ones need far more.
    - 2604.07822 (already known): depth extrapolation with overthinking.

**Not found in any citing paper:** a structural dependency-depth predictor of needed passes in tool calling, controlling for #calls and prompt length (C1 as stated); a bandit or RL per-request loop-count controller for tool use (C4); and the Ouro-1.4B@2k vs Ouro-2.6B@k matched-compute comparison. SanSi compares SanSi-1.4B and 2.6B at their own loop counts but does not report a matched-compute crossing; that is my reading of the full text and not exhaustively checked.

## Known dead ends (reported failures)

- **Per-task loop allocation from frozen hidden-state probes, or from Ouro's native gate, does not beat matched-histogram random allocation** (2607.18553, Ouro-2.6B, GSM8K and Horizon Logic). Steering and minimal LoRA "direction binding" also give no gain there.
- **Label-level routers fail where per-prompt oracles succeed** (2610.15218). Ouro's exit rule spends the same depth on harmful and harmless prompts.
- **A learned linear router for per-token depth gives no compute reduction**, while a training-free convergence rule saves 38% (2607.14427).
- **ACT and PonderNet halting attached post hoc to a frozen looped backbone collapse to one loop** (2609.03379).
- **Ouro's gate cannot by itself realise savings with standard KV caching.** The exit prior homogenises depth (2610.09013). SanSi measures that the gate's saving falls from 44% to about 25% of loops under static batching.
- **Running beyond the trained depth:**
  - Knowledge degrades (2609.36636: 62.80 to 52.11), and long relation chains gain less than short ones.
  - Overthinking (2604.07822; 2511.08577: tokens that are correct after the first pass get revised into errors).
  - STARS (2605.26733) reports performance peaking at some depth and then collapsing.
  - Retention collapses: 22% of answers kept at loop 64 when trained for 32 (2610.19768).
  - SanSi: 16 loops = 8 loops, i.e. no extension.
  - 2609.36585: 12 loops hurt every chain length.
- **Frozen Ouro gains little from extra loops on pointer-chasing**: reach plateaus at about 2.2 lines after loop 3 (2609.36585). The gain appears only after a LoRA.
- **RL on looped LMs:** "standard objectives such as GRPO … have failed" to improve LoopLMs because credit goes only to the final latent state (2602.10520 RLTT). This motivates RLTT, LoopRPT and LoopOPD.
- **Matched-compute looping vs dense:**
  - Looped models are less accurate than a non-looped post-trained baseline at matched test compute (2609.35748).
  - φ = 0.46: a recurrence is worth less than a unique layer; truncated BPTT lowers φ to 0.38 (2604.21106).
  - Dense looped models scale worse than standard models; only Looped-MoE scales better (2605.09165).
- **Conversion choices:**
  - Upcycling a dense model into a looped one is worse than training looped from scratch, and sharing the KV cache across loops loses gains (2607.22083 Nanbeige).
  - Naive block re-application on frozen checkpoints degrades performance (2605.23872).
  - Retrofitted looped models are weak on deeply nested tool workflows (2608.18171).
- **Exit-gate transfer to SSMs:** Ouro's two-stage gate on Looped Mamba "degrades severely" without cache-hole adaptation (2607.10110).
- **Training choices:**
  - Last-loop-only fine-tuning collapses early loops (SanSi: 35.6% at loop 1 vs 58.4%).
  - Randomly dropping LoRA applications without rescaling does not help (2609.34218).
  - Ouro's gate collapses early in from-scratch training without LR warmup or a larger KL (2610.00673).
- **Quantization:** one-step GPTQ is worse than round-to-nearest (RTN) on 5 of 9 looped checkpoints (2609.30820).

---

## Table of relevant papers (categories a–e)

Legend: "scratch" = trained from scratch; "Ouro-ft" = fine-tuned or adapted Ouro; "Ouro-frozen" = frozen released Ouro checkpoints. In the overlap column, "none" means no substantive overlap with C1–C4. "(full text)" means a relevant section was read beyond the abstract. A blank negative-results cell ("—") means the abstract reports none.

### (e) Per-request (sequence-level) compute allocation

| arXiv id | title | date | cat | models | tasks | key finding | overlap C1–C4 | negative results / dead ends |
|---|---|---|---|---|---|---|---|---|
| 2609.03379 | RecurTrace: Adaptive Latent Reasoning with Loop-Time Memory | 2026-09-03 | e | retrofitted loop on frozen Qwen3 0.6–8B | MathQA, other QA | Sequence-level halting head distilled from a "deeper loop lowers loss" oracle; 56.9% at about 2.0 loops vs best fixed 54.7% (full text) | C4 medium-high (per-example depth controller; not Ouro, not tools, no structure features) | ACT and PonderNet collapse to 1 loop; CALM 54.1% at 5.6 loops |
| 2609.39740 | LatentHarness: Learning Latent Actions for Memory and Reasoning via Counterfactual Policy Distillation | 2026-09-30 | e | Ouro-1.4B/2.6B-Thinking (ft) | general + long-context reasoning (MATH500, GPQA, multi-hop QA) | Learned THINK/RECALL/EXIT policy replaces Ouro's exit; +2.8% / +10.0% relative; 55% of full-depth compute; 5.9x faster than long-context baseline (full text) | C4 medium (learned controller on Ouro; per position, not per request; no tools) | — |

### (d) Looped models for reasoning, agents, tool use or RL

| arXiv id | title | date | cat | models | tasks | key finding | overlap C1–C4 | negative results / dead ends |
|---|---|---|---|---|---|---|---|---|
| 2610.07730 | SanSi: A Looped Typed Decision Model for System 1.5 Thinking | 2026-10-06 | d | Ouro-1.4B/2.6B + LoRA (ft, 8 loops) | 10,027 typed decisions; liar chains; object swaps | +13.5 pts vs same-shape non-looped; gate keeps loop-8 acc. at 4.5 loops; gate depth tracks chain depth (3.6, 6.5, 7.7 loops) (full text) | **C1 high, C2 contradicts, C3 contradicts** (see Threats #1) | 16 loops = 8 loops; last-loop-only training collapses early loops; static batching cuts gate saving to about 25% |
| 2608.18171 | Looped Language Models Improve Compositional Tool Calling | 2026-08-17 | d | Ouro-1.4B/2.6B LoRA-SFT; retrofitted Llama-3.2-1B, OLMo-2-1B | BFCL, NESTful, API-Bank | Depth helps multi-call/nested; Ouro gate gives better compute trade-off (full text) | C1 partial, C2 tension, C4 none (known) | retrofitted models weak on deep nesting |
| 2607.22083 | Nanbeige4.2-3B: Unlocking Agentic Capabilities in a Compact Model | 2026-07-24 | d | Nanbeige4.2-3B, looped from scratch on 28T tokens (2 passes) | code-agent, office-agent, tool use | Beats Qwen3.5-9B and Gemma4-12B on agentic benchmarks (full text) | low (looped agentic model; fixed 2 passes; no depth analysis). A possible extra testbed | upcycling worse than scratch; >2 passes marginal and unstable; KV sharing loses gains |
| 2604.25917 | Recursive Multi-Agent Systems | 2026-04-28 | d | Qwen3/3.5, Llama-3, Gemma3, Mistral agents; Ouro-2.6B baseline | 9 benchmarks incl. tool-integrated deliberation | +8.3% avg acc; 1.2–2.4x speedup; 34.6–75.6% fewer tokens; tool-caller +4.8% (full text) | none (recursion across agents) | — |
| 2602.10520 | RLTT: Rewarding Latent Thought Trajectories in Looped LMs | 2026-02-11 | d | Ouro-1.4B/2.6B-Thinking (RL) | MATH-500, AIME24/26, BeyondAIME | +5.8% (1.4B) / +10.9% (2.6B) over GRPO | none (RL training, not depth control) | GRPO on LoopLMs "failed" |
| 2603.19714 | LoopRPT: Reinforcement Pre-Training for Looped Language Models | 2026-03-20 | d | Ouro (several scales) | general benchmarks | RL on latent steps; Pareto accuracy-compute; gains on hard tokens (S2 misses it; already known) | low (compresses reasoning into fewer iterations) | — |
| 2610.10623 | Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation (LoopOPD) | 2026-10-07 | d | Ouro-Thinking | math, general, code | Terminal-loop teacher distils into intermediate loops; improves math and transfers | low (makes shallow loops better, which changes the needed-pass distribution) | reward-based supervision sparse/costly (motivation) |
| 2609.37432 | Looped Actor | 2026-09-29 | d | scratch transformer policies | 22 RL tasks | Matches untied baseline with 16x params; Boxoban loops rise with remaining pushes | C1 analogous (known) | — |
| 2609.24196 | LoopCD: Loop-wise Contrastive Decoding | 2026-09-21 | d | recent LoopLMs | reasoning | Contrasting early vs last-loop logits improves reasoning, training-free | none | loop instability creates "hard" tokens |
| 2610.02185 | Decoding Looped Transformers Better for (Almost) Free | 2026-10-01 | d | Ouro-2.6B-Thinking, Huginn, 2 others | AIME24, HumanEval | AIME 61.88 to 73.33%; HumanEval 22.56 to 31.71%; halving loops matches full depth (−22.5 to −48.2% FLOPs) | low (half the loops suffice once decoding is changed, which confounds "needed passes") | — |
| 2609.34218 | Loop Dropout | 2026-09-28 | d | looped LMs + LoRA | math, instruction, code | Stochastic adapter masking with inverse-survival rescale; stronger early-loop adaptation | none | late-loop bias of LoRA; plain random omission does not help |
| 2609.36159 | Principled Thoughts for Latent Recursive LLM Systems (REST) | 2026-09-28 | d | latent single and multi-agent systems | 7 benchmarks | Up to +7.5 pts over CE-only; convergence +30% | none | CE-only training collapses thoughts |
| 2607.18618 | LatentMT: Machine Translation with Latent Reasoning | 2026-07-21 | d | 2.6B LoopLM (likely Ouro; not verified) | 32 translation directions | Comparable to 3–5x larger; gains saturate after early steps | low (saturation with steps) | quick saturation |
| 2602.07845 | Recurrent-Depth VLA | 2026-02-08 | d | RD-VLA action head | robot manipulation | Hard tasks: 0% at 1 iteration to over 90% at 4; simple tasks saturate; adaptive stop by convergence; up to 80x speedup | C1 analogous (task difficulty vs iterations) | — |
| 2605.09948 | LoopVLA | 2026-05-11 | d | recurrent VLA | LIBERO, VLA-Arena | Learned sufficiency score per iteration; −45% params; 1.7x throughput | C4 weak (learned per-step sufficiency) | heuristic early-exit rules criticised |
| 2608.24136 | Steering Recurrent Reasoners with Readout Feedback | 2026-08-25 | d | AKOrN, ItrSA++, TRM | Sudoku, Maze | Readout feedback gains on 4 of 6 model-task pairs | none | 2 of 6 pairs no or small gain |
| 2608.29376 | Evaluating Tiny Recursive Models for Code Generation | 2026-08-29 | d | 28M TRM-AR (scratch) | NL-to-Python | Recovers about 45% of val-loss gap at equal params; at equal depth the larger model wins | none | advantage is resistance to overfitting, not capability |

### (c) Theory or interpretability of looping

| arXiv id | title | date | cat | models | tasks | key finding | overlap C1–C4 | negative results / dead ends |
|---|---|---|---|---|---|---|---|---|
| 2609.36585 | Transformers Stop Thinking Too Early, and a Tiny LoRA Fixes It | 2026-09-29 | c | 13 base LMs, Ouro-1.4B/2.6B (frozen + LoRA), Huginn | pointer chains, fictional multi-hop, MuSiQue | see Threats #2 (full text) | **C1 high, C3 high** | frozen loops plateau; 12 loops hurt |
| 2609.36636 | What Makes Recurrence Effective in Looped Language Models? | 2026-09-29 | c | scratch Llama3.1-1B / Qwen3-0.6B shapes; BaseLoop, LoopFormer, Ouro-style, Huginn-style | knowledge + reasoning (ProofWriter, CLUTRR, BBH) | Beyond horizon: reasoning +2.97, knowledge −10.7; "harder instances do not consistently benefit more" (full text) | **C3 medium (partly contradicts)**, C1 low | knowledge degrades; initial-state injection not robust |
| 2610.15218 | Running Fewer Loops Removes Refusal Behaviour in Recurrent-Depth LMs | 2026-10-10 | c | Ouro (4 checkpoints), Huginn-0125 | GSM8K, harmful/harmless prompts | Half depth: GSM8K 0.775 to 0.692, harmful help 0.038 to 0.205 (5.3x) (full text) | **C2 supportive, C4 cautionary** | gate ignores harm; label router fails; standard safety SFT over-refuses (0.865) |
| 2607.18553 | Operational Proto-Introspection in Looped LMs | 2026-07-20 | c | Ouro-RLTT / Ouro-Thinking 2.6B (frozen) | GSM8K, Horizon Logic | Pre-answer probe AUROC 0.797 vs 0.731 surface; selection gains; allocation null (full text) | **C2/C4 high (negative precedent)** | tap and gate allocation = random; steering negative; LoRA binding null |
| 2610.18127 | Monitoring latent reasoning in looped language models | 2026-10-13 | c | 8 looped LMs (Ouro, LoopUS, Parcae, Rec-OLMo, Huginn) | 5 interpretability lenses | carry-loop vs re-read-loop distinction | low (known) | — |
| 2609.21383 | Prediction Dynamics in Depth-Recurrent Language Models | 2026-09-18 | c | Huginn-3.5B, Ouro-1.4B | answer-scoring tasks | Margin decomposition cuts the mean earliest qualifying depth by 22.5–34.4% of total depth | C2 low (earliest-correct-depth estimators) | retrospective only (needs full trajectories) |
| 2609.01924 | Looped Transformers under the Jacobian Lens | 2026-09-01 | c | Ouro-2.6B, Huginn-0125, Qwen3.6-27B | workspace probes | Ouro rebuilds workspace each loop; Huginn carries it across about 2 recurrences | none | linear transport fails across Ouro loop boundaries |
| 2604.11791 | A Mechanistic Analysis of Looped Reasoning LMs | 2026-04-13 | c | several looped LMs | latent dynamics | Each layer converges to a cyclic fixed point; inference stages repeat per loop | none | — |
| 2604.07822 | Loop, Think, & Generalize | 2026-04-09 | c | scratch recurrent-depth | k-hop composition | Depth extrapolation (5 to 10 hops) by more iterations; overthinking | C1/C3 analogous (known) | overthinking |
| 2607.20594 | When Does Recurrence Become an Algorithm? Convergence Selection | 2026-07-22 | c | scratch weight-tied | group word problems | Frontier speed v ≈ n_train/T_train; halting rule T* = ceil(n/v); extra loops rescue late positions | C1 analogous (needed loops ∝ structure size), C3 analogous | untied models extrapolate worst; S5 wall |
| 2610.19768 | Learning to Compute Is Not Learning to Retain or Reuse | 2026-10-14 | c | scratch looped | Mealy-machine execution | Trained at loop 32, keeps 22% of answers at loop 64; random-position supervision keeps 99.8% | C3 analogous (beyond-horizon loss) | truncated gradients do not fix it |
| 2602.08100 | Emergent Search and Backtracking in Latent Reasoning Models | 2026-02-08 | c | a latent reasoning transformer (model not named in abstract) | MCQA | Backtracking in 32% of instances, +34% acc; implausible distractors shorten exploration by 54% | C1 weak (exploration adapts to difficulty) | — |
| 2601.10242 | Loop as a Bridge | 2026-01-15 | c | looped transformers | knowledge vs output gap | Gap narrows with loops partly via degraded internal knowledge | none | introspection only in final loop |
| 2602.16490 | From Growing to Looping | 2026-02-18 | c | depth-grown + looped models | reasoning primitives | Inference-time looping of middle blocks up to 2x on some primitives | none | — |
| 2602.10097 | Step-resolved data attribution for looped transformers | 2026-02-10 | c | looped GPT-style | algorithmic | Per-iteration TracIn (SDI) | none | — |
| 2508.16745 | Beyond Memorization: Reasoning Depth with Recurrence, Memory and TTC | 2025-08-22 | c | scratch architectures + LLMs | 1D cellular automata | Accuracy drops with required steps; recurrence/ACT help but bounded | C1 analogous | LLMs fail the NL proxy |
| 2509.25239 | A Formal Comparison Between CoT and Latent Thought | 2025-09-25 | c | theory | — | Latent thought more parallel; CoT enables approximate counting/sampling | none | — |
| 2604.17121 | The Topological Trouble With Transformers | 2026-04-18 | c | position/taxonomy | state tracking | Taxonomy of recurrence axes | none | — |
| 2604.21106 | How Much Is One Recurrence Worth? Iso-Depth Scaling Laws | 2026-04-22 | c | scratch looped LMs | pretraining | φ = 0.46; 410M r=4 ≈ 580M non-looped at 1B training cost | low (relevant to 1.4B@2k vs 2.6B@k framing) | truncated BPTT φ = 0.38 |
| 2605.30757 | CoT and Compressed Looped Transformers: Memory-Budget Separation | 2026-05-29 | c | theory | pointer chasing, recall | Compressed loops cannot decide P-complete problems (under an assumption) | none | — |
| 2606.24898 | Dense Supervision Is Not Enough: Readout Blind Spot | 2026-06-12 | c | 44M/129M scratch | LM | Per-loop CE lets hidden norms reach the thousands | none | RMSNorm readout hides scale |
| 2606.18524 | On the Residual Scaling of Looped Transformers | 2026-06-16 | c | scratch | LM | ε = λ/(N√L); LR depends on unique layers only | none | 1/√N scaling insufficient |
| 2606.20737 | Repeated Shared Access Enables Grokking, but Edit Propagation Needs Addressable Memory | 2026-06-17 | c | scratch 2x2 grid | synthetic KG QA | Loop edit propagation 0.04–0.30 vs memory 0.71–0.96 | none | loops alone weak for edits |
| 2608.07261 | Why Knowing Both Hops Is Not Enough | 2026-08-07 | c | scratch | two-hop | Recurrent-style training fixes OOD second hop | none | — |
| 2609.16665 | Right Direction, Wrong Step: Finite-Step Failure | 2026-09-15 | c | two looped families | math, commonsense | Quarter step gives gains in 72.2–83.2% of selected failures | C3 weak (why extra loops hurt) | full-step overshoot |
| 2609.39892 | Shared Weights, Selected Computations | 2026-09-30 | c | scratch graph-walk models | graph walks | Hidden state + attention route which transition a loop performs | none | — |
| 2610.10625 | Safe at One Loop, Risky at Another (SafeBridge) | 2026-10-07 | c | LoopLMs (Ouro-type; not verified) | jailbreak benchmarks | Attack success can rise at deeper depths; SafeBridge reduces it | none | SFT and preference alignment leave depth gaps |
| 2602.00927 | Beyond What Seems Necessary: Hidden Gains from Scaling Training-Time Reasoning Length | 2026-01-31 | c | scratch looped (synthetic) + LLM RL | synthetic, math | OOD keeps improving with more training loops after ID saturates | low | — |
| 2604.09870 | Relational Preference Encoding in Looped Transformer Internal States | 2026-04-10 | c | Ouro-2.6B frozen + heads | HH-RLHF | Erratum: headline 95.2% was an artifact; corrected 63.9% antisymmetrized | none | evaluation leaks and ordering artifacts |
| 2609.25778 | Statistical Gains from Looped Estimation under Parameter Budgets | 2026-09-22 | c | theory | — | Looped sieve MLE attains minimax rates with fixed parameters | none | — |
| 2610.02741 | On the CoT Monitorability of Looped Language Models | 2026-10-02 | c | LoopLMs (Ouro family) | MonitorBench (8 tasks) | Deeper loops reduce monitorability on some stress tests; no systematic loop vs dense gap | none | — |

### (b) Adaptive depth, halting, early exit, efficiency

| arXiv id | title | date | cat | models | tasks | key finding | overlap C1–C4 | negative results / dead ends |
|---|---|---|---|---|---|---|---|---|
| 2607.20519 | Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates | 2026-07-08 | b | synthetic + Ouro-1.4B/2.6B | mod. arithmetic, parity, Ouro evals | Post-hoc confidence readouts match or beat learned gates; Ouro gates competitive, not Pareto-optimal | C2 medium (known) | joint gate training harms the trajectory |
| 2610.09013 | Enabling Dynamic Computation in Looped LMs | 2026-10-06 | b | Ouro + scratch small loops | LM benchmarks | Best-available KV caching −30% FLOPs and KV at full-depth performance; fix to the exit prior | C2 medium (known) | Ouro prior makes depth homogeneous |
| 2609.35748 | Improving Test-Time Scaling with Adaptive Looped Transformers (TaH2) | 2026-09-28 | b | Qwen3-Base post-trained; Ouro/Huginn-style loops | AIME, code, QA, **BFCL** | Slope 2.74 vs 1.79; +3.4 pts at matched compute; depth 2 to 8 gain +2.8 to +3.9 (full text) | C4 medium (token-level decider; includes tool use) | Ouro-style loops less accurate than dense at matched compute |
| 2511.08577 | Think-at-Hard: Dynamic Looped Transformers | 2025-11-11 | b | Qwen3 + LoRA + decider | 9 benchmarks | Oracle iteration policy up to +7.3%; skips 93% of tokens; +3.8–4.4% vs always-iterate | C4 low (token-level) | "latent overthinking": correct tokens revised into errors |
| 2607.14427 | Per-Token Fixed-Point Convergence in Depth-Recurrent Transformers | 2026-07-15 | b | 135M scratch | FineWeb-Edu LM | Convergence halting at 4.94 avg loops = depth-8 quality (−38%) (full text) | C4 cautionary | trained linear router needs about 8 loops |
| 2602.08864 | Understanding Dynamic Compute Allocation in Recurrent Transformers (ANIRA) | 2026-02-09 | b | scratch ANIRA | CLRS, synthetic | Allocation tracks complexity; early deciders use structural cues (full text partial) | C1 medium (token-level, scratch) | no extrapolation to larger inputs |
| 2609.34488 | FlexLoop | 2026-09-28 | b | looped RL policies | 30 goal-conditioned envs | −43% avg depth; 1.34x wall-clock | C4 low (known; state-wise) | pretrained looped policies are depth-specialised |
| 2610.14252 | Adaptive-Depth Is a Finishing Stage | 2026-10-09 | b | 200M/1B scratch | LM | Vary depth only in final phase: −31–45% modeled training compute, NLL within 0.8% | none | — |
| 2606.29983 | Stabilizing Extrapolation via Learned Stochastic Stopping | 2026-06-29 | b | scratch looped | addition, Dyck-1, etc. | Random or RL-learned loop counts reduce OOD variance | C3 analogous | can stabilise a suboptimal computation |
| 2606.18206 | Fixed-Point Reasoners (FPRM) | 2026-06-16 | b | scratch | Sudoku, Maze, ARC | Fixed-point halting adapts compute to difficulty | C1 weak | — |
| 2603.01914 | AdaPonderLM | 2026-03-02 | b | Pythia 70M–2.8B | LM | Token-wise exit; about 10% compute reduction; more compute to high-NLL tokens | C4 low | — |
| 2603.08391 | Adaptive Loops and Memory in Transformers | 2026-03-09 | b | scratch | math, commonsense | Looping helps math, memory helps commonsense | none | — |
| 2602.08220 | Pretraining with Token-Level Adaptive Latent CoT | 2026-02-09 | b | Llama-arch scratch | LM + downstream | Variable-length latent CoT per token; fewer FLOPs | none | — |
| 2602.11451 | LoopFormer: Elastic-Depth Looped Transformers | 2026-02-11 | b | scratch | LM + reasoning | Shortcut-consistency over variable trajectory lengths; budget-conditioned | low | — |
| 2609.15160 | T-LoopFormer: Token-Level Elastic-Depth with Dynamic Routing | 2026-09-14 | b | scratch | LM, zero-shot reasoning | Token-choice routing + recursion-wise KV | C4 low | — |
| 2510.13879 | Catch Your Breath | 2025-10-13 | b | LMs with pause tokens | LM | Per-token adaptive pauses beat CE | none | — |
| 2511.18105 | AdaPerceiver | 2025-11-22 | b | vision | ImageNet, seg., depth | 85.4% at +36% throughput; policy −24–33% FLOPs | none | — |
| 2604.17224 | LASER: Low-Rank Activation SVD for Efficient Recursion | 2026-04-19 | b | TRM | reasoning | About 60% activation-memory savings | none | — |
| 2605.07721 | MELT: Memory-Efficient Looped Transformer | 2026-05-08 | b | Ouro-ft | LM, reasoning | Single gated KV per layer across loops | none | — |
| 2605.16343 | LoopQ | 2026-05-08 | b | 4 looped backbones | LM | W4A4: +67.8% relative accuracy | none | standard PTQ degrades |
| 2605.13190 | N-vium: Mixture-of-Exits | 2026-05-13 | b | up to 1.5B scratch (non-looped) | LM | 57.9% wall-clock speedup, no perplexity cost | none | — |
| 2607.10110 | CHASE: Cache-Hole-Adapted Skip Exit for Looped SSMs | 2026-07-11 | b | Looped Mamba / hybrid | Mano, p-hop, LM | Ouro's 2-stage gate adapted; about half the recurrent steps | none | naive gate transfer degrades severely |
| 2607.15456 | Looped Latent Attention: Cross-Loop KV Compression | 2026-07-16 | b | Ouro-1.4B/2.6B, Huginn | LM, MATH-500 | 21.3x compression; batch capacity 32 to 768; MATH-500 0.43 to 0.66 at 4x | none | — |
| 2608.09444 | Continuous Depth Batching | 2026-08-10 | b | Ouro-1.4B, Huginn-3.5B | serving | Up to 99% of estimated max speedup | C4 systems-relevant (serving variable depth) | non-looped layers complicate scheduling |
| 2609.17184 | LoopSpec | 2026-09-15 | b | several looped | reasoning, coding | Up to 6.83x speedup | none | — |
| 2609.23033 | WaveFront Decoding | 2026-09-19 | b | Ouro-2.6B, Huginn-3.5B | Spec-Bench | 2.42x (Ouro), up to 4.81x (Huginn) | none | — |
| 2609.29812 | FlashLoop | 2026-09-24 | b | several looped | — | 1.64x speedup; 6x KV reduction, lossless | none | — |
| 2609.30820 | Quantizing Looped Transformers | 2026-09-25 | b | 9 checkpoints (Huginn etc.) | — | Accumulated-Hessian GPTQ wins on all 9 | none | one-step GPTQ < RTN on 5/9 |
| 2609.36653 | Scheduling Recursive Reasoning (TAPS) | 2026-09-29 | b | several recurrent | structured reasoning | Adaptive step size; 1.56x speedup | none | — |
| 2609.39277 | A Tilted Bowl Is Not a Slippery Slope: Compressing Looped Models | 2026-09-30 | b | 30+ models, 5 families | Sudoku, Maze | Halting-head + 8-bit finish: up to +15 pts at a third of weight traffic | none | collapse only for non-settling loops |
| 2610.04198 | ALoDLM | 2026-10-03 | b | 1.7B/8B diffusion LM | 11 benchmarks | Token-adaptive recurrence in DLMs | none | — |
| 2610.05265 | Loopy | 2026-10-04 | b | Ouro-1.4B etc. | LAMBADA etc. | W4A4 −36.5% perplexity vs SpinQuant | none | rankings change with depth |
| 2610.07940 | Hybrid Latent Attention | 2026-10-06 | b | Ouro-1.4B/2.6B uptrained | math, knowledge, long-context | KV 10.7x smaller; 2.5–7.4x throughput; over 97% accuracy retained | none | — |
| 2610.10381 | ResidualQuant | 2026-10-07 | b | looped transformers | math, code | −80.7% KV; up to 4.15x peak throughput | none | — |
| 2610.16368 | LoopQuant | 2026-10-12 | b | Ouro-2.6B etc. | — | W4A4 beats W4A8 baselines; −65.3% memory; 1.7x decode | none | — |
| (no id) | Think-at-Hard: Selective Latent Iterations… (earlier S2 record) | 2025 | b | duplicate of 2511.08577 | — | — | — | — |
| (no id) | T-LoopFormer (S2 record without arXiv id) | — | b | duplicate of 2609.15160 | — | — | — | — |

### (a) New looped or recurrent architectures, or pretrained models

None of these overlap substantively with C1–C4. A row is kept in this section when it has a relevant negative result or a model we could test on.

| arXiv id | title | date | cat | models | tasks | key finding | overlap | negative results / dead ends |
|---|---|---|---|---|---|---|---|---|
| 2610.12582 | One Model to Loop Them All: Elastic Latent Recurrence | 2026-10-08 | a | 4B dense, 30B-A3B MoE; Koi-4B | 10-task avg | 1-loop at par; 4 loops +7.25 / +4.75 pts | low (known) | "one-loop tax" in prior models |
| 2610.00673 | Closing the Loop: Practical Training Recipes for Looped LMs | 2026-09-30 | a | 1.4B LoopLM (310B tokens); Looped Qwen3-1.7B | 12 benchmarks | Beats dense twin: GSM8K +14, MATH +10, DROP +22; approaches 3.9B dense at matched inference compute | low (matched-compute framing) | exit gate collapses without warmup / larger KL |
| 2605.20670 | LT2: Linear-Time Looped Transformers | 2026-05-20 | a | LT2 variants; Ouro-hybrid-1.4B (converted) | recall, LM | Converted Ouro-hybrid-1.4B competitive with industry 4B | none | — |
| 2605.11011 | LoopUS | 2026-05-10 | a | pretrained LLM retrofitted | reasoning | Encoder / loop / decoder; confidence head for early exit | low | hidden-state drift |
| 2605.23872 | Training-Free Looped Transformers | 2026-05-22 | a | Qwen3-4B, Qwen3-30B-A3B, Moonlight-16B | MMLU-Pro, CSQA, OBQA | +2.64 / +1.14 / +1.20 pp with damped sub-steps | none | naive block reapplication degrades |
| 2604.12946 | Parcae | 2026-04-14 | a | up to 1.3B scratch | CORE | −6.3% val ppl vs prior looped; test-time saturating exponential | none | residual explosion and loss spikes in prior recipes |
| 2605.09165 | Sparse Layers are Critical to Scaling Looped LMs | 2026-05-09 | a | scratch dense / MoE | LM | Looped-MoE scales; loop boundaries are better exits | none | dense looped scales worse |
| 2607.16051 | Loop the Loopies! | 2026-07-17 | a | 20B-A2B, 6B-A0.6B MoE | reasoning | Beats vanilla at equal compute | none | — |
| 2609.01343 | SMELT | 2026-09-01 | a | up to 54B MoE | LM | 6.8–18.0% training FLOP savings | none | — |
| 2609.40316 | Scaling Laws for Looped MoE | 2026-09-30 | a | MoE | LM | Recurrence about 2x total-parameter efficiency on reasoning | none | — |
| 2606.04438 | LoopMoE | 2026-06-03 | a | 3B / 9B | 9 benchmarks | +1 pt (3B) to about 3 pts (9B) over matched MoE | none | — |
| 2609.35751 | How to Loop MoE (Foil) | 2026-09-28 | a | MoE | LM | Flatten experts, untie attention: −0.012 nat | none | — |
| 2610.19975 | LatentLoop | 2026-10-14 | a | 1.3–7B MoE | 9 benchmarks | Compute savings 19.0% / 7.9% vs MoE / looped baseline | none | — |
| 2605.20613 | HRM-Text | 2026-05-20 | a | 1B scratch | MMLU 60.7, GSM8K 84.5 | 100–900x fewer tokens | none | — |
| 2605.12466 | Attractor Models | 2026-05-12 | a | 770M LM; 27M reasoner | LM; Sudoku-Extreme 91.4%, Maze-Hard 93.1% | Iterations chosen by convergence | C1 weak | — |
| 2605.21488 | Equilibrium Reasoners | 2026-05-20 | a | scratch | Sudoku-Extreme 2.6% to over 99% | Easy cases 1–5 steps, hard cases need many | C1 analogous | — |
| 2610.06833 | Towards Looped Models Done Right, Part II | 2026-10-05 | a | 100M–1.6B | LM | Learned depth prior + orthogonal injection; 3x smaller KV at parity | none | fixed-depth training breaks KV sharing |
| 2610.02383 | Shared Memory in Looped Transformers (LPT) | 2026-10-01 | a | 150M–1B | FineWeb-Edu | −1.12–1.82 ppl, −76–79% context memory | none | — |
| 2609.36108 | LoopICL | 2026-09-28 | a | single-block tabular | TabArena, TALENT | About 90% fewer params, learned exit gate | none | — |

Other (a) papers with no overlap (title, date, model):

- 2512.14531 VersatileFFN (2025-12-16)
- 2601.21582 Dreamer depth-recurrent attention mixtures (2026-01-29; 2–8x fewer tokens)
- 2602.02156 LoopViT (2026-02-02; ARC-AGI-1 65.8%, 18M; entropy-based dynamic exit)
- 2602.09080 RecursiveVLM (2026-02-09; +3% / +7%)
- 2602.11698 SpiralFormer (2026-02-12)
- 2603.23998 Sparse Growing Transformer (2026-03-25)
- 2604.01577 Fast-Slow Latent Recurrence (2026-04-02)
- 2604.18839 Denoising Recursion Models (2026-04-20)
- 2604.19550 LoopCTR (2026-04-21; oracle analysis shows 0.02–0.04 AUC headroom for adaptive inference, which is C4-analogous)
- 2604.21254 Hyperloop Transformers (2026-04-23; about 50% fewer params)
- 2605.06322 SMolLM (2026-05-07)
- 2605.18797 Fully Looped Transformer (2026-05-11; stable to 12 loops, +13.2%)
- 2605.16048 Reshape and Recur (2026-05-15)
- 2605.18464 PERL (2026-05-18)
- 2605.26099 Offline Recurrence / "sleep" (2026-05-25)
- 2605.26106 LoopMDM (2026-05-25; up to +8.5 GSM8K)
- 2605.26733 STARS (2026-05-26; known; performance peaks then collapses with depth)
- 2605.30202 Dual-path (2026-05-28)
- 2606.09357 Recursive-Transformer ASR (2026-06-08)
- 2606.16825 Tied expert layers (2026-06-15)
- 2606.18208 Looped World Models (2026-06-16)
- 2606.31779 LOTUS (2026-06-30; 3B, 2.5–6.9x lower thought latency)
- 2607.00341 DiscoLoop (2026-07-01)
- 2607.10128 Energy-guided Recursive Model (2026-07-11; "deeper recurrence does not necessarily improve")
- 2607.13491 DeepLoop (2026-07-15)
- 2607.15178 T²MLR (2026-07-16)
- 2608.03624 LoopMTP (2026-08-04; +8.1% relative, stable to 15 loops)
- 2608.08888 Full-bandwidth transformer (2026-08-09)
- 2608.15062 Gated Recurrent Transformers (2026-08-15)
- 2608.18230 MixerLoop (2026-08-18)
- 2609.19521 LSTM-UT (2026-09-17)
- 2609.28809 Stream Recursion Model (2026-09-23)
- 2609.37379 Looped Transformers as Optimizers (2026-09-29)
- 2609.37631 Procedural Core (2026-09-29)
- 2609.38426 LoopVL (2026-09-29)
- 2610.00541 Random Recursive Models (2026-09-30)
- 2610.14030 Rotate the Loop (2026-09-30)
- 2610.11472 LLoCoT (2026-10-08)
- 2610.16415 Chain Dimension Scaling (2026-10-12)
- 2607.17843 Mobius Learning (2026-07-20)
- 2607.27656 Source-Centered State Evolution (2026-07-30)
- 2608.17981 Recirculation (2026-08-18)
- No arXiv id (S2 only; not verified): "LpCoder∞: Scaling Code Intelligence via Looped LMs" and "Language Models Need Sleep" (probably 2605.26099).

---

## Category (f): cited only in passing, or unrelated (40)

- 2505.16782: Reasoning Beyond Language: A Comprehensive Survey on Latent CoT Reasoning
- 2510.04573: LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning
- 2511.00907: Transformers as Intrinsic Optimizers: Forward Inference through the Energy Principle
- 2512.10941: Mull-Tokens: Modality-Agnostic Latent Thinking
- 2512.22955: Diversity or Precision? A Deep Dive into Next Token Prediction
- (no id): Beyond Test-Time Training: Learning to Reason via Hardware-Efficient Optimal Control
- 2601.22954: Residual Context Diffusion Language Models
- 2602.01705: LaDi-RL
- 2602.04925: Internalizing LLM Reasoning via Discovery and Replay of Latent Actions (STIR)
- 2602.08783: Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure
- 2602.10229: Latent Thoughts Tuning
- 2602.11683: ThinkRouter
- (no id): Towards efficient and reliable AI through neuromorphic principles
- 2603.09221: Beyond Test-Time Memory: State-Space Optimal Control for LLM Reasoning
- 2603.17837: The Silent Thought (FLAIR)
- 2604.04902: Are Latent Reasoning Models Easily Interpretable?
- 2604.18637: NeuroAI and Beyond
- 2604.17892: LEPO: Latent Reasoning Policy Optimization
- 2604.22951: The Power of Power Law: Asymmetry Enables Compositional Reasoning
- 2605.06546: Efficient Pre-Training with Token Superposition
- 2605.09630: Scratchpad Patching
- 2605.15913: Towards Generalization of Block Attention…
- 2605.16638: TTE-Flash
- 2605.21260: On the Cost and Benefit of Chain of Thought: A Learning-Theoretic Perspective
- 2605.22504: LACO: Adaptive Latent Communication for Collaborative Driving
- 2605.30813: Incremental BPE Tokenization
- 2606.01243: Unlocking the Black Box of Latent Reasoning
- 2606.06447: Latent Reasoning with Normalizing Flows (NF-CoT)
- 2606.20075: What Makes Effective Supervision in Latent CoT
- 2606.29712: Interpretable Discrete Latent Reasoning via Rendered Compression
- 2608.13570: Think in Latent, Explain in Language (SELR)
- 2607.08186: Hidden Decoding at Scale
- 2607.16972: Training Continuous CoT Models: A Tale of Two Regimes
- 2607.20952: The Weight of Silence (latent chess reasoning)
- 2608.14019: Emergent Models: Intelligence from Tiny Substrates
- 2609.27156: Giving Credit Where It's Due (RECAP)
- 2609.38972: Making LLMs Say What They Think (CIA)
- 2610.06666: What Matters for Latent Reasoning with Flow Matching (FLaRe)
- (no id): WhiteMatter: All-to-All Cross-Layer Connections via KV Mixing
- 2608.06014: Iterate or Widen? Test-Time Refinement for LiDAR Scene Completion (arXiv supplement)

## Caveats

- **S2 total citation count was not obtained (HTTP 429).** The 175 records are everything the citations endpoint returned; the supplement shows S2 is incomplete.
- Each row's category comes from title and abstract. Full texts were read only for the papers marked "(full text)": 2610.07730, 2609.36585, 2608.18171, 2607.18553, 2610.15218, 2609.36636, 2609.03379, 2609.35748, 2609.39740, 2607.14427, 2602.08864, 2607.22083, 2604.25917, 2610.00673.
- Where the abstract does not name the backbone, the row says "not verified": 2607.18618 LatentMT, 2610.10625 SafeBridge, 2602.08100.
- SanSi's per-depth gate numbers (3.6, 6.5, 7.7 and 2.8, 4.8, 6.2) are quoted from its Table 33 text. The exact depth values those numbers refer to were lost in HTML math extraction, so I call them increasing depths.
