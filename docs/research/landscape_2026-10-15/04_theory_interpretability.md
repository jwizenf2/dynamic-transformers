# 04 — Theory, generalization and interpretability of looped transformers

Slice of the landscape survey. Written 2026-10-15. Sources: arXiv abstract pages and, where marked
**[FT]**, the full HTML text (read via a summarizing fetcher, so recheck quoted numbers against
the PDF before citing). **[abs]** = only the abstract was read. **[snippet]** = known only from
a search-result snippet or another paper's citation; unverified. Venues are taken from the arXiv
comments field unless marked "(unverified)".

Our claims for reference: C1 needed passes grow with dependency depth (≈3.0 at depth 2 →
≈4.7 at depth 6, frozen few-shot Ouro); C2 Ouro's exit gate does not track this; C3 past the
trained max of 4 passes, shallow requests lose and deep chains gain; C4 a structure-aware
per-request loop controller.

## 1. Table

### 1a. Theory: how much depth / how many loops a k-step problem needs

| arXiv | Title | Date / venue | Type | Key claim (precise) | Setting | Relation to C1/C3 | Prediction for our curve |
|---|---|---|---|---|---|---|---|
| 2402.09268 [FT] | Transformers, parallel computation, and logarithmic depth (Sanford, Hsu, Telgarsky) | Feb 2024; arXiv lists no venue (ICML 2024, unverified) | Theory + small from-scratch exp. | Thm 4.2: hop_k is computed by a masked transformer of depth ⌊log₂k⌋+2, width O(1), one head per layer. Cor 4.3: Ω(log k) depth needed if mH = O(k^(1−ε)), *conditional on* the one-vs-two-cycle conjecture. Experiments with depths 2–6 and k ≤ 16: the largest learnable k roughly doubles per layer (6 layers: k ≤ 16; 5: k ≤ 8; 4: k ≤ 4) | Non-looped, synthetic pointer chasing | Source of the "logarithmic in hops" prediction for C1 | Needed depth ≈ log₂(hops) + const: about one extra block per doubling of depth |
| 2502.17416 [FT] | Reasoning with Latent Thoughts: On the Power of Looped Transformers (Saunshi, Dikkala, Li, Kumar, Reddi) | Feb 2025 (v1 says submitted Oct 2024); ICLR 2025 per the PDF header | Theory + from-scratch + pretraining | Cor 5.3: a 1-layer transformer looped ⌊log₂p⌋+2 times solves p-hop (d = O(1), ≤ 3 heads, O(log n) precision), matching Sanford's non-looped bound. Thm 5.4: m CoT steps of an L-layer model are simulated by m loops of an (L+O(1))-layer model. Empirically k⊗L ≈ kL-layer: p-hop n=256, 1⊗6 scores 99.9/99.5 for p = 16/32 vs 48.9/49.0 for 1⊗1. Downstream accuracy ≈ α·log(effective depth)+β; reasoning tasks gain more | Synthetic from scratch (addition, p-hop, i-GSM); 1B-scale pretraining | Strongest theoretical anchor for C1. Does **not** study more loops than trained (C3) | Log growth, if the needed computation is pointer chasing done in parallel (doubling). Linear growth if it is sequential, CoT-like (Thm 5.4: T steps ↔ T loops) |
| 2503.03961 [abs] | A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers (Merrill, Sabharwal) | Mar 2025; NeurIPS 2025 | Theory + from-scratch | Depth Θ(log n) suffices for regular-language recognition and graph connectivity. Neither is expressible at fixed depth under standard conjectures. Depth scaling is more efficient than width or CoT. Required depth matched trained depth in experiments | Uniform transformers; n = input length | Dependency resolution resembles reachability / topological order on the call DAG | Needed loops ∝ log(size of the dependency graph), not ∝ its depth |
| 2505.18948 [abs] | Exact Expressive Power of Transformers with Padding (Merrill, Sabharwal) | May 2025; NeurIPS 2025 | Theory | Polynomially padded transformers = FO-uniform TC⁰. Padding plus O(logᵈ n) looping = FO-uniform TC^d. Polylog looping gives NC | Asymptotic in n | Background: looping helps only with log-many iterations for parallelizable problems | Asymptotic only; no constants usable for n ≤ 6 |
| 2210.10749 [abs] | Transformers Learn Shortcuts to Automata (Liu, Ash, Goel, Krishnamurthy, Zhang) | Oct 2022; arXiv lists no venue (ICLR 2023, unverified) | Theory + from-scratch | Every finite automaton on length T has an O(log T)-depth polynomial-size simulator; often O(1) depth (Krohn–Rhodes). Learned shortcuts are brittle OOD | State tracking | Shortcuts can make needed depth flat in the number of steps | Can be flat (O(1)) or log, depending on algebraic structure |
| 2301.13196 [abs] | Looped Transformers as Programmable Computers (Giannou et al.) | Jan 2023; arXiv lists no venue (ICML 2023, unverified) | Theory | A 13-layer looped transformer emulates a small instruction-set computer; one loop per instruction | Constructive | Looping as sequential program execution | Linear in the number of sequential instructions |
| 2402.01107 [abs] | Simulation of Graph Algorithms with Looped Transformers (de Luca, Fountoulakis) | Feb 2024; arXiv lists no venue | Theory | Looped transformers with extra graph-attention heads simulate Dijkstra, BFS, DFS and Kosaraju with parameters independent of graph size. Finite precision limits simulation. Turing complete | Constructive | Iterations follow algorithm steps (e.g. BFS frontier ∝ depth) | Linear in path length for sequential BFS-style simulation |
| 2410.01405 [abs] | On Expressive Power of Looped Transformers: … Timestep Encoding (Xu, Sato) | Oct 2024; ICML 2025 | Theory + exp. | Approximation rate of looped transformers via a modulus of continuity. Identifies a looping-specific limitation, fixed by per-loop timestep scaling. More loops help empirically | Function approximation | Indirect: per-loop conditioning helps (Ouro has none, as far as we know; unverified) | — |
| 2505.12514 [snippet] | Reasoning by Superposition (Zhu, Hao, Hu, Jiao, Russell, Tian) | May 2025; NeurIPS 2025 (unverified) | Theory (continuous CoT, context only) | A 2-layer transformer solves directed reachability with D continuous-thought steps (D = diameter) by BFS in superposition; discrete CoT needs O(n²) | Coconut-style, not looping | Contrast: latent sequential steps scale linearly with diameter | Linear in diameter for the sequential latent-step model |
| 2610.14456 [FT] | Learning to Extrapolate in Depth with Looped Transformers (Yang, Huang, Wang, Mirzasoleiman) | 10 Oct 2026; arXiv preprint | Theory + synthetic | All-pairs connectivity. Each loop cubes the propagation operator (P^{3^T}), so recovery needs T > Hard(G), which is logarithmic in component size and in 1/(−log ρ). The trained block works only inside a finite window Hard(G) < T < T_max(G;K): too few loops give no propagation, too many give **false connections** (when the training residual δ_K > 0). Depth gain beyond training is T−K = Θ(H·3^{K−H}) | Population training at depth K; exchangeable graphs | **Very close to C3** (shallow/easy instances eventually hurt, hard ones gain beyond K) and to C1 (log in hardness) | Log growth; plus an extrapolation window whose upper edge explains shallow degradation |

### 1b. From-scratch experiments: generalization, extrapolation, overthinking

| arXiv | Title | Date / venue | Type | Key claim (precise) | Setting | Relation to C1/C3 | Prediction for our curve |
|---|---|---|---|---|---|---|---|
| 2604.07822 [FT] | Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers (Kohli, Parthasarathy, Sun, Yao) | Apr 2026 (v2 Aug); COLM 2026 | From scratch | Recurrent-depth models learn systematic 2-hop composition (3-stage grokking) and **depth extrapolation**: R=6 trained to 13 hops reached 17; R=8 trained to 16 reached 24. The benefit of scaling inference loops "only emerges for R>4". Overthinking: dynamic-R models gain nothing beyond 15 iterations. The logit margin rises, peaks, then falls "regardless of task complexity", and the peak drops for harder hops. Halting iterations (KL<0.01 & entropy<3) rise with hop count, **but no per-hop numbers are given**, and no iterations-per-hop law is stated | 4-layer block, d=768, synthetic KG, ≤40 hops | Closest empirical precedent for C1 (halting rises with hops) and C3 (deeper hops need more loops; overthinking) | Monotone rise; the form (log vs linear) is not reported |
| 2409.15647 [FT] | Looped Transformers for Length Generalization (Fan, Du, Ramchandran, Lee) | Sep 2024; ICLR 2025 | From scratch | n-RASP-L tasks. Training supervises the output after a ground-truth step count T(n) = problem length (multiplication: product of lengths). Inference stops at maximum confidence (batch or per-sample). Several tasks "converge" after solving; per-sample stopping is worse on Parity and Binary Sum | Arithmetic and algorithmic | Shows loops ≈ linear in problem size **when supervised that way** | Linear, but imposed by the training schedule rather than discovered |
| 2311.12424 [abs] | Looped Transformers are Better at Learning Learning Algorithms (Yang, Lee, Nowak, Papailiopoulos) | Nov 2023; ICLR 2024 | From scratch | Looped transformer matches a standard one on in-context data fitting with <10% of the parameters | ICL regression | Background | — |
| 2202.05826 [abs] | End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking (Bansal et al.) | Feb 2022; arXiv lists no venue (NeurIPS 2022, unverified) | From scratch (recurrent CNNs) | Coins "overthinking": accuracy degrades with many iterations. Fixed by recall (re-injecting the input) and progressive training | Mazes, prefix sums, chess | C3 vocabulary; recall ≈ "re-read" loops | Extra iterations safe only with input re-injection and iteration-agnostic training |
| 2606.29983 [abs] | Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping (Kuo, Chayti, Reizinger, Brendel, Jaggi) | Jun 2026; preprint | From scratch | OOD variance comes from a spurious correlation between length and loop count. Random loop counts during training help; RL-Halting improves the accuracy–stability trade-off but can lock in a suboptimal computation | Addition, Dyck-1, Unique Set, Copy | C3 and C4 (halting) | — |
| 2609.33144 [abs] | Beyond the Training Horizon: Mechanisms and Limits of Length Generalization in Looped Transformers (Liang, Jin, Pan) | Sep 2026; preprint | From scratch + mechanistic | Extrapolation works without faithfully representing intermediate states. Both looped variants fail at greater depth as local errors compound | Polynomial iteration, FSM composition, KG traversal | Caution: "needed loops" need not equal the number of reasoning steps | — |
| 2603.21676 [abs] | Thinking Deeper, Not Longer … Depth-Recurrent Transformers for Compositional Generalization (H.-H. Chen) | Mar 2026 (v2 Sep); preprint | From scratch | Accuracy rises once thinking steps match task complexity. On graphs the jump from chance is abrupt; on nested boolean logic and relational text it is gradual. Extrapolates on graphs. Per-step supervision hurts extrapolation. (No overthinking up to 24 steps per a search snippet; not verified in full text) | <1M params; graph reachability, nested boolean logic | Direct "steps ≈ complexity" evidence, from scratch | Threshold (step-like) curves; per-request needed loops ≈ structural depth |
| 2608.11233 [abs] | Retrofitting Recurrent Depth into a Pretrained Language Model (Shapiro) | Jul 2026; preprint | Retrofit (Qwen2.5-0.5B) | Extrapolates to ~1.5× supervised depth while keeping 70% accuracy through depth 18. Beyond depth 10: 53% vs 2.5% for a scratchpad model. "Learned depth selection remains unsolved" | Synthetic rule-depth tasks | C3-like extrapolation in a pretrained base; C4 named as open | — |
| 2607.00341 [abs] | DiscoLoop (Fu, Guo, Wang, Zhu, Lee, Jiao, Russell, Mei) | Jul 2026; preprint | From scratch + pretraining | Looping partly fixes the "depth-local storage problem" in 2-hop reasoning. After loop 1 the bridge entity is decodable but misaligned with its token embedding; realignment nearly closes the OOD gap | 2-hop | Mechanism for why one extra pass per hop may be needed | Suggests ≈1 pass per hop for knowledge hops (linear), if bridges are re-read per loop |
| 2405.15071 [abs] | Grokked Transformers are Implicit Reasoners (Wang, Yue, Su, Sun) | May 2024; NeurIPS 2024 | From scratch | Composition generalizes ID only after grokking and fails OOD. The authors suggest (do not test) cross-layer sharing | 2-hop KG | Motivation for looping | — |

### 1c. Pretrained-model analyses (looped and standard LMs)

| arXiv | Title | Date / venue | Type | Key claim (precise) | Setting | Relation to C1/C3 | Prediction for our curve |
|---|---|---|---|---|---|---|---|
| 2510.25741 [FT, partial] | Ouro (LoopLM) | Oct 2025 | Pretrained looped LM | Looping does not raise knowledge capacity (~2 bits/param) but helps manipulation and multi-hop (claimed in the intro; the dependence on loop or hop count was not visible in the section we read). Trained T=4. Base models degrade moderately at T=5–8 (1.4B MMLU 67.45 → 66.64 → 64.49; 2.6B Winogrande 75.53 → **75.93** at T=5). Thinking models degrade sharply (2.6B-Thinking OlympiadBench 76.44 → 71.85 → 39.26) but SuperGPQA **rises** 53.68 → 56.45 at T=5. Appendix B.5 "Theory: latent thought" not read | Our model | Baseline for C3: a task-dependent T=5 effect already appears in their tables, at benchmark level | Benchmark-level mixed T=5 effects, consistent with a structure-dependent split |
| 2610.18127 [FT] | Monitoring latent reasoning in looped language models (Olszowiec et al.) | 13 Oct 2026; preprint | Pretrained analysis, 8 models / 4 families | Five lenses (settling, decision time, best iteration, head fingerprints, noise). Ouro: 82–90% of MATH answer tokens are decided at iteration 1. The gate fires 3.2–4.1 iterations after the decision. At T=6 base Ouro peaks at 3–4 (1.4B 0.605 → 0.580): "a four-step pipeline". **Depth use does not track MATH level 1–5** (flips change <0.08/token); the exception is Huginn on arithmetic (flips 1.3 → 1.7 for 1 → 3 digits). Carry loops (Ouro) are fragile to noise; re-read loops (Huginn) do not degrade with more iterations. No multi-hop analysis | Ouro, LoopUS, Parcae, Rec-OLMo, Huginn | Supports C2. Partly contradicts the premise of C1 at the level of difficulty (but uses MATH level, not structural depth). Average C3 degradation | Predicts a flat curve if "difficulty" is what matters; our structural variable is the open question |
| 2502.05171 [snippet] | Scaling up Test-Time Compute with Latent Reasoning (Huginn; Geiping et al.) | Feb 2025; NeurIPS 2025 (unverified) | Pretrained recurrent-depth LM | Trained with random recurrence (mean ~32). Easier tasks saturate sooner; math/code keep improving. Per-token convergence varies by context (unverified detail) | Huginn-3.5B | Task-level "harder needs more", not per request | Saturating per task |
| 2507.02199 [abs] | Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer (Lu et al.) | Jul 2025; COLM 2025 workshop | Pretrained interpretability | Logit/Coda lens on Huginn arithmetic: "limited evidence of interpretable latent CoT". Probing is inconsistent across blocks; more recurrence gives marginal gains | Huginn-3.5B | Dead end for reading intermediate results per loop | — |
| 2604.11791 [abs] | A Mechanistic Analysis of Looped Reasoning Language Models (Blayney et al.) | Apr 2026; preprint | Pretrained + trained analysis | In many models each layer of the cycle converges to a distinct fixed point (cyclic trajectory). Attention stabilizes, and blocks repeat feedforward-like inference stages each iteration | Several looped LMs (not named in the abstract) | Explains why extra loops can be no-ops or drift (C3) | Gains stop once at the fixed point |
| 2608.18222 [abs] | Think Shallow, Solve Deep: Controlling Recurrent Dynamics for Reliable Test-Time Depth (Viakhirev et al.) | Aug 2026; submitted to AAAI-27 | From scratch + Huginn analysis | Settling, marginal or drifting operators predict whether more depth helps. Depth-safety condition: per-step displacement small relative to the decoder margin. Huginn is non-settling; freezing does not track difficulty ("guess-freezing"); carry task 0.69 at r=8 → 0.00 for r≥32 (from a snippet). Calibrating depth on 30 examples recovers the peak | Algorithmic, Sudoku, Huginn | C3 mechanism; per-task depth calibration ≈ weak C4 | — |
| 2609.16665 [abs] | Right Direction, Wrong Step (Guo et al.) | Sep 2026; preprint | Pretrained analysis | Harmful extra iterations often point in an improving direction but overshoot. A quarter-step gives positive gains on 72.2–83.2% of selected failures | Two looped families, math and commonsense | Mechanism for shallow degradation at pass 5 (C3) | — |
| 2609.21383 [abs] | Prediction Dynamics in Depth-Recurrent Language Models (Luo, Yu) | Sep 2026; preprint | Pretrained analysis | Margin decomposition explains finite-depth answer preservation. On Huginn-3.5B and Ouro-1.4B it reduces the earliest qualifying depth by 22.5–34.4% of total depth (retrospective) | Huginn, Ouro-1.4B | Better per-request "needed depth" estimators (C2/C4) | — |
| 2609.19934 [abs] | Beyond Depth Truncation: Controlled Evaluation of Depth Utilization in Recursive LMs (Dau et al.) | Sep 2026; preprint | Methodology | The truncation slope mixes fewer block applications, less distinct computation, and an OOD readout. Proposes controls | Protocol only | **Threat to the C1 measurement**: an early-exit readout at pass k may fail from readout shift, not missing computation | — |
| 2607.20519 [abs] | Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts (Popescu, Sáez de Ocáriz Borde, Liò) | Jul 2026; preprint | From scratch + Ouro | Post-hoc confidence readouts match or beat learned gates. Failures come from the gate-induced trajectory. Ouro's gates are "not uniformly Pareto-optimal" | Mod-arith, parity, Ouro-1.4B/2.6B | C2, C4 baselines | — |
| 2608.18171 [FT] | Looped Language Models Improve Compositional Tool Calling (same authors) | Aug 2026; preprint | Fine-tuned Ouro + retrofits | Recurrence helps compositional tool use more than single calls. Ouro-2.6B "near-saturation after three iterations". NESTful only aggregate; **no breakdown by nesting/dependency depth, no per-example oracle, no Ouro >4 loops**. Tables say depth 8 for looped models while §4.3 says Ouro 1–4 (internal inconsistency) | API-Bank, BFCL, NESTful | Nearest application paper; leaves C1/C3 per-structure open | Saturation by pass ~3 on aggregate |
| 2505.13898 [abs] | Do Language Models Use Their Depth Efficiently? (Csordás, Manning, Potts) | May 2025; NeurIPS 2025 | Standard-LM analysis | Second-half layers contribute much less. "Unable to find evidence" that models use extra depth to compose subresults on multi-hop | Llama 3.1, Qwen 3, OLMo 2 | Contrast: non-looped LMs show no per-hop depth use. C1 on Ouro would be a positive counterpart | — |
| 2406.12775 [abs] | Hopping Too Late (Biran, Gottesman, Yang, Geva, Globerson) | Jun 2024; EMNLP 2024 | Standard-LM analysis | Bridge entity is resolved early and the second hop late. Back-patching fixes up to 66% of failures, i.e. the model runs out of depth | 2-hop queries | Mechanistic motivation: looping = repeated back-patching | ≥1 extra block per hop |
| 2402.16837 [snippet] | Do LLMs Latently Perform Multi-Hop Reasoning? (Yang et al.) | Feb 2024; ACL 2024 (snippet) | Standard-LM analysis | First hop scales with model size; second hop does not | Llama-2 | Background | — |
| 2601.03542 [snippet] | Layer-Order Inversion: Rethinking Latent Multi-Hop Reasoning in LLMs | Jan 2026 | Standard-LM analysis | For 2–4 hops, the final answer becomes decodable *earlier* than bridge entities, against hop-aligned layer circuits (unverified; from a snippet) | GPT-J, Llama3 | Challenges "one block per hop" | Weakens the linear-per-hop prediction |

### 1d. Fixed points, stability and scaling (context for C3/C4)

| arXiv | Title | Date / venue | Type | Key claim | Relation |
|---|---|---|---|---|---|
| 2605.26733 [abs] | STARS: Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in LoopLMs (Yang et al.) | May 2026; comments say ICML 2026 | Training method | Performance "often peaks at a certain iteration depth and then collapses". Jacobian spectral-radius regularization plus random loop sampling gives stable fixed points and reliable scaling | C3: degradation is a property of how Ouro was trained, not inevitable |
| 2606.18206 [snippet] | Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers (Movahedi et al.) | Jun 2026; AdaptFM workshop (unverified) | Architecture | Pre-norm plus residual scaling for looped fixed-point transformers; halts on iterate residual (per citing paper 2609.33540) | C4 halting baseline (residual-based) |
| 2610.06833 [abs] | Towards Looped Models Done Right, Part II: Rethinking at Fixed Points (Huang et al.) | Oct 2026; preprint | Training method | A learned depth prior plus orthogonal input injection. Near fixed points, terminal KV sharing is almost lossless | Background |
| 2509.23314 [abs] | Two-Scale Latent Dynamics for Recurrent-Depth Transformers (Pappone et al.) | Sep 2025; preprint | Analysis + exit rule | Loop steps shrink and become orthogonal. A second-order step-size exit beats KL exit | C4 baseline |
| 2604.21106 [abs] | How Much Is One Recurrence Worth? Iso-Depth Scaling Laws (Schwethelm et al.) | Apr 2026; preprint | Scaling law | φ = 0.46: one recurrence is worth less than one unique block (r=4: 410M looped ≈ 580M dense in loss, at 1B training cost) | Calibrates "a pass ≠ a layer" |
| 2604.15259 [snippet] | Stability and Generalization in Looped Transformers (Labovich) | Apr 2026 | Theory framework | Fixed-point stability along reachability, input dependence and geometry | Background |

## 2. Predicted shape of needed-loops vs dependency depth

**What theory predicts depends on what computation one pass performs.**

1. **Logarithmic (parallel composition / pointer doubling).** p-hop needs ⌊log₂p⌋+2 loops of a
   1-layer block (Saunshi Cor 5.3; Sanford Thm 4.2, tight under a conjecture). Connectivity and
   regular languages need depth Θ(log n) (Merrill & Sabharwal). In 2610.14456 each loop cubes the
   propagation operator, so loops ∝ log₃(hardness). Assumptions: the model *learned* the
   doubling algorithm (constructions exist, but learnability is shown only in small synthetic
   settings); hops are uniform lookups; precision is O(log n). For a pretrained 24-layer-ish
   block (Ouro), one pass contains many layers, so the constants do not transfer. Only the slope
   in log(depth) is a usable prediction.
2. **Linear (sequential, CoT-like steps).** Thm 5.4 (T CoT steps ↔ T loops), Giannou's
   one-instruction-per-loop computer, de Luca's BFS simulation, continuous-CoT reachability in
   D steps (Zhu et al.), and Fan et al., where T(n) = length is *imposed* by supervision. This
   applies if the model resolves one dependency per pass (DiscoLoop/Hopping-Too-Late mechanism:
   bridge entity decoded after one pass, next hop needs another).
3. **Flat / O(1) (shortcuts, or offloading to generated tokens).** Liu et al.'s shortcuts. More
   importantly for tool calling, outputs are autoregressive: if each call's arguments only
   reference already-emitted calls (`$var1`), every token is a constant-depth step and per-token
   needed depth should be **flat** in chain depth (Thm 5.4 read in reverse: CoT tokens replace
   loops). Growth in C1 therefore suggests that some token must resolve the *whole* dependency
   structure, for example choosing the first call requires knowing the topological order. That
   is a connectivity-type problem, which the log-depth theory covers.
4. **Saturating / windowed (finite trained depth).** With trained max K = 4, measured needed
   passes are **censored** at 4 (or 5 with extrapolation). 2610.14456 predicts a window
   Hard(G) < T < T_max: hard instances gain past K, easy ones eventually develop false
   positives. 2604.07822 reports the logit-margin peak shifting later and lower with hops. Both
   predict a C3-like split, and both predict that the observed curve flattens near K whatever
   the true law.

**Our numbers.** Needed ≈ 3.0 at depth 2 and ≈ 4.7 at depth 6 fit both log₂(d)+2 (3.0, 4.58)
and linear 2.15+0.425·d (3.0, 4.70) almost equally. The two curves differ most at d = 1
(2.0 vs 2.57) and d = 3 (3.58 vs 3.42). At d ≥ 8 the predictions (5.0 vs 5.55) lie beyond the
censoring ceiling. The match with Sanford/Saunshi's ⌊log₂p⌋+2 is numerically striking but very
likely coincidental (different block sizes, and "hop" ≠ "dependency depth"). Do not claim it
without the tests below.

**How to test it.**
- Fit linear, log₂, and saturating (a − b·e^{−d/c}) models with **interval censoring**: a request
  first solved at pass k has needed ∈ (k−1, k]; a request never solved is right-censored at
  4 or 5. Compare by likelihood/AIC with bootstrap CIs. Report d = 1 and d = 3 strata
  explicitly, since they discriminate. Include the per-request controls (#calls, length).
- Decouple **DAG depth** (longest path) from **DAG size** (#calls) and **width**: the log-n
  theory (Merrill) predicts dependence on size, the hop theory on longest path, the
  sequential-CoT view on nothing per token. Use matched NESTful subsets or synthetic
  rewrites of xLAM/NESTful prompts with the same functions but different dependency wiring.
- **Localize the hard token** (2610.18127 Lens 2 adapted): per pass, find which output token
  first flips to correct. If it is the first call or the first `$var` reference, the
  "plan the whole DAG" account holds; if it moves along the chain, the per-hop sequential
  account holds.
- **Synthetic calibration on frozen Ouro**: few-shot p-hop pointer chasing and variable-binding
  chains (x1=f(x0) …) at p = 1…8. That gives Ouro's native passes-vs-hops curve to set
  against the tool-call curve.
- Guard against readout confounds (2609.19934): Ouro trains its exits at every pass ≤ 4, so
  early-pass readouts are on-distribution, but pass 5 is not. Report pass-5 gains as
  "extrapolation including readout shift".

## 3. Threats to novelty

- **2604.07822 (COLM 2026)**: halting iterations rise with hop count, and deeper hops are solved
  by extra inference recurrence beyond training, with overthinking. From scratch, synthetic
  KG, aggregate only, no law. C1/C3 must be framed as *pretrained, frozen, real tool data,
  per request, controlled for confounds*.
- **2610.14456 (10 Oct 2026)**: theory plus experiments predicting exactly the C3 pattern
  (hard instances gain past the training depth, while excessive loops create false positives)
  with an explicit window. Cite it as the theoretical expectation; our contribution is
  observing it per request in a pretrained LM, split by structure.
- **2608.18171**: fine-tuned Ouro on NESTful/BFCL with depth curves. It has no per-depth
  breakdown, no oracle, and no T>4, but a quick follow-up by these authors (who also wrote
  2607.20519 on gates) could cover C1/C2. This is the fastest-moving competitor.
- **Saunshi / Sanford**: anyone could say "of course needed loops grow with hops". The novelty
  must be measurement on a pretrained model plus the shape (log vs linear) and the controls.
- **2610.18127**: already shows the Ouro gate fires late and that depth does not track MATH
  difficulty. This narrows C2 to "does not track *structural* depth in tool chains", which
  is still open.
- **Ouro paper tables 10–13**: T=5 improves some benchmarks (2.6B Winogrande, 2.6B-Thinking
  SuperGPQA) and hurts others. So "T=5 helps some inputs" is visible at benchmark level; our
  C3 needs the per-request, structure-conditioned split.
- **2608.18222**: per-task depth calibration from 30 examples is a simple competitor for C4.
  Post-hoc confidence readouts (2607.20519) and second-order step size (2509.23314) are
  required C4 baselines.

## 4. Known dead ends / contested results

- **Does loop use track difficulty?** No on MATH levels for pretrained looped LMs (2610.18127)
  and no for Huginn per token (2608.18222 "guess-freezing"). Yes, from scratch, on hops
  (2604.07822, 2603.21676). The only pretrained exception is Huginn arithmetic flips
  1.3 → 1.7. Difficulty level ≠ structural depth, which is the gap C1 fills, but reviewers
  will cite these.
- **Overthinking: inherent or a training artifact?** Inherent under fixed-depth training with
  residual bias (2610.14456 window; 2604.07822 margin peak; Ouro tables; STARS: "peaks then
  collapses"). Avoidable with recall/progressive training (Bansal), contraction / fixed-point
  objectives (STARS, 2608.18222, FPRM), or re-read loops (Huginn does not degrade per
  2610.18127). Since Ouro is a carry loop trained at fixed max 4, degradation is expected.
- **Hop-per-layer circuits in standard LMs**: sequential late hopping (Biran; Yang 2402.16837)
  vs layer-order inversion (2601.03542, snippet only). Csordás finds no evidence that extra
  depth composes subresults. "One block per hop" is not established in pretrained LMs.
- **Logit lens on recurrent LMs** gives limited, inconsistent latent-CoT evidence on Huginn
  (2507.02199). Prefer the model's own per-pass readout (Ouro has trained exits), as 2610.18127
  argues.
- **Truncation curves are confounded** (2609.19934): fewer applications, less distinct
  computation, and readout shift are mixed together. This applies directly to how we define
  "needed passes".
- **Extrapolation ≠ faithful algorithm**: success past the horizon without faithful
  intermediate states (2609.33144). Variance is driven by the length/loop-count spurious
  correlation (2606.29983). The trained depth may not recover the ideal iterative rule
  (2610.14456, citing "Ye et al., 2026", which we did not find).
- **A loop ≠ a layer**: φ = 0.46 (2604.21106). Constants from layer-count theory should not be
  transferred to pass counts.
- **Theory is asymptotic.** Merrill–Sabharwal padding/looping (TC^d with logᵈ n loops) and
  Liu shortcuts say nothing quantitative at chain depths 1–6. Only the log-vs-linear
  *form* is testable at our scale, and censoring at 4–5 passes limits even that.

## 5. Not verified / follow-ups

- Venues marked "(unverified)": Sanford (ICML 2024?), Giannou (ICML 2023?), Liu (ICLR 2023?),
  Bansal (NeurIPS 2022?), Geiping (NeurIPS 2025?), Zhu (NeurIPS 2025?).
- Ouro Appendix B.5 "Theory: latent thought with LoopLM" and its Mano/multi-hop synthetic
  results were not read (page truncated).
- 2601.03542, 2402.16837, 2505.12514, 2606.18206, 2604.15259 and 2502.05171 details are from
  snippets. 2603.21676's "no overthinking up to 24 steps" is from a snippet.
- Not found: any paper giving per-request needed loops vs structural depth on a pretrained
  looped LM; "Ye et al. 2026" (cited by 2610.14456).
- Not covered here: Merrill & Sabharwal 2310.07923 (CoT expressivity) and Merrill et al.
  2404.08819 (Illusion of State). Both are relevant background on CoT vs depth; ids from
  memory, not re-fetched.
