# 02 — Looped / recurrent-depth architectures and available models (2018–2026)

Compiled 2026-10-15 from arXiv abstract/HTML pages, Hugging Face model cards and GitHub READMEs
fetched that day (WebFetch summaries, not full reads of every paper). Markers:
- **[V]** verified on the arXiv/HF/GitHub page today.
- **[V-sec]** taken from a secondary source fetched today (citing paper, aggregator, community card).
- **[M]** from memory, id/details not re-fetched today. Check before citing.
- "n/s" = the fetched page did not state it. That does not mean it is absent.

Our own measured numbers (Ouro on xLAM/NESTful) come from `docs/research/findings_2026-10-14_routing_and_looped_models.md`
and are not repeated here, except where they disagree with a model card.

---

## 1. Main table

Types: **PT** = looped LM pretrained from scratch at scale; **RF** = retrofit/conversion of a pretrained
non-looped LM; **SM** = small-scale (≤~1.6B) pretraining study; **SY** = synthetic/theory; **INF** = inference/systems.

### 1a. Lineage (2016–2025)

| arXiv | Name | Date / venue | Type | Sizes & weights | Recurrence | Halting | Tooling | Evaluated on | Key result | Weaknesses |
|---|---|---|---|---|---|---|---|---|---|---|
| 1603.08983 [M] | Adaptive Computation Time (Graves) | 2016 | SY (RNN) | none | variable | learned halting prob. + ponder cost | – | synthetic, char-LM | first learned halting | ponder cost hyper-sensitive |
| 1807.03819 [M] | Universal Transformer | 2018, ICLR'19 | SM | none (tensor2tensor code) | shared layer looped T times | per-position ACT | – | bAbI, LAMBADA, MT | beats Transformer on algorithmic/bAbI | ACT unstable; does not scale cheaply |
| 1909.11942 [M] | ALBERT | 2019, ICLR'20 | encoder, cross-layer sharing | HF `albert-*` (Apache-2.0) | fixed (12/24 shares) | none | HF | GLUE, SQuAD, RACE | parameter-efficient BERT | compute not reduced; not adaptive |
| 2104.06022 [M] | Lessons on Parameter Sharing (Takase & Kiyono) | 2021 | SM | none | sequence/cycle sharing | none | – | MT, LM | "cycle" sharing best | – |
| 2107.05407 [M] | PonderNet | 2021 | SY | none | variable | learned halting distribution, KL to geometric prior | – | parity, bAbI | stable halting vs ACT | small scale |
| 2301.13196 [M] | Looped Transformers as Programmable Computers | 2023, ICML | SY/theory | none | looped | none | – | constructions | looped TF can emulate programs | theory |
| 2310.07096 [M] | Sparse Universal Transformer | 2023, EMNLP | SM | none | looped + MoE | stick-breaking dynamic halting | – | MT, logical inference | MoE + UT at lower compute | small scale |
| 2405.16039 [M] | MoEUT | 2024, NeurIPS | SM | code only [M] | shared MoE layer groups | none | – | LM ppl, zero-shot | UT competitive with dense at LM scale ≤~1B | small scale |
| 2410.20672 [V] | Relaxed Recursive Transformers | Oct 2024 (rev. Feb 2025), ICLR'25 [M for venue] | RF | Gemma-1B-scale recursive models; weights n/s | 2–3 loops of shared block; per-loop LoRA | none (proposes continuous depth-wise batching / early exit) | – | few-shot LM benchmarks | recursive Gemma 1B beats TinyLlama 1.1B / Pythia 1B; recovers most of original | uptraining needed; LoRA breaks pure sharing |
| 2502.05171 [V] | Huginn-0125 ("recurrent depth approach") | Feb 2025 (NeurIPS'25 per a citing paper, [V-sec]) | PT | **`tomg-group-umd/huginn-0125`**, 3.5B, Apache-2.0 [V] | prelude 2 / core 4 / coda 2 layers, hidden 5280; trained r ~ log-normal Poisson, mean 32, truncated BPTT k=8; usable at arbitrary r, gains to ~64 then flat [V] | none trained; zero-shot per-token exit (`kl`, `entropy-diff`, `latent-diff`, `argmax-stability`, threshold 5e-4) [V] | HF `trust_remote_code`, custom `HuginnDynamicCache`; KV sharing `compress-s16`; no MLX port found [V]; CDB serving (2608.09444) | ARC, HellaSwag, MMLU, OBQA, PiQA, SciQ, WinoGrande, GSM8K, MATH, MBPP, HumanEval, MT-Bench | r=1 near chance (GSM8K 0.00); r=32 GSM8K CoT 34.8/42.1 (sys prompt); "compute equiv. 50B" [V] | 800B tokens only, LR never cooled, not instruction-tuned; weak closed-book QA; many steps needed (slow); r=1 useless |
| 2502.17416 [V] | Reasoning with Latent Thoughts (Saunshi et al.) | Feb 2025, ICLR'25 | SY + SM | none | k layers looped L times | none | – | addition, p-hop, math, LM downstream | looped k×L ≈ kL-layer on reasoning; T loops simulate T CoT steps | memorization weaker than reasoning |
| 2502.07503 [V-sec] | Recursive Inference Scaling | Feb 2025 | SM | n/s | – | – | – | – | cited baseline in SMELT/Loopie | not checked |
| 2507.10524 [V] | Mixture-of-Recursions (MoR) | Jul 2025, NeurIPS'25 [V-sec] | SM | 135M–1.7B; no official weights found; community `sudeshmu/mixture-of-recursions-360m` [V-sec] | Nr ∈ {2,3,4}, middle-cycle sharing | **per-token router** (expert-choice or token-choice) chooses recursion depth | official code `raymin0223/mixture_of_recursions` | few-shot LM avg | equal FLOPs: 43.1 vs 42.3 avg with ~50% fewer params; up to 2.06–2.18× throughput (text inconsistent) [V] | ≤1.7B; no reasoning eval; expert-choice routers saturate, can't change capacity post-hoc; KV sharing hurts expert-choice |
| 2510.07358 [V-sec] | Encode–Think–Decode (ETD) | Oct 2025 | RF (mid-training) | OLMo-2 1B; weights n/s | 7 enc / 4 think × k / 5 dec; k=5 eval | adaptive-depth variant tested | – | 17 reasoning benchmarks | GSM8K +28.4% rel., MATH +36% rel. at k=5 | small model |
| 2510.25741 [V] | **Ouro** (LoopLM) | Oct 2025 (v6 Oct 2026) | PT | **`ByteDance/Ouro-1.4B`, `Ouro-2.6B`, `Ouro-1.4B-Thinking`, `Ouro-2.6B-Thinking`**, Apache-2.0 [V] | 1.4B: 24 layers × 4; 2.6B: 48 × 4 (project page) [V]. Trained R ≤ 4. Beyond 4: 6-bench avg 1.4B 72.39 (T4) → 71.32 (T5) → 69.87 (T8); 2.6B 77.44 → 77.23 → 76.44 [V] | per-loop exit gate + LM head; entropy-regularized stage 1, gate-only stage 2; Q-exit on cumulative prob. ≥ q [V] | HF `trust_remote_code`, transformers <4.56 [V]; vLLM native v0.11.1–v0.26.0, always full 4 loops [V]; MLX: `GeorgiyDemo/ouro-mlx` (2.6B-Thinking, `--steps N`, gates loaded but unused) [V], `mlx-community/Ouro-2.6B-Thinking-4bit`, `ArcadaLabs/Ouro-2.6B-Thinking-mlx-bf16` [V-sec]; llama.cpp PR, chatllm.cpp, OpenVINO, TransformerLens [V] | MMLU, GSM8K, MATH500, BBH, HumanEval, AIME, GPQA | 2.6B: MMLU 74.60, MATH500 90.85, BBH 80.46 (vs Qwen3-8B 76.63/62.30/77.65); 7.7T tokens; "matches up to 12B" [V] | per-loop KV cache so early exit saves little (2610.09013); gate fires late (2610.18127); Thinking variants degrade after iter 1–2; HF card for 2.6B says "24 layers" (contradicts project page and our measured 48) |
| 2511.07384 [V] | Retrofitted recurrence (McLeish et al.); "Rec-OLMo" in 2610.18127 | Nov 2025 | RF | **19 checkpoints** under `smcleish/`, e.g. `Recurrent-OLMo-2-0425-train-recurrence-{4,8,16,32}`, `Recurrent-Llama-3.2-train-recurrence-*` (1B), `Recurrent-TinyLlama-3T-*` (0.8B) [V]; repo Apache-2.0, weight license n/s | prelude/core/coda e.g. (4,8,4), (4,6,4); recurrence curriculum; r up to 32 at test | none (fixed r) | own code (Huginn-style); HF collection | GSM8K CoT, MATH, val loss | OLMo-2-1B MATH 25.1 → 40.6 at (4,6,4), r=32 [V-sec] | ~50B tokens math-heavy (Nemotron-CC-Math); "Rec-OLMo" = this family is an inference [unverified mapping]; gains plateau after a few iterations (2610.18127) |

### 1b. 2026 architectures and models

| arXiv | Name | Date / venue | Type | Sizes & weights | Recurrence | Halting | Tooling | Evaluated on | Key result | Weaknesses |
|---|---|---|---|---|---|---|---|---|---|---|
| 2602.11451 [V] | LoopFormer (elastic depth, shortcut modulation) | Feb 2026, ICLR'26 | SM | weights n/s | elastic depth | n/s | n/s | "latent reasoning" LM benchmarks | no numbers in abstract | not checked in full |
| 2604.12946 [V] | **Parcae** | Apr 2026 | SM | **`SandyResearch/parcae-140m`, `-370m`, `-770m`, `-1.3b`** (FineWeb-Edu, base only; license n/s on card) [V] | prelude/core/coda 8/8/8 for 1.3B, default recurrence 8; per-sequence Poisson depth; tested to T=24; saturating-exponential test-time law, little gain past trained depth [V] | implicit (fixed T) | `pip install parcae-lm`, not HF transformers; card ids mismatch repo names [V] | val ppl, WikiText, Core/Core-Extended | 1.3B Core 28.44 vs 25.45 Transformer; 770M ≈ 1.3B TF; up to 6.3% lower ppl vs prior RDMs; γ_μ≈0.40, γ_D≈0.78 [V] | small, base-only, no chat/tool ability; custom loader |
| 2604.21106 [V] | How Much Is One Recurrence Worth? (iso-depth scaling) | Apr 2026 | SM | n/s | r ∈ {1,2,4,8} | – | – | scaling fits | φ = 0.46 (TBPTT 0.38, hyper-connections 0.65); r=4 410M looped ≈ 580M dense but costs like 1B to train [V] | – |
| 2604.21254 [V] | Hyperloop Transformers (loop-level hyper-connections) | Apr 2026 (v3 Jul) | SM | 240M, 1B, 2B; weights n/s | begin/middle/end, middle looped | none | n/s | ppl | lower ppl than depth-matched TF with ~50% params; beats mHC [V] | ppl only in abstract |
| 2605.09165 [V] | Sparse Layers are Critical to Scaling Looped LMs (Looped-MoE) | May 2026 | SM | n/s | – | early exit at loop boundaries | – | scaling | looped-MoE scales better than dense; dense looped does not [V] | no numbers in abstract |
| 2605.11011 [V] | **LoopUS** | May 2026 | RF | Qwen3-1.7B/4B/8B, Phi-4 14B converted; released: **`Thrillcrazyer/Qwen3_1.7B_LoopUS`** (Apache-2.0 on card) [V] | enc / looped reasoning block / dec; trained B=20 steps, K=5 supervised; inference max 8; tested 40–100 on 4B [V] | learned confidence head, threshold 0.6; avg 3.39/8 steps on Qwen3-4B [V] | own repo `uv run chat.py`; no HF transformers path [V] | MMLU, HellaSwag, ARC, PIQA, WinoGrande, OBQA | +1.6 to +2.2 avg points (e.g. Qwen3-8B 63.2 → 65.4) [V] | MMLU/HellaSwag flat or down; gate fires ~16.6 iters late (2610.18127); only 1.7B released |
| 2605.18797 [V-sec] | Fully Looped Transformer | May 2026 | SM | n/s | – | – | – | – | stabilization | not checked |
| 2605.23872 [V] | Training-Free Looped Transformers | May 2026 | RF (inference only) | wrapper over frozen Qwen3-4B-Instruct, Qwen3-30B-A3B, Moonlight-16B-A3B (7 families) | loops a contiguous mid-block with damped Euler sub-steps | none | wrapper code n/s | MMLU-Pro, CSQA, OBQA | +2.64 MMLU-Pro (Qwen3-4B), +1.1–1.2 elsewhere [V] | naive block reapplication usually degrades |
| 2606.18023 [V] | LoopCoder-v2 | Jun 2026 | PT (code) | 7B, 18T tokens; weights n/s | 2 loops best; ≥3 regress [V] | none | n/s | SWE-bench Verified, Multi-SWE | SWE-V 43.0 → 64.4 (non-looped → 2-loop) [V] | more loops hurt |
| 2607.10110 [V] | CHASE (looped Mamba exit) | Jul 2026 | SM | n/s | – | Ouro-style gate adapted with cache-hole training | – | MANO, p-hop, LM | ~half the recurrent steps at near-full ppl [V] | SSM state skipping breaks without adaptation |
| 2607.13491 [V-sec] | DeepLoop | Jul 2026 | SM | n/s | – | – | – | – | DeepNorm-style init breaks under sharing | not checked |
| 2607.16051 [V] | **Loop the Loopies!** (Loopie MoE) | Jul 2026 | PT (MoE) | **`IQuestLab/Loopie-20B-A2B-preview`, `IQuestLab/Loopie-6B-A0.6B-preview`** [V; HF card returned 401 to fetch, license unverified] | layer-loop (each layer twice), R=2 fixed; no other R evaluated [V] | none | Megatron + vLLM variants in `IQuestLab/loopie` [V] | AIME, AMC, MMLU | 20B-A2B Thinking AIME24 92.09 vs Qwen3-30B-A3B 90.10 with 3.5T vs 36T tokens [V] | fixed 2 loops; no inference-depth flexibility |
| 2607.22083 [V] | **Nanbeige4.2-3B** (looped agentic model) | Jul 2026 | PT (+SFT/RL, agentic) | **`Nanbeige/Nanbeige4.2-3B`, `Nanbeige/Nanbeige4.2-3B-Base`**, Apache-2.0 [V]; `mlx-community/Nanbeige4.2-3B-OptiQ-4bit` [V] | 22 layers × `num_loops: 2` (config) [V]; 2 passes chosen; more passes "marginal" [V]; other R at inference not reported | none | HF `trust_remote_code` (transformers 4.45.1 in examples); vLLM/SGLang/llama.cpp **forks only**; MLX needs OptiQ ≥0.4.6 [V]; 5 MPS bugs fixed in 2608.13987 [V] | SWE-bench, Terminal-Bench, MCP-Atlas, GDPval, Claw-Eval, GPQA | SWE-V 63.6, MCP-Atlas 57.8 (vs Qwen3.5-9B 47.4) [V]; 2608.13987: near-perfect BFCL single calls, fails most multi-tool [V] | only 2 loops trained (tiny action space); per-loop KV cache; prefill attention memory doubled |
| 2609.01343 [V] | SMELT (compute-matched MoE looped scaling) | Sep 2026 | SM–large | up to 54B non-emb.; weights n/s | middle half looped twice | none | n/s | scaling + downstream | 6.8–18.0% training-FLOP savings at matched FLOPs, params, KV [V] | – |
| 2610.06833 [V] | Towards Looped Models Done Right, Part II | Oct 2026 | SM | 100M–1.6B; **49 checkpoints `IFM/LoopedLM-P1-*`, `IFM/LoopedLM-P2-*`** (xLLM native format, not HF transformers) [V] | fixed/sampled/learned depth prior | learned depth prior (not request-level halting) | `ifm-ai/xllm-loop` | ppl, downstream avg | learned prior + 3× smaller KV matches full-cache fixed depth at 1.6B; prefill 1.79× faster [V] | custom format |
| 2610.12582 [V] | **ELR / Koi-4B** (Elastic Latent Recurrence, NVIDIA authors) | 8 Oct 2026 | PT (+mid/SFT/RL) | 4B dense hybrid (GDP-Hybrid-4B: 23 Gated DeltaProduct + 6 attention + 23 FFN; 14/24/14 split) and 30B-A3B MoE hybrid; **no HF id found** (searches today) | trained mixed k=1–4 (RLVR 4:1:1:1); beyond 4 not evaluated [V] | none: loop count is a user-set budget input; adaptive depth listed as future work [V] | n/s; last-loop KV option keeps attention KV at 1-loop size (−0–3 pts at k=4) [V] | AA Intelligence Index v4.1, Terminal-Bench 2.1, τ3-Bench Banking, GDPval-AA, IFBench, LCB v6, MATH-500 | Index ≈ 18/26/28/29 at k=1/2/3/4; τ3-Bench Banking 11.3 → 21.8, Terminal-Bench 12.1 → 28.0 (k=1 → 4) [V] | weights not public (as of today); own-harness scores |

### 1c. Inference / analysis papers that use existing looped checkpoints (relevant tooling)

| arXiv | Name | Models | What it gives us |
|---|---|---|---|
| 2608.09444 [V] | Continuous Depth Batching | Ouro-1.4B, Huginn | serving system for per-token depth; up to 99% of max speedup; fully looped models suit adaptivity best |
| 2610.09013 [V] | Enabling Dynamic Computation in Looped LMs | Ouro | "best-available" KV: −30% FLOPs and KV at full accuracy; Ouro early exit otherwise saves nothing |
| 2610.05265 [V] | Loopy (low-bit quant for looped LMs) | Ouro-1.4B | W4A4: −36.5% LAMBADA ppl vs SpinQuant |
| 2608.13987 [V] | Nanbeige4.2-3B on Apple Silicon | Nanbeige4.2-3B | 5 MPS bugs, chunked prefill (2.7× context on 32 GiB), BFCL/MCPMark results |
| 2610.18127 [V] | Monitoring latent reasoning | Ouro ×4, LoopUS-1.7B, Parcae-1.3B, Huginn, Rec-OLMo | depth use does not track MATH difficulty; Ouro gate 3.2–4.1 iters late; Huginn improves to 16 steps, Ouro degrades past 4 |
| 2608.18171 [V] | Looped LMs Improve Compositional Tool Calling | Ouro-1.4B/2.6B, retrofitted Llama-3.2-1B and OLMo-2-1B (McLeish) | LoRA-SFT; BFCL 86.4 (Ouro-2.6B); NESTful win 0.371; depth curves only in figures; adaptive = Ouro gate thresholds |
| 2609.19934 [V] | Beyond Depth Truncation (Depth Control Protocol) | n/s | methodological controls for "does the model use its depth": separate #applications, distinct computation, readout distribution shift. **Directly relevant to how we argue C1/C3** |
| 2608.18230 [V] | Allocating Recurrent Compute (MixerLoop) | 15M/110M from scratch | which sublayer to loop; not per-request |
| 2604.07822, 2607.20519, 2603.19714 (LoopRPT), 2605.26733 (STARS), 2602.10520 (RLTT) | see novelty file | Ouro and from-scratch | covered in our findings doc |

Not checked (seen in search results only): 2605.26797 Latent Recurrent Transformer, 2606.01495 CART, 2607.27656 source-centered state evolution, 2606.24898 readout blind spot, 2608.03624 LoopMTP, 2609.29812 FlashLoop, 2609.17184 LoopSpec, 2609.23033 WaveFront, 2610.07730 SanSi, 2602.16490 Growing↔Looping, 2510.04871 TRM / 2506.21734 HRM [M] (small synthetic recursive reasoners, not LMs).

---

## 2. Which models could we use to test generality of C1–C4 (Mac Studio M5 Max, 128 GB, MLX / PyTorch-MPS)

Criteria: open weights; inference-time loop count variable over ≥3 values; enough instruction/tool
ability to get non-trivial xLAM/NESTful accuracy few-shot (or after light LoRA); runnable locally.

1. **Ouro-1.4B / 2.6B and the -Thinking variants (ByteDance, Apache-2.0).** Already in use. Adding the
   Thinking variants is the cheapest generality check (same code), but 2610.18127 reports they peak at
   iteration 1–2 and degrade after, so expect a different needed-depth profile. That is itself a test of C1/C3.
2. **Huginn-0125 (3.5B, Apache-2.0).** The only open, reasonably large model with a *wide* recurrence range
   (1–64) and a different family (re-read loop, random init state, prelude/coda). It is the strongest test of C1
   (does needed r grow with dependency depth when r can go to 32+?) and of C3 (it was trained on a heavy-tailed
   r distribution, so "beyond trained max" is fuzzy; 2610.18127 sees no degradation to 16). Costs: no MLX port
   (PyTorch-MPS with `trust_remote_code`; ~8 GB bf16 weights; r=32 means ~132 effective layers per token, so
   slow), base model only with weak GSM8K (~35–42 at r=32), so tool calling likely needs few-shot plus maybe LoRA.
   Its zero-shot KL exit gives a ready "native halting" baseline for C2.
3. **Retrofitted recurrence checkpoints (`smcleish/Recurrent-{OLMo-2-0425,Llama-3.2,TinyLlama-3T}-train-recurrence-{4..32}`, ~1B).**
   Variable r, cheap to run, same family 2608.18171 used for tool calling. Gives a "retrofit" data point and
   lets us compare train-recurrence 4 vs 32 (a direct handle on C3: does the trained max move the crossover?).
   Weak base capability (math-tuned 1B), so tool accuracy will be low without LoRA SFT. Weight license n/s.
4. **LoopUS Qwen3-1.7B (`Thrillcrazyer/Qwen3_1.7B_LoopUS`, Apache-2.0).** Qwen3 tokenizer/chat template, so
   tool-call formatting is easier than for the base-only models; has a learned confidence halting head (a
   second "native gate" for C2) and up to 8 (tested to 100) steps. Risks: custom repo loader, only one size
   released, base performance near Qwen3-1.7B, and its gate fires very late per 2610.18127.
5. **Nanbeige4.2-3B (Apache-2.0, agentic, MLX 4-bit exists).** Best actual tool caller among looped models,
   but only `num_loops: 2` trained, so the action set is {1, 2} plus extrapolation to 3–4. Useful as a
   "C3 on a production agentic looped model" probe (does a 3rd pass help deep chains and hurt shallow ones?) and
   as a real-world 1-vs-2 router test. Needs the OptiQ MLX loader or the MPS patches from 2608.13987;
   whether `num_loops` can be changed at load time must be checked in `modeling_nanbeige.py`.
6. **Parcae-770m / 1.3B (SandyResearch).** Clean variable-depth (to 24) base models, but too weak for tool
   calling few-shot and need `parcae-lm`. Only useful for a synthetic dependency-depth probe (e.g. p-hop /
   composed function evaluation) to show C1 outside tool use.
7. **Loopie-6B-A0.6B-preview** (fixed 2 loops, MoE, Qwen3-MoE-like; license unverified, HF card 401) and
   **IFM/LoopedLM-P1/P2** (xLLM-native, ≤1.6B base): low priority.
8. **Not usable now:** Koi-4B / ELR (no public weights found as of 2026-10-15; would be ideal: k=1–4 trained,
   τ3-Bench tool evals, NVIDIA hybrid); LoopCoder-v2 (weights n/s); MoR (no official weights); Hyperloop,
   SMELT, Looped-MoE (no weights found).

Training-free looping of a standard model (2605.23872) is a cheap non-looped control: apply our needed-depth
analysis to Qwen3-4B with a looped mid-block to see whether C1 is a property of looped *training*.

---

## 3. Threats to novelty (C1–C4)

- **2608.18171 (Looped LMs improve compositional tool calling).** Same models (Ouro-1.4B/2.6B), same benchmarks
  (NESTful, BFCL, API-Bank), claims "recurrent computation benefits compositional and dependency-aware tool use"
  and "adaptive inference gives a better trade-off". Overlaps the *direction* of C1 and the motivation of C4.
  Gaps we fill: no per-request needed-depth, no control for #calls/length, depth curves only aggregate, adaptive
  = Ouro gate thresholds only (no oracle comparison, no learned controller), retrofitted models only at 1/2/4/8
  and never adaptively. Their Table 1 lists Ouro at depth 8 although Ouro max is 4: worth checking whether they
  already report Ouro >4 behaviour on NESTful (C3 threat if so).
- **2610.18127 (Monitoring latent reasoning).** Ouro gate fires 3.2–4.1 iterations late, depth use does not track
  MATH difficulty, Ouro degrades past 4. Partly covers C2 and the average part of C3 (on math, per token).
- **2607.20519, 2604.07822, LoopRPT 2603.19714, STARS 2605.26733:** already in the findings doc.
- **2610.12582 ELR/Koi-4B.** Budget-conditioned loop count with τ3-Bench and Terminal-Bench gains per k. They list
  per-request adaptive depth as future work, so C4 is open, but they are the most likely group to publish it next
  and they have the stronger agentic model.
- **2610.06833 (learned depth prior)** and **MoR (per-token routers)**: learned *training-time* depth allocation,
  per token. Not per-request, not tool use, but reviewers will ask why a per-request bandit beats a learned router.
- **LoopUS confidence head**, **Huginn KL exit**, **CHASE**: alternative native halting rules; C2 should be shown
  on more than Ouro's gate to be general.
- **2609.19934 (Depth Control Protocol):** not a novelty threat but a methodological one: truncation curves mix
  #applications, distinct computation and readout distribution shift. Our "needed passes" uses truncation, so C1
  should add their linchpin control (full applications, one distinct iteration) or at least discuss it.
- Not found: any paper doing per-request loop-count selection by bandit/RL for tool use, or splitting
  beyond-max-loop effects by dependency depth on a pretrained looped LM (searches today; not exhaustive).

---

## 4. Known dead ends / cautions

- **Extrapolating Ouro beyond 4 loops on average hurts** (Ouro paper; 2610.18127; STARS); Thinking variants
  degrade already after 1–2. Only by-structure splits (our C3) show gains.
- **More than 2 loops did not help** in production-style models trained with 2 (LoopCoder-v2: ≥3 regress;
  Nanbeige: more passes "marginal"; Loopie fixed at 2). Extrapolation on these will likely be flat or negative.
- **Early exit without KV-cache changes saves no compute** in Ouro (2610.09013); vLLM always runs all 4 loops.
  Any latency claim needs best-available or last-loop KV (2610.09013, ELR) and measured wall-clock.
- **Naive block repetition of a frozen non-looped model degrades it** (2605.23872); our own frozen-hybrid
  layer-removal results agree that frozen dense/hybrid models do not tolerate depth edits.
- **Upcycling into loops < training from scratch** (Nanbeige report); retrofits need ~50B tokens (McLeish) or
  depth curricula; LoopUS gains are +2 points and MMLU can drop.
- **Dense looping at matched compute is weaker than adding params** (φ = 0.46, 2604.21106); MoE or
  hyper-connections help (2605.09165, Hyperloop, SMELT). Do not frame looping as a free lunch.
- **Re-read vs carry loops behave differently** (2610.18127): re-read (Huginn, Parcae, Rec-OLMo) plateau without
  degrading and are noise-robust; carry loops (Ouro, LoopUS) decide early and degrade. Generality of C3 may hold only
  for carry loops: state this as a hypothesis.
- **Tooling traps:** Ouro needs transformers <4.56; Ouro-2.6B HF card says 24 layers (project page: 48);
  stock mlx-lm lacks Ouro and Nanbeige classes; Nanbeige has 5 MPS bugs out of the box; Parcae and IFM
  checkpoints need their own libraries; Parcae card model ids differ from repo ids; Huginn's custom cache class
  may break with HF updates.

Sources (fetched 2026-10-15): arxiv.org/abs and /html pages for every id marked [V]; huggingface.co pages for
ByteDance/Ouro-1.4B, ByteDance/Ouro-2.6B, tomg-group-umd/huginn-0125, SandyResearch/parcae-1.3b,
Thrillcrazyer/Qwen3_1.7B_LoopUS, Nanbeige/Nanbeige4.2-3B (+config.json), mlx-community/Nanbeige4.2-3B-OptiQ-4bit,
collections/tomg-group-umd/retrofitting-recurrence; ouro-llm.github.io; github.com/mcleish7/retrofitting-recurrence,
github.com/GeorgiyDemo/ouro-mlx, github.com/ifm-ai/xllm-loop; thrillcrazyer.github.io/LoopUS.
