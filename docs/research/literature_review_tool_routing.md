# Dynamic compute routing for tool-calling language models: a literature review

Written 2026-09-23. Scope: work relevant to extending DHT (Meng et al., 2024), which chooses
layers and attention heads per input with contextual bandits, from text classification to
**tool (function) calling** with a small decoder-only model. Status markers: **[peer]**
peer-reviewed venue; **[preprint]** arXiv only; **[withdrawn]**. Abstracts were read for every
paper marked ✓ in the reference list; numbers for the others are as reported in search
summaries of the papers and should be checked against the PDF before being cited in a paper.

## 1. The question

Can a small tool-calling model decide, **per request**, how much of itself to run (which
layers, which heads) so that easy calls are cheap and hard calls keep full capacity, and can
that decision be learned with bandits or reinforcement learning using the objective
correct/incorrect signal that tool-calling benchmarks provide?

The literature splits into four strands that each touch part of this question:

| Strand | What is routed | Typical learner | Tool-calling specific? |
| --- | --- | --- | --- |
| A. Model routing and cascades | the request, to one of several whole models | supervised classifier; some bandits | one paper (Switchcraft) |
| B. In-model dynamic compute | layers / heads / tokens inside one model | differentiable gates; rarely RL | one line of work (LayerRoute) |
| C. Reasoning-budget control | how many reasoning tokens before answering | RL (GRPO / PPO) or prompting | one paper (Brief Is Better) |
| D. Tool-calling substrate | — (benchmarks, data, small models) | supervised fine-tuning | yes |

## 2. Starting point: DHT

Meng et al. (2024) **[peer]** route each sample through a BERT-base encoder: LinUCB picks depth
and head count, and a contextual combinatorial Thompson sampler (per-head Bayesian logistic
regression) picks which heads run. Decisions are made once per sample from a 3-layer prefix, and
the bandits are trained jointly with the network from a loss×cost reward. The paper reports
AG News F1 on par with full BERT at 0.35–0.81× inference FLOPs. Our reproduction reached .940/.941
macro F1 (tree/integrated) against .949 for full BERT at ~0.4× task FLOPs, and needed batched
execution before those FLOP savings became wall-clock savings. Semantic Scholar lists 17 citing
papers, none of which extends the method, so related work has to come from the wider field.

## 3. Strand A: routing a request between whole models

**Tool-calling routers.** *Switchcraft* (Agarwal et al., 2026) **[preprint]** is the only router
built specifically for agentic tool calling that we found. A 66M-parameter DistilBERT classifier
reads the query and agent context and picks the cheapest of eight models expected to produce a
correct call, trained on an evaluation suite built from five function-calling benchmarks. It
matches the best single model (82.9% vs 82.3%) at 84% lower cost. Two findings matter for us:
larger models were **not** consistently better at tool calls, and simple heuristics (route by
number of tools, input length, or turns) underperformed the learned router, so tool-call
difficulty is not trivially readable from surface features. *Toward Super Agent System with
Hybrid AI Routers* (Yao et al., 2025) **[preprint]** is a position paper arguing for
local/cloud routing inside agent systems; it has no experiments to compare against.

**General LLM routers** (built for chat, used here as method background):
*Hybrid LLM* (Ding et al., ICLR 2024) **[peer]** and *RouteLLM* (Ong et al., ICLR 2025)
**[peer]** train supervised small-vs-large routers; *FrugalGPT* (Chen et al., TMLR 2024)
**[peer]** uses a cascade. Two routers use the same learner family as DHT: *MixLLM* (Wang et al.,
NAACL 2025) **[peer]**, a contextual bandit, and *PILOT* (Panda et al., Findings of EMNLP 2025)
**[peer]**, LinUCB under a budget constraint. PILOT is the closest methodological analogue to a
budgeted version of DHT.

**Cascades and escalation.** *Language Model Cascades: Token-Level Uncertainty and Beyond*
(Gupta et al., 2024) **[preprint]** shows sequence-level uncertainty has a length bias and
learns deferral rules from token-level uncertainty. *When Does Confidence-Based Cascade Deferral
Suffice?* (Jitkrittum et al., NeurIPS 2023) **[peer]** characterises when simple confidence
thresholds are optimal. *Signed Rescue Routing* (Wang et al., 2026) **[preprint]** argues
escalation should be ranked by the *signed* gain (large model fixes vs. breaks the small
model's answer), not by uncertainty; evaluated with Qwen3-4B→8B on multiple-choice tasks.
*Bayesian Self-Escalation in Hierarchical LLM Agents* (Shaikh, 2026) **[preprint]** makes the
escalation decision **during** generation as an optimal-stopping problem with a time-varying
threshold; its experiment is a Qwen2.5-Coder 1.5B→7B cascade on MBPP (code, not tool calls).

*Relevance:* strand A answers "which model", not "how much of this model". It supplies the
evaluation habits we should copy (cost-accuracy curves, heuristic baselines, a best-single-model
control) and, in PILOT/MixLLM, evidence that bandit routers are competitive.

## 4. Strand B: dynamic compute inside one model

**Agent-specific.** *LayerRoute* (Sikdar, 2026) is the only in-model routing work aimed at agents.
Version 1 (arXiv 2606.01838) **[withdrawn]** claimed tool-call steps skip more layers than
planning steps in Qwen2.5-0.5B-Instruct; the author withdrew it after the effect failed to
reproduce across 10 seeds (range −19.6% to +5.1%). Version 2 (arXiv 2609.13682) **[preprint]**
reframes the claim: per-layer straight-through gates plus rank-8 LoRA, trained end to end
(no RL) in under 7 minutes, converge to the **same** skip-eligible band (layers 8–16) in all
10 seeds, gate decisions change outcomes for 87–100% of held-out samples, perplexity improves,
and wall-clock speedup is **1.02–1.06×**. Takeaway: a learned router can look input-dependent
while delivering little more than a fixed pruning pattern, which is why a
fixed-architecture-at-matched-compute control is essential.

**Layer skipping and early exit in LLMs (not tool-specific).** Learned routers without RL:
*Router-Tuning* (He et al., EMNLP 2025) **[peer]**, *FlexiDepth* (Luo et al., COLM 2025)
**[peer]**, *Dr.LLM* (Heakl et al., ICLR 2026) **[peer]**, routers supervised by MCTS search;
*GateSkip* (Laitenberger et al., ICLR 2026) **[peer]**, smooth residual gates, ~15% savings on
long-form reasoning and near-baseline quality at ~50% savings on instruction-tuned models;
*Mixture-of-Depths* (Raposo et al., 2024) **[preprint]**. Confidence-based exits: *CALM*
(Schuster et al., NeurIPS 2022) **[peer]**, *LayerSkip* (Elhoushi et al., ACL 2024) **[peer]**,
*SkipDecode* (Del Corro et al., 2023) **[preprint]**, which forces exit depth to be monotone
decreasing so batching and the KV cache keep working. With RL: *LEDE* (Zhu et al., 2026)
**[preprint]** uses offline RL to pick an exit layer and speculation length per decoding step.

**Head-level routing.** *Mixture of Attention Heads* (Zhang et al., EMNLP 2022) **[peer]** and
*MoH* (Jin et al., 2024) **[preprint]** route each token to a top-k subset of heads; MoH-LLaMA3-8B
used 75% of heads with a reported +2.4 average accuracy after continued training. Both use a
learned softmax router, not bandits, and route per token, not per request.

**Reported limits on real speedups.** Several papers find that per-token dynamic depth rarely
turns into wall-clock gains in autoregressive decoding: skipped layers leave holes in the KV
cache that must be copied (quality loss) or recomputed (lost savings) (*River-LLM*, 2026;
*FFN-SkipLLM*, 2024); in batched decoding the batch waits for its deepest token, capping even
oracle early exit at ~1.14× in SLEB's analysis (Song et al., ICML 2024) **[peer]**; gains shrink
with batch size (*LEAP*, 2026); and modern LLMs show diminishing early-exit returns (*The
Diminishing Returns of Early-Exit Decoding in Modern LLMs*, 2026). Reasoning is fragile:
FlexiDepth reports a weaker router dropping GSM8K from 0.657 to 0.131.

*Relevance:* per-request routing, as in DHT, sidesteps the KV-cache and batching problems
because every token of a response shares one architecture. Per-token routing is where most of
the reported failures sit.

## 5. Strand C: controlling how much the model reasons

*Brief Is Better* (Qi, 2026) **[preprint]** is the one tool-calling study here. On BFCL v3
*multiple* with Qwen2.5-1.5B-Instruct, ~32 reasoning tokens before the call raise accuracy
from 44% to 64%, while 256 tokens drop it to 25%; the short reasoning acts as a
function-selection step (wrong-function rate 30.5% → 1.5%). Outside tool calling, RL is the
standard tool for budget control: *L1* (Aggarwal & Welleck, COLM 2025) **[peer]**, *Thinkless*
(Fang et al., NeurIPS 2025) **[peer]**, *AdaptThink* (Zhang et al., EMNLP 2025) **[peer]**, and
*S-GRPO* (Dai et al., 2025) **[preprint]**, which rewards earlier correct exits from a chain of
thought.

*Relevance:* this is compute measured in tokens rather than layers, but it shows (i) the
optimal compute for tool calls is small and request-dependent, and (ii) GRPO-style group-relative
rewards over sampled attempts are an established recipe for learning compute policies.

## 6. Strand D: benchmarks, data and small tool-calling models

- **Benchmark.** *BFCL* (Patil et al., ICML 2025) **[peer]** grades calls by parsing them into
  an abstract syntax tree and checking function, required arguments, types and values
  (order-insensitive; alternatives allowed), plus executable checks and *irrelevance* cases
  where the correct output is no call. This gives the objective 0/1 reward a bandit needs.
- **Training data.** *APIGen* / *xlam-function-calling-60k* (Liu et al., NeurIPS 2024 Datasets
  & Benchmarks) **[peer]**: 60k calls verified by format, execution and semantic checks.
- **Earlier foundations.** *Toolformer* (Schick et al., NeurIPS 2023) **[peer]**; *Gorilla*
  (Patil et al., NeurIPS 2024) **[peer]**.
- **Small models.** *Hammer* (Lin et al., 2024) **[preprint]**, Qwen-based 0.5–7B
  function-calling models with function masking and an irrelevance set; *TinyAgent* (Erdogan
  et al., EMNLP 2024 Demos) **[peer]**, 1.1B/7B on-device agents with tool retrieval to shorten
  prompts; *Small Language Models for Efficient Agentic Tool Calling* (Jhandi et al., AAAI 2026
  workshop) **[peer, workshop]**, a fine-tuned OPT-350M reporting a 77.55% ToolBench pass rate;
  *Small Reasoning Models are Instruction Followers in Function Calling* (Taheri et al., 2026)
  **[preprint]**, which delegates call formatting to a smaller instruction-following model.
  Independent evaluations (TinyLLM, 2025) find sub-2B models reasonable on single-turn BFCL but
  near zero on multi-turn.

*Relevance:* single-turn BFCL with a ~0.5B model is a feasible, objectively graded testbed;
multi-turn is not, at this scale.

## 7. Synthesis

| Work | Routes | Granularity | Learner | Tool calls | Real speedup reported |
| --- | --- | --- | --- | --- | --- |
| DHT (2024) | layers + heads | per input | LinUCB + Thompson sampling | no (classification) | not reported (ours: 1.12× inference after batching) |
| Switchcraft (2026) | whole models | per request | supervised classifier | **yes** | cost −84% (API pricing) |
| PILOT / MixLLM (2025) | whole models | per request | contextual bandit | no | cost only |
| LayerRoute v2 (2026) | layers | per input | straight-through gates + LoRA | agentic data | 1.02–1.06× |
| Dr.LLM / FlexiDepth / Router-Tuning | layers | per token or per layer | supervised / MCTS labels | no | mixed |
| MoA / MoH | heads | per token | softmax router | no | not the focus |
| LEDE (2026) | exit layer | per decoding step | offline RL | no | 2.0–2.7× vs token-by-token decoding (with speculation) |
| Brief Is Better (2026) | reasoning length | per request | prompting / templates | **yes** | fewer tokens |
| S-GRPO / L1 / Thinkless | reasoning length | per request | RL | no | fewer tokens |

**Patterns.**
1. Routing that is *tool-calling specific* exists only at the whole-model level (Switchcraft)
   or reasoning-length level (Brief Is Better).
2. In-model routing for agents has one line of work (LayerRoute), trained without RL, with a
   withdrawn first version and ~1.04× real speedup in the second.
3. RL for compute allocation is common at the token-budget level (GRPO family) and rare at the
   architecture level (LEDE, offline; BlockDrop/SkipNet in vision).
4. Per-token dynamic depth repeatedly fails to deliver wall-clock savings; per-request
   decisions do not have the KV-cache or batching problem.

## 8. Gap and positioning

We found no work that routes **layers and heads inside a model, per request, for tool calling,
with a bandit or RL policy trained on an objective correctness reward**. The proposed project
sits between Switchcraft (per-request, tool-calling, but between models) and LayerRoute
(in-model, agentic, but gradient-trained gates with little real speedup), and inherits DHT's
bandit formulation, replacing its loss×cost ratio with `𝟙[call correct] − λ·compute`.

Minimum evaluation implied by the literature:
- a **fixed architecture at matched average compute** (LayerRoute's lesson);
- simple **heuristic routers** (number of tools, prompt length; Switchcraft's baselines);
- **wall-clock** throughput at realistic batch sizes, not only FLOPs (SLEB, LEAP, our D017);
- **≥3 seeds** with variance (LayerRoute v1's failure mode);
- per-category BFCL results, including irrelevance, since skipping may raise false calls.

## 9. Open questions

1. Is tool-call difficulty predictable from a shallow prefix? Switchcraft's weak heuristics
   suggest surface features are insufficient; whether early hidden states suffice is untested.
2. Does per-request routing beat the best fixed sub-network, or does it collapse to one
   (LayerRoute v2's same-band convergence)?
3. Is a bandit enough, or does a GRPO-style policy over sampled architectures learn a better
   frontier for the same training budget?
4. Can a single switch point (reasoning text → call emission) add adaptivity without the
   per-token costs catalogued in §4?

## References

✓ = abstract or proceedings page read on 2026-09-22/23.

- Meng, Zhang, Chen, Wang. Sample-Based Dynamic Hierarchical Transformer with Layer and Head
  Flexibility via Contextual Bandit. *Proceedings on Engineering Sciences* 6(2), 2024. arXiv:2312.03038 ✓
- Agarwal, Namyar, Wolman, Ambavat, Gupta, Zhang. Switchcraft: AI Model Router for Agentic Tool Calling. arXiv:2605.07112, 2026 ✓
- Yao et al. Toward Super Agent System with Hybrid AI Routers. arXiv:2504.10519, 2025 ✓
- Ding et al. Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing. ICLR 2024. arXiv:2404.14618
- Ong et al. RouteLLM: Learning to Route LLMs from Preference Data. ICLR 2025. arXiv:2406.18665 ✓
- Chen, Zaharia, Zou. FrugalGPT. TMLR 2024. arXiv:2305.05176 ✓
- Wang et al. MixLLM: Dynamic Routing in Mixed Large Language Models. NAACL 2025. arXiv:2502.18482 ✓
- Panda et al. Adaptive LLM Routing under Budget Constraints (PILOT). Findings of EMNLP 2025. arXiv:2508.21141 ✓
- Gupta, Narasimhan, Jitkrittum, Rawat, Menon, Kumar. Language Model Cascades: Token-Level Uncertainty and Beyond. arXiv:2404.10136, 2024 ✓
- Jitkrittum et al. When Does Confidence-Based Cascade Deferral Suffice? NeurIPS 2023. arXiv:2307.02764
- Wang, Li, Song, Chen, Song, Liu. Signed Rescue Routing: Harm-Aware Cascades for Efficient LLM Inference. arXiv:2609.07786, 2026 ✓
- Shaikh. Knowing When to Ask for Help: Bayesian Self-Escalation in Hierarchical LLM Agents. arXiv:2608.24087, 2026 ✓
- Sikdar. LayerRoute: Input-Conditioned Adaptive Layer Skipping via LoRA Fine-Tuning for Agentic Language Models. arXiv:2606.01838, 2026 (withdrawn) ✓
- Sikdar. LayerRoute: Adaptive Layer-Skipping with LoRA-Preserved Quality for Efficient LLM Inference. arXiv:2609.13682, 2026 ✓
- He et al. Router-Tuning. EMNLP 2025. arXiv:2410.13184 ✓
- Luo, Wang, Yan. Adaptive Layer-skipping in Pre-trained LLMs (FlexiDepth). COLM 2025. arXiv:2503.23798 ✓
- Heakl et al. Dr.LLM: Dynamic Layer Routing in LLMs. ICLR 2026. arXiv:2510.12773 ✓
- Laitenberger et al. What Layers When: Learning to Skip Compute in LLMs with Residual Gates (GateSkip). ICLR 2026. arXiv:2510.13876 ✓
- Raposo et al. Mixture-of-Depths. arXiv:2404.02258, 2024 ✓
- Schuster et al. Confident Adaptive Language Modeling (CALM). NeurIPS 2022. arXiv:2207.07061 ✓
- Elhoushi et al. LayerSkip. ACL 2024. arXiv:2404.16710 ✓
- Del Corro et al. SkipDecode. arXiv:2307.02628, 2023 ✓
- Zhu et al. Experience-Driven Dynamic Exits for LLMs with Reinforcement Learning (LEDE). arXiv:2606.03113, 2026 ✓
- Zhang et al. Mixture of Attention Heads. EMNLP 2022. arXiv:2210.05144 ✓
- Jin et al. MoH: Multi-Head Attention as Mixture-of-Head Attention. arXiv:2410.11842, 2024 ✓
- Song et al. SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks. ICML 2024. arXiv:2402.09025
- River-LLM: Large Language Model Seamless Exit Based on KV Share. arXiv:2604.18396, 2026
- FFN-SkipLLM. arXiv:2404.03865, 2024
- LEAP: Layer-wise Exit-Aware Pretraining for Efficient Transformer Inference. arXiv:2605.01058, 2026
- The Diminishing Returns of Early-Exit Decoding in Modern LLMs. arXiv:2603.23701, 2026
- Qi. Brief Is Better: Non-Monotonic Chain-of-Thought Budget Effects in Function-Calling Language Agents. arXiv:2604.02155, 2026 ✓
- Aggarwal, Welleck. L1. COLM 2025. arXiv:2503.04697 ✓
- Fang, Ma, Wang. Thinkless. NeurIPS 2025. arXiv:2505.13379 ✓
- Zhang et al. AdaptThink. EMNLP 2025. arXiv:2505.13417 ✓
- Dai et al. S-GRPO: Early Exit via Reinforcement Learning in Reasoning Models. arXiv:2505.07686, 2025
- Patil et al. The Berkeley Function Calling Leaderboard (BFCL). ICML 2025 ✓
- Liu et al. APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets. NeurIPS 2024 D&B. arXiv:2406.18518
- Schick et al. Toolformer. NeurIPS 2023. arXiv:2302.04761
- Patil et al. Gorilla: Large Language Model Connected with Massive APIs. NeurIPS 2024. arXiv:2305.15334
- Lin et al. Hammer: Robust Function-Calling for On-Device Language Models via Function Masking. arXiv:2410.04587, 2024 ✓
- Erdogan et al. TinyAgent: Function Calling at the Edge. EMNLP 2024 System Demonstrations ✓
- Jhandi, Kazi, Subramanian, Sendas. Small Language Models for Efficient Agentic Tool Calling. AAAI 2026 Workshop. arXiv:2512.15943 ✓
- Taheri, Heydari, Naaman, Fatemi. Small Reasoning Models are Instruction Followers in Function Calling. arXiv:2608.22472, 2026 ✓
- TinyLLM: Evaluation and Optimization of Small Language Models for Agentic Tasks on Edge Devices. arXiv:2511.22138, 2025
