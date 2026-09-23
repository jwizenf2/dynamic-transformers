# Next research topics: dynamic transformers, especially with RL

Written 2026-09-22 after the AG News fixed / T-DHT / F-DHT runs. Every citation below
was checked against arXiv, ACL Anthology, OpenReview or proceedings pages, and
corrections were applied (e.g. FrugalGPT is TMLR 2024; RouteLLM's published title says
"from"; PonderNet appeared at an ICML 2021 *workshop*). Listing a paper here is not a
novelty claim about any proposed project: check recent literature again before
committing to one.

## Where DHT sits, and what our runs taught us

DHT (Meng et al., 2024) makes **per-sample, one-shot** decisions with **contextual
bandits** (LinUCB for depth/count, logistic Thompson sampling for head identity), trained
**jointly** with the network. Among the methods below, only a handful use RL at all
(BlockDrop, SkipNet, Runtime Neural Pruning, TR-BERT, ConsistentEE, NAS-RL, ENAS and the
2025 "when to think" LLM papers), and bandits are rarer still (CeeBERT, MixLLM, PILOT).
That gap is itself an opportunity.

Our measured observations, each of which motivates a topic below:

1. **Routing trend is the reverse of the paper's Fig. 2.** Depth 3 takes 89–97% of
   samples at the start, then falls to ~55–60%. The loss×depth ratio reward
   plausibly rewards shallowness before the model has learned anything (topic A1, A5).
2. **Reward ratios hit the cap 81,574 times** in one T-DHT run: the ratio reward is
   badly scaled when cross-entropy approaches 0 (A1, A5).
3. **Depth-3 misroutes:** ~46% of the hardest samples still exit at depth 3, and most
   of T-DHT's accuracy loss comes from them. A one-shot decision at layer 3 can't
   revise itself after seeing layer 6 (A3).
4. **FLOPs ≠ wall-clock time.** 0.39× task FLOPs trained 1.58× *slower* per sample
   until batched execution was added (D017). Any compute claim needs measured time (B2).
5. **AG News inputs are ~60% padding** at length 128 (median 51 tokens), so token-level
   savings may dwarf head-level savings (B2).

---

## A. Direct extensions of DHT (best fit for a semester project)

### A1. Budget-constrained routing (bandits with knapsacks)
Replace the ratio reward with an explicit objective: maximise accuracy subject to an
average compute budget, e.g. a Lagrangian reward `correct − λ·FLOPs` with λ updated
online (primal-dual). Then compare the frontier traced by λ with DHT's single point.

- Badanidiyuru, Kleinberg, Slivkins. *Bandits with Knapsacks.* FOCS 2013. arXiv:1305.2545.
- Badanidiyuru, Langford, Slivkins. *Resourceful Contextual Bandits.* COLT 2014. arXiv:1402.6779.
- Agrawal, Devanur. *Linear Contextual Bandits with Knapsacks.* NeurIPS 2016. arXiv:1507.06738.
- Agrawal, Devanur, Li. *An efficient algorithm for contextual bandits with knapsacks, and an
  extension to concave objectives.* COLT 2016. arXiv:1506.03374.
- Panda et al. *Adaptive LLM Routing under Budget Constraints* (PILOT, LinUCB + budget).
  Findings of EMNLP 2025. arXiv:2508.21141. The closest modern analogue.

**First experiment:** add `reward_rule = "lagrangian"` beside the literal ratio, and
sweep 3–4 budgets on AG News with batched execution. Directly addresses findings 1–2.

### A2. Neural contextual bandits for the router
LinUCB over hand-pooled 84-dim features is a strong linearity assumption. Neural bandits
learn the reward model and keep principled exploration.

- Zhou, Li, Gu. *Neural Contextual Bandits with UCB-based Exploration* (NeuralUCB). ICML 2020. arXiv:1911.04462.
- Zhang, Zhou, Li, Gu. *Neural Thompson Sampling.* ICLR 2021. arXiv:2010.00827.

**First experiment:** swap `LinearUCB` for a small NeuralUCB over the same contexts
(the interface is `scores/select/update`); keep everything else fixed.

### A3. From one-shot bandits to sequential RL (exit/continue as an MDP)
Model routing as an episode: after each c-layer group the policy sees the current
representation and chooses {exit, continue with k heads}. The reward is task correctness
minus compute. This fixes finding 3: the decision is revisited with deeper evidence.

- Wu et al. *BlockDrop: Dynamic Inference Paths in Residual Networks.* CVPR 2018. arXiv:1711.08393.
  (policy gradient drops residual blocks per input)
- Wang, Yu, Dou, Darrell, Gonzalez. *SkipNet: Learning Dynamic Routing in Convolutional Networks.*
  ECCV 2018. arXiv:1711.09485. (hybrid supervised + RL gating)
- Lin, Rao, Lu, Zhou. *Runtime Neural Pruning.* NeurIPS 2017. (per-input channel pruning as an MDP,
  Q-learning)
- Zeng et al. *ConsistentEE: A Consistent and Hardness-Guided Early Exiting Method for Accelerating
  Language Models Inference.* AAAI 2024. arXiv:2312.11882. **The closest RL competitor for
  BERT-style classification.**

**First experiment:** REINFORCE / actor-critic exit policy at layers 3/6/9 on the same
BERT backbone, compared with T-DHT/F-DHT at matched average depth.

### A4. Off-policy evaluation from logged routing decisions
Every training step already logs each sample's context, action and reward
(`metrics.jsonl`, bandit statistics). Doubly-robust off-policy evaluation could score a
*new* routing policy or reward rule without a 3-hour retrain, turning reward design into
a fast loop. The caveat to state clearly: the policy also changes the network being
trained, so OPE is exact for frozen-model inference routing and only approximate for
joint training.

- Dudík, Langford, Li. *Doubly Robust Policy Evaluation and Learning.* ICML 2011. arXiv:1103.4601.

**First experiment:** log propensities (UCB is deterministic, so add ε-exploration or use
TS's sampling probabilities), then evaluate candidate inference-time routers on held-out
validation exports.

### A5. Reward-design ablations (cheapest, strongest paper-grounded study)
Literal vs corrected binary feedback (already implemented), cap values, ratio vs
difference rewards, per-group vs shared history, and warm-starting routing only after
the task loss stabilises. Finding 1 (reversed dynamics) is the headline question:
**which reward/initialisation reproduces the paper's Fig. 2 trend?** Multiple seeds are
needed; one seed cannot establish variance.

---

## B. Related dynamic-transformer families (baselines and related work)

### B1. Early exit (adaptive depth by confidence), the main competitor class
- Xin et al. *DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference.* ACL 2020. arXiv:2004.12993.
- Liu et al. *FastBERT: a Self-distilling BERT with Adaptive Inference Time.* ACL 2020. arXiv:2004.02178.
  (a baseline in the DHT paper)
- Zhou et al. *BERT Loses Patience: Fast and Robust Inference with Early Exit* (PABEE). NeurIPS 2020.
  arXiv:2006.04152.
- *CeeBERT: Cross-Domain Inference in Early Exit BERT.* Findings of ACL 2024. arXiv:2405.15039.
  **A multi-armed bandit over exit thresholds**: the most direct bandit comparison for DHT.
- Schuster et al. *Confident Adaptive Language Modeling* (CALM). NeurIPS 2022. arXiv:2207.07061.
- Elhoushi et al. *LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding.* ACL 2024.
  arXiv:2404.16710.

*Project:* add a DeeBERT/PABEE baseline to the same harness; it's cheap and makes the
DHT comparison honest (the paper only compares against FastBERT/DistilBERT).

### B2. Token-level dynamic compute (likely larger savings on padded short text)
- Goyal et al. *PoWER-BERT.* ICML 2020. arXiv:2001.08950.
- Kim et al. *Learned Token Pruning for Transformers.* KDD 2022. arXiv:2107.00910.
- Ye, Lin, Huang, Sun. *TR-BERT: Dynamic Token Reduction for Accelerating BERT Inference.* NAACL 2021.
  arXiv:2105.11618. **REINFORCE decides per-token skipping; tested on Yelp/IMDB-style
  classification, i.e. RL on the paper's own dataset family.**
- Raposo et al. *Mixture-of-Depths.* arXiv:2404.02258 (2024, preprint).
- Bolya et al. *Token Merging: Your ViT But Faster.* ICLR 2023. arXiv:2210.09461.
- Rao et al. *DynamicViT.* NeurIPS 2021. arXiv:2106.02034.
- Yin et al. *A-ViT: Adaptive Tokens for Efficient Vision Transformer.* CVPR 2022. arXiv:2112.07658
  (arXiv title: "AdaViT").

*Project:* a combined "which heads AND which tokens" bandit/RL router; the EDA shows
~60% padding on AG News and 16% truncation on Yelp.

### B3. Head and width selection
- Michel, Levy, Neubig. *Are Sixteen Heads Really Better than One?* NeurIPS 2019. arXiv:1905.10650.
- Voita et al. *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest
  Can Be Pruned.* ACL 2019. arXiv:1905.09418. (cited by the DHT paper)
- Zhang et al. *Mixture of Attention Heads: Selecting Attention Heads Per Token.* EMNLP 2022.
  arXiv:2210.05144. The closest learned-router analogue to DHT's TS head selection.
- Hou et al. *DynaBERT: Dynamic BERT with Adaptive Width and Depth.* NeurIPS 2020. arXiv:2004.04037.
  (width/depth chosen at deployment, not per input)

### B4. Conditional computation / Mixture-of-Experts
- Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer.* ICLR 2017.
  arXiv:1701.06538.
- Fedus, Zoph, Shazeer. *Switch Transformers.* JMLR 2022. arXiv:2101.03961.
- Zhou et al. *Mixture-of-Experts with Expert Choice Routing.* NeurIPS 2022. arXiv:2202.09368.

### B5. Learned halting
- Graves. *Adaptive Computation Time for Recurrent Neural Networks.* arXiv:1603.08983 (2016).
- Dehghani et al. *Universal Transformers.* ICLR 2019. arXiv:1807.03819.
- Banino, Balaguer, Blundell. *PonderNet: Learning to Ponder.* ICML 2021 AutoML Workshop.
  arXiv:2107.05407.

---

## C. RL for adaptive compute in LLMs (frontier directions)

### C1. Learning *how long* / *whether* to think
- Aggarwal, Welleck. *L1: Controlling How Long A Reasoning Model Thinks With Reinforcement
  Learning.* COLM 2025. arXiv:2503.04697.
- Fang, Ma, Wang. *Thinkless: LLM Learns When to Think.* NeurIPS 2025. arXiv:2505.13379.
- Zhang et al. *AdaptThink: Reasoning Models Can Learn When to Think.* EMNLP 2025. arXiv:2505.13417.

This is the natural LLM-era successor of DHT's idea: per-input compute chosen by a learned
policy with a cost-aware reward, applied to test-time tokens instead of layers.

### C2. Choosing *which model* per query (routing and cascades)
- Wang et al. *MixLLM: Dynamic Routing in Mixed Large Language Models.* NAACL 2025. arXiv:2502.18482.
  (contextual bandit)
- Panda et al. *Adaptive LLM Routing under Budget Constraints.* Findings of EMNLP 2025.
  arXiv:2508.21141. (LinUCB + knapsack-style budget)
- Ong et al. *RouteLLM: Learning to Route LLMs from Preference Data.* ICLR 2025. arXiv:2406.18665.
  (supervised; contrast case)
- Chen, Zaharia, Zou. *FrugalGPT.* TMLR 2024. arXiv:2305.05176. (cascade; contrast case)

### C3. Layer skipping inside LLMs
- He et al. *Router-Tuning: A Simple and Effective Approach for Enabling Dynamic-Depth in
  Transformers.* EMNLP 2025. arXiv:2410.13184.
- Luo, Wang, Yan. *Adaptive Layer-skipping in Pre-trained LLMs* (FlexiDepth). COLM 2025. arXiv:2503.23798.
- Jiang et al. *D-LLM: A Token Adaptive Computing Resource Allocation Strategy for Large Language
  Models.* NeurIPS 2024.
- Heakl et al. *Dr.LLM: Dynamic Layer Routing in LLMs.* ICLR 2026. arXiv:2510.12773. (routers
  supervised by MCTS)
- Zhu et al. *LEDE: Experience-Driven Dynamic Exits for LLMs with Reinforcement Learning.*
  arXiv:2606.03113 (2026, not yet peer reviewed). Offline RL for exit layer and speculation length.

None of the first four uses RL; a bandit/RL version of per-token layer skipping in a
small open LLM is a clear gap, but it is a large project.

---

## D. Background: architecture search (static, not per input)

- Zoph, Le. *Neural Architecture Search with Reinforcement Learning.* ICLR 2017. arXiv:1611.01578.
- Pham et al. *Efficient Neural Architecture Search via Parameter Sharing* (ENAS). ICML 2018.
  arXiv:1802.03268.
- Cai et al. *Once-for-All.* ICLR 2020. arXiv:1908.09791. (evolutionary search)
- Wang et al. *HAT: Hardware-Aware Transformers.* ACL 2020. arXiv:2005.14187. (latency-guided)

Useful for the related-work section: DHT's contrast with NAS is *per-sample* and
*during training*.

---

## Recommendation

| Rank | Topic | Why this one | Uses existing code | RL content | Risk |
|---:|---|---|---|---|---|
| 1 | **A3 sequential RL exit policy vs DHT bandits** | Targets our biggest measured weakness (depth-3 misroutes); clean RL-vs-bandit comparison; ConsistentEE gives a published reference point | Backbone, data, batched executor, FLOP/time harness | Policy gradient / actor-critic | Medium |
| 2 | **A1 budget-constrained (knapsack/Lagrangian) routing** | Fixes the clipped, badly scaled ratio reward; produces an accuracy–compute *frontier* instead of one point | Controller + reward module only | Constrained contextual bandits | Low–medium |
| 3 | **A5 reward ablations + multiple seeds** | Cheapest; answers "why is our Fig. 2 reversed?"; needed for any credible claim | Configs only | Bandit reward design | Low |
| stretch | B2 bandit/RL token + head routing, or C1 on a small LLM | Largest potential savings / most topical | Partial | RL | High |

Whatever is chosen: keep the matched fixed baseline, report **measured wall-clock
time** (now meaningful with `execution = "batched"`) next to FLOP estimates, and run
≥3 seeds.
