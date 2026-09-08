# Paper audit

Implementation status: the audit below preserves what was found in the original
PDF. Working UCB, logistic TS and tree/integrated DHT were subsequently implemented
under decisions D005–D008. Statements about Phase 0/1 describe the initial audit;
see `algorithm_mapping.md` and `../code_walkthrough.md` for current code.

Fanfei Meng, Lele Zhang, Yu Chen, Yuxin Wang. *Sample-Based Dynamic Hierarchical
Transformer with Layer and Head Flexibility via Contextual Bandit*. Proceedings on
Engineering Sciences 6(2), 439–452 (2024). DOI: 10.24874/PES06.02.001.

Source: supplied `dynamic_transformers_reference_paper.pdf`, SHA256
`f9117b8504bb952edc5b05e87f74a5f3f72b3822fb96dc7c9373a0698be97ad4`.
PDF page 1 = printed 439; page 14 = printed 452. Entire extracted text read,
including references; rendered PDF pages 3–11 inspected for equations, algorithms,
workflow, experimental lists, tables and features. Page 452 contains only a running
header. No author implementation was used. Many displayed equations lack usable
numbers: names below identify their location rather than inventing numbering.

## Motivation and scope (Sections 1–2, pp. 439–441)

DHT adapts the network to individual text samples during training and inference.
It combines numerical choices (depth/head count) with subset choices (head identity),
without an auxiliary neural controller. The paper contrasts this with distillation,
post-training compression, head importance pruning, and neural architecture search.
The semester title mentions reinforcement learning; this reproduction specifically
uses contextual bandits, not a general multi-step RL algorithm. Novelty assertions
in the paper are authors' claims, not independently verified claims for our project.

## Linear contextual UCB (Section 3.1 p. 441, Algorithm 1)

An arm feature z is concatenated with sample context x: w=[x;z]. With both of
dimension d the aggregate has dimension 2d. Expected reward is wᵀθ_arm.
For selected observations D and scalar rewards y, the dimensionally consistent
ridge solution is θ=(DᵀD+λI)⁻¹Dᵀy. Algorithm 1 maintains separate statistics per arm:

```text
A_arm = λ I_p; b_arm = zeros(p)
θ_arm = solve(A_arm, b_arm)
score_arm = w_armᵀθ_arm + α sqrt(w_armᵀ solve(A_arm,w_arm))
choose argmax score
A_chosen += w_chosen w_chosenᵀ
b_chosen += observed_reward * w_chosen
```

The paper prints d×d statistics despite a 2d aggregate and calls the reward matrix
k×2d. Those dimensions do not yield its scalar-reward parameter vector. Recommended
interpretation: p is actual aggregate width, D is n×p, y is n, A is p×p, b and θ
are p. λ is ridge regularization, α is exploration scale; the reward also calls a
different scale α, so code must distinguish them. This is contextual linear UCB,
not UCB1. Only executed arms receive updates.

## Logistic Thompson semi-bandit (Section 3.2 pp. 441–442)

Each head is an arm. The text initializes a mean g=I_(2d×1) and inverse covariance
s=0_(2d×2d), samples p_arm from N(g_arm,s_arm⁻¹), computes
E_arm=σ(p_armᵀw_arm), and selects the K highest E as a superarm Φ*. It observes
a binary reward for the selected subset and refreshes a Gaussian approximation to
logistic parameters.

The rendered optimization on p. 442 literally uses an argmax over g, a **positive**
quadratic term λ/2 Σ_j s_arm(p_arm−g_arm)², and
Σ_i log σ(r_i p_armᵀw_i,arm), with r declared 0/1. A positive quadratic in a
maximization is not a proper Gaussian log prior; r=0 makes the likelihood term
constant. It also mixes the optimization variable and sampled parameter. The
precision increment is printed as
s_arm ← s_arm + Σ_i σ(pᵀw_i)(1−σ(pᵀw_i)) w_i,arm².
The component square suggests diagonal precision, while initialization is a matrix.

These are unresolved specification defects, not permission to substitute Beta
Bernoulli TS. A proposed executable repair is diagonal Laplace logistic TS:
positive prior precision, penalized binary cross entropy, Gaussian sampling, and
selected-head-only posterior updates using the observed superarm label. Full
covariance is an alternative. See A03–A05 before implementation.

## Attention and architecture search (Sections 4.1–4.2 pp. 442–443)

The masked multi-head output is a sum of ξ_h Att_h(x), ξ_h∈{0,1}; the head
count is Σ_h ξ_h. Retain the original output projection subspace for each selected
head. Masking is a semantic definition and alone does not avoid Q/K/V computation.

Arm features are data-derived, not just one-hot architecture IDs. The paper first
passes all samples through the full network without backpropagation. Depth-arm
features average representations at the corresponding depth; head-count features
average representations after random/even head-count assignments; combined arms
condition on both depth and head count. TSB features split representation matrices
along the embedding dimension into one submatrix per head and average them.
Frequency of rebuilding features and whether the same full pass can produce every
head-conditioned representation are unclear. Never use validation/test data for this
calibration, and account for its cost.

Tree workflow (Algorithm 2 p. 444, Figure 1 p. 445):

1. Run b full-head prefix layers with no task gradient; UCB_L chooses total depth.
2. Put the original sample and routing decision in its depth queue.
3. When ready, rerun the prefix with gradients under current model weights.
4. For each c-layer dynamic group, concatenate context, head-arm feature and
   one-hot layer marker; UCB_H chooses count and TSB chooses identities.
   All c layers use the same relative head positions.
5. Execute exactly the selected total depth, compute loss, update the model,
   compute observed rewards, and update the corresponding bandits.

Integrated search replaces UCB_L/UCB_H with UCB_cp selecting (depth,count) once;
the count stays constant across dynamic groups but TSB may change identities.
Use D001's total-depth interpretation; the paper also writes b+L inconsistently.

## Rewards (Section 4.3 pp. 443–444)

The explicit depth cost is δ_L,t,i = loss(x_t,i) L_t,i.
The printed depth ratio is α times the previous batch's average δ divided by
the current sample's δ. The subscripts change between δ_L, δ_1 and δ with no
explanation. The corresponding head and combined displays use the same form:

```text
r_L(t,i)    = depth_scale * mean_previous_batch(δ_L) / δ_L(t,i)
r_H(t,i,l)  = head_scale  * mean_previous_batch(δ_H at l) / δ_H(t,i,l)
r_cp(t,i)   = pair_scale  * mean_previous_batch(δ_cp) / δ_cp(t,i)
```

The prose refers to products of loss, depth and head count, but the promised
explicit definitions of δ_H and δ_cp are absent in the rendered file. Candidates
are loss×H and loss×L×H respectively, or loss×L×H for both. Neither is verified.
Higher ratios are described as desirable.

The binary equation on p. 444 **literally assigns 0 for r_H (or r_cp)>1, and 1
for ≤1**. Parentheses appear to denote the integrated alternative, not a product
or function application. This conflicts with the prose describing 1 as more
effective and with top-K maximization. Preserve literal and corrected-direction
options separately; do not silently flip the inequality.

Reward scales, initial history, epsilon, saturation/clipping, group missingness,
and the meaning of previous batch across asynchronous depth queues are unspecified.
Proposed safeguards and their fidelity status are in the ambiguity log. No reward
is implemented in Phase 0/1.

## Queuing (Section 5 p. 444; Algorithm 3 p. 445)

Maintain one FIFO per depth. Release a batch when its size reaches q, then clear
the released entries. Routing can use stale prefix weights while a queue fills;
the training pass reruns the prefix. Figure 1 confirms two stages and L=b+oc.
The queue algorithm image is on printed **445**, although its discussion begins
on 444. No policy for epoch-end partial queues, starvation, or within-batch
head heterogeneity is specified. Checkpointing will need original sample IDs,
selected actions, decision-time contexts/features, queue order and data cursor.

## Experiments (Section 6 pp. 445–448)

Published backbone: 12 layers, 12 heads, width 768, b=c=3.

- T-DHT: depth {3,6,9,12}, count {8,10,12}.
- S-DHT: (3,12),(6,12),(9,12),(12,8),(12,10),(12,12).
- F-DHT: (3,12),(6,8),(6,10),(6,12),(9,8),(9,10),(9,12),(12,8),(12,10),(12,12).
- P-DHT: F-DHT for two epochs, then the specialized list below.

Exact P-DHT list transcribed from Section 6.1 p. 446:

```text
1L8H+2L12H, 1L10H+2L12H, 1L12H+2L12H,
4L8H+2L12H, 4L10H+2L12H, 4L12H+2L12H,
7L6H+2L12H, 7L9H+2L12H, 7L12H+2L12H,
10L8H+2L12H, 10L10H+2L12H, 10L12H+2L12H
```

The unusual 6/9 heads in the seven-layer group are as printed. The sentence
“train all but the last two layers by freezing their weights” is self-contradictory.
The intended freeze schedule and identity of final layers need separate treatment.
The 1L8H prefix also contradicts ordinary b=3 full-head protection.

| Dataset | Train | Test | Classes | Sequence length |
| --- | ---: | ---: | ---: | ---: |
| AG News | 120000 | 7600 | 4 | 128 |
| DBpedia | 560000 | 70000 | 14 | 128 |
| Yelp Polarity | 560000 | 38000 | 2 | 300 |
| Yelp Full | 650000 | 50000 | 5 | 300 |

Section 6.2 says DHT is compared “from scratch”; this does not fully establish
which checkpoint/tokenizer/pretraining procedure every model uses. Baselines are
12-layer vanilla BERT, 1/3/6-layer DistilBERT, and FastBERT inference settings.
All Table 1/2 numbers are preserved in `references/paper_results.csv` (64 records).
Table 1's F1 is inference F1 even though its cost column concerns training.
F1 averaging is unspecified. Ratios estimate operations from architecture choices;
74% savings means 1−0.26, not a measured wall-clock reduction.

Figures 2–3 show authors' selected-depth/head trends and decreasing loss. These
plots do not establish reproducibility or statistical uncertainty. Section 7.4
references absent Figures 4/6/7 and uses F-IDHT/S-IDHT names without clarification.

## Features/settings/conclusion (Sections 7–7.4 pp. 448–449)

Mean pooling first over sequence segments then embedding segments reduces
representations. The multiscale example uses 1+2+4 bins. Depth/pair pooling says
sequence 1+2 and embedding 4+8+16, but incorrectly sums the latter to 26 and
later says 3×36; arithmetic yields 3×28=84. Head-count pooling says 3×14=42,
TSB 1×12=12. Section 7.3 instead reports training-vector widths 168,87,13.
168 is consistent with two 84-vectors and 87 with two 42-vectors plus three
group markers, but this is inference; 13 remains unexplained.

Section 7.2 suggests rolling normalization without an algorithm/window/epsilon.
Section 7.3 says adaptive learning rates, typically 3–6 epochs, and GTX 1080,
GTX 2080, Tesla V100 16G runs, without complete per-run settings. No seeds or
uncertainty estimates are supplied. References occupy pp. 449–451; we have not
validated every cited related work. Future extensions remain candidate questions.
