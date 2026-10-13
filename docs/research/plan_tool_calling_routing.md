# Plan: per-request layer-group routing for a hybrid-attention tool-calling model, with a UCT teacher

First written 2026-09-23 after the progress meeting, in which the professor endorsed the
tool-calling direction and suggested (1) Pi as the agent harness and (2) Upper Confidence Trees
(UCT). **Revised 2026-10-13**: the Mac Studio (ai-lab: M5 Max, 128 GB) is available, and the
model family moved from Qwen2.5 to the current hybrid-attention Qwen (Qwen3.5 / Qwen3.8). Background and
citations: `literature_review_tool_routing.md` (§7 UCT/Dr.LLM, §8 RL for tool calling, §9 Pi).
Nothing here has been run yet. Every cost estimate is a guess, to be replaced by Phase 0
measurements.

## Thesis

**Claim under test.** Consider a frozen tool-calling LLM. A small router reads each request once,
using features from the first layer group, and chooses which middle layer groups to skip. It can
cut average compute while keeping BFCL correctness, and it beats the best *static* pruned model
at matched compute.

Sub-questions (each is a reportable result either way):

1. **Headroom.** Do different requests actually need different depths, or is one static pruned
   architecture as good as any per-request choice? (Gate 1.)
2. **Skippable layer types.** The models interleave Gated DeltaNet (linear-attention) layers and
   full-attention layers. Which type tolerates skipping, and how does that change from 0.8B to
   27B? Routing on this hybrid architecture has not been studied; Dr.LLM used standard
   transformers.
3. **How should the router learn?** Compare an online contextual bandit (DHT), imitation of a UCT
   teacher (Dr.LLM-style), GRPO, and Expert Iteration (router-guided search, then retraining).

## Models

The newest open-weight Qwen is **Qwen3.8-27B** (Apache-2.0, released 2026-08). It is a dense
vision-language model with tool calling. Thinking mode is on by default; we disable it with
`enable_thinking=False`. Qwen3.8 has no smaller dense sizes. Qwen3.8-27B's config uses the same
`qwen3_5` architecture as the Qwen3.5 family, which does come in small sizes. That gives a
size ladder within one architecture:

| Model | Role |
| --- | --- |
| Qwen3.5-0.8B, Qwen3.5-4B | development and debugging |
| Qwen3.5-9B | mid-size point |
| Qwen3.5-27B | same size as the 3.8 model, older training |
| **Qwen3.8-27B** | main model |

Architecture (Qwen3.8-27B config): 64 layers in 16 groups of `[DeltaNet, DeltaNet, DeltaNet,
full attention]`, each followed by a feed-forward layer. Hidden size 5,120; feed-forward size
17,408; full attention with 24 query and 4 KV heads; one multi-token-prediction layer. bf16
weights are about 54 GB, which fits in the lab's ~81 GB free for models. Text only: the vision
encoder is not loaded.

**Framework: MLX.** mlx-lm supports `qwen3_5` with a Metal kernel for DeltaNet, but only for
inference: the kernel has no backward pass. Since the model is frozen and routers train on
cached features, this does not block the main plan. It does make joint fine-tuning (a later
ablation) harder. PyTorch on MPS has no fast DeltaNet kernel.

The previous plan's head-count dimension is dropped. Query-head trimming has no counterpart in
DeltaNet layers, and our AG News ablation found that head identity barely matters.

## Datasets

| Role | Dataset | Notes |
| --- | --- | --- |
| Final test (run once per selected router) | **BFCL** single-turn categories: simple, multiple, parallel, parallel_multiple, irrelevance | Graded by an abstract-syntax-tree (AST) check: parse the generated call and compare function name, argument names and allowed values with the answer key. For irrelevance, a request is correct only if no call is made. Never used for search, training or selection. |
| Search, training, validation | **Salesforce xlam-function-calling-60k** | 60k automatically generated and verified (query, tools, answers) examples; the same AST-style check applies. Gated on Hugging Face: terms accepted with the owner's account, token on the lab. |
| Fallback pool | ToolRL's ~4k open mix (ToolACE / Hammer / xLAM subsets) | Used if xLAM is unavailable. |
| Agent evaluation (Phase 3) | 30–50 custom Pi tasks with automatic checks | |

The protocol matches the DHT reproduction: search and train on training data, select on held-out
validation, run the BFCL test once per selected router.

## Objective

The LLM is frozen, so there is no language-model loss; only the router learns. Each request x
and architecture a (the set of middle groups that run) receives the reward

```
r(x, a) = 1[generated call passes the AST check] − λ · cost(a) / cost(full)
```

`cost` is analytical FLOPs, and measured time is reported alongside it. Sweeping λ traces an
accuracy–compute frontier. The headline comparison is the router frontier against the static
pruning frontier. The search stores every (architecture, correct, cost) triple, so one search
serves every λ.

**Router input.** Mean-pooled hidden state after group 1 (always run, like DHT's prefix),
computed once and cached.

**Router losses:**

| Learner | Loss / update |
| --- | --- |
| R1 bandit (LinUCB, DHT-style, `src/dht/bandits/linear_ucb.py`) | per-architecture ridge regression of r on features; act by upper confidence bound |
| R2 UCT → imitation | cross-entropy to the UCT label: the best architecture found, or the soft visit distribution. The architecture may be factored into one binary skip decision per group, with a summed binary cross-entropy. |
| R3 GRPO | sample G architectures per request; `L = −Σᵢ Âᵢ log π(aᵢ \| x)` with `Âᵢ = (rᵢ − mean r)/std r` |
| R4 Expert Iteration | PUCT search guided by the current router, then cross-entropy to the new visit distribution; 1–2 rounds |

## UCT: role and design

The 16 groups are reduced to 14 decisions by always running the first group (it produces the
features) and the last. That gives 2¹⁴ = 16,384 architectures per request, too many to
enumerate at 27B, which is why search is needed.

- **Tree.** Decisions are made in depth order. A node at depth k fixes skip/run for the first k
  middle groups.
- **Selection.** UCB1: choose the child maximising `Q + c·√(ln N_parent / N_child)`.
- **Rollout.** Complete the remaining decisions with a default policy (run all remaining
  groups), generate the call greedily, run the AST check, and compute r.
- **Backup.** Propagate r along the path.
- **Prefix sharing.** Paths that share their first k decisions compute identical prefill
  activations up to group k. These are cached at tree nodes, which cuts the dominant prefill
  cost. Decoding diverges per architecture and is batched across leaves.
- **Variant.** The tree can be extended to {skip DeltaNet triplet, skip attention layer} per
  group (4¹⁴) to answer sub-question 2 directly.

UCT is used offline:
1. **Headroom measurement (Gate 1).** It finds a near-cheapest correct architecture per
   training request. To avoid a lucky-pass bias (many tries with a 0/1 reward), the headroom
   number comes from a probe evaluated on *held-out* requests, not from the per-request best.
2. **Teacher** for R2 and R4.

At deployment the router is a single forward pass of a small network; UCT does not run. Whether
the professor intended UCT as the online decision rule itself remains an open question (below).

## Phases and go/no-go gates

### Phase 0: infrastructure and baselines (ai-lab)
- Push the repo to `wizenfeld-lab`; set up the lab environment; download bf16 weights, BFCL
  and xLAM (explicit opt-in).
- MLX routable forward that skips layer groups, with an equivalence test for the full
  architecture against the stock model.
- Qwen chat template with tools and `enable_thinking=False`; a call parser; an AST checker
  (from `bfcl-eval` if usable offline, otherwise reimplemented).
- Baselines on validation: the full model; every single-group skip; a greedy static frontier;
  the DeltaNet-only vs attention-only skip comparison; measured latency at batch 1 and 8; the
  cost of one architecture evaluation.
- **Gate 0:** full-model accuracy is plausible relative to published tool-calling results, on
  0.8B through 27B.

### Phase 1: UCT headroom on Qwen3.8-27B
- UCT on ~300 training requests (then up to ~1,000), 32–64 simulations each, with prefix
  caching and batched leaves.
- Report the oracle frontier against the static frontier, the diversity of best architectures
  (entropy, top-architecture share), and held-out probe accuracy.
- **Gate 1:** proceed only if routing beats the best static architecture at matched compute by a
  clear margin *on held-out requests*. Otherwise write up "little per-request headroom". The
  fallback is lossless self-speculative decoding using the built-in multi-token-prediction
  layer or a layer-skipped draft (needs a short literature check first).

### Phase 2: learn the router
R0 controls (always-full, best static at matched compute, random, heuristics such as tool count
and prompt length, and the model's own `reasoning_effort` setting at matched compute), then R1–R4.
Select on validation; test once on BFCL single-turn. Report accuracy per category, FLOPs,
measured latency and throughput, and 3 seeds.

### Phase 3: Pi harness
Serve the routed model behind an OpenAI-compatible endpoint, register it in Pi's `models.json`,
and compare full / best static / best router on 30–50 checkable tasks (success, steps,
wall-clock, compute). A 27B model makes this a real agent evaluation. Stretch: route per step.

### Phase 4: write-up
Frontier plots (FLOPs and measured time), the size trend, DeltaNet vs attention skippability,
the learner comparison, the Gate 1 analysis, the Pi results, limitations.

## Risks
- **No headroom** → Gate 1 makes it a reportable result; there is a speculative-decoding
  fallback.
- **Skipping breaks the JSON format of tool calls** → measured directly in Phase 0.
- **Search cost at 27B** (prefill dominates) → prefix caching, batching, and starting at 300
  requests.
- **Gated xLAM** → ToolRL mix fallback.
- **MLX training limits** → the frozen model with cached features avoids them; joint
  fine-tuning is deferred.
- **Shared lab** → jobs use `ailab_jobs` to yield and checkpoint; memory estimates are declared.

## Questions for the professor
1. Is the Gate 1 headroom result a sufficient intermediate milestone?
2. Is UCT as an offline teacher plus Expert Iteration what he intended, or UCT as the online
   decision rule?
3. Is a 27B model with Pi tasks that have automatic checks an acceptable agent evaluation?
