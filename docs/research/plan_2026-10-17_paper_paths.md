# Plan: paper paths after the full SFT runs (2026-10-17)

Goal: run the experiments behind each candidate headline, then ask the professor which one
the paper should lead with. Measured numbers only; ceilings are labelled as ceilings.

## Where we are (measured)

Full SFT queue, job 03ce6df5 (commit f76d0b3): LoRA SFT of Ouro-2.6B on Hermes FC v1, the
2608.18171 recipe (Ouro loss) and our depth-aware loss, 2 epochs each, evaluated with
`ouro_sft_eval.py --adapters` at fixed depth 1–4 in the Hermes format. Grader: our BFCL-style
AST checker (not the official `bfcl-eval`). One training seed per loss.

| BFCL non-live, request-weighted (n) | d1 | d2 | d3 | d4 |
|---|---|---|---|---|
| Ouro loss (1,000) | 0.621 | 0.892 | 0.921 | 0.921 |
| Depth-aware loss (989 of 1,000 at writing) | 0.686 | 0.888 | 0.911 | 0.917 |

* Base Ouro produced no Hermes-format tool calls; both fine-tuned models reach ~92%.
* Depth 3 = depth 4 on BFCL; on xLAM val (Ouro loss) d3 0.83 > d4 0.815.
* Depth-aware loss: depth 1 better in every BFCL category (+3 to +12 points, largest on
  parallel_multiple), within ~1 point at depths 2–4. Single seed, no intervals yet.
* Oracle per-request depth (cheapest correct depth, a ceiling): Ouro loss 95.1% at 1.38 mean
  loops, depth-aware 94.0% at 1.30, vs ~92% at 4 loops fixed. xLAM: 85.5% at 1.40.
* NESTful (Ouro loss, partial 62/120): 0/2/8/13% at d1–4. Hermes data has no nested calls.
* Base model, earlier: router v3 (supervised, pass-1 + early probe) 81% at ~3.3 effective
  passes vs fixed-4 82% on xLAM val (≈×1.22 batched); Ouro's per-token gate ×1.01–1.06
  batched; LinUCB never reached 81%; NESTful oracle over depths 1–4 22.5% vs fixed-4 18.9%,
  over 1–6 33.6% vs 22.8% at 5.

## Candidate headlines

**A. Serving: per-request depth for looped LMs.** Per-token exit loses its saving under
batching; a per-request router on a depth-aware fine-tuned looped model keeps it.
Headline number: measured throughput at equal BFCL accuracy (×X). Status: every link has
evidence except X.

**B. Compute-aware RL over loops on tool-call chains.** A small policy picks the loop count
per call in multi-step chains, rewarded by execution success minus loop cost; target: above
fine-tuned fixed depth 4 on NESTful accuracy with fewer loops. Status: needs a model that can
do chains (nested SFT) and a go/no-go on per-call headroom. Check novelty against RL with
compute/length penalties for reasoning models before claiming it.

**C. Both:** A for single calls, B for chains, one model family and one serving story.

The depth-aware loss is shared by all three as a component and ablation, not a headline on
its own (+3–12 points at depth 1 is incremental alone).

## Experiments

Order is by information per GPU hour. Gates say when a path stops.

### 0. Free analyses (CPU, from existing outputs)
* Bootstrap 95% intervals per category and depth; paired bootstrap for depth-aware minus
  Ouro loss at each depth.
* Baseline table from existing data: fixed depths, oracle, matched-histogram random.
* First estimate of X: how much of the oracle saving a pass-1-feature router recovers on the
  fine-tuned models (needs pass-1 features for the fine-tuned models; if not cheap, defer).
* Confirm our checker against the official BFCL grader on a sample (or document the gap).

### 1. Path A: router on the fine-tuned models
1. Router labels: each fine-tuned model at depths 1–4 on a training pool that is not BFCL
   (xLAM train ids and/or held-out Hermes conversations), plus probe features.
2. Train the router there; test on BFCL (cross-benchmark transfer) and xLAM val.
3. Live probe-then-commit generation and a batched serving bench grouped by chosen depth;
   report accuracy vs measured tokens/s against fixed depth, Ouro's gate, confidence
   threshold, matched random, oracle.
* Gate A: X ≥ ~1.5× at equal accuracy makes A a credible main-track headline; ~1.2× keeps A
  as a workshop-level result or a section of C.

### 2. Path B: chains
0. First (cheap, no retraining): multi-turn NESTful eval. Finding 2026-10-18: the fine-tuned
   models never emit `"label"` (0/120 at d4, Ouro loss) and write all calls in one turn,
   hard-coding intermediate values they compute themselves; only 31/120 outputs contain a
   `$var_N.result$` reference. Hermes SFT teaches the multi-turn protocol (call →
   `<tool_response>` → next call), so the single-turn label convention is a format mismatch.
   Evaluate chains by executing each turn's calls and feeding results back; this is also the
   loop the per-call depth policy needs. Re-decide whether nested SFT data is needed after it.
1. Nested-call training data: survey candidates (NesTools, Seal-Tools, ToolACE, others; check
   licence, real dependent calls, overlap with NESTful) and/or synthetic chains with
   templates and functions disjoint from NESTful. NESTful stays test-only.
2. Nested SFT variant (depth-aware loss, Hermes + nested data), ~11 h + eval. Kept separate so
   the Hermes-only runs remain the clean 2608.18171 reproduction and ablation.
3. Per-call oracle experiment (go/no-go): ~100 chains; at each call boundary branch depths
   1–4 for that call, continue greedily; compare best per-call path vs best per-request
   depth vs fixed 4. Depth switches at `</tool_call>`; skipped-loop KV filled from the
   deepest computed loop (as in the serving bench).
* Gate B: per-call oracle clearly above per-request oracle (accuracy or loops). If not, the
  supervised per-request router suffices and B is dropped.
4. If go: RL depth policy (frozen LM, cached rollouts, reward = execution success − λ·loops),
   compared with fixed depth, per-request router, Ouro's gate and the per-call oracle.

### 3. Training seeds (for the loss comparison in every path)
* What: seeds change LoRA initialisation and data order. Evaluation needs no seeds (greedy
  decoding is deterministic); its uncertainty comes from the bootstrap in step 0.
* Plan: 2 more seeds per loss (seeds 1, 2) → 4 runs, same recipe, ~11 h each, sequential
  (two at once share the GPU and are not faster). One queue job via `sft_queue.py`.
* Eval for seed runs: BFCL only, depths 1–4 (xLAM/NESTful made the first evals slow).
* Report mean ± spread over 3 seeds and the paired bootstrap per seed. Claim "depth-aware
  improves shallow depths" only if it holds across seeds.
* When: whenever the lab is free between path A and B jobs; before writing the paper.
  If path B is chosen, the nested variant needs seeds too (decide after Gate B).

### Also needed for any main-track version
* Ouro-1.4B as a second scale (SFT + eval, ~1 day).
* Official BFCL grader numbers or a documented agreement check.

## Lab budget (approximate, sequential)

| Item | GPU time |
|---|---|
| Router labels + live serving bench (A) | 1–2 days |
| Nested SFT + per-call oracle (B, to the gate) | ~1 day |
| 4 seed runs + BFCL evals | ~2 days |
| RL policy (B, if go) | 1–2 days |
| Ouro-1.4B | ~1 day |

## Questions for the professor (after the experiments)
1. Which headline: A (serving, single calls), B (RL over loops on chains), or C (both)?
2. Is the depth-aware loss better presented as a contribution or as an ablation component?
3. Is our BFCL-style checker acceptable with an agreement check, or are official grader
   numbers required?
4. Is LinUCB a sufficient bandit/RL baseline for the per-request choice, given the
   full-information labels (supervised beat it)?
