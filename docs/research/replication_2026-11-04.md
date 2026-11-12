# Replication runs: unseen requests, training seeds, Ouro-1.4B (pre-registered 2026-11-04)

Purpose: the analysis-paper findings rest on one fine-tuned model (chain v3, seed 0) and on NESTful
requests [0, 720) that were all inspected. These runs (1) confirm them on unseen requests with the
same model, (2) repeat them for two more training seeds, (3) repeat them on a second model size
whose accuracy rises with depth. All predictions and their analysis
(`scripts/lab/replication_metrics.py`, smoke-tested on the existing chain v3 data, where it
reproduces every earlier number) are fixed before any of the runs.

## Predictions (per model and slice; SUPPORTED / NOT SUPPORTED / N/A)

| | Prediction | Rule |
|---|---|---|
| P1 | Depth fixes formulation | among requests wrong at 2 and right at 4 loops, share of `wrong_plan` + `wrong_operation` 2-loop errors > .5 (dev: .96) |
| P2 | Gains compound with chain length | 2→4 gain on 6–8 gold calls minus on 2–3, bootstrap 90% lower bound > 0 (dev: +.12) |
| P3 | A given plan removes part of the gain | gain_none − gain_full_plan, 90% lower bound > 0 (dev: +.046) |
| P4 | The need for depth is stable (unseen slice only, depth 6 run) | share of 4-loop fixes also right at 6 loops >= .6 (dev: .82) |
| P5 | Rescuability is not predictable early | rescue AUC (early-feature router, 4 right among 2-loop failures) <= .72 (dev: .66) |
| P6 | Short horizons help failure detection only (BFCL runs) | BFCL "4 helps" AUC > the same model's NESTful AUC, and BFCL rescue AUC <= .72 |

A finding counts as replicated across training runs if it is SUPPORTED for at least two of the
three seeds (0, 1, 2); the paper reports each seed's value and the mean and spread.

## Runs (one resumable lab job, `run_stages.py --plan replication`; stages in this order)

1. **Unseen confirmation**, chain v3 seed 0, NESTful [720, 1200) (never run; 480 requests, 228
   with 2–3 calls and 73 with 6–8): greedy dialogues at 1, 2, 3, 4, 6 loops, the 48-token probe,
   the full-plan condition at 2 and 4 loops. Analysis: `--nestful-tag nestful720 --plan-tag
   plan_full720 --start 720 --n 480 --rescue-train nestful nestful240` (rescue router frozen on
   the model's own [0, 720) data). ~16 h.
2. **Chain v3, seed 1** (`sft_queue.py --plan chains-v3-s1`: chain v3's data and recipe, warm
   start from the same stage-1 adapters, kept at `~/AI/keep/ouro-depth-full`, sha256 cb5ebeb3…;
   the seed changes the data order and which 3,000 Hermes conversations are replayed
   (the LoRA weights come from the warm start); BFCL at 1/2/4 loops), then on NESTful [0, 720): dialogues at
   1–4 loops, probe, full plan at 2 and 4 loops; BFCL probe. Analysis with cross-validated rescue
   router. ~32 h.
3. **Ouro-1.4B** (`--plan ouro14`: stage 1 = depth-full's Hermes SFT with the depth loss, stage
   2 = chain v3's mix), then the same evaluation as run 2 with `--model ByteDance/Ouro-1.4B`
   (smoke test: 2 training steps run; the base model's validation loss by loop
   .98/.42/.31/.30). ~25 h.
4. **Chain v3, seed 2**, as run 2. ~32 h.

Scope notes, stated in the paper: the seeds re-run only the stage-2 fine-tune (stage 1 shared);
NESTful is the only multi-step benchmark (BFCL is the single-turn control); Dory is reported as a
model whose accuracy barely changes with depth on this task (loop count acts as a perturbation),
not as a replication.

## Results (job d5e54bae, finished 2026-11-08; analysed 2026-11-12)

Accuracy at 1/2/3/4 loops on NESTful [0, 720): seed 0 .168/.374/.449/.500; seed 1
.181/.389/.465/.522; seed 2 .164/.374/.475/.494; Ouro-1.4B .099/.258/.357/.381. Unseen [720, 1200),
seed 0: .213/.458/.548/.594 (6 loops .583).

| Prediction | Seed 0 [0,720) | Seed 1 | Seed 2 | Seeds replicated | Ouro-1.4B | Seed 0 unseen |
|---|---|---|---|---|---|---|
| P1 formulation (share) | ✓ .96 | ✓ .83 | ✓ .85 | 3/3 | ✓ .84 | ✓ .87 |
| P2 compounding (6–8 minus 2–3) | ✓ +.12 | ✓ +.09 | ✓ +.10 | 3/3 | ✓ +.11 | ✗ +.10 (CI −.007 to +.21) |
| P3 plan ablation (gain difference) | ✓ +.046 | ✗ +.019 | ✓ +.039 | 2/3 | ✗ −.029 | ✓ +.102 |
| P4 stability (right at 6) | — | — | — | — | — | ✓ .80 |
| P5 rescue AUC <= .72 | ✓ .66 | ✓ .61 | ✗ .731 | 2/3 | ✗ .723 | ✓ .57 (frozen) |
| P6 BFCL (AUC BFCL vs NESTful; BFCL rescue) | ✓ .79 vs .60; .67 | ✓ .75 vs .56; .63 | ✓ .67 vs .64; .66 | 3/3 | ✓ .72 vs .60; .64 | — |

By the pre-registered rule (SUPPORTED for at least two of three seeds) P1, P2, P3, P5 and P6 replicate.
Ouro-1.4B supports P1, P2, P6 and misses P3 and, narrowly, P5. The unseen confirmation supports
P1, P3, P4, P5 and misses P2 (73 long requests).

Exploratory (not pre-registered): P3 split by chain length. 2→4 gain on 2–5-call chains without /
with the full plan: seed 0 +.095/+.005, seed 1 +.108/+.010, seed 2 +.095/+.008, Ouro-1.4B
+.085/+.049, seed 0 unseen +.117/+.008. On 9+-call chains the plan does not remove the gain (2-loop
accuracy with the plan .06–.57: the shallow model cannot execute long plans). The aggregate P3
depends on the mix of chain lengths; the stable statement is that a given plan removes the
benefit of depth where execution is easy.

Reading for the paper: across three training seeds, two model sizes and unseen requests, extra
loops fix formulation errors (83–96% of the requests they fix) and help more on longer chains;
early signals rank rescuable failures only weakly (AUC .57–.73) and routing does not beat the
random mix (`router_confirm2.py`); single-turn tasks make failure easier to detect but not
rescue. The plan ablation holds where execution is easy and is composition-dependent overall.
