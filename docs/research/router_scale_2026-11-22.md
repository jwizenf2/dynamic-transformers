# Router scale-up on GSM8K train, and GSM8K for seeds 1 and 2 (pre-registered 2026-11-22)

Background (`router_nonlinear_2026-11-18.md`): with loop-1/2 hidden states of the 48-token
probe, an MLP router reaches AUC .664 for "4 loops help" at ~2,025 training problems (2,531 in
total), still rising ~.03 per doubling; extrapolated, ~8,000 problems would give ~.72. The
pre-registered rule said a large collection was not worthwhile (.664 < .70); the authors chose to
run it anyway to measure the curve rather than extrapolate it. This is that run, plus GSM8K test
for training seeds 1 and 2, so the second benchmark covers all three seeds like NESTful.

## Data (`run_stages.py --plan router-scale`, one job, sequential)

1. GSM8K test (800, depth-balanced order) for chain v3 seeds 1 and 2: greedy dialogues at 2, 4, 6
   loops, the 48-token probe, and the full-plan condition at 2 and 4 loops (as seed 0's
   `gsm8k-p4` stages, without depths 1 and 3).
2. GSM8K train (openai/gsm8k @ 740312a, `make_gsm8k.py --split train`: 5,510 of 7,473 problems
   kept by the same conversion and filters as the test split; 2–15 calls), chain v3 seed 0, in
   chunks of 1,000: greedy dialogues at 2 and 4 loops (`router_smoke.py`), then the 48-token probe
   with loop-1/2 hidden states on the same requests (`router_probe2.py --hidden`). No model is
   fine-tuned on GSM8K train; it only supplies router labels. GSM8K train and test are disjoint
   by construction of the dataset. The depth-balanced order interleaves dependency depths over the
whole file, so the late chunks hold mostly 2–3-deep problems; an early stop gives a sample deeper
than GSM8K overall (reported if it happens).

With the existing 2,531 problems (NESTful [0, 1200), GSM8K test 800, MathQA 531), 8,041 problems
have hidden states and 2-/4-loop outcomes.

## Analysis 1: the learning curve (answers "do we reach .72?")

Exactly the follow-up protocol of `router_nonlinear.py --hidden-followup` (inputs: mean loop-2
state and mean h_2 − h_1 over the probe tokens plus the benchmark one-hot; label "4 helps" =
wrong at 2, right at 4; PCA32/PCA64-logistic and the MLP; 5 seeds x 5 folds over problems), on all
8,041 problems, training sizes 250, 500, 1,000, 2,000, 4,000 and the full fold (~6,430).

Reading: the best model's mean AUC at full size >= .72 means the extrapolation held; below it, the
curve flattened. Reported with the seed sd, per-benchmark AUC of the out-of-fold scores, and the
rescue AUC (4 right among 2-loop failures). The earlier analyses put the AUC a router needs to
match fixed-4 at ~0.8x compute near .75.

## Analysis 2: held-out routing (does the bigger router beat the random mix?)

Router trained only on GSM8K train + MathQA (6,041 problems; no benchmark one-hot): the MLP,
fixed now (the best hidden-state model of the 2026-11-18 follow-up), not chosen by Analysis 1,
whose cross-validation includes the test problems. Policy (as `router_confirm2`): start at 2 loops;
escalate the requests scoring above a threshold; cost stay 2·T_2, escalate 4·T_4 + 2·48 (probe),
T = prompt + generated tokens of the greedy dialogue at that depth. The threshold is fixed on the
training set from out-of-fold scores (5 folds) so that the policy spends 0.85x of fixed-4 cost
there; it is applied unchanged to the test sets.

* Primary: GSM8K test (800, in distribution for the training labels). Gain = policy accuracy
  minus the best random mix of fixed depths at the policy's realised cost (upper concave hull of
  fixed 1–4 loops, as `router_confirm2`; if the cost exceeds fixed-4's, the difference from
  fixed-4); paired bootstrap 90% interval over
  requests (scores fixed, 2,000 resamples, seed 0). CONFIRMED if the lower bound > 0, else NOT
  CONFIRMED.
* Secondary: the same on NESTful [0, 1200) (cross-benchmark); the gain against the 2–4 chord;
  a quantile threshold (escalate the same share of test requests as the out-of-fold training
  policy did), since a refitted MLP's score scale can shift the realised cost; AUC on each test
  set; the accuracy-cost curve over all thresholds; the oracle 2-vs-4 router.

Code: a new `scripts/lab/router_scale.py` (reusing `router_nonlinear`'s loading, MLP and
training), so the published `router_nonlinear.py` numbers are untouched (its SOURCES, one-hot
width and RNG sequence over SIZES stay as they were). GSM8K train rows get the GSM8K benchmark
one-hot and a separate split field for Analysis 2's train/test selection.

Review before queueing (2026-11-22, applied): model pre-committed instead of selected on data
that includes the test sets; the baseline kept as the 1–4 hull of the earlier confirmation (the
chord would be softer); quantile threshold added as a secondary; analysis code separated.

## Analysis 3: GSM8K on three seeds

`replication_metrics.py` for seeds 1 and 2 on GSM8K (`--tasks results/gsm8k/test_tasks.jsonl
--nestful-tag gsm8k --plan-tag gsm8k_plan_full --start 0 --n 800`), predictions P1–P5 with the
rules of `replication_2026-11-04.md`; a prediction replicates on GSM8K if SUPPORTED for at least
two of the three seeds (seed 0 supported all five).

## Cost

From the earlier GSM8K job (~1.7 s per request-loop on the lab), ~6 h per seed for step 1 and
~10 s per request for 2 + 4 loops, so ~16–18 h for GSM8K train: ~29 h in total. Each train chunk is
usable on its own if the run is stopped early; the analyses then use the chunks that finished.
