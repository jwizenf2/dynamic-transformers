# Nonlinear routers and the value of more labels (pre-registered 2026-11-18)

Suggestion (advisor): a truly nonlinear router with many interactions, e.g. a neural network,
scaled up. Question: would more labels and a bigger model make early-signal routing work, or is
the information not there? Decided by a learning curve on the data we already have, before any
new label collection.

Data (chain v3 seed 0; greedy outcomes at 2 and 4 loops; the 48-token probe of the first answer):
NESTful [0, 1200) (1,200), GSM8K test (800), MathQA clean val + test pool (531): 2,531 problems.
Inputs are deployable at 2 loops: per token and loop 1–2, exit gate, entropy, top-2 margin, the
loop-2 state change, loop-1/2 argmax disagreement (as in the earlier routers), the token type
(function name, number, structure, other), and the benchmark. Label: "4 loops help" (wrong at 2,
right at 4).

Models (`scripts/lab/router_nonlinear.py`):
1. linear: L2 logistic on the 12 summary features of the earlier routers;
2. interactions: L2 logistic on the 12 features and all their pairwise products;
3. MLP: two hidden layers on ~70 pooled per-token statistics (mean/max/min of every signal,
   overall and per token type) plus the 12 features;
4. sequence network: per-token embedding, a 1-D convolution over the 48 tokens, masked mean
   and max pooling, an MLP head with the summary features;
5. on the 531 MathQA problems only (the only ones with saved hidden states): MLP on the mean
   loop-2 hidden state and mean h_2 − h_1 of the probe tokens, vs the PCA-logistic of
   `phase_a_router.py`.

Protocol: 5 seeds x 5 folds over problems; training sets subsampled to 250, 500, 1,000 and the
full fold (~2,025); each fitted model scored on the same held-out fold; neural models stop early
on 10% of their training set. Metric: AUC for "4 helps"; also the rescue AUC among 2-loop failures
for the best model at full size.

Decision rule: collecting thousands of new labels (GSM8K train, ~2–3 lab days) is worthwhile only
if the best nonlinear model at full size beats the linear model by >= .03 AUC AND its AUC still
rises from 1,000 to full size by >= .01 (both averaged over seeds, with the bootstrap spread
reported). Otherwise the conclusion is that early signals do not contain the needed information,
whatever the model.

## Result (2026-11-18, `results/router_nonlinear/report.json`)

AUC for "4 helps" (17.4% of 2,531 problems), mean over 5 seeds (sd), by training size
250 / 500 / 1,000 / full (~2,025):

| Model | 250 | 500 | 1,000 | full | rescue AUC (full) |
|---|---|---|---|---|---|
| linear, 12 features | .554 | .579 | .574 | .587 (.006) | .556 |
| all pairwise interactions | .551 | .574 | .582 | .588 (.008) | .549 |
| MLP on pooled per-token statistics | .578 | .588 | .597 | .611 (.010) | .564 |
| sequence network over 48 tokens | .533 | .543 | .568 | .590 (.022) | .533 |

MathQA hidden states (531 problems; training 100 / 200 / full ~424): PCA-logistic .572 / .613 /
.653, MLP .611 / .560 / .625.

Decision: best nonlinear (MLP) +.023 over linear at full size (rule: >= .03) and +.014 from 1,000
to full (rule: >= .01): the gain criterion fails, so collecting thousands of new labels for
summary-signal routers is not worthwhile. The MLP gains about .01 AUC per doubling of data;
extrapolated to ~8,000 labels it would reach ~.63, far below the ~.75 that matching fixed-4 at
0.8x compute needs. Interactions add nothing; the sequence model is no better than the linear one.
The one curve still rising steeply is the hidden-state PCA model on MathQA (+.04 per doubling,
only 424 training problems), which the existing labelled NESTful and GSM8K problems could test at
~2,000 problems by re-running only the probe with hidden states (no new dialogues).

## Follow-up: hidden states at ~2,000 problems (pre-registered 2026-11-18, before the probe run)

Re-run the 48-token probe with loop-1/2 hidden states on the already-labelled NESTful [0, 1200)
and GSM8K (800) problems (`run_stages.py --plan hidden-probes`; no new dialogues), giving 2,531
problems with hidden states. Inputs: mean loop-2 state and mean h_2 − h_1 over the probe tokens
(as on MathQA) plus the benchmark. Models: PCA (32 or 64 components, fitted on the training fold)
+ L2 logistic, and an MLP; same seeds, folds and training sizes as above
(`router_nonlinear.py --hidden-followup`). Decision: a large hidden-state label collection (the
advisor's scale-up) is worthwhile only if the best hidden-state model reaches AUC >= .70 at full
size and still rises by >= .01 from 1,000 to full; otherwise early hidden states, like the summary
signals, do not carry enough information to route.

### Result (job d8a9267a; `results/router_nonlinear/hidden_followup.json`, 2026-11-20)

2,531 problems with hidden states. AUC for "4 helps" at 250 / 500 / 1,000 / full (~2,025):
PCA32-logistic .588 / .609 / .637 / .642; PCA64-logistic .585 / .605 / .636 / .644; MLP .590 /
.605 / .635 / .664 (sd .01). Decision: best (MLP) .664 < .70: NOT worthwhile by the rule, although
it still rises (+.029 from 1,000 to full). The hidden-state MLP is the best early router found
(summary-feature linear .587, summary MLP .611). Its curve gains ~.03 per doubling; extrapolated,
~8,000 labels would give ~.72 and ~16,000 ~.75, the level the earlier policy simulation needed to
match fixed-4 at ~0.8x compute. Collecting that many labels (2- and 4-loop dialogues at ~30 s per
problem, ~60 h per 7,000; GSM8K train is the only large unused source) is a judgement call left to
the authors; the pre-registered rule says no.

## Decision window, attention over hidden states, cleaner labels (2026-11-20, exploratory)

`router_window_hidden.py`, `results/router_nonlinear/window_hidden.json`.

Window (MathQA 531 2-loop traces; MLP / logistic AUC for "4 helps"; timely = share of "4 helps"
requests not yet diverged from the 4-loop dialogue at the decision point; switch policy accuracy
lower bound at <= 0.90x fixed-4 cost vs the random mix, fixed-4 .437):

| Decide after | AUC MLP | AUC logistic | timely | gain vs mix (MLP) |
|---|---|---|---|---|
| 16 tokens | .602 | .570 | .81 | +.000 |
| 32 | .601 | .547 | .64 | −.028 |
| 48 | .634 | .578 | .57 | −.041 |
| 64 | .665 | .649 | .36 | −.063 |
| 96 | .680 | .656 | .22 | −.065 |
| 128 | .693 | .651 | .14 | −.072 |
| 192 | .696 | .657 | .04 | −.055 |
| 256 | .714 | .691 | .01 | −.037 |
| end of turn 1 | .640 | .619 | .36 | −.042 |
| end of turn 2 | .679 | .646 | .12 | −.073 |
| end (restart) | .718 | .695 | 0 | −.012 |

Ranking improves with every token the router waits, but the requests it would rescue diverge
from the 4-loop path at the same rate: by 64 tokens two thirds of them have, and by 192 nearly all.
The information that makes "4 helps" predictable arrives with the divergence itself, so no window
both ranks well and switches in time. Caveat: a switch after divergence is scored as not
rescuing (lower bound; midpoint = the 2-loop outcome); whether the 4-loop continuation of a
diverged 2-loop prefix recovers is not in the data.

Hidden states (2,531 problems, full training size, AUC greedy / stable label): mean-pooled MLP
trained on greedy labels .664 / .662; trained on stable labels (wrong at 2, right at 4 and 6;
14.6% vs 17.4%) .649 / .651; attention pooling over the 48 tokens .618 / .615. Neither the
attention model nor cleaner labels improve on the mean-pooled MLP.
