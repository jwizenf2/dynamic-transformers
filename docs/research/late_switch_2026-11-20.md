# Late switching: does continuing at 4 loops after the divergence still rescue? (pre-registered 2026-11-20)

Background (`router_window_hidden.py`): an early-signal router ranks "4 loops help" better the
longer it waits (MathQA AUC .60 at 16 tokens → .69 at 128 → .72 at the end of the 2-loop dialogue),
but the requests it would rescue leave the 4-loop path at the same rate (81% still on it at 16
tokens, 14% at 128). A switch before the divergence reproduces the 4-loop dialogue exactly; a
switch after it was scored as not rescuing, because its outcome was not in the data. If late
switches often recover, deciding late becomes viable.

Collection (`late_switch.py`, `run_stages.py --plan late-switch`, chain v3 seed 0): for every
request, the 2-loop greedy dialogue with per-token loop-1/2 signals and ids; at decision points
T = 64 and 128 trace steps, the 2-loop context at T is continued at 4 loops to the end of the
dialogue (prefill of the whole context at 4 loops = a switch that reuses the loop-1/2 caches).
Sets: MathQA clean val + test pool (531, development), NESTful [0, 720) (test), GSM8K test first 800
(test). Validation: where the 2- and 4-loop runs have not diverged by T, the switch must reproduce
the stored 4-loop outcome; the agreement rate is reported (lab GPU, same as the stored runs).

Analysis (`late_switch_analysis.py`). Costs (loop-tokens): 2-loop run 2·T_2; fixed 4 loops 4·T_4;
a switch at T charged 4 x all tokens of the switched dialogue (context at T, tokens generated after,
tool turns after); a T beyond the end of the 2-loop dialogue is a restart (2·T_2 + 4·T_4, the
4-loop outcome). Router at T: features of the 2-loop trace up to T plus the 12 early probe features
(`router_window_hidden.window_features`), logistic or MLP, 5x5 cross-validation within the set on
the label "the switch at T is right and the 2-loop run is wrong". Policy: escalate the top-q scores;
curve over q; random-mix baseline: the chord between fixed 2 and fixed 4 loops.

Selection (development, MathQA): the decision point T in {64, 128} and model in {logistic, MLP}
with the highest accuracy gain over the mix at <= 0.90x fixed-4 cost.

Primary (test, NESTful [0, 720), chosen T and model): gain over the random mix at <= 0.90x
fixed-4 cost, paired bootstrap 90% interval over requests (scores fixed, 2,000 resamples, seed
0): CONFIRMED if the lower bound > 0, else NOT CONFIRMED. Secondary: the same on GSM8K; accuracy at
<= 1.0x vs fixed 4 loops; both T and both models per set; the recovery table (share of requests
wrong at 2 loops that the switch at T gets right, split by whether the 4-loop run from scratch is
right); the oracle switch (escalate exactly the requests the switch rescues) as an upper bound.

Reading rules: CONFIRMED means late switching makes per-request depth routing beat the fixed-depth
mix on held-out problems (the original goal, reached by moving the decision later); NOT CONFIRMED
with a high recovery rate means the information is there but the router still ranks too weakly;
a low recovery rate means a diverged 2-loop prefix cannot be rescued, closing the routing question.

Review before queueing (2026-11-20, applied): the validation (switch before divergence equals the
stored 4-loop outcome) is computed in the analysis, with the divergence step found by comparing
the 2-loop trace texts with the stored 4-loop texts (which are cut at 300 characters per turn, so a
few divergences are found late); it need not be 100%, since the switch prefills the whole context
in one pass where the stored run prefilled chunk by chunk (different kernel paths). The collection
asserts that every task matches the stored labelled runs (query, gold chain length) before any
GPU time; decision points count the trace steps of each turn (turns cut at 256 tokens have no
stop step).

## Results (job 8e59c776, finished 2026-11-21; `results/late_switch/analysis.json`)

Validation: a switch before the 2-/4-loop divergence reproduces the stored 4-loop outcome on
.987–1.000 of such requests (MathQA .993/.994, NESTful .987/.990, GSM8K .991/1.000 at T = 64/128).

Recovery (share of requests wrong at 2 loops and right at 4 that the switch at T rescues; at 64 /
128): MathQA .48 / .46, NESTful .52 / .42, GSM8K .54 / .40. Requests wrong at both depths are almost
never rescued (.01–.12); switching loses .03–.08 of the requests right at 2 loops. Switching every
request at 64 gives MathQA .365, NESTful .446, GSM8K .710 against fixed 4 loops .437 / .500 / .764
at similar cost.

Selection on MathQA: T = 64, MLP (gain −.038, the best of the four). Primary, NESTful [0, 720):
gain over the random mix at <= 0.90x −.031 (90% CI −.046 to −.007): NOT CONFIRMED. GSM8K: −.030 (CI
−.048 to −.010): NOT CONFIRMED. Router AUC for "the switch at T rescues": .58–.72. Oracle late switch
(exactly the rescued requests): +.068 to +.106 over the mix at 0.54–0.57x.

Reading (by the pre-registered rules: not confirmed, with a moderate recovery rate): about half of
the requests that 4 loops fix can still be fixed after the 2-loop run has diverged, so deciding
later keeps part of the headroom; but the router ranks those requests too weakly, and each switch
costs about a full 4-loop run, so late switching does not beat the fixed-depth mix. With the
window analysis this completes the routing picture: early decisions are timely but uninformed,
late decisions are better informed but can recover only about half of the benefit, and neither
reaches a useful policy.
