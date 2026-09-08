# Independent UCB / logistic TS implementation specification

The implementation described here is now executable. See `docs/code_walkthrough.md`
for a guided tour and D005–D008 for adopted interpretations. This document retains
its original plan filename for continuity.
Source: §3.1 Algorithm 1 p.441 and §3.2 pp.441–442 of the supplied PDF.

## Linear UCB

Implemented in `src/dht/bandits/linear_ucb.py`, tested in
`tests/unit/test_bandits.py`. Uses a fixed ordered set of arm IDs and actual
aggregate width p (A02). Store CPU float64 per-arm A=λI_p and b=zeros(p).
API: `scores(features_by_arm)`, `select(features_by_arm)`,
`update(selected_arm, decision_time_features, reward)`, `state_dict/load_state_dict`.
Require finite features/rewards, λ>0, exploration>=0 and exact widths.

Compute θ via a linear solve, and mean wᵀθ plus exploration×sqrt(wᵀsolve(A,w)).
Clamp only tiny negative roundoff in the quadratic form; report larger invalid
values. Use deterministic arm-order tie breaking as a documented engineering
choice. Update only the selected arm, A+=wwᵀ and b+=r w. Accept externally
constructed context-arm features so unresolved pooling widths do not block UCB.

Acceptance: direct ridge comparison after several known observations; hand-computed
mean/bonus; unobserved-arm exploration; untouched unselected statistics; strict
validation; reproducible ties; checkpoint equality; seeded synthetic contextual
environment with distinguishable linear rewards. No neural network is needed.

## Second task: logistic Gaussian Thompson semi-bandit

Adopted under A03–A05/D005: diagonal online Laplace approximation, matching
the component-square curvature expression most closely. Retain the paper's all-ones
initial mean as an interpretation of its rectangular I, initialize precision to
λ_ts>0 rather than the noninvertible printed zero, and clearly label this repair.

For each head arm, sample θ=g+ε/sqrt(s), ε~N(0,I) from a dedicated CPU generator.
Compute σ(wᵀθ), choose K distinct highest probabilities with stable tie breaking.
Record features and selected subset before delayed feedback. On observed subset
label y∈{0,1}, update selected heads only; broadcast that one observed label without
claiming independently observed individual head rewards. Never update unused heads.

Given prior mean g_old and precision s_old, minimize in θ:

```text
0.5 * sum_j s_old[j] * (θ[j] - g_old[j])²
  + sum_new_observations [softplus(wᵀθ) - y * wᵀθ]
```

Use the minimizer as g_new. Update diagonal precision by adding
sum_new σ(wᵀg_new)(1−σ(wᵀg_new)) w² to s_old. λ_ts enters via the initial prior;
do not silently multiply accumulated precision by λ at each update. Store convergence
status and optimizer tolerances. Using only new observations avoids counting history
again when it is already represented in the Gaussian prior. These are proposed
repairs to the printed optimization, not verified author conventions. Full Hessian
covariance is a distinct alternative and should have a separate named configuration.

Acceptance: finite/stable objective for extreme logits, both labels affect the
objective, optimization decreases it, positive precision, empirical sampling
moments, valid top-K, selected-only updates, reproducible RNG and state round trips,
and learning in a small logistic environment. Do not use Beta-Bernoulli TS.

## Integrated DHT

D006–D008 now specify and implement A06–A09: head products, binary feedback,
bootstrap and pooling. The default preserves the literal p.444 inequality; the
prose-consistent inversion is a separately labeled correction setting. TS remains
independent of that reward policy. Train-only arm calibration, frozen normalization,
depth queues and both search modes are implemented and covered by integration tests.
