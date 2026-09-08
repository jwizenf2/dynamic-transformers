# Notation glossary

All citations refer to printed pages of the supplied PDF. Code uses zero-based
layer/head indices and counts for depth. p always means aggregate feature width
in our bandit APIs, not the paper's sampled TS parameter vector.

| Paper symbol | Meaning / proposed code name | Source |
| --- | --- | --- |
| NL, maximum L | available encoder layers / model.total_depth | §4.2 p.442 |
| L, L* | selected total depth / ArchitectureSpec.total_depth (D001) | §4.2 pp.442–443; Fig.1 p.445 |
| b | unpruned prefix / prefix_depth | §4.2 p.443 |
| c | shared-decision block / group_size | §4.2 p.443 |
| o | dynamic group count = (total_depth−prefix_depth)/group_size | Fig.1 p.445 |
| l | layer/depth marker; convert to zero-based layer index | §4.2 p.443 |
| NH or N_h | full available head count / num_heads | §4.1 p.442 |
| H_l | selected head_count | §4.1 p.442 |
| ξ_h | binary head mask | §4.1 p.442 |
| Π_l, Φ_ts* | selected superarm / head_indices | §§3.2,4.1 pp.442–443 |
| x, z, w=[x;z] | sample, arm, aggregate features | §3.1 p.441 |
| e_l | one-hot depth/group marker for UCB_H | §4.2.1 p.443 |
| d, 2d | raw/aggregate widths (inconsistent elsewhere) | §3.1 p.441 |
| D, C | observation design matrix, scalar reward vector (dimension repair) | §3.1 p.441 |
| A_γ, b_γ | per-arm ridge matrix and reward-weighted sum | Alg.1 p.441 |
| θ_γ | linear reward parameter estimate | Alg.1 p.441 |
| λ | ridge or logistic prior strength; separate configs | §§3.1–3.2 pp.441–442 |
| α in Alg.1 | ucb_exploration_scale | Alg.1 p.441 |
| g_γ, s_γ | TS parameter mean and precision (diagonal/matrix unresolved) | §3.2 pp.441–442 |
| p_i,γ | sampled parameter vector / sampled_theta | §3.2 p.442 |
| E_i,γ | sigmoid score / success_probability | §3.2 p.442 |
| K | number of selected head arms / head_count | §3.2 p.442 |
| δ_L, δ_1 | loss_depth_product (aliases appear in print) | §4.3 p.443 |
| δ_H, δ_2, δ_cp | underspecified head/pair cost products | §4.3 pp.443–444 |
| α in reward | depth_reward_scale (not UCB exploration) | §4.3 p.443 |
| β_1, β_cp | head_reward_scale, pair_reward_scale | §4.3 p.444 |
| t, i, bs | batch round, sample within batch, batch size | §4.3 p.443 |
| Q_L, q | depth FIFO, release capacity / queue_capacity | §5 pp.444–445 |
| S1, S2,0 | routing-prefix context, training-prefix context | Fig.1 p.445 |
