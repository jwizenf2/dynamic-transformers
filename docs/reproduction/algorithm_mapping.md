# Algorithm-to-code mapping

Paths are under src/dht; pages are printed PDF pages. Adopting an executable
convention does not resolve the source inconsistency. See ../code_walkthrough.md.

| Component | Paper source | Implementation | Fidelity / nuance |
| --- | --- | --- | --- |
| Masked head sum/original positions | §4.1 p.442 | models/attention.py | Sliced rows/columns, dense reference; D003 |
| Full-width encoder/classifier | §4.1 p.442; Fig.1 p.445 | models/transformer.py; pretrained.py | Random scaffold or strict pretrained encoder import; shared CLS linear head; D002/D010 |
| Total depth/full prefix | §4.2 pp.442–443; Fig.1 p.445 | models/architecture.py | Total includes prefix; D001 |
| Routing prefix / fresh task prefix | §4.2.1–4.2.2 p.443; Alg.2 line7 p.444; Fig.1 p.445 | DHTController.route/execute; DHTTrainer.step; TransformerClassifier.prefix | Same trainable weights, no-grad routing then fresh task graph; CPU/MPS calls/updates tested in test_prefix_passes.py; D008/D014 |
| Shared subset across c layers | §4.2.1 p.443 | controller.execute; ArchitectureSpec.validate | Explicitly specified |
| Context/arm concatenation/group marker | §3.1 p.441; §4.2.1 p.443 | bandits/features.py: aggregate | Actual widths from bins; D006 |
| Per-arm ridge/UCB bonus | Alg.1 p.441 | bandits/linear_ucb.py | Independent implementation, repaired dimensions; D005 |
| Gaussian/sigmoid/top-K | §3.2 pp.441–442 | bandits/logistic_ts.py | Positive prior, repaired logistic objective; D005 |
| Data-derived arm means | §4.2 pp.442–443 | FeatureBank.calibrate | Train subset, all count paths, measured modification; D006 |
| Pooling/normalization | §§7.1–7.2 p.449 | pooled_features, FrozenStandardizer | Explicit bins, frozen train moments; D006 |
| Previous-batch products/ratios | §4.3 pp.443–444 | policies/rewards.py; controller.update | Missing products interpreted, bootstrap/floor/cap; D007 |
| Binary subset label | §4.3 p.444 | RewardHistory.binary_feedback | Literal default, named correction |
| Tree search | §4.2.1 p.443; Alg.2 p.444 | policies/controller.py | Route UCB then per-group UCB_H/TS; D008 |
| Integrated search | §4.2.2 p.443 | same controller, mode=integrated | Pair once, fixed count, per-group TS |
| Depth queues/tails | §5 p.444; Alg.3 p.445 | training/queues.py; next_batch | Original sample ID/context, explicit flush |
| Model/bandit updates | Alg.2 p.444; Fig.1 p.445 | DHTTrainer.step | Mean task gradient then executed-action feedback |
| Complete checkpoint/resume | Not specified fully | checkpointing.py; trainer.state_dict | Model, optimizer, policies, history, calibration, queues, RNGs |
| Main calibration subset | §4.2 pp.442–443 all-data description; our approximation | training/calibration.py | Seeded512 training-only, class-balanced IDs; D013; frozen normalizers remain a deviation |
| Epoch/LR/validation protocol | §7.3 p.449 incomplete | training/epochs.py; TrainingConfig | Equal corpus exposure, tails, explicit sample schedule, validation selection, atomic schema3 recovery; D013 |
| F1/architecture distributions | §6.2 p.447 | metrics.py; selection_summary | Explicit averaging and group frequencies |
| FLOPs/time | §6.2 p.447 | flops.py; trainer; CLI | Partial task/routing/calibration estimates and actual stage timing |
| Physical execution evidence | §4.1 p.442 semantics; our verification | evaluation/execution.py; scripts/verify_execution.py | Real mm/addmm/bmm shapes, layer traces, sliced/dense outputs; not complete training FLOPs |
| AG News preprocessing/initialization | §6.1 pp.445–446 incomplete | ag_news.py; models/pretrained.py | Pinned CSV/WordPiece/BERT-base, held-out validation, explicit pilot; D010 |
| Published S/F spaces | §6.1 p.446 | configurable pairs; references/paper_settings.json | Mechanism supported; published-scale experiments unrun |
| P-DHT | §6.1 p.446 | Reference list only | Freeze semantics deferred |

Mathematical checks: tests/unit/test_bandits.py. Feature/reward/queue checks:
tests/unit/test_features_rewards.py. Exact resume, frozen inference, heterogeneous
heads and epoch tails: tests/integration/test_dht.py.
