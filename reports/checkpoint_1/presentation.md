# Checkpoint 1 — Paper presentation outline

## 1. Problem and selected paper
Dynamic Transformers with Reinforcement Learning: Improving Sample-Based
Architecture Selection. Meng et al., PES 6(2), 439–452, 2024,
DOI 10.24874/PES06.02.001. Question: can each sample choose how much Transformer
computation to use during training and inference?

## 2. Core method
Prefix context → UCB depth → depth queue → rerun prefix → repeated UCB head count
and Gaussian logistic TS head subset → classification → model/bandit updates.
Explain total-depth convention and contextual bandits (no neural controller).

## 3. Mechanism
UCB learns contextual reward plus uncertainty; TSB samples logistic parameters and
keeps top-K heads. Rewards compare a loss-cost product with the previous batch.
Tree mode exposes three choices; integrated mode selects depth/count jointly.

## 4. Authors' evidence
Table 1/2, p.447: AG News full BERT F1 .944, T-DHT .945 with .42× training and
.81× inference FLOPs; F-DHT .943 with .59×/.35×. These are reported estimates,
not our results or measured latency. Complete table: references/paper_results.csv.

## 5. Independent implementation checkpoint
Current: independent linear UCB, logistic Gaussian TS, calibrated contexts, reward
ratios, tree/integrated routing, depth queues, and selected-head execution all work
on synthetic CPU data. Tests cover mathematics, frozen inference and exact resume.
No author code used. Next: real-data preprocessing and a fixed/T-DHT AG News
comparison. Synthetic sanity demonstrates mechanics, not NLP performance.

## 6. Limits and semester plan
Show feature-size, TS objective/feedback and missing training-protocol ambiguities.
Present adopted interpretations and proposed repairs separately. First reproduce,
then select a research extension and evaluate it against baseline/ablations.
