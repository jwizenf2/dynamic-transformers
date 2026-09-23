# AG News F-DHT (integrated search) and batched execution

Date: 2026-09-22. These are our measured results, not the authors' tables. F-DHT used
`configs/experiment/ag_news_fdht_full.toml` (D016): identical to the completed T-DHT run
except `mode = "integrated"` and the paper's F-DHT pair list (§6.1 p.446). Same pinned
BERT-base, AG News WordPiece split, seed 7, batch 8, three 114,000-sample epochs,
validation selection, and one public-test evaluation of the selected checkpoint.

## Accuracy and compute (public test, 7,600 samples)

| Model | Val macro F1 | Test macro F1 | Paper F1 | Task FLOPs | + routing prefix | Paper inference FLOPs | Mean depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed BERT | .94801 | .94903 | .944 | 1.000x | 1.000x | 1.00x | 12 |
| T-DHT (tree) | .94383 | .93995 | .945 | .386x | .636x | .81x | 4.75 |
| F-DHT (integrated) | .94523 | .94093 | .943 | .432x | .682x | .35x | 5.35 |

F-DHT's validation-selected checkpoint is epoch 2 (validation .93757 → .94523 → .94010).
F-DHT and T-DHT differ by 7 test samples (F-DHT right/T-DHT wrong: 86; reverse: 79),
a tie on one seed. Both trail our fixed baseline by ~0.9 points. The paper's ordering of
inference compute (F-DHT .35x < T-DHT .81x) is not reproduced: ours are .43x and .39x.

Test architecture choices differ sharply: T-DHT depths 3/6/9/12 = 62.6/19.9/14.0/3.4%;
F-DHT = 30.7/62.6/4.4/2.3% with head counts 8/10/12 = 50.5/16.6/33.0% of executed groups.
The two agree on 97.67% of predictions but only 50.8% of depths. Both modes reverse the
paper's reported 3-layer trend: 3 layers dominates early and declines (T-DHT to ~60%,
F-DHT to ~30–40% while 6 layers rises to ~60%). Capped reward ratios: 57,579 for the full
F-DHT run (31,474 at its selected epoch-2 checkpoint) vs 81,574 for T-DHT.

Training step time with the original per-sample engine: fixed 1.88 h, T-DHT 2.96 h,
F-DHT 3.72 h. Per-sample exports reproduce each recorded test macro F1 exactly.

## Batched execution (D017)

`[dht] execution = "batched"` runs the same algorithm for parallel hardware. Tests pin
decisions, TS RNG consumption, logits/gradients and 6-step trajectories to the per-sample
reference; on 512 real validation samples each benchmark found identical architectures and
max logit difference 0.0. Apple M2 Max, MPS, otherwise idle, sequential runs:

| Inference samples/s | batch 8 | batch 32 |
| --- | ---: | ---: |
| Fixed BERT | 181.1 | 263.0 |
| T-DHT per-sample → batched | 69.2 → 159.2 | 67.7 → 236.6 |
| F-DHT per-sample → batched | 62.8 → 133.0 | 52.1 → 221.6 |

| Training samples/s (batch 8, 150 timed steps after 20 warmup, from pretrained BERT) | Fixed | Per-sample DHT | Batched DHT |
| --- | ---: | ---: | ---: |
| T-DHT config | 47.4 | 28.8 (0.61×) | 73.4 (**1.55×**) |
| F-DHT config | 48.2 | 21.5 (0.45×) | 42.2 (0.88×) |

Training benchmarks cover routing, forward, backward, AdamW and bandit updates; one-off
calibration is excluded. They measure the early-training policy (T-DHT .33x / F-DHT .54x
FLOPs in the window), not full-run averages. On CPU, batched T-DHT inference was ~2× faster
than fixed BERT. On MPS, a profile of batched T-DHT inference (batch 32, 4.8 ms/sample)
attributed ~1.1 ms to device→CPU transfer plus float64 pooling of bandit contexts and
~1.1 ms to the prefix. That motivated opt-in GPU-side pooling (D018, below), kept separate
because float32 pooling cannot promise bit-identical contexts. The tables above are from the
first benchmark session; their raw JSON files were overwritten by the D018 session.

## Device-side context pooling (D018)

`context_pooling = "device"` pools bandit contexts on the GPU (float32, one prefix sum per
batch) and scores UCB arms per batch; bandit statistics stay float64 on the CPU. Latest
idle-MPS session (numbers vary roughly ±10% between sessions):

| | Inference b=32 (× fixed) | Training b=8 (× fixed) |
| --- | ---: | ---: |
| T-DHT per-sample / batched / batched+GPU pooling | 0.26 / 0.91 / **1.12** | 0.59 / 1.50 / **1.60** |
| F-DHT per-sample / batched / batched+GPU pooling | 0.24 / 0.87 / **1.11** | 0.44 / 0.87 / **0.96** |

Fixed BERT: 264.5 samples/s inference (b=32), 50.3 / 48.8 samples/s training in the two
sessions. On all 6,000 validation samples GPU pooling chose the same architecture and
prediction as exact float64 pooling for both checkpoints (macro F1 .94383 / .94523,
identical to recorded). In training the backward pass now dominates DHT step time.

## Limitations

One seed; no variance estimate. S-DHT and P-DHT remain unrun. FLOP ratios exclude
backward/optimizer/bandit work and are not the authors' undefined scope. Benchmarks are
single-machine snapshots. The completed full runs used the per-sample engine; no full run
has been retrained with batched execution.

## Artifacts

- Run: `artifacts/runs/ag-news-fdht-full-seed7/` (`final_test.json`, `validation.json`)
- Exports/benchmarks: `artifacts/analysis/` (git-ignored)
- Notebooks: `notebooks/01_dataset_eda_and_dht_examples.ipynb`, `notebooks/02_tree_vs_integrated_search.ipynb`
- Curated record: `experiments/02_related_work/ag_news_fdht_batched_2026-09-22.json`
