# Main AG News run protocol and recovery guide

This is the explicit **D013 reproduction protocol**, not a recovered set of author
hyperparameters. The paper specifies the backbone/actions and suggests 3–6 epochs
(§6.1 pp.445–446; §7.3 p.449), but leaves important experimental details missing.
No full-data run is launched by the readiness checks. Get runtime approval first.

## Frozen choices before looking at main-run results

- Fixed BERT and T-DHT: same pinned pretrained encoder, classifier initialization,
  WordPiece data, model seed 7 and policy seed 17; 12 layers, 12 heads, width 768,
  sequence length 128. T-DHT uses b=c=3 and the published depth/count options.
- Three complete epochs on all **114,000** prepared training records, with no
  cross-epoch wrapping or dropped tails. DHT queues may reorder records and produce
  different numbers of optimizer updates, but each record trains exactly once per
  epoch. Both see 342,000 total sample exposures. Prepared data order is fixed across
  epochs; we do not silently introduce an epoch shuffle or change the dataset split.
- DHT calibration: **512 distinct training records, 128/class**, private seed 42.
  Selection is across the entire configured training set, not its first four rows.
  Class labels are used only to stratify training records, not as bandit context.
  Chosen IDs/counts are saved in `calibration.json` and the controller checkpoint.
  Every count path is calibrated; frozen normalization/once-only descriptors remain
  documented approximations. This is not equivalent to the paper's all-data pass.
- AdamW, peak LR 2e-5, weight decay .01 on all parameters, batch size 8, dropout .1.
  Linear warmup for 10% of trained-sample exposure, then linear decay. No new gradient
  clipping, optimizer parameter grouping or alternative reward formula is introduced.
- LR is evaluated for the next 1-based sample exposure and stays fixed within one
  batch. `scheduled_learning_rate` documents exact endpoints. Progress comes from
  trained samples, not routed samples or optimizer-step counts; resume retains it.
- Evaluate the complete **6,000-example validation split after every epoch**.
  Select maximum macro F1, retaining the earliest checkpoint on an exact tie.
  Evaluate the public **7,600-example test once per selected final model**, only
  after an explicit evaluation command. Full-run configs have `evaluate_test=false`.
- Keep `paper_literal` binary feedback in the primary T-DHT configuration. The
  alternative's readiness result must not silently redefine the reproduced method.

One seed is the initial affordable comparison, not a variance estimate. Subsequent
seeds should use new named configurations/runs and the same data split/protocol.

## Run commands (only after approval)

```sh
uv sync --locked --extra nlp
uv run --extra nlp dht train --config configs/experiment/ag_news_fixed_full.toml --output artifacts/runs/ag-news-fixed-full-seed7
uv run --extra nlp dht train --config configs/experiment/ag_news_tree_full.toml --output artifacts/runs/ag-news-tree-full-seed7
```

The `[training]` section enables the epoch runner. Do not copy the old pilot's
`steps=128` into these configs. Its 1,024-record subset is not the full corpus.
MPS is explicit with fallback disabled. Use another named config for CPU/CUDA;
changing device/config during resume is not promised bitwise reproducible.

`metrics.jsonl` and `metrics.csv` stream and flush each update. They are segment
logs on resume; `summary.json`/checkpoint counters are cumulative. No list of all
sample traces is held in memory. `validation.json` contains cumulative epoch results.
Training timing excludes calibration, validation and checkpoint I/O; do not call
it total end-to-end experiment time. Routing/calibration FLOPs stay separate.

## Checkpoints: latest is not necessarily best

- `checkpoint.pt`: complete latest state, atomically replaced every 1,000 updates,
  at each epoch boundary, and at an explicit stop. An initial post-calibration
  checkpoint protects the expensive descriptor preparation before the first update.
- `best-step-XXXXXXXX.pt`: immutable complete checkpoint for a validation improvement.
  There are at most three candidates for a three-epoch run. Keep them with latest:
  an older recovery checkpoint can legitimately reference an older best candidate.
- `best.pt`: atomically published same-directory hard link to the current candidate;
  it does not duplicate a gigabyte-scale file. Use this for final evaluation.
- Schema 3 stores model/optimizer, bandits, calibration/IDs, queues and contexts,
  CPU/MPS/CUDA and independent RNG states, epoch/cursor/sample counters, schedule
  progress, validation history and best-selection provenance. Schemas 1/2 remain
  readable for previous pilots but cannot be resumed as the new epoch protocol.

For a planned stop, `--stop-after 100` means the **absolute optimizer step**, not
100 additional steps and not an epoch budget. Resuming must keep the original
configuration (including the original total epochs/schedule horizon):

```sh
uv run --extra nlp dht train --config configs/experiment/ag_news_tree_full.toml --stop-after 100 --output artifacts/runs/partial-full
uv run --extra nlp dht train --config configs/experiment/ag_news_tree_full.toml --resume artifacts/runs/partial-full/checkpoint.pt --output artifacts/runs/resumed-full
```

An arbitrary Ctrl-C or exception falls back to the latest completed atomic save.
Do not serialize partially executed optimizer/bandit updates as a clean checkpoint.
Resume writes a NEW directory, copies its referenced best candidate, and carries
selection history forward without recalibration. Missing/changed best artifacts fail
explicitly. Copy/archive the directory together rather than taking only a latest
file that references a best candidate elsewhere. Only load trusted local checkpoints.

After completed training and final protocol review:

```sh
uv run --extra nlp dht evaluate --checkpoint artifacts/runs/ag-news-fixed-full-seed7/best.pt --device mps --split test --output artifacts/runs/ag-news-fixed-full-seed7/final_test.json
uv run --extra nlp dht evaluate --checkpoint artifacts/runs/ag-news-tree-full-seed7/best.pt --device mps --split test --output artifacts/runs/ag-news-tree-full-seed7/final_test.json
```

Do not rerun these evaluations while tuning. Report the paper's reference values
separately and include actual latency; attention slicing does not eliminate dense
optimizer, FFN, routing or controller costs.

## Bounded reward-feedback/readiness check

`scripts/check_long_run_readiness.py` uses separately named readiness configs:
1,024 training records, two epochs, 512 balanced calibration records, 512 validation
records, seed 7. It runs fixed/literal/improvement, then replays literal from its
step-64 periodic checkpoint and attempts exact model/optimizer/controller/RNG state.
It never invokes a full-data config for training or evaluates public test.

```sh
uv run --extra nlp python scripts/check_long_run_readiness.py --output artifacts/runs/my-readiness
```

The two DHT configurations differ only in name and the disputed feedback direction.
The script checks that invariant and logs actual binary-feedback counts. The comparison
is a short one-seed sensitivity diagnostic, not a literature novelty claim or proof
of the authors' intended inequality. The main config stays literal regardless of outcome.

### Observed MPS recovery qualification

The first readiness run's bitwise comparison **failed**; it is not a passing exact
replay. The follow-up `scripts/inspect_readiness.py` leaves that failure intact and
quantifies the discrepancy without changing a tolerance to manufacture a pass.
Live MPS restoration matches the saved weights, optimizer and controller exactly
before further computation. Across the197 replayed updates, sample/head actions
and RNG states match; both epoch validation scores and best-step selection match.
Final weights differ slightly (largest absolute difference about2.07e-5), and a
32-example validation probe has maximum logit difference1.43e-6 with identical
predictions and architectures. The exact backend cause is not established.

CPU fixed/tree/integrated replay remains bitwise tested. MPS is operationally
resumable on the checked stack, not certified bitwise reproducible. This is a
documented limitation, not a reason to remove strict deterministic algorithms or
change the paper method. See the readiness report for complete observed results.
