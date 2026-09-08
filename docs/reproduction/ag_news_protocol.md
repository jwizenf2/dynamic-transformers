# AG News: first paper-backbone pilot

This is a bounded real-data workflow check before a full reproduction, not a claim
to match the paper's Tables1/2. See decision D010 and the ambiguities log. The
published backbone dimensions/actions are specified; initialization, optimizer,
tokenization, validation protocol and some controller details remain interpretations.

## Sources and preprocessing

- CSV data: [AG News mirror and dataset readme](https://github.com/mhjabreel/CharCnn_Keras/tree/555590db4219b1243abb1918effd6a7425a2d75f/data/ag_news_csv),
  revision `555590db4219b1243abb1918effd6a7425a2d75f`. It credits the original
  Zhang/Zhao/LeCun dataset construction; no implementation code from that repository
  is downloaded or used.
- Tokenizer/weights: [BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/86b5e0934494bd15c9632b12f734a8a67f723594),
  revision `86b5e0934494bd15c9632b12f734a8a67f723594`. Standard pretrained encoder,
  NOT an AG News-fine-tuned checkpoint and not author DHT code.
- Use its serialized WordPiece tokenizer, title + space + description, one segment,
  `[CLS]`/`[SEP]`, truncation/padding to128. No extra text cleanup is inferred.
- Labels1..4 map to0..3. Stratified validation holds out1500/class from the original
  training set, leaving114000 training rows. Fixed preparation seed42; independent
  of the model/policy seeds. Public test remains7600 rows. Arrays retain source IDs.
- Every asset and prepared split has a SHA256; the experiment config pins the
  manifest's SHA256. Loading/resuming fails if the corpus changes underneath a run.

## Model and comparison

Both modes have12 layers,12 heads,768 width,3072 FFN, dropout0.1 and LayerNorm
epsilon1e-12. The learned positional table retains BERT's512 positions; the data
and actual execution length are128. Token vocabulary30522. Copy every embedding
and encoder weight into our own physically selectable modules with strict shapes.
Legacy LayerNorm gamma/beta names map exactly to weight/bias. No pooler/MLM/NSP
head is used. A shared first-token four-class linear head is initialized identically
in the two modes (seed7, normalstd0.02, bias0). This head choice is not fully specified
by the paper, so do not claim exact Hugging Face sequence-classifier equivalence.

Fixed baseline always executes12 layers/all12 heads. Tree uses b=c=3, depths
3/6/9/12, head counts8/10/12, non-Beta Gaussian logistic TS, and D005–D008 conventions.
The pilot's four-record calibration is intentionally bounded, not a recommended
full-reproduction feature estimate. Literal paper feedback remains unchanged.

## Predeclared pilot budget and interpretation

128 AdamW updates, batch8, learning rate2e-5, weight decay0.01 on all parameters,
no warmup/clipping/schedule, first1024 shuffled training rows, first128 validation
rows, no test evaluation. Train order is fixed; DHT depth queues can reorder samples
and release partial batches. Record actual exposures: equal updates need not mean
equal samples. No test-based hyperparameter search or best-checkpoint selection.
Use final validation accuracy and macro/micro/weightedF1. Small validation uncertainty
is substantial; do not subtract these numbers from the paper's public test F1.

After the two pilots completed, widen read-only evaluation to ALL6000 already
held-out validation examples for BOTH unchanged checkpoints. No extra updates,
calibration, hyperparameter choices or checkpoint selection occur. This diagnostic
extension makes the pilot comparison less sensitive to the first128 examples; it
does not turn validation results into public-test results. The script
`scripts/evaluate_full_validation.py` saves both the original checkpoint config and
the effective evaluation size. Public test remains untouched.

The paper reports3–6 epochs (§7.3 p.449) without sufficient optimization settings.
Before a full run, fix an epoch/sample-exposure budget, validation checkpoint
selection, schedule/warmup, calibration size, seeds and hardware/runtime budget.
Run matched fixed and T-DHT, evaluate selected checkpoints once on the full test,
then compare against authors' AG News FullBERT .944 / T-DHT .945 and reported
training/inference ratios1/1 versus.42/.81. Our partial matmul counts are not those
training ratios. Actual routing/controller overhead must be measured separately.

## Commands

```sh
uv sync --locked --extra nlp
uv run --extra nlp python scripts/prepare_ag_news.py
uv run --extra nlp python scripts/fetch_bert.py
uv run --extra nlp dht train --config configs/experiment/ag_news_fixed_pilot.toml --output artifacts/runs/my-ag-fixed
uv run --extra nlp dht train --config configs/experiment/ag_news_tree_pilot.toml --output artifacts/runs/my-ag-tree
uv run --extra nlp python scripts/verify_execution.py --checkpoint artifacts/runs/my-ag-tree/checkpoint.pt --output artifacts/runs/my-ag-tree/execution.json
uv run --extra nlp dht benchmark --checkpoint artifacts/runs/my-ag-tree/checkpoint.pt --device mps --warmup 5 --repeats 20 --output artifacts/runs/my-ag-tree/benchmark.json
```

Preparation requires a NEW output directory. Training/evaluation never download
anything. Confirm the printed manifest hash matches the config (do not bypass
integrity checks if it differs). CPU use is supported by changing the device in
a new named configuration; the published-size pilot configs explicitly require MPS.
Read live `metrics.jsonl` during training. `--stop-after` saves a resumable checkpoint
at a chosen update boundary; complete weights/optimizer/policy state can be large.
