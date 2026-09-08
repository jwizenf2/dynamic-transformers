# Decision log

Classification: **explicitly specified** = directly supported by the supplied PDF;
**reasonable interpretation** = makes incomplete text executable; **our modification**
= an engineering choice or alternative that is not established by the paper.

## D001 — Total depth boundary (2026-09-07, adopted for scaffold)

Section 4.2 pp. 442–443 first defines L as executed depth but later stops at b+L;
Algorithm 2 p. 444 repeats this inconsistency. Figure 1 p. 445 says L=b+oc;
Section 6.1 pp. 445–446 uses 3/6/9/12 in a 12-layer model with b=c=3.
Alternatives: treat L as total depth, or remaining depth (which would execute 15
layers for the published 12 option). Use `total_depth` including prefix, and
`dynamic_depth = total_depth - prefix_depth`. Execute indices [0,total_depth).
This is a **reasonable interpretation**, not a verified correction of the paper.
Ordinary DHT requires dynamic_depth divisible by group_size. Prefix-only actions
execute no head bandits. This decision precedes paper-specific scaffold code.

## D002 — Development backbone (2026-09-07, adopted)

**Our modification:** a small randomly initialized BERT-style post-LayerNorm encoder,
learned token/position/type embeddings, GELU feed-forward, and first-token linear
classifier. No pretrained BERT equivalence or paper initialization fidelity claimed.
Section 4.1 p. 442 specifies masked attention; Figure 1 p. 445 ends with linear
classification. Hidden size, dropout, classifier pooling, FFN width, and initialization
are explicit configuration choices. Published dimensions are reference-only for now.

## D003 — Physical head slicing (2026-09-07, adopted)

**Reasonable interpretation** of Section 4.1 p. 442 masked sum: slice Q/K/V projection
rows and output projection columns before matrix multiplication, preserving original
head identities. Keep a dense mask reference path for correctness. Residual/FFN widths
remain full. One ArchitectureSpec applies to one homogeneous batch; heterogeneous
sample execution and depth queues are deferred, not approximated by a shared policy.
Zero task gradient on inactive slices does not imply optimizer momentum or weight
decay cannot change them. No runtime speedup is assumed.

## D004 — Local workflow (2026-09-07, adopted)

**Our modification:** uv lockfile, Python 3.12.12 (installed), PyTorch, Pydantic validation,
TOML, pytest/Ruff. CPU synthetic smoke, AdamW and explicit seeded RNGs. Runs save
configuration, environment, code state, metrics and trusted checkpoints. These defaults
are engineering validation settings, not reconstructed paper hyperparameters.

## D005 — Executable bandits (2026-09-07, adopted)

The user requested a reliable, readable reproduction base without waiting for every
paper ambiguity to be resolved. Adopt the following transparent interpretations.
§3.1 p.441/Algorithm 1: per-arm CPU float64 ridge statistics use the ACTUAL aggregate
width p, with linear solves and deterministic arm-order ties. This is a dimensional
repair, not UCB1. §3.2 pp.441–442: retain the all-ones TS mean interpretation,
replace singular zero precision with positive diagonal prior precision, minimize
stable binary cross entropy plus a Gaussian prior penalty, and use diagonal
Laplace curvature updates from NEW observations only. The MAP solve uses damped
Newton steps with a full temporary Hessian; the stored covariance is diagonal.
Nonconvergent optimization raises explicitly. Share the observed subset label only
among selected heads. Save the posterior and its dedicated RNG. These are
**reasonable interpretations/necessary mathematical repairs**, not verified author
code. Unrestricted top-K implements 495/66/1 possibilities for 8/10/12 of 12 heads;
the unexplained 70/28 restriction is not invented (§6.1 p.446, A11).

## D006 — Contexts/calibration (2026-09-07, adopted)

§4.2 pp.442–443 and §7.1 p.449: use multiscale mean pooling and concatenated
sample/arm vectors; append a dynamic-group one-hot to UCB_H. Explicit paper bins
produce raw widths 84/42/12 and aggregate widths 168/87/24 for the published
three-group backbone. TSB's 24 (context+arm) does not claim to explain printed 13.
The tiny CPU configs use smaller bins for fast tests. Adaptive pooling removes
padding first and overlaps regions when an input dimension is smaller than bins.

**Our modification:** calibrate once on a configurable training-only subset,
executing each allowed head count plus a full-head path for every calibration
sample. Reuse the prefix within calibration and measure all these path costs.
Average corresponding depth/group/count outputs for UCB arms; average original
embedding-position chunks for TS arm descriptors. These chunks are NOT raw isolated
attention-head outputs. Fit normalization with Welford moments during calibration
and then freeze it (population variance; constant dimensions use scale one).
This simplifies the unspecified rolling normalization/refresh schedule and keeps
the ridge coordinate system stable. Calibration subset and frozen normalization
are deviations, not a claim of exact full-training-set calibration fidelity.

## D007 — Reward interpretation (2026-09-07, adopted)

§4.3 pp.443–444: use cross-entropy times total depth for depth rewards, loss×head
count for head rewards, and loss×total depth×head count for integrated rewards.
The latter two are **reasonable interpretations** of missing product definitions.
All rewards retain previous-observed-batch-mean/current-sample-product form.
Use separate histories for depth/pair and each observed head group; skip updates
for a key's first batch, then commit its actual batch mean. Absent groups retain
history and produce no fictitious feedback. Histories are global across depth
queues, except head keys are indexed by group. One shared configurable scale is
an engineering simplification of separate α/β values (defaults all effectively 1).

**Our numerical safeguards:** denominator floor 1e-8, finite input validation,
configurable reward cap 100 (None disables), and a count of clipped rewards.
`paper_literal` preserves 1 for ratio≤1 on printed p.444; `improvement` flips the
direction as a separately named correction. The default preserves print despite
its conflict with the prose; the implementation does not claim the conflict is resolved.

## D008 — Routing, queues and evaluation (2026-09-07, adopted)

§4.2/Algorithm 2 and §5/Algorithm 3 pp.443–445: no-gradient full-prefix routing,
depth FIFO, gradient-enabled prefix re-execution, per-group count/subset decisions,
one mean-loss optimizer step per released batch, then observed-action updates using
stored decision-time contexts and pre-optimizer per-sample losses. Tree and
integrated search share `policies/controller.py` instead of near-duplicate modules.

**Our engineering choices:** routing disables dropout; training execution respects
configured dropout. Samples execute individually inside the same depth batch so
heterogeneous heads are correct. Epoch tails flush in depth order with no drops;
a step-limited stop preserves pending queues. Fixed cyclic input order avoids
unstated sampler behavior. Checkpoints save queues, calibration, policies, histories,
epoch/cursor and all RNGs. Evaluation uses mean UCB/TS estimates without updates
or sampling, processes samples without queues, and counts both prefix passes.
Per-sample Python execution is a correctness baseline, not a speed-optimized kernel.

## D010 — First real AG News pilot (2026-09-07, adopted before training)

The user requested a paper-baseline comparison and proof of physical skipping.
Use the published 12-layer/12-head/768-width backbone, b=c=3, length128 and T-DHT
actions from §6.1 pp.445–446. The explicitly approved download is the standard
`google-bert/bert-base-uncased` revision `86b5e0934494bd15c9632b12f734a8a67f723594`.
Copy its embeddings/encoder into our selectable implementation, not a dense
wrapper. Preserve the existing first-token linear classifier (no BERT pooler),
initialize it with normal std0.02/zero bias equally in both modes.

**Our protocol, not established by the paper:** pinned AG News CSV mirror,
single-segment title-space-description, serialized uncased WordPiece tokenizer,
128 padded/truncated tokens, 6000 stratified validation records from the original
120000 training rows, fixed split/order seed42 independent of model seed7. The
7600 public test records remain untouched by pilot evaluation. Record byte hashes.
The pilot uses the first1024 shuffled training and first128 validation rows,
128 optimizer updates, batch8, AdamW lr2e-5/weight decay0.01 on all parameters,
dropout0.1, no warmup or clipping. This is an explicit small pilot budget, not the
paper's 3–6-epoch training (§7.3 p.449). Do not tune based on public test scores.
T-DHT retains D005–D008 interpretations and calibrates on only four training records
to bound initial profiling cost. Equal optimizer-step budgets do not guarantee
equal sample exposure because queue tails can be partial. Record actual counts.

Profiler audit: compare full versus 9-layer/8-head sliced versus same-depth dense
zero-mask reference. Count executed CPU mm/addmm/bmm shapes and layer calls. These
are supported forward matrix operations, not total training FLOPs or speed proof.
Direct full-head projections now avoid unnecessary index-select copies; sparse
paths still gather before multiplication. No architecture/reward rule was changed.

## D011 — MPS context transfer compatibility (2026-09-07)

The first real MPS T-DHT calibration failed its finite-context check before any
training update. Read-only CPU/MPS comparisons found finite embedding/prefix
activations on BOTH devices, but nonfinite values after the old combined
`hidden[mask].to(device="cpu", dtype=float64)` transfer on this PyTorch2.14/MPS
stack. This is observed backend behavior, not proof of a universal upstream bug.
Transfer original-dtype hidden/mask to CPU first, then index and cast to float64
on CPU. No NaN replacement, dropped features, reward change or altered precision.
This **engineering compatibility repair** preserves our existing math. Add an
optional MPS regression and a paper-backbone read-only diagnostic. Record the failed
launch separately; fixed training does not use FeatureBank and is unaffected.

## D012 — Unique-channel deterministic backward (2026-09-07)

The second T-DHT launch completed a prefix-only update, then MPS rejected the
general index-select accumulating backward under strict deterministic algorithms.
Do not disable the strict setting silently. Attention's head/channel IDs are already
validated unique, so their gather derivative can assign each incoming gradient to
exactly one original channel (`index_copy_`) instead of general duplicate-index
accumulation. Adopt a private custom autograd gather for the sparse path only.
This is an **engineering implementation**, not a changed forward function or
research algorithm. It still allocates dense zero-filled parameter gradients.
Verify first/second derivatives by CPU finite differences and compare all parameter
and input gradients against dense masked attention on CPU and MPS. Full-head
projections bypass the gather and the fixed baseline is unaffected.

## D013 — Long-run readiness protocol (2026-09-08, adopted before diagnostics)

The user approved three targeted readiness improvements, NOT the long runs yet.
Replace four-record calibration in new reproduction configs with512 distinct
training records,128/class for AG News, selected with private seed42. Labels only
stratify this training-only subset; the bandit still receives representation features,
not labels as context. Persist selected IDs/counts and full calibration state.
Retain frozen normalization and once-only descriptors (D006), and explicitly retain
the subset-versus-all-data deviation from §4.2 pp.442–443. Do not repeatedly refresh
coordinates underneath existing ridge/posterior statistics.

New opt-in [training] protocol: three complete training epochs, all114000 prepared
training records per epoch for fixed/T-DHT, full6000 validation at each epoch end,
macroF1 selection with earliest tie, and public test off until a separately requested
final evaluation. Both routes flush partial batches WITHOUT wrapping/repeating or
dropping training records. Data order remains the same prepared shuffle each epoch.
The paper's §7.3 p.449 suggests3–6 epochs but does not establish our optimizer policy:
keep AdamW lr2e-5/weight decay.01 on all parameters, use10% warmup then linear decay
as a function of trained-sample exposure, evaluated for the next1-based sample.
Within a variable-size batch the LR stays fixed. No new clipping/reward schedule.
Record this as **our explicit training protocol**, not recovered author settings.

Keep an atomic latest checkpoint every1000 updates, at epoch boundaries and on
explicit --stop-after; save an initial post-calibration checkpoint. Schema3 adds
epoch/sample/LR-progress counters, validation history and best-selection provenance.
No scheduler restart on resume. Store immutable best candidates; publish best.pt
via an atomic same-directory hard link so a crash cannot invalidate the older best
referenced by a saved latest checkpoint. Keep streaming per-step JSONL/CSV instead
of retaining a semester-scale list of sample traces in memory.

Reward diagnostic is a **separate sensitivity experiment**, not a replacement of
the reproduction setting: two epochs on the same1024 training examples,512 balanced
calibration records,512 validation examples, seed7/policy17, identical model and
schedule. Compare paper_literal (1 if r<=1) with improvement (1 if r>1), §4.3 p.444.
No public test evaluation, no selection between methods based on that test, and no
claim that one small comparison resolves the paper. The main full-run T-DHT config
will keep paper_literal irrespective of which diagnostic score is larger.

The bounded readiness checks also exercise real-backbone fixed training and DHT
interruption/resumption. No multi-hour/full-data run is launched in this session.

## D014 — Readiness qualification and explicit prefix verification (2026-09-08)

**Engineering qualification, not a paper modification:** the bounded real MPS
replay from step 64 to 261 failed bitwise equality. Do not weaken the strict check
or claim it passed. Separate read-only inspection verified exact live restoration
before computation, matching discrete choices/RNG/validation results afterward,
and small floating differences during continued training. Record these in the
readiness report; the precise backend cause is unverified. CPU exact replay remains
tested. Operational MPS recovery is sufficient for the user's good-enough target,
but future long restarts could diverge at action boundaries. No algorithm or
deterministic-mode setting is changed to conceal this limitation.

The requested prefix audit confirms existing D008, not a new method: §4.2.1
Steps 1–2 and §4.2.2 p.443; §5 / Algorithm 2 p.444; Figure 1 p.445 specify routing
with historical weights, then prefix re-execution for training. Same parameters,
no routing update, no detached activation reuse for the task. Explicit b=c=3
CPU/MPS regressions verify calls, task gradients and actual prefix weight changes.
Routing dropout-off remains our documented choice, not a recovered paper setting.

## D009 — Earlier synthetic diagnostic (2026-09-07)

The user's request to run/monitor/compare authorized a bounded
synthetic diagnostic. Use 300 optimizer steps, seeds 7/11/19, shared tiny model/data
settings and literal feedback. Add a fixed-prefix control after observing early
exits on training/validation logs. This is **our diagnostic experiment**, not an
AG News reproduction protocol. No hyperparameters were tuned on test outcomes.
Audit observation hooks leave core math/RNG/action choices unchanged; their training
timing overhead is excluded from speed claims. Benchmark trained checkpoints
separately, sequentially, with the normal uninstrumented CLI.

## Still open

The source inconsistencies remain unresolved even where we adopted a convention.
Real-data/pretrained pilot choices are adopted under D010, not verified as the
authors' choices. Main training settings are now explicit under D013; P-DHT freeze semantics, cross-device
reproducibility and complete training FLOP/memory accounting remain later milestones.
No speculative research extension has been added.
