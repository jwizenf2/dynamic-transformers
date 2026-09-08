# Reproducibility issue log

Status as of 2026-09-08, updated after real-data readiness checks. Adopted conventions
D005–D014 implement gaps and engineering repairs transparently; they do not resolve the source. The supplied PDF is authoritative. “Adopted” below
describes our documented implementation choice, not resolution of the source.
No author repository or supplementary artifact has been verified.

| ID | Evidence / issue | Alternatives and recommendation | Status / impact |
| --- | --- | --- | --- |
| A01 | §4.2 pp.442–443, Alg.2 p.444: total versus remaining L | Total depth (recommended/adopted D001) matches Fig.1 p.445 and §6.1 options; remaining depth would exceed backbone | Source unresolved; scaffold uses total |
| A02 | §3.1 p.441: w is 2d, A is d×d, C is k×2d | Use actual aggregate p, A p×p, b p, scalar y; alternative literal shapes cannot produce scalar ridge model | Implemented dimensional repair; D005; source unresolved |
| A03 | §3.2 pp.441–442: zero inverse covariance is singular; mean I for a vector | Positive λ_ts diagonal precision and ones mean (closest textual interpretation); zero mean is alternative modification | Implemented positive diagonal precision/ones mean; D005 |
| A04 | §3.2 p.442: positive quadratic under argmax; r∈{0,1} in log σ(r score); g versus p optimization | Literal objective invalid/unbounded. Recommend minimize binary cross entropy plus positive quadratic, with prior mean/precision fixed during update; signed labels would be another repair | Implemented stable penalized Bernoulli objective under user direction; D005 |
| A05 | §3.2 p.442: precision matrix versus component-square update; subset-only scalar feedback | Recommend diagonal Gaussian, new observations only per online update, broadcast observed subset label to selected heads only. Alternative full Hessian uses outer products. Superarm-specific model is not specified | Implemented diagonal online Laplace and selected-only shared label; D005 |
| A06 | §4.3 pp.443–444: δ_H and δ_cp described but not explicitly defined | Candidate loss×H and loss×L×H respectively; alternative loss×L×H for both. Keep loss type explicit (cross entropy proposed) | Implemented loss×H and loss×L×H interpretations; D007 |
| A07 | §4.3 p.444: literal TSB 0 for continuous reward >1, 1 otherwise, conflicts with prose/top-K | Preserve literal reproduction setting; name inverted feedback as a separate ablation | Two-epoch bounded diagnostic completed (D013); similar validation scores do not resolve source; primary stays literal |
| A08 | §4.3 pp.443–444: no initial history, zero-denominator, clipping or scales | Recommend first completed batch initializes history without bandit updates; clamp denominator to configurable ε, fail nonfinite input, scale=1 engineering default; no clipping unless separate modification | Implemented first-key bootstrap, epsilon and configurable cap; D007 |
| A09 | §§7.1–7.3 p.449: 4+8+16=26 then 3×36; training widths 168/87/13 | Recommend literal bin lists yielding 84/42/12 raw contexts, computed concatenation widths; 168=84+84 and 87=42+42+3 plausible, 13 unexplained. Do not pad/truncate to force widths | Explicit bin-derived widths implemented; 168/87/24 for paper bins; D006 |
| A10 | §6 pp.445–447; §7.3 p.449: backbone init/tokenizer/pretraining, optimizer/LR/batch/warmup/decay/seeds incomplete | “From scratch” p.447 is evidence but insufficient for exact protocol. D010 adopts standard pretrained BERT/WordPiece; D013 specifies three epochs, AdamW/warmup/decay and validation selection | Full protocol executable and readiness-tested; choices are ours, source unresolved |
| A11 | §6.1 p.446: 70/28/1 combinations for 8/10/12 of 12 | Unrestricted counts are 495/66/1. Recommend unrestricted top-K as labeled interpretation; no invented restriction | Unrestricted top-K implemented; unexplained reported restriction stays unresolved |
| A12 | §5 pp.444–445: same depth, different head subsets per batch | Per-sample execution is simplest correctness reference; homogeneous subgroup execution later. Do not force shared head mask and call sample-specific | Per-sample execution within depth batch implemented; D008; optimization deferred |
| A13 | §6.2 p.447: FLOPs estimated by selected layers/heads; detailed accounting absent | Report architecture matmul estimate and actual total time separately. Include FFN/prefix/classifier, routing/calibration, optimizer/backward/controller where measured. Dense masks cannot imply savings | CPU operator profiles verify actual sliced dimensions; separate task/routing/calibration and timing; full training accounting open |
| A14 | §7.4 p.449 references Figures 4/6/7 absent; inconsistent F-IDHT/S-IDHT | Record missing material; do not reconstruct curves | Unresolved |
| A15 | §6.1 p.446 P-DHT contradictory freeze wording; 1L8H violates ordinary prefix; 7L6H/7L9H unusual | Preserve exact list. Freeze all but last two is candidate; which last two (backbone vs selected path) unknown. Separate execution contract needed | Deferred Phase 5; not supported by ordinary specs |
| A16 | §§4.2–5 pp.442–445 arm calibration all samples/full model, refresh frequency and head sampling ambiguous | Use training split only, measured calibration and immutable decision-time features | D013 replaces four pilot records with 512 balanced training records in main configs; once-only subset remains a D006 modification |
| A17 | §7.2 p.449 rolling normalization without algorithm | Online mean/variance versus finite window; recommend training-only running moments, frozen evaluation statistics, epsilon and checkpoint state | Train-calibrated frozen moments implemented; D006 modification |
| A18 | §5 pp.444–445 no tail flush/starvation; §4.3 previous-batch meaning with depth queues | Recommend flush partial queues at epoch end, count actual batch size, per-group history only when observed; global versus depth-conditioned history changes rewards | Epoch-tail flush, actual batch sizes and per-observed-group histories implemented; D007/D008 |
| A19 | §6.2 p.447 F1 averaging, validation split, inference exploration unspecified | Report accuracy and macro/micro/weighted F1; validation-only tuning; deterministic posterior-mean inference candidate versus continued sampling | Explicit F1/frozen inference and D010 training-only validation implemented; source unresolved |
| A20 | §4.1 p.442 head matrices all called d×d, classifier/pooling/bias unspecified | Standard head width hidden/num_heads, full residual width, shared first-token linear classifier (D002/D003/D010), not the standard BERT pooler | Strict pretrained encoder tensor import and pilot tested; exact author classifier unverified |
| A21 | Alg.2 p.444 and §4.2 p.443 update/loss ordering underspecified | Recommend compute detached per-sample losses from executed training pass, one model optimizer step, then observed-action updates using stored contexts; no hypothetical rewards | Stored decision context and detached pre-optimizer loss updates implemented; D008 |
| A22 | Running headers pp.440 etc. name unrelated MSME paper; §6.1 titled Reward Formulation; broken equation labels and repeated Fig.1 prose | Preserve exact PDF provenance; use page+equation description instead of guessing identifiers | Editorial defects; no inference about research validity |

## Evidence search

Additional implementation verification (not a source ambiguity): two-pass prefix
behavior is explicit in §§4.2.1–4.2.2 p.443, §5/Algorithm2 p.444 and Fig.1 p.445.
CPU/MPS tests confirm no routing updates and a fresh trainable second pass; see
[the explanation](two_pass_prefix.md). MPS bitwise continuation failed despite
exact state restoration and matching bounded-run decisions/validation; D014 and
the readiness report preserve the measured numerical drift and unresolved cause.

On 2026-09-07 opened DOI https://doi.org/10.24874/PES06.02.001 (browser resolver
failed) and searched the exact DOI. The publisher issue index
https://pesjournal.net/number2.php lists the supplied title/authors/DOI.
An indexed PDF mirror was surfaced at https://oaji.net/articles/2023/8593-1717677118.pdf
but was not used to replace the supplied PDF. This limited metadata check does
not establish availability or absence of author code or supplements. No author
code was downloaded; no contact messages were sent.
