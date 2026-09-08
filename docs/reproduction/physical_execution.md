# What is really skipped?

The production path does **not** multiply inactive heads by zero. That operation
exists only in `dense_masked=True`, a deliberately inefficient correctness reference.
The methods follow §4.1 p.442 semantically; gathering weights before multiplication
is our physical implementation of its masked attention sum (decision D003).

## Follow one selected head through the code

Each original head owns `head_dim = hidden_size / num_heads` contiguous channels.
For original head IDs `(0,2,5)`, attention.py constructs their original channel IDs:

1. Gather only those **rows** of each Q/K/V weight and bias BEFORE `F.linear`.
   Q, K and V have shape `[batch,3,sequence,head_dim]`, not all available heads.
2. Multiply QK and AV only for these three heads. No scores, softmax or attention
   outputs exist for unselected heads in this forward graph.
3. Gather the matching original **columns** of the output projection. The result
   returns to the full residual width. Do not use the first three head columns:
   that would change head identities and produce the wrong function.
4. Run the full-width feed-forward sublayer of the retained layer. Head selection
   does not prune FFN channels or the residual stream.

Full, canonically ordered head sets bypass gathering entirely, saving unnecessary
copies and backward scatters. Permuted full head sets still preserve positions by
gathering. Dense masking always computes every head, then multiplies by the mask.

Depth is physical control flow: `range(spec.total_depth)` in the fixed/spec path;
the controller breaks before any group starting at/after selected total depth.
Later layers are not called, so their attention AND FFN work are avoided. Ordinary
DHT always executes the first b=3 layers with all heads. Training's no-grad routing
pass executes those three again; this overhead cannot be credited as savings.

## Evidence, not just a mask count

`scripts/verify_execution.py` runs a CPU operator profiler on the actual full,
sliced and dense-reference functions with identical weights and real input tokens.
It checks layer-call traces, actual mm/addmm/bmm dimensions, exact agreement with
our same-scope analytic count, and sliced/reference logit equivalence. Given a
learned checkpoint it also traces the routing prefix and selected task path,
asserting that every visited layer and head set agrees with the returned spec.

The offline regression test checks flattened attention bmm batch dimensions:
batch2 × full4 heads = 8, versus batch2 × active2 heads = 4. Tests separately
check inactive task gradients, invariance to changing inactive weights, output
column identities, and absence of any gradient in wholly skipped layers.

PyTorch only estimates FLOPs for supported operators. We sum mm/addmm/bmm, not
every nonzero profiler FLOP field, to match our documented estimator's scope.
See [PyTorch profiler documentation](https://docs.pytorch.org/docs/2.14/profiler.html).
Profiler instrumentation is NOT used as an inference latency benchmark.

## What still costs work

- Padding is still processed in projections/FFNs, even though attention ignores
  padded keys. Shorter batches or variable-length kernels are separate optimizations.
- Gathering copies active weights. Its memory traffic and launches can dominate
  a small matrix multiplication, particularly with per-sample masks.
- Backpropagation computes smaller selected projection matmuls, but our unique-index
  gather backward writes into **dense** zero-filled parameter-gradient buffers.
  The unique-index assignment avoids the nondeterministic accumulating scatter
  rejected by MPS; CPU finite-difference and CPU/MPS dense-reference gradient tests
  verify its derivative (D012). Inactive slices have zero task gradients, not zero
  allocation or memory-write cost.
- AdamW uses dense tensors for a partially active parameter. Momentum/weight decay
  can update inactive slices, and perform arithmetic on them. Entire skipped layers
  with `grad=None` are different: PyTorch's optimizer skips those parameters on that
  step. A head inactive for one sample can also be active for another in the batch.
- Prefix routing, train-only calibration, CPU bandits, CPU/GPU context transfers,
  queues and Python dispatch are real costs. Controller-selected per-sample paths
  are not as efficiently batched as the fixed baseline.

Therefore: physical slicing avoids inactive-head attention operations, but does
not guarantee proportional TRAINING or wall-clock savings. Removing dense optimizer
work would require a separately designed sparse/per-head optimizer/state layout;
that is not silently introduced as a faithful paper reproduction. A future safe
execution optimization is batching samples with identical currently selected
head sets at each group; head choices must still depend on the actual preceding
hidden states, not on hypothetical unexecuted groups.
