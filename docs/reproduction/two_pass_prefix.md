# Why the same first three layers execute twice

**Yes: the ordinary DHT training implementation follows the paper's two stages.**
The first pass does not update the network. The second pass recomputes the prefix
and includes it in task backpropagation. The prefix is not permanently frozen.

## Direct paper evidence

Source: the supplied `dynamic_transformers_reference_paper.pdf`, inspected as text
and rendered pages. Printed page numbers below differ from the PDF page index.

- §4.2.1, printed p.443, Step 1: run the first b layers with inherited weights to
  obtain S1, then use UCB_L to select depth. Step 2 explicitly says the sample
  passes through those layers again to S2,0 and describes backpropagation updates.
- §4.2.2, p.443: integrated search also obtains its pair decision at S1, then
  reruns the first b layers to S2,0.
- §5, p.444: routing uses historical prefix weights without updating the network
  until a training batch is formed. Samples may have entered a queue at different
  times; routing representations need not come from identical model versions.
- Algorithm 2, p.444, lines 2–6 describe routing and line 7 explicitly reruns the
  prefix. Figure 1, p.445, draws separate Stage 1 and Stage 2 b-layer blocks.
- §6.1, pp.445–446, sets b=c=3 for the ordinary experiments.

The paper's Step 2 wording does not mean SGD occurs halfway through a forward.
Our implementation computes final cross entropy, runs backward through the entire
selected Stage 2 path, then makes one optimizer update for the released depth batch.
This ordering is the documented D008 interpretation. The two blocks use the same
parameters; a separate frozen encoder or EMA copy would be a different method.

## Code path

| Stage | Actual work | Gradient/update behavior |
| --- | --- | --- |
| `DHTController.route` | Embed tokens; execute layers 0,1,2 with all heads; select depth/pair | `@torch.no_grad()`; no graph, optimizer or bandit update |
| `DHTTrainer.next_batch` | Store sample IDs and original decision contexts in depth queues | No cached task activations; queues can span optimizer updates |
| `DHTController.execute` | Embed tokens AGAIN; execute layers 0,1,2 AGAIN; choose and execute later groups | Fresh task graph through embeddings, prefix, selected later heads/layers and classifier |
| `DHTTrainer.step` | Mean cross entropy, backward, optimizer step, observed bandit updates | Prefix receives task gradients and is updated with the rest of the executed path |

`model.prefix()` is intentionally neutral about gradient mode. Calling it from
`route()` suppresses gradients; calling it from the training execution preserves
them. `model.eval()` is not a substitute for `no_grad()`: eval disables dropout,
whereas no_grad controls the graph. Routing dropout-off is our D008 engineering
choice; the paper does not specify dropout behavior in each stage.

For a sample routed to depth6, the layer-call sequence is
`[0,1,2] -> queue -> [0,1,2,3,4,5]`; layers6–11 do not execute. For depth3,
both passes still execute `[0,1,2]`, and no head-selection group is needed. All
prefix heads remain active. Routing's extra three-layer forward costs FLOPs even
though it does not backpropagate; estimates report it separately from task work.
Evaluation freezes both passes and the policies, while still reporting routing cost.

## Executable regression

`tests/integration/test_prefix_passes.py` instruments real layer calls and outputs
through `DHTTrainer.step()`, including real queue release and optimizer update.
It tests tree/integrated, depth3/depth6, and CPU/available MPS with b=c=3 and
12 head identities (tiny hidden width; synthetic tokens; no downloads).

It verifies both prefixes actually execute, all prefix heads are present, routing
has no graph or weight changes, training uses distinct activations with gradients,
each prefix layer receives a nonzero task gradient and changes after the update,
and layers past the selected depth are neither executed nor updated. Existing
attention tests additionally check inactive head gradients and matrix dimensions.
Calibration is finished before hooks are installed: its separate once-only
descriptor passes are not miscounted as per-sample routing or training.

```sh
uv run --extra nlp pytest tests/integration/test_prefix_passes.py -v
```

This ordinary-prefix behavior is separate from P-DHT's later frozen-layer phase,
which remains deferred because its published configurations have additional ambiguities.
