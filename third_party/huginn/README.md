# huginn (vendored)

Modeling code of Huginn-0125 (Geiping et al. 2025, "Scaling up Test-Time Compute with Latent
Reasoning: A Recurrent Depth Approach", arXiv 2502.05171): a 3.5B recurrent-depth model with a
2-layer prelude, a 4-layer recurrent block applied `num_steps` times with the input embedding
re-injected at every step (an adapter on `[state, embedding]`), and a 2-layer coda; trained with a
random number of steps per sample (log-normal Poisson, mean 32, backpropagation through the last 8).
Copied unmodified from the Hugging Face repo `tomg-group-umd/huginn-0125` at revision
`bb6621b65e90b6a4b9b29ef88dc83866d450470c` (license Apache-2.0 per the model card; the repo has
no separate LICENSE file). Vendored so the model is loaded from audited, pinned code instead of
`trust_remote_code`.

| File | Upstream path | SHA256 |
|---|---|---|
| `raven_modeling_minimal.py` | `raven_modeling_minimal.py` | `a1d447da93c605a6dc11fbc17cbc7185663f4a41ae12335b34b5847edf0f84aa` |
| `raven_config_minimal.py` | `raven_config_minimal.py` | `36a39939e76dc9e8f7563187c6498e012ca7f64a3d86b32ff1daa7f4c41a4278` |

Audit (2026-11-12): imports only `torch`, `transformers`, the standard library and the sibling
config module; no file, network, subprocess or dynamic-import code; the only side effect at import
is registering the classes with `transformers`' Auto classes. Attention uses
`torch.nn.functional.scaled_dot_product_attention` unless a `flex_attention` block mask is passed
(the forward pass never builds one: `prepared_attn_mask = None`).

Notes for use (from the code and the model card):
* The recurrent state is initialised with random noise on every forward call
  (`initialize_state`), so greedy decoding is deterministic only with a fixed torch seed per call.
* The upstream `HuginnDynamicCache` stores one tensor per token and re-stacks them on every call;
  `scripts/lab/huginn_model.py` supplies a contiguous cache with the same `update` interface for
  fixed-step decoding.
* Fewer than 4 steps give coarse outputs; the card reports gains up to ~64 steps; run in bfloat16.
