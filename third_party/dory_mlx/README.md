# dory-mlx (vendored)

MLX implementation of NVIDIA Nemotron-Labs Dory 1.0 4B (`DoryForCausalLM`, a looped nGPT-style
model: 8 input blocks, a 20-block shared recurrent stack run 1–4 times with input re-injection,
8 output blocks). Copied unmodified from https://github.com/internet-projects/dory-mlx at commit
`0184a4537fe13e519dc5f90d862fcdd5c561a3bc` (Apache-2.0, see `LICENSE`; community port by
NetProjects, not affiliated with NVIDIA). Vendored because lab jobs reach Hugging Face but not
GitHub, and our `transformers` 5.17 has no `dory` architecture.

| File | Upstream path | SHA256 |
|---|---|---|
| `model.py` | `dory_mlx/model.py` | `658742f412aa41163232affd65c0ef25ff790a17decff2186645fd93ea773404` |
| `__init__.py` | `dory_mlx/__init__.py` | `fe5b95147d7241d06eca0f173ff3fce16d913340d6563a5ac961de0a52737432` |

Upstream validation (its README, not re-measured here): logits vs an independent fp32 PyTorch
reference, 2 loops, 55 tokens: top-1 agreement 100%, KL 0.001–0.003; WikiText-2 perplexity at
1/2/4 loops 12.17/11.76/11.69 (bf16). Code audited 2026-10-21: plain MLX modules, no network
access other than `huggingface_hub.snapshot_download` in `load`, no dynamic code execution.

Weights are NVIDIA's BF16 checkpoint (`nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`,
OpenMDW-1.1), loaded directly: `Model.sanitize` fuses q/k/v/g and gate/up projections at load.

Loop count: `model.args.n_recurrent_loops` is read on every forward call. With
`recurrent_kv_cache_mode="per_loop"` the cache plan is built on the first call for the loop count
set then, so set the maximum (4) before the first call and lower it per call to run fewer loops.
