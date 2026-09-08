"""Measure executed CPU matrix operations, not a count inferred from head masks.

PyTorch's profiler estimates FLOPs only for supported operators. We deliberately
sum mm/addmm/bmm events and compare those with our same-scope analytic estimate.
Neither quantity measures total training work. Operator shapes and layer hooks
provide independent evidence that inactive heads/layers do not execute.
"""

from collections.abc import Callable

import torch
from torch import Tensor

from dht.evaluation.flops import forward_matmul_flops
from dht.models.architecture import ArchitectureSpec
from dht.models.transformer import TransformerClassifier


def profile_forward(model: TransformerClassifier, forward: Callable[[], Tensor]) -> dict:
    """Profile one eval/no-grad CPU call and restore the model's previous mode.

    `forward` can include routing/policy work, but the estimator comparison in
    verify_execution below applies only to a deterministic architecture task path.
    Profiling is not latency benchmarking: instrumentation itself adds overhead.
    """
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("operator FLOP verification currently requires CPU")
    visited = []
    handles = [
        layer.register_forward_pre_hook(
            lambda module, args, i=i: visited.append(
                {"layer": i, "batch_size": len(args[0]), "head_indices": list(args[2])}
            )
        )
        for i, layer in enumerate(model.layers)
    ]
    was_training = model.training
    model.eval()
    try:
        with (
            torch.no_grad(),
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                with_flops=True,
            ) as trace,
        ):
            logits = forward()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite profiled logits")
    finally:
        model.train(was_training)
        for handle in handles:
            handle.remove()
    operators = [
        {
            "operator": event.name,
            "input_shapes": event.input_shapes,
            "flops": event.flops,
        }
        for event in trace.events()
        if event.name in {"aten::mm", "aten::addmm", "aten::bmm"}
    ]
    return {
        "scope": "executed forward mm/addmm/bmm; not total runtime or training FLOPs",
        "layers": visited,
        "operators": operators,
        "matmul_flops": sum(event["flops"] for event in operators),
    }


def verify_execution(
    model: TransformerClassifier, tokens: Tensor, selected: ArchitectureSpec
) -> dict:
    """Compare full, sliced and dense-zero-mask paths with identical weights/inputs.

    The dense reference still skips layers past selected.total_depth. Its retained
    layers compute ALL heads, then multiply inactive outputs by zero. Thus it must
    match the sliced logits, but NOT the sliced head-operation count.
    """
    selected.validate(model.config)
    full = ArchitectureSpec(model.config.total_depth)
    modes = {"full": (full, False), "sliced": (selected, False), "dense_masked": (selected, True)}
    reports, outputs = {}, {}
    for name, (spec, dense) in modes.items():

        def execute(spec=spec, dense=dense, name=name):
            outputs[name] = model(tokens, architecture=spec, dense_masked=dense)
            return outputs[name]

        reports[name] = profile_forward(model, execute)
        expected_spec = ArchitectureSpec(spec.total_depth) if dense else spec
        expected = len(tokens) * forward_matmul_flops(model.config, expected_spec, tokens.shape[1])
        if reports[name]["matmul_flops"] != expected:
            raise AssertionError(f"{name}: profiler/analytic matmul FLOPs disagree")
        if [row["layer"] for row in reports[name]["layers"]] != list(range(spec.total_depth)):
            raise AssertionError(f"{name}: unexpected layer execution")
        reports[name]["analytic_matmul_flops"] = expected
    torch.testing.assert_close(outputs["sliced"], outputs["dense_masked"], rtol=1e-5, atol=1e-6)
    return {
        "selected_architecture": selected.to_dict(),
        "batch_size": len(tokens),
        "sequence_length": tokens.shape[1],
        "max_sliced_dense_logit_difference": (outputs["sliced"] - outputs["dense_masked"])
        .abs()
        .max()
        .item(),
        "paths": reports,
        "sliced_to_full_matmul_ratio": reports["sliced"]["matmul_flops"]
        / reports["full"]["matmul_flops"],
        "sliced_to_dense_masked_matmul_ratio": reports["sliced"]["matmul_flops"]
        / reports["dense_masked"]["matmul_flops"],
    }
