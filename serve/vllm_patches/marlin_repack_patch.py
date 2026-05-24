"""
Monkey-patch for vLLM's `prepare_nvfp4_moe_layer_for_marlin` (in
`vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py:288`).

PROBLEM
=======
The upstream implementation per MoE layer does:

    tensor_list = []
    for i in range(E):                                     # E = 512 experts
        tensor_list.append(ops.gptq_marlin_repack(weight[i]))
    return torch.cat([x.unsqueeze(0) for x in tensor_list], 0)

This holds all 512 per-expert tensors AND the cat-output simultaneously,
peaking at roughly 2x the layer's weight size per call. With 8 MoE layers
times 2 weight types (w13, w2) and lazy garbage collection, the transient
can accumulate to ~50 GB on top of the model footprint. On a unified-memory
GPU with 130 GB total (DGX Spark / GB10), the load itself sits at ~69 GB
and the repack peak pushes the system past the physical ceiling.

PATCH
=====
Replace the `tensor_list + torch.cat` pattern with a preallocated output
tensor and in-place per-expert writes. After the first expert is repacked
to discover the output shape, the destination is allocated once and each
subsequent expert is written directly into its slice. Per-call live memory
becomes:

    original weight (3 GB) + output destination (3 GB) + one expert temp = ~6 GB

vs the upstream ~9-12 GB per call (or 50+ GB across calls with lazy GC).

The same pattern is applied to `premute_scales`.

CALL THIS BEFORE vLLM's loader runs:
    from marlin_repack_patch import apply_patch
    apply_patch()

Verified safe for vLLM 0.21 source (file marlin_utils_fp4.py:288-394).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def _patched_prepare_nvfp4_moe_layer_for_marlin(
    layer,
    w13,
    w13_scale,
    w13_scale_2,
    w2,
    w2_scale,
    w2_scale_2,
    is_act_and_mul,
):
    # Lazy import - only resolve symbols after vLLM is loaded
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        _nvfp4_compute_scale_factor,
        get_marlin_input_dtype,
        marlin_make_workspace_new,
        marlin_permute_scales,
        nvfp4_marlin_process_global_scale,
        nvfp4_marlin_process_scales,
    )

    # Original "FP4 emulation via Marlin" warning is upstream - we don't repeat it.

    input_dtype = get_marlin_input_dtype(prefix="")
    if input_dtype is not None and input_dtype.itemsize == 1:
        raise RuntimeError("NVFP4 weight + INT8/FP8 activation is not supported.")

    GROUP_SIZE = 16
    E = layer.num_experts
    K = layer.hidden_size
    N = layer.intermediate_size_per_partition

    device = w13.device
    param_dtype = layer.params_dtype
    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1

    # Workspace + empty perm tensor are upstream-identical
    layer.workspace = marlin_make_workspace_new(device, 4)
    perm = torch.empty(0, dtype=torch.int, device=device)

    def repack_weight_inplace(weight, name):
        """Chunked / preallocated replacement for repack_weight."""
        num_shards = 2 if is_act_and_mul else 1
        if "w13" in name:
            size_n, size_k = N * num_shards, K
        else:
            size_n, size_k = K, N

        assert weight.shape == (E, size_n, size_k // 2), (
            f"weight shape mismatch: got {tuple(weight.shape)} "
            f"expected ({E}, {size_n}, {size_k // 2})"
        )

        # First expert: discover output shape + dtype, allocate destination.
        first_qw = weight[0].view(torch.int32).T.contiguous()
        first_out = ops.gptq_marlin_repack(
            b_q_weight=first_qw,
            perm=perm,
            size_k=size_k,
            size_n=size_n,
            num_bits=4,
            is_a_8bit=is_a_8bit,
        )
        del first_qw
        out = torch.empty(
            (E, *first_out.shape), dtype=first_out.dtype, device=device
        )
        out[0].copy_(first_out)
        del first_out

        # Remaining experts: in-place write, drop temporaries each iteration.
        for i in range(1, E):
            qw = weight[i].view(torch.int32).T.contiguous()
            mq = ops.gptq_marlin_repack(
                b_q_weight=qw,
                perm=perm,
                size_k=size_k,
                size_n=size_n,
                num_bits=4,
                is_a_8bit=is_a_8bit,
            )
            out[i].copy_(mq)
            del qw, mq

        return out

    w13 = repack_weight_inplace(w13, "w13")
    w2 = repack_weight_inplace(w2, "w2")

    def premute_scales_inplace(scales, g_scales, name):
        """Chunked / preallocated replacement for premute_scales."""
        scales = scales.to(param_dtype)
        num_shards = 2 if is_act_and_mul else 1
        if "w13" in name:
            size_n, size_k = N * num_shards, K
        else:
            size_n, size_k = K, N

        combined_scale_factor = _nvfp4_compute_scale_factor(scales, param_dtype)

        # First expert: discover scale tensor shape + dtype.
        first_perm = marlin_permute_scales(
            s=scales[0].T,
            size_k=size_k,
            size_n=size_n,
            group_size=GROUP_SIZE,
            is_a_8bit=is_a_8bit,
        )
        first_proc, _ = nvfp4_marlin_process_scales(
            first_perm, scale_factor=combined_scale_factor, a_dtype=param_dtype
        )
        del first_perm
        out_scales = torch.empty(
            (E, *first_proc.shape), dtype=first_proc.dtype, device=device
        )
        out_scales[0].copy_(first_proc)
        del first_proc

        for i in range(1, E):
            sp = marlin_permute_scales(
                s=scales[i].T,
                size_k=size_k,
                size_n=size_n,
                group_size=GROUP_SIZE,
                is_a_8bit=is_a_8bit,
            )
            proc, _ = nvfp4_marlin_process_scales(
                sp, scale_factor=combined_scale_factor, a_dtype=param_dtype
            )
            out_scales[i].copy_(proc)
            del sp, proc

        g_scales = nvfp4_marlin_process_global_scale(g_scales, param_dtype)
        g_scales = g_scales / combined_scale_factor
        return out_scales, g_scales

    w13_scale, w13_scale_2 = premute_scales_inplace(w13_scale, w13_scale_2, "w13")
    w2_scale, w2_scale_2 = premute_scales_inplace(w2_scale, w2_scale_2, "w2")

    return w13, w13_scale, w13_scale_2, w2, w2_scale, w2_scale_2


def apply_patch():
    """Replace the upstream function. Call once before vLLM loads weights."""
    import vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 as mod

    if getattr(mod.prepare_nvfp4_moe_layer_for_marlin, "_is_chunked_patch", False):
        return  # already patched

    original = mod.prepare_nvfp4_moe_layer_for_marlin
    _patched_prepare_nvfp4_moe_layer_for_marlin._is_chunked_patch = True
    _patched_prepare_nvfp4_moe_layer_for_marlin._original = original
    mod.prepare_nvfp4_moe_layer_for_marlin = _patched_prepare_nvfp4_moe_layer_for_marlin

    # Also patch the caller's local name if it has already imported the symbol.
    try:
        import vllm.model_executor.layers.fused_moe.oracle.nvfp4 as caller_mod

        if hasattr(caller_mod, "prepare_nvfp4_moe_layer_for_marlin"):
            caller_mod.prepare_nvfp4_moe_layer_for_marlin = (
                _patched_prepare_nvfp4_moe_layer_for_marlin
            )
    except ImportError:
        pass

    logger.info(
        "Applied marlin_repack_patch: chunked + preallocated NVFP4 MoE repack."
    )
