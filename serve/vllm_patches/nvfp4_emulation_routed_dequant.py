"""
Runtime monkeypatch: ROUTED-ONLY dequant for the NVFP4 emulation MoE backend
(vLLM 0.22.1, DGX Spark GB10/sm_121). Opt-in via VLLM_PATCH_ROUTED_DEQUANT=1.

PROBLEM
=======
`Nvfp4QuantizationEmulationTritonExperts.apply()` (model_executor/layers/fused_moe/
experts/nvfp4_emulation_moe.py:83-164) dequantizes the FULL w1/w2 -- ALL experts --
to bf16 on EVERY forward (:117-136), THEN calls TritonExperts.apply() which only uses
the top-k ROUTED experts (:148-164). For GLM-4.5-Air (~128 experts/layer) this
re-materializes tens of GB of bf16 per forward to compute a handful of routed experts,
~10-20x slower than the (LoRA-incapable) cutlass backend. The weights are also static
(frozen base), so re-dequantizing every forward is pure waste. This is the bottleneck
behind slow expert-LoRA serving (emulation is the only LoRA-capable NVFP4 MoE path here).

FIX
===
Dequantize only the UNIQUE routed experts this forward: gather w1/w2 + their per-block
scales + per-expert global scales by the routed expert ids, dequant the compact [k,...]
tensors, remap topk_ids into the compact id space, and call TritonExperts.apply() with
the compact weights, global_num_experts=k, and expert_map=None. Expected ~E/k speedup
on the dequant (decode with small top-k: large; big prefill that touches most experts:
little). EXACT vs the full-dequant path (identical math, fewer experts materialized).

When an expert-LoRA context is active, its per-expert tensors are gathered on expert
dimension 1 in the same routed order and installed as a temporary compact
MoELoRAContext for the TritonExperts.apply() call. That keeps base weights, topk_ids,
and LoRA tensors in the same compact id space; the original context is restored before
returning. If the context shape does not match the non-EP global expert set, the
defensive exception path falls back to full dequant.

SAFETY
======
Defensive by construction: falls back to the ORIGINAL apply() (correct-but-slow) on
expert-parallel runs (expert_map is not None), when all experts are routed (no gain),
or on ANY exception. Worst case is the status quo, never a wrong result.

VALIDATED 2026-07-02 on Nemotron-3-Nano-30B-A3B-NVFP4 + expert-LoRA (GB10/sm_121, vLLM 0.22.1):
bit-exact parity (max|logprob delta|=0, base AND adapter) and 5.0x decode (2.56 -> 12.86 tok/s),
0 runtime fallbacks. Evidence: results/cross_arch/emulation_speedup/. Bit-exact parity subsumes
the Spider-EM check (identical logprobs => identical greedy decode). 120B magnitude is a follow-up.

GPU-VALIDATION CHECKLIST (gate before trusting/shipping -- see
docs/plans/emulation_speedup_impl_plan.md):
  1. PARITY: routed-only output == full-dequant output within tight tol (ideally bit-exact),
     base AND with expert LoRA. A diff = remap/scale-gather/workspace bug.
  2. **THE critical check** -- LoRA correctness: with an expert-LoRA adapter, base vs myft must
     diverge AND myft output must EQUAL the unpatched-emulation myft output. This patch gathers the
     global-indexed LoRA tensors into the same compact routed order as the base weights, remaps
     topk_ids to compact ids, and passes expert_map=None/global_num_experts=k. A diff = compact
     id, scale gather, or compact LoRA-context bug.
  3. SPEEDUP: time a fixed token budget patched vs unpatched.
  4. Re-run the Spider EM eval patched -> must match unpatched EM exactly (base AND myft).
OPEN RISKS to confirm on GPU: (a) compact MoELoRAContext must match Punica's expert-dim contract;
parity-with-adapter catches it; (b) workspace13/workspace2 are token-shaped (triton_moe.py:157,
230-235) so passthrough is safe; (c) per-expert global-scale (g1/g2_alphas) gathered only when dim0=E.
"""
from dataclasses import replace
import os
import sys

import torch

_orig_apply = None


def _routed_apply(
    self,
    output,
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    activation,
    global_num_experts,
    expert_map,
    a1q_scale,
    a2_scale,
    workspace13,
    workspace2,
    expert_tokens_meta,
    apply_router_weight_on_input,
):
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        dequantize_to_dtype,
    )

    def _full():
        return _orig_apply(
            self, output, hidden_states, w1, w2, topk_weights, topk_ids,
            activation, global_num_experts, expert_map, a1q_scale, a2_scale,
            workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input,
        )

    try:
        # Expert-parallel (expert_map set) -> don't optimize; original handles the remap.
        if expert_map is not None:
            return _full()

        E = int(global_num_experts)
        routed = torch.unique(topk_ids)
        routed = routed[(routed >= 0) & (routed < E)]
        k = int(routed.numel())
        if k == 0 or k >= E:
            return _full()  # nothing to save

        # --- gather the compact routed-expert weight + scale set (dim0 = expert) ---
        w1_c = w1.index_select(0, routed)
        w2_c = w2.index_select(0, routed)
        w1s_c = self.w1_scale_val.index_select(0, routed)
        w2s_c = self.w2_scale_val.index_select(0, routed)

        def _gather_gscale(g):
            if torch.is_tensor(g) and g.dim() >= 1 and g.shape[0] == E:
                return g.index_select(0, routed)
            return g  # scalar / non-per-expert -> unchanged

        g1_c = _gather_gscale(self.quant_config.g1_alphas)
        g2_c = _gather_gscale(self.quant_config.g2_alphas)

        def _compact_lora_context(ctx):
            if ctx is None:
                return None
            if int(ctx.local_num_experts) != E:
                raise RuntimeError(
                    "active MoE LoRA context does not match the non-EP global "
                    f"expert set: local_num_experts={ctx.local_num_experts}, E={E}"
                )

            def _gather_tuple(tensors, name):
                compact = []
                for tensor in tensors:
                    if tensor.dim() < 2 or tensor.shape[1] != E:
                        raise RuntimeError(
                            f"cannot compact {name}: expected expert dim 1 "
                            f"of size {E}, got shape {tuple(tensor.shape)}"
                        )
                    compact.append(tensor.index_select(1, routed))
                return tuple(compact)

            return replace(
                ctx,
                w13_lora_a_stacked=_gather_tuple(
                    ctx.w13_lora_a_stacked, "w13_lora_a_stacked"
                ),
                w13_lora_b_stacked=_gather_tuple(
                    ctx.w13_lora_b_stacked, "w13_lora_b_stacked"
                ),
                w2_lora_a_stacked=_gather_tuple(
                    ctx.w2_lora_a_stacked, "w2_lora_a_stacked"
                ),
                w2_lora_b_stacked=_gather_tuple(
                    ctx.w2_lora_b_stacked, "w2_lora_b_stacked"
                ),
                local_num_experts=k,
            )

        # --- dequant only the compact set ---
        w1_dq = dequantize_to_dtype(
            tensor_fp4=w1_c, tensor_sf=w1s_c, global_scale=g1_c,
            dtype=hidden_states.dtype, block_size=16, swizzle=False,
        )
        w2_dq = dequantize_to_dtype(
            tensor_fp4=w2_c, tensor_sf=w2s_c, global_scale=g2_c,
            dtype=hidden_states.dtype, block_size=16, swizzle=False,
        )

        # TritonExperts' compact contract is: compact weights, compact topk_ids,
        # global_num_experts=k, expert_map=None. Expert-LoRA tensors are global
        # indexed in vLLM's context, so gather them into this same routed order.
        g2c = topk_ids.new_full((E,), -1)
        g2c[routed] = torch.arange(k, device=topk_ids.device, dtype=topk_ids.dtype)
        topk_ids_c = topk_ids.clone()
        valid_topk = (topk_ids >= 0) & (topk_ids < E)
        topk_ids_c[valid_topk] = g2c[topk_ids[valid_topk]]
        lora_context = getattr(self, "_lora_context", None)
        compact_lora_context = _compact_lora_context(lora_context)

        # activation quant -- identical to the original path
        hs, _ = moe_kernel_quantize_input(
            A=hidden_states, A_scale=self.quant_config.a1_gscale,
            quant_dtype="nvfp4", per_act_token_quant=False, quantization_emulation=True,
        )

        # expert_tokens_meta is full-E; force recompute for the compact set (None).
        if lora_context is not None:
            self._lora_context = compact_lora_context
        try:
            TritonExperts.apply(
                self, output=output, hidden_states=hs, w1=w1_dq, w2=w2_dq,
                topk_weights=topk_weights, topk_ids=topk_ids_c,
                activation=activation, global_num_experts=k, expert_map=None,
                a1q_scale=None, a2_scale=self.quant_config.a2_gscale,
                workspace13=workspace13, workspace2=workspace2,
                expert_tokens_meta=None,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        finally:
            if lora_context is not None:
                self._lora_context = lora_context
    except Exception as e:  # noqa: BLE001 -- never let the optimization break correctness
        sys.stderr.write(f"[routed_dequant] fallback to full dequant: {e!r}\n")
        return _full()


def apply_patch():
    global _orig_apply
    from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
        Nvfp4QuantizationEmulationTritonExperts as C,
    )
    if getattr(C.apply, "_routed_dequant_patched", False):
        return
    _orig_apply = C.apply
    _routed_apply._routed_dequant_patched = True
    C.apply = _routed_apply
    sys.stderr.write(
        f"[routed_dequant pid={os.getpid()}] patched "
        "Nvfp4QuantizationEmulationTritonExperts.apply (routed-only dequant)\n"
    )
