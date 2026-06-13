"""Frozen NVFP4 fused-3D MoE experts.

Drop-in runtime container for model families that expose routed experts as
fused 3D tensors in memory while storing compressed-tensors NVFP4 expert
weights as per-expert safetensors keys on disk.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dequant import dequantize_nvfp4_weight, dequantize_nvfp4_weight_batched


class _DequantExpertLinear(torch.autograd.Function):
    """Linear over one frozen NVFP4 expert weight.

    The dequantized weight is materialized in forward and recomputed in
    backward so the autograd graph never retains a bf16/fp16 shadow weight.
    """

    @staticmethod
    def forward(ctx, x, packed, scale, gscale, group_size: int, fmt: str = "compressed_tensors"):
        ctx.save_for_backward(packed, scale, gscale)
        ctx.group_size = int(group_size)
        ctx.fmt = fmt
        W = dequantize_nvfp4_weight(
            packed,
            scale,
            gscale,
            group_size=ctx.group_size,
            out_dtype=x.dtype,
            format=fmt,
        )
        return F.linear(x, W)

    @staticmethod
    def backward(ctx, grad_output):
        packed, scale, gscale = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            W = dequantize_nvfp4_weight(
                packed,
                scale,
                gscale,
                group_size=ctx.group_size,
                out_dtype=grad_output.dtype,
                format=ctx.fmt,
            )
            grad_x = grad_output @ W
        return grad_x, None, None, None, None, None


class _GroupedDequantExpertLinear(torch.autograd.Function):
    """Batched linear over K frozen NVFP4 expert weights.

    forward: x (K, L, in) x expert weights (K, out, in) -> (K, L, out), where
    the K weights are dequantized in ONE batched call into a transient bf16
    workspace and applied with one torch.bmm. backward recomputes the dequant
    and does grad_x = bmm(grad_out, W); only the activation gradient is
    produced (expert weights are frozen). Same contract as
    _DequantExpertLinear: ctx saves only the packed/scale buffers (views of
    the persistent module buffers, zero extra memory) plus the small expert
    index, so no bf16 expert weight is retained in the graph between forward
    and backward.

    Operation-count argument (GPU benchmarks pending): the legacy loop issues
    1 dequant (itself 8-10 eager dispatches on the fallback path, 1 Triton
    launch otherwise) + 1 GEMM per hit expert per projection, i.e. up to
    2 * 256 = 512 sequential launches per projection per layer, each GEMM
    averaging only ~64 tokens (seq 2048 x top-8 / 256 experts) and badly
    under-occupying the SMs. The grouped path issues 2 launches per K-batch
    (1 batched dequant grid + 1 bmm), i.e. 2 * ceil(256/K) = 64 at K=8 - an
    8x cut in launch count - and each bmm carries K experts' worth of work,
    amortizing kernel launch latency and raising GEMM occupancy. Backward gets
    the identical reduction since dequant is recomputed there. Total dequant
    bytes and matmul FLOPs are unchanged except for padding the per-expert
    token groups to the max group length within each K-batch.

    The transient bf16 workspace is K * out * in * 2 bytes per call (allocated
    here, dead after the bmm); see NVFP4Experts3D.expert_batch_size for the
    sizing policy.
    """

    @staticmethod
    def forward(ctx, x, packed, scale, gscale, expert_idx, group_size: int, fmt: str = "compressed_tensors"):
        ctx.save_for_backward(packed, scale, gscale, expert_idx)
        ctx.group_size = int(group_size)
        ctx.fmt = fmt
        W = dequantize_nvfp4_weight_batched(
            packed,
            scale,
            gscale,
            group_size=ctx.group_size,
            out_dtype=x.dtype,
            format=fmt,
            expert_idx=expert_idx,
        )
        return torch.bmm(x, W.transpose(1, 2))

    @staticmethod
    def backward(ctx, grad_output):
        packed, scale, gscale, expert_idx = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            W = dequantize_nvfp4_weight_batched(
                packed,
                scale,
                gscale,
                group_size=ctx.group_size,
                out_dtype=grad_output.dtype,
                format=ctx.fmt,
                expert_idx=expert_idx,
            )
            grad_x = torch.bmm(grad_output, W)
        return grad_x, None, None, None, None, None, None


class NVFP4Experts3D(nn.Module):
    """Frozen NVFP4 container for fused-3D routed MoE experts.

    Shapes mirror Qwen3.5/Mistral4 fused expert tensors:
      gate_up: (num_experts, 2 * intermediate_dim, hidden_dim)
      down:    (num_experts, hidden_dim, intermediate_dim)

    quant_format selects the dequant scale convention of the source
    checkpoint: "compressed_tensors" (weight_packed / weight_global_scale,
    global scale stored as a divisor) or "modelopt" (weight / weight_scale_2,
    global scale stored as a multiplier). The packed-nibble and fp8
    block-scale layout is identical between the two.

    split_gate_up_scales=False (default) stores gate and up fused along the
    output axis with ONE global scale per expert; this is the validated fast
    path (one dequant + one bmm per K-batch) and requires the checkpoint's
    gate/up global scales to be equal per expert. split_gate_up_scales=True
    stores gate and up as separate 3D buffers with their own global scales,
    for checkpoints whose quantizer scaled the two projections independently;
    the gate_up projection then costs two dequant+bmm pairs per K-batch.
    The trainer probes the checkpoint's scales and picks the mode.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        group_size: int = 16,
        act_fn: nn.Module | callable | None = None,
        device: torch.device | str | None = None,
        quant_format: str = "compressed_tensors",
        split_gate_up_scales: bool = False,
    ):
        super().__init__()
        if quant_format not in ("compressed_tensors", "modelopt"):
            raise ValueError(f"quant_format must be 'compressed_tensors' or 'modelopt', got {quant_format!r}")
        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.group_size = int(group_size)
        self.quant_format = quant_format
        self.split_gate_up_scales = bool(split_gate_up_scales)
        self.act_fn = act_fn if act_fn is not None else nn.SiLU()

        # Grouped expert path: process hit experts in batches of
        # `expert_batch_size` (K), dequantizing K weights per call and running
        # one bmm over the padded per-expert token groups. Set
        # NVFP4_GROUPED_EXPERTS=0 to fall back to the legacy per-expert loop.
        #
        # K sizes the transient bf16 dequant workspace, which exists only
        # inside one _GroupedDequantExpertLinear call:
        #   gate_up: K * (2 * intermediate_dim) * hidden_dim * 2 bytes
        #   down:    K * hidden_dim * intermediate_dim * 2 bytes
        # For Qwen3.5-122B routed experts (hidden 3072, intermediate 1024)
        # that is 12.6 MB / 6.3 MB per expert, so K=8 peaks around 100 MB for
        # gate_up (151 MB if both projections were ever live at once - they
        # are not). NEVER set K anywhere near num_experts: all 256 experts at
        # once is ~4.8 GB per layer for gate_up alone.
        self.expert_batch_size = 8
        self.grouped_experts = os.environ.get("NVFP4_GROUPED_EXPERTS", "1") != "0"

        if self.hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim must be even for uint8 fp4 packing, got {self.hidden_dim}")
        if self.intermediate_dim % 2 != 0:
            raise ValueError(f"intermediate_dim must be even for uint8 fp4 packing, got {self.intermediate_dim}")
        if self.hidden_dim % self.group_size != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by group_size={self.group_size}"
            )
        if self.intermediate_dim % self.group_size != 0:
            raise ValueError(
                f"intermediate_dim={self.intermediate_dim} must be divisible by group_size={self.group_size}"
            )

        if self.split_gate_up_scales:
            # Separate, fully contiguous per-projection buffers: same total
            # bytes as the fused layout, but each half carries its own
            # per-expert global scale.
            for proj in ("gate", "up"):
                self.register_buffer(
                    f"{proj}_packed",
                    torch.zeros(
                        self.num_experts, self.intermediate_dim, self.hidden_dim // 2,
                        dtype=torch.uint8, device=device,
                    ),
                )
                self.register_buffer(
                    f"{proj}_scale",
                    torch.ones(
                        self.num_experts,
                        self.intermediate_dim,
                        self.hidden_dim // self.group_size,
                        dtype=torch.float8_e4m3fn,
                        device=device,
                    ),
                )
                self.register_buffer(
                    f"{proj}_global_scale",
                    torch.ones(self.num_experts, 1, dtype=torch.float32, device=device),
                )
        else:
            gate_up_out = 2 * self.intermediate_dim
            self.register_buffer(
                "gate_up_packed",
                torch.zeros(self.num_experts, gate_up_out, self.hidden_dim // 2, dtype=torch.uint8, device=device),
            )
            self.register_buffer(
                "gate_up_scale",
                torch.ones(
                    self.num_experts,
                    gate_up_out,
                    self.hidden_dim // self.group_size,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
            )
            self.register_buffer(
                "gate_up_global_scale",
                torch.ones(self.num_experts, 1, dtype=torch.float32, device=device),
            )

        self.register_buffer(
            "down_packed",
            torch.zeros(
                self.num_experts,
                self.hidden_dim,
                self.intermediate_dim // 2,
                dtype=torch.uint8,
                device=device,
            ),
        )
        self.register_buffer(
            "down_scale",
            torch.ones(
                self.num_experts,
                self.hidden_dim,
                self.intermediate_dim // self.group_size,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
        )
        self.register_buffer(
            "down_global_scale",
            torch.ones(self.num_experts, 1, dtype=torch.float32, device=device),
        )

    def _gate_up_acted(self, x: torch.Tensor, expert_idx: torch.Tensor | int) -> torch.Tensor:
        """act(gate(x)) * up(x) for one expert, honoring the storage mode."""
        i = int(expert_idx)
        if self.split_gate_up_scales:
            gate = _DequantExpertLinear.apply(
                x,
                self.gate_packed[i].contiguous(),
                self.gate_scale[i].contiguous(),
                self.gate_global_scale[i],
                self.group_size,
                self.quant_format,
            )
            up = _DequantExpertLinear.apply(
                x,
                self.up_packed[i].contiguous(),
                self.up_scale[i].contiguous(),
                self.up_global_scale[i],
                self.group_size,
                self.quant_format,
            )
        else:
            gate, up = _DequantExpertLinear.apply(
                x,
                self.gate_up_packed[i].contiguous(),
                self.gate_up_scale[i].contiguous(),
                self.gate_up_global_scale[i],
                self.group_size,
                self.quant_format,
            ).chunk(2, dim=-1)
        return self.act_fn(gate) * up

    def _down_linear(self, x: torch.Tensor, expert_idx: torch.Tensor | int) -> torch.Tensor:
        i = int(expert_idx)
        return _DequantExpertLinear.apply(
            x,
            self.down_packed[i].contiguous(),
            self.down_scale[i].contiguous(),
            self.down_global_scale[i],
            self.group_size,
            self.quant_format,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.grouped_experts:
            return self._forward_grouped(hidden_states, top_k_index, top_k_weights)
        return self._forward_per_expert(hidden_states, top_k_index, top_k_weights)

    def _forward_per_expert(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Legacy sequential per-expert loop (NVFP4_GROUPED_EXPERTS=0)."""
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self._gate_up_acted(current_state, expert_idx)
            current_hidden_states = self._down_linear(current_hidden_states, expert_idx)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states

    def _forward_grouped(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Grouped expert MLP: sort-trick token regrouping + K-expert batched GEMMs.

        Routing math is identical to the per-expert loop: the same
        (token, top_k_pos) pairs hit the same experts with the same
        `top_k_weights[token, pos]` multipliers; only the order in which
        contributions accumulate into `final_hidden_states` differs (sorted by
        expert, token-major within an expert, vs the loop's top_k-pos-major),
        which matters only at bf16 rounding level.
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        device = hidden_states.device
        top_k = top_k_index.shape[1]

        with torch.no_grad():
            # Sort the flattened (token, top_k_pos) assignments by expert so
            # each expert's tokens form one contiguous segment.
            flat_expert = top_k_index.reshape(-1)
            sort_order = torch.argsort(flat_expert, stable=True)
            token_idx_sorted = torch.div(sort_order, top_k, rounding_mode="floor")
            counts = torch.bincount(flat_expert, minlength=self.num_experts)
            # Segment bookkeeping happens on CPU (python loop bounds); one sync.
            counts_cpu = counts.to("cpu")
            offsets_cpu = torch.cumsum(counts_cpu, dim=0) - counts_cpu
            hit_experts_cpu = counts_cpu.nonzero(as_tuple=False).flatten()

        if hit_experts_cpu.numel() == 0:
            return final_hidden_states

        # One gather up front; one index_add_ scatter at the end.
        x_sorted = hidden_states[token_idx_sorted]
        w_sorted = top_k_weights.reshape(-1)[sort_order]

        batch_size = max(1, int(self.expert_batch_size))
        out_chunks = []
        for start in range(0, hit_experts_cpu.numel(), batch_size):
            batch_cpu = hit_experts_cpu[start : start + batch_size]
            k = batch_cpu.numel()
            seg_len = int(counts_cpu[batch_cpu].max())

            # Pad each expert's token segment to the longest in this batch so
            # the K groups stack into one (k, seg_len, hidden) bmm operand.
            # Padded lanes alias row 0 of x_sorted; their outputs are dropped
            # by the `valid` mask below, so no gradient flows through them.
            seg_pos = torch.arange(seg_len, device=device)
            seg_counts = counts_cpu[batch_cpu].to(device)
            seg_offsets = offsets_cpu[batch_cpu].to(device)
            valid = seg_pos[None, :] < seg_counts[:, None]
            gather_idx = seg_offsets[:, None] + seg_pos[None, :]
            gather_idx = torch.where(valid, gather_idx, torch.zeros_like(gather_idx))
            current_state = x_sorted[gather_idx.reshape(-1)].view(k, seg_len, self.hidden_dim)

            expert_idx = batch_cpu.to(device)
            if self.split_gate_up_scales:
                gate = _GroupedDequantExpertLinear.apply(
                    current_state,
                    self.gate_packed,
                    self.gate_scale,
                    self.gate_global_scale,
                    expert_idx,
                    self.group_size,
                    self.quant_format,
                )
                up = _GroupedDequantExpertLinear.apply(
                    current_state,
                    self.up_packed,
                    self.up_scale,
                    self.up_global_scale,
                    expert_idx,
                    self.group_size,
                    self.quant_format,
                )
            else:
                gate, up = _GroupedDequantExpertLinear.apply(
                    current_state,
                    self.gate_up_packed,
                    self.gate_up_scale,
                    self.gate_up_global_scale,
                    expert_idx,
                    self.group_size,
                    self.quant_format,
                ).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = _GroupedDequantExpertLinear.apply(
                current_hidden_states,
                self.down_packed,
                self.down_scale,
                self.down_global_scale,
                expert_idx,
                self.group_size,
                self.quant_format,
            )
            out_chunks.append(
                current_hidden_states.reshape(k * seg_len, self.hidden_dim)[valid.reshape(-1)]
            )

        # Valid rows concatenate back into exact sorted order: hit experts are
        # visited ascending and rows within a segment keep their sorted order.
        out_sorted = torch.cat(out_chunks, dim=0) if len(out_chunks) > 1 else out_chunks[0]
        out_sorted = out_sorted * w_sorted[:, None]
        final_hidden_states.index_add_(0, token_idx_sorted, out_sorted.to(final_hidden_states.dtype))
        return final_hidden_states


def _load_safetensor_key(model_dir: Path, weight_map: Mapping[str, str], key: str) -> torch.Tensor:
    """Load a single tensor via the lazy `safe_open` API (NOT `load_file` which reads
    the whole shard). Critical for full-model assembly where we touch hundreds of
    thousands of small per-expert keys from large shards.
    """
    try:
        shard_name = weight_map[key]
    except KeyError as exc:
        raise KeyError(f"missing safetensors key {key!r}") from exc

    import safetensors
    shard_path = model_dir / shard_name
    with safetensors.safe_open(str(shard_path), framework="pt", device="cpu") as f:
        try:
            return f.get_tensor(key)
        except Exception as exc:
            raise KeyError(f"key {key!r} not found in shard {shard_path}") from exc


def _expert_key_suffixes(quant_format: str) -> tuple[str, str, str]:
    """(packed, block-scale, global-scale) key suffixes for a storage format."""
    if quant_format == "modelopt":
        return "weight", "weight_scale", "weight_scale_2"
    return "weight_packed", "weight_scale", "weight_global_scale"


def _stuff_expert_buffers(
    module: NVFP4Experts3D,
    expert_idx: int,
    base: str,
    gate_packed, gate_scale, gate_gscale,
    up_packed, up_scale, up_gscale,
    down_packed, down_scale, down_gscale,
) -> None:
    """Copy one expert's three projections into the module buffers, honoring
    the fused vs split gate/up storage mode."""
    if module.split_gate_up_scales:
        for proj, packed, scale, gscale in (
            ("gate", gate_packed, gate_scale, gate_gscale),
            ("up", up_packed, up_scale, up_gscale),
        ):
            getattr(module, f"{proj}_packed")[expert_idx].copy_(
                packed.to(device=getattr(module, f"{proj}_packed").device, dtype=torch.uint8)
            )
            getattr(module, f"{proj}_scale")[expert_idx].copy_(
                scale.to(device=getattr(module, f"{proj}_scale").device, dtype=torch.float8_e4m3fn)
            )
            getattr(module, f"{proj}_global_scale")[expert_idx].copy_(
                gscale.reshape(1).to(
                    device=getattr(module, f"{proj}_global_scale").device, dtype=torch.float32
                )
            )
    else:
        module.gate_up_packed[expert_idx, : module.intermediate_dim].copy_(
            gate_packed.to(device=module.gate_up_packed.device, dtype=torch.uint8)
        )
        module.gate_up_packed[expert_idx, module.intermediate_dim :].copy_(
            up_packed.to(device=module.gate_up_packed.device, dtype=torch.uint8)
        )
        module.gate_up_scale[expert_idx, : module.intermediate_dim].copy_(
            gate_scale.to(device=module.gate_up_scale.device, dtype=torch.float8_e4m3fn)
        )
        module.gate_up_scale[expert_idx, module.intermediate_dim :].copy_(
            up_scale.to(device=module.gate_up_scale.device, dtype=torch.float8_e4m3fn)
        )
        if not torch.equal(gate_gscale.reshape(1), up_gscale.reshape(1)):
            raise ValueError(
                f"{base}.gate_proj and up_proj per-tensor scales differ but the module "
                "was built with fused gate_up storage (one global scale per expert). "
                "Rebuild with split_gate_up_scales=True; the unified trainer probes "
                "the checkpoint and selects the mode automatically."
            )
        module.gate_up_global_scale[expert_idx].copy_(
            gate_gscale.reshape(1).to(device=module.gate_up_global_scale.device, dtype=torch.float32)
        )
    module.down_packed[expert_idx].copy_(
        down_packed.to(device=module.down_packed.device, dtype=torch.uint8)
    )
    module.down_scale[expert_idx].copy_(
        down_scale.to(device=module.down_scale.device, dtype=torch.float8_e4m3fn)
    )
    module.down_global_scale[expert_idx].copy_(
        down_gscale.reshape(1).to(device=module.down_global_scale.device, dtype=torch.float32)
    )


def assemble_nvfp4_experts3d_batched(
    module: NVFP4Experts3D,
    st_prefix: str,
    model_dir: Path,
    weight_map: Mapping[str, str],
) -> None:
    """Batched assembly: group keys by shard, open each shard once, extract all
    needed per-expert tensors. Critical for full-model load — naive per-key
    opens scale as O(num_experts × 3 × 3 × shard_open_cost).

    Key naming follows module.quant_format (compressed-tensors or ModelOpt).
    """
    import safetensors
    model_dir = Path(model_dir)
    w_key, s_key, g_key = _expert_key_suffixes(module.quant_format)

    # Plan all keys for this MoE block
    needed: list[str] = []
    for expert_idx in range(module.num_experts):
        base = f"{st_prefix}.{expert_idx}"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix in (w_key, s_key, g_key):
                needed.append(f"{base}.{proj}.{suffix}")

    # Group by shard
    by_shard: dict[str, list[str]] = {}
    for key in needed:
        if key not in weight_map:
            raise KeyError(f"missing safetensors key {key!r}")
        by_shard.setdefault(weight_map[key], []).append(key)

    # Open each shard once, extract
    tensors: dict[str, torch.Tensor] = {}
    for shard_name, keys in by_shard.items():
        shard_path = model_dir / shard_name
        with safetensors.safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for k in keys:
                tensors[k] = f.get_tensor(k)

    # Now stuff buffers
    with torch.no_grad():
        for expert_idx in range(module.num_experts):
            base = f"{st_prefix}.{expert_idx}"
            _stuff_expert_buffers(
                module, expert_idx, base,
                tensors[f"{base}.gate_proj.{w_key}"],
                tensors[f"{base}.gate_proj.{s_key}"],
                tensors[f"{base}.gate_proj.{g_key}"],
                tensors[f"{base}.up_proj.{w_key}"],
                tensors[f"{base}.up_proj.{s_key}"],
                tensors[f"{base}.up_proj.{g_key}"],
                tensors[f"{base}.down_proj.{w_key}"],
                tensors[f"{base}.down_proj.{s_key}"],
                tensors[f"{base}.down_proj.{g_key}"],
            )


def detect_moe_expert_storage(model_dir: Path, weight_map: Mapping[str, str]) -> dict | None:
    """Probe how a checkpoint stores its per-expert routed-MoE projections.

    Returns None when no per-expert expert keys exist, else
      {"quant_format": "compressed_tensors" | "modelopt",
       "split_gate_up_scales": bool}
    where split_gate_up_scales is True iff ANY expert's gate/up per-tensor
    scales differ (read from the shards; grouped by shard, ~1 s even for
    10k+ experts on a 100B-class checkpoint).
    """
    import re as _re

    import safetensors

    expert_re = _re.compile(r"^(?P<base>.+\.experts\.\d+\.gate_proj)\.(?P<suffix>weight_packed|weight_scale_2)$")
    fmt = None
    for key in weight_map:
        m = expert_re.match(key)
        if m:
            fmt = "compressed_tensors" if m.group("suffix") == "weight_packed" else "modelopt"
            break
    if fmt is None:
        # ModelOpt expert weights are bare `.weight`; presence of the fp32
        # per-tensor scale (matched above) is the discriminator, so reaching
        # here means there are no NVFP4 per-expert projections at all.
        return None
    _, _, g_key = _expert_key_suffixes(fmt)

    pairs: list[tuple[str, str]] = []
    for key in weight_map:
        if key.endswith(f".gate_proj.{g_key}") and ".experts." in key:
            up = key.replace(f".gate_proj.{g_key}", f".up_proj.{g_key}")
            if up in weight_map:
                pairs.append((key, up))
    by_shard: dict[str, list[str]] = {}
    for g, u in pairs:
        by_shard.setdefault(weight_map[g], []).append(g)
        by_shard.setdefault(weight_map[u], []).append(u)
    scales: dict[str, float] = {}
    model_dir = Path(model_dir)
    for shard, keys in by_shard.items():
        with safetensors.safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
            for k in keys:
                scales[k] = float(f.get_tensor(k).reshape(-1)[0])
    split = any(scales[g] != scales[u] for g, u in pairs)
    return {"quant_format": fmt, "split_gate_up_scales": split}


def replace_moe_experts_with_nvfp4_3d(
    model: "torch.nn.Module",
    model_family: str,
    device: "torch.device | str | None" = None,
    quant_format: str = "compressed_tensors",
    split_gate_up_scales: bool = False,
) -> int:
    """Walk the model and replace each fused-3D MoE experts block with NVFP4Experts3D.

    Returns the number of blocks replaced.
    """
    import torch.nn as nn

    from .families import FAMILIES

    fam = FAMILIES.get(model_family)
    target_cls_name = fam.get("moe_experts_class") if fam is not None else None
    if target_cls_name is None:
        raise RuntimeError(
            f"replace_moe_experts_with_nvfp4_3d does not have a fused-3D MoE class for "
            f"model_family={model_family!r}. Add moe_experts_class to the family's "
            f"entry in nvfp4_lora/families.py (see docs/SUPPORTED_TOPOLOGIES.md)."
        )

    # Activation: probe the model config for hidden_act, fall back to SiLU which is
    # the standard for both Qwen3.5 and Mistral4
    from transformers.activations import ACT2FN
    cfg = getattr(model, "config", None)
    # text_config carries hidden_act when wrapped via VLM
    text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
    hidden_act = None
    for c in (cfg, text_cfg):
        if c is not None and hasattr(c, "hidden_act"):
            hidden_act = c.hidden_act
            break
    act_fn = ACT2FN[hidden_act] if hidden_act in ACT2FN else nn.SiLU()

    replaced = 0
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ != target_cls_name:
            continue
        # Old module exposes num_experts/hidden_dim/intermediate_dim
        num_experts = int(module.num_experts)
        hidden_dim = int(module.hidden_dim)
        intermediate_dim = int(module.intermediate_dim)
        new_module = NVFP4Experts3D(
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            group_size=16,
            act_fn=act_fn() if isinstance(act_fn, type) else act_fn,
            device=device,
            quant_format=quant_format,
            split_gate_up_scales=split_gate_up_scales,
        )
        # Place new module in the model tree
        parent_name, _, child_attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_attr, new_module)
        replaced += 1
    return replaced


def assemble_nvfp4_experts3d_from_safetensors_keys(
    module: NVFP4Experts3D,
    st_prefix: str,
    model_dir: Path,
    weight_map: Mapping[str, str],
) -> None:
    """Assemble per-expert NVFP4 safetensors keys into 3D buffers.

    Expected keys, per expert and per projection (gate_proj, up_proj,
    down_proj), with suffixes from module.quant_format:
      compressed-tensors: weight_packed / weight_scale / weight_global_scale
      ModelOpt:           weight        / weight_scale / weight_scale_2

    gate/up are stacked along output axis as [gate, up], matching the
    reference MoE forward's `.chunk(2, dim=-1)` ordering (fused mode), or
    stored separately when module.split_gate_up_scales is set.
    """
    model_dir = Path(model_dir)
    w_key, s_key, g_key = _expert_key_suffixes(module.quant_format)

    with torch.no_grad():
        for expert_idx in range(module.num_experts):
            base = f"{st_prefix}.{expert_idx}"

            gate_packed = _load_safetensor_key(model_dir, weight_map, f"{base}.gate_proj.{w_key}")
            gate_scale = _load_safetensor_key(model_dir, weight_map, f"{base}.gate_proj.{s_key}")
            gate_gscale = _load_safetensor_key(model_dir, weight_map, f"{base}.gate_proj.{g_key}")

            up_packed = _load_safetensor_key(model_dir, weight_map, f"{base}.up_proj.{w_key}")
            up_scale = _load_safetensor_key(model_dir, weight_map, f"{base}.up_proj.{s_key}")
            up_gscale = _load_safetensor_key(model_dir, weight_map, f"{base}.up_proj.{g_key}")

            down_packed = _load_safetensor_key(model_dir, weight_map, f"{base}.down_proj.{w_key}")
            down_scale = _load_safetensor_key(model_dir, weight_map, f"{base}.down_proj.{s_key}")
            down_gscale = _load_safetensor_key(model_dir, weight_map, f"{base}.down_proj.{g_key}")

            expected_gate_up_piece = (module.intermediate_dim, module.hidden_dim // 2)
            expected_gate_up_scale = (module.intermediate_dim, module.hidden_dim // module.group_size)
            expected_down = (module.hidden_dim, module.intermediate_dim // 2)
            expected_down_scale = (module.hidden_dim, module.intermediate_dim // module.group_size)

            if tuple(gate_packed.shape) != expected_gate_up_piece:
                raise ValueError(
                    f"{base}.gate_proj.{w_key} shape {tuple(gate_packed.shape)} "
                    f"!= {expected_gate_up_piece}"
                )
            if tuple(up_packed.shape) != expected_gate_up_piece:
                raise ValueError(
                    f"{base}.up_proj.{w_key} shape {tuple(up_packed.shape)} "
                    f"!= {expected_gate_up_piece}"
                )
            if tuple(gate_scale.shape) != expected_gate_up_scale:
                raise ValueError(
                    f"{base}.gate_proj.{s_key} shape {tuple(gate_scale.shape)} "
                    f"!= {expected_gate_up_scale}"
                )
            if tuple(up_scale.shape) != expected_gate_up_scale:
                raise ValueError(
                    f"{base}.up_proj.{s_key} shape {tuple(up_scale.shape)} "
                    f"!= {expected_gate_up_scale}"
                )
            if tuple(down_packed.shape) != expected_down:
                raise ValueError(
                    f"{base}.down_proj.{w_key} shape {tuple(down_packed.shape)} != {expected_down}"
                )
            if tuple(down_scale.shape) != expected_down_scale:
                raise ValueError(
                    f"{base}.down_proj.{s_key} shape {tuple(down_scale.shape)} != {expected_down_scale}"
                )

            _stuff_expert_buffers(
                module, expert_idx, base,
                gate_packed, gate_scale, gate_gscale,
                up_packed, up_scale, up_gscale,
                down_packed, down_scale, down_gscale,
            )
