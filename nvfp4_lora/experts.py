"""Frozen NVFP4 fused-3D MoE experts.

Drop-in runtime container for model families that expose routed experts as
fused 3D tensors in memory while storing compressed-tensors NVFP4 expert
weights as per-expert safetensors keys on disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dequant import dequantize_nvfp4_weight


class _DequantExpertLinear(torch.autograd.Function):
    """Linear over one frozen NVFP4 expert weight.

    The dequantized weight is materialized in forward and recomputed in
    backward so the autograd graph never retains a bf16/fp16 shadow weight.
    """

    @staticmethod
    def forward(ctx, x, packed, scale, gscale, group_size: int):
        ctx.save_for_backward(packed, scale, gscale)
        ctx.group_size = int(group_size)
        W = dequantize_nvfp4_weight(
            packed,
            scale,
            gscale,
            group_size=ctx.group_size,
            out_dtype=x.dtype,
            format="compressed_tensors",
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
                format="compressed_tensors",
            )
            grad_x = grad_output @ W
        return grad_x, None, None, None, None


class NVFP4Experts3D(nn.Module):
    """Frozen NVFP4 container for fused-3D routed MoE experts.

    Shapes mirror Qwen3.5/Mistral4 fused expert tensors:
      gate_up: (num_experts, 2 * intermediate_dim, hidden_dim)
      down:    (num_experts, hidden_dim, intermediate_dim)
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        group_size: int = 16,
        act_fn: nn.Module | callable | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.group_size = int(group_size)
        self.act_fn = act_fn if act_fn is not None else nn.SiLU()

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

    def _gate_up_linear(self, x: torch.Tensor, expert_idx: torch.Tensor | int) -> torch.Tensor:
        i = int(expert_idx)
        return _DequantExpertLinear.apply(
            x,
            self.gate_up_packed[i].contiguous(),
            self.gate_up_scale[i].contiguous(),
            self.gate_up_global_scale[i],
            self.group_size,
        )

    def _down_linear(self, x: torch.Tensor, expert_idx: torch.Tensor | int) -> torch.Tensor:
        i = int(expert_idx)
        return _DequantExpertLinear.apply(
            x,
            self.down_packed[i].contiguous(),
            self.down_scale[i].contiguous(),
            self.down_global_scale[i],
            self.group_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
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
            gate, up = self._gate_up_linear(current_state, expert_idx).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = self._down_linear(current_hidden_states, expert_idx)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
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


def assemble_nvfp4_experts3d_batched(
    module: NVFP4Experts3D,
    st_prefix: str,
    model_dir: Path,
    weight_map: Mapping[str, str],
) -> None:
    """Batched assembly: group keys by shard, open each shard once, extract all
    needed per-expert tensors. Critical for full-model load — naive per-key
    opens scale as O(num_experts × 3 × 3 × shard_open_cost).
    """
    import safetensors
    model_dir = Path(model_dir)

    # Plan all keys for this MoE block
    needed: list[str] = []
    for expert_idx in range(module.num_experts):
        base = f"{st_prefix}.{expert_idx}"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight_packed", "weight_scale", "weight_global_scale"):
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
            gate_packed = tensors[f"{base}.gate_proj.weight_packed"]
            gate_scale = tensors[f"{base}.gate_proj.weight_scale"]
            gate_gscale = tensors[f"{base}.gate_proj.weight_global_scale"]
            up_packed = tensors[f"{base}.up_proj.weight_packed"]
            up_scale = tensors[f"{base}.up_proj.weight_scale"]
            up_gscale = tensors[f"{base}.up_proj.weight_global_scale"]
            down_packed = tensors[f"{base}.down_proj.weight_packed"]
            down_scale = tensors[f"{base}.down_proj.weight_scale"]
            down_gscale = tensors[f"{base}.down_proj.weight_global_scale"]

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
                    f"{base}.gate_proj.weight_global_scale and up_proj.weight_global_scale differ; "
                    "fused gate_up storage has one global scale per expert"
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


def replace_moe_experts_with_nvfp4_3d(
    model: "torch.nn.Module",
    model_family: str,
    device: "torch.device | str | None" = None,
) -> int:
    """Walk the model and replace each fused-3D MoE experts block with NVFP4Experts3D.

    Returns the number of blocks replaced.
    """
    import torch.nn as nn

    family_class_names = {
        "qwen3_5_moe": "Qwen3_5MoeExperts",
        "qwen3_5_moe_text": "Qwen3_5MoeExperts",
        "mistral3": "Mistral4NaiveMoe",
        "mistral4": "Mistral4NaiveMoe",
    }
    target_cls_name = family_class_names.get(model_family)
    if target_cls_name is None:
        raise RuntimeError(
            f"replace_moe_experts_with_nvfp4_3d does not have a fused-3D MoE class for "
            f"model_family={model_family!r}. Add a mapping here."
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
    """Assemble per-expert compressed-tensors safetensors keys into 3D buffers.

    Expected keys:
      {st_prefix}.{expert}.gate_proj.weight_packed
      {st_prefix}.{expert}.gate_proj.weight_scale
      {st_prefix}.{expert}.gate_proj.weight_global_scale
      {st_prefix}.{expert}.up_proj.weight_packed
      {st_prefix}.{expert}.up_proj.weight_scale
      {st_prefix}.{expert}.up_proj.weight_global_scale
      {st_prefix}.{expert}.down_proj.weight_packed
      {st_prefix}.{expert}.down_proj.weight_scale
      {st_prefix}.{expert}.down_proj.weight_global_scale

    gate/up are stacked along output axis as [gate, up], matching the
    reference MoE forward's `.chunk(2, dim=-1)` ordering.
    """
    model_dir = Path(model_dir)

    with torch.no_grad():
        for expert_idx in range(module.num_experts):
            base = f"{st_prefix}.{expert_idx}"

            gate_packed = _load_safetensor_key(model_dir, weight_map, f"{base}.gate_proj.weight_packed")
            gate_scale = _load_safetensor_key(model_dir, weight_map, f"{base}.gate_proj.weight_scale")
            gate_gscale = _load_safetensor_key(model_dir, weight_map, f"{base}.gate_proj.weight_global_scale")

            up_packed = _load_safetensor_key(model_dir, weight_map, f"{base}.up_proj.weight_packed")
            up_scale = _load_safetensor_key(model_dir, weight_map, f"{base}.up_proj.weight_scale")
            up_gscale = _load_safetensor_key(model_dir, weight_map, f"{base}.up_proj.weight_global_scale")

            down_packed = _load_safetensor_key(model_dir, weight_map, f"{base}.down_proj.weight_packed")
            down_scale = _load_safetensor_key(model_dir, weight_map, f"{base}.down_proj.weight_scale")
            down_gscale = _load_safetensor_key(model_dir, weight_map, f"{base}.down_proj.weight_global_scale")

            expected_gate_up_piece = (module.intermediate_dim, module.hidden_dim // 2)
            expected_gate_up_scale = (module.intermediate_dim, module.hidden_dim // module.group_size)
            expected_down = (module.hidden_dim, module.intermediate_dim // 2)
            expected_down_scale = (module.hidden_dim, module.intermediate_dim // module.group_size)

            if tuple(gate_packed.shape) != expected_gate_up_piece:
                raise ValueError(
                    f"{base}.gate_proj.weight_packed shape {tuple(gate_packed.shape)} "
                    f"!= {expected_gate_up_piece}"
                )
            if tuple(up_packed.shape) != expected_gate_up_piece:
                raise ValueError(
                    f"{base}.up_proj.weight_packed shape {tuple(up_packed.shape)} "
                    f"!= {expected_gate_up_piece}"
                )
            if tuple(gate_scale.shape) != expected_gate_up_scale:
                raise ValueError(
                    f"{base}.gate_proj.weight_scale shape {tuple(gate_scale.shape)} "
                    f"!= {expected_gate_up_scale}"
                )
            if tuple(up_scale.shape) != expected_gate_up_scale:
                raise ValueError(
                    f"{base}.up_proj.weight_scale shape {tuple(up_scale.shape)} "
                    f"!= {expected_gate_up_scale}"
                )
            if tuple(down_packed.shape) != expected_down:
                raise ValueError(
                    f"{base}.down_proj.weight_packed shape {tuple(down_packed.shape)} != {expected_down}"
                )
            if tuple(down_scale.shape) != expected_down_scale:
                raise ValueError(
                    f"{base}.down_proj.weight_scale shape {tuple(down_scale.shape)} != {expected_down_scale}"
                )

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
                    f"{base}.gate_proj.weight_global_scale and up_proj.weight_global_scale differ; "
                    "fused gate_up storage has one global scale per expert"
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
