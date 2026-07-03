#!/usr/bin/env python3
"""Quantize Mistral-Small-4-119B-2603-BF16 → NVFP4 (compressed-tensors HF layout).

Why a custom script (not llmcompressor):
  llmcompressor 0.11.0 hard-pins transformers<=4.57.6, but Mistral-Small-4 uses
  `mistral4` text-backbone model_type which requires transformers>=5.x. They cannot
  coexist. compressed-tensors 0.15.0.1 (already in the qwen-serve venv) ships all
  the NVFP4 primitives we need; this script uses them directly.

Output layout matches Qwen3.5-122B-A10B-NVFP4 (CT NVFP4) so the existing loader
(nvfp4_lora/loader.py + nvfp4_lora/experts.py) reads it natively:
  - {prefix}.weight_packed        uint8,  (out, in//2)
  - {prefix}.weight_scale         fp8_e4m3fn, (out, in//GROUP_SIZE)
  - {prefix}.weight_global_scale  fp32,    (1,)

Fused 3D MoE source tensors are split per expert per projection so each Linear-like
slice becomes a separate trio of keys:
  source `experts.gate_up_proj` shape (E, 2*I, H) →
      `experts.{e}.gate_proj.{weight_packed,weight_scale,weight_global_scale}` for e in range(E)
      `experts.{e}.up_proj.{...}`
  source `experts.down_proj` shape (E, H, I) →
      `experts.{e}.down_proj.{...}`

Ignored (preserved as bf16 in the output): lm_head, embed_tokens, all norms,
the MoE router gate, and the vision tower / multi-modal projector.

Run (from the repo root):
    /home/veritan-spark-01/Veritan/.venvs/qwen-serve/bin/python \\
        scripts/quantize_mistral_to_nvfp4.py
"""
from __future__ import annotations

import gc
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# The quantization math now lives in the nvfp4_lora package; this script is a
# thin CLI wrapper over it (behaviour and output layout unchanged).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nvfp4_lora.quantize import GROUP_SIZE, quantize_nvfp4_2d

SOURCE_DIR = Path("/home/veritan-spark-01/Veritan/Models/Mistral-Small-4-119B-2603-BF16-HF")
OUTPUT_DIR = Path("/home/veritan-spark-01/Veritan/Models/Mistral-Small-4-119B-2603-NVFP4-HF")

# Aim for ~5 GB output shards (HF convention; large enough to keep file count manageable).
MAX_SHARD_BYTES = 5 * 1024 ** 3

# Compute is forced to CUDA when available (GB10 is ~50x faster than CPU for the
# LUT-search argmin). The package returns CPU tensors regardless.
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quantize_to_nvfp4_2d(
    W: torch.Tensor,
    per_tensor_max_override: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight (out, in) -> NVFP4 CT trio.

    Thin wrapper over nvfp4_lora.quantize.quantize_nvfp4_2d(layout="ct"); the
    trio return shape is kept for callers here and in the merge scripts.

    `per_tensor_max_override` lets the caller supply an externally-computed
    per-tensor abs-max (used for fused gate_up_proj where gate and up must share a
    single global scale, so the loader's NVFP4Experts3D equality assert holds).

    Returns:
      weight_packed:        uint8, shape (out, in/2), on CPU
      weight_scale_fp8:     float8_e4m3fn, shape (out, in/GROUP_SIZE), on CPU
      weight_global_scale:  float32, shape (1,), on CPU
    """
    d = quantize_nvfp4_2d(
        W.to(_DEVICE, non_blocking=True),
        layout="ct",
        per_tensor_max_override=per_tensor_max_override,
    )
    return d["weight_packed"], d["weight_scale"], d["weight_global_scale"]


def quantize_to_nvfp4_3d_per_slice(
    W: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Quantize each 2D slice of a 3D weight (E, out, in) independently.

    Returns a list of (packed, scale_fp8, global_scale) trios, one per slice.
    Caller is responsible for placing them under per-slice keys.
    """
    if W.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {tuple(W.shape)}")
    return [quantize_to_nvfp4_2d(W[e].contiguous()) for e in range(W.shape[0])]


def is_norm_or_gate_or_embed(key: str) -> bool:
    """Identify tensors that MUST be preserved as bf16, not quantized."""
    parts = key.split(".")
    last = parts[-1]
    # 1D weights are norms/scales — preserve as-is. Also lm_head/embed_tokens, gates, biases.
    bad_suffix = (
        "embed_tokens.weight", "lm_head.weight",
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "q_a_layernorm.weight", "kv_a_layernorm.weight",
        "norm.weight",
        "mlp.gate.weight",  # MoE router gate
    )
    if any(key.endswith(s) for s in bad_suffix):
        return True
    # Vision branch — skip quantization (we don't train it)
    if "vision_tower" in key or "multi_modal_projector" in key:
        return True
    return False


def is_fused_3d_moe_weight(key: str) -> str | None:
    """Recognise the two fused-MoE source keys; return projection-pair tag."""
    if key.endswith("mlp.experts.gate_up_proj"):
        return "gate_up_proj"
    if key.endswith("mlp.experts.down_proj"):
        return "down_proj"
    return None


def split_gate_up_proj(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split fused gate_up_proj (E, 2*I, H) into (gate (E, I, H), up (E, I, H)).

    Mistral4NaiveMoe / Qwen3_5MoeExperts both stack rows as [gate; up] along dim 1,
    matching the canonical fused FFN convention.
    """
    if W.ndim != 3:
        raise ValueError(f"Expected 3D fused gate_up_proj, got {tuple(W.shape)}")
    E, two_I, H = W.shape
    if two_I % 2 != 0:
        raise ValueError(f"gate_up_proj dim 1 not even: {two_I}")
    I = two_I // 2
    gate = W[:, :I, :].contiguous()
    up = W[:, I:, :].contiguous()
    return gate, up


def _flush_shard(
    buffer: dict[str, torch.Tensor],
    out_dir: Path,
    shard_idx: int,
    weight_map: dict[str, str],
    shard_template: str,
) -> None:
    """Write the current buffer to a shard file and update weight_map."""
    if not buffer:
        return
    shard_name = shard_template.format(idx=shard_idx)
    out_path = out_dir / shard_name
    save_file(buffer, str(out_path), metadata={"format": "pt"})
    for k in buffer:
        weight_map[k] = shard_name
    print(f"  wrote {shard_name}: {len(buffer)} keys, {sum(t.numel() * t.element_size() for t in buffer.values()) / 1e9:.2f} GB")


def make_quantization_config(ignore_globs: list[str]) -> dict:
    """Build the compressed-tensors quantization_config block for config.json."""
    return {
        "config_groups": {
            "group_0": {
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": 16,
                    "num_bits": 4,
                    "observer": "memoryless_minmax",
                    "observer_kwargs": {},
                    "scale_dtype": "float8_e4m3fn",
                    "strategy": "tensor_group",
                    "symmetric": True,
                    "type": "float",
                    "zp_dtype": "float8_e4m3fn",
                },
            }
        },
        "format": "nvfp4-pack-quantized",
        "global_compression_ratio": None,
        "ignore": ignore_globs,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {},
        "version": "0.15.0",
    }


def main() -> None:
    assert SOURCE_DIR.exists(), f"Source BF16 dir not found: {SOURCE_DIR}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Mistral BF16 → NVFP4 (CT) quantization ===")
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    # All BF16 shards present?
    expected_shards = [SOURCE_DIR / f"model-{i:05d}-of-00003.safetensors" for i in range(1, 4)]
    missing = [s for s in expected_shards if not s.exists()]
    if missing:
        raise SystemExit(f"Missing BF16 shards: {missing}. Wait for download to complete.")

    idx = json.loads((SOURCE_DIR / "model.safetensors.index.json").read_text())
    src_weight_map: dict[str, str] = idx["weight_map"]

    # Group source keys by source shard
    shards_to_keys: dict[str, list[str]] = {}
    for k, sh in src_weight_map.items():
        shards_to_keys.setdefault(sh, []).append(k)

    weight_map_out: dict[str, str] = {}
    out_shard_template = "model-{idx:05d}.safetensors"
    out_shard_idx = 1
    current_buffer: dict[str, torch.Tensor] = {}
    current_size = 0

    t0 = time.time()
    n_quant_2d = 0
    n_quant_3d_slices = 0
    n_passthrough = 0

    def emit(key: str, tensor: torch.Tensor) -> None:
        nonlocal current_buffer, current_size, out_shard_idx
        sz = tensor.numel() * tensor.element_size()
        if current_size + sz > MAX_SHARD_BYTES and current_buffer:
            _flush_shard(current_buffer, OUTPUT_DIR, out_shard_idx, weight_map_out, out_shard_template)
            current_buffer = {}
            current_size = 0
            out_shard_idx += 1
        current_buffer[key] = tensor.contiguous()
        current_size += sz

    for src_shard_name in sorted(shards_to_keys.keys()):
        src_shard_path = SOURCE_DIR / src_shard_name
        keys_in_shard = sorted(shards_to_keys[src_shard_name])
        print(f"\n--- processing {src_shard_name} ({len(keys_in_shard)} keys) ---")
        with safe_open(str(src_shard_path), framework="pt") as st:
            for i, k in enumerate(keys_in_shard):
                tensor = st.get_tensor(k)

                # Drop the multimodal wrapper for the quantization config; we still keep the
                # underlying language_model.* keys unchanged (the loader strips the prefix).
                fused_tag = is_fused_3d_moe_weight(k)
                if fused_tag is not None:
                    if fused_tag == "gate_up_proj":
                        # Split into gate (E, I, H) + up (E, I, H). Gate and up of expert
                        # e MUST share a single per-tensor max so their `weight_global_scale`
                        # values match — NVFP4Experts3D.assemble_nvfp4_experts3d_batched
                        # asserts equality and fuses them into one buffer.
                        gate, up = split_gate_up_proj(tensor)
                        base = k[: -len(".gate_up_proj")]
                        E = gate.shape[0]
                        for e_idx in range(E):
                            shared_max = max(
                                gate[e_idx].abs().amax().float().item(),
                                up[e_idx].abs().amax().float().item(),
                            )
                            for proj_name, W2d in (
                                ("gate_proj", gate[e_idx].contiguous()),
                                ("up_proj",   up[e_idx].contiguous()),
                            ):
                                p, s, g = quantize_to_nvfp4_2d(W2d, per_tensor_max_override=shared_max)
                                pref = f"{base}.{e_idx}.{proj_name}"
                                emit(f"{pref}.weight_packed", p)
                                emit(f"{pref}.weight_scale", s)
                                emit(f"{pref}.weight_global_scale", g)
                                n_quant_3d_slices += 1
                        del gate, up
                    else:  # "down_proj"
                        slice_results = quantize_to_nvfp4_3d_per_slice(tensor)
                        base = k[: -len(".down_proj")]
                        for e_idx, (p, s, g) in enumerate(slice_results):
                            pref = f"{base}.{e_idx}.down_proj"
                            emit(f"{pref}.weight_packed", p)
                            emit(f"{pref}.weight_scale", s)
                            emit(f"{pref}.weight_global_scale", g)
                            n_quant_3d_slices += 1
                    del tensor
                    gc.collect()
                    continue

                if is_norm_or_gate_or_embed(k):
                    emit(k, tensor)
                    n_passthrough += 1
                    continue

                # Linear 2D weight
                if tensor.ndim == 2 and k.endswith(".weight"):
                    p, s, g = quantize_to_nvfp4_2d(tensor)
                    base = k[: -len(".weight")]
                    emit(f"{base}.weight_packed", p)
                    emit(f"{base}.weight_scale", s)
                    emit(f"{base}.weight_global_scale", g)
                    n_quant_2d += 1
                    del tensor
                    continue

                # Fallback: pass through (biases, anything else 1D)
                emit(k, tensor)
                n_passthrough += 1

                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    print(
                        f"  [{i+1}/{len(keys_in_shard)}]  2d={n_quant_2d}  3d_slices={n_quant_3d_slices}  "
                        f"passthrough={n_passthrough}  elapsed={elapsed:.1f}s"
                    )

    # Flush final shard
    if current_buffer:
        _flush_shard(current_buffer, OUTPUT_DIR, out_shard_idx, weight_map_out, out_shard_template)

    print(f"\nWriting model.safetensors.index.json…")
    total_size = sum(
        (OUTPUT_DIR / shard).stat().st_size for shard in set(weight_map_out.values())
    )
    index_out = {"metadata": {"total_size": total_size}, "weight_map": weight_map_out}
    (OUTPUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index_out, indent=2))

    print(f"Writing config.json with quantization_config block…")
    cfg = json.loads((SOURCE_DIR / "config.json").read_text())
    cfg["quantization_config"] = make_quantization_config(
        ignore_globs=[
            "lm_head",
            "re:.*embed_tokens$",
            "re:.*gate$",
            "re:.*vision_tower.*",
            "re:.*multi_modal_projector.*",
            "re:.*layernorm.*",
            "re:.*\\.norm$",
        ]
    )
    (OUTPUT_DIR / "config.json").write_text(json.dumps(cfg, indent=2))

    # Copy tokenizer + chat template + processor + any other non-shard files
    print(f"Copying tokenizer / processor files…")
    for f in SOURCE_DIR.iterdir():
        if f.suffix in (".json", ".jinja", ".txt", ".md", ".model") and not f.name.startswith("model"):
            shutil.copy2(f, OUTPUT_DIR / f.name)
        if f.name == "tokenizer.json" or f.name == "tokenizer.model":
            shutil.copy2(f, OUTPUT_DIR / f.name)

    elapsed = time.time() - t0
    print(f"\n=== Quantization complete in {elapsed/60:.1f} min ===")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Stats: 2D-quantized={n_quant_2d}, 3D-slice-quantized={n_quant_3d_slices}, passthrough={n_passthrough}")
    print(f"Total output size: {total_size/1e9:.2f} GB")


if __name__ == "__main__":
    main()
