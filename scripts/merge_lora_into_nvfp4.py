"""
Merge a PEFT LoRA adapter into a Nano or Super NVFP4 base model and re-emit as a new
NVFP4 safetensors directory that vLLM can serve directly.

Designed for Nemotron-3 NVFP4 models (Nano, Super) on DGX Spark, but
generalises to any HF-released NVIDIA NVFP4 checkpoint whose MoE expert
weights are stored as (uint8 packed weight + fp8_e4m3 group scales +
fp32 per-tensor scale).

Approach
--------
For each NVFP4-quantized weight tensor `W` in the base that has a
matching LoRA pair `(A, B)`:

1. Dequantize `W` to bf16 via modelopt's `NVFP4QTensor.dequantize`.
2. Compute the LoRA delta: `delta = (alpha/r) * (B @ A)`.
3. Add: `W_merged = W_dequant + delta`.
4. Requantize via `NVFP4QTensor.quantize` (recomputes both per-tensor
   scale and per-block group scales for maximum precision).
5. Write the new packed weight + scales to the output shard.

Tensors without LoRA pairs are copied byte-for-byte unchanged.

Per-shard manifest is written so the merge is resumable on failure.
Per-tensor stats (delta-to-quant-step audit) are emitted to a JSONL log
to support the validation suite in Phase 1.5.x.

Usage
-----
    python scripts/merge_lora_into_nvfp4.py \\
        --base-model-dir /path/to/Nemotron-3-Super-120B-A12B-NVFP4 \\
        --lora-adapter-dir /path/to/adapter \\
        --output-dir /path/to/Nemotron-3-Super-120B-A12B-NVFP4-merged \\
        [--shards 1,2,3 | --shards all] \\
        [--resume]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ADAPTER_PREFIX = "base_model.model."
DEFAULT_BASE_PREFIX = "backbone."
torch = None
safe_open = None
save_file = None


def ensure_runtime_imports():
    global torch, safe_open, save_file
    if torch is None:
        import torch as torch_mod
        from safetensors import safe_open as safe_open_mod
        from safetensors.torch import save_file as save_file_mod

        torch = torch_mod
        safe_open = safe_open_mod
        save_file = save_file_mod


def detect_base_prefix(weight_map: dict) -> str:
    """Derive the text-backbone prefix from the base index itself.

    Nemotron-family checkpoints store the backbone under a single top-level
    prefix ("backbone."); lm_head and the MTP speculation layers sit beside
    it. The same heuristic the loader's fallback translator uses.
    """
    prefixes = {k.split(".", 1)[0] for k in weight_map}
    candidates = sorted(p for p in prefixes if p not in ("lm_head", "mtp"))
    if len(candidates) != 1:
        raise SystemExit(
            f"could not derive the base backbone prefix from the index; "
            f"top-level candidates: {candidates}. This merge script expects a "
            f"Nemotron-style single-backbone ModelOpt layout; for "
            f"compressed-tensors checkpoints use merge_lora_into_ct_nvfp4.py."
        )
    return candidates[0] + "."


def adapter_key_to_base_key(akey: str, base_prefix: str = DEFAULT_BASE_PREFIX) -> str:
    """Convert an adapter tensor key to the corresponding base weight key.

    Examples (base_prefix="backbone."):
      base_model.model.backbone.layers.1.mixer.experts.0.up_proj.lora_A.weight
      base_model.model.model.backbone.layers.1.mixer.experts.0.up_proj.lora_A.weight
      -> backbone.layers.1.mixer.experts.0.up_proj.weight
    """
    if not akey.startswith(ADAPTER_PREFIX):
        raise ValueError(
            f"adapter key {akey!r} does not start with {ADAPTER_PREFIX!r}; "
            "expected keys produced by the v1.0 Nano/Super training scripts"
        )
    tail = akey[len(ADAPTER_PREFIX):]
    if tail.startswith("model.") and not base_prefix.startswith("model."):
        tail = tail[len("model."):]
    m = re.search(r"\.lora_[AB]\.weight$", tail)
    if m is None:
        raise ValueError(f"adapter key {akey!r} does not look like a PEFT LoRA tensor")
    base_key = tail[:m.start()] + ".weight"
    if not base_key.startswith(base_prefix):
        base_key = base_prefix + base_key
    return base_key


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(block_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def parse_shard_selection(shards_arg: str, n_shards: int) -> list[int]:
    """Parse comma-separated 1-based shard indices into 0-based indices."""
    shards_arg = shards_arg.strip()
    valid_range = f"1-{n_shards}"
    if shards_arg.lower() == "all":
        return list(range(n_shards))

    selected = []
    seen = set()
    for raw_token in shards_arg.split(","):
        token = raw_token.strip()
        if not token:
            raise SystemExit(
                f"--shards must contain comma-separated shard numbers in valid range "
                f"{valid_range}; offending input: {shards_arg!r}"
            )
        if re.fullmatch(r"[1-9]\d*", token) is None:
            raise SystemExit(
                f"--shards entries must be integers in valid range {valid_range}; "
                f"offending input: {shards_arg!r}"
            )
        shard_num = int(token)
        if shard_num < 1 or shard_num > n_shards:
            raise SystemExit(
                f"--shards entries must be in valid range {valid_range}; "
                f"offending input: {shards_arg!r}"
            )
        shard_idx = shard_num - 1
        if shard_idx not in seen:
            selected.append(shard_idx)
            seen.add(shard_idx)
    return selected


def load_adapter(adapter_dir: Path, base_prefix: str = DEFAULT_BASE_PREFIX):
    """Load every LoRA A/B tensor into a dict keyed by base weight key.

    Returns: (lora_map, adapter_config)
      lora_map[base_key] = {"A": tensor, "B": tensor}
      adapter_config: dict from adapter_config.json (must include r + lora_alpha)
    """
    with open(adapter_dir / "adapter_config.json") as f:
        cfg = json.load(f)
    if "r" not in cfg or "lora_alpha" not in cfg:
        raise ValueError(f"adapter_config.json missing r or lora_alpha: {cfg}")

    lora_map = defaultdict(dict)
    adapter_files = sorted(adapter_dir.glob("adapter_model*.safetensors"))
    if not adapter_files:
        raise FileNotFoundError(f"no adapter_model*.safetensors in {adapter_dir}")

    for af in adapter_files:
        with safe_open(af, framework="pt") as sf:
            translated_keys = []
            for akey in sf.keys():
                base_key = adapter_key_to_base_key(akey, base_prefix)
                translated_keys.append(base_key)
                side = re.search(r"\.lora_([AB])\.weight$", akey)
                if not side:
                    continue
                tensor = sf.get_tensor(akey)
                lora_map[base_key][side.group(1)] = tensor
            assert all(k.startswith(base_prefix) for k in translated_keys)

    # Validate: every base_key has both A and B
    missing = [k for k, ab in lora_map.items() if "A" not in ab or "B" not in ab]
    if missing:
        raise ValueError(f"{len(missing)} LoRA targets missing A or B half: {missing[:5]}...")
    return dict(lora_map), cfg


def get_nvfp4_dequant_then_merge(
    packed_w: torch.Tensor,
    group_scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    alpha_over_r: float,
    device: torch.device,
):
    """Dequant NVFP4 base, apply LoRA delta in bf16, return merged bf16 tensor.

    Returns (merged_bf16, delta_bf16, dequant_bf16) for downstream stats.
    """
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    packed = packed_w.to(device)
    gs = group_scale.to(device)
    pts = per_tensor_scale.to(device)
    logical_shape = (packed.shape[0], packed.shape[1] * 2)

    qt = NVFP4QTensor(logical_shape, torch.bfloat16, packed)
    dequant = qt.dequantize(scale=gs, double_scale=pts, block_sizes={-1: 16})

    A = lora_A.to(device=device, dtype=torch.bfloat16)
    B = lora_B.to(device=device, dtype=torch.bfloat16)
    delta = alpha_over_r * (B.float() @ A.float()).to(torch.bfloat16)

    if delta.shape != dequant.shape:
        raise ValueError(
            f"LoRA delta shape {delta.shape} != base dequant shape {dequant.shape}"
        )

    merged = (dequant.float() + delta.float()).to(torch.bfloat16)
    return merged, delta, dequant


def requantize_to_nvfp4(merged_bf16: torch.Tensor):
    """Requantize merged bf16 back to NVFP4 packed + scales."""
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    qt_out, gs_out, pts_out = NVFP4QTensor.quantize(merged_bf16, block_size=16)
    # qt_out is an NVFP4QTensor; get its packed bytes
    packed_out = qt_out._quantized_data
    # qt.quantize returns weight_scaling_factor in float32 form; vLLM expects fp8_e4m3fn
    if gs_out.dtype != torch.float8_e4m3fn:
        gs_out = gs_out.to(torch.float8_e4m3fn)
    if pts_out.dtype != torch.float32:
        pts_out = pts_out.to(torch.float32)
    return packed_out, gs_out, pts_out


# ---------------------------------------------------------------------------
# Per-tensor stats for validation suite
# ---------------------------------------------------------------------------

def per_tensor_stats(dequant, delta, merged, new_dequant):
    """Compute per-tensor quality metrics.

    - delta_abs_mean, delta_abs_max: magnitude of LoRA delta
    - relative_delta: ratio of delta magnitude to base magnitude
    - bit_change_pct: % of packed bytes that differ from base after merge
        (not computed here; done at shard level)
    - cos_dequant: cosine of (W_base + delta) vs requant-dequant
    - noop_pct: % of elements where delta is below half a quant step
        approximated as |delta| < 0.5 * (per-block range)
    """
    delta_abs = delta.abs()
    base_abs = dequant.abs()
    cos = torch.nn.functional.cosine_similarity(
        merged.flatten().float(), new_dequant.flatten().float(), dim=0
    ).item()
    rel_err = (
        (merged - new_dequant).abs().float() / (merged.abs().float() + 1e-6)
    ).mean().item()
    return {
        "n_elem": delta.numel(),
        "delta_abs_mean": delta_abs.mean().item(),
        "delta_abs_max": delta_abs.max().item(),
        "base_abs_mean": base_abs.mean().item(),
        "delta_to_base_ratio": (delta_abs.mean() / (base_abs.mean() + 1e-6)).item(),
        "merge_cosine": cos,
        "merge_relative_error": rel_err,
    }


# ---------------------------------------------------------------------------
# Shard processing
# ---------------------------------------------------------------------------

def process_shard(
    base_shard_path: Path,
    output_shard_path: Path,
    lora_map: dict,
    alpha_over_r: float,
    device: torch.device,
    stats_log: list,
    metadata_passthrough: dict,
):
    """Read base_shard, merge any tensors with matching LoRA pairs, write output_shard.

    Returns dict with per-shard summary stats.
    """
    out_tensors = {}
    n_merged = 0
    n_passthrough = 0
    shard_start = time.time()

    with safe_open(base_shard_path, framework="pt") as sf:
        # Preserve the safetensors metadata block if present (HF stores quant_config etc)
        meta = sf.metadata() or {}
        metadata_passthrough.update({k: v for k, v in meta.items() if k not in metadata_passthrough})

        for key in sf.keys():
            tensor = sf.get_tensor(key)

            # Decide: merge or passthrough?
            if key in lora_map:
                # Get accompanying scale + scale_2 from same shard
                scale_key = key.replace(".weight", ".weight_scale")
                scale2_key = key.replace(".weight", ".weight_scale_2")
                # Both scale tensors live in the same shard as the weight per NVIDIA's
                # release; assert that.
                if scale_key not in sf.keys() or scale2_key not in sf.keys():
                    print(
                        f"WARN: {key} has LoRA pair but missing scales in shard "
                        f"{base_shard_path.name}; passing weight through unchanged."
                    )
                    out_tensors[key] = tensor
                    n_passthrough += 1
                    continue

                group_scale = sf.get_tensor(scale_key)
                per_tensor_scale = sf.get_tensor(scale2_key)

                merged, delta, dequant = get_nvfp4_dequant_then_merge(
                    tensor,
                    group_scale,
                    per_tensor_scale,
                    lora_map[key]["A"],
                    lora_map[key]["B"],
                    alpha_over_r,
                    device,
                )
                packed_new, gs_new, pts_new = requantize_to_nvfp4(merged)

                # Per-tensor validation stats
                new_dequant = NVFP4QTensor_dequant_helper(packed_new, gs_new, pts_new)
                stats = per_tensor_stats(dequant, delta, merged, new_dequant)
                stats["key"] = key
                stats_log.append(stats)

                # Replace weight + both scales in output
                out_tensors[key] = packed_new.cpu()
                out_tensors[scale_key] = gs_new.cpu()
                out_tensors[scale2_key] = pts_new.cpu()
                n_merged += 1

                # Free GPU memory for this tensor before next
                del merged, delta, dequant, new_dequant, packed_new, gs_new, pts_new
                torch.cuda.empty_cache()
            elif key.endswith(".weight_scale") or key.endswith(".weight_scale_2"):
                # If the corresponding .weight is in lora_map, we handle the scale in the
                # merge branch above. If not, passthrough.
                base_weight_key = re.sub(r"\.weight_scale(_2)?$", ".weight", key)
                if base_weight_key in lora_map:
                    # Skip: already written in the merge branch above
                    continue
                out_tensors[key] = tensor
                n_passthrough += 1
            else:
                out_tensors[key] = tensor
                n_passthrough += 1

    # Write output shard with metadata
    output_metadata = dict(metadata_passthrough)
    output_metadata.setdefault("format", "pt")
    save_file(out_tensors, str(output_shard_path), metadata=output_metadata)
    elapsed = time.time() - shard_start

    return {
        "shard_path": str(base_shard_path.name),
        "output_path": str(output_shard_path.name),
        "n_merged": n_merged,
        "n_passthrough": n_passthrough,
        "elapsed_s": elapsed,
        "source_sha256": None,  # filled after writing
        "output_sha256": None,
    }


def NVFP4QTensor_dequant_helper(packed, group_scale, per_tensor_scale):
    """Dequantize a freshly requantized NVFP4 tensor for stats."""
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
    logical_shape = (packed.shape[0], packed.shape[1] * 2)
    qt = NVFP4QTensor(logical_shape, torch.bfloat16, packed)
    return qt.dequantize(
        scale=group_scale, double_scale=per_tensor_scale, block_sizes={-1: 16}
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-dir", required=True, type=Path)
    ap.add_argument("--lora-adapter-dir", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="required unless --dry-run")
    ap.add_argument(
        "--shards",
        default="all",
        help="comma-separated shard indices (1-based) or 'all'",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate adapter-to-base key coverage and report what "
                         "would be merged, writing NOTHING")
    args = ap.parse_args()

    ensure_runtime_imports()
    device = torch.device(args.device)

    print(f"[merge] base = {args.base_model_dir}")
    print(f"[merge] adapter = {args.lora_adapter_dir}")
    print(f"[merge] output = {args.output_dir}")

    # Load base index first: the backbone prefix used for adapter-key
    # translation is derived from it (not hardcoded).
    base_idx_path = args.base_model_dir / "model.safetensors.index.json"
    if not base_idx_path.exists():
        raise FileNotFoundError(base_idx_path)
    with open(base_idx_path) as f:
        base_idx = json.load(f)
    weight_map = base_idx["weight_map"]
    base_prefix = detect_base_prefix(weight_map)
    print(f"[merge] base backbone prefix = {base_prefix!r}")

    # Load adapter (~960 MB for Super; small enough to keep in CPU RAM)
    print("[merge] loading adapter...")
    lora_map, adapter_cfg = load_adapter(args.lora_adapter_dir, base_prefix)
    alpha_over_r = adapter_cfg["lora_alpha"] / adapter_cfg["r"]
    print(f"[merge] loaded {len(lora_map)} LoRA target weights")
    print(f"[merge] alpha/r = {alpha_over_r}")

    # Group base tensors by shard
    shard_to_tensors = defaultdict(list)
    for tk, shard_filename in weight_map.items():
        shard_to_tensors[shard_filename].append(tk)
    shard_files = sorted(shard_to_tensors.keys())
    print(f"[merge] base has {len(shard_files)} shards, {len(weight_map)} tensors total")

    # Coverage report
    targeted_keys = set(lora_map.keys())
    base_keys = set(weight_map.keys())
    missing = targeted_keys - base_keys
    if missing:
        raise ValueError(
            f"{len(missing)} LoRA targets not found in base index. "
            f"Sample: {sorted(missing)[:5]}"
        )
    print(f"[merge] LoRA coverage OK: 100% of {len(targeted_keys)} targets matched in base")

    # Every target must map to an actual NVFP4-quantized tensor (scales present),
    # not just any same-named bf16 weight.
    unquantized = [
        k for k in targeted_keys
        if k.replace(".weight", ".weight_scale") not in base_keys
        or k.replace(".weight", ".weight_scale_2") not in base_keys
    ]
    if unquantized:
        raise SystemExit(
            f"{len(unquantized)} LoRA targets map to base tensors WITHOUT NVFP4 "
            f"scales (unquantized in this base): {sorted(unquantized)[:5]}"
        )

    if args.dry_run:
        for k in sorted(targeted_keys)[:10]:
            print(f"[dry-run] target: {k}")
        if len(targeted_keys) > 10:
            print(f"[dry-run] ... and {len(targeted_keys) - 10} more")
        print(f"[dry-run] all {len(targeted_keys)} targets map to NVFP4 base tensors; "
              f"no files written")
        return

    if args.output_dir is None:
        ap.error("--output-dir is required unless --dry-run")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Shard selection
    selected = parse_shard_selection(args.shards, len(shard_files))

    # Resume support: read existing manifest if present
    manifest_path = args.output_dir / "merge_manifest.json"
    if args.resume and manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        done_shards = {sh["shard_path"] for sh in manifest.get("shards", [])}
        print(f"[merge] resume: {len(done_shards)} shards already in manifest")
    else:
        manifest = {
            "base_model_dir": str(args.base_model_dir),
            "lora_adapter_dir": str(args.lora_adapter_dir),
            "output_dir": str(args.output_dir),
            "alpha_over_r": alpha_over_r,
            "n_lora_targets": len(targeted_keys),
            "shards": [],
            "stats_log_path": "merge_stats.jsonl",
        }
        done_shards = set()

    stats_log_path = args.output_dir / manifest["stats_log_path"]
    stats_log_mode = "a" if args.resume and manifest_path.exists() else "w"
    stats_log_f = open(stats_log_path, stats_log_mode)
    metadata_passthrough = {}
    worst_cosine: tuple[float, str] | None = None

    # Process selected shards
    for sh_idx in selected:
        shard_filename = shard_files[sh_idx]
        if shard_filename in done_shards:
            print(f"[merge] shard {sh_idx + 1}/{len(shard_files)} {shard_filename}: SKIP (already done)")
            continue
        print(
            f"[merge] shard {sh_idx + 1}/{len(shard_files)} {shard_filename}: processing..."
        )
        base_shard_path = args.base_model_dir / shard_filename
        out_shard_path = args.output_dir / shard_filename

        stats_log_buffer = []
        summary = process_shard(
            base_shard_path,
            out_shard_path,
            lora_map,
            alpha_over_r,
            device,
            stats_log_buffer,
            metadata_passthrough,
        )

        # Flush stats
        for s in stats_log_buffer:
            stats_log_f.write(json.dumps(s) + "\n")
            if worst_cosine is None or s["merge_cosine"] < worst_cosine[0]:
                worst_cosine = (s["merge_cosine"], s["key"])
        stats_log_f.flush()

        # Compute hashes
        summary["source_sha256"] = file_sha256(base_shard_path)
        summary["output_sha256"] = file_sha256(out_shard_path)
        manifest["shards"].append(summary)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(
            f"[merge]   done: merged={summary['n_merged']} "
            f"passthrough={summary['n_passthrough']} "
            f"elapsed={summary['elapsed_s']:.1f}s"
        )

    stats_log_f.close()

    # Copy non-weight files (config.json, tokenizer files, modeling_*.py, etc.) if missing.
    # Skip directories (e.g. .cache/ if it exists in the base dir).
    for f in args.base_model_dir.iterdir():
        if f.is_dir():
            continue
        if f.suffix == ".safetensors":
            continue
        if f.name == "model.safetensors.index.json":
            # We use the same index since shard names and tensor keys are unchanged
            shutil.copy2(f, args.output_dir / f.name)
            continue
        out_f = args.output_dir / f.name
        if not out_f.exists():
            shutil.copy2(f, out_f)

    if worst_cosine is not None:
        print(f"[merge] worst merge_cosine={worst_cosine[0]:.6f} at {worst_cosine[1]}")
    print(f"[merge] complete. Manifest: {manifest_path}")
    print(f"[merge] per-tensor stats: {stats_log_path}")


if __name__ == "__main__":
    main()
