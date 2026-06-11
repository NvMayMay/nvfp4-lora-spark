#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a compressed-tensors NVFP4 base checkpoint.

DO NOT RUN UNTIL TRAINING COMPLETES. The 13h Qwen3.5 run owns the GPU until
roughly 01:30; this script may initialize CUDA at import time (the quantizer
module probes torch.cuda.is_available()), so do not even --dry-run it while
the trainer is alive unless you pass --device cpu AND set
CUDA_VISIBLE_DEVICES="" in the environment.

Compressed-tensors counterpart of scripts/merge_lora_into_nvfp4.py (which is
modelopt/Nemotron specific). Differences, by design:

- Key layout: CT stores each quantized linear as
      {prefix}.weight_packed        uint8   (out, in/2)
      {prefix}.weight_scale         fp8_e4m3(out, in/16)
      {prefix}.weight_global_scale  fp32    (1,)   <- stored as a DIVISOR
      {prefix}.input_global_scale   fp32    (1,)   <- activation side, untouched
- Adapter naming: the Qwen3.5 v3.5 trainer saved keys like
      base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight
  while the base checkpoint uses model.language_model.layers.3....
  This script inserts the missing "language_model." segment.
- Dequant uses the repo's own nvfp4_lora.dequant.dequantize_nvfp4_weight with
  format="compressed_tensors", i.e. EXACTLY the function the trainer used in
  its forward pass, so merged = trained function up to one requantization.
- Requant uses quantize_to_nvfp4_2d from scripts/quantize_mistral_to_nvfp4.py
  (CT scale convention). Each layer's q/k/v are requantized with a SHARED
  per-tensor abs-max so their weight_global_scale stays identical; vLLM fuses
  q/k/v into one qkv_proj and degrades accuracy (with a warning) if the global
  scales differ (see vllm .../compressed_tensors_w4a4_nvfp4.py
  process_weights_after_loading).

Phases:
  1. Load adapter, map every LoRA pair to a base prefix, group q/k/v trios.
  2. For each target: dequant -> add (alpha/r) * B @ A -> requantize.
     Per-tensor stats (cosine of merged vs requant-dequant, relative error,
     delta magnitudes) go to <output>/merge_stats.jsonl.
  3. Rewrite each base shard once, swapping in the new packed weights and
     scales; everything else (including input_global_scale) passes through.
     NOTE: the Qwen base is 2 shards x ~36 GB, so expect ~40 GB host RAM
     peak per shard. Fine once training has released the UMA.
  4. Copy index + config/tokenizer files unchanged (tensor keys, shapes and
     dtypes are unchanged, so the original index stays valid).

Usage (post-training):
    cd /home/veritan-spark-01/Veritan/Sandbox/repos/nvfp4-lora-spark
    /home/veritan-spark-01/Veritan/.venvs/qwen-serve/bin/python \
        scripts/merge_lora_into_ct_nvfp4.py \
        --base-model-dir /home/veritan-spark-01/Veritan/Models/RedHatAI-Qwen3.5-122B-A10B-NVFP4 \
        --lora-adapter-dir /home/veritan-spark-01/Veritan/Sandbox/adapters/qwen3_5_122b_a10b_rh_nvfp4_lora_ich_v3_5 \
        --output-dir /home/veritan-spark-01/Veritan/Models/RedHatAI-Qwen3.5-122B-A10B-NVFP4-ich-v3.5

Cheap checks first:
    ... --self-test                  # CPU round-trip on random tensors, no model files
    ... --dry-run                    # adapter/base coverage report, no writes
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))          # for nvfp4_lora
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # for quantize_mistral_to_nvfp4

ADAPTER_PREFIX = "base_model.model."
LM_PREFIX = "model.language_model."

_QKV_RE = re.compile(
    r"^(?P<layer>model\.language_model\.layers\.\d+)\.self_attn\.(?P<proj>q_proj|k_proj|v_proj)$"
)


# ---------------------------------------------------------------------------
# Adapter key mapping
# ---------------------------------------------------------------------------

def adapter_key_to_base_prefix(akey: str) -> tuple[str, str]:
    """Map a PEFT adapter tensor key to (base module prefix, 'A'|'B').

    base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight
        -> ("model.language_model.layers.3.self_attn.q_proj", "A")
    base_model.model.model.language_model.layers.3....   (already prefixed)
        -> passthrough
    """
    if not akey.startswith(ADAPTER_PREFIX):
        raise ValueError(f"adapter key {akey!r} does not start with {ADAPTER_PREFIX!r}")
    tail = akey[len(ADAPTER_PREFIX):]
    m = re.search(r"\.lora_(?P<side>[AB])\.weight$", tail)
    if m is None:
        raise ValueError(f"adapter key {akey!r} is not a lora_A/lora_B weight")
    prefix = tail[: m.start()]
    if prefix.startswith(LM_PREFIX):
        pass
    elif prefix.startswith("model."):
        prefix = LM_PREFIX + prefix[len("model."):]
    else:
        raise ValueError(f"adapter key {akey!r} has unrecognized module path {prefix!r}")
    return prefix, m.group("side")


def load_adapter(adapter_dir: Path):
    from safetensors import safe_open

    cfg_path = adapter_dir / "adapter_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    if "r" not in cfg or "lora_alpha" not in cfg:
        raise ValueError(f"{cfg_path} missing r or lora_alpha")
    if cfg.get("use_rslora"):
        scale = cfg["lora_alpha"] / math.sqrt(cfg["r"])
    else:
        scale = cfg["lora_alpha"] / cfg["r"]

    adapter_files = sorted(adapter_dir.glob("adapter_model*.safetensors"))
    if not adapter_files:
        raise FileNotFoundError(f"no adapter_model*.safetensors in {adapter_dir}")

    lora_map: dict[str, dict] = defaultdict(dict)
    for af in adapter_files:
        with safe_open(af, framework="pt") as sf:
            for akey in sf.keys():
                prefix, side = adapter_key_to_base_prefix(akey)
                lora_map[prefix][side] = sf.get_tensor(akey)

    missing = [k for k, ab in lora_map.items() if "A" not in ab or "B" not in ab]
    if missing:
        raise ValueError(f"{len(missing)} LoRA targets missing one half: {missing[:5]}")
    return dict(lora_map), cfg, scale


def scale_groups(prefixes: list[str]) -> list[list[str]]:
    """Group target prefixes that must share one requant per-tensor max.

    q/k/v of the same layer form one group (vLLM fuses them into qkv_proj and
    requires equal weight_global_scale); everything else is a singleton.
    """
    qkv: dict[str, list[str]] = defaultdict(list)
    singles: list[list[str]] = []
    for p in prefixes:
        m = _QKV_RE.match(p)
        if m:
            qkv[m.group("layer")].append(p)
        else:
            singles.append([p])
    groups = [sorted(v) for _, v in sorted(qkv.items())] + sorted(singles)
    return groups


# ---------------------------------------------------------------------------
# Merge math
# ---------------------------------------------------------------------------

def _read_ct_trio(weight_map: dict, base_dir: Path, prefix: str, opened: dict):
    """Fetch (weight_packed, weight_scale, weight_global_scale) for a prefix."""
    from safetensors import safe_open

    out = []
    for suffix in (".weight_packed", ".weight_scale", ".weight_global_scale"):
        key = prefix + suffix
        if key not in weight_map:
            raise KeyError(f"{key} not found in base index (is this target quantized?)")
        shard = weight_map[key]
        if shard not in opened:
            opened[shard] = safe_open(base_dir / shard, framework="pt")
        out.append(opened[shard].get_tensor(key))
    return tuple(out)


def dequant_merge(packed, gs_fp8, global_scale, A, B, scale, device):
    """Return (merged_fp32, dequant_bf16, delta_fp32) on `device`."""
    import torch
    from nvfp4_lora.dequant import dequantize_nvfp4_weight

    dequant = dequantize_nvfp4_weight(
        packed.to(device),
        gs_fp8.to(device),
        global_scale.to(torch.float32).to(device),
        group_size=16,
        out_dtype=torch.bfloat16,
        format="compressed_tensors",
    )
    delta = scale * (B.to(device, torch.float32) @ A.to(device, torch.float32))
    if tuple(delta.shape) != tuple(dequant.shape):
        raise ValueError(f"delta shape {tuple(delta.shape)} != base {tuple(dequant.shape)}")
    merged = dequant.to(torch.float32) + delta
    return merged, dequant, delta


def merge_targets(lora_map, scale, base_dir: Path, weight_map: dict, device,
                  stats_out: list) -> dict:
    """Compute requantized CT trios for every LoRA target.

    Returns {tensor_key: cpu_tensor} covering weight_packed / weight_scale /
    weight_global_scale for each target prefix.
    """
    import torch
    from nvfp4_lora.dequant import dequantize_nvfp4_weight
    from quantize_mistral_to_nvfp4 import quantize_to_nvfp4_2d

    opened: dict = {}
    replacements: dict = {}
    groups = scale_groups(sorted(lora_map.keys()))
    print(f"[merge] {len(lora_map)} targets in {len(groups)} scale groups")

    for group in groups:
        merged_members = []
        for prefix in group:
            packed, gs_fp8, gsc = _read_ct_trio(weight_map, base_dir, prefix, opened)
            merged, dequant, delta = dequant_merge(
                packed, gs_fp8, gsc, lora_map[prefix]["A"], lora_map[prefix]["B"],
                scale, device,
            )
            merged_members.append((prefix, merged, dequant, delta, gsc))

        shared_max = max(m.abs().max().item() for _, m, _, _, _ in merged_members)
        share = len(group) > 1

        for prefix, merged, dequant, delta, old_gsc in merged_members:
            packed_new, scale_new, gsc_new = quantize_to_nvfp4_2d(
                merged, per_tensor_max_override=shared_max if share else None
            )
            new_dequant = dequantize_nvfp4_weight(
                packed_new.to(device),
                scale_new.to(device),
                gsc_new.to(device),
                group_size=16,
                out_dtype=torch.bfloat16,
                format="compressed_tensors",
            )
            mflat = merged.flatten()
            nflat = new_dequant.to(torch.float32).flatten()
            cos = torch.nn.functional.cosine_similarity(mflat, nflat, dim=0).item()
            rel = ((mflat - nflat).abs() / (mflat.abs() + 1e-6)).mean().item()
            stats_out.append({
                "key": prefix,
                "n_elem": merged.numel(),
                "delta_abs_mean": delta.abs().mean().item(),
                "delta_abs_max": delta.abs().max().item(),
                "base_abs_mean": dequant.to(torch.float32).abs().mean().item(),
                "delta_to_base_ratio": (
                    delta.abs().mean() / (dequant.to(torch.float32).abs().mean() + 1e-6)
                ).item(),
                "merge_cosine": cos,
                "merge_relative_error": rel,
                "old_global_scale": float(old_gsc.reshape(-1)[0]),
                "new_global_scale": float(gsc_new.reshape(-1)[0]),
                "shared_scale_group": group if share else None,
            })
            replacements[prefix + ".weight_packed"] = packed_new.cpu()
            replacements[prefix + ".weight_scale"] = scale_new.cpu()
            replacements[prefix + ".weight_global_scale"] = gsc_new.cpu()
            print(f"[merge]   {prefix}: cosine={cos:.6f} rel_err={rel:.5f}")
            del merged, dequant, delta, new_dequant
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return replacements


# ---------------------------------------------------------------------------
# Shard rewrite
# ---------------------------------------------------------------------------

def rewrite_shards(base_dir: Path, output_dir: Path, weight_map: dict,
                   replacements: dict) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    shard_files = sorted(set(weight_map.values()))
    used = set()
    for i, shard in enumerate(shard_files, 1):
        t0 = time.time()
        print(f"[merge] shard {i}/{len(shard_files)} {shard}: rewriting...")
        out_tensors = {}
        n_replaced = 0
        with safe_open(base_dir / shard, framework="pt") as sf:
            meta = sf.metadata() or {}
            for key in sf.keys():
                if key in replacements:
                    out_tensors[key] = replacements[key]
                    used.add(key)
                    n_replaced += 1
                else:
                    out_tensors[key] = sf.get_tensor(key)
        meta.setdefault("format", "pt")
        save_file(out_tensors, str(output_dir / shard), metadata=meta)
        del out_tensors
        print(f"[merge]   replaced {n_replaced} tensors in {time.time() - t0:.1f}s")
    unused = set(replacements) - used
    if unused:
        raise RuntimeError(f"{len(unused)} merged tensors never placed: {sorted(unused)[:5]}")


def copy_aux_files(base_dir: Path, output_dir: Path) -> None:
    for f in base_dir.iterdir():
        if f.is_dir() or f.suffix == ".safetensors":
            continue
        dest = output_dir / f.name
        if not dest.exists():
            shutil.copy2(f, dest)


# ---------------------------------------------------------------------------
# Self test (CPU, no model files)
# ---------------------------------------------------------------------------

def self_test() -> int:
    import torch
    from nvfp4_lora.dequant import dequantize_nvfp4_weight
    from quantize_mistral_to_nvfp4 import quantize_to_nvfp4_2d

    torch.manual_seed(0)
    failures = 0
    for out_f, in_f in ((64, 128), (96, 256)):
        W = torch.randn(out_f, in_f, dtype=torch.float32) * 0.02
        packed, gs, gsc = quantize_to_nvfp4_2d(W)
        W_dq = dequantize_nvfp4_weight(
            packed, gs, gsc, group_size=16,
            out_dtype=torch.bfloat16, format="compressed_tensors",
        ).to(torch.float32)
        cos = torch.nn.functional.cosine_similarity(W.flatten(), W_dq.flatten(), dim=0).item()
        print(f"[self-test] quant/dequant ({out_f}x{in_f}): cosine={cos:.6f}")
        if cos < 0.98:
            failures += 1

        # Round-trip stability: requantizing a dequantized tensor should be
        # near-lossless (values already sit on the NVFP4 grid).
        packed2, gs2, gsc2 = quantize_to_nvfp4_2d(W_dq)
        W_dq2 = dequantize_nvfp4_weight(
            packed2, gs2, gsc2, group_size=16,
            out_dtype=torch.bfloat16, format="compressed_tensors",
        ).to(torch.float32)
        cos2 = torch.nn.functional.cosine_similarity(W_dq.flatten(), W_dq2.flatten(), dim=0).item()
        print(f"[self-test] round-trip ({out_f}x{in_f}): cosine={cos2:.6f}")
        if cos2 < 0.999:
            failures += 1

        # Zero-delta merge must reproduce the round trip.
        A = torch.zeros(16, in_f)
        B = torch.zeros(out_f, 16)
        merged = W_dq + 2.0 * (B @ A)
        if not torch.equal(merged, W_dq):
            failures += 1

        # Shared-max grouping must produce identical global scales.
        Wb = torch.randn(out_f, in_f, dtype=torch.float32) * 0.01
        m = max(W_dq.abs().max().item(), Wb.abs().max().item())
        _, _, g1 = quantize_to_nvfp4_2d(W_dq, per_tensor_max_override=m)
        _, _, g2 = quantize_to_nvfp4_2d(Wb, per_tensor_max_override=m)
        if not torch.equal(g1, g2):
            print("[self-test] FAIL: shared-max global scales differ")
            failures += 1

    print(f"[self-test] {'PASS' if failures == 0 else f'FAIL ({failures} checks)'}")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-model-dir", type=Path)
    ap.add_argument("--lora-adapter-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--device", default="cuda",
                    help="device for dequant/merge math; 'cpu' works (slower)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report adapter coverage against the base index, write nothing")
    ap.add_argument("--self-test", action="store_true",
                    help="CPU round-trip sanity check on random tensors, no model files")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if not (args.base_model_dir and args.lora_adapter_dir):
        ap.error("--base-model-dir and --lora-adapter-dir are required (or use --self-test)")

    import torch
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    print(f"[merge] base    = {args.base_model_dir}")
    print(f"[merge] adapter = {args.lora_adapter_dir}")
    print(f"[merge] output  = {args.output_dir}")
    print(f"[merge] device  = {device}")

    lora_map, cfg, scale = load_adapter(args.lora_adapter_dir)
    print(f"[merge] {len(lora_map)} LoRA targets, r={cfg['r']} alpha={cfg['lora_alpha']} scale={scale}")

    idx_path = args.base_model_dir / "model.safetensors.index.json"
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]

    missing = [p for p in lora_map if p + ".weight_packed" not in weight_map]
    if missing:
        raise SystemExit(
            f"{len(missing)} targets have no .weight_packed in the base index "
            f"(unquantized or misnamed): {sorted(missing)[:5]}"
        )
    groups = scale_groups(sorted(lora_map.keys()))
    n_trios = sum(1 for g in groups if len(g) > 1)
    print(f"[merge] coverage OK: {len(lora_map)} targets, {n_trios} shared-scale q/k/v groups")

    if args.dry_run:
        for g in groups:
            print(f"[dry-run] group: {g}")
        print("[dry-run] no files written")
        return 0

    if args.output_dir is None:
        ap.error("--output-dir is required unless --dry-run/--self-test")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats: list[dict] = []
    replacements = merge_targets(
        lora_map, scale, args.base_model_dir, weight_map, device, stats
    )

    stats_path = args.output_dir / "merge_stats.jsonl"
    with open(stats_path, "w") as f:
        for s in stats:
            f.write(json.dumps(s) + "\n")

    rewrite_shards(args.base_model_dir, args.output_dir, weight_map, replacements)
    shutil.copy2(idx_path, args.output_dir / idx_path.name)
    copy_aux_files(args.base_model_dir, args.output_dir)

    worst = min(stats, key=lambda s: s["merge_cosine"])
    print(f"[merge] complete. worst merge_cosine={worst['merge_cosine']:.6f} at {worst['key']}")
    print(f"[merge] stats: {stats_path}")
    manifest = {
        "base_model_dir": str(args.base_model_dir),
        "lora_adapter_dir": str(args.lora_adapter_dir),
        "scale_alpha_over_r": scale,
        "n_targets": len(lora_map),
        "merged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(args.output_dir / "merge_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    raise SystemExit(main())
