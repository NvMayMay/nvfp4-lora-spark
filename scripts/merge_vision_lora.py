#!/usr/bin/env python3
"""Merge a VISION LoRA adapter into an NVFP4 VLM's unquantized (bf16) vision tower.

Vision counterpart of scripts/merge_lora_into_nvfp4.py (ModelOpt) and
scripts/merge_lora_into_ct_nvfp4.py (compressed-tensors). It reuses their
adapter-key mapping + shard-rewrite mechanics but is DELIBERATELY SIMPLER: a
`--train-target vision` adapter only ever targets the tower + multimodal
projector, which every reference NVFP4 VLM leaves UNQUANTIZED (bf16). So the
merge is a plain

    W_merged = W + (alpha / r) * (B @ A)

in the base weight's own dtype -- NO dequantize / requantize, no NVFP4 kernels,
no scale bookkeeping. The frozen 4-bit LLM backbone is never touched.

Why this exists (scoping doc section 6 / PLAN Phase V2): vLLM runtime-LoRA
applies adapters to the LLM backbone only, so a vision-tower adapter has NO
runtime-LoRA path. The supported vision serve story is merge-to-bf16-base: bake
the vision delta into a copy of the base checkpoint and serve the merged VLM
(serve/run_mistral24b_vision_merged.sh). See docs/SERVING.md section 6.

Guardrails:
  * Every adapter target must land on a bf16 tower/projector weight. If a target
    resolves to an NVFP4 (`.weight_packed` / `.weight`+`.weight_scale`) tensor
    the merge REFUSES and points you at merge_lora_into_nvfp4.py /
    merge_lora_into_ct_nvfp4.py -- a quantized target is a different (dequant ->
    add -> requant) problem this script will not silently mishandle.
  * Only shard(s) that actually hold a merged tensor are rewritten; every other
    shard is copied BYTE-FOR-BYTE. A rewritten shard preserves its NVFP4 backbone
    tensors exactly (read-through of the raw packed/scale bytes); only the bf16
    tower tensors change. The index / config / processor / tokenizer files are
    copied unchanged (keys, shapes and dtypes are all preserved, so the base
    index stays valid).

Adapter-key mapping (reuses families.adapter_key_to_base_prefix per rule):
  base_model.model.model.vision_tower.transformer.layers.0.attention.q_proj.lora_A.weight
      -> vision_tower.transformer.layers.0.attention.q_proj.weight
  base_model.model.model.multi_modal_projector.linear_1.lora_B.weight
      -> multi_modal_projector.linear_1.weight
The in-memory <-> on-disk prefix pairs come from the family registry's
`vision_st_to_model`, so the merge translates keys with EXACTLY the rule the
vision trainer used at load time.

Usage:
    python scripts/merge_vision_lora.py \
        --base-model-dir /path/to/Mistral-Small-3.2-24B-NVFP4 \
        --adapter-dir /path/to/vision-adapter \
        --out-dir /path/to/Mistral-Small-3.2-24B-NVFP4-vision-merged

Cheap checks first:
    ... --self-test          # CPU round-trip on random tensors, no model files
    ... --dry-run            # adapter/base coverage report, no writes
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # for nvfp4_lora

from nvfp4_lora.families import resolve_family, family_supports_vision  # noqa: E402
from nvfp4_lora.loader import classify_module_storage  # noqa: E402

ADAPTER_PREFIX = "base_model.model."


# ---------------------------------------------------------------------------
# Adapter-key mapping (vision: tower + projector, both unquantized)
# ---------------------------------------------------------------------------

def vision_prefix_pairs(family: dict) -> list[tuple[str, str]]:
    """(in_memory_prefix, on_disk_prefix) pairs for the tower + projector.

    Derived from the family's `vision_st_to_model` rewrite rules, which are stored
    as (on_disk_prefix, in_memory_prefix); we return them mem-first because the
    adapter carries IN-MEMORY module paths and we translate them to on-disk keys.
    """
    pairs = [(mem, st) for st, mem in family.get("vision_st_to_model", ())]
    if not pairs:
        raise SystemExit(
            "family declares no `vision_st_to_model`; this base does not support "
            "--train-target vision, so it has no vision adapter to merge."
        )
    return pairs


def adapter_key_to_base_key(akey: str, prefix_pairs: list[tuple[str, str]],
                            adapter_prefix: str = ADAPTER_PREFIX) -> str:
    """Map a PEFT vision-LoRA tensor key to its on-disk base WEIGHT key.

    Tries each (in_memory_prefix, on_disk_prefix) pair (tower, projector). The
    per-pair translation is exactly families.adapter_key_to_base_prefix's rule
    (strip the adapter prefix, strip `.lora_{A,B}.weight`, swap the mem prefix for
    the on-disk prefix), then append `.weight`. Keys already carrying an on-disk
    prefix pass through unchanged.
    """
    import re

    if not akey.startswith(adapter_prefix):
        raise ValueError(f"adapter key {akey!r} does not start with {adapter_prefix!r}")
    tail = akey[len(adapter_prefix):]
    m = re.search(r"\.lora_(?P<side>[AB])\.weight$", tail)
    if m is None:
        raise ValueError(f"adapter key {akey!r} is not a lora_A/lora_B weight")
    prefix = tail[: m.start()]
    for mem_prefix, st_prefix in prefix_pairs:
        if prefix.startswith(st_prefix):
            return prefix + ".weight"           # already on-disk form
        if prefix.startswith(mem_prefix):
            return st_prefix + prefix[len(mem_prefix):] + ".weight"
    raise ValueError(
        f"adapter key {akey!r} has module path {prefix!r} that matches no vision "
        f"prefix {[m for m, _ in prefix_pairs]!r}; is this a VISION adapter (a "
        f"--train-target vision run), or a text-backbone adapter for the NVFP4 "
        f"merge scripts (merge_lora_into_nvfp4.py / merge_lora_into_ct_nvfp4.py)?"
    )


def adapter_side(akey: str) -> str:
    import re
    m = re.search(r"\.lora_(?P<side>[AB])\.weight$", akey)
    if m is None:
        raise ValueError(f"adapter key {akey!r} is not a lora_A/lora_B weight")
    return m.group("side")


def load_adapter(adapter_dir: Path, prefix_pairs: list[tuple[str, str]]):
    """Load every LoRA A/B tensor keyed by on-disk base weight key.

    Returns (lora_map, cfg, scale):
      lora_map[base_key] = {"A": tensor, "B": tensor}
      cfg: adapter_config.json dict (must carry r + lora_alpha)
      scale: alpha/r, or alpha/sqrt(r) when use_rslora (matches the CT merge).
    """
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
                base_key = adapter_key_to_base_key(akey, prefix_pairs)
                lora_map[base_key][adapter_side(akey)] = sf.get_tensor(akey)

    missing = [k for k, ab in lora_map.items() if "A" not in ab or "B" not in ab]
    if missing:
        raise ValueError(f"{len(missing)} LoRA targets missing one half: {missing[:5]}")
    return dict(lora_map), cfg, scale


# ---------------------------------------------------------------------------
# Merge math (bf16, no dequant/requant)
# ---------------------------------------------------------------------------

# Unquantized storage classes a vision target is allowed to land on. Anything with
# NVFP4 / FP8 scale siblings is REFUSED (a different merge problem).
_MERGEABLE_STORAGE = ("bf16",)


def assert_mergeable_target(index_keys: set, base_key: str) -> None:
    """Refuse any vision target that is NOT an unquantized (bf16) weight.

    classify_module_storage keys off the index alone: a bf16 module has ONLY
    `.weight` (no `.weight_scale` / `.weight_packed`). An NVFP4/FP8 target here is a
    dequant->add->requant job for the NVFP4 merge scripts, not this one.
    """
    module = base_key[: -len(".weight")] if base_key.endswith(".weight") else base_key
    storage = classify_module_storage(index_keys, module)
    if storage == "absent":
        raise SystemExit(
            f"vision target {base_key!r} is not present in the base index. Check the "
            f"adapter was trained against THIS base, and that the family's "
            f"vision_st_to_model maps its module path correctly."
        )
    if storage not in _MERGEABLE_STORAGE:
        raise SystemExit(
            f"vision target {base_key!r} is stored as {storage!r}, not an unquantized "
            f"bf16 weight. This merge only handles the bf16 tower/projector; a "
            f"quantized target must be merged with scripts/merge_lora_into_nvfp4.py "
            f"(ModelOpt) or scripts/merge_lora_into_ct_nvfp4.py (compressed-tensors), "
            f"which dequantize, add the delta, then requantize. (Vision towers are "
            f"unquantized in every reference VLM; a quantized one is unexpected -- "
            f"re-check the adapter/base pairing.)"
        )


def merged_weight(W, A, B, scale: float):
    """W_merged = W + scale * (B @ A), computed in fp32, cast back to W's dtype.

    B is (out, r), A is (r, in), so B @ A is (out, in) == W. Casting the delta up
    to fp32 for the matmul then back to the base dtype mirrors the NVFP4 merge
    scripts' precision handling.
    """
    import torch

    delta = scale * (B.to(torch.float32) @ A.to(torch.float32))
    if tuple(delta.shape) != tuple(W.shape):
        raise ValueError(
            f"LoRA delta shape {tuple(delta.shape)} != base weight shape {tuple(W.shape)}"
        )
    return (W.to(torch.float32) + delta).to(W.dtype)


def compute_replacements(lora_map: dict, scale: float, base_dir: Path,
                         weight_map: dict, device) -> tuple[dict, list]:
    """Compute the merged bf16 weight for every vision target.

    Returns ({base_key: merged_cpu_tensor}, stats_rows). Reads each base weight
    from its shard, asserts it is a bf16 target, merges, records a per-tensor stat
    row (delta magnitude relative to base -- a no-op detector, cf. validate_merge).
    """
    import torch
    from safetensors import safe_open

    index_keys = set(weight_map.keys())
    opened: dict = {}
    replacements: dict = {}
    stats: list = []
    for base_key in sorted(lora_map):
        assert_mergeable_target(index_keys, base_key)
        shard = weight_map[base_key]
        if shard not in opened:
            opened[shard] = safe_open(base_dir / shard, framework="pt")
        W = opened[shard].get_tensor(base_key).to(device)
        A = lora_map[base_key]["A"].to(device)
        B = lora_map[base_key]["B"].to(device)
        merged = merged_weight(W, A, B, scale)
        delta_abs_mean = (merged.to(torch.float32) - W.to(torch.float32)).abs().mean().item()
        base_abs_mean = W.to(torch.float32).abs().mean().item()
        stats.append({
            "key": base_key,
            "dtype": str(W.dtype).replace("torch.", ""),
            "shape": list(W.shape),
            "delta_abs_mean": delta_abs_mean,
            "base_abs_mean": base_abs_mean,
            "delta_to_base_ratio": delta_abs_mean / (base_abs_mean + 1e-9),
        })
        replacements[base_key] = merged.cpu()
        print(f"[merge]   {base_key}: dtype={stats[-1]['dtype']} "
              f"delta/base={stats[-1]['delta_to_base_ratio']:.5f}")
        del W, A, B, merged
    return replacements, stats


# ---------------------------------------------------------------------------
# Shard rewrite (only the shards that hold a merged tensor; others byte-for-byte)
# ---------------------------------------------------------------------------

def rewrite_shards(base_dir: Path, out_dir: Path, weight_map: dict,
                   replacements: dict) -> dict:
    """Write the merged checkpoint's shards.

    A shard is REWRITTEN iff it holds at least one merged tensor -- in that pass its
    NVFP4 backbone tensors are read and written back verbatim (the raw packed/scale
    bytes round-trip unchanged) and only the bf16 tower tensors are swapped. Every
    other shard is COPIED byte-for-byte. Returns a summary dict.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    all_shards = sorted(set(weight_map.values()))
    # Only replacement keys the base index actually knows drive a shard rewrite; a
    # key absent from the index is caught by the "never placed" guard at the end.
    shards_with_merge = sorted({weight_map[k] for k in replacements if k in weight_map})
    used: set = set()
    for shard in all_shards:
        src = base_dir / shard
        dst = out_dir / shard
        if shard not in shards_with_merge:
            shutil.copy2(src, dst)                          # untouched, byte-for-byte
            continue
        t0 = time.time()
        out_tensors = {}
        n_replaced = 0
        with safe_open(src, framework="pt") as sf:
            meta = sf.metadata() or {}
            for key in sf.keys():
                if key in replacements:
                    out_tensors[key] = replacements[key]    # merged bf16 tower weight
                    used.add(key)
                    n_replaced += 1
                else:
                    out_tensors[key] = sf.get_tensor(key)   # NVFP4 backbone: verbatim
        meta.setdefault("format", "pt")
        save_file(out_tensors, str(dst), metadata=meta)
        del out_tensors
        print(f"[merge] shard {shard}: rewrote {n_replaced} bf16 tensor(s), "
              f"copied the rest verbatim ({time.time() - t0:.1f}s)")
    unused = set(replacements) - used
    if unused:
        raise RuntimeError(f"{len(unused)} merged tensors never placed: {sorted(unused)[:5]}")
    return {
        "n_shards_total": len(all_shards),
        "n_shards_rewritten": len(shards_with_merge),
        "n_shards_copied": len(all_shards) - len(shards_with_merge),
        "shards_rewritten": shards_with_merge,
    }


def copy_aux_files(base_dir: Path, out_dir: Path) -> None:
    """Copy every non-shard file (index, config, processor, tokenizer) unchanged.

    The tensor keys / shapes / dtypes are all preserved by the merge, so the base
    `model.safetensors.index.json` stays valid verbatim (no relocation, unlike the
    NVFP4 merge scripts). Directories (e.g. a `.cache/`) are skipped.
    """
    for f in base_dir.iterdir():
        if f.is_dir() or f.suffix == ".safetensors":
            continue
        dst = out_dir / f.name
        if not dst.exists():
            shutil.copy2(f, dst)


# ---------------------------------------------------------------------------
# Self test (CPU, no model files)
# ---------------------------------------------------------------------------

def self_test() -> int:
    import torch

    failures = 0
    torch.manual_seed(0)

    # 1. Known small case: W=0, so merged == scale * B@A.
    W = torch.zeros(4, 3, dtype=torch.bfloat16)
    A = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.bfloat16)   # (r=1, in=3)
    B = torch.tensor([[2.0], [0.0], [0.0], [0.0]], dtype=torch.bfloat16)  # (out=4, r=1)
    m = merged_weight(W, A, B, scale=3.0)
    expect = torch.zeros(4, 3, dtype=torch.bfloat16)
    expect[0, 0] = 6.0                                          # 3 * (2*1)
    if not torch.equal(m, expect):
        print("[self-test] FAIL: known-case merged weight wrong")
        failures += 1
    else:
        print("[self-test] known-case merge (W=0, 3*B@A): OK")

    # 2. Zero adapter (A=0) is a no-op: merged == W exactly.
    Wb = torch.randn(8, 6, dtype=torch.bfloat16)
    A0 = torch.zeros(2, 6, dtype=torch.bfloat16)
    B0 = torch.randn(8, 2, dtype=torch.bfloat16)
    if not torch.equal(merged_weight(Wb, A0, B0, scale=4.0), Wb):
        print("[self-test] FAIL: zero-A adapter changed the weight")
        failures += 1
    else:
        print("[self-test] zero-adapter no-op: OK")

    # 3. dtype preserved (bf16 in -> bf16 out).
    if merged_weight(Wb, torch.randn(2, 6), torch.randn(8, 2), 1.0).dtype != torch.bfloat16:
        print("[self-test] FAIL: merge did not preserve base dtype")
        failures += 1
    else:
        print("[self-test] dtype preserved: OK")

    # 4. bf16-only guard: an NVFP4 (weight_packed) target is refused.
    ct_keys = {"vision_tower.x.weight_packed", "vision_tower.x.weight_scale",
               "vision_tower.x.weight_global_scale"}
    try:
        assert_mergeable_target(ct_keys, "vision_tower.x.weight")
        print("[self-test] FAIL: NVFP4 target not refused")
        failures += 1
    except SystemExit:
        print("[self-test] NVFP4-target guard: OK")

    print(f"[self-test] {'PASS' if failures == 0 else f'FAIL ({failures} checks)'}")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-model-dir", type=Path,
                    help="NVFP4 VLM: 4-bit LLM backbone + bf16 vision tower/projector")
    ap.add_argument("--adapter-dir", "--lora-adapter-dir", dest="adapter_dir", type=Path,
                    help="PEFT adapter from a --train-target vision run")
    ap.add_argument("--out-dir", "--output-dir", dest="out_dir", type=Path,
                    help="destination for the merged VLM (required unless --dry-run)")
    ap.add_argument("--device", default="cpu",
                    help="device for the (tiny, bf16) merge math; 'cpu' is the default "
                         "and plenty -- the tower is <1 GB and there is no dequant")
    ap.add_argument("--family-config", type=Path, default=None,
                    help="explicit family spec (the --family-config escape hatch); "
                         "must carry the vision_* fields")
    ap.add_argument("--prefix-pair", action="append", default=None, metavar="MEM:DISK",
                    help="Override the tower prefix pairs with explicit MEM:DISK module-prefix "
                         "pairs (repeatable). Use to merge a NON-vision bf16 adapter, e.g. the "
                         "LLM half of a --train-target both split: "
                         "`--prefix-pair language_model.:language_model.`. Bypasses the "
                         "vision-family requirement; the adapter keys drive the mapping. The "
                         "merge is the same dequant-free bf16 delta-add (targets must be bf16).")
    ap.add_argument("--dry-run", action="store_true",
                    help="report adapter->base coverage + target storage, write nothing")
    ap.add_argument("--self-test", action="store_true",
                    help="CPU merge-math sanity check on synthetic tensors, no model files")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if not (args.base_model_dir and args.adapter_dir):
        ap.error("--base-model-dir and --adapter-dir are required (or use --self-test)")

    import torch
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    print(f"[merge] base    = {args.base_model_dir}")
    print(f"[merge] adapter = {args.adapter_dir}")
    print(f"[merge] out     = {args.out_dir}")
    print(f"[merge] device  = {device}")

    # Vision prefix pairs come from the family registry (so the merge translates
    # adapter keys with the same rule the vision trainer used at load time).
    model_type, family = resolve_family(
        args.base_model_dir, family_config=args.family_config
    )
    if args.prefix_pair:
        # Explicit override: merge a bf16 adapter under arbitrary module prefixes (the LLM half
        # of a --train-target both split -> `language_model.:language_model.`). The adapter keys
        # drive the mapping, so no vision-family requirement. Same bf16 merge downstream.
        prefix_pairs = []
        for pp in args.prefix_pair:
            if ":" not in pp:
                ap.error(f"--prefix-pair must be MEM:DISK, got {pp!r}")
            mem, disk = pp.split(":", 1)
            prefix_pairs.append((mem, disk))
        print(f"[merge] explicit prefix pairs (bf16, non-vision): {prefix_pairs}")
    else:
        if not family_supports_vision(family):
            raise SystemExit(
                f"model_type={model_type!r} declares no vision scope, so it has no "
                f"--train-target vision adapter to merge. Vision families: add "
                f"vision_peft_scope / vision_st_to_model to its FAMILIES entry (see "
                f"mistral3), or pass --family-config."
            )
        prefix_pairs = vision_prefix_pairs(family)
        print(f"[merge] family  = {model_type} (vision prefixes: "
              f"{[m for m, _ in prefix_pairs]})")

    lora_map, cfg, scale = load_adapter(args.adapter_dir, prefix_pairs)
    print(f"[merge] {len(lora_map)} vision LoRA targets, r={cfg['r']} "
          f"alpha={cfg['lora_alpha']} use_rslora={bool(cfg.get('use_rslora'))} scale={scale}")

    idx_path = args.base_model_dir / "model.safetensors.index.json"
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]

    # Coverage + storage guard (index-only, no shard reads yet).
    index_keys = set(weight_map)
    missing = [k for k in lora_map if k not in index_keys]
    if missing:
        raise SystemExit(
            f"{len(missing)} vision targets not found in the base index "
            f"(unquantized/misnamed): {sorted(missing)[:5]}"
        )
    for base_key in lora_map:
        assert_mergeable_target(index_keys, base_key)
    shards_hit = sorted({weight_map[k] for k in lora_map})
    print(f"[merge] coverage OK: {len(lora_map)} bf16 targets across "
          f"{len(shards_hit)} shard(s): {shards_hit}")

    if args.dry_run:
        for k in sorted(lora_map)[:10]:
            print(f"[dry-run] target: {k} -> shard {weight_map[k]}")
        if len(lora_map) > 10:
            print(f"[dry-run] ... and {len(lora_map) - 10} more")
        print(f"[dry-run] {len(set(weight_map.values())) - len(shards_hit)} shard(s) "
              f"would be copied byte-for-byte; {len(shards_hit)} rewritten. No files written.")
        return 0

    if args.out_dir is None:
        ap.error("--out-dir is required unless --dry-run/--self-test")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    replacements, stats = compute_replacements(
        lora_map, scale, args.base_model_dir, weight_map, device
    )
    stats_path = args.out_dir / "merge_stats.jsonl"
    with open(stats_path, "w") as f:
        for s in stats:
            f.write(json.dumps(s) + "\n")

    shard_summary = rewrite_shards(
        args.base_model_dir, args.out_dir, weight_map, replacements
    )
    copy_aux_files(args.base_model_dir, args.out_dir)

    worst = max(stats, key=lambda s: s["delta_to_base_ratio"]) if stats else None
    n_noop = sum(1 for s in stats if s["delta_to_base_ratio"] < 1e-4)
    manifest = {
        "merge_kind": "vision_bf16",
        "base_model_dir": str(args.base_model_dir),
        "adapter_dir": str(args.adapter_dir),
        "model_type": model_type,
        "scale_alpha_over_r": scale,
        "use_rslora": bool(cfg.get("use_rslora")),
        "n_targets": len(lora_map),
        "n_near_zero_updates": n_noop,
        "merge_dtype": stats[0]["dtype"] if stats else None,
        "merged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **shard_summary,
    }
    with open(args.out_dir / "merge_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[merge] complete: merged {len(lora_map)} bf16 vision target(s) "
          f"(dtype={manifest['merge_dtype']}), rewrote "
          f"{shard_summary['n_shards_rewritten']}/{shard_summary['n_shards_total']} "
          f"shard(s), copied {shard_summary['n_shards_copied']} byte-for-byte.")
    if worst is not None:
        print(f"[merge] largest delta/base ratio: {worst['delta_to_base_ratio']:.5f} "
              f"at {worst['key']}")
    if n_noop:
        print(f"[merge] WARN: {n_noop}/{len(lora_map)} targets are near-zero updates "
              f"(delta/base < 1e-4) -- check the adapter actually trained.")
    print(f"[merge] stats: {stats_path}")
    print(f"[merge] manifest: {args.out_dir / 'merge_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
