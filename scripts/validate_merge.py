"""
P1.5 validation suite for a merged NVFP4 model.

Runs the standard validation checks on a merged checkpoint produced
by `scripts/merge_lora_into_nvfp4.py`:

  P1.5.1: Layerwise delta-to-quant-step audit (was done inline by merge
          script; this just aggregates merge_stats.jsonl).
  P1.5.2: Post-merge per-tensor cosine similarity + relative error report
          (also in merge_stats.jsonl; aggregate here).
  P1.5.5: Tokenizer / config / special-tokens byte-identity check vs base.
  P1.5.6: No-op guardrail: detect tensors with near-zero effective updates.
  P1.5.X: Coverage report - confirm count of merged tensors matches expected.

NOT done here (separate scripts, require running vLLM server):
  P1.5.3: Logit-level parity test on 100-200 prompts (needs vLLM server).
  P1.5.4: Scripted FT eval metric (needs vLLM server + eval data).

Usage:
  python scripts/validate_merge.py \\
      --base-model-dir /path/to/base \\
      --merged-model-dir /path/to/merged
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections import Counter
from pathlib import Path

from merge_lora_into_nvfp4 import adapter_key_to_base_key


# Files we expect to be byte-identical between base and merged dirs.
INTEGRITY_FILES = [
    "config.json",
    "configuration_nemotron_h.py",
    "modeling_nemotron_h.py",
    "generation_config.json",
    "hf_quant_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "__init__.py",
    "super_v3_reasoning_parser.py",
]


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(block_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def expected_shard_count(base_model_dir: Path) -> int:
    index_path = base_model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing base model safetensors index: {index_path}")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} does not contain a weight_map object")
    return len(set(weight_map.values()))


def adapter_manifest_values(adapter_dir: Path) -> tuple[float, int]:
    with open(adapter_dir / "adapter_config.json") as f:
        cfg = json.load(f)
    if "r" not in cfg or "lora_alpha" not in cfg:
        raise ValueError(f"adapter_config.json missing r or lora_alpha: {cfg}")
    alpha_over_r = float(cfg["lora_alpha"]) / float(cfg["r"])

    adapter_files = sorted(adapter_dir.glob("adapter_model*.safetensors"))
    if not adapter_files:
        raise FileNotFoundError(f"no adapter_model*.safetensors in {adapter_dir}")

    targets = set()
    for adapter_file in adapter_files:
        with open(adapter_file, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        for adapter_key in header:
            if adapter_key != "__metadata__":
                targets.add(adapter_key_to_base_key(adapter_key))
    return alpha_over_r, len(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-dir", required=True, type=Path)
    ap.add_argument("--merged-model-dir", required=True, type=Path)
    ap.add_argument(
        "--lora-adapter-dir", "--adapter-dir", dest="lora_adapter_dir", type=Path,
        help="Optional adapter directory used to verify manifest alpha/r and target count",
    )
    ap.add_argument(
        "--noop-warn-frac", default=0.10, type=float,
        help="Warn if more than this fraction of merged tensors are near-zero updates",
    )
    ap.add_argument(
        "--noop-fail-frac", default=0.50, type=float,
        help="Fail if more than this fraction are near-zero updates",
    )
    ap.add_argument(
        "--cosine-fail-threshold", default=0.99, type=float,
        help="Fail if any merged tensor has cosine below this",
    )
    args = ap.parse_args()

    print(f"[validate] base   = {args.base_model_dir}")
    print(f"[validate] merged = {args.merged_model_dir}")

    failures = []
    warnings_list = []

    # P1.5.5: integrity check on non-weight files
    print("\n=== P1.5.5: integrity check on non-weight files ===")
    for fname in INTEGRITY_FILES:
        base_f = args.base_model_dir / fname
        merged_f = args.merged_model_dir / fname
        if not base_f.exists():
            print(f"  [skip] {fname}: not in base")
            continue
        if not merged_f.exists():
            failures.append(f"missing in merged: {fname}")
            print(f"  [FAIL] {fname}: missing in merged")
            continue
        base_h = file_sha256(base_f)
        merged_h = file_sha256(merged_f)
        if base_h == merged_h:
            print(f"  [ok]   {fname}: bytes identical")
        else:
            failures.append(f"hash mismatch: {fname}")
            print(f"  [FAIL] {fname}: hash differs base={base_h[:12]} merged={merged_h[:12]}")

    # Load merge_manifest.json
    manifest_path = args.merged_model_dir / "merge_manifest.json"
    if not manifest_path.exists():
        failures.append("missing merge_manifest.json")
        print(f"\n[FAIL] no merge_manifest.json at {manifest_path}")
        sys.exit(1)
    with open(manifest_path) as f:
        manifest = json.load(f)

    # P1.5 coverage report
    print(f"\n=== Coverage report ===")
    n_shards = len(manifest["shards"])
    total_merged = sum(s["n_merged"] for s in manifest["shards"])
    total_passthrough = sum(s["n_passthrough"] for s in manifest["shards"])
    expected = manifest["n_lora_targets"]
    expected_shards = expected_shard_count(args.base_model_dir)
    print(f"  shards processed: {n_shards}/{expected_shards} (expected {expected_shards})")
    print(f"  tensors merged: {total_merged}")
    print(f"  tensors passthrough: {total_passthrough}")
    print(f"  expected LoRA targets in adapter: {expected}")
    if n_shards != expected_shards:
        failures.append(f"only {n_shards}/{expected_shards} shards processed")
    if total_merged != expected:
        failures.append(
            f"merged count {total_merged} != expected adapter LoRA targets {expected}"
        )
    else:
        print(f"  [ok]   merged count matches expected adapter LoRA targets")

    if args.lora_adapter_dir is None:
        print("  alpha_over_r / adapter consistency: SKIPPED (no --lora-adapter-dir)")
    else:
        try:
            adapter_alpha_over_r, adapter_targets = adapter_manifest_values(args.lora_adapter_dir)
            manifest_alpha_over_r = float(manifest["alpha_over_r"])
            print(
                "  alpha_over_r / adapter consistency: "
                f"manifest={manifest_alpha_over_r:g} adapter={adapter_alpha_over_r:g}"
            )
            if abs(manifest_alpha_over_r - adapter_alpha_over_r) > 1e-12:
                failures.append(
                    f"alpha_over_r mismatch: manifest {manifest_alpha_over_r:g} "
                    f"!= adapter {adapter_alpha_over_r:g}"
                )
                print("  [FAIL] alpha_over_r does not match adapter_config.json")
            else:
                print("  [ok]   alpha_over_r matches adapter_config.json")

            print(f"  adapter unique translated LoRA targets: {adapter_targets}")
            if expected != adapter_targets:
                failures.append(
                    f"manifest n_lora_targets {expected} != adapter translated targets {adapter_targets}"
                )
                print("  [FAIL] manifest n_lora_targets does not match adapter")
            else:
                print("  [ok]   manifest n_lora_targets matches adapter")
        except Exception as e:
            failures.append(f"adapter consistency check failed: {e!r}")
            print(f"  [FAIL] adapter consistency check failed: {e!r}")

    # P1.5.1 + P1.5.2 + P1.5.6: per-tensor stats analysis
    stats_path = args.merged_model_dir / manifest["stats_log_path"]
    if not stats_path.exists():
        failures.append(f"missing stats log {stats_path}")
        print(f"\n[FAIL] no per-tensor stats at {stats_path}")
        sys.exit(1)

    print(f"\n=== P1.5.1 + P1.5.2 + P1.5.6 per-tensor stats ===")
    stats = []
    with open(stats_path) as f:
        for line in f:
            stats.append(json.loads(line))
    print(f"  total stats entries: {len(stats)}")

    if not stats:
        failures.append("stats log is empty")
        sys.exit(1)

    deltas = [s["delta_abs_mean"] for s in stats]
    delta_to_base = [s["delta_to_base_ratio"] for s in stats]
    cosines = [s["merge_cosine"] for s in stats]
    rel_errs = [s["merge_relative_error"] for s in stats]

    def pctile(xs, q):
        sx = sorted(xs)
        i = int(len(sx) * q)
        return sx[min(i, len(sx) - 1)]

    print(f"\n  delta_abs_mean      p50={pctile(deltas, 0.5):.2e} p99={pctile(deltas, 0.99):.2e}")
    print(f"  delta_to_base_ratio p50={pctile(delta_to_base, 0.5):.4f} p99={pctile(delta_to_base, 0.99):.4f}")
    print(f"  merge_cosine        p01={pctile(cosines, 0.01):.6f} p50={pctile(cosines, 0.5):.6f}")
    print(f"  merge_relative_err  p50={pctile(rel_errs, 0.5):.4f} p99={pctile(rel_errs, 0.99):.4f}")

    # P1.5.6 no-op guardrail: count tensors where delta_to_base_ratio < 1e-4
    near_zero_threshold = 1e-4
    n_noop = sum(1 for r in delta_to_base if r < near_zero_threshold)
    noop_frac = n_noop / len(delta_to_base)
    print(f"\n  near-zero updates (delta/base < {near_zero_threshold}): {n_noop}/{len(delta_to_base)} = {noop_frac*100:.2f}%")
    if noop_frac > args.noop_fail_frac:
        failures.append(f"no-op fraction {noop_frac*100:.2f}% > fail threshold {args.noop_fail_frac*100}%")
    elif noop_frac > args.noop_warn_frac:
        warnings_list.append(f"no-op fraction {noop_frac*100:.2f}% > warn threshold {args.noop_warn_frac*100}%")
    else:
        print(f"  [ok]   no-op fraction within tolerance")

    # P1.5.2 cosine failure check
    n_bad_cosine = sum(1 for c in cosines if c < args.cosine_fail_threshold)
    if n_bad_cosine > 0:
        failures.append(f"{n_bad_cosine} tensors have merge_cosine < {args.cosine_fail_threshold}")
        print(f"  [FAIL] {n_bad_cosine} tensors have merge_cosine below threshold")
    else:
        print(f"  [ok]   all merge_cosine values >= {args.cosine_fail_threshold}")

    # Final summary
    print("\n=== FINAL ===")
    if warnings_list:
        print(f"WARNINGS ({len(warnings_list)}):")
        for w in warnings_list:
            print(f"  - {w}")
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
