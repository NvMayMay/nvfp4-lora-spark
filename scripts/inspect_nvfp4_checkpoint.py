#!/usr/bin/env python3
"""Inspect an NVFP4 checkpoint and report what this repo can do with it.

THE first command to run on any new checkpoint, before training or merging:

    python scripts/inspect_nvfp4_checkpoint.py /path/to/model
    python scripts/inspect_nvfp4_checkpoint.py /path/to/model \\
        --target-modules q_proj,k_proj,v_proj,o_proj
    python scripts/inspect_nvfp4_checkpoint.py /path/to/model --json

Reads ONLY config.json and model.safetensors.index.json (no weights, no GPU,
runs in seconds even for 100B+ checkpoints) and reports:

  * model_type and whether it maps to a supported family
  * per-module storage census: ModelOpt NVFP4, compressed-tensors NVFP4,
    FP8 per-tensor, plain BF16
  * per-suffix coverage incl. layer-level gaps (partial quantization)
  * routed-expert (MoE) topology and whether the fused-3D path supports it
  * for --target-modules: the exact LoRA mechanism a training run would use
    (native NVFP4 / PEFT), or the precise reason it would be rejected

With --deep (and shard files present) it additionally reads the per-expert
gate/up per-tensor-scale scalars and reports whether they are equal (the
fused gate_up fast path) or differ (handled via split gate/up storage,
selected automatically by the trainer).

Exit codes: 0 = inspected fine (and targets, if given, are trainable);
2 = a requested --target-modules set would be rejected; 1 = bad input.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nvfp4_lora.families import FAMILIES, model_type_from_config  # noqa: E402
from nvfp4_lora.loader import (  # noqa: E402
    classify_module_storage,
    decide_lora_mode,
    list_weight_module_prefixes,
)

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"^(?P<moe>.+\.experts)\.(?P<idx>\d+)\.(?P<proj>[A-Za-z0-9_]+)$")


def _num_layers_from_config(cfg: dict) -> int | None:
    for c in (cfg, cfg.get("text_config") or {}):
        if isinstance(c, dict) and "num_hidden_layers" in c:
            return int(c["num_hidden_layers"])
    return None


def build_report(model_dir: Path, target_suffixes: list[str] | None, deep: bool) -> dict:
    cfg_path = model_dir / "config.json"
    idx_path = model_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        raise SystemExit(f"no model.safetensors.index.json under {model_dir}")
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    weight_map = json.loads(idx_path.read_text())["weight_map"]
    keys = set(weight_map.keys())

    model_type = cfg.get("model_type")
    family = FAMILIES.get(model_type)
    report: dict = {
        "model_dir": str(model_dir),
        "model_type": model_type,
        "family_supported": family is not None,
        "known_families": sorted(FAMILIES),
        "num_layers": _num_layers_from_config(cfg),
        "n_tensors": len(weight_map),
    }
    qc = cfg.get("quantization_config") or {}
    if qc:
        report["quant_config"] = {
            "quant_method": qc.get("quant_method"),
            "format": qc.get("format"),
            "n_ignore_rules": len(qc.get("ignore", [])),
        }

    # ---- storage census over every weight-owning module -------------------
    prefixes = list_weight_module_prefixes(keys)
    census: dict[str, int] = defaultdict(int)
    by_suffix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    suffix_layers: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for p in sorted(prefixes):
        cls = classify_module_storage(keys, p)
        census[cls] += 1
        suffix = p.rsplit(".", 1)[-1]
        by_suffix[suffix][cls] += 1
        m = _LAYER_RE.search(p)
        if m:
            suffix_layers[suffix][cls].add(int(m.group(1)))
    report["storage_census"] = dict(census)
    report["suffixes"] = {
        s: {
            "counts": dict(c),
            "layers": {cls: len(v) for cls, v in suffix_layers[s].items()},
        }
        for s, c in sorted(by_suffix.items())
    }

    # ---- MoE topology ------------------------------------------------------
    moe_blocks: dict[str, dict] = {}
    for p in prefixes:
        m = _EXPERT_RE.match(p)
        if not m:
            continue
        blk = moe_blocks.setdefault(
            m.group("moe"), {"max_expert": -1, "projections": set(), "classes": set()}
        )
        blk["max_expert"] = max(blk["max_expert"], int(m.group("idx")))
        blk["projections"].add(m.group("proj"))
        blk["classes"].add(classify_module_storage(keys, p))
    if moe_blocks:
        n_experts = sorted({b["max_expert"] + 1 for b in moe_blocks.values()})
        projections = sorted(set().union(*(b["projections"] for b in moe_blocks.values())))
        classes = sorted(set().union(*(b["classes"] for b in moe_blocks.values())))
        gate_up_down = {"gate_proj", "up_proj", "down_proj"}.issubset(set(projections))
        report["moe"] = {
            "n_expert_blocks": len(moe_blocks),
            "experts_per_block": n_experts,
            "projections": projections,
            "storage_classes": classes,
            "per_expert_keys": True,
            "fused3d_supported": (
                family is not None
                and family.get("moe_experts_class") is not None
                and gate_up_down
                and classes in (["nvfp4_ct"], ["nvfp4_modelopt"])
            ),
        }
    else:
        report["moe"] = None

    # ---- deep check: gate/up global-scale equality (topology v1) -----------
    if deep and moe_blocks:
        try:
            import safetensors
        except ImportError:
            report["deep_gate_up_scale_check"] = "skipped (safetensors not installed)"
        else:
            # Collect every gate/up per-tensor-scale key (suffix depends on the
            # storage format), group by shard so each shard is opened once,
            # then compare per expert. A mismatch is handled at load time via
            # split gate/up storage, so this is informational, not a failure.
            pairs: list[tuple[str, str]] = []
            for moe, blk in moe_blocks.items():
                for i in range(blk["max_expert"] + 1):
                    for g_suffix in ("weight_global_scale", "weight_scale_2"):
                        g = f"{moe}.{i}.gate_proj.{g_suffix}"
                        u = f"{moe}.{i}.up_proj.{g_suffix}"
                        if g in weight_map and u in weight_map:
                            pairs.append((g, u))
                            break
            by_shard: dict[str, list[str]] = defaultdict(list)
            for g, u in pairs:
                by_shard[weight_map[g]].append(g)
                by_shard[weight_map[u]].append(u)
            scales: dict[str, float] = {}
            missing_shards = False
            for shard, shard_keys in by_shard.items():
                shard_path = model_dir / shard
                if not shard_path.exists():
                    missing_shards = True
                    continue
                with safetensors.safe_open(str(shard_path), framework="pt", device="cpu") as f:
                    for k in shard_keys:
                        scales[k] = float(f.get_tensor(k).reshape(-1)[0])
            checked = 0
            mismatches: list[str] = []
            for g, u in pairs:
                if g in scales and u in scales:
                    checked += 1
                    if scales[g] != scales[u]:
                        mismatches.append(g.rsplit(".gate_proj.weight_global_scale", 1)[0])
            report["deep_gate_up_scale_check"] = {
                "experts_checked": checked,
                "mismatched": len(mismatches),
                "mismatch_examples": mismatches[:5],
                "shards_missing": missing_shards,
            }

    # ---- target verdict -----------------------------------------------------
    if target_suffixes:
        try:
            mode, coverage = decide_lora_mode(model_dir, target_suffixes)
            report["target_verdict"] = {"ok": True, "mode": mode, "coverage": coverage}
        except SystemExit as e:
            report["target_verdict"] = {"ok": False, "reason": str(e)}

    return report


def print_human(report: dict) -> None:
    p = print
    p(f"checkpoint : {report['model_dir']}")
    fam = "supported" if report["family_supported"] else (
        f"NOT in the family registry (known: {', '.join(report['known_families'])})"
    )
    p(f"model_type : {report['model_type']!r} - {fam}")
    if report.get("num_layers") is not None:
        p(f"layers     : {report['num_layers']}")
    if "quant_config" in report:
        qcfg = report["quant_config"]
        p(f"quant_cfg  : method={qcfg['quant_method']} format={qcfg['format']} "
          f"ignore_rules={qcfg['n_ignore_rules']}")
    p(f"tensors    : {report['n_tensors']}")
    p("")
    p("storage census (weight-owning modules):")
    for cls, n in sorted(report["storage_census"].items()):
        p(f"  {cls:16s} {n}")
    p("")
    p("per-suffix coverage (counts by storage class; #layers in parens):")
    for suffix, info in report["suffixes"].items():
        parts = []
        for cls, n in sorted(info["counts"].items()):
            nl = info["layers"].get(cls)
            parts.append(f"{cls}={n}" + (f" ({nl} layers)" if nl else ""))
        flag = " <-- MIXED" if len({c for c in info["counts"]
                                    if c in ("nvfp4_ct", "nvfp4_modelopt", "bf16", "fp8")}) > 1 else ""
        p(f"  {suffix:24s} {', '.join(parts)}{flag}")
    p("")
    moe = report.get("moe")
    if moe:
        p("routed experts (MoE):")
        p(f"  blocks={moe['n_expert_blocks']} experts_per_block={moe['experts_per_block']} "
          f"projections={moe['projections']}")
        p(f"  storage={moe['storage_classes']} fused3d_supported={moe['fused3d_supported']}")
    else:
        p("routed experts (MoE): none detected (dense model)")
    deep = report.get("deep_gate_up_scale_check")
    if deep is not None:
        if isinstance(deep, dict):
            if deep["mismatched"] == 0:
                verdict = "OK (fused gate_up fast path applies)"
            else:
                verdict = (f"{deep['mismatched']} differ (handled via split gate/up "
                           f"storage; the trainer selects it automatically)")
            p(f"  gate/up per-tensor-scale equality: {deep['experts_checked']} experts checked, {verdict}")
            for ex in deep["mismatch_examples"]:
                p(f"    differs: {ex}")
        else:
            p(f"  gate/up global-scale check: {deep}")
    tv = report.get("target_verdict")
    if tv is not None:
        p("")
        if tv["ok"]:
            p(f"target verdict: OK - LoRA mechanism = {tv['mode']}")
            for suffix, info in tv["coverage"]["inventory"].items():
                p(f"  {suffix}: {info['counts']}")
        else:
            p("target verdict: REJECTED")
            p(tv["reason"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--target-modules", default=None,
                    help="Comma-separated suffixes to evaluate as LoRA targets")
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    ap.add_argument("--deep", action="store_true",
                    help="also read per-expert global scales from the shards and "
                         "verify the fused gate/up equal-scale assumption")
    args = ap.parse_args()

    targets = ([t.strip() for t in args.target_modules.split(",") if t.strip()]
               if args.target_modules else None)
    report = build_report(args.model_dir, targets, deep=args.deep)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human(report)
    tv = report.get("target_verdict")
    return 2 if (tv is not None and not tv["ok"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
