#!/usr/bin/env python3
"""Split a `--train-target both` adapter into a tower sub-adapter + an LLM sub-adapter.

A `both` run saves ONE unified native adapter whose `adapter_model.safetensors` holds LoRA
keys for BOTH halves: the bf16 vision tower/projector AND the LLM backbone. Neither existing
merge tool accepts that unified file -- `merge_vision_lora.adapter_key_to_base_key` raises on
a text-backbone key, and `merge_lora_into_nvfp4` would try to merge tower keys into a base
that has no such quantized target. This tool splits the unified adapter by SCOPE (read from
the `both` block that the trainer writes into `adapter_config.json`) into two standard
sub-adapters, each consumable by its own merge tool:

    <out>/tower/   -> scripts/merge_vision_lora.py                      (bf16 tower + projector)
    <out>/llm/     -> scripts/merge_vision_lora.py --prefix-pair ...    (bf16 LLM backbone)

v1 serve path (see docs/plans/train_target_both_plan.md, Phase 0): vLLM 0.22.1 cannot
runtime-LoRA the nemotron VLM (the multimodal wrapper does not declare SupportsLoRA), so BOTH
halves are MERGED and the result is served as a plain VLM. Both halves are bf16 (the tower,
and a nemotron `both` run targets bf16 q/k/v), so both merge via the SAME dequant-free
merge_vision_lora path -- the LLM half just needs an explicit `--prefix-pair` because it lives
under a non-vision module prefix. (merge_lora_into_nvfp4 is NOT used: it is single-backbone +
NVFP4-target only, and drops bf16 targets.) The merge ORDER matters -- tower first, then the
LLM against the tower-merged base:

    python scripts/merge_vision_lora.py --base-model-dir <BASE> \
        --adapter-dir <out>/tower --out-dir <BASE.tower>
    python scripts/merge_vision_lora.py --base-model-dir <BASE.tower> \
        --adapter-dir <out>/llm --out-dir <BASE.both> --prefix-pair language_model.:language_model.

Merging the LLM half is lossless because its targets are bf16 (nemotron q/k/v); FP8/NVFP4 LLM
targets would take a quantized-merge quality hit (a wrapper-patch runtime-LoRA path is the
quality-preserving alternative, deferred -- see the plan).

Pure-Python, no torch import at module top: the key classification is unit-testable without a
model, and the tensor copy uses safetensors directly.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# base_model.model.<module_path>.lora_{A,B}.weight  (the trainer's native save format)
_ADAPTER_PREFIX = "base_model.model."
_LORA_SUFFIX_RE = re.compile(r"\.lora_[AB]\.weight$")


def module_path_of(adapter_key: str) -> str:
    """The in-memory module path an adapter key targets (what the scope regexes match).

    `base_model.model.vision_model.blk.qkv.lora_A.weight` -> `vision_model.blk.qkv`
    `base_model.model.model.vision_tower.<...>.q_proj.lora_B.weight` -> `model.vision_tower.<...>.q_proj`
    """
    if not adapter_key.startswith(_ADAPTER_PREFIX):
        raise ValueError(
            f"adapter key {adapter_key!r} does not start with {_ADAPTER_PREFIX!r}; not a "
            f"native `both` adapter key")
    if not _LORA_SUFFIX_RE.search(adapter_key):
        raise ValueError(
            f"adapter key {adapter_key!r} is not a `.lora_A.weight`/`.lora_B.weight` tensor "
            f"(a `both` adapter should not carry expert-LoRA or other key shapes)")
    inner = adapter_key[len(_ADAPTER_PREFIX):]
    return _LORA_SUFFIX_RE.sub("", inner)


def scope_to_prefix(scope: str) -> str:
    """The literal module prefix an anchored scope regex matches.

    `^language_model\\.` -> `language_model.`  ;  `^model\\.language_model\\.` -> `model.language_model.`
    Used to build the `--prefix-pair` for merging the LLM half via merge_vision_lora.
    """
    s = scope.lstrip("^")
    return re.sub(r"\\(.)", r"\1", s)


def classify_keys(keys, vision_peft_scope: str, projector_scopes) -> tuple[list, list]:
    """Partition adapter keys into (tower_keys, llm_keys) by the both-config vision scopes.

    A key is a TOWER key iff its module path matches the vision tower scope OR any projector
    scope; everything else is an LLM key. Order of the returned lists follows input order.
    """
    vision_res = [re.compile(vision_peft_scope)] if vision_peft_scope else []
    vision_res += [re.compile(p) for p in (projector_scopes or ())]
    tower, llm = [], []
    for k in keys:
        mp = module_path_of(k)
        (tower if any(r.search(mp) for r in vision_res) else llm).append(k)
    return tower, llm


def _load_both_config(adapter_dir: Path) -> dict:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"no adapter_config.json in {adapter_dir}")
    cfg = json.loads(cfg_path.read_text())
    both = cfg.get("both")
    if cfg.get("train_target") != "both" or not both:
        raise SystemExit(
            f"{cfg_path} is not a `--train-target both` adapter (no `both` block / "
            f"train_target != both). This splitter only applies to both-adapters.")
    for req in ("vision_peft_scope", "vision_target_modules", "text_target_modules"):
        if req not in both:
            raise SystemExit(f"{cfg_path} `both` block missing required field {req!r}")
    return cfg


def _sub_config(cfg: dict, *, target_modules, base_model, extra: dict) -> dict:
    """A standard PEFT-shaped sub-adapter config (merge tools read r + lora_alpha)."""
    out = {
        "base_model_name_or_path": base_model,
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "r": cfg["r"], "lora_alpha": cfg["lora_alpha"],
        "lora_dropout": cfg.get("lora_dropout", 0.0),
        "bias": "none", "target_modules": list(target_modules),
        "inference_mode": True, "fan_in_fan_out": False,
    }
    out.update(extra)
    return out


def split_both_adapter(adapter_dir: str | Path, output_dir: str | Path) -> dict:
    """Split a unified both-adapter into <out>/tower and <out>/llm. Returns a summary dict."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    adapter_dir = Path(adapter_dir)
    output_dir = Path(output_dir)
    cfg = _load_both_config(adapter_dir)
    both = cfg["both"]

    files = sorted(adapter_dir.glob("adapter_model*.safetensors"))
    if not files:
        raise SystemExit(f"no adapter_model*.safetensors in {adapter_dir}")

    # Load every tensor (a both-adapter is small: LoRA A/B only).
    state: dict = {}
    for af in files:
        with safe_open(af, framework="pt") as sf:
            for k in sf.keys():
                state[k] = sf.get_tensor(k)

    tower_keys, llm_keys = classify_keys(
        state.keys(), both["vision_peft_scope"], both.get("projector_scopes", ()))

    # R6: a both-adapter MUST carry BOTH scopes -- a missing half means the run trained only
    # one side (or the scopes are wrong). Refuse loudly rather than emit a silent half-merge.
    if not tower_keys:
        raise SystemExit(
            "split refused: ZERO tower/projector keys matched the vision scope "
            f"({both['vision_peft_scope']!r} + projector {both.get('projector_scopes', ())}). "
            "This is not a valid both-adapter (or the scopes are wrong).")
    if not llm_keys:
        raise SystemExit(
            "split refused: ZERO LLM keys (every key matched the vision scope). This is a "
            "vision-only adapter, not a both-adapter -- use merge_vision_lora.py directly.")

    base_model = cfg.get("base_model_name_or_path", both.get("base_model_name_or_path"))
    tower_dir = output_dir / "tower"
    llm_dir = output_dir / "llm"
    for d in (tower_dir, llm_dir):
        d.mkdir(parents=True, exist_ok=True)

    save_file({k: state[k].contiguous() for k in tower_keys},
              str(tower_dir / "adapter_model.safetensors"))
    save_file({k: state[k].contiguous() for k in llm_keys},
              str(llm_dir / "adapter_model.safetensors"))

    tower_cfg = _sub_config(
        cfg, target_modules=both["vision_target_modules"], base_model=base_model,
        extra={"train_target": "vision", "include_projector": both.get("include_projector", True),
               "_split_from": "both", "_note": "tower/projector half; merge with merge_vision_lora.py"})
    # The LLM half must be merged AGAINST the tower-merged base (merge order matters); the base
    # pointer is advisory (merge_lora_into_nvfp4 takes --base-model-dir), but we record intent.
    llm_cfg = _sub_config(
        cfg, target_modules=both["text_target_modules"], base_model=base_model,
        extra={"train_target": "text", "_split_from": "both",
               "_note": "LLM half; merge with merge_lora_into_nvfp4.py AGAINST the tower-merged base"})
    (tower_dir / "adapter_config.json").write_text(json.dumps(tower_cfg, indent=2))
    (llm_dir / "adapter_config.json").write_text(json.dumps(llm_cfg, indent=2))

    summary = {
        "adapter_dir": str(adapter_dir), "output_dir": str(output_dir),
        "tower_keys": len(tower_keys), "llm_keys": len(llm_keys),
        "base_model": base_model,
        "tower_dir": str(tower_dir), "llm_dir": str(llm_dir),
        # Module prefix of the LLM half, for merge_vision_lora's --prefix-pair. Identity
        # (MEM==DISK) for families whose LLM in-memory path equals its on-disk path (nemotron);
        # a family whose st_to_model rewrites the LLM prefix (e.g. mistral's language_model.model.)
        # needs the DISK side adjusted to match its on-disk keys.
        "llm_prefix": scope_to_prefix(both["text_peft_scope"]),
    }
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter-dir", required=True, type=Path,
                    help="A `--train-target both` adapter dir (adapter_model.safetensors + "
                         "adapter_config.json carrying a `both` block).")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Destination; writes <output-dir>/tower and <output-dir>/llm.")
    args = ap.parse_args(argv)

    s = split_both_adapter(args.adapter_dir, args.output_dir)
    print(json.dumps(s, indent=2))
    p = s["llm_prefix"]
    print("\nNext (fully-merge, v1 serve path -- ORDER MATTERS):")
    print(f"  python scripts/merge_vision_lora.py --base-model-dir <BASE> \\")
    print(f"      --adapter-dir {s['tower_dir']} --out-dir <BASE.tower>")
    # The LLM half is bf16 (target q/k/v) -> same dequant-free merge as the tower, via
    # merge_vision_lora with an explicit --prefix-pair (NOT merge_lora_into_nvfp4, which is
    # single-backbone + NVFP4-target only). Merge it AGAINST the tower-merged base.
    print(f"  python scripts/merge_vision_lora.py --base-model-dir <BASE.tower> \\")
    print(f"      --adapter-dir {s['llm_dir']} --out-dir <BASE.both> --prefix-pair {p}:{p}")
    print("  # then: serve <BASE.both> as a plain VLM (NO --enable-lora; the nemotron VLM")
    print("  #       wrapper does not support runtime-LoRA on vLLM 0.22.1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
