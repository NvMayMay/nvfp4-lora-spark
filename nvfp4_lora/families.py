"""Single source of truth for per-family model knowledge.

Every component that needs to know how a model family lays out its checkpoint
(the unified trainer, the loader's key translator, the fused-3D MoE replacer,
the checkpoint inspector, and the merge scripts) reads from this registry.
Adding support for a new NVFP4 family should mean adding ONE entry here plus
tests, not editing loader/trainer/merge bodies.

Registry fields, per `config.json` `model_type`:

  auto_class            which transformers Auto* builds the right text-trainable
                        graph: "causal_lm" or "image_text_to_text"
  expert_prefix         (in_memory_prefix, safetensors_prefix) for the text
                        backbone. Used for routed-expert key assembly AND for
                        translating PEFT adapter keys to on-disk base keys at
                        merge time (the two are the same translation).
  peft_scope            regex prefix anchoring PEFT target_modules to the text
                        backbone (so multimodal towers can never match a bare
                        suffix)
  freeze                submodules of model.model to freeze (multimodal towers)
  skip_st_prefixes      safetensors key prefixes that are intentionally never
                        loaded for text-only training (vision tower, projector,
                        MTP speculation layers). This is the explicit allowlist
                        for "tensor on disk, absent in the training graph".
  st_to_model           ordered (st_prefix, model_prefix) rewrite rules mapping
                        safetensors keys to in-memory attribute paths; first
                        match wins, unmatched keys pass through verbatim.
                        None means the layout cannot be expressed statically
                        and the loader's dynamic fallback translator applies
                        (Nemotron: Nano materializes the backbone as
                        `backbone.*` but Super as `model.*` under the SAME
                        model_type, so the prefix is probed from the live
                        module tree)
  meta_allowed_prefixes in-memory parameter-path prefixes allowed to remain on
                        the meta device after loading (text-only training never
                        materializes the multimodal towers). Everything else on
                        meta after load is a load bug and fails the no-meta
                        assertion.
  moe_experts_class     HF module class name of the fused-3D routed-experts
                        block this family uses (replaced by NVFP4Experts3D),
                        or None if the family has no supported fused-3D MoE
"""
from __future__ import annotations

import json
from pathlib import Path

FAMILIES: dict[str, dict] = {
    "qwen3_5_moe": {
        "auto_class": "causal_lm",
        "expert_prefix": ("model.", "model.language_model."),
        "peft_scope": r"^model\.layers\.",
        "freeze": (),
        "skip_st_prefixes": ("model.visual.",),
        "st_to_model": (("model.language_model.", "model."),),
        "meta_allowed_prefixes": (),
        "moe_experts_class": "Qwen3_5MoeExperts",
    },
    "mistral3": {
        "auto_class": "image_text_to_text",
        "expert_prefix": ("model.language_model.", "language_model.model."),
        "peft_scope": r"^model\.language_model\.",
        "freeze": ("vision_tower", "multi_modal_projector"),
        "skip_st_prefixes": ("vision_tower.", "multi_modal_projector."),
        "st_to_model": (
            ("language_model.model.", "model.language_model."),
            ("language_model.lm_head.", "lm_head."),
        ),
        "meta_allowed_prefixes": ("model.vision_tower.", "model.multi_modal_projector."),
        "moe_experts_class": "Mistral4NaiveMoe",
    },
    # Nemotron-3 Nano/Super (the original v1.0 family). Routed experts are
    # per-expert NVFP4 nn.Linear modules (no fused-3D container), so
    # expert_prefix and moe_experts_class are None and replace_nvfp4_modules
    # handles everything. Key translation is dynamic (st_to_model=None): the
    # checkpoint stores `backbone.*` but Nano materializes it as `backbone.*`
    # while Super materializes it as `model.*`, decided by the live module
    # tree, so the loader's fallback heuristic does the mapping. `mtp.*`
    # (Multi-Token Prediction speculation layers, serve-only) is skipped
    # there as well.
    "nemotron_h": {
        "auto_class": "causal_lm",
        "expert_prefix": None,
        "peft_scope": r"^(model|backbone)\.layers\.",
        "freeze": (),
        "skip_st_prefixes": ("mtp.",),
        "st_to_model": None,
        "meta_allowed_prefixes": (),
        "moe_experts_class": None,
    },
    # GLM-4.5-Air (106B-A12B). The checkpoint stores routed experts per-expert
    # (model.layers.N.mlp.experts.E.{gate,up,down}_proj), but transformers
    # materializes them as a FUSED-3D block (Glm4MoeNaiveMoe: gate_up_proj +
    # down_proj batched over experts) — structurally identical to
    # Mistral4NaiveMoe. So this is the fused-3D path: replace_moe_experts_with_
    # nvfp4_3d swaps Glm4MoeNaiveMoe -> NVFP4Experts3D and assemble_nvfp4_
    # experts3d_batched gathers the per-expert NVFP4 tensors into the fused
    # buffers (split_gate_up_scales is auto-probed from the shards). Text-only
    # causal LM, so in-memory and on-disk share the `model.` backbone prefix:
    # expert_prefix=("model.","model.") and st_to_model=None routes non-expert
    # weights through the loader's dynamic heuristic (identity translation,
    # safetensors_prefix==model_prefix=="model"). The router gate
    # (mlp.gate.weight, BF16), shared expert and q/k/v attention biases load as
    # frozen non-NVFP4 weights. No MTP/visual tensors to skip.
    "glm4_moe": {
        "auto_class": "causal_lm",
        "expert_prefix": ("model.", "model."),
        "peft_scope": r"^model\.layers\.",
        "freeze": (),
        "skip_st_prefixes": (),
        "st_to_model": None,
        "meta_allowed_prefixes": (),
        "moe_experts_class": "Glm4MoeNaiveMoe",
    },
    # Qwen3 dense (e.g. Qwen3-32B-NVFP4): a plain Qwen3ForCausalLM, no MoE, standard
    # attention. `model.layers.*` on disk and in memory, so st_to_model=None uses the
    # loader's dynamic identity translator; no fused experts (moe_experts_class=None).
    # NVFP4 attention + NVFP4 dense MLP, so q/k/v/o + gate/up/down all train natively.
    "qwen3": {
        "auto_class": "causal_lm",
        "expert_prefix": None,
        "peft_scope": r"^model\.layers\.",
        "freeze": (),
        "skip_st_prefixes": (),
        "st_to_model": None,
        "meta_allowed_prefixes": (),
        "moe_experts_class": None,
    },
    # Llama dense (e.g. Llama-3.1-8B-Instruct-NVFP4): a plain LlamaForCausalLM, no MoE,
    # standard attention. Same on-disk/in-memory `model.layers.*` layout as qwen3 dense,
    # so st_to_model=None uses the identity translator and moe_experts_class=None. NVFP4
    # attention + NVFP4 dense MLP, so q/k/v/o + gate/up/down all train natively.
    "llama": {
        "auto_class": "causal_lm",
        "expert_prefix": None,
        "peft_scope": r"^model\.layers\.",
        "freeze": (),
        "skip_st_prefixes": (),
        "st_to_model": None,
        "meta_allowed_prefixes": (),
        "moe_experts_class": None,
    },
}

# These share the exact checkpoint layout of their base entry (qwen3_5_moe_text is the
# text-only config split of qwen3_5_moe; mistral4 is the v4 point release of mistral3),
# so they alias to ONE object rather than a copied literal that could silently drift.
FAMILIES["qwen3_5_moe_text"] = FAMILIES["qwen3_5_moe"]
FAMILIES["mistral4"] = FAMILIES["mistral3"]


def model_type_from_config(model_dir: str | Path) -> str | None:
    """Read `model_type` straight from config.json.

    Deliberately does NOT go through AutoConfig: this works even when the
    installed transformers version does not know the model type, and it keeps
    the inspector and the CPU test suite free of a transformers dependency for
    family resolution.
    """
    cfg_path = Path(model_dir) / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    return cfg.get("model_type")


# Best-effort defaults for an unregistered but structurally-standard NVFP4
# checkpoint: a flat causal-LM whose backbone is `model.layers.*` (or nemotron's
# `backbone.layers.*`) on disk and in memory. st_to_model=None routes non-expert
# weights through the loader's dynamic identity translator; moe_experts_class=None
# means any routed experts are handled per-expert by replace_nvfp4_modules (like
# nemotron_h / qwen3 / llama). This deliberately does NOT cover multimodal-wrapped
# bases (they need an explicit st_to_model rewrite) or unregistered FUSED-3D MoE
# blocks (they need an HF class name); both are caught downstream by the strict-load
# / coverage / no-meta gates, which fail fast rather than train a silent mismatch.
_GENERIC_FAMILY_DEFAULTS: dict = {
    "auto_class": "causal_lm",
    "expert_prefix": None,
    "peft_scope": r"^(model|backbone)\.layers\.",
    "freeze": (),
    "skip_st_prefixes": (),
    "st_to_model": None,
    "meta_allowed_prefixes": (),
    "moe_experts_class": None,
}

# The fields every family entry (registered or user-supplied) must define.
_REQUIRED_FAMILY_KEYS = (
    "auto_class", "expert_prefix", "peft_scope", "freeze", "skip_st_prefixes",
    "st_to_model", "meta_allowed_prefixes", "moe_experts_class",
)


def synthesize_generic_family(model_dir: str | Path) -> dict:
    """Best-effort family for an unregistered flat causal-LM NVFP4 checkpoint.

    Refuses (SystemExit, with a --family-config hint) for multimodal-wrapped
    architectures, whose decoder lives under a `language_model.` prefix and needs an
    explicit st_to_model rewrite the generic identity mapping cannot supply. The
    returned dict is tagged `_unverified` so callers can warn + stamp provenance.
    """
    cfg_path = Path(model_dir) / "config.json"
    cfg = {}
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError):
        pass
    arch = (cfg.get("architectures") or [None])[0] or ""
    if arch.endswith("ForConditionalGeneration"):
        raise SystemExit(
            f"model_type={cfg.get('model_type')!r} is an unregistered multimodal-wrapped "
            f"architecture ({arch}); the generic fallback only handles flat causal-LM "
            f"layouts. Supply an explicit --family-config family.json describing the "
            f"wrapped st_to_model rewrite (e.g. [[\"language_model.model.\", "
            f"\"model.language_model.\"]]) and peft_scope."
        )
    fam = dict(_GENERIC_FAMILY_DEFAULTS)
    fam["_generic"] = True
    fam["_unverified"] = True
    fam["_note"] = (f"synthesized generic family for model_type={cfg.get('model_type')!r} "
                    f"arch={arch!r}; verify with a short train + coverage check")
    return fam


def load_family_config(path: str | Path) -> dict:
    """Load a user-supplied family spec (the --family-config escape hatch).

    Lets a user onboard a model without editing library source. Validates that every
    required field is present and coerces list fields to the tuples consumers expect.
    """
    obj = json.loads(Path(path).read_text())
    missing = [k for k in _REQUIRED_FAMILY_KEYS if k not in obj]
    if missing:
        raise SystemExit(
            f"--family-config {path}: missing required keys {missing}. "
            f"Required: {list(_REQUIRED_FAMILY_KEYS)}."
        )
    for k in ("freeze", "skip_st_prefixes", "meta_allowed_prefixes"):
        obj[k] = tuple(obj[k]) if obj[k] is not None else ()
    if obj["st_to_model"] is not None:
        obj["st_to_model"] = tuple(tuple(r) for r in obj["st_to_model"])
    if obj["expert_prefix"] is not None:
        obj["expert_prefix"] = tuple(obj["expert_prefix"])
    obj["_source"] = str(path)
    return obj


def resolve_family(model_dir: str | Path, *, allow_generic: bool = False,
                   family_config: str | Path | None = None) -> tuple[str, dict]:
    """Map a checkpoint directory to (model_type, family registry entry).

    Resolution order: an explicit `family_config` wins; then the registry; then, only
    if `allow_generic`, a best-effort synthesized family for a flat causal-LM checkpoint
    (tagged `_unverified`). Otherwise raises SystemExit with the porting options. The
    default (allow_generic=False, no family_config) preserves the strict fail-fast.
    """
    model_type = model_type_from_config(model_dir)
    if family_config is not None:
        return (model_type or "custom", load_family_config(family_config))
    fam = FAMILIES.get(model_type)
    if fam is not None:
        return model_type, fam
    if allow_generic:
        return model_type, synthesize_generic_family(model_dir)
    raise SystemExit(
        f"Unsupported model_type={model_type!r}. Known: {sorted(FAMILIES)}. Options: "
        f"add a FAMILIES entry in nvfp4_lora/families.py; pass --family-config family.json; "
        f"or re-run with --allow-unverified-family for a best-effort flat causal-LM mapping "
        f"(guarded by the strict-load / coverage gates). Run "
        f"scripts/inspect_nvfp4_checkpoint.py on the checkpoint first to see its layout."
    )


def make_family_translator(fam: dict):
    """Build `translate(safetensors_key) -> model_path | None` from registry data.

    None means "intentionally skipped" (multimodal tower / MTP); unmatched keys
    pass through verbatim (lm_head etc.).
    """
    skip_prefixes = tuple(fam["skip_st_prefixes"])
    rules = tuple(fam["st_to_model"])

    def translate(key: str):
        for skip in skip_prefixes:
            if key.startswith(skip):
                return None
        for st_prefix, model_prefix in rules:
            if key.startswith(st_prefix):
                return model_prefix + key[len(st_prefix):]
        return key

    return translate


def translator_log_prefixes(fam: dict) -> tuple[str, str]:
    """(safetensors_prefix, model_prefix) of the primary rewrite rule, without
    trailing dots, for log lines."""
    st_prefix, model_prefix = fam["st_to_model"][0]
    return st_prefix.rstrip("."), model_prefix.rstrip(".")


def adapter_key_to_base_prefix(akey: str, mem_prefix: str, st_prefix: str,
                               adapter_prefix: str = "base_model.model.") -> tuple[str, str]:
    """Map a PEFT adapter tensor key to (on-disk base module prefix, 'A'|'B').

    The translation is the family's text-backbone prefix swap, i.e. exactly the
    inverse of what the trainer's loader does at load time:

      qwen3_5 (mem "model.", st "model.language_model."):
        base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight
            -> ("model.language_model.layers.3.self_attn.q_proj", "A")
      mistral3/4 (mem "model.language_model.", st "language_model.model."):
        base_model.model.model.language_model.layers.0.mlp.experts.0.gate_proj.lora_B.weight
            -> ("language_model.model.layers.0.mlp.experts.0.gate_proj", "B")

    Keys already carrying the on-disk prefix pass through unchanged.
    """
    import re

    if not akey.startswith(adapter_prefix):
        raise ValueError(f"adapter key {akey!r} does not start with {adapter_prefix!r}")
    tail = akey[len(adapter_prefix):]
    m = re.search(r"\.lora_(?P<side>[AB])\.weight$", tail)
    if m is None:
        raise ValueError(f"adapter key {akey!r} is not a lora_A/lora_B weight")
    prefix = tail[: m.start()]
    if prefix.startswith(st_prefix):
        pass  # already in on-disk form
    elif prefix.startswith(mem_prefix):
        prefix = st_prefix + prefix[len(mem_prefix):]
    else:
        raise ValueError(
            f"adapter key {akey!r} has unrecognized module path {prefix!r}; "
            f"expected it to start with {mem_prefix!r} (in-memory) or "
            f"{st_prefix!r} (on-disk)"
        )
    return prefix, m.group("side")
