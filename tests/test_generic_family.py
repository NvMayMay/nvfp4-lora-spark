"""Generic-family fallback + --family-config escape hatch (nvfp4_lora.families).

An unregistered but structurally-standard NVFP4 checkpoint should be trainable via a
best-effort synthesized family (tagged UNVERIFIED, guarded downstream by strict-load +
coverage), instead of the old hard SystemExit. A multimodal-wrapped arch must still be
refused (its decoder needs an explicit rewrite). The strict default is preserved so
existing callers/tests are unaffected. Pure config.json reads: no torch, no weights.
"""
from __future__ import annotations

import json

import pytest

from nvfp4_lora import families
from nvfp4_lora.families import (_REQUIRED_FAMILY_KEYS, load_family_config,
                                 resolve_family, synthesize_generic_family)


def _cfg(d, model_type, arch):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": model_type,
                                               "architectures": [arch]}))
    return d


def test_strict_default_still_raises(tmp_path):
    # Preserve the fail-fast contract for callers that do not opt in.
    _cfg(tmp_path, "brandnew_moe", "BrandNewForCausalLM")
    with pytest.raises(SystemExit):
        resolve_family(tmp_path)


def test_generic_fallback_synthesizes_for_flat_causal_lm(tmp_path):
    _cfg(tmp_path, "brandnew", "BrandNewForCausalLM")
    model_type, fam = resolve_family(tmp_path, allow_generic=True)
    assert model_type == "brandnew"
    assert fam["_unverified"] is True and fam["_generic"] is True
    # Has every field the consumers read.
    for k in _REQUIRED_FAMILY_KEYS:
        assert k in fam
    assert fam["st_to_model"] is None            # rides the loader identity translator
    assert fam["moe_experts_class"] is None       # per-expert / dense, not fused-3D
    assert fam["auto_class"] == "causal_lm"


def test_generic_fallback_refuses_multimodal_wrapped(tmp_path):
    _cfg(tmp_path, "brandnew_vl", "BrandNewVLForConditionalGeneration")
    with pytest.raises(SystemExit) as e:
        resolve_family(tmp_path, allow_generic=True)
    assert "family-config" in str(e.value)


def test_registered_family_unaffected_by_generic(tmp_path):
    # A known model_type resolves to the registry entry, never the generic tag.
    _cfg(tmp_path, "qwen3", "Qwen3ForCausalLM")
    model_type, fam = resolve_family(tmp_path, allow_generic=True)
    assert model_type == "qwen3"
    assert "_unverified" not in fam
    assert fam is families.FAMILIES["qwen3"]


def _valid_family_json(p):
    p.write_text(json.dumps({
        "auto_class": "causal_lm",
        "expert_prefix": None,
        "peft_scope": r"^model\.layers\.",
        "freeze": [],
        "skip_st_prefixes": ["vision_tower."],
        "st_to_model": [["language_model.model.", "model.language_model."]],
        "meta_allowed_prefixes": [],
        "moe_experts_class": None,
    }))
    return p


def test_family_config_loads_and_coerces_tuples(tmp_path):
    fam = load_family_config(_valid_family_json(tmp_path / "family.json"))
    assert isinstance(fam["skip_st_prefixes"], tuple)
    assert fam["skip_st_prefixes"] == ("vision_tower.",)
    assert fam["st_to_model"] == (("language_model.model.", "model.language_model."),)
    assert fam["_source"].endswith("family.json")


def test_family_config_missing_keys_refused(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"auto_class": "causal_lm"}))  # missing the rest
    with pytest.raises(SystemExit) as e:
        load_family_config(p)
    assert "missing required keys" in str(e.value)


def test_family_config_wins_over_registry(tmp_path):
    # Even for a REGISTERED model_type, an explicit family_config takes precedence.
    _cfg(tmp_path, "qwen3", "Qwen3ForCausalLM")
    fam_path = _valid_family_json(tmp_path / "family.json")
    model_type, fam = resolve_family(tmp_path, family_config=fam_path)
    assert fam["_source"].endswith("family.json")
    assert fam is not families.FAMILIES["qwen3"]


def test_synthesize_direct_tags_note(tmp_path):
    _cfg(tmp_path, "brandnew", "BrandNewForCausalLM")
    fam = synthesize_generic_family(tmp_path)
    assert "brandnew" in fam["_note"]
