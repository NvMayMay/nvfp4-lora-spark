"""resolve_family: model_type -> (family_name, family_dict) dispatch.

Uses real config.json files copied verbatim from the two public NVFP4 checkpoints
(Qwen3.5-122B and Mistral-Small-4-119B) as fixtures. resolve_family only calls
AutoConfig.from_pretrained, which reads config.json on CPU; no weights are loaded.
"""
from __future__ import annotations

import pytest


def test_qwen_resolves_to_causal_lm(train_mod, fixtures_dir):
    model_type, family = train_mod.resolve_family(fixtures_dir / "qwen3_5_moe")
    assert model_type == "qwen3_5_moe"
    assert family["auto_class"] == "causal_lm"
    # Family registry surface the trainer relies on.
    assert family["expert_prefix"] == ("model.", "model.language_model.")
    assert family["freeze"] == ()


def test_mistral_resolves_to_image_text_to_text(train_mod, fixtures_dir):
    model_type, family = train_mod.resolve_family(fixtures_dir / "mistral3")
    assert model_type == "mistral3"
    assert family["auto_class"] == "image_text_to_text"
    assert family["expert_prefix"] == ("model.language_model.", "language_model.model.")
    # Multimodal towers are frozen for text-only training.
    assert family["freeze"] == ("vision_tower", "multi_modal_projector")


def test_families_registry_contents(train_mod):
    # Guard the registry keys so a refactor that renames or drops a family is caught.
    assert set(train_mod.FAMILIES) == {
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "mistral3",
        "mistral4",
        "nemotron_h",
    }
    for fam in train_mod.FAMILIES.values():
        assert fam["auto_class"] in ("causal_lm", "image_text_to_text")
        # expert_prefix is a (mem, st) pair for fused-3D MoE families, None for
        # per-expert-linear families (Nemotron).
        if fam["expert_prefix"] is not None:
            assert isinstance(fam["expert_prefix"], tuple) and len(fam["expert_prefix"]) == 2
        assert isinstance(fam["freeze"], tuple)
        # st_to_model=None declares a dynamic layout; the dynamic family stores
        # experts as per-module linears, so no fused-MoE expectations.
        if fam["st_to_model"] is None:
            assert fam["expert_prefix"] is None
            assert fam["moe_experts_class"] is None


def test_nemotron_resolves(train_mod, fixtures_dir):
    model_type, family = train_mod.resolve_family(fixtures_dir / "fp8_demoted")
    assert model_type == "nemotron_h"
    assert family["auto_class"] == "causal_lm"
    assert family["moe_experts_class"] is None
    assert family["st_to_model"] is None


def test_unsupported_model_type_raises_systemexit(train_mod, fixtures_dir):
    # `llama` is a model_type transformers recognizes (so AutoConfig.from_pretrained
    # succeeds) but which is NOT in FAMILIES. This is the path resolve_family is meant
    # to guard: it must raise SystemExit with the helpful "Add a FAMILIES entry" message.
    with pytest.raises(SystemExit) as exc:
        train_mod.resolve_family(fixtures_dir / "unsupported_family")
    msg = str(exc.value)
    assert "Unsupported model_type='llama'" in msg
    assert "FAMILIES entry" in msg
    assert "make_key_translator" in msg
