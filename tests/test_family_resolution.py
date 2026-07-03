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
        "glm4_moe",
        "qwen3",
        "llama",
    }
    for fam in train_mod.FAMILIES.values():
        assert fam["auto_class"] in ("causal_lm", "image_text_to_text")
        # A family either declares a fused-3D routed-MoE (an expert_prefix pair AND a
        # moe_experts_class) or it is per-module/dense (both None); the two fields must
        # agree. Nemotron is per-module (both None); GLM is fused-3D with an identity
        # model.->model. prefix.
        has_fused_moe = fam["moe_experts_class"] is not None
        assert (fam["expert_prefix"] is not None) == has_fused_moe
        if fam["expert_prefix"] is not None:
            assert isinstance(fam["expert_prefix"], tuple) and len(fam["expert_prefix"]) == 2
        assert isinstance(fam["freeze"], tuple)
        # st_to_model is either a static (st_prefix, model_prefix) rewrite ruleset or
        # None, which selects the loader's dynamic translator (Nemotron's probed
        # backbone prefix, or GLM's identity map). Independent of the MoE structure.
        assert fam["st_to_model"] is None or isinstance(fam["st_to_model"], tuple)


def test_nemotron_resolves(train_mod, fixtures_dir):
    model_type, family = train_mod.resolve_family(fixtures_dir / "fp8_demoted")
    assert model_type == "nemotron_h"
    assert family["auto_class"] == "causal_lm"
    assert family["moe_experts_class"] is None
    assert family["st_to_model"] is None


def test_unsupported_model_type_raises_systemexit(train_mod, fixtures_dir):
    # `gpt2` is a model_type transformers recognizes (so AutoConfig.from_pretrained
    # succeeds) but which is NOT in FAMILIES. With no opt-in (allow_generic=False, no
    # family_config), resolve_family preserves the strict fail-fast: SystemExit whose
    # message names all three porting affordances.
    with pytest.raises(SystemExit) as exc:
        train_mod.resolve_family(fixtures_dir / "unsupported_family")
    msg = str(exc.value)
    assert "Unsupported model_type='gpt2'" in msg
    assert "FAMILIES entry" in msg
    assert "--family-config" in msg
    assert "--allow-unverified-family" in msg
