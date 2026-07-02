"""Unit tests for nvfp4_lora.adapter_keys -- the single source of truth for the LoRA
adapter key schema and the rekey transforms.

Everything that used to be duplicated across nybbloris/plan.py, the two rekey scripts,
and the vLLM patch now lives in one module; these tests lock its API (schema parse,
strip-prefix, and each rekey transform) so a drift can't silently reintroduce the
"silent no-op adapter" class the consolidation was meant to kill.
"""
from __future__ import annotations

import pytest

from nvfp4_lora.adapter_keys import (
    PEFT_PREFIX,
    REKEYS,
    adapter_module_path,
    identity,
    is_lora_key,
    rekey_by_name,
    strip_base_prefix,
    wrapped_remap_module,
    wrapped_remap_safetensors_key,
)


# --------------------------------------------------------------------------------------
# Key schema: recognition + reduction to a bare module path.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("key,expected", [
    # PEFT dense form (with .weight)
    ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
     "model.layers.0.self_attn.q_proj"),
    ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
     "model.layers.0.self_attn.q_proj"),
    # Native STACKED expert form (NO .weight suffix) -- must still be recognized.
    ("base_model.model.model.layers.3.mlp.experts.gate_up.lora_A",
     "model.layers.3.mlp.experts.gate_up"),
    ("base_model.model.model.layers.3.mlp.experts.down.lora_B",
     "model.layers.3.mlp.experts.down"),
    # Already-wrapped serve layout passes through the strip unchanged.
    ("base_model.model.language_model.model.layers.0.self_attn.o_proj.lora_A.weight",
     "language_model.model.layers.0.self_attn.o_proj"),
])
def test_adapter_module_path(key, expected):
    assert adapter_module_path(key) == expected


@pytest.mark.parametrize("key", [
    "__metadata__",
    "base_model.model.model.layers.0.self_attn.q_proj.weight",  # base weight, not a LoRA tensor
    "some.random.tensor",
])
def test_non_lora_keys_are_none(key):
    assert adapter_module_path(key) is None
    assert is_lora_key(key) is False


def test_is_lora_key_both_suffix_forms():
    assert is_lora_key("base_model.model.x.lora_A.weight")
    assert is_lora_key("base_model.model.x.experts.down.lora_B")  # native, no .weight


def test_strip_base_prefix():
    assert strip_base_prefix(PEFT_PREFIX + "model.layers.0.q") == "model.layers.0.q"
    # No prefix -> unchanged (idempotent, never strips twice).
    assert strip_base_prefix("model.layers.0.q") == "model.layers.0.q"
    assert strip_base_prefix(strip_base_prefix(PEFT_PREFIX + "a")) == "a"


# --------------------------------------------------------------------------------------
# Rekey transforms.
# --------------------------------------------------------------------------------------
def test_identity_is_noop():
    for k in ("model.layers.0.self_attn.q_proj", "language_model.model.layers.0.x", "lm_head"):
        assert identity(k) == k


def test_wrapped_remap_module():
    assert (wrapped_remap_module("model.layers.5.self_attn.k_proj")
            == "language_model.model.layers.5.self_attn.k_proj")
    # Only the flat decoder prefix is swapped; layout-invariant + already-wrapped pass through.
    assert wrapped_remap_module("lm_head") == "lm_head"
    assert (wrapped_remap_module("language_model.model.layers.5.x")
            == "language_model.model.layers.5.x")


def test_wrapped_remap_module_swaps_once():
    """A second application is a fixpoint (the output no longer matches the flat prefix)."""
    once = wrapped_remap_module("model.layers.0.mlp.experts")
    assert wrapped_remap_module(once) == once


def test_wrapped_remap_safetensors_key():
    src = "base_model.model.model.layers.7.self_attn.v_proj.lora_A.weight"
    dst = "base_model.model.language_model.model.layers.7.self_attn.v_proj.lora_A.weight"
    assert wrapped_remap_safetensors_key(src) == dst
    # Idempotent on already-wrapped keys.
    assert wrapped_remap_safetensors_key(dst) == dst
    # Native expert key (no .weight) is still remapped.
    exp = "base_model.model.model.layers.2.mlp.experts.gate_up.lora_A"
    assert wrapped_remap_safetensors_key(exp).startswith(
        "base_model.model.language_model.model.layers.2.mlp.experts")


def test_module_and_safetensors_remap_agree():
    """The two spellings describe the SAME transform: rekey the module path, then
    re-attach the PEFT prefix, and you get the safetensors-key transform."""
    mod = "model.layers.4.self_attn.q_proj"
    st_key = PEFT_PREFIX + mod + ".lora_A.weight"
    via_module = PEFT_PREFIX + wrapped_remap_module(mod) + ".lora_A.weight"
    assert wrapped_remap_safetensors_key(st_key) == via_module


# --------------------------------------------------------------------------------------
# The canonical REKEYS table + lookup.
# --------------------------------------------------------------------------------------
def test_rekeys_table_is_canonical():
    names = [n for n, _ in REKEYS]
    assert names == ["identity", "language_model"]
    assert all(callable(fn) for _, fn in REKEYS)


def test_rekey_by_name():
    assert rekey_by_name("identity") is identity
    assert rekey_by_name("language_model") is wrapped_remap_module
    with pytest.raises(KeyError):
        rekey_by_name("nonesuch")


def test_full_roundtrip_via_module_path_then_rekey():
    """End-to-end: a flat PEFT key -> module path -> wrapped module path matches
    what the safetensors rekey produces (round-trip consistency across the API)."""
    flat = "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight"
    mod = adapter_module_path(flat)
    wrapped_mod = rekey_by_name("language_model")(mod)
    assert wrapped_mod == "language_model.model.layers.0.self_attn.o_proj"
    # The module path of the safetensors-remapped key equals the module-path rekey.
    assert adapter_module_path(wrapped_remap_safetensors_key(flat)) == wrapped_mod
