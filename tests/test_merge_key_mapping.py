"""Merge-side adapter-key translation must match the trainer-side load layout.

Both merge scripts now derive their key mapping from the same place the
trainer does:

  * merge_lora_into_ct_nvfp4.py uses families.adapter_key_to_base_prefix with
    the family's expert_prefix pair (in-memory prefix, on-disk prefix);
  * merge_lora_into_nvfp4.py derives the Nemotron backbone prefix from the
    base index itself (detect_base_prefix).

The consistency tests check translated adapter keys against the REAL fixture
indexes: a key the merge would write must exist in the base checkpoint.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from nvfp4_lora.families import FAMILIES, adapter_key_to_base_prefix

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def merge_ct():
    return _load_script("merge_lora_into_ct_nvfp4.py")


@pytest.fixture(scope="module")
def merge_modelopt():
    return _load_script("merge_lora_into_nvfp4.py")


def _index_keys(fixtures_dir: Path, family_dir: str) -> set[str]:
    idx = json.loads((fixtures_dir / family_dir / "model.safetensors.index.json").read_text())
    return set(idx["weight_map"].keys())


# ---------------------------------------------------------------------------
# Registry-driven CT translation
# ---------------------------------------------------------------------------

def test_qwen_adapter_key_translation(fixtures_dir):
    mem, st = FAMILIES["qwen3_5_moe"]["expert_prefix"]
    prefix, side = adapter_key_to_base_prefix(
        "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight", mem, st
    )
    assert prefix == "model.language_model.layers.3.self_attn.q_proj"
    assert side == "A"
    # The merge writes {prefix}.weight_packed: it must exist in the real layout.
    assert f"{prefix}.weight_packed" in _index_keys(fixtures_dir, "qwen3_5_moe")


def test_mistral_adapter_key_translation(fixtures_dir):
    mem, st = FAMILIES["mistral3"]["expert_prefix"]
    prefix, side = adapter_key_to_base_prefix(
        "base_model.model.model.language_model.layers.0.mlp.experts.0.gate_proj.lora_B.weight",
        mem, st,
    )
    assert prefix == "language_model.model.layers.0.mlp.experts.0.gate_proj"
    assert side == "B"
    assert f"{prefix}.weight_packed" in _index_keys(fixtures_dir, "mistral3")


def test_already_disk_prefixed_key_passes_through():
    mem, st = FAMILIES["qwen3_5_moe"]["expert_prefix"]
    prefix, _ = adapter_key_to_base_prefix(
        "base_model.model.model.language_model.layers.3.self_attn.q_proj.lora_A.weight",
        mem, st,
    )
    assert prefix == "model.language_model.layers.3.self_attn.q_proj"


def test_unrecognized_adapter_key_raises():
    mem, st = FAMILIES["mistral3"]["expert_prefix"]
    with pytest.raises(ValueError):
        adapter_key_to_base_prefix(
            "base_model.model.unexpected.layers.0.q_proj.lora_A.weight", mem, st
        )
    with pytest.raises(ValueError):
        adapter_key_to_base_prefix("base_model.model.model.layers.0.q_proj.weight", mem, st)


def test_qkv_scale_grouping_uses_family_prefix(merge_ct):
    mem, st = FAMILIES["qwen3_5_moe"]["expert_prefix"]
    qkv_re = merge_ct.make_qkv_regex(st)
    prefixes = [
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.k_proj",
        "model.language_model.layers.3.self_attn.v_proj",
        "model.language_model.layers.3.self_attn.o_proj",
        "model.language_model.layers.7.self_attn.q_proj",
    ]
    groups = merge_ct.scale_groups(prefixes, qkv_re)
    trio = [g for g in groups if len(g) == 3]
    assert len(trio) == 1 and all("layers.3" in p for p in trio[0])
    singles = [g for g in groups if len(g) == 1]
    assert sorted(g[0].rsplit(".", 1)[-1] for g in singles) == ["o_proj", "q_proj"]


def test_mistral_qkv_regex_matches_its_layout(merge_ct):
    mem, st = FAMILIES["mistral3"]["expert_prefix"]
    qkv_re = merge_ct.make_qkv_regex(st)
    assert qkv_re.match("language_model.model.layers.0.self_attn.q_proj")
    assert not qkv_re.match("model.language_model.layers.0.self_attn.q_proj")


def test_resolve_text_backbone_prefix_prefers_st_to_model(merge_ct):
    """The CT merge derives its prefix from st_to_model[0] (the text-backbone rule), not
    expert_prefix -- correct for the general case, same result for these families."""
    for name in ("mistral3", "qwen3_5_moe"):
        fam = FAMILIES[name]
        mem, st = merge_ct.resolve_text_backbone_prefix(fam)
        exp_st, exp_mem = fam["st_to_model"][0]          # stored (on_disk, in_memory)
        assert (mem, st) == (exp_mem, exp_st)


def test_resolve_text_backbone_prefix_falls_back_to_expert(merge_ct):
    """A family with no st_to_model falls back to expert_prefix."""
    fam = FAMILIES["glm4_moe"]
    assert fam.get("st_to_model") is None
    assert merge_ct.resolve_text_backbone_prefix(fam) == tuple(fam["expert_prefix"])


def test_scale_groups_warns_on_incomplete_qkv_trio(merge_ct, capsys):
    """A partial q/k/v merge breaks vLLM's fused-qkv shared-scale invariant -> loud warning."""
    _, st = merge_ct.resolve_text_backbone_prefix(FAMILIES["mistral3"])
    qkv_re = merge_ct.make_qkv_regex(st)
    prefixes = [
        # layer 0: q + v only (NO k) -> incomplete trio.
        "language_model.model.layers.0.self_attn.q_proj",
        "language_model.model.layers.0.self_attn.v_proj",
        # layer 1: full trio -> fine.
        "language_model.model.layers.1.self_attn.q_proj",
        "language_model.model.layers.1.self_attn.k_proj",
        "language_model.model.layers.1.self_attn.v_proj",
    ]
    groups = merge_ct.scale_groups(prefixes, qkv_re)
    out = capsys.readouterr().out
    assert "INCOMPLETE" in out
    # The complete trio is still grouped correctly.
    assert any(len(g) == 3 and all("layers.1" in p for p in g) for g in groups)


# ---------------------------------------------------------------------------
# ModelOpt/Nemotron prefix derivation
# ---------------------------------------------------------------------------

def test_detect_base_prefix(merge_modelopt):
    wm = {
        "backbone.layers.0.mixer.experts.0.up_proj.weight": "a",
        "backbone.embeddings.weight": "a",
        "lm_head.weight": "a",
        "mtp.layers.0.mixer.up_proj.weight": "a",
    }
    assert merge_modelopt.detect_base_prefix(wm) == "backbone."


def test_detect_base_prefix_ambiguous_raises(merge_modelopt):
    wm = {"backbone.x.weight": "a", "vision.x.weight": "a"}
    with pytest.raises(SystemExit):
        merge_modelopt.detect_base_prefix(wm)


def test_modelopt_adapter_key_translation(merge_modelopt):
    f = merge_modelopt.adapter_key_to_base_key
    expected = "backbone.layers.1.mixer.experts.0.up_proj.weight"
    assert f("base_model.model.backbone.layers.1.mixer.experts.0.up_proj.lora_A.weight",
             "backbone.") == expected
    # PEFT sometimes emits a doubled "model." segment.
    assert f("base_model.model.model.backbone.layers.1.mixer.experts.0.up_proj.lora_B.weight",
             "backbone.") == expected
    with pytest.raises(ValueError):
        f("not_an_adapter_key.weight", "backbone.")
