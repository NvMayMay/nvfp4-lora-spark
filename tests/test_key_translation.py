"""make_key_translator: safetensors-key -> model-attribute-path mapping per family.

make_key_translator(model, model_dir) dispatches on model.config.model_type. The
Qwen and Mistral branches need only `model.config.model_type` (they never touch
model_dir or named_children), so we drive them with tiny SimpleNamespace stubs. The
Nemotron fallback branch reads model.safetensors.index.json and walks
model.named_children() looking for a child with `.layers`, so that branch gets a
minimal real nn.Module plus a synthesized index file.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import torch.nn as nn

from nvfp4_lora.loader import make_key_translator


def _stub(model_type: str):
    return types.SimpleNamespace(config=types.SimpleNamespace(model_type=model_type))


# --------------------------------------------------------------------------------------
# Qwen3.5 MoE
# --------------------------------------------------------------------------------------
def test_qwen_translate(tmp_path):
    model = _stub("qwen3_5_moe")
    # model_dir is unused by the Qwen branch but must be a valid path argument.
    translate, st_prefix, model_prefix = make_key_translator(model, tmp_path)
    assert st_prefix == "model.language_model"
    assert model_prefix == "model"

    # language_model.* -> model.* (strip the .language_model. multimodal wrapper)
    assert (
        translate("model.language_model.layers.0.mlp.experts.0.gate_proj.weight_packed")
        == "model.layers.0.mlp.experts.0.gate_proj.weight_packed"
    )
    assert (
        translate("model.language_model.layers.3.self_attn.q_proj.weight_packed")
        == "model.layers.3.self_attn.q_proj.weight_packed"
    )
    # vision tower is skipped for text-only training
    assert translate("model.visual.blocks.0.attn.qkv.weight") is None
    # lm_head passes through unchanged
    assert translate("lm_head.weight") == "lm_head.weight"


def test_qwen_text_variant_uses_same_branch(tmp_path):
    # AutoModelForCausalLM.from_config yields the text-only "qwen3_5_moe_text" type;
    # the translator must handle both names identically.
    model = _stub("qwen3_5_moe_text")
    translate, _, _ = make_key_translator(model, tmp_path)
    assert (
        translate("model.language_model.norm.weight") == "model.norm.weight"
    )
    assert translate("model.visual.merger.linear_fc1.weight") is None


# --------------------------------------------------------------------------------------
# Mistral3 / Mistral4
# --------------------------------------------------------------------------------------
def test_mistral_translate(tmp_path):
    model = _stub("mistral3")
    translate, st_prefix, model_prefix = make_key_translator(model, tmp_path)
    assert st_prefix == "language_model.model"
    assert model_prefix == "model.language_model"

    # text backbone: language_model.model.* -> model.language_model.*
    assert (
        translate("language_model.model.layers.1.self_attn.o_proj.weight")
        == "model.language_model.layers.1.self_attn.o_proj.weight"
    )
    assert (
        translate("language_model.model.layers.0.mlp.experts.0.gate_proj.weight_packed")
        == "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_packed"
    )
    # multimodal branches skipped
    assert translate("vision_tower.transformer.layers.0.attention.q_proj.weight") is None
    assert translate("multi_modal_projector.linear_1.weight") is None
    # lm_head: language_model.lm_head.weight -> lm_head.weight
    assert translate("language_model.lm_head.weight") == "lm_head.weight"


def test_mistral4_text_variant_uses_same_branch(tmp_path):
    model = _stub("mistral4")
    translate, _, _ = make_key_translator(model, tmp_path)
    assert (
        translate("language_model.model.embed_tokens.weight")
        == "model.language_model.embed_tokens.weight"
    )


# --------------------------------------------------------------------------------------
# Nemotron-H fallback heuristic
# --------------------------------------------------------------------------------------
class _InnerWithLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity()])


class _NemotronStub(nn.Module):
    """Mimics Super-120B: self.model = NemotronHModel(...) -> in-memory prefix 'model'."""

    def __init__(self):
        super().__init__()
        self.model = _InnerWithLayers()
        self.config = types.SimpleNamespace(model_type="nemotron_h")


def _write_index(dir_path: Path, weight_map: dict) -> None:
    (dir_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


def test_nemotron_fallback_translate(tmp_path):
    _write_index(
        tmp_path,
        {
            # safetensors prefix 'backbone' -> in-memory 'model'
            "backbone.layers.0.mixer.in_proj.weight": "s.safetensors",
            "backbone.layers.0.mixer.in_proj.weight_scale": "s.safetensors",
            "lm_head.weight": "s.safetensors",
            # MTP speculation layers: skipped (serve-only, never trained)
            "mtp.0.fc.weight": "s.safetensors",
        },
    )
    model = _NemotronStub()
    translate, st_prefix, model_prefix = make_key_translator(model, tmp_path)
    assert st_prefix == "backbone"
    assert model_prefix == "model"

    assert (
        translate("backbone.layers.0.mixer.in_proj.weight")
        == "model.layers.0.mixer.in_proj.weight"
    )
    assert translate("mtp.0.fc.weight") is None
    assert translate("lm_head.weight") == "lm_head.weight"


class _NemotronNanoStub(nn.Module):
    """Mimics Nano-30B: self.backbone = NemotronHModel(...) -> in-memory prefix 'backbone'."""

    def __init__(self):
        super().__init__()
        self.backbone = _InnerWithLayers()
        self.config = types.SimpleNamespace(model_type="nemotron_h")


def test_nemotron_nano_identity_prefix(tmp_path):
    # nemotron_h IS in the family registry, but with st_to_model=None, which
    # must fall through to the dynamic heuristic: Nano's in-memory prefix is
    # 'backbone' (identity mapping), unlike Super's 'model'. A static registry
    # rule could not express both, which is why the entry stays dynamic.
    _write_index(
        tmp_path,
        {
            "backbone.layers.0.mixer.in_proj.weight": "s.safetensors",
            "lm_head.weight": "s.safetensors",
        },
    )
    model = _NemotronNanoStub()
    translate, st_prefix, model_prefix = make_key_translator(model, tmp_path)
    assert (st_prefix, model_prefix) == ("backbone", "backbone")
    assert (
        translate("backbone.layers.0.mixer.in_proj.weight")
        == "backbone.layers.0.mixer.in_proj.weight"
    )
