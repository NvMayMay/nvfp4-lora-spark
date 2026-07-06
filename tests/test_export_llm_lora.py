import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.export_llm_lora import export_llm_lora
from scripts.split_both_adapter import classify_keys


def _tensor():
    return torch.ones((1, 1), dtype=torch.float32)


def _lora_pair(module_path: str) -> dict:
    return {
        f"base_model.model.{module_path}.lora_A.weight": _tensor(),
        f"base_model.model.{module_path}.lora_B.weight": _tensor(),
    }


def _write_both_adapter(adapter_dir: Path, state: dict) -> None:
    adapter_dir.mkdir(parents=True)
    cfg = {
        "base_model_name_or_path": "base-model",
        "train_target": "both",
        "r": 1,
        "lora_alpha": 2,
        "lora_dropout": 0.0,
        "both": {
            "base_model_name_or_path": "base-model",
            "vision_peft_scope": "^vision_model\\.",
            "projector_scopes": ["^mlp1\\."],
            "vision_target_modules": ["qkv", "proj"],
            "text_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
    }
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(cfg),
        encoding="utf-8",
    )
    save_file(state, str(adapter_dir / "adapter_model.safetensors"))


def _read_keys(adapter_file: Path) -> set[str]:
    with safe_open(adapter_file, framework="pt") as sf:
        return set(sf.keys())


def test_export_llm_lora_keeps_only_llm_keys(tmp_path):
    adapter_dir = tmp_path / "both"
    output_dir = tmp_path / "llm"

    state = {}
    state.update(_lora_pair("vision_model.blocks.0.attn.qkv"))
    state.update(_lora_pair("mlp1.0.proj"))
    state.update(_lora_pair("language_model.backbone.layers.0.self_attn.q_proj"))
    state.update(_lora_pair("language_model.backbone.layers.0.self_attn.k_proj"))
    state.update(_lora_pair("language_model.backbone.layers.0.self_attn.v_proj"))
    state.update(_lora_pair("language_model.backbone.layers.0.self_attn.o_proj"))
    _write_both_adapter(adapter_dir, state)

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    tower_keys, llm_keys = classify_keys(
        state.keys(),
        cfg["both"]["vision_peft_scope"],
        cfg["both"]["projector_scopes"],
    )
    assert len(tower_keys) == 4
    assert len(llm_keys) == 8

    summary = export_llm_lora(adapter_dir, output_dir)

    assert summary["retained_llm_tensors"] == 8
    assert summary["retained_attention_tensors"] == 6
    assert summary["dropped_vision_projector_tensors"] == 4

    exported_keys = _read_keys(output_dir / "adapter_model.safetensors")
    assert exported_keys == set(llm_keys)
    assert all("vision_model." not in key for key in exported_keys)
    assert all("mlp1." not in key for key in exported_keys)

    exported_cfg = json.loads((output_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert exported_cfg["train_target"] == "text"
    assert exported_cfg["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_export_llm_lora_refuses_zero_attention_keys(tmp_path):
    adapter_dir = tmp_path / "both"
    output_dir = tmp_path / "llm"

    state = {}
    state.update(_lora_pair("vision_model.blocks.0.attn.qkv"))
    state.update(_lora_pair("language_model.backbone.layers.0.self_attn.o_proj"))
    _write_both_adapter(adapter_dir, state)

    with pytest.raises(SystemExit, match="ZERO LLM attention LoRA tensors"):
        export_llm_lora(adapter_dir, output_dir)

    assert not (output_dir / "adapter_model.safetensors").exists()
