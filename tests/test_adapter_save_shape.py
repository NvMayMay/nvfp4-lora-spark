from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from nvfp4_lora.linear import BF16LoRALinear


class _FakeTokenizer:
    def save_pretrained(self, path):
        (Path(path) / "tokenizer_config.json").write_text("{}")


class _FakeNativeModel:
    def __init__(self):
        self.proj = BF16LoRALinear(
            4,
            3,
            torch.zeros(3, 4, dtype=torch.bfloat16),
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            device=torch.device("cpu"),
        )

    def named_modules(self):
        return iter([("layers.0.self_attn.q_proj", self.proj)])


def test_save_adapter_atomic_native_key_shape_and_target_suffixes(train_mod, tmp_path):
    out = tmp_path / "adapter"

    train_mod._save_adapter_atomic(
        _FakeNativeModel(),
        _FakeTokenizer(),
        out,
        lambda event, **kw: None,
        lora_mode="native",
        base_model_dir="/models/base",
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_suffixes=["q_proj"],
    )

    state = load_file(str(out / "adapter_model.safetensors"))
    prefix = "base_model.model.layers.0.self_attn.q_proj"
    assert set(state) == {
        f"{prefix}.lora_A.weight",
        f"{prefix}.lora_B.weight",
    }
    assert tuple(state[f"{prefix}.lora_A.weight"].shape) == (2, 4)
    assert tuple(state[f"{prefix}.lora_B.weight"].shape) == (3, 2)

    cfg = json.loads((out / "adapter_config.json").read_text())
    assert cfg["target_modules"] == ["q_proj"]


@pytest.mark.integration
@pytest.mark.xfail(strict=False, reason="needs a real compatible minimal base fixture")
def test_stock_peft_round_trip_placeholder():
    pytest.fail("Documented placeholder for PEFT PeftModel.from_pretrained round-trip.")
