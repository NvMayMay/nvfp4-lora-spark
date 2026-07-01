"""CPU regressions for trainer fail-loud correctness guards.

These cover cases where a run could previously report success while producing
or resuming from a silently wrong adapter artifact.
"""
from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from nvfp4_lora.linear import BF16LoRALinear


class _Tok:
    def save_pretrained(self, dest):
        pass


def test_native_resume_zero_matching_modules_is_fatal(train_mod, tmp_path):
    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = BF16LoRALinear(
                3, 2, torch.zeros(2, 3), r=2, lora_alpha=4, dtype=torch.float32
            )

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    save_file(
        {
            "base_model.model.other_proj.lora_A.weight": torch.zeros(2, 3),
            "base_model.model.other_proj.lora_B.weight": torch.zeros(2, 2),
        },
        str(adapter / "adapter_model.safetensors"),
    )

    events = []
    with pytest.raises(RuntimeError, match="matched 0/1 expected LoRA"):
        train_mod._load_adapter_weights(
            Wrap(), adapter, "native", lambda event, **kw: events.append((event, kw))
        )
    assert events[0][0] == "resume_adapter_mismatch"
    assert events[0][1]["modules"] == "0/1"


def test_native_resume_partial_module_coverage_is_fatal(train_mod, tmp_path):
    events = []
    with pytest.raises(RuntimeError, match="matched 1/2 expected LoRA"):
        train_mod._validate_native_resume_coverage(
            adapter_dir=tmp_path / "adapter",
            expected_modules=2,
            loaded_modules=1,
            expected_expert_blocks=0,
            loaded_expert_blocks=0,
            expert_missing=0,
            log_fn=lambda event, **kw: events.append((event, kw)),
        )
    assert events == [
        (
            "resume_adapter_mismatch",
            {
                "modules": "1/2",
                "expert_blocks": "0/0",
                "expert_missing": 0,
                "path": str(tmp_path / "adapter"),
            },
        )
    ]


def test_native_save_meta_lora_tensor_is_fatal(train_mod, tmp_path):
    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = BF16LoRALinear(
                3,
                2,
                torch.zeros(2, 3),
                r=2,
                lora_alpha=4,
                dtype=torch.float32,
                lora_A_tensor=torch.empty(2, 3, device="meta"),
                lora_B_tensor=torch.zeros(2, 2),
            )

    with pytest.raises(RuntimeError, match=r"q_proj\.lora_A"):
        train_mod._save_adapter_atomic(
            Wrap(),
            _Tok(),
            tmp_path / "adapter",
            lambda *a, **k: None,
            lora_mode="native",
            base_model_dir="x",
            lora_r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_suffixes=["q_proj"],
        )


def test_peft_save_meta_key_guard_is_fatal(train_mod, tmp_path, monkeypatch):
    class FakePeft(nn.Module):
        active_adapter = "default"

        def __init__(self):
            super().__init__()
            self.peft_config = {"default": object()}

    fake_peft_pkg = types.ModuleType("peft")
    fake_peft = types.ModuleType("peft.utils")
    fake_peft.get_peft_model_state_dict = lambda model: {
        "base_model.model.q_proj.lora_A.weight": torch.empty(2, 3, device="meta")
    }
    fake_peft_pkg.utils = fake_peft
    monkeypatch.setitem(sys.modules, "peft", fake_peft_pkg)
    monkeypatch.setitem(sys.modules, "peft.utils", fake_peft)

    with pytest.raises(RuntimeError, match="base_model.model.q_proj.lora_A.weight"):
        train_mod._save_adapter_atomic(
            FakePeft(),
            _Tok(),
            tmp_path / "adapter",
            lambda *a, **k: None,
            lora_mode="peft",
            base_model_dir="x",
            lora_r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_suffixes=["q_proj"],
        )


def test_final_save_timeout_exits_nonzero_and_logs(train_mod, monkeypatch):
    events = []

    def fake_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(train_mod.os, "_exit", fake_exit)

    with pytest.raises(SystemExit) as exc:
        train_mod._exit_after_final_save_timeout(
            Path("/tmp/run"), lambda event, **kw: events.append((event, kw))
        )

    assert exc.value.code == train_mod.FINAL_SAVE_TIMEOUT_EXIT_CODE == 3
    assert events == [
        (
            "final_save_timeout",
            {
                "path": "/tmp/run",
                "fatal": True,
                "exit_code": 3,
                "note": "root adapter save did not complete; refusing to report success",
            },
        )
    ]
