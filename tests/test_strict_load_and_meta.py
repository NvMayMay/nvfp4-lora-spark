"""Loader hardening: strict load_non_nvfp4_weights + assert_no_meta_tensors.

Models are built under init_empty_weights, so the two silent failure modes are:
  1. an on-disk tensor maps to a path the model does not have (warn-and-continue
     used to defer the explosion to first forward) -> strict load raises with
     the offending keys named;
  2. a model parameter that no on-disk tensor ever reaches stays on the meta
     device -> assert_no_meta_tensors raises unless the prefix is explicitly
     allowlisted (frozen multimodal towers).

Everything here runs on CPU with tiny synthetic safetensors shards.
"""
from __future__ import annotations

import json
import types

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from nvfp4_lora.loader import assert_no_meta_tensors, load_non_nvfp4_weights

CPU = torch.device("cpu")


class _TinyQwenShaped(nn.Module):
    """Minimal module whose paths match the qwen3_5 family translation:
    safetensors `model.language_model.X` -> in-memory `model.X`."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.x = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 4, bias=False)
        self.config = types.SimpleNamespace(model_type="qwen3_5_moe_text")


def _write_checkpoint(tmp_path, tensors: dict):
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, str(tmp_path / shard))
    index = {"weight_map": {k: shard for k in tensors}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))


def _base_tensors():
    return {
        "model.language_model.x.weight": torch.randn(4, 4),
        "model.language_model.x.bias": torch.randn(4),
        "lm_head.weight": torch.randn(4, 4),
    }


def test_clean_strict_load(tmp_path):
    _write_checkpoint(tmp_path, _base_tensors())
    model = _TinyQwenShaped()
    n = load_non_nvfp4_weights(model, tmp_path, device=CPU, dtype=torch.float32, strict=True)
    assert n == 3
    assert torch.isfinite(model.model.x.weight).all()


def test_strict_load_raises_on_unmapped_tensor(tmp_path):
    tensors = _base_tensors()
    tensors["model.language_model.zzz.weight"] = torch.randn(2, 2)
    _write_checkpoint(tmp_path, tensors)
    model = _TinyQwenShaped()
    with pytest.raises(RuntimeError) as exc:
        load_non_nvfp4_weights(model, tmp_path, device=CPU, dtype=torch.float32, strict=True)
    msg = str(exc.value)
    assert "model.zzz.weight" in msg
    assert "skip_st_prefixes" in msg  # the message tells you how to allowlist


def test_permissive_load_warns_and_continues(tmp_path, capsys):
    tensors = _base_tensors()
    tensors["model.language_model.zzz.weight"] = torch.randn(2, 2)
    _write_checkpoint(tmp_path, tensors)
    model = _TinyQwenShaped()
    n = load_non_nvfp4_weights(model, tmp_path, device=CPU, dtype=torch.float32, strict=False)
    assert n == 3  # the three good tensors still load
    assert "WARN: path not found" in capsys.readouterr().out


def test_family_skip_list_is_not_an_error(tmp_path):
    # Vision-tower keys are intentionally absent from the text-only graph:
    # the family translator skips them, strict or not.
    tensors = _base_tensors()
    tensors["model.visual.blocks.0.attn.qkv.weight"] = torch.randn(2, 2)
    _write_checkpoint(tmp_path, tensors)
    model = _TinyQwenShaped()
    n = load_non_nvfp4_weights(model, tmp_path, device=CPU, dtype=torch.float32, strict=True)
    assert n == 3


# ---------------------------------------------------------------------------
# assert_no_meta_tensors
# ---------------------------------------------------------------------------

def _meta_param_module():
    m = nn.Module()
    m.good = nn.Linear(2, 2)
    m.tower = nn.Linear(2, 2, device="meta")
    return m


def test_no_meta_assert_raises_on_meta_param():
    with pytest.raises(RuntimeError) as exc:
        assert_no_meta_tensors(_meta_param_module())
    msg = str(exc.value)
    assert "tower.weight" in msg
    assert "meta_allowed_prefixes" in msg


def test_no_meta_assert_allowlist():
    assert_no_meta_tensors(_meta_param_module(), allowed_prefixes=("tower.",))


def test_no_meta_assert_catches_buffers():
    m = nn.Module()
    m.register_buffer("rope", torch.empty(2, device="meta"))
    with pytest.raises(RuntimeError) as exc:
        assert_no_meta_tensors(m)
    assert "buffer rope" in str(exc.value)


def test_no_meta_assert_passes_clean_model():
    assert_no_meta_tensors(nn.Linear(2, 2))
