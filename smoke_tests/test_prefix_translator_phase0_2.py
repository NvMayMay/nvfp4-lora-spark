#!/usr/bin/env python3
"""Phase 0.2 smoke tests for per-family prefix-map in make_key_translator()."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from nvfp4_lora.loader import make_key_translator, list_quantized_modules


QWEN35_DIR = Path("/home/veritan-spark-01/Veritan/Models/Qwen3.5-122B-A10B-NVFP4")
NANO_DIR = Path("/home/veritan-spark-01/Veritan/Models/Nemotron-3-Nano-30B-A3B-NVFP4")


def _require(p: Path) -> Path:
    if not (p / "model.safetensors.index.json").exists():
        pytest.skip(f"Model not found at {p}")
    return p


def _build_model_empty(model_dir: Path):
    """Construct a meta model via init_empty_weights — CPU only, no GPU init cost."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    return model


def test_qwen35_translator_returns_expected_prefixes():
    model_dir = _require(QWEN35_DIR)
    model = _build_model_empty(model_dir)
    translate, st_prefix, model_prefix = make_key_translator(model, model_dir)
    assert st_prefix == "model.language_model"
    assert model_prefix == "model"


def test_qwen35_translator_strips_language_model_infix():
    model_dir = _require(QWEN35_DIR)
    model = _build_model_empty(model_dir)
    translate, _, _ = make_key_translator(model, model_dir)
    assert (
        translate("model.language_model.layers.3.self_attn.q_proj.weight_packed")
        == "model.layers.3.self_attn.q_proj.weight_packed"
    )
    assert (
        translate("model.language_model.embed_tokens.weight")
        == "model.embed_tokens.weight"
    )
    assert (
        translate("model.language_model.norm.weight")
        == "model.norm.weight"
    )


def test_qwen35_translator_passes_through_lm_head():
    model_dir = _require(QWEN35_DIR)
    model = _build_model_empty(model_dir)
    translate, _, _ = make_key_translator(model, model_dir)
    assert translate("lm_head.weight") == "lm_head.weight"


def test_qwen35_translator_skips_visual_branch():
    model_dir = _require(QWEN35_DIR)
    model = _build_model_empty(model_dir)
    translate, _, _ = make_key_translator(model, model_dir)
    assert translate("model.visual.blocks.0.attn.qkv.weight") is None
    assert translate("model.visual.norm.weight") is None


def test_qwen35_attention_and_shared_expert_modules_resolve_in_named_modules():
    """Phase 0.2 + 0.1 acceptance: every NON-routed-expert quantized module name
    from list_quantized_modules() must map to a real in-memory module path.

    The 36,864 routed-expert keys are EXPECTED misses (fused-3D handled by Phase 0.6).
    The remaining 192 (= 48 attention + 144 shared_expert) must all hit."""
    model_dir = _require(QWEN35_DIR)
    model = _build_model_empty(model_dir)
    translate, _, _ = make_key_translator(model, model_dir)

    all_named = {n for n, _ in model.named_modules()}
    quant_set = list_quantized_modules(model_dir)

    non_routed_expert = [
        st_name for st_name in quant_set
        if "experts." not in st_name or "shared_expert" in st_name
    ]
    routed_expert = [
        st_name for st_name in quant_set
        if "experts." in st_name and "shared_expert" not in st_name
    ]

    # All non-routed should resolve
    misses = [
        st_name for st_name in non_routed_expert
        if translate(st_name) is not None and translate(st_name) not in all_named
    ]
    assert misses == [], (
        f"Non-routed-expert quantized modules failed translation: {misses[:5]} "
        f"(total misses: {len(misses)})"
    )

    # Routed experts SHOULD miss (they are fused-3D, no individual nn.Module)
    routed_hits = [
        st_name for st_name in routed_expert
        if translate(st_name) in all_named
    ]
    assert routed_hits == [], (
        f"Routed expert keys unexpectedly found as nn.Module — expected ALL to be "
        f"fused-3D (covered by Phase 0.6). First 5 hits: {routed_hits[:5]}"
    )


def test_nemotron_nano_heuristic_still_works():
    """Phase 0.2 must not regress Nemotron-H. The default heuristic path should still
    find backbone prefix and translate every NVFP4 module."""
    model_dir = _require(NANO_DIR)
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, dtype=torch.bfloat16)

    translate, st_prefix, model_prefix = make_key_translator(model, model_dir)
    assert st_prefix == "backbone"
    assert model_prefix == "backbone"

    all_named = {n for n, _ in model.named_modules()}
    quant_set = list_quantized_modules(model_dir)
    hits = sum(1 for st_name in quant_set if translate(st_name) in all_named)
    assert hits == len(quant_set), (
        f"Nemotron Nano regression: only {hits}/{len(quant_set)} quantized modules "
        "translated successfully (Nemotron experts are individual nn.Linear, so 100% hit expected)."
    )
