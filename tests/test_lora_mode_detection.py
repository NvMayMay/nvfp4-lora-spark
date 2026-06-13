"""detect_lora_mode + the target-coverage inventory: native vs PEFT decision.

detect_lora_mode wraps nvfp4_lora.loader.decide_lora_mode, which reads ONLY
model.safetensors.index.json and classifies EVERY module matching each target
suffix individually:

  .weight_packed                              -> nvfp4_ct
  .weight + .weight_scale + .weight_scale_2   -> nvfp4_modelopt
  .weight + .weight_scale                     -> fp8 (per-tensor)
  .weight only                                -> bf16

Key checkpoint facts that drive these tests:
  * Qwen3.5: full-attention layers carry NVFP4-quantized self_attn q/k/v/o_proj, so
    targeting q_proj,k_proj,v_proj,o_proj -> "native".
  * Mistral-Small-4: the quant config ignores `re:.*self_attn.*`, so q_b_proj /
    kv_b_proj / o_proj are plain bf16 (no weight_scale) -> "peft". Only the MoE
    expert gate/up/down_proj are NVFP4.
  * partial_quant fixture: o_proj NVFP4 in layer 0 but BF16 in layer 1 -> hard
    error unless allow_partial_targets (the silent-partial-training hole).
  * fp8_demoted fixture: Nemotron-style FP8 shared experts / attention -> hard
    error unless allow_fp8_targets (FP8 modules are demoted to frozen).
"""
from __future__ import annotations

import pytest

from nvfp4_lora.loader import (
    build_target_inventory,
    classify_module_storage,
    decide_lora_mode,
    list_quantized_modules,
)


def test_list_quantized_modules_qwen(fixtures_dir):
    q = list_quantized_modules(fixtures_dir / "qwen3_5_moe")
    # self_attn projections on the full-attention layer are quantized
    assert "model.language_model.layers.3.self_attn.q_proj" in q
    assert "model.language_model.layers.3.self_attn.o_proj" in q
    # routed experts are quantized
    assert "model.language_model.layers.0.mlp.experts.0.gate_proj" in q
    # plain bf16 modules (norms, embeddings, lm_head, linear_attn) are NOT quantized
    assert "model.language_model.embed_tokens" not in q
    assert "lm_head" not in q
    assert "model.language_model.layers.0.linear_attn.out_proj" not in q
    assert "model.language_model.layers.0.input_layernorm" not in q


def test_list_quantized_modules_mistral(fixtures_dir):
    q = list_quantized_modules(fixtures_dir / "mistral3")
    # MoE experts + shared_experts are quantized
    assert "language_model.model.layers.0.mlp.experts.0.gate_proj" in q
    assert "language_model.model.layers.0.mlp.shared_experts.up_proj" in q
    # self_attn is excluded by the quant config -> plain bf16, not quantized
    assert "language_model.model.layers.0.self_attn.q_b_proj" not in q
    assert "language_model.model.layers.0.self_attn.kv_b_proj" not in q
    assert "language_model.model.layers.0.self_attn.o_proj" not in q
    # gate / lm_head / embeddings not quantized
    assert "language_model.model.layers.0.mlp.gate" not in q
    assert "language_model.lm_head" not in q


def test_classify_module_storage():
    keys = {
        "a.ct.weight_packed", "a.ct.weight_scale", "a.ct.weight_global_scale",
        "a.mo.weight", "a.mo.weight_scale", "a.mo.weight_scale_2",
        "a.fp8.weight", "a.fp8.weight_scale",
        "a.plain.weight",
    }
    assert classify_module_storage(keys, "a.ct") == "nvfp4_ct"
    assert classify_module_storage(keys, "a.mo") == "nvfp4_modelopt"
    assert classify_module_storage(keys, "a.fp8") == "fp8"
    assert classify_module_storage(keys, "a.plain") == "bf16"
    assert classify_module_storage(keys, "a.nothing") == "absent"


def test_qwen_targets_detect_native(train_mod, fixtures_dir):
    mode, coverage = train_mod.detect_lora_mode(
        fixtures_dir / "qwen3_5_moe", ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    assert mode == "native"
    assert coverage["mode"] == "native"
    assert coverage["inventory"]["q_proj"]["counts"] == {"nvfp4_ct": 1}


def test_mistral_targets_detect_peft(train_mod, fixtures_dir):
    mode, coverage = train_mod.detect_lora_mode(
        fixtures_dir / "mistral3", ["q_b_proj", "kv_b_proj", "o_proj"]
    )
    assert mode == "peft"
    assert coverage["inventory"]["o_proj"]["counts"] == {"bf16": 1}


def test_mixed_targets_raise_systemexit(train_mod, fixtures_dir):
    # gate_proj is quantized, o_proj is not -> native and PEFT can't be combined.
    with pytest.raises(SystemExit) as exc:
        train_mod.detect_lora_mode(fixtures_dir / "mixed_quant", ["gate_proj", "o_proj"])
    msg = str(exc.value)
    assert "Mixed LoRA targets" in msg
    assert "gate_proj" in msg
    assert "o_proj" in msg


def test_native_requires_all_targets_quantized(train_mod, fixtures_dir):
    # If even one target suffix is unquantized alongside a quantized one, that's mixed.
    # If NONE are quantized -> peft. Here gate_proj alone -> native.
    assert train_mod.detect_lora_mode(fixtures_dir / "mixed_quant", ["gate_proj"])[0] == "native"
    assert train_mod.detect_lora_mode(fixtures_dir / "mixed_quant", ["o_proj"])[0] == "peft"


def test_unknown_suffix_is_hard_error(train_mod, fixtures_dir):
    # The v1 heuristic silently classified a typo'd suffix as "not quantized";
    # combined with all-bf16 targets that meant a clean "peft" run training
    # nothing for the typo'd module. Now: hard error.
    with pytest.raises(SystemExit) as exc:
        train_mod.detect_lora_mode(fixtures_dir / "qwen3_5_moe", ["q_prj"])
    assert "matches no module" in str(exc.value)


# ---------------------------------------------------------------------------
# Partial quantization across layers (the silent-partial-training hole)
# ---------------------------------------------------------------------------

def test_partial_quantization_is_hard_error(fixtures_dir):
    with pytest.raises(SystemExit) as exc:
        decide_lora_mode(fixtures_dir / "partial_quant", ["q_proj", "o_proj"])
    msg = str(exc.value)
    assert "PARTIALLY quantized" in msg
    assert "o_proj" in msg
    assert "--allow-partial-targets" in msg


def test_partial_quantization_allowed_with_flag(fixtures_dir):
    mode, coverage = decide_lora_mode(
        fixtures_dir / "partial_quant", ["q_proj", "o_proj"],
        allow_partial_targets=True,
    )
    assert mode == "native"
    assert coverage["inventory"]["o_proj"]["counts"] == {"nvfp4_ct": 1, "bf16": 1}
    # Layer-level visibility: layer 0 quantized, layer 1 not.
    assert coverage["inventory"]["o_proj"]["layers"]["nvfp4_ct"] == [0]
    assert coverage["inventory"]["o_proj"]["layers"]["bf16"] == [1]


def test_fully_quantized_suffix_unaffected(fixtures_dir):
    mode, coverage = decide_lora_mode(fixtures_dir / "partial_quant", ["q_proj"])
    assert mode == "native"
    assert coverage["inventory"]["q_proj"]["counts"] == {"nvfp4_ct": 2}


# ---------------------------------------------------------------------------
# FP8-demoted targets (loader freezes them; silent no-training without consent)
# ---------------------------------------------------------------------------

def test_fp8_target_is_hard_error(fixtures_dir):
    with pytest.raises(SystemExit) as exc:
        decide_lora_mode(fixtures_dir / "fp8_demoted", ["up_proj"])
    msg = str(exc.value)
    assert "FP8" in msg
    assert "--allow-fp8-targets" in msg


def test_fp8_target_allowed_with_flag(fixtures_dir):
    mode, coverage = decide_lora_mode(
        fixtures_dir / "fp8_demoted", ["up_proj"], allow_fp8_targets=True
    )
    assert mode == "native"
    assert coverage["inventory"]["up_proj"]["counts"] == {"nvfp4_modelopt": 1, "fp8": 1}


def test_fp8_only_suffix_trains_via_peft(fixtures_dir):
    # q_proj in this fixture is FP8 everywhere. The loader dequantizes FP8 to a
    # frozen BF16 nn.Linear, which PEFT can wrap, so an FP8-only suffix resolves
    # to peft and trains (no nvfp4 -> no native demotion, no flag needed).
    mode, coverage = decide_lora_mode(fixtures_dir / "fp8_demoted", ["q_proj"])
    assert mode == "peft"
    assert coverage["inventory"]["q_proj"]["counts"] == {"fp8": 1}


def test_bf16_fp8_mix_resolves_to_peft_without_flags(fixtures_dir):
    # The Super attention o_proj shape: some layers BF16, some FP8, no NVFP4.
    # Under PEFT both are wrappable, so this must NOT require --allow-fp8-targets.
    mode, coverage = decide_lora_mode(fixtures_dir / "peft_fp8_mix", ["q_proj", "o_proj"])
    assert mode == "peft"
    assert coverage["inventory"]["o_proj"]["counts"] == {"bf16": 1, "fp8": 1}
    assert coverage["inventory"]["q_proj"]["counts"] == {"bf16": 2}


def test_native_suffix_with_fp8_still_blocks(fixtures_dir):
    # up_proj is NVFP4 + FP8 (no bf16): a native suffix, so the FP8 stragglers
    # would stay frozen -> still a hard error until --allow-fp8-targets.
    with pytest.raises(SystemExit) as exc:
        decide_lora_mode(fixtures_dir / "fp8_demoted", ["up_proj"])
    assert "--allow-fp8-targets" in str(exc.value)


def test_build_target_inventory_shape(fixtures_dir):
    inv = build_target_inventory(fixtures_dir / "qwen3_5_moe", ["gate_proj", "nope"])
    assert inv["gate_proj"]["counts"] == {"nvfp4_ct": 1}
    assert inv["nope"]["counts"] == {}
    # examples are capped at 3 and name real modules
    for ex in inv["gate_proj"]["examples"]["nvfp4_ct"]:
        assert ex.endswith(".gate_proj")
