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
  * fp8_demoted fixture: Nemotron-style FP8 shared experts / attention now train
    natively (FP8LoRALinear), so an FP8 target makes it a native run.
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


def test_mixed_nvfp4_bf16_co_trains_native(train_mod, fixtures_dir):
    # gate_proj is NVFP4, o_proj is bf16. With BF16LoRALinear both co-train in ONE native
    # run (NVFP4 -> NVFP4LoRALinear, BF16 -> BF16LoRALinear) -- no longer a hard error.
    mode, coverage = train_mod.detect_lora_mode(fixtures_dir / "mixed_quant", ["gate_proj", "o_proj"])
    assert mode == "native"
    assert coverage["mode"] == "native"


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

def test_partial_quantization_co_trains_native(fixtures_dir):
    # partial_quant: a suffix that is NVFP4 in some layers and BF16 in others. With
    # BF16LoRALinear the BF16 instances co-train, so it resolves to a native run (was a
    # hard error pre-BF16LoRALinear).
    mode, _ = decide_lora_mode(fixtures_dir / "partial_quant", ["q_proj", "o_proj"])
    assert mode == "native"


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
# FP8 targets now train NATIVELY via FP8LoRALinear (frozen FP8 base + bf16 LoRA),
# so a native run adapts FP8 attention instead of freezing it.
# ---------------------------------------------------------------------------

def test_nvfp4_fp8_suffix_trains_native(fixtures_dir):
    # up_proj is NVFP4 + FP8 (no bf16). FP8 trains natively now, so a native run
    # adapts both -- no error, no flag needed (formerly a hard error).
    mode, coverage = decide_lora_mode(fixtures_dir / "fp8_demoted", ["up_proj"])
    assert mode == "native"
    assert coverage["inventory"]["up_proj"]["counts"] == {"nvfp4_modelopt": 1, "fp8": 1}


def test_fp8_only_suffix_trains_native(fixtures_dir):
    # q_proj is FP8 everywhere. An FP8-only suffix (no nvfp4, no bf16) trains
    # natively via FP8LoRALinear -- no flag needed.
    mode, coverage = decide_lora_mode(fixtures_dir / "fp8_demoted", ["q_proj"])
    assert mode == "native"
    assert coverage["inventory"]["q_proj"]["counts"] == {"fp8": 1}


def test_bf16_fp8_mix_co_trains_native(fixtures_dir):
    # peft_fp8_mix: o_proj is bf16+fp8, q_proj is bf16, no NVFP4. With FP8LoRALinear +
    # BF16LoRALinear both train natively, so any FP8 present -> a native run (was peft
    # pre-BF16LoRALinear). The inventory is unchanged; only the mode.
    mode, coverage = decide_lora_mode(fixtures_dir / "peft_fp8_mix", ["q_proj", "o_proj"])
    assert mode == "native"
    assert coverage["inventory"]["o_proj"]["counts"] == {"bf16": 1, "fp8": 1}
    assert coverage["inventory"]["q_proj"]["counts"] == {"bf16": 2}


def test_inventory_excludes_family_skip_list(fixtures_dir):
    # nemotron_h skips mtp.* (Multi-Token Prediction head). The MTP attention
    # block has q_proj/o_proj too, but it never trains, so it must not inflate
    # the target counts: 2 backbone attention layers, not 3.
    inv = build_target_inventory(fixtures_dir / "nemotron_mtp", ["q_proj", "o_proj"])
    assert inv["q_proj"]["counts"] == {"bf16": 2}
    assert inv["o_proj"]["counts"] == {"bf16": 2}
    assert all("mtp." not in ex for ex in inv["q_proj"]["examples"].get("bf16", []))


def test_skip_list_aware_mode_decision(fixtures_dir):
    mode, coverage = decide_lora_mode(fixtures_dir / "nemotron_mtp", ["q_proj", "o_proj"])
    assert mode == "peft"
    assert coverage["inventory"]["q_proj"]["counts"] == {"bf16": 2}


def test_build_target_inventory_shape(fixtures_dir):
    inv = build_target_inventory(fixtures_dir / "qwen3_5_moe", ["gate_proj", "nope"])
    assert inv["gate_proj"]["counts"] == {"nvfp4_ct": 1}
    assert inv["nope"]["counts"] == {}
    # examples are capped at 3 and name real modules
    for ex in inv["gate_proj"]["examples"]["nvfp4_ct"]:
        assert ex.endswith(".gate_proj")
