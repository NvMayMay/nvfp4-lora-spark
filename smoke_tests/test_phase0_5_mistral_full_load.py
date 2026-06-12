#!/usr/bin/env python3
"""Phase 0.5 acceptance gate — full Mistral-Small-4-119B-NVFP4 load + 1-token forward.

Mirrors test_phase0_5_qwen35_full_load.py with Mistral-specific differences:
- model_family="mistral3" (HF model_type) → translator dispatches to MLA-aware path
- LoRA targets ("q_b_proj", "kv_b_proj", "o_proj") — MLA names, NOT q_proj/v_proj
- Vision-encoder freeze (model.model.vision_tower, model.model.multi_modal_projector)
- NO SSM layers → expected to clear the NVRM descriptor cliff that blocks Qwen3.5

Run:
    cd /home/veritan-spark-01/Veritan/Sandbox/repos/nvfp4-lora-spark
    RUN_PHASE0_5_FULL_LOAD=1 \\
        /home/veritan-spark-01/Veritan/.venvs/qwen-peft/bin/python -u \\
        smoke_tests/test_phase0_5_mistral_full_load.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch


MODEL_DIR = Path("/home/veritan-spark-01/Veritan/Models/Mistral-Small-4-119B-2603-NVFP4-HF")


def _require_model():
    if not (MODEL_DIR / "model.safetensors.index.json").exists():
        pytest.skip(f"Mistral-Small-4 NVFP4 model not at {MODEL_DIR}; run scripts/quantize_mistral_to_nvfp4.py first")


def test_phase0_5_mistral_full_load_and_one_token_forward():
    """The Phase 0 acceptance gate for Mistral-Small-4."""
    _require_model()
    if not os.environ.get("RUN_PHASE0_5_FULL_LOAD"):
        pytest.skip("Set RUN_PHASE0_5_FULL_LOAD=1 to run the full-load acceptance gate")

    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForImageTextToText
    from nvfp4_lora.experts import (
        replace_moe_experts_with_nvfp4_3d,
        assemble_nvfp4_experts3d_batched,
        NVFP4Experts3D,
    )
    from nvfp4_lora.loader import (
        list_quantized_modules,
        make_key_translator,
        replace_nvfp4_modules,
        load_non_nvfp4_weights,
        _assign_dequant_workspaces,
    )

    t0 = time.time()
    print(f"\n=== Phase 0.5 acceptance gate: Mistral-Small-4-119B-NVFP4 ===")
    print(f"Step 1: building model on meta...")
    cfg = AutoConfig.from_pretrained(str(MODEL_DIR))
    with init_empty_weights():
        # Mistral-Small-4-119B is a multimodal model — Mistral3 outer wrapper around
        # the mistral4 text backbone + Pixtral vision tower. Use the
        # image-text-to-text auto class; vision branch gets frozen below.
        model = AutoModelForImageTextToText.from_config(cfg)
    t1 = time.time()
    print(f"  built in {t1-t0:.1f}s; model_type={model.config.model_type}")

    print(f"Step 2: replacing fused-3D MoE blocks (Mistral4NaiveMoe) with NVFP4Experts3D...")
    # model_family inferred from model.config.model_type ('mistral4' for the text backbone
    # via AutoModelForCausalLM.from_config dispatch; replace_moe_experts_with_nvfp4_3d
    # accepts both 'mistral3' and 'mistral4').
    family = getattr(model.config, "model_type", "mistral4")
    n_moe_replaced = replace_moe_experts_with_nvfp4_3d(model, model_family=family)
    t2 = time.time()
    print(f"  replaced {n_moe_replaced} MoE blocks in {t2-t1:.1f}s")
    assert n_moe_replaced > 0, "Expected at least one Mistral4NaiveMoe block to replace"

    print(f"Step 3: replacing nn.Linear NVFP4 modules (MLA attention, shared expert)...")
    idx = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    # Mistral4 uses MLA — q_b_proj / kv_b_proj / o_proj as LoRA targets
    counts = replace_nvfp4_modules(
        model,
        MODEL_DIR,
        target_lora_suffixes=("q_b_proj", "kv_b_proj", "o_proj"),
        r=8,
        lora_alpha=16,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.bfloat16,
    )
    t3 = time.time()
    print(f"  replaced: {counts}; took {t3-t2:.1f}s")
    # Mistral-Small-4 has 36 layers; 36 × 3 MLA projections targeted = 108 LoRA modules expected.
    assert counts["lora"] == 108, (
        f"Expected 108 LoRA modules (36 layers × 3 MLA projections q_b_proj/kv_b_proj/o_proj); "
        f"got {counts['lora']}. Suffix list may be wrong (Mistral4 uses MLA names, NOT q_proj/v_proj)."
    )

    print(f"Step 4: assembling routed-expert NVFP4 buffers from CT keys...")
    translate, st_prefix, model_prefix = make_key_translator(model, MODEL_DIR)
    n_assembled = 0
    for name, module in model.named_modules():
        if not isinstance(module, NVFP4Experts3D):
            continue
        # In-memory:  model.language_model.layers.X.mlp.experts
        # Safetensors: language_model.model.layers.X.mlp.experts
        assert name.startswith("model.language_model."), f"Unexpected in-memory path: {name}"
        st_name = "language_model.model." + name[len("model.language_model."):]
        assemble_nvfp4_experts3d_batched(module, st_name, MODEL_DIR, wm)
        n_assembled += 1
        if n_assembled % 6 == 0:
            print(f"  assembled {n_assembled}/{n_moe_replaced} MoE blocks ({(time.time()-t3)/n_assembled:.1f}s/block)")
    t4 = time.time()
    print(f"  assembled {n_assembled} MoE blocks in {t4-t3:.1f}s")

    print(f"Step 5: loading non-NVFP4 weights (embeddings, norms, lm_head)...")
    n_loaded = load_non_nvfp4_weights(
        model, MODEL_DIR,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.bfloat16,
    )
    t5 = time.time()
    print(f"  loaded {n_loaded} non-NVFP4 tensors in {t5-t4:.1f}s")

    print(f"Step 6: assigning dequant workspaces...")
    workspace_pool = _assign_dequant_workspaces(
        model,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.bfloat16,
    )
    print(f"  {len(workspace_pool)} unique workspace shapes")

    print(f"Step 7: checking for meta tensors...")
    meta_params = [name for name, p in model.named_parameters() if p.is_meta]
    meta_buffers = [name for name, b in model.named_buffers() if b.is_meta]
    # Vision branch — Pixtral encoder, multi_modal_projector — intentionally skipped via translator
    def _is_vision(n: str) -> bool:
        return "vision_tower" in n or "multi_modal_projector" in n
    meta_params_text = [n for n in meta_params if not _is_vision(n)]
    meta_buffers_text = [n for n in meta_buffers if not _is_vision(n)]
    assert not meta_params_text, f"Non-vision params still on meta: {meta_params_text[:5]}"
    assert not meta_buffers_text, f"Non-vision buffers still on meta: {meta_buffers_text[:5]}"

    print(f"Step 7.5: freezing vision encoder (text-only training)...")
    n_frozen = 0
    for attr_name in ("vision_tower", "multi_modal_projector"):
        sub = getattr(model.model, attr_name, None)
        if sub is None:
            continue
        for p in sub.parameters():
            p.requires_grad = False
            n_frozen += 1
    print(f"  froze {n_frozen} vision/projector params")

    print(f"Step 7.6: moving CPU-side buffers (RoPE inv_freq etc.) to target device...")
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_moved = 0
    for mod in model.modules():
        for name, buf in list(mod.named_buffers(recurse=False)):
            if buf.device.type == "cpu":
                mod._buffers[name] = buf.to(target_device)
                n_moved += 1
        for name, par in list(mod.named_parameters(recurse=False)):
            if par.device.type == "cpu":
                mod._parameters[name] = torch.nn.Parameter(par.data.to(target_device), requires_grad=par.requires_grad)
                n_moved += 1
    print(f"  moved {n_moved} CPU tensors to {target_device}")

    print(f"Step 8: 1-token forward (model.train() to skip eval-cache; pure attention + MoE — no SSM)...")
    model.train()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device=target_device)
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits
    print(f"  logits shape: {logits.shape}, dtype: {logits.dtype}")
    assert torch.isfinite(logits).all(), "Forward produced non-finite logits"

    # Step 8.5: SEMANTIC sanity. The earlier 'finite logits' check passed cleanly while the
    # quant was 10^10× off and the model was outputting uniform probabilities — silent garbage.
    # A small token-level greedy probe catches this class of bug for ~3 s of wall clock.
    print(f"Step 8.5: greedy semantic sanity ('The capital of France is' → ' Paris')...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    probe_ids = tok.encode("The capital of France is", return_tensors="pt").to(target_device)
    with torch.no_grad():
        out2 = model(probe_ids)
    logits2 = out2.logits[0, -1, :].float()
    # The std of healthy LM logits is O(1); uniform-output logits have std ~ 0.
    logits_std = logits2.std().item()
    top_idx = int(logits2.argmax().item())
    top_token = tok.decode([top_idx])
    top_prob = torch.softmax(logits2, dim=-1)[top_idx].item()
    print(f"  logits std={logits_std:.3f}  top-1 token={top_token!r}  p={top_prob:.4f}")
    # Two cheap signals: (1) logits aren't uniform; (2) top-1 is ' Paris'. We require BOTH —
    # a healthy model has logits std ~ 2 and confidently emits ' Paris' (~0.7 at NVFP4A16).
    assert logits_std > 0.5, (
        f"Logits std={logits_std:.3f} — output looks near-uniform. "
        f"Likely cause: NVFP4 dequant format mismatch (CT vs ModelOpt). "
        f"See nvfp4_lora/dequant.py's format= switch."
    )
    assert top_token.strip().lower() == "paris", (
        f"Top-1 token is {top_token!r}, expected ' Paris'. "
        f"The model is loading + running but its predictions are wrong — investigate."
    )
    t6 = time.time()
    print(f"  forward + semantic check in {t6-t5:.1f}s; total elapsed {t6-t0:.1f}s")

    print("\n=== PHASE 0.5 MISTRAL ACCEPTANCE GATE: PASSED ===")


if __name__ == "__main__":
    test_phase0_5_mistral_full_load_and_one_token_forward()
