#!/usr/bin/env python3
"""Phase 0.5 acceptance gate — full Qwen3.5 model load through our custom loader path.

This is the MOMENT-OF-TRUTH test for Phase 0. It:
1. Builds the Qwen3.5-122B-A10B-NVFP4 model on meta via init_empty_weights
2. Replaces Qwen3_5MoeExperts (fused 3D) with NVFP4Experts3D (our quantized container)
3. Replaces NVFP4 nn.Linear instances (attention, shared_expert) with NVFP4LoRALinear
4. Assembles routed expert weights from per-expert CT safetensors keys
5. Loads non-NVFP4 weights (embeddings, norms)
6. Asserts no meta tensors remain
7. Runs a 1-token forward, asserts non-NaN

This is slow (~5-10 minutes per run due to ~60 GB of weights to load).
Mark with `slow` so it can be skipped in normal smoke runs.

Run explicitly:
    pytest smoke_tests/test_phase0_5_qwen35_full_load.py -v --runslow
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


MODEL_DIR = Path("/home/veritan-spark-01/Veritan/Models/Qwen3.5-122B-A10B-NVFP4")


def _require_model():
    if not (MODEL_DIR / "model.safetensors.index.json").exists():
        pytest.skip(f"Qwen3.5 model not at {MODEL_DIR}")


def test_phase0_5_qwen35_full_load_and_one_token_forward():
    """The Phase 0 acceptance gate for Qwen3.5."""
    _require_model()
    # Skip unless explicit env var, to avoid slow runs in normal pytest
    if not os.environ.get("RUN_PHASE0_5_FULL_LOAD"):
        pytest.skip("Set RUN_PHASE0_5_FULL_LOAD=1 to run the full-load acceptance gate (slow, ~60 GB)")

    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
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
    print(f"\n=== Phase 0.5 acceptance gate: Qwen3.5-122B-A10B-NVFP4 ===")
    print(f"Step 1: building model on meta...")
    cfg = AutoConfig.from_pretrained(str(MODEL_DIR))
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg)
    t1 = time.time()
    print(f"  built in {t1-t0:.1f}s; model_type={model.config.model_type}")

    print(f"Step 2: replacing fused-3D MoE blocks with NVFP4Experts3D...")
    n_moe_replaced = replace_moe_experts_with_nvfp4_3d(model, model_family="qwen3_5_moe_text")
    t2 = time.time()
    print(f"  replaced {n_moe_replaced} MoE blocks in {t2-t1:.1f}s")
    assert n_moe_replaced > 0, "Expected at least one MoE block to replace"

    print(f"Step 3: replacing nn.Linear NVFP4 modules (attention, shared_expert)...")
    idx = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    counts = replace_nvfp4_modules(
        model,
        MODEL_DIR,
        target_lora_suffixes=("q_proj", "v_proj", "o_proj"),
        r=8,
        lora_alpha=16,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.bfloat16,
    )
    t3 = time.time()
    print(f"  replaced (LoRA targets + frozen NVFP4 records): {counts}; took {t3-t2:.1f}s")

    print(f"Step 4: assembling routed-expert NVFP4 buffers from CT keys (this is the slow part)...")
    translate, st_prefix, model_prefix = make_key_translator(model, MODEL_DIR)
    n_assembled = 0
    for name, module in model.named_modules():
        if not isinstance(module, NVFP4Experts3D):
            continue
        # Reverse-translate: in-memory `model.layers.X.mlp.experts` -> st `model.language_model.layers.X.mlp.experts`
        # The MoE block lives at `model.layers.X.mlp.experts` in memory.
        # Its safetensors prefix mirrors with language_model in the middle.
        assert name.startswith("model.layers."), name
        st_name = "model.language_model." + name[len("model."):]
        assemble_nvfp4_experts3d_batched(module, st_name, MODEL_DIR, wm)
        n_assembled += 1
        if n_assembled % 8 == 0:
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
    # Vision branch is intentionally meta — we skip it via translator
    meta_params_non_vision = [n for n in meta_params if "visual" not in n]
    meta_buffers_non_vision = [n for n in meta_buffers if "visual" not in n]
    if meta_params_non_vision:
        print(f"  WARNING: {len(meta_params_non_vision)} non-vision params still on meta")
        print(f"    samples: {meta_params_non_vision[:5]}")
    if meta_buffers_non_vision:
        print(f"  WARNING: {len(meta_buffers_non_vision)} non-vision buffers still on meta")
        print(f"    samples: {meta_buffers_non_vision[:5]}")
    assert not meta_params_non_vision, f"Non-vision params still on meta: {meta_params_non_vision[:3]}"
    assert not meta_buffers_non_vision, f"Non-vision buffers still on meta: {meta_buffers_non_vision[:3]}"

    print(f"Step 7.5: moving CPU-side buffers (RoPE inv_freq etc.) to target device...")
    # NVFP4 buffers + LoRA params + loaded non-NVFP4 weights are already on the target
    # device (we passed device=cuda to replace_nvfp4_modules and load_non_nvfp4_weights).
    # The remaining CPU-side tensors are typically: RoPE inv_freq (computed in __init__,
    # never in safetensors). A blanket `model.to(device)` OOMs because it walks every
    # tensor at once. Targeted: move only buffers that are currently on CPU.
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

    print(f"Step 8: 1-token forward...")
    # train() mode skips NVFP4LoRALinear's bf16 _eval_weight cache (which can grow to
    # 30 GB across 192 linear modules). For a one-off forward sanity check, per-call
    # dequant uses much less peak memory. Acceptance gate doesn't need gradients —
    # just that forward produces finite logits.
    model.train()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device=target_device)
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits
    print(f"  logits shape: {logits.shape}, dtype: {logits.dtype}")
    assert torch.isfinite(logits).all(), "Forward produced non-finite logits"
    t6 = time.time()
    print(f"  1-token forward in {t6-t5:.1f}s; total elapsed {t6-t0:.1f}s")

    print("\n=== PHASE 0.5 ACCEPTANCE GATE: PASSED ===")


if __name__ == "__main__":
    test_phase0_5_qwen35_full_load_and_one_token_forward()
