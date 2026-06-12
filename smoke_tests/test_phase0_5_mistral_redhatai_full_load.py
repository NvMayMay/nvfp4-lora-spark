#!/usr/bin/env python3
"""Phase 0.5 acceptance — full load of the RedHatAI-quanted Mistral-Small-4-119B-2603
converted to HuggingFace layout (`...-NVFP4-HF`).

Architecture this artifact uses (different from our v1 in-house quant):
  - Attention modules (q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj) are
    kept BF16 — they are plain `nn.Linear` after loading. LoRA is attached with
    standard PEFT (`LoraConfig` + `get_peft_model`), no custom NVFP4LoRALinear.
  - Routed MoE experts and shared experts are NVFP4 (weight + dynamic-activation)
    and load via NVFP4Experts3D / NVFP4LoRALinear paths as before.

Result of this test = portable training base: an adapter trained against this exact
artifact is byte-equivalent to one trained against RedHatAI's consolidated checkpoint,
because the underlying weights are identical (verify_conversion.py confirmed 15/15
spot checks bit-identical).

Run:
    cd /home/veritan-spark-01/Veritan/Sandbox/repos/nvfp4-lora-spark
    RUN_PHASE0_5_FULL_LOAD=1 \\
        /home/veritan-spark-01/Veritan/.venvs/qwen-serve/bin/python -u \\
        smoke_tests/test_phase0_5_mistral_redhatai_full_load.py
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


MODEL_DIR = Path("/home/veritan-spark-01/Veritan/Models/RedHatAI-Mistral-Small-4-119B-2603-NVFP4-HF")


def _require_model():
    if not (MODEL_DIR / "model.safetensors.index.json").exists():
        pytest.skip(f"Converted RedHatAI Mistral HF artifact not at {MODEL_DIR}; "
                    f"run scripts/convert_mistral_consolidated_to_hf.py first")


def test_phase0_5_mistral_redhatai_full_load_and_one_token_forward():
    """Phase 0.5 acceptance gate for the RedHatAI-quanted Mistral, HF layout."""
    _require_model()
    if not os.environ.get("RUN_PHASE0_5_FULL_LOAD"):
        pytest.skip("Set RUN_PHASE0_5_FULL_LOAD=1 to run the full-load acceptance gate")

    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
    from nvfp4_lora.experts import (
        replace_moe_experts_with_nvfp4_3d,
        assemble_nvfp4_experts3d_batched,
        NVFP4Experts3D,
    )
    from nvfp4_lora.loader import (
        replace_nvfp4_modules,
        load_non_nvfp4_weights,
        _assign_dequant_workspaces,
    )

    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    print(f"\n=== Phase 0.5 acceptance: RedHatAI Mistral-Small-4-119B-NVFP4 (HF layout) ===")

    print(f"Step 1: building model on meta…")
    cfg = AutoConfig.from_pretrained(str(MODEL_DIR))
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(cfg)
    t1 = time.time()
    print(f"  built in {t1-t0:.1f}s; model_type={model.config.model_type}")

    print(f"Step 2: replacing fused-3D MoE blocks (NVFP4) with NVFP4Experts3D…")
    family = getattr(model.config, "model_type", "mistral4")
    n_moe = replace_moe_experts_with_nvfp4_3d(model, model_family=family)
    t2 = time.time()
    print(f"  replaced {n_moe} MoE blocks in {t2-t1:.1f}s")
    assert n_moe == 36, f"Expected 36 MoE blocks, got {n_moe}"

    print(f"Step 3: replacing NVFP4 nn.Linear (shared_experts only — attention is BF16)…")
    idx = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    counts = replace_nvfp4_modules(
        model, MODEL_DIR,
        # No NVFP4 attention modules in this artifact, so target_lora_suffixes is empty —
        # this only iterates shared_experts (which ARE NVFP4) and wraps them as FROZEN
        # NVFP4 (r=0). LoRA on attention is attached separately via PEFT in step 7.
        target_lora_suffixes=(),
        r=0, lora_alpha=0,
        device=target_device, dtype=torch.bfloat16,
    )
    t3 = time.time()
    print(f"  replaced: {counts}; took {t3-t2:.1f}s")
    # Mistral-Small-4 has 36 layers × 3 shared_expert projections (gate/up/down) = 108
    # frozen NVFP4 shared-expert modules. Routed experts go through NVFP4Experts3D in step 2.
    assert counts["frozen_nvfp4"] == 108, (
        f"Expected 108 frozen NVFP4 shared-expert modules (36 layers × 3 projections); "
        f"got {counts['frozen_nvfp4']}. "
        f"Did the conversion key-mapping miss something?"
    )
    assert counts["lora"] == 0, "No NVFP4 LoRA targets in RedHatAI recipe (attention is BF16)"

    print(f"Step 4: assembling routed-expert NVFP4 buffers…")
    n_assembled = 0
    for name, module in model.named_modules():
        if not isinstance(module, NVFP4Experts3D):
            continue
        # In-memory: model.language_model.layers.X.mlp.experts
        # Safetensors: language_model.model.layers.X.mlp.experts
        assert name.startswith("model.language_model."), f"Unexpected path: {name}"
        st_name = "language_model.model." + name[len("model.language_model."):]
        assemble_nvfp4_experts3d_batched(module, st_name, MODEL_DIR, wm)
        n_assembled += 1
    t4 = time.time()
    print(f"  assembled {n_assembled} MoE blocks in {t4-t3:.1f}s")

    print(f"Step 5: loading non-NVFP4 weights (BF16 attention + embeddings + norms + lm_head)…")
    n_loaded = load_non_nvfp4_weights(
        model, MODEL_DIR,
        device=target_device, dtype=torch.bfloat16,
    )
    t5 = time.time()
    print(f"  loaded {n_loaded} BF16 tensors in {t5-t4:.1f}s")

    print(f"Step 6: assigning NVFP4 dequant workspaces (for shared experts)…")
    workspace_pool = _assign_dequant_workspaces(
        model, device=target_device, dtype=torch.bfloat16,
    )
    print(f"  {len(workspace_pool)} unique workspace shapes")

    print(f"Step 7: checking for meta tensors (vision tower allowed)…")
    def _is_vision(n: str) -> bool:
        return "vision_tower" in n or "multi_modal_projector" in n
    meta_params_text = [n for n, p in model.named_parameters() if p.is_meta and not _is_vision(n)]
    meta_buffers_text = [n for n, b in model.named_buffers() if b.is_meta and not _is_vision(n)]
    assert not meta_params_text, f"Non-vision params on meta: {meta_params_text[:5]}"
    assert not meta_buffers_text, f"Non-vision buffers on meta: {meta_buffers_text[:5]}"

    print(f"Step 7.5: freezing vision encoder + multi_modal_projector…")
    n_frozen = 0
    for attr in ("vision_tower", "multi_modal_projector"):
        sub = getattr(model.model, attr, None)
        if sub is None:
            continue
        for p in sub.parameters():
            p.requires_grad = False
            n_frozen += 1
    print(f"  froze {n_frozen} vision/projector params")

    print(f"Step 7.6: migrating CPU-side buffers (RoPE inv_freq etc.) to {target_device}…")
    n_moved = 0
    for mod in model.modules():
        for nm, buf in list(mod.named_buffers(recurse=False)):
            if buf.device.type == "cpu":
                mod._buffers[nm] = buf.to(target_device)
                n_moved += 1
        for nm, par in list(mod.named_parameters(recurse=False)):
            if par.device.type == "cpu":
                mod._parameters[nm] = torch.nn.Parameter(
                    par.data.to(target_device), requires_grad=par.requires_grad
                )
                n_moved += 1
    print(f"  moved {n_moved} CPU tensors to {target_device}")

    print(f"Step 8: attaching PEFT LoRA to BF16 attention modules…")
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_b_proj", "kv_b_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {n_trainable:,} / {n_total:,} = {n_trainable/n_total*100:.4f}%")
    # 36 layers × 3 MLA projections × (rank=8 × in_features × 2 LoRA dirs)
    # q_b_proj: in=1024 out=4096 → A:(8,1024)+B:(4096,8) = 40960 params
    # kv_b_proj: in=256 out=6144 → A:(8,256)+B:(6144,8) = 51200
    # o_proj: in=4096 out=4096 → A:(8,4096)+B:(4096,8) = 65536
    # per layer: 157696. 36 layers: 5,677,056. Approximately. Sanity check it's >1M.
    assert 1_000_000 < n_trainable < 20_000_000, (
        f"trainable count {n_trainable} outside expected window for r=8 on 36×3 MLA modules"
    )

    print(f"Step 9: 1-token forward (train mode to skip eval-cache; pure attention + MoE — no SSM)…")
    model.train()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device=target_device)
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits
    print(f"  logits shape: {logits.shape}, dtype: {logits.dtype}")
    assert torch.isfinite(logits).all(), "Forward produced non-finite logits"

    print(f"Step 9.5: greedy semantic sanity ('The capital of France is' → ' Paris')…")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    probe_ids = tok.encode("The capital of France is", return_tensors="pt").to(target_device)
    with torch.no_grad():
        out2 = model(probe_ids)
    logits2 = out2.logits[0, -1, :].float()
    logits_std = logits2.std().item()
    top_idx = int(logits2.argmax().item())
    top_token = tok.decode([top_idx])
    top_prob = torch.softmax(logits2, dim=-1)[top_idx].item()
    print(f"  logits std={logits_std:.3f}  top-1 token={top_token!r}  p={top_prob:.4f}")
    assert logits_std > 0.5, f"Logits std={logits_std:.3f} — output looks near-uniform"
    assert top_token.strip().lower() == "paris", (
        f"Top-1 token is {top_token!r}, expected ' Paris'."
    )
    t6 = time.time()
    print(f"  forward + semantic check in {t6-t5:.1f}s; total elapsed {t6-t0:.1f}s")

    print("\n=== PHASE 0.5 (RedHatAI-quanted Mistral, HF layout): PASSED ===")


if __name__ == "__main__":
    test_phase0_5_mistral_redhatai_full_load_and_one_token_forward()
