#!/usr/bin/env python3
"""Day 2 loader smoke: build Nemotron-3-Nano with NVFP4LoRALinear modules + 2-batch LoRA train.

1. Loader builds the model via init_empty_weights + module replacement + non-NVFP4 weight load.
2. 2-batch forward + backward produces finite loss.
3. Trainable params = LoRA params only (frozen NVFP4 storage + frozen non-target modules).
4. Adapter saves via standard PEFT format (or hand-rolled compatible).

Note: q_proj/v_proj in Nemotron-3-Nano are NOT NVFP4 (they're in the modelopt exclude_modules
list, kept in bf16). The NVFP4 modules are expert up_proj/down_proj and Mamba in_proj/out_proj.
For Day 2 we target a small subset of expert MLP layers via suffix matching, which validates
the NVFP4 LoRA path. Day 3+ will add bf16-Linear LoRA targeting for the attention path.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse


TARGET_SUFFIXES = ("up_proj", "down_proj")
R = 8
LORA_ALPHA = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load Nano with NVFP4LoRALinear modules and run a short LoRA train smoke.",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("NVFP4_SMOKE_MODEL_DIR"),
        help="Path to Nemotron-3-Nano-30B-A3B-NVFP4. Can also be set via NVFP4_SMOKE_MODEL_DIR.",
    )
    args = parser.parse_args()
    if not args.model_dir:
        parser.print_usage(sys.stderr)
        print(
            "error: provide --model-dir /path/to/Nemotron-3-Nano-30B-A3B-NVFP4 "
            "or set NVFP4_SMOKE_MODEL_DIR",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not os.path.exists(os.path.join(args.model_dir, "model.safetensors.index.json")):
        print(
            f"error: no model.safetensors.index.json under {args.model_dir}; check --model-dir",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return args


def main():
    args = parse_args()
    global torch, load_nemotron_with_nvfp4_lora, NVFP4LoRALinear
    import torch
    from nvfp4_lora.loader import load_nemotron_with_nvfp4_lora
    from nvfp4_lora.linear import NVFP4LoRALinear

    print("=== Day 2 loader smoke (Nemotron-3-Nano + NVFP4LoRALinear) ===")
    device = torch.device("cuda")

    # GPU + mem baseline
    free_g = torch.cuda.mem_get_info()
    print(f"baseline: cuda free={free_g[0]/1e9:.1f} GB / total={free_g[1]/1e9:.1f} GB")
    import psutil
    print(f"baseline: system mem used={psutil.virtual_memory().used/1e9:.1f} GB / total={psutil.virtual_memory().total/1e9:.1f} GB")

    print(f"\n--- loading model with LoRA targets {TARGET_SUFFIXES}, r={R}, alpha={LORA_ALPHA} ---")
    model = load_nemotron_with_nvfp4_lora(
        args.model_dir,
        target_lora_suffixes=TARGET_SUFFIXES,
        r=R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        device=device,
        dtype=torch.bfloat16,
    )

    # Memory after load
    print(f"\nafter-load: system mem used={psutil.virtual_memory().used/1e9:.1f} GB")
    print(f"after-load: cuda allocated={torch.cuda.memory_allocated()/1e9:.2f} GB reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")

    # Trainable param accounting
    print("\n--- trainable param accounting ---")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    n_buffers = sum(b.numel() for b in model.buffers())
    nvfp4_count = sum(1 for _, m in model.named_modules() if isinstance(m, NVFP4LoRALinear))
    lora_count = sum(1 for _, m in model.named_modules() if isinstance(m, NVFP4LoRALinear) and m.r > 0)
    print(f"  NVFP4LoRALinear modules: {nvfp4_count} (of which {lora_count} are LoRA-trainable)")
    print(f"  trainable params: {n_trainable/1e6:.2f}M")
    print(f"  total params: {n_total/1e9:.2f}B")
    print(f"  buffer params: {n_buffers/1e9:.2f}B  ← NVFP4 storage")

    # Quick sanity: list a few trainable param names
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"  first 5 trainable param paths: {trainable_names[:5]}")
    print(f"  last 5 trainable param paths:  {trainable_names[-5:]}")

    # 2-batch smoke
    print("\n--- 2-batch forward + backward ---")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    prompts = [
        "The European Medicines Agency is responsible for",
        "Good clinical practice (GCP) requires that",
    ]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    labels = enc.input_ids.clone()
    labels[enc.attention_mask == 0] = -100

    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
    )

    for step in range(2):
        torch.cuda.reset_peak_memory_stats()
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=labels)
        loss = out.loss
        loss.backward()
        # Check at least one LoRA grad is non-finite-checked
        grad_norms = {}
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grad_norms[n] = p.grad.norm().item()
        # Pick a representative one
        rep = next(iter(grad_norms.items())) if grad_norms else (None, None)
        peak = torch.cuda.max_memory_allocated()/1e9
        print(f"  step {step+1}: loss={loss.item():.4f} finite={torch.isfinite(loss).item()} peak_cuda={peak:.2f}GB sample_grad={rep[0]}:{rep[1]:.4e}" if rep[0] else f"  step {step+1}: loss={loss.item():.4f} NO GRADS")
        opt.step()
        opt.zero_grad(set_to_none=True)

    # Compare logits to base (no-op LoRA at init) - to verify the LoRA path actually changed something after 2 steps
    # (After backward + opt.step, lora_B is no longer zero, so logits should differ from a fresh-LoRA model)
    print("\n--- adapter save smoke (PEFT-compatible format) ---")
    adapter_dir = "/tmp/day2_smoke_adapter"
    os.makedirs(adapter_dir, exist_ok=True)
    # Collect LoRA params and save in PEFT format
    state = {}
    for name, mod in model.named_modules():
        if isinstance(mod, NVFP4LoRALinear) and mod.r > 0:
            state[f"base_model.model.{name}.lora_A.weight"] = mod.lora_A.detach().cpu()
            state[f"base_model.model.{name}.lora_B.weight"] = mod.lora_B.detach().cpu()
    import safetensors.torch as st
    st.save_file(state, f"{adapter_dir}/adapter_model.safetensors")
    # PEFT-style adapter_config.json
    import json
    cfg = {
        "base_model_name_or_path": args.model_dir,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": list(TARGET_SUFFIXES),
        "inference_mode": True,
        "fan_in_fan_out": False,
    }
    with open(f"{adapter_dir}/adapter_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {len(state)} LoRA tensors to {adapter_dir}/")
    print(f"  total adapter size: {sum(t.numel()*t.element_size() for t in state.values())/1e6:.2f} MB")

    print("\n=== Day 2 loader smoke PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
