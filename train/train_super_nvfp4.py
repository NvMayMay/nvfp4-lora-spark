#!/usr/bin/env python3
"""Day 5 production training: full Super-120B-NVFP4 LoRA on ICH v3.1.

Config (committed after Day 4.5 pre-flight):
- model: Nemotron-3-Super-120B-A12B-NVFP4
- LoRA targets: up_proj, down_proj (40961 NVFP4 modules; 76 FP8 shared-expert ones auto-demoted to frozen)
- r=8, lora_alpha=16, lora_dropout=0
- max_length=1536 (covers ~95% of ICH v3.1 train examples without truncation)
- gradient checkpointing enabled (essential at this seq length)
- batch_size=1, grad_accum=4 → effective batch 4
- AdamW, lr=1e-4
- 1 epoch over 1081 train examples
- Adapter checkpoint every 200 forward+backward steps (~25 min of training each)
- Expected wall time: 1081 × ~127 s = ~38 h
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import safetensors.torch as st

from nvfp4_lora.loader import load_nemotron_with_nvfp4_lora
from nvfp4_lora.linear import NVFP4LoRALinear


MODEL_DIR = "/path/to/Models/Nemotron-3-Super-120B-A12B-NVFP4"
TRAIN_FILE = "/path/to/datasets/ich_v3_1_final_1272rows/ich_v3_1_final.train.jsonl"
ADAPTER_DIR = "/path/to/adapters/nemotron_3_super_nvfp4_lora_ichv31_1epoch"

TARGET_SUFFIXES = ("up_proj", "down_proj")
R = 8
LORA_ALPHA = 16
MAX_LEN = 1536
GRAD_ACCUM = 4
LR = 1e-4
N_EPOCHS = 1
SAVE_EVERY = 200  # forward+backward steps


def save_adapter(model, save_dir, loss_history, step, epoch):
    os.makedirs(save_dir, exist_ok=True)
    state = {}
    for name, mod in model.named_modules():
        if isinstance(mod, NVFP4LoRALinear) and mod.r > 0:
            state[f"base_model.model.{name}.lora_A.weight"] = mod.lora_A.detach().cpu().contiguous()
            state[f"base_model.model.{name}.lora_B.weight"] = mod.lora_B.detach().cpu().contiguous()
    st.save_file(state, f"{save_dir}/adapter_model.safetensors")
    cfg = {
        "base_model_name_or_path": MODEL_DIR,
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "r": R, "lora_alpha": LORA_ALPHA, "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": list(TARGET_SUFFIXES),
        "inference_mode": True, "fan_in_fan_out": False,
    }
    with open(f"{save_dir}/adapter_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    with open(f"{save_dir}/training_progress.json", "w") as f:
        json.dump({
            "step": step, "epoch": epoch,
            "loss_history_tail_500": loss_history[-500:],
            "total_steps_logged": len(loss_history),
            "config": {
                "max_len": MAX_LEN, "grad_accum": GRAD_ACCUM, "lr": LR,
                "n_epochs": N_EPOCHS, "r": R, "lora_alpha": LORA_ALPHA,
                "targets": list(TARGET_SUFFIXES),
            },
        }, f, indent=2)
    return len(state)


def main():
    print("=== Day 5 production: Super-120B LoRA on ICH v3.1, 1 epoch, max_len=1536 ===")
    device = torch.device("cuda")
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    free_g = torch.cuda.mem_get_info()
    print(f"baseline: cuda free={free_g[0]/1e9:.1f} GB / total={free_g[1]/1e9:.1f} GB")

    print("\n--- loading model ---")
    t0 = time.time()
    model = load_nemotron_with_nvfp4_lora(
        MODEL_DIR,
        target_lora_suffixes=TARGET_SUFFIXES,
        r=R, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
        device=device, dtype=torch.bfloat16,
    )
    print(f"  load wall: {time.time()-t0:.1f}s, cuda_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB")

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        for m in model.modules():
            if hasattr(m, "gradient_checkpointing"):
                m.gradient_checkpointing = True
    print("  gradient checkpointing: enabled")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"\n--- loading training data from {TRAIN_FILE} ---")
    train_examples = []
    with open(TRAIN_FILE) as f:
        for line in f:
            train_examples.append(json.loads(line))
    print(f"  loaded {len(train_examples)} train examples")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    opt = torch.optim.AdamW(trainable, lr=LR)
    print(f"  optimizer: AdamW lr={LR} on {n_trainable/1e6:.2f}M trainable params across {len(trainable)} tensors")

    print(f"\n--- training {N_EPOCHS} epoch x {len(train_examples)} examples (max_len={MAX_LEN}, grad_accum={GRAD_ACCUM}, save_every={SAVE_EVERY}) ---")
    model.train()
    loss_history: list[float] = []
    t_start = time.time()
    step = 0
    total_steps = len(train_examples) * N_EPOCHS

    for epoch in range(N_EPOCHS):
        for ex_idx, ex in enumerate(train_examples):
            text = tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
            enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
            labels = enc.input_ids.clone()

            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=labels)
            scaled_loss = out.loss / GRAD_ACCUM
            scaled_loss.backward()
            loss_history.append(float(out.loss.item()))

            step += 1
            if step % GRAD_ACCUM == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)

            if step % 10 == 0 or step == 1:
                elapsed = time.time() - t_start
                eta_h = (elapsed / step) * (total_steps - step) / 3600
                avg_loss = sum(loss_history[-20:]) / max(1, len(loss_history[-20:]))
                print(
                    f"  step {step}/{total_steps}: loss={loss_history[-1]:.4f} avg20={avg_loss:.4f} "
                    f"elapsed={elapsed/60:.1f}m eta={eta_h:.2f}h "
                    f"cuda_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB",
                    flush=True,
                )

            if step % SAVE_EVERY == 0:
                n_saved = save_adapter(model, ADAPTER_DIR, loss_history, step, epoch)
                print(f"  [checkpoint @ step {step}] saved {n_saved} LoRA tensors -> {ADAPTER_DIR}/")

    # final flush + save
    if step % GRAD_ACCUM != 0:
        opt.step()
        opt.zero_grad(set_to_none=True)
    save_adapter(model, ADAPTER_DIR, loss_history, step, epoch)
    print(f"\n=== Day 5 DONE: {step} steps, wall={(time.time()-t_start)/3600:.2f}h ===")
    print(f"final adapter: {ADAPTER_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
