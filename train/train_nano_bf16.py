#!/usr/bin/env python3
"""Day 5c: bf16 Nano-30B LoRA training on ICH v3.1 - quant ablation baseline for Day 5b.

Mirror of `train_nano_nvfp4.py` but loads the bf16 base (no NVFP4 quantization) and
uses stock PEFT LoRA instead of `NVFP4LoRALinear`. Identical hyperparams + data + chat
template + max_len so the only delta between Day 5b and Day 5c is the weight precision
on the frozen base.

Result lets us argue the NVFP4 quantization tax for LoRA FT on Nemotron-3 MoE: compare
final loss + downstream eval delta between:
  - Day 5b adapter on NVFP4 base (the publishable path on a Spark)
  - Day 5c adapter on bf16 base  (the unquantized reference)

Run AFTER Day 5b finishes - the GPU is single-tenant.

Memory budget on Spark (Nano-30B bf16):
  weights ~60 GB + grad-ckpt activations ~10 GB + LoRA optim state ~3 GB = ~75 GB / 120 GB
  comfortable headroom for max_len=1536, batch_size=1, grad_accum=4.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Larger contiguous VA ranges reduce NVRM descriptor pool churn during model
# load on GB10 unified memory. Must be set before torch.cuda is initialized.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


MODEL_DIR = "/path/to/Models/Nemotron-3-Nano-30B-A3B-BF16"
TRAIN_FILE = "/path/to/datasets/ich_v3_1_final_1272rows/ich_v3_1_final.train.jsonl"
ADAPTER_DIR = "/path/to/adapters/nemotron_3_nano_bf16_lora_ichv31_1epoch"

TARGET_MODULES = ["up_proj", "down_proj"]
R = 8
LORA_ALPHA = 16
MAX_LEN = 1536
GRAD_ACCUM = 4
LR = 1e-4
N_EPOCHS = 1
SAVE_EVERY = 200


def save_progress(save_dir, loss_history, step, epoch):
    with open(f"{save_dir}/training_progress.json", "w") as f:
        json.dump({
            "step": step, "epoch": epoch,
            "loss_history_tail_500": loss_history[-500:],
            "total_steps_logged": len(loss_history),
            "config": {
                "base_dtype": "bf16",
                "max_len": MAX_LEN, "grad_accum": GRAD_ACCUM, "lr": LR,
                "n_epochs": N_EPOCHS, "r": R, "lora_alpha": LORA_ALPHA,
                "targets": TARGET_MODULES,
            },
        }, f, indent=2)


def main():
    print("=== Day 5c: bf16 Nano-30B LoRA on ICH v3.1, 1 epoch, max_len=1536 (quant ablation baseline) ===")
    device = torch.device("cuda")
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    free_g = torch.cuda.mem_get_info()
    print(f"baseline: cuda free={free_g[0]/1e9:.1f} GB / total={free_g[1]/1e9:.1f} GB")

    print("\n--- loading bf16 model ---")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda",
    )
    print(f"  load wall: {time.time()-t0:.1f}s, cuda_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB")

    print("\n--- wrapping with PEFT LoRA ---")
    lora_cfg = LoraConfig(
        r=R, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
        bias="none",
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    print("  gradient checkpointing: enabled")

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
                model.save_pretrained(ADAPTER_DIR)
                save_progress(ADAPTER_DIR, loss_history, step, epoch)
                print(f"  [checkpoint @ step {step}] saved adapter -> {ADAPTER_DIR}/")

    if step % GRAD_ACCUM != 0:
        opt.step()
        opt.zero_grad(set_to_none=True)
    model.save_pretrained(ADAPTER_DIR)
    save_progress(ADAPTER_DIR, loss_history, step, epoch)
    print(f"\n=== Day 5c DONE: {step} steps, wall={(time.time()-t_start)/3600:.2f}h ===")
    print(f"final adapter: {ADAPTER_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
