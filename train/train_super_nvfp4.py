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
- Expected wall time: 1081 x ~136 s = ~40 h on the v1.0 production config (b=1, ml=1536)
v1.0 published runs used unmasked labels; `--mask-prompt-labels` is the
recommended setting for new training and will produce different loss curves.

Optimizer state saving is enabled by default and can add roughly 10 GB for the
Super adapter. Use `--no-save-optimizer-state` if disk space is tighter than
resume fidelity.
"""
import sys, os, json, time, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reduce NVRM memdesc churn during the ~40K-allocation Super load on GB10 unified
# memory. Larger contiguous VA ranges = fewer descriptor pool entries. Must be set
# before torch.cuda is initialized.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import numpy as np
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


def save_adapter(
    model, save_dir, loss_history, step, epoch, batch, max_len, grad_accum,
    grad_accum_arg, mask_prompt_labels_enabled,
    optimizer=None, save_optimizer_state=True,
):
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
                "batch": batch, "max_len": max_len,
                "grad_accum": grad_accum, "grad_accum_arg": grad_accum_arg,
                "mask_prompt_labels": bool(mask_prompt_labels_enabled), "lr": LR,
                "n_epochs": N_EPOCHS, "r": R, "lora_alpha": LORA_ALPHA,
                "targets": list(TARGET_SUFFIXES),
            },
        }, f, indent=2)
    if save_optimizer_state and optimizer is not None:
        torch.save(optimizer.state_dict(), f"{save_dir}/optimizer_state.pt")
    rng_state = {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "random": random.getstate(),
        "numpy": np.random.get_state(),
    }
    torch.save(rng_state, f"{save_dir}/rng_state.pt")
    return len(state)


def load_adapter_weights(model, adapter_dir):
    ckpt_path = f"{adapter_dir}/adapter_model.safetensors"
    prog_path = f"{adapter_dir}/training_progress.json"
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"--resume-from requested but no checkpoint at {ckpt_path}")

    adapter_state = st.load_file(ckpt_path)
    loaded = 0
    for name, mod in model.named_modules():
        if isinstance(mod, NVFP4LoRALinear) and mod.r > 0:
            k_a = f"base_model.model.{name}.lora_A.weight"
            k_b = f"base_model.model.{name}.lora_B.weight"
            if k_a in adapter_state and k_b in adapter_state:
                mod.lora_A.data.copy_(adapter_state[k_a].to(mod.lora_A.device, mod.lora_A.dtype))
                mod.lora_B.data.copy_(adapter_state[k_b].to(mod.lora_B.device, mod.lora_B.dtype))
                loaded += 1

    loss_history = []
    progress_step = None
    progress_epoch = 0
    if os.path.exists(prog_path):
        try:
            with open(prog_path) as f:
                progress = json.load(f)
            progress_step = progress.get("step")
            progress_epoch = progress.get("epoch", 0)
            loss_history = progress.get("loss_history_tail_500", [])
            progress_config = progress.get("config", {})
        except (OSError, json.JSONDecodeError):
            progress_config = {}
    else:
        progress_config = {}

    return loaded, progress_step, progress_epoch, loss_history, progress_config


def validate_resume_config(saved_config, batch, max_len, grad_accum, grad_accum_arg, mask_prompt_labels_enabled, force):
    expected = {
        "batch": batch,
        "max_len": max_len,
        "grad_accum": grad_accum,
        "mask_prompt_labels": bool(mask_prompt_labels_enabled),
    }
    differences = []
    legacy_missing = []
    for key, current_value in expected.items():
        saved_value = saved_config.get(key)
        if saved_value is None:
            legacy_missing.append(key)
        if saved_value is not None and saved_value != current_value:
            differences.append(f"{key}: checkpoint={saved_value}, current={current_value}")
    saved_grad_accum_arg = saved_config.get("grad_accum_arg")
    if saved_grad_accum_arg is None:
        legacy_missing.append("grad_accum_arg")
    if saved_grad_accum_arg is not None and saved_grad_accum_arg != grad_accum_arg:
        print(
            f"  WARNING: checkpoint grad_accum_arg={saved_grad_accum_arg}, "
            f"current --grad-accum={grad_accum_arg}; effective grad_accum remains {grad_accum}."
        )
    if legacy_missing:
        print(
            "  WARNING: checkpoint pre-dates resume validation fields "
            f"({', '.join(legacy_missing)}); config validation is not exhaustive. "
            "Resume will proceed normally; pass --force-config-mismatch only if you "
            'also see a hard "config differs" error.'
        )
    if differences and not force:
        raise SystemExit(
            "Checkpoint config differs from current CLI values: "
            + "; ".join(differences)
            + ". Resume with the original config or pass --force-config-mismatch."
        )
    if differences:
        print("  WARNING: forcing resume despite checkpoint config mismatch: " + "; ".join(differences))


def load_resume_state(adapter_dir, optimizer):
    opt_path = f"{adapter_dir}/optimizer_state.pt"
    rng_path = f"{adapter_dir}/rng_state.pt"
    optimizer_loaded = False
    rng_loaded = False

    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=False))
        optimizer_loaded = True

    if os.path.exists(rng_path):
        rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state(rng_state["torch_cuda"])
        random.setstate(rng_state["random"])
        np.random.set_state(rng_state["numpy"])
        rng_loaded = True

    return optimizer_loaded, rng_loaded


def _assistant_response_start_char(example, text):
    for msg in reversed(example["messages"]):
        if msg.get("role") == "assistant":
            content = msg.get("content") or ""
            if content:
                idx = text.rfind(content)
                if idx >= 0:
                    return idx
            break

    markers = ["<|assistant|>", "<|im_start|>assistant", "Assistant:", "assistant\n"]
    marker_hits = [(text.rfind(marker), marker) for marker in markers]
    marker_hits = [(idx, marker) for idx, marker in marker_hits if idx >= 0]
    if marker_hits:
        idx, marker = max(marker_hits, key=lambda item: item[0])
        return idx + len(marker)
    raise ValueError("could not find assistant response boundary in rendered chat template")


def mask_prompt_labels(labels, texts, examples, tokenizer, max_len):
    for row_idx, (text, example) in enumerate(zip(texts, examples)):
        start_char = _assistant_response_start_char(example, text)
        prefix_ids = tokenizer(text[:start_char], truncation=True, max_length=max_len)["input_ids"]
        labels[row_idx, :min(len(prefix_ids), labels.shape[1])] = -100


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1,
                    help="physical batch size; batch > 1 enables padded batched training")
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--grad-accum", type=int, default=GRAD_ACCUM,
                    help="gradient accumulation for the batch=1 compatibility path")
    ap.add_argument("--stop-at-step", type=int, default=None,
                    help="stop cleanly after this many forward/backward steps")
    ap.add_argument("--resume-from", type=int, default=None,
                    help="resume from this position using ADAPTER_DIR: example index for batch=1, batch index for batch>1")
    ap.add_argument("--force-config-mismatch", action="store_true",
                    help="allow resume when checkpoint batch/max_len/grad_accum differ from current CLI values")
    ap.add_argument("--mask-prompt-labels", action="store_true",
                    help="train only on assistant response tokens; v1.0 published runs left this off")
    ap.add_argument("--save-optimizer-state", action=argparse.BooleanOptionalAction, default=True,
                    help="save AdamW state for exact resume; disable to save disk space")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.batch < 1:
        raise SystemExit("--batch must be >= 1")
    if args.grad_accum < 1:
        raise SystemExit("--grad-accum must be >= 1")

    effective_grad_accum = args.grad_accum if args.batch == 1 else 1
    print(f"=== Day 5 production: Super-120B LoRA on ICH v3.1, 1 epoch, batch={args.batch}, max_len={args.max_len} ===")
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
    tok.padding_side = "right"

    print(f"\n--- loading training data from {TRAIN_FILE} ---")
    train_examples = []
    with open(TRAIN_FILE) as f:
        for line in f:
            train_examples.append(json.loads(line))
    print(f"  loaded {len(train_examples)} train examples")

    resume_step = 0
    resume_epoch = 0
    loss_history: list[float] = []
    if args.resume_from is not None:
        if args.resume_from < 0:
            raise SystemExit(f"--resume-from must be >= 0; got {args.resume_from}")
        print(f"\n--- resuming from step {args.resume_from} ---")
        loaded, progress_step, progress_epoch, restored_history, progress_config = load_adapter_weights(model, ADAPTER_DIR)
        validate_resume_config(
            progress_config,
            args.batch,
            args.max_len,
            effective_grad_accum,
            args.grad_accum,
            args.mask_prompt_labels,
            args.force_config_mismatch,
        )
        if progress_step is None and not args.force_config_mismatch:
            raise SystemExit(
                "ADAPTER_DIR contains an adapter but no training_progress.json. "
                "Cannot validate resume config against original training config. "
                "Pass --force-config-mismatch to resume anyway."
            )
        if progress_step is not None and progress_step != args.resume_from:
            print(
                f"  WARNING: --resume-from={args.resume_from} but checkpoint reports step={progress_step}; "
                f"using checkpoint value (it matches the saved optimizer/RNG state). "
                f"If you intended a different step, point ADAPTER_DIR at a different checkpoint."
            )
        resume_step = progress_step if progress_step is not None else args.resume_from
        resume_epoch = progress_epoch
        loss_history = list(restored_history)
        print(f"  loaded LoRA weights for {loaded} modules")
        print(f"  prior step={resume_step}, loss_history entries restored: {len(loss_history)}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    opt = torch.optim.AdamW(trainable, lr=LR)
    print(f"  optimizer: AdamW lr={LR} on {n_trainable/1e6:.2f}M trainable params across {len(trainable)} tensors")
    if args.resume_from is not None:
        opt_loaded, rng_loaded = load_resume_state(ADAPTER_DIR, opt)
        print(f"  resume state: optimizer_state={'loaded' if opt_loaded else 'missing'}, rng_state={'loaded' if rng_loaded else 'missing'}")

    max_loop_idx = len(train_examples) if args.batch == 1 else (len(train_examples) // args.batch)
    # Defensive: catches a manually-edited training_progress.json that claims epoch >= N_EPOCHS.
    # The script's own save path never writes such a state.
    if resume_epoch >= N_EPOCHS:
        raise SystemExit(
            f"Resume epoch ({resume_epoch}) >= N_EPOCHS ({N_EPOCHS}); checkpoint is already at end of training. "
            f"Nothing to train."
        )
    if resume_step >= max_loop_idx:
        raise SystemExit(
            f"--resume-from yields resume_step={resume_step} which is >= the {('example' if args.batch==1 else 'batch')} count "
            f"({max_loop_idx}) for this dataset+batch config. Nothing to train. "
            f"Check ADAPTER_DIR or pass a different --batch."
        )

    if args.batch == 1:
        total_steps = len(train_examples) * N_EPOCHS
        print(f"\n--- training {N_EPOCHS} epoch x {len(train_examples)} examples (max_len={args.max_len}, grad_accum={effective_grad_accum}, save_every={SAVE_EVERY}) ---")
    else:
        n_full_batches = len(train_examples) // args.batch
        total_steps = n_full_batches * N_EPOCHS
        print(f"\n--- training {N_EPOCHS} epoch x {n_full_batches} full batches (batch={args.batch}, max_len={args.max_len}, save_every={SAVE_EVERY}) ---")
    model.train()
    t_start = time.time()
    step = resume_step
    epoch = resume_epoch

    if args.batch == 1:
        for epoch in range(resume_epoch, N_EPOCHS):
            start_ex_idx = resume_step if epoch == resume_epoch else 0
            for ex_idx, ex in enumerate(train_examples[start_ex_idx:], start=start_ex_idx):
                text = tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
                enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
                labels = enc.input_ids.clone()
                if args.mask_prompt_labels:
                    mask_prompt_labels(labels, [text], [ex], tok, args.max_len)

                out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=labels)
                scaled_loss = out.loss / effective_grad_accum
                scaled_loss.backward()
                loss_history.append(float(out.loss.item()))

                step += 1
                if step % effective_grad_accum == 0:
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                if step % 10 == 0 or step == 1:
                    elapsed = time.time() - t_start
                    eta_h = (elapsed / max(1, step)) * (total_steps - step) / 3600
                    avg_loss = sum(loss_history[-20:]) / max(1, len(loss_history[-20:]))
                    print(
                        f"  step {step}/{total_steps}: loss={loss_history[-1]:.4f} avg20={avg_loss:.4f} "
                        f"elapsed={elapsed/60:.1f}m eta={eta_h:.2f}h "
                        f"cuda_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB",
                        flush=True,
                    )

                if step % SAVE_EVERY == 0:
                    n_saved = save_adapter(
                        model, ADAPTER_DIR, loss_history, step, epoch,
                        args.batch, args.max_len, effective_grad_accum,
                        args.grad_accum, args.mask_prompt_labels,
                        optimizer=opt, save_optimizer_state=args.save_optimizer_state,
                    )
                    print(f"  [checkpoint @ step {step}] saved {n_saved} LoRA tensors -> {ADAPTER_DIR}/")

                if args.stop_at_step is not None and step >= args.stop_at_step:
                    print(f"  [stop-at-step={args.stop_at_step} reached]")
                    break
            if args.stop_at_step is not None and step >= args.stop_at_step:
                break
    else:
        n_full_batches = len(train_examples) // args.batch
        for epoch in range(resume_epoch, N_EPOCHS):
            start_batch = resume_step if epoch == resume_epoch else 0
            for batch_idx in range(start_batch, n_full_batches):
                batch_ex = train_examples[batch_idx * args.batch:(batch_idx + 1) * args.batch]
                texts = [
                    tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
                    for ex in batch_ex
                ]
                enc = tok(
                    texts, return_tensors="pt", truncation=True,
                    max_length=args.max_len, padding="max_length",
                ).to(device)
                labels = enc.input_ids.clone()
                if args.mask_prompt_labels:
                    mask_prompt_labels(labels, texts, batch_ex, tok, args.max_len)
                labels[enc.attention_mask == 0] = -100

                out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=labels)
                out.loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                loss_history.append(float(out.loss.item()))

                step += 1
                if step % 10 == 0 or step == 1:
                    elapsed = time.time() - t_start
                    eta_h = (elapsed / max(1, step)) * (total_steps - step) / 3600
                    avg_loss = sum(loss_history[-20:]) / max(1, len(loss_history[-20:]))
                    print(
                        f"  step {step}/{total_steps}: loss={loss_history[-1]:.4f} avg20={avg_loss:.4f} "
                        f"elapsed={elapsed/60:.1f}m eta={eta_h:.2f}h "
                        f"cuda_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB",
                        flush=True,
                    )

                if step % SAVE_EVERY == 0:
                    n_saved = save_adapter(
                        model, ADAPTER_DIR, loss_history, step, epoch,
                        args.batch, args.max_len, effective_grad_accum,
                        args.grad_accum, args.mask_prompt_labels,
                        optimizer=opt, save_optimizer_state=args.save_optimizer_state,
                    )
                    print(f"  [checkpoint @ step {step}] saved {n_saved} LoRA tensors -> {ADAPTER_DIR}/")

                if args.stop_at_step is not None and step >= args.stop_at_step:
                    print(f"  [stop-at-step={args.stop_at_step} reached]")
                    break
            if args.stop_at_step is not None and step >= args.stop_at_step:
                break

    # final flush + save
    if args.batch == 1 and step % effective_grad_accum != 0:
        opt.step()
        opt.zero_grad(set_to_none=True)
    save_adapter(
        model, ADAPTER_DIR, loss_history, step, epoch,
        args.batch, args.max_len, effective_grad_accum,
        args.grad_accum, args.mask_prompt_labels,
        optimizer=opt, save_optimizer_state=args.save_optimizer_state,
    )
    print(f"\n=== Day 5 DONE: {step} steps, wall={(time.time()-t_start)/3600:.2f}h ===")
    print(f"final adapter: {ADAPTER_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
