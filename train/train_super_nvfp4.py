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
import sys, os, json, time, argparse, random, importlib, types, threading, subprocess, contextlib
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
from nvfp4_lora.loss import chunked_frozen_lm_head_ce, liger_fused_lm_head_ce


def _lm_head_ce(loss_mode, hidden_states, labels, lm_head, chunk_tokens, logits_fp32):
    """Dispatch frozen-lm-head CE based on selected loss mode."""
    if loss_mode == "liger_flce":
        return liger_fused_lm_head_ce(hidden_states, labels, lm_head)
    return chunked_frozen_lm_head_ce(
        hidden_states, labels, lm_head, chunk_tokens, logits_fp32=logits_fp32
    )


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


def _mem_available_bytes():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _format_gib(n_bytes):
    return f"{n_bytes / (1024 ** 3):.2f}GiB"


_CURRENT_PHASE = "init"


def set_current_phase(label):
    global _CURRENT_PHASE
    _CURRENT_PHASE = label


def get_current_phase():
    return _CURRENT_PHASE


def print_memory_snapshot(label, step=None):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    available = _mem_available_bytes()
    parts = [f"  [mem {label}"]
    if step is not None:
        parts.append(f" step={step}")
    parts.append(
        f"] cuda_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
        f"cuda_peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB "
        f"cuda_reserved={torch.cuda.memory_reserved()/1e9:.2f}GB"
    )
    if torch.cuda.is_available():
        stats = torch.cuda.memory_stats()
        parts.append(
            f" alloc_retries={int(stats.get('num_alloc_retries', 0))}"
            f" ooms={int(stats.get('num_ooms', 0))}"
            f" dev_alloc={int(stats.get('num_device_alloc', 0))}"
            f" dev_free={int(stats.get('num_device_free', 0))}"
        )
    if available is not None:
        parts.append(f" mem_available={_format_gib(available)}")
    print("".join(parts), flush=True)
    set_current_phase(label)


def start_safety_watchdog(min_available_gb=0.0, watch_nvrm_errors=False, interval_s=0.5, defer_nvrm: bool = False):
    """Abort before unified-memory pressure turns into host/driver instability."""
    min_available_bytes = int(min_available_gb * (1024 ** 3))
    enabled = min_available_bytes > 0 or watch_nvrm_errors
    if not enabled:
        return None

    stop_event = threading.Event()

    def abort(reason):
        phase = get_current_phase()
        print(f"\n[FATAL watchdog phase={phase}] {reason}", flush=True)
        os._exit(90)

    def memory_loop():
        while not stop_event.is_set():
            available = _mem_available_bytes()
            if available is not None and min_available_bytes > 0 and available < min_available_bytes:
                abort(
                    "MemAvailable "
                    f"{_format_gib(available)} below threshold {_format_gib(min_available_bytes)}"
                )
            stop_event.wait(interval_s)

    def nvrm_loop():
        proc = None
        try:
            proc = subprocess.Popen(
                ["journalctl", "-k", "-f", "-n", "0", "-o", "cat"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                if stop_event.is_set():
                    break
                if (
                    "NVRM:" in line
                    and ("NV_ERR_NO_MEMORY" in line or "mem_desc.c:1359" in line)
                ):
                    abort("NVIDIA kernel allocator error observed: " + line.strip())
        except (OSError, AssertionError):
            print("  WARNING: watchdog could not follow kernel journal for NVRM errors", flush=True)
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()

    def _start_nvrm_thread():
        t = threading.Thread(target=nvrm_loop, name="nvrm-watchdog", daemon=True)
        t.start()

    if min_available_bytes > 0:
        t = threading.Thread(target=memory_loop, name="ram-watchdog", daemon=True)
        t.start()
    if watch_nvrm_errors:
        if defer_nvrm:
            stop_event.start_nvrm = _start_nvrm_thread
        else:
            _start_nvrm_thread()

    print(
        "  safety watchdog: enabled"
        + (f", min MemAvailable={min_available_gb:.2f}GiB" if min_available_bytes > 0 else "")
        + (", NVRM allocator errors=abort" if watch_nvrm_errors else ""),
        flush=True,
    )
    return stop_event


def stop_safety_watchdog(stop_event):
    if stop_event is not None:
        stop_event.set()


_OFFLOAD_PIN_MEMORY = False


def set_offload_pin_memory(enabled):
    global _OFFLOAD_PIN_MEMORY
    _OFFLOAD_PIN_MEMORY = bool(enabled)


def activation_offload_context(mode):
    if mode == "none":
        return contextlib.nullcontext()
    if mode == "save_on_cpu":
        return torch.autograd.graph.save_on_cpu(pin_memory=_OFFLOAD_PIN_MEMORY)
    raise ValueError(f"unknown activation offload mode: {mode}")


def save_adapter(
    model, base_model_dir, save_dir, loss_history, step, epoch, batch, max_len, grad_accum,
    grad_accum_arg, mask_prompt_labels_enabled, dynamic_padding_enabled,
    length_bucketing_enabled, pad_to_multiple_of, limit_examples, select_longest_examples,
    synthetic_repeat_to_len, synthetic_examples, training_mode, train_suffix_len, prefix_chunk_len,
    loss_mode, loss_chunk_tokens, loss_logits_dtype, activation_offload, optimizer_name,
    sdpa_causal_no_mask_enabled, pooled_loader_buffers_enabled, moe_sparse_no_one_hot_enabled,
    checkpoint_group_size, mamba_chunk_size, lora_r, lora_alpha, target_suffixes,
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
        "base_model_name_or_path": base_model_dir,
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": list(target_suffixes),
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
                "mask_prompt_labels": bool(mask_prompt_labels_enabled),
                "dynamic_padding": bool(dynamic_padding_enabled),
                "length_bucketing": bool(length_bucketing_enabled),
                "pad_to_multiple_of": pad_to_multiple_of,
                "limit_examples": limit_examples,
                "select_longest_examples": select_longest_examples,
                "synthetic_repeat_to_len": synthetic_repeat_to_len,
                "synthetic_examples": synthetic_examples,
                "training_mode": training_mode,
                "train_suffix_len": train_suffix_len,
                "prefix_chunk_len": prefix_chunk_len,
                "loss_mode": loss_mode,
                "loss_chunk_tokens": loss_chunk_tokens,
                "loss_logits_dtype": loss_logits_dtype,
                "activation_offload": activation_offload,
                "optimizer": optimizer_name,
                "sdpa_causal_no_mask": bool(sdpa_causal_no_mask_enabled),
                "pooled_loader_buffers": bool(pooled_loader_buffers_enabled),
                "moe_sparse_no_one_hot": bool(moe_sparse_no_one_hot_enabled),
                "checkpoint_group_size": checkpoint_group_size,
                "mamba_chunk_size": mamba_chunk_size,
                "lr": LR,
                "n_epochs": N_EPOCHS, "r": lora_r, "lora_alpha": lora_alpha,
                "targets": list(target_suffixes),
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


def validate_resume_config(
    saved_config, batch, max_len, grad_accum, grad_accum_arg, mask_prompt_labels_enabled,
    dynamic_padding_enabled, length_bucketing_enabled, pad_to_multiple_of,
    limit_examples, select_longest_examples, synthetic_repeat_to_len, synthetic_examples,
    training_mode, train_suffix_len, prefix_chunk_len,
    loss_mode, loss_chunk_tokens, loss_logits_dtype, activation_offload, optimizer_name, sdpa_causal_no_mask_enabled,
    pooled_loader_buffers_enabled, moe_sparse_no_one_hot_enabled, checkpoint_group_size,
    mamba_chunk_size,
    lora_r, lora_alpha, target_suffixes, force,
):
    expected = {
        "batch": batch,
        "max_len": max_len,
        "grad_accum": grad_accum,
        "mask_prompt_labels": bool(mask_prompt_labels_enabled),
        "dynamic_padding": bool(dynamic_padding_enabled),
        "length_bucketing": bool(length_bucketing_enabled),
        "pad_to_multiple_of": pad_to_multiple_of,
        "limit_examples": limit_examples,
        "select_longest_examples": select_longest_examples,
        "synthetic_repeat_to_len": synthetic_repeat_to_len,
        "synthetic_examples": synthetic_examples,
        "training_mode": training_mode,
        "train_suffix_len": train_suffix_len,
        "prefix_chunk_len": prefix_chunk_len,
        "loss_mode": loss_mode,
        "loss_chunk_tokens": loss_chunk_tokens,
        "loss_logits_dtype": loss_logits_dtype,
        "activation_offload": activation_offload,
        "optimizer": optimizer_name,
        "sdpa_causal_no_mask": bool(sdpa_causal_no_mask_enabled),
        "pooled_loader_buffers": bool(pooled_loader_buffers_enabled),
        "moe_sparse_no_one_hot": bool(moe_sparse_no_one_hot_enabled),
        "checkpoint_group_size": checkpoint_group_size,
        "mamba_chunk_size": mamba_chunk_size,
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "targets": list(target_suffixes),
    }
    legacy_defaults = {
        "dynamic_padding": False,
        "length_bucketing": False,
        "pad_to_multiple_of": None,
        "limit_examples": None,
        "select_longest_examples": None,
        "synthetic_repeat_to_len": None,
        "synthetic_examples": None,
        "training_mode": "full_sequence",
        "train_suffix_len": None,
        "prefix_chunk_len": 8192,
        "loss_mode": "hf",
        "loss_chunk_tokens": 128,
        "loss_logits_dtype": "fp32",
        "activation_offload": "none",
        "optimizer": "adamw",
        "sdpa_causal_no_mask": False,
        "pooled_loader_buffers": False,
        "moe_sparse_no_one_hot": False,
        "checkpoint_group_size": 1,
        "mamba_chunk_size": None,
        "r": R,
        "lora_alpha": LORA_ALPHA,
        "targets": list(TARGET_SUFFIXES),
    }
    differences = []
    legacy_missing = []
    for key, current_value in expected.items():
        if key in saved_config:
            saved_value = saved_config[key]
        elif key in legacy_defaults:
            saved_value = legacy_defaults[key]
        else:
            legacy_missing.append(key)
            continue
        if saved_value != current_value:
            # dynamic_padding has no behavioral effect at batch=1; do not flag a mismatch
            if key == "dynamic_padding" and batch == 1:
                continue
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


def prepare_training_records(train_examples, tokenizer, max_len, length_bucketing_enabled, select_longest_examples):
    records = []
    for idx, ex in enumerate(train_examples):
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        token_len = len(tokenizer(text, truncation=True, max_length=max_len)["input_ids"])
        records.append({"idx": idx, "example": ex, "text": text, "token_len": token_len})

    if select_longest_examples is not None:
        records = sorted(records, key=lambda r: (-r["token_len"], r["idx"]))[:select_longest_examples]
    if length_bucketing_enabled:
        records.sort(key=lambda r: (r["token_len"], r["idx"]))
    return records


def build_synthetic_pressure_records(records, tokenizer, target_len, n_examples, max_len):
    if not records:
        raise SystemExit("cannot build synthetic pressure records from an empty dataset")
    if target_len > max_len:
        raise SystemExit("--synthetic-repeat-to-len must be <= --max-len")
    seed = max(records, key=lambda r: r["token_len"])
    seed_ids = tokenizer(seed["text"], truncation=True, max_length=max_len)["input_ids"]
    if not seed_ids:
        raise SystemExit("cannot build synthetic pressure records from an empty tokenization")
    repeats = (target_len + len(seed_ids) - 1) // len(seed_ids)
    synthetic_ids = (seed_ids * repeats)[:target_len]
    return [
        {
            "idx": idx,
            "example": None,
            "text": None,
            "input_ids": synthetic_ids,
            "token_len": len(synthetic_ids),
            "synthetic": True,
            "seed_idx": seed["idx"],
            "seed_token_len": seed["token_len"],
        }
        for idx in range(n_examples)
    ]


def build_optimizer(trainable, optimizer_name):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(trainable, lr=LR), "AdamW"
    if optimizer_name == "adamw8bit":
        from torchao.optim import AdamW8bit
        return AdamW8bit(trainable, lr=LR), "torchao AdamW8bit"
    if optimizer_name == "adafactor":
        from transformers.optimization import Adafactor
        return (
            Adafactor(
                trainable,
                lr=LR,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
                weight_decay=0.0,
            ),
            "Transformers Adafactor",
        )
    raise SystemExit(f"unknown optimizer: {optimizer_name}")


def parse_target_suffixes(raw_value):
    suffixes = tuple(s.strip() for s in raw_value.split(",") if s.strip())
    if not suffixes:
        raise SystemExit("--target-suffixes must include at least one suffix")
    return suffixes


def enable_sdpa_causal_no_mask(model):
    """Let SDPA use its internal causal path for unpadded full-sequence training.

    The remote Nemotron-H code creates a 4D causal mask even for unpadded SDPA
    training. At long context this square mask is expensive and also prevents
    torch SDPA from using `is_causal=True`. This patch is intentionally opt-in:
    callers must pass no padding mask, and padded batches keep the original path.
    """
    if getattr(model.config, "_attn_implementation", None) != "sdpa":
        raise SystemExit(
            "--sdpa-causal-no-mask requires config._attn_implementation == 'sdpa'; "
            f"got {getattr(model.config, '_attn_implementation', None)!r}"
        )

    base_model = model.model
    original_update_causal_mask = base_model._update_causal_mask

    def patched_update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is None
            and cache_position is not None
        ):
            if int(cache_position[0].item()) == 0:
                return None
            from torch.nn.attention import bias

            return bias.causal_lower_right(
                input_tensor.shape[1], int(cache_position[-1].item()) + 1
            )
        return original_update_causal_mask(attention_mask, input_tensor, cache_position)

    base_model._update_causal_mask = types.MethodType(patched_update_causal_mask, base_model)

    n_patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "NemotronHAttention":
            continue
        remote_mod = importlib.import_module(module.__class__.__module__)
        all_attention_functions = remote_mod.ALL_ATTENTION_FUNCTIONS

        def patched_attention_forward(
            self,
            hidden_states,
            attention_mask=None,
            past_key_values=None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

            if (
                self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and query_states.shape[-2] != key_states.shape[-2]
            ):
                # Cached suffixes need lower-right causal alignment. Calling
                # Transformers' SDPA wrapper with a non-None mask can expand
                # GQA K/V heads before SDPA; call PyTorch SDPA directly so
                # enable_gqa=True preserves the compact K/V representation.
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=attention_mask,
                    dropout_p=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    scale=self.scaling,
                    enable_gqa=(self.num_heads != self.num_key_value_heads),
                )
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_weights = None
                attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, attn_weights, past_key_values

            attention_interface = remote_mod.eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = all_attention_functions[self.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                is_causal=(attention_mask is None),
                **kwargs,
            )

            attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights, past_key_values

        module.forward = types.MethodType(patched_attention_forward, module)
        n_patched += 1

    if n_patched == 0:
        raise SystemExit("--sdpa-causal-no-mask could not find NemotronHAttention modules to patch")
    return n_patched


def maybe_omit_full_attention_mask(attention_mask, enabled):
    if not enabled:
        return attention_mask
    if attention_mask is None:
        return None
    if torch.all(attention_mask == 1):
        return None
    return attention_mask


def _moe_sparse_no_one_hot_impl(self, hidden_states, topk_indices, topk_weights):
    final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)

    # Build one sorted route list instead of scanning every expert. This keeps
    # the trainable graph focused on active LoRA adapters and avoids hundreds of
    # empty routing masks per MoE layer at long context.
    flat_experts = topk_indices.reshape(-1)
    flat_weights = topk_weights.reshape(-1)
    flat_tokens = (
        torch.arange(hidden_states.shape[0], device=hidden_states.device)
        .repeat_interleave(topk_indices.shape[1])
    )
    sorted_experts, order = torch.sort(flat_experts)
    sorted_weights = flat_weights[order]
    sorted_tokens = flat_tokens[order]
    active_experts, route_counts = torch.unique_consecutive(sorted_experts, return_counts=True)

    route_start = 0
    for expert_idx, route_count in zip(active_experts.tolist(), route_counts.tolist()):
        expert = self.experts[expert_idx]
        route_end = route_start + route_count
        token_indices = sorted_tokens[route_start:route_end]

        expert_weights = sorted_weights[route_start:route_end]
        expert_input = hidden_states[token_indices]
        expert_output = expert(expert_input)
        weighted_output = expert_output * expert_weights.unsqueeze(-1)
        final_hidden_states.index_add_(0, token_indices, weighted_output)
        route_start = route_end

    return final_hidden_states.type(hidden_states.dtype)


def enable_moe_sparse_no_one_hot(model):
    """Avoid materializing [tokens, top_k, n_experts] one-hot routing masks."""
    n_patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "NemotronHMoE":
            continue

        module.moe = types.MethodType(_moe_sparse_no_one_hot_impl, module)
        n_patched += 1

    if n_patched == 0:
        raise SystemExit("--moe-sparse-no-one-hot could not find NemotronHMoE modules to patch")
    return n_patched


def enable_mamba_cached_multitoken(model):
    """Allow cached Nemotron-H Mamba layers to continue with multi-token chunks.

    The remote implementation's cached path is generation-oriented. When
    `has_previous_state` is true it uses single-token update kernels, which is
    not suitable for a trainable suffix chunk. This patch uses the full scan
    kernels with the cached conv/SSM states as initial states for multi-token
    continuation, while leaving the original paths untouched.
    """
    n_patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "NemotronHMamba2Mixer":
            continue

        original_cuda_forward = module.cuda_kernels_forward
        remote_mod = importlib.import_module(module.__class__.__module__)

        def patched_cuda_kernels_forward(
            self, hidden_states, cache_params=None, attention_mask=None,
            original_cuda_forward=original_cuda_forward, remote_mod=remote_mod,
        ):
            if (
                cache_params is None
                or not cache_params.has_previous_state
                or hidden_states.shape[1] <= 1
            ):
                return original_cuda_forward(hidden_states, cache_params, attention_mask)

            batch_size, seq_len, _ = hidden_states.shape
            groups_time_state_size = self.n_groups * self.ssm_state_size
            d_to_remove = 2 * self.intermediate_size + 2 * self.n_groups * self.ssm_state_size + self.num_heads

            if attention_mask is not None and not torch.all(attention_mask == 1):
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)

            readonly_suffix = getattr(cache_params, "readonly_suffix", False)
            projected_states = self.in_proj(hidden_states)
            d_mlp = (projected_states.shape[-1] - d_to_remove) // 2
            _, _, gate, hidden_states_b_c, time_step = torch.split(
                projected_states,
                [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads],
                dim=-1,
            )

            # The double-transpose is intentional: it remaps the conv-state slice into
            # the channel-contiguous layout (stride (C*(K-1), 1, C)) that
            # `causal_conv1d_fn` requires for `initial_states`. The slice `[..., 1:]`
            # alone leaves the parent strides (C*K, K, 1) unchanged so its second-dim
            # stride is K, not 1; `.transpose(1, 2).contiguous().transpose(1, 2)`
            # forces the channel-fastest layout we need. Without this we hit
            # `Expected initial_states.stride(1) == 1` (see LC-040 in
            # docs/LONG_CONTEXT_EXPERIMENTS.md). Do not collapse to a single
            # `.contiguous()` without re-testing against causal_conv1d's current ABI.
            conv_initial = (
                cache_params.conv_states[self.layer_idx][..., 1:]
                .transpose(1, 2)
                .contiguous()
                .transpose(1, 2)
            )
            conv_result = remote_mod.causal_conv1d_fn(
                x=hidden_states_b_c.transpose(1, 2),
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                initial_states=conv_initial,
                return_final_states=(not readonly_suffix),
            )
            if isinstance(conv_result, tuple):
                hidden_states_b_c, conv_state = conv_result
                cache_params.conv_states[self.layer_idx].zero_()
                cache_params.conv_states[self.layer_idx][..., 1:].copy_(conv_state)
            else:
                hidden_states_b_c = conv_result
            hidden_states_b_c = hidden_states_b_c.transpose(1, 2)[:, :seq_len]

            hidden_states, b_state, c_state = torch.split(
                hidden_states_b_c,
                [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                dim=-1,
            )
            if attention_mask is not None and not torch.all(attention_mask == 1):
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)

            a_state = -torch.exp(self.A_log.float())
            dt_limit_kwargs = {} if self.time_step_limit is None else {"dt_limit": self.time_step_limit}
            scan_result = remote_mod.mamba_chunk_scan_combined(
                hidden_states.view(batch_size, seq_len, -1, self.head_dim),
                time_step,
                a_state,
                b_state.view(batch_size, seq_len, self.n_groups, -1),
                c_state.view(batch_size, seq_len, self.n_groups, -1),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=None,
                initial_states=cache_params.ssm_states[self.layer_idx],
                return_final_states=(not readonly_suffix),
                dt_bias=self.dt_bias,
                dt_softplus=True,
                **dt_limit_kwargs,
            )
            if readonly_suffix:
                scan_output = scan_result
            else:
                scan_output, ssm_state = scan_result
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
            scan_output = scan_output.view(batch_size, seq_len, -1)
            scan_output = self.norm(scan_output, gate)
            return self.out_proj(scan_output)

        module.cuda_kernels_forward = types.MethodType(patched_cuda_kernels_forward, module)
        n_patched += 1

    if n_patched == 0:
        raise SystemExit("cached-prefix mode could not find NemotronHMamba2Mixer modules to patch")
    return n_patched


def set_mamba_chunk_size(model, chunk_size):
    if chunk_size is None:
        return 0
    n_changed = 0
    for module in model.modules():
        if module.__class__.__name__ != "NemotronHMamba2Mixer":
            continue
        if hasattr(module, "chunk_size"):
            module.chunk_size = int(chunk_size)
            n_changed += 1
    if n_changed == 0:
        raise SystemExit("--mamba-chunk-size could not find NemotronHMamba2Mixer modules to patch")
    return n_changed


def set_gradient_checkpointing_flags(model, enabled):
    n_changed = 0
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = bool(enabled)
            n_changed += 1
    return n_changed


def enable_grouped_layer_checkpointing(model, group_size):
    """Checkpoint groups of Nemotron-H layers instead of every layer separately."""
    if group_size <= 1:
        return 0

    base_model = model.model
    original_forward = base_model.forward
    remote_mod = importlib.import_module(base_model.__class__.__module__)
    output_cls = remote_mod.NemotronHOutput

    def patched_forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        if (
            not self.training
            or not self.gradient_checkpointing
            or output_attentions
            or output_hidden_states
            or past_key_values is not None
        ):
            return original_forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        hidden_states = inputs_embeds
        if cache_position is None:
            cache_position = torch.arange(
                0, hidden_states.shape[1], device=hidden_states.device
            )
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)
        mamba_mask = self._update_mamba_mask(attention_mask, cache_position)

        def layer_mask_for(block):
            if block.block_type == "mamba":
                return mamba_mask
            if block.block_type == "attention":
                return causal_mask
            if block.block_type in ["mlp", "moe"]:
                return None
            raise ValueError(f"Invalid block_type: {block.block_type}")

        for group_start in range(0, len(self.layers), group_size):
            group_end = min(group_start + group_size, len(self.layers))
            group_layers = self.layers[group_start:group_end]
            group_masks = [layer_mask_for(block) for block in group_layers]

            def run_group(group_hidden_states, group_layers=tuple(group_layers), group_masks=tuple(group_masks)):
                for block, layer_mask in zip(group_layers, group_masks):
                    group_hidden_states = block(
                        group_hidden_states,
                        past_key_values=None,
                        cache_position=cache_position,
                        attention_mask=layer_mask,
                        output_attentions=False,
                    )
                return group_hidden_states

            hidden_states = self._gradient_checkpointing_func(run_group, hidden_states)

        hidden_states = self.norm_f(hidden_states)
        if not return_dict:
            return (hidden_states,)
        return output_cls(
            last_hidden_state=hidden_states,
            past_key_values=None if not use_cache else past_key_values,
            hidden_states=None,
        )

    base_model.forward = types.MethodType(patched_forward, base_model)
    return len(base_model.layers)


def run_routing_census(model, input_ids, attention_mask, output_path=None, top_n=16):
    counts = {}
    handles = []

    for name, module in model.named_modules():
        if module.__class__.__name__ != "NemotronHTopkRouter":
            continue
        n_experts = int(module.config.n_routed_experts)
        counts[name] = torch.zeros(n_experts, dtype=torch.long)

        def hook(_module, _inputs, output, name=name):
            topk_indices = output[0].detach().flatten().to("cpu")
            counts[name] += torch.bincount(topk_indices, minlength=counts[name].numel())

        handles.append(module.register_forward_hook(hook))

    if not handles:
        raise SystemExit("--routing-census-only could not find NemotronHTopkRouter modules")

    try:
        model.train()
        with torch.no_grad():
            _ = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    result = {}
    for name, count in counts.items():
        total = int(count.sum().item())
        top = torch.topk(count, k=min(top_n, count.numel()))
        result[name] = {
            "total_routes": total,
            "active_experts": int((count > 0).sum().item()),
            "top_experts": [
                {"expert": int(idx), "routes": int(val)}
                for val, idx in zip(top.values.tolist(), top.indices.tolist())
            ],
        }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    active_counts = [item["active_experts"] for item in result.values()]
    print(
        "  routing census: "
        f"{len(result)} routers, active_experts min={min(active_counts)} "
        f"median={sorted(active_counts)[len(active_counts)//2]} max={max(active_counts)}",
        flush=True,
    )
    if output_path:
        print(f"  routing census saved: {output_path}", flush=True)
    return result


def make_hybrid_cache(model, batch_size, device):
    remote_mod = importlib.import_module(model.model.__class__.__module__)
    cache_cls = remote_mod.NemotronHHybridDynamicCache
    dtype = getattr(model, "dtype", torch.bfloat16)
    return cache_cls(model.config, batch_size=batch_size, dtype=dtype, device=device)


def make_static_prefix_attention_cache(cache, prefix_len):
    """Preallocate prefix K/V cache and fill it by slice instead of repeated cat."""
    cache.static_prefix_len = int(prefix_len)
    cache.static_prefix_write_pos = {}

    def static_update(key_states, value_states, layer_idx, cache_kwargs=None):
        write_pos = cache.static_prefix_write_pos.get(layer_idx, 0)
        end = write_pos + key_states.shape[-2]
        if end > cache.static_prefix_len:
            raise RuntimeError(
                f"static prefix cache overflow at layer {layer_idx}: "
                f"{end} > {cache.static_prefix_len}"
            )
        if cache.key_cache[layer_idx].numel() == 0:
            key_shape = list(key_states.shape)
            value_shape = list(value_states.shape)
            key_shape[-2] = cache.static_prefix_len
            value_shape[-2] = cache.static_prefix_len
            cache.key_cache[layer_idx] = torch.empty(
                key_shape, device=key_states.device, dtype=key_states.dtype
            )
            cache.value_cache[layer_idx] = torch.empty(
                value_shape, device=value_states.device, dtype=value_states.dtype
            )
        cache.key_cache[layer_idx][:, :, write_pos:end, :].copy_(key_states)
        cache.value_cache[layer_idx][:, :, write_pos:end, :].copy_(value_states)
        cache.static_prefix_write_pos[layer_idx] = end
        return (
            cache.key_cache[layer_idx][:, :, :end, :],
            cache.value_cache[layer_idx][:, :, :end, :],
        )

    cache.update = static_update
    return cache


def make_cache_readonly_for_suffix(cache):
    """Let checkpointed suffix forwards read prefix cache without mutating it."""
    if cache is None:
        return None
    if getattr(cache, "readonly_suffix", False):
        return cache

    def readonly_update(key_states, value_states, layer_idx, cache_kwargs=None):
        cached_key = cache.key_cache[layer_idx]
        cached_value = cache.value_cache[layer_idx]
        if cached_key.numel() == 0:
            return key_states, value_states
        return (
            torch.cat([cached_key.detach(), key_states], dim=2),
            torch.cat([cached_value.detach(), value_states], dim=2),
        )

    cache.update = readonly_update
    cache.readonly_suffix = True
    return cache


def prefill_prefix_cache(model, input_ids, prefix_len, prefix_chunk_len, sdpa_causal_no_mask):
    if prefix_len <= 0:
        return None
    cache = make_hybrid_cache(model, input_ids.shape[0], input_ids.device)
    cache = make_static_prefix_attention_cache(cache, prefix_len)
    was_training = model.training
    n_gc_disabled = set_gradient_checkpointing_flags(model, False)
    # Keep train mode to avoid NVFP4LoRALinear's eval-mode bf16 weight cache.
    model.train(True)
    try:
        with torch.no_grad():
            n_chunks = (prefix_len + prefix_chunk_len - 1) // prefix_chunk_len
            progress_every = max(1, n_chunks // 8) if prefix_len >= 65536 else 0
            for chunk_idx, start in enumerate(range(0, prefix_len, prefix_chunk_len), start=1):
                end = min(start + prefix_chunk_len, prefix_len)
                chunk_ids = input_ids[:, start:end]
                chunk_mask = torch.ones_like(chunk_ids)
                model_attention_mask = maybe_omit_full_attention_mask(chunk_mask, sdpa_causal_no_mask)
                cache_position = torch.arange(start, end, device=input_ids.device)
                outputs = model.model(
                    input_ids=chunk_ids,
                    attention_mask=model_attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=cache_position,
                    return_dict=True,
                )
                cache = outputs.past_key_values or cache
                if progress_every and (chunk_idx % progress_every == 0 or end == prefix_len):
                    print(
                        "  cached-prefix prefill: "
                        f"chunk {chunk_idx}/{n_chunks} tokens={end}/{prefix_len}",
                        flush=True,
                    )
    finally:
        if n_gc_disabled:
            set_gradient_checkpointing_flags(model, True)
        model.train(was_training)
    return cache


def cached_prefix_suffix_loss(
    model, input_ids, labels, prefix_len, suffix_len, prefix_chunk_len,
    sdpa_causal_no_mask, loss_chunk_tokens, loss_logits_dtype,
    activation_offload="none", profile_step=None, loss_mode="chunked_frozen_ce",
):
    set_current_phase("cached_prefix_prefill")
    cache = prefill_prefix_cache(model, input_ids, prefix_len, prefix_chunk_len, sdpa_causal_no_mask)
    cache = make_cache_readonly_for_suffix(cache)
    if profile_step is not None:
        print_memory_snapshot("after_cached_prefix_prefill", profile_step)
    suffix_ids = input_ids[:, prefix_len:prefix_len + suffix_len]
    suffix_labels = labels[:, prefix_len:prefix_len + suffix_len]
    suffix_mask = torch.ones_like(suffix_ids)
    model_attention_mask = maybe_omit_full_attention_mask(suffix_mask, sdpa_causal_no_mask)
    cache_position = torch.arange(prefix_len, prefix_len + suffix_ids.shape[1], device=input_ids.device)
    set_current_phase("cached_suffix_forward")
    with activation_offload_context(activation_offload):
        model_outputs = model.model(
            input_ids=suffix_ids,
            attention_mask=model_attention_mask,
            past_key_values=cache,
            use_cache=False,
            cache_position=cache_position,
            return_dict=True,
        )
        if profile_step is not None:
            print_memory_snapshot("after_cached_suffix_forward", profile_step)
        set_current_phase("cached_suffix_loss")
        loss = _lm_head_ce(
            loss_mode, model_outputs[0], suffix_labels, model.lm_head,
            loss_chunk_tokens, (loss_logits_dtype == "fp32"),
        )
    if profile_step is not None:
        print_memory_snapshot("after_cached_suffix_loss_inner", profile_step)
    return loss, model_outputs


def run_cached_prefix_compare(
    model, input_ids, labels, prefix_len, suffix_len, prefix_chunk_len,
    sdpa_causal_no_mask, loss_chunk_tokens, loss_logits_dtype,
    loss_mode="chunked_frozen_ce",
):
    was_training = model.training
    model.train(True)
    with torch.no_grad():
        full_outputs = model.model(
            input_ids=input_ids[:, :prefix_len + suffix_len],
            attention_mask=maybe_omit_full_attention_mask(
                torch.ones_like(input_ids[:, :prefix_len + suffix_len]), sdpa_causal_no_mask
            ),
            use_cache=False,
            return_dict=True,
        )
        full_hidden = full_outputs[0][:, prefix_len:prefix_len + suffix_len, :]
        full_loss = _lm_head_ce(
            loss_mode, full_hidden, labels[:, prefix_len:prefix_len + suffix_len],
            model.lm_head, loss_chunk_tokens, (loss_logits_dtype == "fp32"),
        )
        full_hidden_cpu = full_hidden.detach().float().cpu()
        full_loss_value = float(full_loss)
        del full_outputs, full_hidden, full_loss
        torch.cuda.empty_cache()

        cache = prefill_prefix_cache(model, input_ids, prefix_len, prefix_chunk_len, sdpa_causal_no_mask)
        # The static-prefix attention cache is sized exactly to prefix_len; without
        # the readonly patch the suffix's attention update would try to write at
        # position prefix_len and trip the overflow guard. The training-path helper
        # cached_prefix_suffix_loss() calls this; the compare helper has to do the
        # same to remain valid after the static-prefix-cache patch (LC-051).
        cache = make_cache_readonly_for_suffix(cache)
        suffix_ids = input_ids[:, prefix_len:prefix_len + suffix_len]
        suffix_mask = torch.ones_like(suffix_ids)
        cache_position = torch.arange(prefix_len, prefix_len + suffix_len, device=input_ids.device)
        cached_outputs = model.model(
            input_ids=suffix_ids,
            attention_mask=maybe_omit_full_attention_mask(suffix_mask, sdpa_causal_no_mask),
            past_key_values=cache,
            use_cache=False,
            cache_position=cache_position,
            return_dict=True,
        )
        cached_hidden = cached_outputs[0]
        cached_loss = _lm_head_ce(
            loss_mode, cached_hidden, labels[:, prefix_len:prefix_len + suffix_len],
            model.lm_head, loss_chunk_tokens, (loss_logits_dtype == "fp32"),
        )
        cached_hidden_cpu = cached_hidden.detach().float().cpu()
        cached_loss_value = float(cached_loss)
        del cached_outputs, cached_hidden, cached_loss
        torch.cuda.empty_cache()

    max_abs = (full_hidden_cpu - cached_hidden_cpu).abs().max().item()
    mean_abs = (full_hidden_cpu - cached_hidden_cpu).abs().float().mean().item()
    loss_diff = abs(full_loss_value - cached_loss_value)
    print(
        "  cached-prefix compare: "
        f"full_loss={full_loss_value:.6f} cached_loss={cached_loss_value:.6f} "
        f"loss_abs_diff={loss_diff:.6f} hidden_max_abs_diff={max_abs:.6f} "
        f"hidden_mean_abs_diff={mean_abs:.6f}",
        flush=True,
    )
    model.train(was_training)
    return {"loss_abs_diff": loss_diff, "hidden_max_abs_diff": max_abs, "hidden_mean_abs_diff": mean_abs}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR,
                    help="base model directory; defaults to the script constant")
    ap.add_argument("--train-file", default=TRAIN_FILE,
                    help="JSONL training file; defaults to the script constant")
    ap.add_argument("--adapter-dir", default=ADAPTER_DIR,
                    help="adapter output/resume directory; defaults to the script constant")
    ap.add_argument("--batch", type=int, default=1,
                    help="physical batch size; batch > 1 enables padded batched training")
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--grad-accum", type=int, default=GRAD_ACCUM,
                    help="gradient accumulation steps")
    ap.add_argument("--pad-to-max-length", action="store_true",
                    help="use legacy batched padding to --max-len instead of padding to the longest item in each batch")
    ap.add_argument("--no-length-bucketing", action="store_false", dest="length_bucketing", default=True,
                    help="keep dataset order for batched runs instead of sorting by tokenized length")
    ap.add_argument("--pad-to-multiple-of", type=int, default=None,
                    help="optionally pad dynamic batches to a multiple, e.g. 8; unset minimizes token count")
    ap.add_argument("--limit-examples", type=int, default=None,
                    help="limit the loaded dataset before batching; useful for bounded smoke tests")
    ap.add_argument("--select-longest-examples", type=int, default=None,
                    help="after tokenizing, keep only the N longest examples for context-pressure smoke tests")
    ap.add_argument("--synthetic-repeat-to-len", type=int, default=None,
                    help="build synthetic token-id records by repeating the longest example to this exact length")
    ap.add_argument("--synthetic-examples", type=int, default=1,
                    help="number of synthetic pressure records to train when --synthetic-repeat-to-len is set")
    ap.add_argument("--training-mode", choices=("full_sequence", "cached_prefix_suffix"), default="full_sequence",
                    help="full exact training or no-grad prefix plus trainable suffix")
    ap.add_argument("--train-suffix-len", type=int, default=None,
                    help="suffix tokens to train in cached_prefix_suffix mode")
    ap.add_argument("--prefix-chunk-len", type=int, default=8192,
                    help="no-grad prefix prefill chunk size for cached_prefix_suffix mode")
    ap.add_argument("--cached-prefix-compare-full", action="store_true",
                    help="compare full forward vs cached-prefix suffix forward, then exit")
    ap.add_argument("--loss-mode", choices=("hf", "chunked_frozen_ce", "liger_flce"), default="hf",
                    help="loss implementation; chunked_frozen_ce avoids full-sequence fp32 logits; liger_flce uses Liger Kernel FusedLinearCrossEntropy for reduced backward allocator pressure (requires liger-kernel)")
    ap.add_argument("--loss-chunk-tokens", type=int, default=128,
                    help="token chunk size for --loss-mode chunked_frozen_ce")
    ap.add_argument("--loss-logits-dtype", choices=("fp32", "bf16"), default="fp32",
                    help="dtype for chunked frozen-CE logits; bf16 reduces scratch but changes numerics slightly")
    ap.add_argument("--activation-offload", choices=("none", "save_on_cpu"), default="none",
                    help="offload tensors saved for backward out of the CUDA allocator")
    ap.add_argument("--activation-offload-pin-memory", action=argparse.BooleanOptionalAction, default=False,
                    help="use pinned host memory for save_on_cpu (default False on UMA; True is the historical pre-Spark default)")
    ap.add_argument("--optimizer", choices=("adamw", "adamw8bit", "adafactor"), default="adamw",
                    help="optimizer implementation; adamw is the v1.0 baseline")
    ap.add_argument("--lora-r", type=int, default=R,
                    help="LoRA rank for NVFP4 target modules")
    ap.add_argument("--lora-alpha", type=int, default=LORA_ALPHA,
                    help="LoRA alpha for NVFP4 target modules")
    ap.add_argument("--target-suffixes", default=",".join(TARGET_SUFFIXES),
                    help="comma-separated NVFP4 module suffixes to receive LoRA, default: up_proj,down_proj")
    ap.add_argument("--sdpa-causal-no-mask", action="store_true",
                    help="experimental: for unpadded SDPA training, omit explicit causal masks and rely on is_causal=True")
    ap.add_argument("--pooled-loader-buffers", action="store_true",
                    help="experimental: back NVFP4 buffers and LoRA params with flat CUDA pools during load")
    ap.add_argument("--moe-sparse-no-one-hot", action="store_true",
                    help="experimental: avoid dense MoE one-hot expert masks in remote model code")
    ap.add_argument("--mamba-cached-multitoken", action=argparse.BooleanOptionalAction, default=True,
                    help="patch Mamba cache continuation so cached-prefix suffix chunks can be longer than one token")
    ap.add_argument("--mamba-chunk-size", type=int, default=None,
                    help="override Nemotron-H Mamba scan chunk_size; experimental allocator knob")
    ap.add_argument("--checkpoint-group-size", type=int, default=1,
                    help="experimental: checkpoint groups of this many layers; 1 keeps the model default")
    ap.add_argument("--stop-after-load", action="store_true",
                    help="load and patch the model, then exit before tokenizer/data/optimizer setup")
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
    ap.add_argument("--save-final-adapter", action=argparse.BooleanOptionalAction, default=True,
                    help="write the final adapter at process exit; disable for bounded crash-only smoke tests")
    ap.add_argument("--watchdog-min-available-gb", type=float, default=0.0,
                    help="abort if host MemAvailable drops below this GiB threshold; 0 disables")
    ap.add_argument("--watchdog-nvrm-errors", action=argparse.BooleanOptionalAction, default=False,
                    help="abort if new NVIDIA kernel NV_ERR_NO_MEMORY/mem_desc allocator errors appear")
    ap.add_argument("--profile-memory-phases", action="store_true",
                    help="print synchronized CUDA/host memory snapshots around forward/loss/backward/optimizer phases")
    ap.add_argument("--routing-census-only", action="store_true",
                    help="run one no-grad forward and count routed MoE expert usage, then exit")
    ap.add_argument("--routing-census-output", default=None,
                    help="optional JSON path for --routing-census-only output")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.batch < 1:
        raise SystemExit("--batch must be >= 1")
    if args.grad_accum < 1:
        raise SystemExit("--grad-accum must be >= 1")
    if args.pad_to_multiple_of is not None and args.pad_to_multiple_of < 1:
        raise SystemExit("--pad-to-multiple-of must be >= 1 when set")
    if args.limit_examples is not None and args.limit_examples < 1:
        raise SystemExit("--limit-examples must be >= 1 when set")
    if args.select_longest_examples is not None and args.select_longest_examples < 1:
        raise SystemExit("--select-longest-examples must be >= 1 when set")
    if args.synthetic_repeat_to_len is not None and args.synthetic_repeat_to_len < 1:
        raise SystemExit("--synthetic-repeat-to-len must be >= 1 when set")
    if args.synthetic_examples < 1:
        raise SystemExit("--synthetic-examples must be >= 1")
    if args.synthetic_repeat_to_len is not None and args.batch != 1:
        raise SystemExit("--synthetic-repeat-to-len currently supports --batch 1 pressure smokes only")
    if args.synthetic_repeat_to_len is not None and args.mask_prompt_labels:
        raise SystemExit("--synthetic-repeat-to-len uses repeated token IDs; omit --mask-prompt-labels")
    if args.training_mode == "cached_prefix_suffix":
        if args.batch != 1:
            raise SystemExit("--training-mode cached_prefix_suffix currently requires --batch 1")
        if args.loss_mode not in ("chunked_frozen_ce", "liger_flce"):
            raise SystemExit("--training-mode cached_prefix_suffix currently requires --loss-mode chunked_frozen_ce or liger_flce")
        if not args.sdpa_causal_no_mask:
            raise SystemExit("--training-mode cached_prefix_suffix currently requires --sdpa-causal-no-mask (the cached suffix mask contract relies on it)")
        if args.train_suffix_len is None or args.train_suffix_len < 2:
            raise SystemExit("--training-mode cached_prefix_suffix requires --train-suffix-len >= 2")
        if args.prefix_chunk_len < 1:
            raise SystemExit("--prefix-chunk-len must be >= 1")
        if not args.mamba_cached_multitoken and args.train_suffix_len > 1:
            raise SystemExit("cached suffix length >1 requires --mamba-cached-multitoken")
    if args.loss_chunk_tokens < 1:
        raise SystemExit("--loss-chunk-tokens must be >= 1")
    if args.lora_r < 1:
        raise SystemExit("--lora-r must be >= 1")
    if args.lora_alpha < 1:
        raise SystemExit("--lora-alpha must be >= 1")
    if args.watchdog_min_available_gb < 0:
        raise SystemExit("--watchdog-min-available-gb must be >= 0")
    if args.checkpoint_group_size < 1:
        raise SystemExit("--checkpoint-group-size must be >= 1")
    if args.mamba_chunk_size is not None and args.mamba_chunk_size < 1:
        raise SystemExit("--mamba-chunk-size must be >= 1")

    model_dir = args.model_dir
    train_file = args.train_file
    adapter_dir = args.adapter_dir
    target_suffixes = parse_target_suffixes(args.target_suffixes)

    effective_grad_accum = args.grad_accum
    dynamic_padding_enabled = not args.pad_to_max_length
    length_bucketing_enabled = bool(args.length_bucketing and args.batch > 1)
    print(
        f"=== Day 5 production: Super-120B LoRA on ICH v3.1, 1 epoch, batch={args.batch}, "
        f"max_len={args.max_len}, grad_accum={effective_grad_accum} ==="
    )
    watchdog_stop = start_safety_watchdog(
        min_available_gb=args.watchdog_min_available_gb,
        watch_nvrm_errors=args.watchdog_nvrm_errors,
        defer_nvrm=True,
    )
    device = torch.device("cuda")
    os.makedirs(adapter_dir, exist_ok=True)

    free_g = torch.cuda.mem_get_info()
    print(f"baseline: cuda free={free_g[0]/1e9:.1f} GB / total={free_g[1]/1e9:.1f} GB")

    print("\n--- loading model ---")
    t0 = time.time()
    try:
        model = load_nemotron_with_nvfp4_lora(
            model_dir,
            target_lora_suffixes=target_suffixes,
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
            device=device, dtype=torch.bfloat16,
            pooled_loader_buffers=args.pooled_loader_buffers,
        )
        print(f"  load wall: {time.time()-t0:.1f}s, cuda_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB")
    finally:
        # Arm the NVRM watchdog regardless of load success/failure so any allocator
        # burst during the exception teardown is still observed. If load succeeded
        # this is the intended post-load arming; if load raised, this catches NVRM
        # noise during cleanup before the process exits. The print is wrapped in a
        # try/except OSError so that a BrokenPipeError from a closed stdout pipe
        # (e.g. `python train/train_super_nvfp4.py | head`) does not replace the
        # original exception in the success-vs-failure flow.
        if hasattr(watchdog_stop, "start_nvrm"):
            watchdog_stop.start_nvrm()
            try:
                print("  NVRM allocator-error watchdog: armed post-load", flush=True)
            except OSError:
                pass
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
        print("  config.use_cache: disabled")
    print(f"  attention implementation: {getattr(model.config, '_attn_implementation', None)}")
    if args.sdpa_causal_no_mask:
        n_attn = enable_sdpa_causal_no_mask(model)
        print(f"  sdpa causal no-mask patch: enabled for {n_attn} attention modules")
    if args.moe_sparse_no_one_hot:
        n_moe = enable_moe_sparse_no_one_hot(model)
        print(f"  MoE sparse no-one-hot patch: enabled for {n_moe} MoE modules")
    if args.training_mode == "cached_prefix_suffix" and args.mamba_cached_multitoken:
        n_mamba = enable_mamba_cached_multitoken(model)
        print(f"  Mamba cached multi-token patch: enabled for {n_mamba} modules")
    if args.mamba_chunk_size is not None:
        n_mamba_chunk = set_mamba_chunk_size(model, args.mamba_chunk_size)
        print(f"  Mamba chunk_size override: {args.mamba_chunk_size} on {n_mamba_chunk} modules")

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        for m in model.modules():
            if hasattr(m, "gradient_checkpointing"):
                m.gradient_checkpointing = True
    print("  gradient checkpointing: enabled")
    if args.training_mode == "cached_prefix_suffix":
        print(
            "  cached-prefix mode: suffix gradient checkpointing kept enabled "
            "with readonly prefix cache"
        )
    if args.checkpoint_group_size > 1:
        n_layers = enable_grouped_layer_checkpointing(model, args.checkpoint_group_size)
        print(
            f"  grouped layer checkpointing: enabled group_size={args.checkpoint_group_size} "
            f"over {n_layers} layers"
        )

    try:
        import causal_conv1d  # noqa: F401
        print("  causal_conv1d: available")
    except ImportError as e:
        raise SystemExit("causal_conv1d is required for tractable Super long-context training") from e
    if args.stop_after_load:
        print("  [stop-after-load reached]")
        stop_safety_watchdog(watchdog_stop)
        return 0

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    print(f"\n--- loading training data from {train_file} ---")
    train_examples = []
    with open(train_file) as f:
        for line in f:
            train_examples.append(json.loads(line))
    if args.limit_examples is not None:
        train_examples = train_examples[:args.limit_examples]
        print(f"  limited to first {len(train_examples)} train examples")
    else:
        print(f"  loaded {len(train_examples)} train examples")

    resume_step = 0
    resume_epoch = 0
    loss_history: list[float] = []
    if args.resume_from is not None:
        if args.resume_from < 0:
            raise SystemExit(f"--resume-from must be >= 0; got {args.resume_from}")
        print(f"\n--- resuming from step {args.resume_from} ---")
        loaded, progress_step, progress_epoch, restored_history, progress_config = load_adapter_weights(model, adapter_dir)
        validate_resume_config(
            progress_config,
            args.batch,
            args.max_len,
            effective_grad_accum,
            args.grad_accum,
            args.mask_prompt_labels,
            dynamic_padding_enabled,
            length_bucketing_enabled,
            args.pad_to_multiple_of,
            args.limit_examples,
            args.select_longest_examples,
            args.synthetic_repeat_to_len,
            args.synthetic_examples,
            args.training_mode,
            args.train_suffix_len,
            args.prefix_chunk_len,
            args.loss_mode,
            args.loss_chunk_tokens,
            args.loss_logits_dtype,
            args.activation_offload,
            args.optimizer,
            args.sdpa_causal_no_mask,
            args.pooled_loader_buffers,
            args.moe_sparse_no_one_hot,
            args.checkpoint_group_size,
            args.mamba_chunk_size,
            args.lora_r,
            args.lora_alpha,
            target_suffixes,
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
    opt, optimizer_label = build_optimizer(trainable, args.optimizer)
    print(
        f"  optimizer: {optimizer_label} lr={LR} "
        f"on {n_trainable/1e6:.2f}M trainable params across {len(trainable)} tensors"
    )
    if args.loss_mode == "liger_flce":
        _chunk_nondefault = args.loss_chunk_tokens != 128
        _dtype_nondefault = args.loss_logits_dtype != "fp32"
        _loss_banner = f"  loss: {args.loss_mode}"
        if _chunk_nondefault:
            _loss_banner += f" chunk_tokens={args.loss_chunk_tokens} (ignored for liger_flce)"
        if _dtype_nondefault:
            _loss_banner += f" logits_dtype={args.loss_logits_dtype} (ignored for liger_flce)"
        print(_loss_banner)
        if _chunk_nondefault or _dtype_nondefault:
            print("  WARNING: --loss-chunk-tokens and/or --loss-logits-dtype are ignored when --loss-mode liger_flce")
    else:
        print(
            f"  loss: {args.loss_mode}"
            + (f" chunk_tokens={args.loss_chunk_tokens}" if args.loss_mode == "chunked_frozen_ce" else "")
            + (f" logits_dtype={args.loss_logits_dtype}" if args.loss_mode == "chunked_frozen_ce" else "")
        )
    set_offload_pin_memory(args.activation_offload_pin_memory)
    print(f"  activation_offload: {args.activation_offload} pin_memory={args.activation_offload_pin_memory}")
    if args.loss_mode in ("chunked_frozen_ce", "liger_flce") and model.lm_head.weight.requires_grad:
        raise SystemExit(f"--loss-mode {args.loss_mode} requires frozen lm_head weights")
    if args.resume_from is not None:
        opt_loaded, rng_loaded = load_resume_state(adapter_dir, opt)
        print(f"  resume state: optimizer_state={'loaded' if opt_loaded else 'missing'}, rng_state={'loaded' if rng_loaded else 'missing'}")

    print("\n--- rendering/tokenizing training records ---")
    train_records = prepare_training_records(
        train_examples, tok, args.max_len, length_bucketing_enabled, args.select_longest_examples
    )
    if args.select_longest_examples is not None:
        print(f"  selected {len(train_records)} longest examples after tokenization")
    if args.synthetic_repeat_to_len is not None:
        train_records = build_synthetic_pressure_records(
            train_records, tok, args.synthetic_repeat_to_len, args.synthetic_examples, args.max_len
        )
        seed = train_records[0]
        print(
            "  synthetic pressure records: "
            f"{len(train_records)} x {args.synthetic_repeat_to_len} tokens "
            f"(seed_idx={seed['seed_idx']}, seed_token_len={seed['seed_token_len']})"
        )
    if train_records:
        lengths = [record["token_len"] for record in train_records]
        print(f"  token lengths: min={min(lengths)} median={sorted(lengths)[len(lengths)//2]} max={max(lengths)}")
    if length_bucketing_enabled:
        print("  length bucketing: enabled for batched training")
    if args.batch > 1:
        print(
            f"  padding: {'dynamic longest-in-batch' if dynamic_padding_enabled else 'legacy max_length'}"
            + (f", pad_to_multiple_of={args.pad_to_multiple_of}" if args.pad_to_multiple_of else "")
        )
    if args.cached_prefix_compare_full:
        if args.training_mode != "cached_prefix_suffix":
            raise SystemExit("--cached-prefix-compare-full requires --training-mode cached_prefix_suffix")
        record = train_records[0]
        if "input_ids" in record:
            input_ids = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
        else:
            enc = tok(record["text"], return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
            input_ids = enc.input_ids
        seq_len = input_ids.shape[1]
        if args.train_suffix_len >= seq_len:
            raise SystemExit(
                f"--train-suffix-len={args.train_suffix_len} must be < seq_len={seq_len} for cached-prefix compare"
            )
        labels = input_ids.clone()
        prefix_len = seq_len - args.train_suffix_len
        print(
            f"  cached-prefix compare setup: effective={seq_len} prefix={prefix_len} "
            f"suffix={args.train_suffix_len} prefix_chunk={args.prefix_chunk_len}",
            flush=True,
        )
        run_cached_prefix_compare(
            model, input_ids, labels, prefix_len, args.train_suffix_len, args.prefix_chunk_len,
            args.sdpa_causal_no_mask, args.loss_chunk_tokens, args.loss_logits_dtype,
            loss_mode=args.loss_mode,
        )
        print("  [cached-prefix-compare-full reached]")
        stop_safety_watchdog(watchdog_stop)
        return 0
    if args.routing_census_only:
        if args.batch != 1:
            raise SystemExit("--routing-census-only currently supports --batch 1")
        record = train_records[0]
        if "input_ids" in record:
            input_ids = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
        else:
            enc = tok(record["text"], return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
            input_ids = enc.input_ids
            attention_mask = enc.attention_mask
        census_attention_mask = maybe_omit_full_attention_mask(attention_mask, args.sdpa_causal_no_mask)
        run_routing_census(
            model,
            input_ids,
            census_attention_mask,
            output_path=args.routing_census_output,
        )
        print("  [routing-census-only reached]")
        stop_safety_watchdog(watchdog_stop)
        return 0

    max_loop_idx = len(train_records) if args.batch == 1 else (len(train_records) // args.batch)
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
        total_steps = len(train_records) * N_EPOCHS
        print(f"\n--- training {N_EPOCHS} epoch x {len(train_records)} examples (max_len={args.max_len}, grad_accum={effective_grad_accum}, save_every={SAVE_EVERY}) ---")
    else:
        n_full_batches = len(train_records) // args.batch
        total_steps = n_full_batches * N_EPOCHS
        print(f"\n--- training {N_EPOCHS} epoch x {n_full_batches} full batches (batch={args.batch}, max_len={args.max_len}, grad_accum={effective_grad_accum}, save_every={SAVE_EVERY}) ---")
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    step = resume_step
    epoch = resume_epoch

    if args.batch == 1:
        for epoch in range(resume_epoch, N_EPOCHS):
            start_ex_idx = resume_step if epoch == resume_epoch else 0
            for ex_idx, record in enumerate(train_records[start_ex_idx:], start=start_ex_idx):
                ex = record["example"]
                text = record["text"]
                if "input_ids" in record:
                    input_ids = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
                    attention_mask = torch.ones_like(input_ids)
                else:
                    enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
                    input_ids = enc.input_ids
                    attention_mask = enc.attention_mask
                model_attention_mask = maybe_omit_full_attention_mask(attention_mask, args.sdpa_causal_no_mask)
                labels = input_ids.clone()
                if args.mask_prompt_labels:
                    mask_prompt_labels(labels, [text], [ex], tok, args.max_len)
                seq_len = input_ids.shape[1]
                real_tokens = int(attention_mask.sum().item())
                if args.profile_memory_phases:
                    torch.cuda.reset_peak_memory_stats()
                    print_memory_snapshot("step_start", step + 1)

                if args.training_mode == "cached_prefix_suffix":
                    if args.train_suffix_len >= seq_len:
                        raise SystemExit(
                            f"--train-suffix-len={args.train_suffix_len} must be < seq_len={seq_len}"
                        )
                    prefix_len = seq_len - args.train_suffix_len
                    loss, model_outputs = cached_prefix_suffix_loss(
                        model, input_ids, labels, prefix_len, args.train_suffix_len,
                        args.prefix_chunk_len, args.sdpa_causal_no_mask,
                        args.loss_chunk_tokens, args.loss_logits_dtype,
                        args.activation_offload,
                        profile_step=(step + 1 if args.profile_memory_phases else None),
                        loss_mode=args.loss_mode,
                    )
                    if args.profile_memory_phases:
                        print_memory_snapshot("after_cached_suffix_loss", step + 1)
                elif args.loss_mode in ("chunked_frozen_ce", "liger_flce"):
                    set_current_phase("base_forward")
                    with activation_offload_context(args.activation_offload):
                        model_outputs = model.model(
                            input_ids=input_ids,
                            attention_mask=model_attention_mask,
                            use_cache=False,
                            return_dict=True,
                        )
                        if args.profile_memory_phases:
                            print_memory_snapshot("after_base_forward", step + 1)
                        set_current_phase("chunked_loss" if args.loss_mode == "chunked_frozen_ce" else "liger_flce_loss")
                        loss = _lm_head_ce(
                            args.loss_mode, model_outputs[0], labels, model.lm_head,
                            args.loss_chunk_tokens, (args.loss_logits_dtype == "fp32"),
                        )
                    if args.profile_memory_phases:
                        print_memory_snapshot("after_chunked_loss", step + 1)
                else:
                    set_current_phase("hf_forward_loss")
                    with activation_offload_context(args.activation_offload):
                        out = model(input_ids=input_ids, attention_mask=model_attention_mask, labels=labels)
                        loss = out.loss
                    if args.profile_memory_phases:
                        print_memory_snapshot("after_hf_forward_loss", step + 1)
                set_current_phase("backward")
                scaled_loss = loss / effective_grad_accum
                scaled_loss.backward()
                if args.profile_memory_phases:
                    print_memory_snapshot("after_backward", step + 1)
                loss_history.append(float(loss.item()))

                step += 1
                if step % effective_grad_accum == 0:
                    set_current_phase("optimizer")
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    if args.profile_memory_phases:
                        print_memory_snapshot("after_optimizer", step)

                if step % 10 == 0 or step == 1:
                    elapsed = time.time() - t_start
                    eta_h = (elapsed / max(1, step)) * (total_steps - step) / 3600
                    avg_loss = sum(loss_history[-20:]) / max(1, len(loss_history[-20:]))
                    print(
                        f"  step {step}/{total_steps}: loss={loss_history[-1]:.4f} avg20={avg_loss:.4f} "
                        f"elapsed={elapsed/60:.1f}m eta={eta_h:.2f}h "
                        f"seq_len={seq_len} real_tokens={real_tokens} "
                        f"cuda_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
                        f"cuda_peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB "
                        f"cuda_reserved={torch.cuda.memory_reserved()/1e9:.2f}GB",
                        flush=True,
                    )

                if step % SAVE_EVERY == 0 and step % effective_grad_accum == 0:
                    n_saved = save_adapter(
                        model, model_dir, adapter_dir, loss_history, step, epoch,
                        args.batch, args.max_len, effective_grad_accum,
                        args.grad_accum, args.mask_prompt_labels,
                        dynamic_padding_enabled, length_bucketing_enabled,
                        args.pad_to_multiple_of, args.limit_examples,
                        args.select_longest_examples,
                        args.synthetic_repeat_to_len, args.synthetic_examples,
                        args.training_mode, args.train_suffix_len, args.prefix_chunk_len,
                        args.loss_mode, args.loss_chunk_tokens, args.loss_logits_dtype,
                        args.activation_offload, args.optimizer,
                        args.sdpa_causal_no_mask,
                        args.pooled_loader_buffers,
                        args.moe_sparse_no_one_hot,
                        args.checkpoint_group_size,
                        args.mamba_chunk_size,
                        args.lora_r, args.lora_alpha, target_suffixes,
                        optimizer=opt, save_optimizer_state=args.save_optimizer_state,
                    )
                    print(f"  [checkpoint @ step {step}] saved {n_saved} LoRA tensors -> {adapter_dir}/")

                if (
                    args.stop_at_step is not None
                    and step >= args.stop_at_step
                    and step % effective_grad_accum == 0
                ):
                    print(f"  [stop-at-step={args.stop_at_step} reached at accumulation boundary]")
                    break
            if args.stop_at_step is not None and step >= args.stop_at_step:
                break
    else:
        n_full_batches = len(train_records) // args.batch
        for epoch in range(resume_epoch, N_EPOCHS):
            start_batch = resume_step if epoch == resume_epoch else 0
            for batch_idx in range(start_batch, n_full_batches):
                batch_records = train_records[batch_idx * args.batch:(batch_idx + 1) * args.batch]
                batch_ex = [record["example"] for record in batch_records]
                texts = [record["text"] for record in batch_records]
                enc = tok(
                    texts, return_tensors="pt", truncation=True,
                    max_length=args.max_len,
                    padding=("longest" if dynamic_padding_enabled else "max_length"),
                    pad_to_multiple_of=args.pad_to_multiple_of,
                ).to(device)
                labels = enc.input_ids.clone()
                if args.mask_prompt_labels:
                    mask_prompt_labels(labels, texts, batch_ex, tok, args.max_len)
                labels[enc.attention_mask == 0] = -100
                seq_len = enc.input_ids.shape[1]
                real_tokens = int(enc.attention_mask.sum().item())
                model_attention_mask = maybe_omit_full_attention_mask(enc.attention_mask, args.sdpa_causal_no_mask)
                if args.profile_memory_phases:
                    torch.cuda.reset_peak_memory_stats()
                    print_memory_snapshot("step_start", step + 1)

                if args.loss_mode in ("chunked_frozen_ce", "liger_flce"):
                    set_current_phase("base_forward")
                    with activation_offload_context(args.activation_offload):
                        model_outputs = model.model(
                            input_ids=enc.input_ids,
                            attention_mask=model_attention_mask,
                            use_cache=False,
                            return_dict=True,
                        )
                        if args.profile_memory_phases:
                            print_memory_snapshot("after_base_forward", step + 1)
                        set_current_phase("chunked_loss" if args.loss_mode == "chunked_frozen_ce" else "liger_flce_loss")
                        loss = _lm_head_ce(
                            args.loss_mode, model_outputs[0], labels, model.lm_head,
                            args.loss_chunk_tokens, (args.loss_logits_dtype == "fp32"),
                        )
                    if args.profile_memory_phases:
                        print_memory_snapshot("after_chunked_loss", step + 1)
                else:
                    set_current_phase("hf_forward_loss")
                    with activation_offload_context(args.activation_offload):
                        out = model(input_ids=enc.input_ids, attention_mask=model_attention_mask, labels=labels)
                        loss = out.loss
                    if args.profile_memory_phases:
                        print_memory_snapshot("after_hf_forward_loss", step + 1)
                set_current_phase("backward")
                scaled_loss = loss / effective_grad_accum
                scaled_loss.backward()
                if args.profile_memory_phases:
                    print_memory_snapshot("after_backward", step + 1)
                loss_history.append(float(loss.item()))

                step += 1
                if step % effective_grad_accum == 0:
                    set_current_phase("optimizer")
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    if args.profile_memory_phases:
                        print_memory_snapshot("after_optimizer", step)

                if step % 10 == 0 or step == 1:
                    elapsed = time.time() - t_start
                    eta_h = (elapsed / max(1, step)) * (total_steps - step) / 3600
                    avg_loss = sum(loss_history[-20:]) / max(1, len(loss_history[-20:]))
                    print(
                        f"  step {step}/{total_steps}: loss={loss_history[-1]:.4f} avg20={avg_loss:.4f} "
                        f"elapsed={elapsed/60:.1f}m eta={eta_h:.2f}h "
                        f"seq_len={seq_len} real_tokens={real_tokens} "
                        f"cuda_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
                        f"cuda_peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB "
                        f"cuda_reserved={torch.cuda.memory_reserved()/1e9:.2f}GB",
                        flush=True,
                    )

                if step % SAVE_EVERY == 0 and step % effective_grad_accum == 0:
                    n_saved = save_adapter(
                        model, model_dir, adapter_dir, loss_history, step, epoch,
                        args.batch, args.max_len, effective_grad_accum,
                        args.grad_accum, args.mask_prompt_labels,
                        dynamic_padding_enabled, length_bucketing_enabled,
                        args.pad_to_multiple_of, args.limit_examples,
                        args.select_longest_examples,
                        args.synthetic_repeat_to_len, args.synthetic_examples,
                        args.training_mode, args.train_suffix_len, args.prefix_chunk_len,
                        args.loss_mode, args.loss_chunk_tokens, args.loss_logits_dtype,
                        args.activation_offload, args.optimizer,
                        args.sdpa_causal_no_mask,
                        args.pooled_loader_buffers,
                        args.moe_sparse_no_one_hot,
                        args.checkpoint_group_size,
                        args.mamba_chunk_size,
                        args.lora_r, args.lora_alpha, target_suffixes,
                        optimizer=opt, save_optimizer_state=args.save_optimizer_state,
                    )
                    print(f"  [checkpoint @ step {step}] saved {n_saved} LoRA tensors -> {adapter_dir}/")

                if (
                    args.stop_at_step is not None
                    and step >= args.stop_at_step
                    and step % effective_grad_accum == 0
                ):
                    print(f"  [stop-at-step={args.stop_at_step} reached at accumulation boundary]")
                    break
            if args.stop_at_step is not None and step >= args.stop_at_step:
                break

    # final flush + save
    if step % effective_grad_accum != 0:
        opt.step()
        opt.zero_grad(set_to_none=True)
    if args.save_final_adapter:
        save_adapter(
            model, model_dir, adapter_dir, loss_history, step, epoch,
            args.batch, args.max_len, effective_grad_accum,
            args.grad_accum, args.mask_prompt_labels,
            dynamic_padding_enabled, length_bucketing_enabled,
            args.pad_to_multiple_of, args.limit_examples,
            args.select_longest_examples,
            args.synthetic_repeat_to_len, args.synthetic_examples,
            args.training_mode, args.train_suffix_len, args.prefix_chunk_len,
            args.loss_mode, args.loss_chunk_tokens, args.loss_logits_dtype,
            args.activation_offload, args.optimizer,
            args.sdpa_causal_no_mask,
            args.pooled_loader_buffers,
            args.moe_sparse_no_one_hot,
            args.checkpoint_group_size,
            args.mamba_chunk_size,
            args.lora_r, args.lora_alpha, target_suffixes,
            optimizer=opt, save_optimizer_state=args.save_optimizer_state,
        )
    else:
        print("  [final adapter save skipped: --no-save-final-adapter]")
    print(f"\n=== Day 5 DONE: {step} steps, wall={(time.time()-t_start)/3600:.2f}h ===")
    print(f"final adapter: {adapter_dir}/")
    stop_safety_watchdog(watchdog_stop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
