#!/usr/bin/env python3
"""Unified LoRA fine-tuning for NVFP4-quantized models on GB10 (DGX Spark).

One trainer for every supported model family. Supersedes the per-model
scripts (train_mistral_rh_nvfp4_lora_ich_smoke.py,
train_qwen3_5_122b_rh_nvfp4_lora_ich.py), which remain as frozen, proven
artifacts of their respective runs.

What is detected instead of hardcoded:
  * Model family (config.json model_type) -> auto class, expert-tensor key
    translation, PEFT scoping, submodules to freeze.
  * LoRA mechanism: if every --target-modules suffix is NVFP4-quantized in the
    checkpoint, LoRA is baked into NVFP4LoRALinear at load (PEFT cannot wrap
    those). If none are quantized (e.g. BF16 attention in the Mistral-RH
    recipe), standard PEFT wrapping with a family-scoped regex. Mixed targets
    are rejected with an explanation.

GB10 UMA lessons are applied via nvfp4_lora.gb10_prep:
  * expandable_segments alloc conf set before the first CUDA allocation
  * weight-sized buffers allocated directly on cuda (never CPU)
  * shard page cache dropped after assembly (posix_fadvise, no sudo)

Crash safety: atomic adapter saves (tmp+rename), checkpoint rotation,
best-by-val-loss tracking at <output_dir>/best/, and full resume via
--resume-from (adapter + optimizer + scheduler + RNG + deterministic
per-epoch data order).

Smoke example (Qwen3.5-122B):
    python -u scripts/train_nvfp4_lora.py \\
        --model-dir /path/to/RedHatAI-Qwen3.5-122B-A10B-NVFP4 \\
        --target-modules q_proj,k_proj,v_proj,o_proj \\
        --max-train-examples 8 --max-val-examples 4 --max-steps 3 \\
        --eval-every 0 --checkpoint-every 0 --output-dir /tmp/unified_smoke
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must run before the first CUDA allocation.
from nvfp4_lora.gb10_prep import set_alloc_conf  # noqa: E402

set_alloc_conf()

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from accelerate import init_empty_weights  # noqa: E402
from torch.nn.utils.rnn import pad_sequence  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from nvfp4_lora.experts import (  # noqa: E402
    NVFP4Experts3D,
    assemble_nvfp4_experts3d_batched,
    replace_moe_experts_with_nvfp4_3d,
)
from nvfp4_lora.gb10_prep import drop_shard_page_cache, memory_snapshot  # noqa: E402
from nvfp4_lora.linear import NVFP4LoRALinear  # noqa: E402
from nvfp4_lora.loader import (  # noqa: E402
    _assign_dequant_workspaces,
    list_quantized_modules,
    load_non_nvfp4_weights,
    replace_nvfp4_modules,
)


# =========================================================================
# Family registry — everything model-specific lives here
# =========================================================================
# auto_class:      which transformers Auto* builds the right text-trainable graph
# expert_prefix:   (in_memory_prefix, safetensors_prefix) for routed-expert keys
# peft_scope:      regex prefix anchoring PEFT target_modules to the text backbone
# freeze:          submodules of model.model to freeze (multimodal towers)
FAMILIES = {
    "qwen3_5_moe": {
        "auto_class": "causal_lm",
        "expert_prefix": ("model.", "model.language_model."),
        "peft_scope": r"^model\.layers\.",
        "freeze": (),
    },
    "qwen3_5_moe_text": {
        "auto_class": "causal_lm",
        "expert_prefix": ("model.", "model.language_model."),
        "peft_scope": r"^model\.layers\.",
        "freeze": (),
    },
    "mistral3": {
        "auto_class": "image_text_to_text",
        "expert_prefix": ("model.language_model.", "language_model.model."),
        "peft_scope": r"^model\.language_model\.",
        "freeze": ("vision_tower", "multi_modal_projector"),
    },
    "mistral4": {
        "auto_class": "image_text_to_text",
        "expert_prefix": ("model.language_model.", "language_model.model."),
        "peft_scope": r"^model\.language_model\.",
        "freeze": ("vision_tower", "multi_modal_projector"),
    },
}


def resolve_family(model_dir: Path) -> tuple[str, dict]:
    cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    model_type = getattr(cfg, "model_type", None)
    fam = FAMILIES.get(model_type)
    if fam is None:
        raise SystemExit(
            f"Unsupported model_type={model_type!r}. Known: {sorted(FAMILIES)}. "
            f"Add a FAMILIES entry (and a make_key_translator branch in loader.py "
            f"if the safetensors layout is new)."
        )
    return model_type, fam


def detect_lora_mode(model_dir: Path, target_suffixes: list[str]) -> str:
    """'native' if every target suffix is NVFP4-quantized in the checkpoint,
    'peft' if none are. Mixed -> hard error."""
    quantized = list_quantized_modules(model_dir)
    quantized_suffixes = {name.rsplit(".", 1)[-1] for name in quantized}
    hits = [s for s in target_suffixes if s in quantized_suffixes]
    if len(hits) == len(target_suffixes):
        return "native"
    if not hits:
        return "peft"
    raise SystemExit(
        f"Mixed LoRA targets: {hits} are NVFP4-quantized but "
        f"{sorted(set(target_suffixes) - set(hits))} are not. Native NVFP4-LoRA "
        f"and PEFT cannot be combined in one run; split the target list."
    )


# =========================================================================
# Dataset (messages -> input_ids/labels with assistant-only loss masking)
# =========================================================================
class ChatJsonlDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int, max_examples: int | None = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items: list[dict[str, torch.Tensor]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if max_examples is not None and len(self.items) >= max_examples:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                item = self._encode(obj["messages"])
                if item is not None:
                    self.items.append(item)

    def _tokenize(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _render(self, messages, add_generation_prompt: bool = False) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def _encode(self, messages):
        full_text = self._render(messages, add_generation_prompt=False)
        input_ids = self._tokenize(full_text)
        if not input_ids:
            return None
        labels = [-100] * len(input_ids)
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            prefix_ids = self._tokenize(self._render(messages[:index], add_generation_prompt=True))
            through_ids = self._tokenize(self._render(messages[: index + 1], add_generation_prompt=False))
            start = len(prefix_ids)
            end = min(len(through_ids), len(input_ids))
            for pos in range(start, end):
                labels[pos] = input_ids[pos]
        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]
        if all(l == -100 for l in labels):
            # Assistant turn fell entirely beyond max_length: zero supervised
            # tokens -> HF returns NaN loss, which would poison the adapter.
            return None
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_batch(batch, pad_token_id: int):
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# =========================================================================
# Model load
# =========================================================================
def _stage(tag: str) -> None:
    snap = memory_snapshot()
    print(f"[load-mem] {tag}: rss={snap['process_rss_gb']}GB cuda_free={snap['cuda_free_gb']}GB", flush=True)


def load_model(
    model_dir: Path,
    family: dict,
    device: torch.device,
    dtype: torch.dtype,
    lora_mode: str,
    target_suffixes: list[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    print("[load] building model on meta…", flush=True)
    cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    auto_cls = AutoModelForCausalLM if family["auto_class"] == "causal_lm" else AutoModelForImageTextToText
    with init_empty_weights():
        model = auto_cls.from_config(cfg, trust_remote_code=True)
    _stage("post-meta-build")

    print("[load] replacing fused-3D MoE blocks with NVFP4Experts3D…", flush=True)
    model_type = getattr(model.config, "model_type", None)
    # device= is load-bearing on GB10 UMA: the default (None) allocates the
    # packed expert buffers (~weight-sized) on CPU, permanently starving CUDA.
    replace_moe_experts_with_nvfp4_3d(model, model_family=model_type, device=device)
    _stage("post-moe-replace")

    native_targets = tuple(target_suffixes) if lora_mode == "native" else ()
    print(f"[load] replacing NVFP4 nn.Linear (mode={lora_mode}, native targets={list(native_targets)})…", flush=True)
    replace_nvfp4_modules(
        model, model_dir,
        target_lora_suffixes=native_targets,
        r=lora_r if native_targets else 0,
        lora_alpha=lora_alpha if native_targets else 0,
        lora_dropout=lora_dropout if native_targets else 0.0,
        device=device, dtype=dtype,
    )
    _stage("post-linear-replace")

    print("[load] assembling routed-expert NVFP4 buffers…", flush=True)
    mem_prefix, st_prefix = family["expert_prefix"]
    idx_obj = json.loads((model_dir / "model.safetensors.index.json").read_text())
    wm = idx_obj["weight_map"]
    for name, module in model.named_modules():
        if isinstance(module, NVFP4Experts3D):
            assert name.startswith(mem_prefix), f"unexpected expert path: {name!r}"
            st_name = st_prefix + name[len(mem_prefix):]
            assemble_nvfp4_experts3d_batched(module, st_name, model_dir, wm)
    _stage("post-expert-assembly")

    print("[load] loading non-NVFP4 weights (attention/embeddings/norms/lm_head)…", flush=True)
    load_non_nvfp4_weights(model, model_dir, device=device, dtype=dtype)
    _stage("post-non-nvfp4-load")

    _assign_dequant_workspaces(model, device=device, dtype=dtype)
    _stage("post-workspaces")

    # Catch any straggler CPU tensors (RoPE inv_freq etc. — should be tiny;
    # anything weight-sized here indicates a placement bug upstream).
    moved_bytes = 0
    for mod in model.modules():
        for nm, buf in list(mod.named_buffers(recurse=False)):
            if buf.device.type == "cpu":
                moved_bytes += buf.numel() * buf.element_size()
                mod._buffers[nm] = buf.to(device)
        for nm, par in list(mod.named_parameters(recurse=False)):
            if par.device.type == "cpu":
                moved_bytes += par.numel() * par.element_size()
                mod._parameters[nm] = torch.nn.Parameter(
                    par.data.to(device), requires_grad=par.requires_grad
                )
    if moved_bytes > 1e9:
        print(f"[load] WARNING: move-loop relocated {moved_bytes/1e9:.1f}GB from CPU; "
              f"a loader stage is allocating weight-sized buffers on the wrong device", flush=True)
    _stage("post-move-loop")

    before, after = drop_shard_page_cache(model_dir)
    print(f"[load] dropped shard page cache: cuda_free {before:.1f}GB -> {after:.1f}GB", flush=True)

    # Freeze multimodal towers (text-only training)
    inner = getattr(model, "model", model)
    for attr in family["freeze"]:
        sub = getattr(inner, attr, None)
        if sub is None:
            continue
        for p in sub.parameters():
            p.requires_grad = False

    return model


def attach_peft_lora(model, family: dict, target_suffixes: list[str],
                     lora_r: int, lora_alpha: int, lora_dropout: float):
    from peft import LoraConfig, get_peft_model
    # Scope to the text backbone so multimodal towers (whose weights may sit on
    # meta) can never match a bare suffix.
    target_regex = family["peft_scope"] + r".*\.(" + "|".join(target_suffixes) + r")$"
    peft_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias="none", target_modules=target_regex,
    )
    return get_peft_model(model, peft_cfg)


# =========================================================================
# Training loop
# =========================================================================
@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        n_tok = (batch["labels"] != -100).sum().item()
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok
    model.train()
    return total_loss / max(1, total_tokens)


def _save_adapter_atomic(model, tokenizer, dest_dir: Path, log_fn, *,
                         lora_mode: str, base_model_dir: str,
                         lora_r: int, lora_alpha: int, lora_dropout: float,
                         target_suffixes) -> None:
    """Atomic (tmp+rename) adapter save for either LoRA mechanism."""
    import shutil
    from safetensors.torch import save_file as safe_save_file

    dest_dir = Path(dest_dir)
    tmp_dir = dest_dir.with_name(dest_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if lora_mode == "native":
        state = {}
        skipped = []
        for name, mod in model.named_modules():
            if isinstance(mod, NVFP4LoRALinear) and mod.r > 0:
                a, b = mod.lora_A, mod.lora_B
                if (hasattr(a, "is_meta") and a.is_meta) or (hasattr(b, "is_meta") and b.is_meta):
                    skipped.append(name)
                    continue
                state[f"base_model.model.{name}.lora_A.weight"] = a.detach().cpu().contiguous()
                state[f"base_model.model.{name}.lora_B.weight"] = b.detach().cpu().contiguous()
        if skipped:
            log_fn("save_warn_dropping_meta_modules", count=len(skipped), sample=skipped[:3])
        safe_save_file(state, str(tmp_dir / "adapter_model.safetensors"))
        cfg = {
            "base_model_name_or_path": base_model_dir,
            "peft_type": "LORA", "task_type": "CAUSAL_LM",
            "r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout,
            "bias": "none", "target_modules": list(target_suffixes),
            "inference_mode": True, "fan_in_fan_out": False,
        }
        (tmp_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    else:
        from peft.utils import get_peft_model_state_dict
        sd = get_peft_model_state_dict(model)
        meta_keys = [k for k, v in sd.items() if hasattr(v, "is_meta") and v.is_meta]
        if meta_keys:
            log_fn("save_warn_dropping_meta_keys", count=len(meta_keys), sample=meta_keys[:3])
            sd = {k: v for k, v in sd.items() if k not in set(meta_keys)}
        safe_save_file({k: v.detach().contiguous() for k, v in sd.items()},
                       str(tmp_dir / "adapter_model.safetensors"),
                       metadata={"format": "pt"})
        model.peft_config[model.active_adapter].save_pretrained(str(tmp_dir))

    tokenizer.save_pretrained(str(tmp_dir))
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    os.rename(str(tmp_dir), str(dest_dir))


def _load_adapter_weights(model, adapter_dir: Path, lora_mode: str, log_fn) -> None:
    from safetensors.torch import load_file
    sd = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    if lora_mode == "native":
        loaded = 0
        for name, mod in model.named_modules():
            if isinstance(mod, NVFP4LoRALinear) and mod.r > 0:
                k_a = f"base_model.model.{name}.lora_A.weight"
                k_b = f"base_model.model.{name}.lora_B.weight"
                if k_a in sd and k_b in sd:
                    mod.lora_A.data.copy_(sd[k_a].to(mod.lora_A.device, mod.lora_A.dtype))
                    mod.lora_B.data.copy_(sd[k_b].to(mod.lora_B.device, mod.lora_B.dtype))
                    loaded += 1
        log_fn("resume_adapter_loaded", modules=loaded, path=str(adapter_dir))
    else:
        from peft.utils import set_peft_model_state_dict
        set_peft_model_state_dict(model, sd)
        log_fn("resume_adapter_loaded", keys=len(sd), path=str(adapter_dir))


def _save_train_state(dest_dir: Path, optim, sched, update_step: int, epoch: int) -> None:
    state = {
        "update_step": update_step,
        "epoch": epoch,
        "optimizer": optim.state_dict(),
        "scheduler": sched.state_dict(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "rng_python": random.getstate(),
    }
    tmp = Path(dest_dir) / "train_state.pt.tmp"
    torch.save(state, str(tmp))
    os.rename(str(tmp), str(Path(dest_dir) / "train_state.pt"))


def _load_train_state(resume_dir: Path, optim, sched, log_fn) -> int:
    state = torch.load(str(Path(resume_dir) / "train_state.pt"), map_location="cpu", weights_only=False)
    optim.load_state_dict(state["optimizer"])
    sched.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng_torch"])
    if state.get("rng_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["rng_cuda"])
    random.setstate(state["rng_python"])
    log_fn("resume_state_loaded", step=state["update_step"], epoch=state["epoch"])
    return state["update_step"]


def _rotate_checkpoints(output_dir: Path, keep: int = 2) -> None:
    import shutil
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return
    ckpts = sorted(
        (p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint_step_")),
        key=lambda p: int(p.name[len("checkpoint_step_"):]),
    )
    for p in ckpts[:-keep] if keep > 0 else ckpts:
        shutil.rmtree(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--train-file", required=True,
                    help="JSONL of {\"messages\": [...]} chat examples")
    ap.add_argument("--val-file", default=None,
                    help="Optional validation JSONL; enables evals + best tracking")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", required=True,
                    help="Comma-separated projection suffixes. The LoRA mechanism "
                         "(native NVFP4 vs PEFT) is detected from whether these are "
                         "quantized in the checkpoint.")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--max-train-examples", type=int, default=None)
    ap.add_argument("--max-val-examples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume-from", default=None,
                    help="checkpoint_step_N/ dir: loads adapter + optimizer/scheduler/RNG "
                         "and fast-forwards the deterministic data order.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.output_dir) / "metrics.jsonl"
    if not args.resume_from:
        metrics_path.unlink(missing_ok=True)

    def log(event: str, **kw):
        rec = {"ts": time.strftime("%H:%M:%S"), "event": event, **kw}
        print(f"[{rec['ts']}] {event}: {json.dumps(kw)}", flush=True)
        with open(metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    log("config", **vars(args))

    model_dir = Path(args.model_dir)
    model_type, family = resolve_family(model_dir)
    target_suffixes = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_mode = detect_lora_mode(model_dir, target_suffixes)
    log("strategy", model_type=model_type, auto_class=family["auto_class"],
        lora_mode=lora_mode, targets=target_suffixes)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    log("tokenizer_loaded", vocab_size=tok.vocab_size, pad_id=tok.pad_token_id)

    train_ds = ChatJsonlDataset(args.train_file, tok, args.max_length, args.max_train_examples)
    val_ds = (ChatJsonlDataset(args.val_file, tok, args.max_length, args.max_val_examples)
              if args.val_file else [])
    log("dataset_encoded", train=len(train_ds), val=len(val_ds))

    # Dedicated generator, re-seeded per epoch: order is a pure function of
    # (seed, epoch) so --resume-from can replay it exactly.
    data_gen = torch.Generator()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=data_gen,
                              collate_fn=lambda b: collate_batch(b, tok.pad_token_id))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate_batch(b, tok.pad_token_id))

    log("model_loading_start")
    t0 = time.time()
    model = load_model(
        model_dir, family, device, dtype,
        lora_mode=lora_mode, target_suffixes=target_suffixes,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    )
    log("model_loaded", seconds=round(time.time() - t0, 1))

    if lora_mode == "peft":
        model = attach_peft_lora(model, family, target_suffixes,
                                 args.lora_r, args.lora_alpha, args.lora_dropout)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_lora_modules = sum(1 for _, m in model.named_modules()
                         if isinstance(m, NVFP4LoRALinear) and m.r > 0)
    log("lora_attached", mode=lora_mode, targets=target_suffixes,
        native_modules=n_lora_modules, trainable=n_train)

    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    updates_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_updates = max(1, updates_per_epoch * args.epochs)
    if args.max_steps is not None:
        total_updates = min(total_updates, args.max_steps)
    warmup_steps = max(1, int(total_updates * args.warmup_ratio))
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_updates)
    log("optimizer_ready", total_updates=total_updates, warmup_steps=warmup_steps)

    resume_step = 0
    resuming = False
    if args.resume_from:
        resume_dir = Path(args.resume_from)
        _load_adapter_weights(model, resume_dir, lora_mode, log)
        resume_step = _load_train_state(resume_dir, optim, sched, log)
        resuming = resume_step > 0

    model.train()
    update_step = 0
    micro_step = 0
    run_start = time.time()
    best_val_loss = float("inf")
    best_dir = Path(args.output_dir) / "best"

    def save_to(dest):
        _save_adapter_atomic(
            model, tok, dest, log,
            lora_mode=lora_mode, base_model_dir=args.model_dir,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, target_suffixes=target_suffixes,
        )

    for epoch in range(args.epochs):
        data_gen.manual_seed(args.seed * 1000 + epoch)
        for batch in train_loader:
            micro_step += 1
            if resuming:
                if micro_step % args.grad_accum == 0:
                    update_step += 1
                    if update_step >= resume_step:
                        resuming = False
                        log("resume_fastforward_done", step=update_step, epoch=epoch)
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                update_step += 1
                log("train_step", step=update_step, epoch=epoch, loss=round(out.loss.item(), 4),
                    lr=round(sched.get_last_lr()[0], 7),
                    elapsed=round(time.time() - run_start, 1))

                if args.eval_every > 0 and update_step % args.eval_every == 0 and len(val_ds) > 0:
                    val_loss = evaluate(model, val_loader, device)
                    log("eval", step=update_step, val_loss=round(val_loss, 4))
                    if val_loss < best_val_loss:
                        prev = best_val_loss
                        best_val_loss = val_loss
                        log("new_best", step=update_step, val_loss=round(val_loss, 4),
                            prev_best=(round(prev, 4) if prev != float("inf") else None),
                            path=str(best_dir))
                        save_to(best_dir)

                if args.checkpoint_every > 0 and update_step % args.checkpoint_every == 0:
                    ckpt_dir = Path(args.output_dir) / f"checkpoint_step_{update_step}"
                    log("checkpoint_start", step=update_step, path=str(ckpt_dir))
                    save_to(ckpt_dir)
                    _save_train_state(ckpt_dir, optim, sched, update_step, epoch)
                    _rotate_checkpoints(Path(args.output_dir), keep=2)
                    log("checkpoint_done", step=update_step)

                if args.max_steps is not None and update_step >= args.max_steps:
                    break
        if args.max_steps is not None and update_step >= args.max_steps:
            break

    if len(val_ds) > 0:
        final_val = evaluate(model, val_loader, device)
        log("final_eval", val_loss=round(final_val, 4))
        if final_val < best_val_loss:
            prev = best_val_loss
            best_val_loss = final_val
            log("new_best", step=update_step, val_loss=round(final_val, 4),
                prev_best=(round(prev, 4) if prev != float("inf") else None),
                path=str(best_dir))
            save_to(best_dir)

    log("saving_adapter", path=args.output_dir)
    save_to(Path(args.output_dir))
    log("done",
        total_seconds=round(time.time() - run_start, 1),
        total_updates=update_step,
        best_val_loss=(round(best_val_loss, 4) if best_val_loss != float("inf") else None))


if __name__ == "__main__":
    main()
