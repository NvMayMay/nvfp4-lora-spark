#!/usr/bin/env python3
"""Unified LoRA fine-tuning for NVFP4-quantized models on GB10 (DGX Spark).

One trainer for every supported model family. Supersedes the per-model
scripts (train_mistral_rh_nvfp4_lora_ich_smoke.py,
train_qwen3_5_122b_rh_nvfp4_lora_ich.py), which remain as frozen, proven
artifacts of their respective runs.

What is detected instead of hardcoded:
  * Model family (config.json model_type) -> auto class, expert-tensor key
    translation, PEFT scoping, submodules to freeze.
  * LoRA mechanism: each target trains through a frozen-base LoRALinear in the
    NATIVE path -- NVFP4 -> NVFP4LoRALinear, FP8 -> FP8LoRALinear, BF16 ->
    BF16LoRALinear -- so NVFP4/FP8/BF16 targets co-train in one adapter. Only an
    all-BF16 target set takes the standard PEFT path (family-scoped regex).

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
import hashlib  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
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
    detect_moe_expert_storage,
    replace_moe_experts_with_nvfp4_3d,
)
from nvfp4_lora.families import FAMILIES, resolve_family  # noqa: E402, F401
from nvfp4_lora.gb10_prep import drop_shard_page_cache, memory_snapshot  # noqa: E402
from nvfp4_lora.chat_encode import encode_chat_example  # noqa: E402
from nvfp4_lora.linear import BF16LoRALinear, FP8LoRALinear, NVFP4LoRALinear  # noqa: E402
from nvfp4_lora.loader import (  # noqa: E402
    _assign_dequant_workspaces,
    assert_no_meta_tensors,
    decide_lora_mode,
    load_non_nvfp4_weights,
    replace_bf16_targets,
    replace_nvfp4_modules,
)

# The family registry (FAMILIES / resolve_family) lives in nvfp4_lora/families.py
# and is shared with the loader, the checkpoint inspector and the merge scripts.
# They are re-exported above so existing callers and tests keep working.


def detect_lora_mode(
    model_dir: Path,
    target_suffixes: list[str],
    allow_partial_targets: bool = False,
) -> tuple[str, dict]:
    """'native' if the target modules are NVFP4-quantized in the checkpoint,
    'peft' if they are plain BF16. Returns (mode, coverage_report).

    Unlike the v1 suffix-set heuristic, this classifies EVERY matching module
    individually (via loader.decide_lora_mode), so partial quantization across
    layers and FP8-demoted targets are hard errors instead of silent gaps.
    """
    return decide_lora_mode(
        model_dir,
        target_suffixes,
        allow_partial_targets=allow_partial_targets,
    )


def _sha256_file(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return None


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_run_meta(args, coverage) -> dict:
    """Pure-ish run metadata bundle; no torch/transformers/peft imports."""
    arg_snapshot = dict(sorted(vars(args).items()))
    files = {
        "train_file": {
            "path": arg_snapshot.get("train_file"),
            "sha256": _sha256_file(arg_snapshot.get("train_file")),
        },
        "val_file": {
            "path": arg_snapshot.get("val_file"),
            "sha256": _sha256_file(arg_snapshot.get("val_file")),
        },
    }
    return {
        "args": arg_snapshot,
        "coverage": coverage,
        "files": files,
        "git_sha": _git_sha(),
        "versions": {
            "peft": _package_version("peft"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
    }


def _cuda_metrics():
    out = {
        "cuda_allocated_gb": None,
        "cuda_free_gb": None,
        "cuda_reserved_gb": None,
    }
    try:
        if not torch.cuda.is_available():
            return out
        out["cuda_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 4)
        out["cuda_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 4)
        free, _total = torch.cuda.mem_get_info()
        out["cuda_free_gb"] = round(free / 1e9, 4)
    except Exception:
        return {
            "cuda_allocated_gb": None,
            "cuda_free_gb": None,
            "cuda_reserved_gb": None,
        }
    return out


def _host_mem_available_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().available / 1e9, 4)
    except Exception:
        return None


def build_metrics_row(
    step,
    total_updates,
    window_supervised_tokens,
    wall_elapsed,
    recent_upd_s,
    loss_window_mean,
):
    updates_s = None
    if recent_upd_s is not None and math.isfinite(recent_upd_s) and recent_upd_s > 0:
        updates_s = 1.0 / recent_upd_s
    supervised_tokens_s = None
    if recent_upd_s is not None and math.isfinite(recent_upd_s) and recent_upd_s > 0:
        supervised_tokens_s = window_supervised_tokens / recent_upd_s
    eta_s = None
    if step > 0 and wall_elapsed is not None and math.isfinite(wall_elapsed):
        eta_s = max(0.0, (total_updates - step) * (wall_elapsed / step))
    row = {
        "eta_s": (round(eta_s, 1) if eta_s is not None else None),
        "loss_window_mean": (round(loss_window_mean, 4) if loss_window_mean is not None else None),
        "supervised_tokens_s": (round(supervised_tokens_s, 2) if supervised_tokens_s is not None else None),
        "updates_s": (round(updates_s, 4) if updates_s is not None else None),
        "window_supervised_tokens": int(window_supervised_tokens),
    }
    row.update(_cuda_metrics())
    row["host_mem_available_gb"] = _host_mem_available_gb()
    return row


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
        # mistral_common (tekken) models tokenize chat token-natively; the HF
        # text-render path below is wrong for them (broken LlamaTokenizerFast).
        if getattr(self.tokenizer, "is_mistral_common", False):
            return self.tokenizer.encode_chat(messages, self.max_length)
        encoded = encode_chat_example(messages, self.tokenizer, self.max_length)
        if encoded["dropped_reason"] is not None:
            return None
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "labels": torch.tensor(encoded["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
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
    strict: bool = True,
):
    print("[load] building model on meta…", flush=True)
    cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    auto_cls = AutoModelForCausalLM if family["auto_class"] == "causal_lm" else AutoModelForImageTextToText
    with init_empty_weights():
        model = auto_cls.from_config(cfg, trust_remote_code=True)
    _stage("post-meta-build")

    model_type = getattr(model.config, "model_type", None)
    if family.get("moe_experts_class"):
        # Probe the checkpoint's expert storage: ModelOpt vs compressed-tensors
        # key naming, and whether gate/up per-tensor scales differ (which
        # requires split storage with one global scale per projection).
        idx_obj = json.loads((model_dir / "model.safetensors.index.json").read_text())
        moe_storage = detect_moe_expert_storage(model_dir, idx_obj["weight_map"])
        if moe_storage is None:
            # Dense variant of a family whose model_type also covers MoE members:
            # mistral3 spans both the dense Mistral-Small-3.2-24B and the MoE
            # Mistral-Small-4-119B. No per-expert keys -> treat as a dense variant,
            # skip fused-3D replacement; replace_nvfp4_modules handles the dense MLP
            # (and the expert-assembly loop below is a no-op with no NVFP4Experts3D).
            print("[load] moe_experts_class declared but checkpoint has no per-expert "
                  "keys; treating as a dense variant of this family (skipping MoE "
                  "replacement)", flush=True)
        else:
            print(f"[load] replacing fused-3D MoE blocks with NVFP4Experts3D "
                  f"(format={moe_storage['quant_format']}, "
                  f"split_gate_up_scales={moe_storage['split_gate_up_scales']})…", flush=True)
            # device= is load-bearing on GB10 UMA: the default (None) allocates the
            # packed expert buffers (~weight-sized) on CPU, permanently starving CUDA.
            replace_moe_experts_with_nvfp4_3d(
                model, model_family=model_type, device=device,
                quant_format=moe_storage["quant_format"],
                split_gate_up_scales=moe_storage["split_gate_up_scales"],
            )
            _stage("post-moe-replace")
    else:
        # Family stores routed experts as per-expert nn.Linear modules
        # (Nemotron); replace_nvfp4_modules handles them like any other linear.
        print("[load] no fused-3D MoE for this family; experts are per-module linears", flush=True)

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

    if family.get("expert_prefix"):
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
    load_non_nvfp4_weights(model, model_dir, device=device, dtype=dtype, strict=strict)
    _stage("post-non-nvfp4-load")

    # Co-train genuinely-BF16 targets (e.g. attention a mixed-precision quantizer left in
    # bf16) alongside the NVFP4/FP8 ones in this single native adapter. Runs AFTER the bf16
    # weights are loaded; family peft_scope keeps out-of-scope BF16 (MTP heads / vision
    # towers) frozen. No-op when there are no in-scope BF16 targets (e.g. the 3.6, whose
    # attention is FP8).
    if lora_mode == "native":
        n_bf16 = replace_bf16_targets(
            model, target_suffixes, family.get("peft_scope"),
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            device=device, dtype=dtype,
        )
        if n_bf16:
            print(f"[load] wrapped {n_bf16} BF16 target Linears with BF16LoRALinear "
                  f"(co-trained natively)", flush=True)
        _stage("post-bf16-targets")

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

    # Tied embeddings never appear as a separate lm_head tensor on disk; re-tie
    # so lm_head does not stay on meta. No-op for untied checkpoints.
    if getattr(cfg, "tie_word_embeddings", False) and hasattr(model, "tie_weights"):
        model.tie_weights()

    # Everything still on meta at this point was never loaded and will explode
    # at first forward; only the family's frozen multimodal towers are allowed.
    meta_allowed = tuple(family.get("meta_allowed_prefixes", ()))
    if strict:
        assert_no_meta_tensors(model, allowed_prefixes=meta_allowed)
    else:
        try:
            assert_no_meta_tensors(model, allowed_prefixes=meta_allowed)
        except RuntimeError as e:
            print(f"[load] WARNING (--permissive-load): {e}", flush=True)

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
            if isinstance(mod, (NVFP4LoRALinear, FP8LoRALinear, BF16LoRALinear)) and mod.r > 0:
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
    # Move files into dest individually (os.replace is atomic per file).
    # Never rmtree dest_dir: the final save targets the OUTPUT ROOT, which
    # also holds best/, checkpoint_step_*/ and metrics.jsonl -- a whole-dir
    # swap deletes them all (this destroyed the best adapter of the Mistral
    # 119B v3.5 run).
    dest_dir.mkdir(parents=True, exist_ok=True)
    for item in tmp_dir.iterdir():
        os.replace(str(item), str(dest_dir / item.name))
    tmp_dir.rmdir()


def _load_adapter_weights(model, adapter_dir: Path, lora_mode: str, log_fn) -> None:
    from safetensors.torch import load_file
    sd = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    if lora_mode == "native":
        loaded = 0
        for name, mod in model.named_modules():
            if isinstance(mod, (NVFP4LoRALinear, FP8LoRALinear, BF16LoRALinear)) and mod.r > 0:
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
    ap.add_argument("--train-file", required=False, default=None,
                    help="JSONL of {\"messages\": [...]} chat examples. Required "
                         "unless --dry-run is set.")
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
    ap.add_argument("--allow-partial-targets", action="store_true",
                    help="DEPRECATED / no-op: a target suffix that is NVFP4 in some "
                         "layers and BF16 in others now co-trains both natively "
                         "(quantized via NVFP4LoRALinear, BF16 via BF16LoRALinear), so "
                         "no flag is needed. Accepted for backward compatibility.")
    ap.add_argument("--permissive-load", action="store_true",
                    help="Bring-up escape hatch: downgrade strict-load errors "
                         "(unmapped on-disk tensors, tensors left on the meta "
                         "device) to warnings. Never use for a real run.")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-subset", type=int, default=0,
                    help="If >0, in-flight evals run only over the first N val "
                         "examples (cheap). A subset eval that beats the current "
                         "best triggers a confirming FULL eval; only the full value "
                         "updates best tracking. 0 means every eval is full.")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--max-train-examples", type=int, default=None)
    ap.add_argument("--max-val-examples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume-from", default=None,
                    help="checkpoint_step_N/ dir: loads adapter + optimizer/scheduler/RNG "
                         "and fast-forwards the deterministic data order.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preflight OOM probe: load the model exactly as a real run "
                         "would (load_model + LoRA + gradient checkpointing + optimizer), "
                         "run one synthetic forward+backward at (batch_size, max_length), "
                         "log a memory reading, and exit WITHOUT saving any adapter. Catches "
                         "out-of-memory in ~12 minutes instead of mid-run. --train-file is "
                         "not required in this mode.")
    ap.add_argument("--fused-linear-ce", action="store_true",
                    help="Bind Liger fused linear cross-entropy (lce_forward) onto the "
                         "causal-LM head: computes the loss without ever materializing the "
                         "full (seq x vocab) logits tensor or its fp32 upcast, cutting ~10 GB "
                         "of peak activation. Required for seq>=8192 on 121 GB UMA. Binds ONLY "
                         "lce_forward (NOT apply_liger_kernel_*, which would rewrite MoE MLPs "
                         "into dense SwiGLU and corrupt NVFP4 experts). GLM-family validated.")
    args = ap.parse_args()

    if not args.dry_run and not args.train_file:
        ap.error("--train-file is required unless --dry-run is set")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.output_dir) / "metrics.jsonl"
    if not args.resume_from:
        metrics_path.unlink(missing_ok=True)

    def log(event: str, **kw):
        rec = {"ts": time.strftime("%H:%M:%S"), "event": event, **kw}
        print(f"[{rec['ts']}] {event}: {json.dumps(kw, sort_keys=True)}", flush=True)
        with open(metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    log("config", **vars(args))

    model_dir = Path(args.model_dir)
    model_type, family = resolve_family(model_dir)
    target_suffixes = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_mode, coverage = detect_lora_mode(
        model_dir, target_suffixes,
        allow_partial_targets=args.allow_partial_targets,
    )
    log("strategy", model_type=model_type, auto_class=family["auto_class"],
        lora_mode=lora_mode, targets=target_suffixes)

    # Persist the exact target coverage next to the adapter so every run is
    # auditable: which modules were trained natively, which via PEFT, which
    # were FP8-demoted or skipped.
    coverage["model_type"] = model_type
    coverage["model_dir"] = str(model_dir)
    (Path(args.output_dir) / "target_coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True)
    )
    run_meta = build_run_meta(args, coverage)
    meta_name = "resume_meta.json" if args.resume_from else "run_meta.json"
    (Path(args.output_dir) / meta_name).write_text(json.dumps(run_meta, indent=2, sort_keys=True))
    if args.resume_from:
        original_meta_path = Path(args.output_dir) / "run_meta.json"
        if original_meta_path.exists():
            try:
                original_args = json.loads(original_meta_path.read_text()).get("args", {})
                changed = sorted(
                    k for k, v in run_meta["args"].items()
                    if k != "resume_from" and original_args.get(k) != v
                )
            except Exception:
                changed = ["<unreadable run_meta.json>"]
            if changed:
                log("resume_args_differ", changed=changed)
    for suffix, info in coverage["inventory"].items():
        log("target_coverage", suffix=suffix, counts=info["counts"])

    device = torch.device("cuda")
    dtype = torch.bfloat16

    from nvfp4_lora.mistral_tok import MistralCommonTokenizer, has_tekken
    if has_tekken(args.model_dir):
        # Mistral repacks ship a broken HF tokenizer; use the native tekken one.
        tok = MistralCommonTokenizer(args.model_dir)
        log("tokenizer_loaded", backend="mistral_common",
            vocab_size=tok.vocab_size, pad_id=tok.pad_token_id)
    else:
        tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        log("tokenizer_loaded", backend="hf",
            vocab_size=tok.vocab_size, pad_id=tok.pad_token_id)

    # In --dry-run we synthesize one batch later, so skip dataset/dataloader
    # construction entirely (no train-file needed, no tokenization cost).
    if not args.dry_run:
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
        strict=not args.permissive_load,
    )
    log("model_loaded", seconds=round(time.time() - t0, 1))

    if lora_mode == "peft":
        model = attach_peft_lora(model, family, target_suffixes,
                                 args.lora_r, args.lora_alpha, args.lora_dropout)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_lora_modules = sum(1 for _, m in model.named_modules()
                         if isinstance(m, (NVFP4LoRALinear, FP8LoRALinear, BF16LoRALinear)) and m.r > 0)
    log("lora_attached", mode=lora_mode, targets=target_suffixes,
        native_modules=n_lora_modules, trainable=n_train)

    if args.fused_linear_ce:
        # Replace lm_head+CE with Liger's chunked fused linear cross-entropy. The
        # full (seq x vocab) logits + its fp32 upcast is the single largest train-time
        # activation spike at long seq (~10 GB at seq 8192, vocab 151552); FLCE never
        # materializes it. Bind ONLY lce_forward onto the causal-LM module: it calls
        # self.model(...) (the backbone, incl. MoE routing and the NVFP4 custom autograd)
        # unchanged and only chunks the head. Do NOT use apply_liger_kernel_to_glm4 — on
        # an instance it rewrites every decoder_layer.mlp into a dense SwiGLU MLP, which
        # corrupts the Glm4MoeNaiveMoe / NVFP4Experts3D blocks.
        from types import MethodType
        from liger_kernel.transformers.model.glm4 import lce_forward as _lce_forward
        ce_target = model.base_model.model if lora_mode == "peft" else model
        if not (hasattr(ce_target, "model") and hasattr(ce_target, "lm_head")):
            raise RuntimeError(
                f"--fused-linear-ce target {type(ce_target).__name__} lacks .model/.lm_head; "
                f"lce_forward binding only supports a causal-LM head."
            )
        ce_target.forward = MethodType(_lce_forward, ce_target)
        log("fused_linear_ce_enabled", target=type(ce_target).__name__)

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

    if args.dry_run:
        # Preflight OOM probe. Everything memory-relevant (full load, LoRA attach,
        # gradient checkpointing, optimizer state) is already constructed above; the
        # only thing left to exercise is a real forward+backward at the configured
        # shape. We synthesize one max-length batch (the worst case the real run
        # will see) rather than touching the dataset, then exit without saving.
        snap_post_load = memory_snapshot()
        torch.cuda.reset_peak_memory_stats()
        synth = torch.randint(
            0, int(tok.vocab_size), (args.batch_size, args.max_length),
            dtype=torch.long, device=device,
        )
        batch = {
            "input_ids": synth,
            "labels": synth.clone(),
            "attention_mask": torch.ones_like(synth),
        }
        out = model(**batch)
        loss = out.loss / args.grad_accum
        loss.backward()
        snap_post_backward = memory_snapshot()
        optim.zero_grad(set_to_none=True)
        log("dry_run_ok",
            batch_size=args.batch_size, max_length=args.max_length,
            loss=round(out.loss.item(), 4),
            post_load=snap_post_load, post_backward=snap_post_backward,
            cuda_max_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
        return

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
    last_update_time = run_start
    window_supervised_tokens = 0
    window_loss_sum = 0.0
    window_loss_n = 0
    best_val_loss = float("inf")
    best_dir = Path(args.output_dir) / "best"

    def save_to(dest):
        _save_adapter_atomic(
            model, tok, dest, log,
            lora_mode=lora_mode, base_model_dir=args.model_dir,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, target_suffixes=target_suffixes,
        )

    # Subset-eval loader: when --eval-subset N>0, in-flight evals run over only
    # the first N val examples. Subset losses are NOT comparable to full-eval
    # losses, so they can never directly update best_val_loss (which always
    # tracks FULL values); a subset improvement only triggers a confirming full
    # eval. Built once over a fixed prefix so the cheap eval is reproducible.
    from torch.utils.data import Subset
    subset_n = args.eval_subset if args.eval_subset > 0 else 0
    if subset_n > 0 and len(val_ds) > 0:
        subset_val_loader = DataLoader(
            Subset(val_ds, range(min(subset_n, len(val_ds)))),
            batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda b: collate_batch(b, tok.pad_token_id))
    else:
        subset_val_loader = None

    def run_inflight_eval(step):
        """Evaluate at `step`, honoring --eval-subset, and update best tracking.

        Subset evals are logged with subset=N. When a subset eval beats the
        current (full) best, an immediate FULL eval (logged subset="full")
        confirms it; only the full value may update best_val_loss / trigger
        new_best / save best/. Both events are emitted so metrics.jsonl is
        unambiguous about which loss is which.
        """
        nonlocal best_val_loss
        if subset_val_loader is not None:
            sub_loss = evaluate(model, subset_val_loader, device)
            log("eval", step=step, val_loss=round(sub_loss, 4), subset=subset_n)
            if sub_loss >= best_val_loss:
                return
            # Subset says we may have improved; confirm with a full eval. Its
            # value (not the subset's) is what is allowed to set the best.
            full_loss = evaluate(model, val_loader, device)
            log("eval", step=step, val_loss=round(full_loss, 4), subset="full")
        else:
            full_loss = evaluate(model, val_loader, device)
            log("eval", step=step, val_loss=round(full_loss, 4), subset="full")
        if full_loss < best_val_loss:
            prev = best_val_loss
            best_val_loss = full_loss
            log("new_best", step=step, val_loss=round(full_loss, 4),
                prev_best=(round(prev, 4) if prev != float("inf") else None),
                path=str(best_dir))
            save_to(best_dir)

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
            if not torch.isfinite(out.loss):
                # A non-finite loss this micro-step means the whole accumulation
                # window is unusable: earlier micro-batches already wrote grads,
                # and stepping on a partial window would bias the update. Drop
                # every accumulated grad and abandon the window (update_step is
                # intentionally NOT incremented).
                log("nonfinite_loss_skipped", step=update_step, micro_step=micro_step,
                    loss=str(out.loss.detach().float().item()))
                optim.zero_grad(set_to_none=True)
                window_supervised_tokens = 0
                window_loss_sum = 0.0
                window_loss_n = 0
                continue
            window_supervised_tokens += int((batch["labels"] != -100).sum().item())
            window_loss_sum += float(out.loss.detach().float().item())
            window_loss_n += 1
            loss = out.loss / args.grad_accum
            loss.backward()
            if micro_step % args.grad_accum == 0:
                total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                if not torch.isfinite(total_norm):
                    # Grads went non-finite (overflow / a bad micro-batch that
                    # still produced a finite loss). Skip optim.step/sched.step,
                    # drop the grads, and do NOT increment update_step so the
                    # cosine schedule and step count stay consistent.
                    log("nonfinite_grad_skipped", step=update_step, micro_step=micro_step,
                        grad_norm=str(total_norm.detach().float().item()))
                    optim.zero_grad(set_to_none=True)
                    window_supervised_tokens = 0
                    window_loss_sum = 0.0
                    window_loss_n = 0
                    continue
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                update_step += 1
                now = time.time()
                metrics_extra = build_metrics_row(
                    update_step,
                    total_updates,
                    window_supervised_tokens,
                    now - run_start,
                    now - last_update_time,
                    window_loss_sum / window_loss_n if window_loss_n else None,
                )
                last_update_time = now
                log("train_step", step=update_step, epoch=epoch, loss=round(out.loss.item(), 4),
                    lr=round(sched.get_last_lr()[0], 7),
                    elapsed=round(now - run_start, 1),
                    **metrics_extra)
                window_supervised_tokens = 0
                window_loss_sum = 0.0
                window_loss_n = 0

                if args.eval_every > 0 and update_step % args.eval_every == 0 and len(val_ds) > 0:
                    run_inflight_eval(update_step)

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
