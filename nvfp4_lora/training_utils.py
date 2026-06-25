"""Phase 0.4 — Shared training utilities for per-model trainer scripts.

The Nemotron-Super trainer (train/train_super_nvfp4.py) keeps its existing
in-file implementations for compat. New per-model trainers (Mistral4, Qwen3.5)
import from here.

This module deliberately does NOT import or run any model-family-specific code
(no Mamba patches, no cached-prefix-suffix, no Nemotron-H assumptions). It only
exposes the model-agnostic primitives that every NVFP4 LoRA trainer needs.
"""
from __future__ import annotations

from typing import Iterable, Tuple


# --------------------------------------------------------------------------------------
# Phase-tagged watchdog labels (model-agnostic)
# --------------------------------------------------------------------------------------
_CURRENT_PHASE = "init"


def set_current_phase(label: str) -> None:
    global _CURRENT_PHASE
    _CURRENT_PHASE = label


def get_current_phase() -> str:
    return _CURRENT_PHASE


# --------------------------------------------------------------------------------------
# Optimizer dispatch with `lr` as explicit parameter (per Sonnet pass-1 note)
# --------------------------------------------------------------------------------------
def build_optimizer(
    trainable: Iterable,
    optimizer_name: str,
    lr: float,
) -> Tuple["torch.optim.Optimizer", str]:
    """Build a torch optimizer over `trainable` params with explicit LR.

    Supported: "adamw", "adamw8bit" (via torchao), "adafactor" (via transformers).
    """
    import torch
    if optimizer_name == "adamw":
        return torch.optim.AdamW(trainable, lr=lr), "AdamW"
    if optimizer_name == "adamw8bit":
        from torchao.optim import AdamW8bit
        return AdamW8bit(trainable, lr=lr), "torchao AdamW8bit"
    if optimizer_name == "adafactor":
        from transformers.optimization import Adafactor
        return (
            Adafactor(
                trainable,
                lr=lr,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
                weight_decay=0.0,
            ),
            "Transformers Adafactor",
        )
    raise SystemExit(f"unknown optimizer: {optimizer_name}")


# --------------------------------------------------------------------------------------
# Re-exports of model-agnostic helpers from train_super_nvfp4.py
# (Wrapping the existing implementations keeps Nemotron training unbroken and
# avoids duplicating ~200 lines. Per-model trainers import via this module so
# they don't pull in Nemotron-specific helpers as a side effect.)
# --------------------------------------------------------------------------------------
def save_adapter(*args, **kwargs):
    """PEFT-format adapter save — agnostic across model families."""
    from train.train_super_nvfp4 import save_adapter as _impl
    return _impl(*args, **kwargs)


def load_adapter_weights(*args, **kwargs):
    """PEFT-format adapter load — agnostic across model families."""
    from train.train_super_nvfp4 import load_adapter_weights as _impl
    return _impl(*args, **kwargs)


def mask_prompt_labels(*args, **kwargs):
    """Label masking — agnostic (tokenizer already parameterized).

    Real portability risk per codex round-1 finding is in the chat-template
    boundary detection (`_assistant_response_start_char`), not in tokenizer
    ownership. Per-model trainers should smoke-test this against their target
    chat template.
    """
    from train.train_super_nvfp4 import mask_prompt_labels as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "_CURRENT_PHASE",
    "set_current_phase",
    "get_current_phase",
    "build_optimizer",
    "save_adapter",
    "load_adapter_weights",
    "mask_prompt_labels",
]
