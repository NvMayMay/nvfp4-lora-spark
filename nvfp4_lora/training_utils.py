"""Shared, model-agnostic training primitives.

This module deliberately does NOT import or run any model-family-specific code
(no Mamba patches, no cached-prefix-suffix, no Nemotron-H assumptions). It only
exposes the model-agnostic primitives that a NVFP4 LoRA trainer might reuse.

Packaging note: this module must import cleanly from an installed wheel, i.e.
without the repo-only top-level ``train/`` package on ``sys.path``. It therefore
imports nothing from ``train.*`` at module scope. The shipped trainers each carry
their own save/load/label-mask implementations (train/train_super_nvfp4.py and
scripts/train_nvfp4_lora.py), so the historical ``save_adapter`` /
``load_adapter_weights`` / ``mask_prompt_labels`` re-export shim that forwarded
into ``train.train_super_nvfp4`` had no callers and has been removed; keeping it
would have made a packaged ``import nvfp4_lora.training_utils`` an ImportError
land-mine the moment anything referenced those names.
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


__all__ = [
    "_CURRENT_PHASE",
    "set_current_phase",
    "get_current_phase",
    "build_optimizer",
]
