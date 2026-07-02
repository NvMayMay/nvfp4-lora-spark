"""nvfp4_lora.adapter_keys -- the ONE source of truth for LoRA adapter key schema
and the rekey transforms that map a trained adapter onto a vLLM serve module tree.

Historically the same two facts were re-implemented in ~5 places:

  * the PEFT key schema (``base_model.model.<module>.lora_{A,B}[.weight]``), and
  * the ``model.layers.* -> language_model.model.layers.*`` "wrapped-model" remap,

open-coded in ``nybbloris/plan.py``, ``scripts/rekey_lora_for_vllm.py``,
``scripts/rekey_expert_lora_for_vllm.py`` and
``serve/vllm_patches/attention_only_lora_cutlass_moe.py``. Three parallel copies
of the wrapped remap drifted, which is the root cause of the "silent no-op
adapter" class (an adapter that loads but binds nothing). This module centralizes
all of it so there is exactly one place to read and change.

Two coordinate systems appear in this file; keep them straight:

  * MODULE PATHS -- the target-module names an adapter binds against, with the
    ``base_model.model.`` PEFT prefix and the ``.lora_{A,B}[.weight]`` suffix
    already stripped (e.g. ``model.layers.0.self_attn.q_proj`` or, on a wrapped
    base, ``language_model.model.layers.0.self_attn.q_proj``). This is what
    ``nybbloris.plan`` resolves against the base weight index. The ``REKEYS``
    table operates on module paths.

  * SAFETENSORS KEYS -- the full on-disk tensor names an offline rekey rewrites,
    ``base_model.model.<module>.lora_{A,B}[.weight]``. ``wrapped_remap_safetensors_key``
    operates on those.

Both spellings of the wrapped remap describe the SAME transform (swap the decoder
under ``language_model.``); they differ only by the prefix the caller carries.

PEFT saves ``lora_A.weight`` / ``lora_B.weight``. Native nvfp4-lora-spark EXPERT
adapters save the stacked expert tensors WITHOUT the ``.weight`` suffix
(``...experts.gate_up.lora_A``); both forms are recognized here so an expert
adapter is never mis-read as empty.
"""
from __future__ import annotations

import re

__all__ = [
    "PEFT_PREFIX",
    "LORA_SUFFIX_RE",
    "strip_base_prefix",
    "adapter_module_path",
    "is_lora_key",
    "identity",
    "wrapped_remap_module",
    "wrapped_remap_safetensors_key",
    "REKEYS",
    "rekey_by_name",
]

# ---------------------------------------------------------------------------
# PEFT adapter key schema
# ---------------------------------------------------------------------------
# PEFT prefixes every target with `base_model.model.`; the tensor then carries a
# `.lora_A`/`.lora_B` role, optionally followed by `.weight`.
PEFT_PREFIX = "base_model.model."

# A LoRA tensor key ends in `.lora_A` / `.lora_B`, optionally `.weight`.
#   dense/PEFT:        base_model.model.<mod>.lora_A.weight
#   native expert:     base_model.model.<block>.experts.gate_up.lora_A   (no .weight)
LORA_SUFFIX_RE = re.compile(r"\.lora_[AB](?:\.weight)?$")


def is_lora_key(key: str) -> bool:
    """True if `key` names a LoRA A/B tensor (with or without the `.weight` suffix)."""
    return key != "__metadata__" and bool(LORA_SUFFIX_RE.search(key))


def strip_base_prefix(key: str) -> str:
    """Drop the PEFT `base_model.model.` prefix if present, else return unchanged."""
    return key[len(PEFT_PREFIX):] if key.startswith(PEFT_PREFIX) else key


def adapter_module_path(key: str) -> str | None:
    """Reduce a LoRA tensor key to its bare target-module path, or None.

    ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight``
        -> ``model.layers.0.self_attn.q_proj``
    ``base_model.model.model.layers.3.mlp.experts.gate_up.lora_A`` (no .weight)
        -> ``model.layers.3.mlp.experts.gate_up``

    Returns None for a non-LoRA key (e.g. ``__metadata__``).
    """
    if not is_lora_key(key):
        return None
    return LORA_SUFFIX_RE.sub("", strip_base_prefix(key))


# ---------------------------------------------------------------------------
# Rekey transforms (operate on MODULE PATHS -- prefix + lora suffix already off)
# ---------------------------------------------------------------------------
# A multimodal `...ForConditionalGeneration` base nests its decoder under
# `language_model.`, so vLLM binds LoRA against `language_model.model.layers.*`.
# A flat text-decoder adapter carries `model.layers.*`; without this swap it
# resolves to a module vLLM never builds and the adapter silently no-ops
# (MEASURED: Qwen3.5-122B, cross_arch_status.md FINDING #3).
_MODULE_FLAT = "model.layers."
_MODULE_WRAPPED = "language_model.model.layers."


def identity(path: str) -> str:
    """No-op rekey (a flat causal-LM adapter binds directly)."""
    return path


def wrapped_remap_module(path: str) -> str:
    """Module-path form: `model.layers.* -> language_model.model.layers.*` (once)."""
    if path.startswith(_MODULE_FLAT):
        return _MODULE_WRAPPED + path[len(_MODULE_FLAT):]
    return path


# Canonical ordered list of (name, fn) rekey candidates, over MODULE PATHS.
# `nybbloris.plan` picks the one that resolves the most targets against the base.
# Extend as new serve layouts appear (add the transform here, nowhere else).
REKEYS: list[tuple[str, "callable[[str], str]"]] = [
    ("identity", identity),
    ("language_model", wrapped_remap_module),
]


def rekey_by_name(name: str):
    """Look up a rekey fn by its canonical name (raises KeyError if unknown)."""
    for n, fn in REKEYS:
        if n == name:
            return fn
    raise KeyError(f"unknown rekey {name!r}; known: {[n for n, _ in REKEYS]}")


# ---------------------------------------------------------------------------
# Rekey transform (operates on full SAFETENSORS KEYS -- prefix + suffix intact)
# ---------------------------------------------------------------------------
# Same `language_model` swap as above, but on the on-disk tensor name an OFFLINE
# rekey rewrites, i.e. with the `base_model.model.` PEFT prefix still attached:
#   base_model.model.model.layers.N...  ->  base_model.model.language_model.model.layers.N...
_ST_FLAT_PREFIX = PEFT_PREFIX + "model.layers."          # base_model.model.model.layers.
_ST_OLD = PEFT_PREFIX + "model."                         # base_model.model.model.
_ST_NEW = PEFT_PREFIX + "language_model.model."          # base_model.model.language_model.model.


def wrapped_remap_safetensors_key(key: str) -> str:
    """Safetensors-key form of the `language_model` wrapped remap.

    ``base_model.model.model.layers.N...``
        -> ``base_model.model.language_model.model.layers.N...``
    Keys not in the flat decoder layout (already-wrapped, expert-only, aux) pass
    through unchanged. This is exactly ``wrapped_remap_module`` with the PEFT
    prefix carried along.
    """
    if key.startswith(_ST_FLAT_PREFIX):
        return _ST_NEW + key[len(_ST_OLD):]
    return key
