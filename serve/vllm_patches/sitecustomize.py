"""
sitecustomize.py - auto-applies house vLLM patches.

Python imports `sitecustomize` automatically on every process startup if it
exists on `sys.path`. We use this to ensure patches are applied in vLLM's
spawned EngineCore subprocess (which is a fresh Python process and does not
inherit monkey-patches from the parent APIServer process).

Activation: prepend this directory to PYTHONPATH before launching vLLM.

Patches:
  - marlin_repack_patch: applied unconditionally (legacy behavior; harmless
    when the marlin backend is not used).
  - attention_only_lora_cutlass_moe: applied only when
    VLLM_PATCH_ATTN_ONLY_LORA_CUTLASS_MOE=1 is set, so launchers that share
    this PYTHONPATH but do not want dynamic LoRA are unaffected. Env vars
    propagate into the spawned EngineCore process.
"""

import os
import sys


def _try_apply_marlin_patch():
    # vLLM is loaded by both the API server and the spawned EngineCore.
    # The patch is only needed where the model loader actually runs (EngineCore).
    # apply_patch is idempotent so applying it twice is harmless.
    try:
        import marlin_repack_patch

        marlin_repack_patch.apply_patch()
        sys.stderr.write(
            f"[sitecustomize pid={os.getpid()}] applied marlin_repack_patch\n"
        )
    except ImportError as e:
        sys.stderr.write(
            f"WARNING: sitecustomize could not import marlin_repack_patch ({e}); "
            "vLLM marlin will run without the memory patch. "
            "Verify PYTHONPATH includes the serve/vllm_patches directory.\n"
        )
    except Exception as e:
        sys.stderr.write(
            f"[sitecustomize pid={os.getpid()}] marlin_repack_patch failed: {e!r}\n"
        )


def _try_apply_attn_only_lora_cutlass_moe_patch():
    # Dynamic attention-only LoRA over the CUTLASS NVFP4 MoE backend for
    # Qwen3.5-122B-A10B CT NVFP4. Opt-in via env var; see
    # attention_only_lora_cutlass_moe.py and
    # docs/plans/DYNAMIC_LORA_CUTLASS_PATCH.md.
    if os.environ.get("VLLM_PATCH_ATTN_ONLY_LORA_CUTLASS_MOE") != "1":
        return
    try:
        import attention_only_lora_cutlass_moe

        attention_only_lora_cutlass_moe.apply_patch()
        sys.stderr.write(
            f"[sitecustomize pid={os.getpid()}] applied "
            "attention_only_lora_cutlass_moe patch\n"
        )
    except ImportError as e:
        sys.stderr.write(
            "WARNING: sitecustomize could not import "
            f"attention_only_lora_cutlass_moe ({e}); dynamic LoRA over "
            "CUTLASS MoE will NOT work in this process. Verify PYTHONPATH "
            "includes the serve/vllm_patches directory.\n"
        )
    except Exception as e:
        sys.stderr.write(
            f"[sitecustomize pid={os.getpid()}] "
            f"attention_only_lora_cutlass_moe failed: {e!r}\n"
        )


_try_apply_marlin_patch()
_try_apply_attn_only_lora_cutlass_moe_patch()
