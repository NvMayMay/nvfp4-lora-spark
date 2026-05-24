"""
sitecustomize.py - auto-applies the Marlin NVFP4 MoE repack memory fix.

Python imports `sitecustomize` automatically on every process startup if it
exists on `sys.path`. We use this to ensure the patch is applied in vLLM's
spawned EngineCore subprocess (which is a fresh Python process and does not
inherit monkey-patches from the parent APIServer process).

Activation: prepend this directory to PYTHONPATH before launching vLLM.
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
    except ImportError:
        # marlin_repack_patch not on path - silently ignore
        pass
    except Exception as e:
        sys.stderr.write(
            f"[sitecustomize pid={os.getpid()}] marlin_repack_patch failed: {e!r}\n"
        )


_try_apply_marlin_patch()
