"""Shared pytest fixtures and path setup for the CPU-only test suite.

Every test in this directory is designed to run without a GPU. We never build a
real model, never load real shards, and never allocate a CUDA tensor: the suite
exercises pure-Python / pure-CPU logic so it can run in GitHub Actions and as a
guard while a GPU training run holds the local device.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

try:
    import torch
except Exception:  # pragma: no cover - torch is always present in CI
    torch = None

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Make the `nvfp4_lora` package importable.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_train_module():
    """Import scripts/train_nvfp4_lora.py as a module.

    The script is under scripts/ (not a package) and runs gb10_prep.set_alloc_conf()
    at import time. That call only sets an environment variable (no CUDA allocation),
    so it is safe to import on a CPU-only / GPU-busy box.
    """
    path = REPO_ROOT / "scripts" / "train_nvfp4_lora.py"
    spec = importlib.util.spec_from_file_location("train_nvfp4_lora", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def train_mod():
    return _load_train_module()


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(autouse=True)
def _no_cuda_allocation():
    """Fail any test that allocates CUDA memory in this process.

    A GPU training run may hold the device while this suite runs, so the suite must
    stay strictly on CPU. torch.cuda.memory_allocated() reports only THIS process's
    allocations (it does not allocate and is unaffected by other processes), so it is
    a safe, cheap tripwire. On a CI runner without a GPU this is a no-op.
    """
    if torch is None or not torch.cuda.is_available():
        yield
        return
    before = torch.cuda.memory_allocated()
    yield
    after = torch.cuda.memory_allocated()
    assert after <= before, (
        f"test allocated {after - before} bytes of CUDA memory; the CPU suite must "
        f"never touch the GPU"
    )
