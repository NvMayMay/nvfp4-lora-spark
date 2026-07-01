"""CPU regression for the runtime-apply verdict (scripts/serve_apply_check.py).

Encodes the Qwen3.5-122B episode: a LOADED-but-NO-OP adapter produces IDENTICAL
prompt-echo logprobs (the wrapped-model rekey bug), while a correctly applied
adapter moves them. A greedy-text check passed this falsely; the logprob delta
did not. This test guards the verdict math (no server / no GPU needed).
"""
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = REPO_ROOT / "scripts" / "serve_apply_check.py"
    spec = importlib.util.spec_from_file_location("serve_apply_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sac = _load()


def test_identical_logprobs_is_noop():
    # The exact Qwen3.5 no-op signature: byte-identical base vs adapter logprobs.
    base = [-1.2, -0.3, -5.5, -2.1, -0.001]
    v = sac.apply_verdict(base, list(base))
    assert v["applies"] is False
    assert v["max_abs_delta"] == 0.0
    assert v["sum_delta"] == 0.0
    assert v["n"] == len(base)


def test_moved_logprobs_applies():
    # The v3 fix: adapter shifts the prompt logprobs well past threshold.
    base = [-1.2, -0.3, -5.5, -2.1]
    adapter = [-1.9, -0.3, -8.2, -2.1]  # max |delta| = 2.7, like the measured fix
    v = sac.apply_verdict(base, adapter)
    assert v["applies"] is True
    assert v["max_abs_delta"] == pytest.approx(2.7, abs=1e-9)
    assert v["sum_delta"] == pytest.approx(-3.4, abs=1e-9)


def test_subthreshold_difference_is_noop():
    # Tiny numeric jitter must NOT read as "applied".
    base = [-1.0, -2.0, -3.0]
    adapter = [-1.0, -2.0, -3.00005]  # 5e-5 < default 1e-4 threshold
    v = sac.apply_verdict(base, adapter)
    assert v["applies"] is False


def test_threshold_is_configurable():
    base = [-1.0, -2.0]
    adapter = [-1.0, -2.01]  # delta 0.01
    assert sac.apply_verdict(base, adapter, threshold=1e-4)["applies"] is True
    assert sac.apply_verdict(base, adapter, threshold=1.0)["applies"] is False


def test_empty_is_noop_not_crash():
    v = sac.apply_verdict([], [])
    assert v["applies"] is False
    assert v["n"] == 0


def test_unequal_lengths_use_common_prefix():
    base = [-1.0, -2.0, -3.0, -4.0]
    adapter = [-1.0, -9.0]  # only 2 comparable; delta on index 1 = 7.0
    v = sac.apply_verdict(base, adapter)
    assert v["n"] == 2
    assert v["max_abs_delta"] == pytest.approx(7.0, abs=1e-9)
