"""Pin the FP8-module signal to ONE definition.

`is_fp8_module` now has a single implementation in `nvfp4_lora.adapter_keys` (torch-free,
so the pre-flight surfaces classify FP8 without importing the loader). `nvfp4_lora.loader`
re-exports it, but still carries `classify_module_storage`, whose richer taxonomy has its
own fp8 branch. If that branch ever drifts from the re-exported predicate, the serve
pre-flight (nybbloris.plan) and the shard loader would label the same module differently --
the silent-desync this test exists to catch.

We assert, across every storage class, that:
  loader.is_fp8_module(keys, p)  ==  (classify_module_storage(keys, p) == "fp8")
and that the loader's re-exported symbol IS adapter_keys' object (one implementation).
"""
from __future__ import annotations

import pytest

from nvfp4_lora import adapter_keys
from nvfp4_lora import loader


def _keys(prefix, *suffixes):
    return {f"{prefix}{s}" for s in suffixes}


P = "model.layers.0.mlp.gate_proj"

# (label, key-suffixes present under P, expected classify result)
CASES = [
    ("fp8", (".weight", ".weight_scale"), "fp8"),
    ("nvfp4_modelopt", (".weight", ".weight_scale", ".weight_scale_2"), "nvfp4_modelopt"),
    ("nvfp4_ct", (".weight_packed", ".weight_scale"), "nvfp4_ct"),
    ("bf16", (".weight",), "bf16"),
    ("absent", (".bias",), "absent"),
    # packed wins even if the modelopt scales are also somehow present.
    ("ct_precedence", (".weight_packed", ".weight", ".weight_scale", ".weight_scale_2"), "nvfp4_ct"),
]


@pytest.mark.parametrize("label,suffixes,expected", CASES, ids=[c[0] for c in CASES])
def test_is_fp8_module_agrees_with_classify(label, suffixes, expected):
    keys = _keys(P, *suffixes)
    assert loader.classify_module_storage(keys, P) == expected
    # The one predicate must match classify's fp8 branch exactly.
    assert adapter_keys.is_fp8_module(keys, P) == (expected == "fp8")
    assert loader.is_fp8_module(keys, P) == (expected == "fp8")


def test_loader_reexports_adapter_keys_implementation():
    # Not two copies: the loader symbol IS the adapter_keys object.
    assert loader.is_fp8_module is adapter_keys.is_fp8_module
