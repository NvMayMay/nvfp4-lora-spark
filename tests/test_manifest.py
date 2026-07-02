"""CPU tests for the adapter provenance manifest + serve-time compat check.

Reuses the synthetic base/adapter builders from test_serve_contract (no weights, no
GPU). Verifies the manifest fingerprints the pair, and that check_compat REFUSES a
base whose fingerprint differs from the one the adapter was trained against (the
wrong-base serve = silent-no-op / garbage class).
"""
from __future__ import annotations

import json

from nybbloris.manifest import (MANIFEST_NAME, base_fingerprint, build_manifest,
                                check_compat, write_manifest)
from test_serve_contract import ATTN, _build_adapter, _build_base, _mods


def _base(d, *, quant="nvfp4", arch="Qwen3MoeForCausalLM", model_type="qwen3_moe"):
    _build_base(d, arch=arch, model_type=model_type, quant_method="modelopt",
                layout="flat", targets=[(r, quant) for r in ATTN])


def test_build_manifest_has_core_fields(tmp_path):
    base, ad = tmp_path / "base", tmp_path / "ad"
    _base(base)
    _build_adapter(ad, _mods("flat", ATTN), r=8, alpha=16)
    m = build_manifest(base, ad)
    assert m["manifest_version"] >= 1
    assert m["base"]["arch"] == "Qwen3MoeForCausalLM"
    assert m["base"]["quant_method"] == "modelopt"
    assert m["base"]["weight_index_sha256"]  # non-empty hash
    assert m["adapter"]["r"] == 8 and m["adapter"]["lora_alpha"] == 16
    assert m["adapter"]["lora_tensor_count"] == 2 * len(ATTN)  # A+B per target
    assert "package_versions" in m["provenance"]


def test_write_manifest_roundtrips(tmp_path):
    base, ad = tmp_path / "base", tmp_path / "ad"
    _base(base)
    _build_adapter(ad, _mods("flat", ATTN))
    dest = write_manifest(base, ad)
    assert dest.name == MANIFEST_NAME
    loaded = json.loads(dest.read_text())
    assert loaded["base"]["weight_index_sha256"] == base_fingerprint(base)["weight_index_sha256"]


def test_check_compat_same_base_ok(tmp_path):
    base, ad = tmp_path / "base", tmp_path / "ad"
    _base(base)
    _build_adapter(ad, _mods("flat", ATTN))
    m = build_manifest(base, ad)
    ok, reasons = check_compat(m, base)
    assert ok and reasons == []


def test_check_compat_different_weights_refused(tmp_path):
    base, ad, other = tmp_path / "base", tmp_path / "ad", tmp_path / "other"
    _base(base)
    _build_adapter(ad, _mods("flat", ATTN))
    m = build_manifest(base, ad)
    # A different base with a different weight index (extra key) -> index hash differs.
    _build_base(other, arch="Qwen3MoeForCausalLM", model_type="qwen3_moe",
                quant_method="modelopt", layout="flat",
                targets=[(r, "nvfp4") for r in ATTN], extra_keys=["lm_head.weight"])
    ok, reasons = check_compat(m, other)
    assert not ok
    assert any("weight_index_sha256" in r for r in reasons)


def test_check_compat_same_index_different_shard_bytes_refused(tmp_path):
    """Same config + index (same tensor layout / shard filenames) but different weight
    BYTES (a re-downloaded/overwritten/re-quantized revision) must be REFUSED -- the
    case a matching index hash alone would false-pass."""
    base, ad, other = tmp_path / "base", tmp_path / "ad", tmp_path / "other"
    _base(base)
    (base / "model-00001-of-00001.safetensors").write_bytes(b"\x00" * 4096)
    _build_adapter(ad, _mods("flat", ATTN))
    m = build_manifest(base, ad)
    # `other` has identical config.json + index.json (same _base), so the index hash
    # matches -- but a differently-sized shard.
    _base(other)
    (other / "model-00001-of-00001.safetensors").write_bytes(b"\x00" * 8192)
    ok, reasons = check_compat(m, other)
    assert not ok
    assert any("shard_bytes" in r for r in reasons)


def test_check_compat_different_arch_refused(tmp_path):
    base, ad, other = tmp_path / "base", tmp_path / "ad", tmp_path / "other"
    _base(base)
    _build_adapter(ad, _mods("flat", ATTN))
    m = build_manifest(base, ad)
    # Same layout keys but a different architecture/model_type -> structural mismatch.
    _build_base(other, arch="LlamaForCausalLM", model_type="llama",
                quant_method="modelopt", layout="flat",
                targets=[(r, "nvfp4") for r in ATTN])
    ok, reasons = check_compat(m, other)
    assert not ok
    assert any("arch mismatch" in r or "model_type mismatch" in r for r in reasons)
