"""Checkpoint inspector: reports layout + the exact training verdict, CPU-only.

build_report reads only config.json + model.safetensors.index.json, so the
trimmed fixtures exercise the same code paths as a 100B checkpoint.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def inspector():
    path = REPO_ROOT / "scripts" / "inspect_nvfp4_checkpoint.py"
    spec = importlib.util.spec_from_file_location("inspect_nvfp4_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_qwen_report(inspector, fixtures_dir):
    r = inspector.build_report(fixtures_dir / "qwen3_5_moe",
                               ["q_proj", "k_proj", "v_proj", "o_proj"], deep=False)
    assert r["model_type"] == "qwen3_5_moe"
    assert r["family_supported"] is True
    assert r["storage_census"]["nvfp4_ct"] == 7
    assert r["moe"] is not None and r["moe"]["per_expert_keys"] is True
    assert r["target_verdict"]["ok"] is True
    assert r["target_verdict"]["mode"] == "native"


def test_mistral_report_peft_verdict(inspector, fixtures_dir):
    r = inspector.build_report(fixtures_dir / "mistral3",
                               ["q_b_proj", "kv_b_proj", "o_proj"], deep=False)
    assert r["family_supported"] is True
    assert r["quant_config"]["quant_method"] == "compressed-tensors"
    assert r["target_verdict"]["ok"] is True
    assert r["target_verdict"]["mode"] == "peft"


def test_partial_quant_rejection_is_reported_not_raised(inspector, fixtures_dir):
    r = inspector.build_report(fixtures_dir / "partial_quant", ["o_proj"], deep=False)
    # no config.json in this fixture: unknown family, still inspectable
    assert r["family_supported"] is False
    tv = r["target_verdict"]
    assert tv["ok"] is False
    assert "PARTIALLY quantized" in tv["reason"]
    # the suffix table flags the mixed suffix
    assert set(r["suffixes"]["o_proj"]["counts"]) == {"nvfp4_ct", "bf16"}


def test_fp8_census(inspector, fixtures_dir):
    r = inspector.build_report(fixtures_dir / "fp8_demoted", None, deep=False)
    assert r["storage_census"]["fp8"] == 2
    assert r["storage_census"]["nvfp4_modelopt"] == 1
    assert "target_verdict" not in r


def test_human_output_smoke(inspector, fixtures_dir, capsys):
    r = inspector.build_report(fixtures_dir / "qwen3_5_moe", ["q_proj"], deep=False)
    inspector.print_human(r)
    out = capsys.readouterr().out
    assert "storage census" in out
    assert "target verdict: OK" in out
