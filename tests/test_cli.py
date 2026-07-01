"""nybbloris CLI surface: documented exit codes, --json stdout, doctor (CPU-only).

Reuses the synthetic base/adapter builders from test_serve_contract so the CLI is
exercised end-to-end (argparse -> serve_plan -> verdict -> exit code) with no weights.
"""
from __future__ import annotations

import json

import pytest

from nybbloris.cli import VERDICT_EXIT, _derive_probe_prompt, main
from test_serve_contract import ATTN, _build_adapter, _build_base, _mods


def _run(argv):
    with pytest.raises(SystemExit) as e:
        main(argv)
    return e.value.code


def _flat_nvfp4_base(d):
    _build_base(d, arch="Qwen3MoeForCausalLM", model_type="qwen3_moe", quant_method="modelopt",
                layout="flat", targets=[(r, "nvfp4") for r in ATTN])


def _wrapped_nvfp4_base(d):
    _build_base(d, arch="Qwen3_5MoeForConditionalGeneration", model_type="qwen3_5_moe",
                quant_method="modelopt", layout="wrapped", targets=[(r, "nvfp4") for r in ATTN])


def test_inspect_pass_exit_0(tmp_path):
    _flat_nvfp4_base(tmp_path / "base")
    _build_adapter(tmp_path / "ad", _mods("flat", ATTN))
    assert _run(["inspect", "--base-model-dir", str(tmp_path / "base"),
                 "--adapter-dir", str(tmp_path / "ad")]) == VERDICT_EXIT["PASS"] == 0


def test_inspect_noop_exit_3(tmp_path):
    # wrapped base + flat adapter = the silent no-op -> distinct nonzero code.
    _wrapped_nvfp4_base(tmp_path / "base")
    _build_adapter(tmp_path / "ad", _mods("flat", ATTN))
    assert _run(["inspect", "--base-model-dir", str(tmp_path / "base"),
                 "--adapter-dir", str(tmp_path / "ad")]) == 3


def test_inspect_fail_exit_1(tmp_path):
    _flat_nvfp4_base(tmp_path / "base")
    _build_adapter(tmp_path / "ad", ["model.layers.0.self_attn.bogus_proj"])
    assert _run(["inspect", "--base-model-dir", str(tmp_path / "base"),
                 "--adapter-dir", str(tmp_path / "ad")]) == 1


def test_inspect_json_stdout_is_parseable(tmp_path, capsys):
    _flat_nvfp4_base(tmp_path / "base")
    _build_adapter(tmp_path / "ad", _mods("flat", ATTN))
    _run(["inspect", "--base-model-dir", str(tmp_path / "base"),
          "--adapter-dir", str(tmp_path / "ad"), "--json"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["verdict"] == "PASS" and plan["targets"]["live"] == len(ATTN)


def test_derive_probe_prompt_from_val_row(tmp_path):
    # --verify's apply-check probe prompt is stitched from the first val row's messages.
    vf = tmp_path / "val.jsonl"
    vf.write_text(json.dumps({"messages": [
        {"role": "user", "content": "count the singers"},
        {"role": "assistant", "content": "SELECT COUNT(*) FROM singer;"}]}) + "\n")
    p = _derive_probe_prompt(str(vf))
    assert "count the singers" in p and "SELECT COUNT(*)" in p


def test_derive_probe_prompt_missing_file_is_none(tmp_path):
    # Missing/unreadable val file falls back to None (checker uses its default probe).
    assert _derive_probe_prompt(str(tmp_path / "nope.jsonl")) is None


def test_doctor_runs_and_reports(capsys):
    code = _run(["doctor"])
    out = capsys.readouterr().out
    assert "nybbloris doctor" in out and "doctor:" in out
    assert code in (0, 1)
