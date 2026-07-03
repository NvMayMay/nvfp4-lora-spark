"""nybbloris CLI surface: documented exit codes, --json stdout, doctor (CPU-only).

Reuses the synthetic base/adapter builders from test_serve_contract so the CLI is
exercised end-to-end (argparse -> serve_plan -> verdict -> exit code) with no weights.
"""
from __future__ import annotations

import json

import pytest

from nybbloris.cli import VERDICT_EXIT, _derive_probe_prompt, _require_script, main
from test_serve_contract import ATTN, _build_adapter, _build_base, _mods


def test_require_script_missing_is_clean_false(tmp_path, capsys):
    # A missing shell-out helper must fail cleanly (False + a clear message), not with an
    # opaque `python: can't open file` when run from a bare wheel.
    assert _require_script(tmp_path / "nope.py") is False
    assert "not found" in capsys.readouterr().out
    present = tmp_path / "there.py"
    present.write_text("x = 1\n")
    assert _require_script(present) is True


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


# --------------------------------------------------------------------------------------
# serve pre-flight (item 4): backend-aware routed serve + manifest gate.
# --------------------------------------------------------------------------------------
from test_serve_contract import ROUTED, _mods as _m  # noqa: E402


def test_serve_routed_forces_emulation_backend_not_abort(tmp_path, capsys):
    """A routed-expert adapter is backend-gated, not merge-only: serve auto-adds
    --moe-backend emulation and reaches --dry-run launch (does NOT hard-abort)."""
    _build_base(tmp_path / "base", arch="Qwen3MoeForCausalLM", model_type="qwen3_moe",
                quant_method="modelopt", layout="flat", targets=[(r, "nvfp4") for r in ROUTED])
    _build_adapter(tmp_path / "ad", _m("flat", ROUTED))
    code = _run(["serve", "--base-model-dir", str(tmp_path / "base"),
                 "--adapter", f"exp={tmp_path / 'ad'}", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0  # reached dry-run launch, not a REFUSING abort
    assert "--moe-backend emulation" in out
    assert "backend-gated" in out


def test_serve_routed_explicit_blocked_backend_refused(tmp_path, capsys):
    """A routed adapter + explicit --moe-backend cutlass (supports_lora=False) would
    silently no-op -> REFUSE, rather than let the explicit backend win."""
    _build_base(tmp_path / "base", arch="Qwen3MoeForCausalLM", model_type="qwen3_moe",
                quant_method="modelopt", layout="flat", targets=[(r, "nvfp4") for r in ROUTED])
    _build_adapter(tmp_path / "ad", _m("flat", ROUTED))
    code = _run(["serve", "--base-model-dir", str(tmp_path / "base"),
                 "--adapter", f"exp={tmp_path / 'ad'}", "--moe-backend", "cutlass", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 1
    assert "REFUSING" in out and "cutlass" in out


def test_serve_routed_unsafe_backend_override_allowed(tmp_path, capsys):
    """--unsafe-moe-backend opts into the base-only (no-adapter-effect) serve."""
    _build_base(tmp_path / "base", arch="Qwen3MoeForCausalLM", model_type="qwen3_moe",
                quant_method="modelopt", layout="flat", targets=[(r, "nvfp4") for r in ROUTED])
    _build_adapter(tmp_path / "ad", _m("flat", ROUTED))
    code = _run(["serve", "--base-model-dir", str(tmp_path / "base"),
                 "--adapter", f"exp={tmp_path / 'ad'}", "--moe-backend", "cutlass",
                 "--unsafe-moe-backend", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "--moe-backend cutlass" in out


def test_serve_manifest_mismatch_refuses(tmp_path, capsys):
    """A manifest next to the adapter whose base fingerprint mismatches -> REFUSE."""
    from nybbloris.manifest import write_manifest
    # Train-time base (wrapped) vs a DIFFERENT serve-time base (flat) => mismatch.
    _wrapped_nvfp4_base(tmp_path / "train_base")
    _flat_nvfp4_base(tmp_path / "serve_base")
    _build_adapter(tmp_path / "ad", _mods("serve", ATTN))  # binds on wrapped
    write_manifest(tmp_path / "train_base", tmp_path / "ad")
    code = _run(["serve", "--base-model-dir", str(tmp_path / "serve_base"),
                 "--adapter", f"a={tmp_path / 'ad'}", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 1
    assert "manifest base fingerprint does not match" in out


def test_serve_wrapped_base_adds_language_model_only(tmp_path, capsys):
    """A vision-wrapped base (*ForConditionalGeneration) must serve text-only, else
    vLLM loads the vision tower. The generated command adds --language-model-only."""
    _wrapped_nvfp4_base(tmp_path / "base")
    _build_adapter(tmp_path / "ad", _mods("serve", ATTN))  # binds on the wrapped base
    code = _run(["serve", "--base-model-dir", str(tmp_path / "base"),
                 "--adapter", f"a={tmp_path / 'ad'}", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "--language-model-only" in out


def test_serve_flat_base_no_language_model_only(tmp_path, capsys):
    """A plain causal-LM base must NOT get --language-model-only."""
    _flat_nvfp4_base(tmp_path / "base")
    _build_adapter(tmp_path / "ad", _mods("flat", ATTN))
    _run(["serve", "--base-model-dir", str(tmp_path / "base"),
          "--adapter", f"a={tmp_path / 'ad'}", "--dry-run"])
    assert "--language-model-only" not in capsys.readouterr().out


def test_serve_manifest_match_allows(tmp_path, capsys):
    """A manifest that matches the serve base does not block the pre-flight."""
    from nybbloris.manifest import write_manifest
    _flat_nvfp4_base(tmp_path / "base")
    _build_adapter(tmp_path / "ad", _mods("flat", ATTN))
    write_manifest(tmp_path / "base", tmp_path / "ad")
    code = _run(["serve", "--base-model-dir", str(tmp_path / "base"),
                 "--adapter", f"a={tmp_path / 'ad'}", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "compat OK" in out
