from __future__ import annotations

from types import SimpleNamespace


def test_build_run_meta_hashes_files_and_tolerates_missing_val(train_mod, tmp_path):
    train_file = tmp_path / "train.jsonl"
    train_file.write_text('{"messages":[]}\n')
    args = SimpleNamespace(
        model_dir="/models/base",
        train_file=str(train_file),
        val_file=str(tmp_path / "missing.jsonl"),
        output_dir=str(tmp_path / "out"),
        dry_run=False,
        resume_from=None,
    )
    coverage = {"inventory": {"q_proj": {"counts": {"nvfp4": 1}}}}

    meta = train_mod.build_run_meta(args, coverage)

    assert meta["args"]["model_dir"] == "/models/base"
    assert meta["coverage"] == coverage
    assert len(meta["files"]["train_file"]["sha256"]) == 64
    assert meta["files"]["val_file"]["sha256"] is None
    assert set(meta["versions"]) == {"peft", "torch", "transformers"}
    assert "git_sha" in meta


def test_build_metrics_row_has_cpu_safe_fields(train_mod, monkeypatch):
    monkeypatch.setattr(train_mod.torch.cuda, "is_available", lambda: False)

    row = train_mod.build_metrics_row(
        step=2,
        total_updates=10,
        window_supervised_tokens=128,
        wall_elapsed=20.0,
        recent_upd_s=5.0,
        loss_window_mean=1.23456,
    )

    assert row["window_supervised_tokens"] == 128
    assert row["supervised_tokens_s"] == 25.6
    assert row["updates_s"] == 0.2
    assert row["loss_window_mean"] == 1.2346
    assert row["eta_s"] == 80.0
    assert row["cuda_allocated_gb"] is None
    assert row["cuda_reserved_gb"] is None
    assert row["cuda_free_gb"] is None
    assert "host_mem_available_gb" in row
