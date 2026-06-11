"""Regression test: the final adapter save must NOT wipe the output root.

The trainer's end-of-run save targets --output-dir itself, which also holds
best/, checkpoint_step_*/ and metrics.jsonl. An earlier implementation did
rmtree(dest)+rename(tmp, dest), which deleted all of them (this destroyed the
best-by-val-loss adapter of the Mistral-Small-4-119B v3.5 run). The save must
move files into the destination individually and leave sibling content alone.
"""
from __future__ import annotations

import json
from pathlib import Path


class _FakeNativeModel:
    """Model with no NVFP4LoRALinear modules: native save writes an empty
    (but valid) safetensors file, which is all this test needs."""

    def named_modules(self):
        return iter(())


class _FakeTokenizer:
    def save_pretrained(self, path):
        (Path(path) / "tokenizer_config.json").write_text("{}")


def _populate_output_root(out: Path) -> dict:
    best = out / "best"
    best.mkdir(parents=True)
    (best / "adapter_model.safetensors").write_bytes(b"best-weights")
    ckpt = out / "checkpoint_step_2"
    ckpt.mkdir()
    (ckpt / "train_state.pt").write_bytes(b"state")
    metrics = out / "metrics.jsonl"
    metrics.write_text(json.dumps({"event": "train_step", "step": 1}) + "\n")
    return {
        "best_weights": best / "adapter_model.safetensors",
        "train_state": ckpt / "train_state.pt",
        "metrics": metrics,
    }


def test_final_save_to_output_root_preserves_best_checkpoints_metrics(
        train_mod, tmp_path):
    out = tmp_path / "run_output"
    survivors = _populate_output_root(out)

    events = []
    train_mod._save_adapter_atomic(
        _FakeNativeModel(), _FakeTokenizer(), out,
        lambda event, **kw: events.append(event),
        lora_mode="native", base_model_dir="/nonexistent/base",
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        target_suffixes=["q_proj"])

    # The adapter landed in the root...
    assert (out / "adapter_model.safetensors").is_file()
    assert (out / "adapter_config.json").is_file()
    assert (out / "tokenizer_config.json").is_file()
    # ...and every pre-existing sibling survived with its content intact.
    assert survivors["best_weights"].read_bytes() == b"best-weights"
    assert survivors["train_state"].read_bytes() == b"state"
    assert "train_step" in survivors["metrics"].read_text()
    # No leftover tmp dir.
    assert not (out.parent / (out.name + ".tmp")).exists()


def test_save_to_subdir_overwrites_previous_adapter_in_place(train_mod, tmp_path):
    out = tmp_path / "run_output"
    best = out / "best"
    best.mkdir(parents=True)
    (best / "adapter_model.safetensors").write_bytes(b"old")
    (best / "stale_extra_file").write_bytes(b"keep-or-not")

    train_mod._save_adapter_atomic(
        _FakeNativeModel(), _FakeTokenizer(), best,
        lambda event, **kw: None,
        lora_mode="native", base_model_dir="/nonexistent/base",
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        target_suffixes=["q_proj"])

    # New save replaced the adapter files...
    assert (best / "adapter_model.safetensors").read_bytes() != b"old"
    assert (best / "adapter_config.json").is_file()
    # ...and per-file replacement means unrelated files are left alone.
    assert (best / "stale_extra_file").read_bytes() == b"keep-or-not"
