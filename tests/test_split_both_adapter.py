"""Contract test for scripts/split_both_adapter.py (CPU-only, no model).

A `--train-target both` run saves ONE unified adapter (tower + LLM LoRA keys). The splitter
partitions it by the scopes recorded in the `both` config block into two standard
sub-adapters (tower/ for merge_vision_lora, llm/ for merge_lora_into_nvfp4), refusing loudly
if either scope is empty (a half-trained or mis-scoped adapter).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

_SPLIT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "split_both_adapter.py"


def _load():
    spec = importlib.util.spec_from_file_location("split_both_adapter", _SPLIT_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S = _load()

_TOWER = [
    "base_model.model.vision_model.blk.qkv.lora_A.weight",
    "base_model.model.vision_model.blk.qkv.lora_B.weight",
    "base_model.model.mlp1.1.lora_A.weight",
    "base_model.model.mlp1.1.lora_B.weight",
]
_LLM = [
    "base_model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
    "base_model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight",
]


def _make_both_adapter(d: Path, tower_keys, llm_keys, *, both_block=True):
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: torch.zeros(2, 2) for k in (list(tower_keys) + list(llm_keys))},
              str(d / "adapter_model.safetensors"))
    cfg = {
        "base_model_name_or_path": "/base", "peft_type": "LORA",
        "r": 16, "lora_alpha": 32, "lora_dropout": 0.0, "target_modules": ["q_proj"],
    }
    if both_block:
        cfg["train_target"] = "both"
        cfg["both"] = {
            "train_target": "both",
            "text_target_modules": ["q_proj", "k_proj", "v_proj"],
            "vision_target_modules": ["qkv"],
            "text_peft_scope": r"^language_model\.",
            "vision_peft_scope": r"^vision_model\.",
            "projector_scopes": [r"^mlp1\."],
            "include_projector": True,
        }
    (d / "adapter_config.json").write_text(json.dumps(cfg))
    return d


# --------------------------------------------------------------------------- pure functions

def test_module_path_of():
    assert S.module_path_of("base_model.model.vision_model.blk.qkv.lora_A.weight") == \
        "vision_model.blk.qkv"
    # A wrapper-prefixed (mistral3-style) tower key keeps its leading `model.`.
    assert S.module_path_of("base_model.model.model.vision_tower.x.q_proj.lora_B.weight") == \
        "model.vision_tower.x.q_proj"
    with pytest.raises(ValueError):
        S.module_path_of("vision_model.blk.qkv.lora_A.weight")        # no adapter prefix
    with pytest.raises(ValueError):
        S.module_path_of("base_model.model.foo.experts.gate_up.lora_A")  # not .weight shape


def test_classify_keys_partitions_by_scope():
    tower, llm = S.classify_keys(_TOWER + _LLM, r"^vision_model\.", [r"^mlp1\."])
    assert set(tower) == set(_TOWER)          # tower Linear + Sequential projector
    assert set(llm) == set(_LLM)
    # disjoint + complete
    assert set(tower) & set(llm) == set()
    assert set(tower) | set(llm) == set(_TOWER + _LLM)


# --------------------------------------------------------------------------- end to end

def test_split_roundtrip(tmp_path):
    adir = _make_both_adapter(tmp_path / "both", _TOWER, _LLM)
    out = tmp_path / "out"
    summary = S.split_both_adapter(adir, out)
    assert summary["tower_keys"] == 4 and summary["llm_keys"] == 2

    from safetensors import safe_open
    with safe_open(out / "tower" / "adapter_model.safetensors", framework="pt") as sf:
        tower_saved = set(sf.keys())
    with safe_open(out / "llm" / "adapter_model.safetensors", framework="pt") as sf:
        llm_saved = set(sf.keys())
    assert tower_saved == set(_TOWER)
    assert llm_saved == set(_LLM)
    assert tower_saved & llm_saved == set()

    # Sub-configs are merge-tool-ready: r + lora_alpha + the right-half target_modules.
    tcfg = json.loads((out / "tower" / "adapter_config.json").read_text())
    lcfg = json.loads((out / "llm" / "adapter_config.json").read_text())
    assert tcfg["r"] == 16 and tcfg["lora_alpha"] == 32
    assert tcfg["target_modules"] == ["qkv"] and tcfg["train_target"] == "vision"
    assert lcfg["target_modules"] == ["q_proj", "k_proj", "v_proj"] and lcfg["train_target"] == "text"


def test_split_mistral3_shaped_adapter(tmp_path):
    """A non-nemotron layout splits correctly: mistral3 towers under model.vision_tower. /
    model.multi_modal_projector., LLM under model.language_model., with the PEFT-doubled
    model.model. wrapper prefix. Exercises the scope match on a second family shape."""
    tower = [
        "base_model.model.model.vision_tower.transformer.layers.0.attention.q_proj.lora_A.weight",
        "base_model.model.model.vision_tower.transformer.layers.0.attention.q_proj.lora_B.weight",
        "base_model.model.model.multi_modal_projector.linear_1.lora_A.weight",
        "base_model.model.model.multi_modal_projector.linear_1.lora_B.weight",
    ]
    llm = [
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight",
    ]
    d = tmp_path / "m3both"
    d.mkdir(parents=True)
    save_file({k: torch.zeros(2, 2) for k in tower + llm}, str(d / "adapter_model.safetensors"))
    cfg = {
        "base_model_name_or_path": "/base", "r": 16, "lora_alpha": 32, "train_target": "both",
        "both": {
            "text_target_modules": ["q_proj"], "vision_target_modules": ["q_proj"],
            "text_peft_scope": r"^model\.language_model\.",
            "vision_peft_scope": r"^model\.vision_tower\.|^model\.multi_modal_projector\.",
            "projector_scopes": [r"^model\.multi_modal_projector\."],
        },
    }
    (d / "adapter_config.json").write_text(json.dumps(cfg))
    s = S.split_both_adapter(d, tmp_path / "out")
    assert s["tower_keys"] == 4 and s["llm_keys"] == 2
    # NON-identity vs on-disk (mistral's on-disk LLM prefix is language_model.model.); the
    # printed --prefix-pair DISK side needs adjusting -- but mistral's LLM is NVFP4 anyway, so
    # its LLM half merges via merge_lora_into_ct_nvfp4 (family-aware), not the bf16 --prefix-pair.
    assert s["llm_prefix"] == "model.language_model."


def test_split_refuses_non_both_adapter(tmp_path):
    adir = _make_both_adapter(tmp_path / "plain", _TOWER, _LLM, both_block=False)
    with pytest.raises(SystemExit) as e:
        S.split_both_adapter(adir, tmp_path / "out")
    assert "not a `--train-target both`" in str(e.value)


def test_scope_to_prefix():
    assert S.scope_to_prefix(r"^language_model\.") == "language_model."
    assert S.scope_to_prefix(r"^model\.language_model\.") == "model.language_model."


def test_summary_carries_llm_prefix(tmp_path):
    adir = _make_both_adapter(tmp_path / "both", _TOWER, _LLM)
    s = S.split_both_adapter(adir, tmp_path / "out")
    assert s["llm_prefix"] == "language_model."       # feeds merge_vision_lora --prefix-pair


def test_merge_vision_lora_prefix_pair_maps_llm_key_identity():
    """The LLM half merges via merge_vision_lora with an identity --prefix-pair; verify the
    tool maps an LLM adapter key to its on-disk base weight key under that pair."""
    mvl_path = Path(__file__).resolve().parent.parent / "scripts" / "merge_vision_lora.py"
    spec = importlib.util.spec_from_file_location("merge_vision_lora", mvl_path)
    mvl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mvl)
    pairs = [("language_model.", "language_model.")]
    k = "base_model.model.language_model.backbone.layers.3.mixer.q_proj.lora_A.weight"
    assert mvl.adapter_key_to_base_key(k, pairs) == \
        "language_model.backbone.layers.3.mixer.q_proj.weight"


def test_split_refuses_when_a_scope_is_empty(tmp_path):
    # Only LLM keys -> zero tower keys -> refuse (R6: both scopes must be present).
    a1 = _make_both_adapter(tmp_path / "llmonly", [], _LLM)
    with pytest.raises(SystemExit) as e:
        S.split_both_adapter(a1, tmp_path / "o1")
    assert "ZERO tower" in str(e.value)
    # Only tower keys -> zero LLM keys -> refuse (it is a vision adapter, not a both-adapter).
    a2 = _make_both_adapter(tmp_path / "toweronly", _TOWER, [])
    with pytest.raises(SystemExit) as e:
        S.split_both_adapter(a2, tmp_path / "o2")
    assert "ZERO LLM" in str(e.value)
