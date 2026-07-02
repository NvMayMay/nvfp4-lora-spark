"""Serve-binding contract matrix: nybbloris.plan.serve_plan verdict truth table.

`serve_plan` is the pre-flight gate that decides whether a (NVFP4 base + LoRA
adapter) pair will actually bind + serve its deltas in vLLM. It reads ONLY two
files -- config.json and model.safetensors.index.json -- plus the adapter header,
so the full matrix runs CPU-only with synthetic fixtures (no weights, no GPU).

This suite is the regression lock for the inverse-binding bug (MEASURED §7h): the
contract once resolved adapter keys against the on-disk index, which is the INVERSE
of vLLM's runtime module tree for multimodal *ForConditionalGeneration wrappers --
it blessed a flat adapter that silently no-ops and failed the re-keyed one that
binds. The `wrapped` cases below pin the corrected behavior.

Each case is a (base, adapter) -> expected-verdict row. The base is a flat causal-LM
(`model.layers.*`) or a multimodal wrapper (on-disk `model.language_model.layers.*`,
which vLLM exposes as `language_model.model.layers.*`); the adapter targets are
either flat or already re-keyed to the serve layout.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from nybbloris.plan import adapter_modules, lm_head_status, render_plan, serve_plan

# Relative module paths by kind (the _kind() classifier keys off these substrings).
ATTN = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
SHARED = ["mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj"]
ROUTED = ["mlp.experts.0.gate_proj", "mlp.experts.0.up_proj", "mlp.experts.0.down_proj"]
# Fused-3D MoE LoRA: the adapter targets the whole expert block (no expert index),
# `mlp.experts` (down) + `mlp.experts.base_layer` (gate_up). The base still stores
# experts per-expert, so serve_plan must resolve the fused target against expert 0.
FUSED = ["mlp.experts", "mlp.experts.base_layer"]


def _full(layout: str, rel: str) -> str:
    """Expand a relative module path into a full key for a given on-disk/runtime layout."""
    if layout == "flat":     # causal-LM checkpoint == vLLM runtime tree
        return f"model.layers.0.{rel}"
    if layout == "wrapped":  # multimodal *ForConditionalGeneration on-disk nesting
        return f"model.language_model.layers.0.{rel}"
    if layout == "serve":    # the runtime tree vLLM binds LoRA against (re-keyed adapter)
        return f"language_model.model.layers.0.{rel}"
    raise ValueError(layout)


def _q_keys(module: str, quant: str) -> list[str]:
    """The base-weight key set that makes `module` classify as a given quant type."""
    if quant == "nvfp4":     # ModelOpt: weight + per-block + global scale
        return [f"{module}.weight", f"{module}.weight_scale", f"{module}.weight_scale_2"]
    if quant == "nvfp4_ct":  # compressed-tensors: packed nibbles + global scale
        return [f"{module}.weight_packed", f"{module}.weight_global_scale", f"{module}.weight_scale"]
    if quant == "fp8":       # fp8_e4m3 weight + activation scale
        return [f"{module}.weight", f"{module}.weight_scale", f"{module}.input_scale"]
    if quant == "bf16":      # bare weight
        return [f"{module}.weight"]
    raise ValueError(quant)


def _build_base(base_dir: Path, *, arch, model_type, quant_method, layout, targets, extra_keys=()):
    """Write a synthetic base checkpoint (config.json + index.json) -- no weights."""
    base_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"architectures": [arch], "model_type": model_type}
    if quant_method:
        cfg["quantization_config"] = {"quant_method": quant_method}
    (base_dir / "config.json").write_text(json.dumps(cfg))
    keys = list(extra_keys)
    for rel, quant in targets:
        keys += _q_keys(_full(layout, rel), quant)
    weight_map = {k: "model-00001-of-00001.safetensors" for k in keys}
    (base_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def _build_adapter(adapter_dir: Path, modules, *, r=16, alpha=32):
    """Write a synthetic PEFT adapter: a header-only safetensors + adapter_config.json.

    serve_plan reads module paths from the safetensors header alone (never the tensor
    bytes), so a valid 8-byte length prefix + JSON header is enough -- no data needed.
    """
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(json.dumps(
        {"r": r, "lora_alpha": alpha, "peft_type": "LORA",
         "target_modules": sorted({m.rsplit(".", 1)[-1] for m in modules})}))
    header = {"__metadata__": {"format": "pt"}}
    for m in modules:
        for ab in ("lora_A", "lora_B"):
            header[f"base_model.model.{m}.{ab}.weight"] = {
                "dtype": "F16", "shape": [r, 1], "data_offsets": [0, 0]}
    blob = json.dumps(header).encode("utf-8")
    with open(adapter_dir / "adapter_model.safetensors", "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)


def _mods(layout: str, rels) -> list[str]:
    return [_full(layout, rel) for rel in rels]


# --------------------------------------------------------------------------------------
# The verdict truth table. Each row builds a base + adapter and asserts serve_plan().
# --------------------------------------------------------------------------------------
CASES = [
    # 1. Flat causal-LM, dense NVFP4 attention+shared targets, flat adapter -> binds directly.
    dict(id="pass_flat_dense_nvfp4",
         base=dict(arch="Qwen3MoeForCausalLM", model_type="qwen3_moe", quant_method="modelopt",
                   layout="flat", targets=[(r, "nvfp4") for r in ATTN + SHARED]),
         adapter=_mods("flat", ATTN + SHARED),
         expect=dict(verdict="PASS", wrapped=False, naive=7, resolved=7, live=7,
                     blocked=0, fp8_dense_live=0, rekey="identity")),

    # 2. THE inverse-binding bug: multimodal-wrapped base + flat adapter -> SILENT NO-OP.
    dict(id="noop_wrapped_flat_adapter",
         base=dict(arch="Qwen3_5MoeForConditionalGeneration", model_type="qwen3_5_moe",
                   quant_method="modelopt", layout="wrapped",
                   targets=[(r, "nvfp4") for r in ATTN + SHARED]),
         adapter=_mods("flat", ATTN + SHARED),
         expect=dict(verdict="NO-OP", wrapped=True, naive=0, resolved=7,
                     needs_rekey=True, rekey="language_model")),

    # 3. The fix: the same wrapped base with a re-keyed adapter binds cleanly.
    dict(id="pass_wrapped_rekeyed_adapter",
         base=dict(arch="Qwen3_5MoeForConditionalGeneration", model_type="qwen3_5_moe",
                   quant_method="modelopt", layout="wrapped",
                   targets=[(r, "nvfp4") for r in ATTN + SHARED]),
         adapter=_mods("serve", ATTN + SHARED),
         expect=dict(verdict="PASS", wrapped=True, naive=7, resolved=7, rekey="identity")),

    # 4. Routed-expert targets are blocked (FusedMoE supports_lora=False upstream).
    dict(id="blocked_routed",
         base=dict(arch="Qwen3MoeForCausalLM", model_type="qwen3_moe", quant_method="modelopt",
                   layout="flat", targets=[(r, "nvfp4") for r in ROUTED]),
         adapter=_mods("flat", ROUTED),
         expect=dict(verdict="BLOCKED-ROUTED", blocked=3, live=0)),

    # 5. A target the base lacks under any re-key -> FAIL (not a silent partial).
    dict(id="fail_unresolved",
         base=dict(arch="Qwen3MoeForCausalLM", model_type="qwen3_moe", quant_method="modelopt",
                   layout="flat", targets=[("self_attn.q_proj", "nvfp4")]),
         adapter=["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.bogus_proj"],
         expect=dict(verdict="FAIL", unresolved_contains="bogus_proj")),

    # 6. An adapter with no LoRA tensors -> EMPTY.
    dict(id="empty_adapter",
         base=dict(arch="Qwen3MoeForCausalLM", model_type="qwen3_moe", quant_method="modelopt",
                   layout="flat", targets=[(r, "nvfp4") for r in ATTN]),
         adapter=[],
         expect=dict(verdict="EMPTY", n_targets=0)),

    # 7. Dense FP8 attention is served LIVE (delta is bf16, independent of base quant);
    #    frozen only by the eager TRAIN loader, so it is counted live + reported as info.
    dict(id="pass_dense_fp8_live",
         base=dict(arch="Qwen3_5MoeForConditionalGeneration", model_type="qwen3_5_moe",
                   quant_method="modelopt", layout="wrapped",
                   targets=[(r, "fp8") for r in ATTN]),
         adapter=_mods("serve", ATTN),
         expect=dict(verdict="PASS", live=4, fp8_dense_live=4, blocked=0)),

    # 8. compressed-tensors NVFP4 (the 122B path) -> PASS, serves on the older vLLM line.
    dict(id="pass_compressed_tensors",
         base=dict(arch="Qwen3MoeForCausalLM", model_type="qwen3_moe",
                   quant_method="compressed-tensors", layout="flat",
                   targets=[(r, "nvfp4_ct") for r in ATTN]),
         adapter=_mods("flat", ATTN),
         expect=dict(verdict="PASS", quant_method="compressed-tensors", live=4)),

    # 9. The partial/defensive branch: a layout-invariant target (lm_head) binds naively
    #    while flat layer targets need the re-key -> naive < resolved -> NEEDS-REKEY.
    dict(id="needs_rekey_partial",
         base=dict(arch="Qwen3_5MoeForConditionalGeneration", model_type="qwen3_5_moe",
                   quant_method="modelopt", layout="wrapped",
                   targets=[(r, "nvfp4") for r in ATTN], extra_keys=["lm_head.weight"]),
         adapter=["lm_head"] + _mods("flat", ATTN),
         expect=dict(verdict="NEEDS-REKEY", naive=1, resolved=5,
                     needs_rekey=True, rekey="language_model")),

    # 10. THE Qwen3.5-122B fused-MoE no-op, at the contract level: a fused-3D expert
    #     adapter carrying the flat `model.layers.*` path against a wrapped base binds
    #     NOTHING as shipped. Before the fused-expert resolver these targets read as
    #     UNRESOLVED (FAIL, indistinguishable from the fix); now they resolve only via
    #     the language_model re-key -> naive 0 -> NO-OP, correctly flagging the silent
    #     no-op that only a runtime logprob-delta check had caught.
    dict(id="noop_wrapped_flat_fused_experts",
         base=dict(arch="Qwen3_5MoeForConditionalGeneration", model_type="qwen3_5_moe",
                   quant_method="modelopt", layout="wrapped",
                   targets=[(r, "nvfp4") for r in ROUTED]),
         adapter=_mods("flat", FUSED),
         expect=dict(verdict="NO-OP", wrapped=True, naive=0, resolved=2,
                     needs_rekey=True, rekey="language_model", blocked=2, live=0)),

    # 11. The fix: the same fused-3D expert adapter re-keyed to the wrapped serve path
    #     resolves directly. Routed experts remain backend-gated -> BLOCKED-ROUTED
    #     (live on emulation/marlin), but it now binds (naive == resolved, no re-key).
    dict(id="pass_wrapped_rekeyed_fused_experts",
         base=dict(arch="Qwen3_5MoeForConditionalGeneration", model_type="qwen3_5_moe",
                   quant_method="modelopt", layout="wrapped",
                   targets=[(r, "nvfp4") for r in ROUTED]),
         adapter=_mods("serve", FUSED),
         expect=dict(verdict="BLOCKED-ROUTED", wrapped=True, naive=2, resolved=2,
                     needs_rekey=False, blocked=2, live=0)),
]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_serve_plan_verdicts(tmp_path, case):
    b, exp = case["base"], case["expect"]
    base_dir, adapter_dir = tmp_path / "base", tmp_path / "adapter"
    _build_base(base_dir, arch=b["arch"], model_type=b["model_type"],
                quant_method=b["quant_method"], layout=b["layout"], targets=b["targets"],
                extra_keys=b.get("extra_keys", ()))
    _build_adapter(adapter_dir, case["adapter"])

    plan = serve_plan(base_dir, adapter_dir)

    assert plan["verdict"] == exp["verdict"], render_plan(plan)
    if "wrapped" in exp:
        assert plan["base"]["wrapped"] is exp["wrapped"]
    if "quant_method" in exp:
        assert plan["base"]["quant_method"] == exp["quant_method"]
    if "naive" in exp:
        assert plan["binding"]["naive_resolve"] == exp["naive"]
    if "resolved" in exp:
        assert plan["binding"]["resolved"] == exp["resolved"]
    if "needs_rekey" in exp:
        assert plan["binding"]["needs_rekey"] is exp["needs_rekey"]
    if "rekey" in exp:
        assert plan["binding"]["rekey"] == exp["rekey"]
    if "live" in exp:
        assert plan["targets"]["live"] == exp["live"]
    if "blocked" in exp:
        assert plan["targets"]["blocked_routed"] == exp["blocked"]
    if "fp8_dense_live" in exp:
        assert plan["targets"]["fp8_dense_live"] == exp["fp8_dense_live"]
    if "n_targets" in exp:
        assert plan["adapter"]["n_targets"] == exp["n_targets"]
    if "unresolved_contains" in exp:
        assert any(exp["unresolved_contains"] in u for u in plan["binding"]["unresolved"])


def test_serve_naming_strings(tmp_path):
    """The human-facing serve_naming string distinguishes the two layouts."""
    flat, wrapped = tmp_path / "flat", tmp_path / "wrapped"
    _build_base(flat, arch="Qwen3MoeForCausalLM", model_type="qwen3_moe", quant_method="modelopt",
                layout="flat", targets=[("self_attn.q_proj", "nvfp4")])
    _build_base(wrapped, arch="Qwen3_5MoeForConditionalGeneration", model_type="qwen3_5_moe",
                quant_method="modelopt", layout="wrapped", targets=[("self_attn.q_proj", "nvfp4")])
    _build_adapter(tmp_path / "a", _mods("flat", ["self_attn.q_proj"]))
    assert "causal-LM" in serve_plan(flat, tmp_path / "a")["base"]["serve_naming"]
    assert "multimodal wrapper" in serve_plan(wrapped, tmp_path / "a")["base"]["serve_naming"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_render_plan_never_crashes(tmp_path, case):
    """render_plan() produces a human report carrying the verdict for every case."""
    b = case["base"]
    _build_base(tmp_path / "base", arch=b["arch"], model_type=b["model_type"],
                quant_method=b["quant_method"], layout=b["layout"], targets=b["targets"],
                extra_keys=b.get("extra_keys", ()))
    _build_adapter(tmp_path / "adapter", case["adapter"])
    text = render_plan(serve_plan(tmp_path / "base", tmp_path / "adapter"))
    assert "VERDICT:" in text
    assert case["expect"]["verdict"] in text


# --------------------------------------------------------------------------------------
# lm_head checkpoint-compat pre-flight (vLLM keeps lm_head in bf16).
# --------------------------------------------------------------------------------------
def _write_index(d: Path, keys):
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model-00001-of-00001.safetensors" for k in keys}}))


def test_lm_head_status_quantized(tmp_path):
    _write_index(tmp_path, ["lm_head.weight", "lm_head.weight_scale", "lm_head.weight_scale_2"])
    st = lm_head_status(tmp_path)
    assert st["present"] is True and st["quantized"] is True
    assert "lm_head.weight_scale_2" in st["scale_keys"]
    assert "dequantize" in st["note"]


def test_lm_head_status_bf16(tmp_path):
    _write_index(tmp_path, ["lm_head.weight"])
    st = lm_head_status(tmp_path)
    assert st["present"] is True and st["quantized"] is False
    assert st["scale_keys"] == []


def test_lm_head_status_missing_index(tmp_path):
    st = lm_head_status(tmp_path)
    assert st["present"] is False and st["quantized"] is False
    assert st["note"] == "no index.json"


# --------------------------------------------------------------------------------------
# Native EXPERT adapter key recognition: the trainer's stacked expert tensors are named
# `...experts.{gate_up,down}.lora_{A,B}` WITHOUT a `.weight` suffix. adapter_modules must
# recognize them (they previously read as EMPTY, so a native expert adapter inspected as
# a no-target adapter -> a hidden EMPTY verdict). adapter_keys is now the schema source.
# --------------------------------------------------------------------------------------
def _write_native_expert_adapter(adapter_dir: Path, blocks=("model.layers.0.mlp.experts",)):
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(json.dumps(
        {"r": 8, "lora_alpha": 16, "peft_type": "LORA", "target_modules": ["experts"]}))
    header = {"__metadata__": {"format": "pt"}}
    for blk in blocks:
        for proj in ("gate_up", "down"):
            for ab in ("lora_A", "lora_B"):
                # NOTE: native stacked expert keys carry NO `.weight` suffix.
                header[f"base_model.model.{blk}.{proj}.{ab}"] = {
                    "dtype": "F16", "shape": [1, 1], "data_offsets": [0, 0]}
    blob = json.dumps(header).encode("utf-8")
    with open(adapter_dir / "adapter_model.safetensors", "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)


def test_adapter_modules_recognizes_native_expert_keys(tmp_path):
    _write_native_expert_adapter(tmp_path / "ad",
                                 blocks=("model.layers.0.mlp.experts", "model.layers.1.mlp.experts"))
    mods = adapter_modules(tmp_path / "ad")
    # 2 blocks x {gate_up, down} = 4 target-module paths, none dropped as "empty".
    assert mods == [
        "model.layers.0.mlp.experts.down", "model.layers.0.mlp.experts.gate_up",
        "model.layers.1.mlp.experts.down", "model.layers.1.mlp.experts.gate_up",
    ]


def test_serve_plan_native_expert_adapter_not_empty(tmp_path):
    """A native expert adapter must NOT inspect as EMPTY (it has real targets)."""
    _build_base(tmp_path / "base", arch="Qwen3MoeForCausalLM", model_type="qwen3_moe",
                quant_method="modelopt", layout="flat", targets=[(r, "nvfp4") for r in ROUTED])
    _write_native_expert_adapter(tmp_path / "ad")
    plan = serve_plan(tmp_path / "base", tmp_path / "ad")
    assert plan["verdict"] != "EMPTY"
    assert plan["adapter"]["n_targets"] == 2
    assert plan["adapter"]["expert_layout"] == "native"


# --------------------------------------------------------------------------------------
# Backend-aware routed verdict: BLOCKED-ROUTED is a BACKEND-GATED state (live on
# emulation/marlin, blocked on cutlass/flashinfer), NOT merge-only. The structured
# `routed` block carries the per-backend truth; render_plan states it.
# --------------------------------------------------------------------------------------
def test_fp8_classify_agrees_with_loader_no_input_scale(tmp_path):
    """FP8 is `.weight` + `.weight_scale` with NO `.weight_scale_2` (the loader's
    is_fp8_module signal). The old plan.classify required `.input_scale` -- an
    ACTIVATION scale that a checkpoint need not carry -- and mislabeled such modules
    BF16. Build an FP8 attention target WITHOUT input_scale and assert it is FP8-live.
    """
    q = "model.layers.0.self_attn.q_proj"
    keys = [f"{q}.weight", f"{q}.weight_scale"]  # FP8: no weight_scale_2, no input_scale
    (tmp_path / "base").mkdir(parents=True, exist_ok=True)
    (tmp_path / "base" / "config.json").write_text(json.dumps(
        {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
         "quantization_config": {"quant_method": "modelopt"}}))
    (tmp_path / "base" / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model-00001-of-00001.safetensors" for k in keys}}))
    _build_adapter(tmp_path / "ad", ["model.layers.0.self_attn.q_proj"])
    plan = serve_plan(tmp_path / "base", tmp_path / "ad")
    assert plan["verdict"] == "PASS"
    assert plan["targets"]["by_quant"].get("FP8") == 1
    assert plan["targets"]["fp8_dense_live"] == 1


def test_routed_verdict_is_backend_aware(tmp_path):
    _build_base(tmp_path / "base", arch="Qwen3MoeForCausalLM", model_type="qwen3_moe",
                quant_method="modelopt", layout="flat", targets=[(r, "nvfp4") for r in ROUTED])
    _build_adapter(tmp_path / "ad", _mods("flat", ROUTED))
    plan = serve_plan(tmp_path / "base", tmp_path / "ad")
    assert plan["verdict"] == "BLOCKED-ROUTED"
    routed = plan["routed"]
    assert routed["n"] == 3
    assert "emulation" in routed["live_on"]
    assert "cutlass" in routed["blocked_on"] and "flashinfer" in routed["blocked_on"]
    assert "not merge-only" in routed["note"].lower()


def test_routed_block_empty_when_no_routed_targets(tmp_path):
    _build_base(tmp_path / "base", arch="Qwen3MoeForCausalLM", model_type="qwen3_moe",
                quant_method="modelopt", layout="flat", targets=[(r, "nvfp4") for r in ATTN])
    _build_adapter(tmp_path / "ad", _mods("flat", ATTN))
    plan = serve_plan(tmp_path / "base", tmp_path / "ad")
    assert plan["verdict"] == "PASS"
    assert plan["routed"]["n"] == 0
    assert plan["routed"]["live_on"] == [] and plan["routed"]["blocked_on"] == []


def test_render_plan_states_backend_gating(tmp_path):
    _build_base(tmp_path / "base", arch="Qwen3MoeForCausalLM", model_type="qwen3_moe",
                quant_method="modelopt", layout="flat", targets=[(r, "nvfp4") for r in ROUTED])
    _build_adapter(tmp_path / "ad", _mods("flat", ROUTED))
    text = render_plan(serve_plan(tmp_path / "base", tmp_path / "ad"))
    assert "BACKEND-GATED" in text
    assert "emulation" in text
    # No longer claims a flat "BLOCKED (routed-MoE)" merge-only line.
    assert "ROUTED-MoE (gated)" in text
