"""Vision-LoRA merge (scripts/merge_vision_lora.py) + its VQA eval scorer.

CPU-only, no model load: every test runs on synthetic tensors / tiny safetensors
shards written into tmp_path. Covers the five contract points:

  (a) merged_weight computes W + (alpha/r)*B@A correctly for a known small case;
  (b) the bf16-only guard REFUSES an NVFP4-typed target (points at the NVFP4 merge);
  (c) a shard mixing a bf16 "tower" tensor and an NVFP4 "backbone" tensor rewrites
      only the bf16 one and leaves the NVFP4 bytes IDENTICAL (and a shard with no
      merged tensor is copied byte-for-byte);
  (d) the adapter-key -> base-key mapping handles the vision module names (tower +
      projector, mistral3 + llama4);
  (e) the VQA exact-match normalizer is case / punctuation / article insensitive.

Style mirrors tests/test_merge_key_mapping.py (importlib-load the scripts).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def merge():
    return _load_script("merge_vision_lora.py")


@pytest.fixture(scope="module")
def evalv():
    return _load_script("eval_vision_retention.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# (a) merge math
# ---------------------------------------------------------------------------

def test_a_merged_weight_known_case(merge):
    # W + (alpha/r)*B@A with alpha/r=2. B(2x1)@A(1x3) is a rank-1 outer product.
    W = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    A = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32)       # (r=1, in=3)
    B = torch.tensor([[1.0], [2.0]], dtype=torch.float32)           # (out=2, r=1)
    merged = merge.merged_weight(W, A, B, scale=2.0)
    # delta = 2 * B@A = 2 * [[1,0,-1],[2,0,-2]] = [[2,0,-2],[4,0,-4]]
    expect = torch.tensor([[3.0, 2.0, 1.0], [8.0, 5.0, 2.0]], dtype=torch.float32)
    assert torch.equal(merged, expect)


def test_a_merged_weight_preserves_bf16_dtype(merge):
    W = torch.randn(6, 4, dtype=torch.bfloat16)
    out = merge.merged_weight(W, torch.randn(2, 4), torch.randn(6, 2), scale=1.5)
    assert out.dtype == torch.bfloat16 and out.shape == W.shape


def test_a_shape_mismatch_raises(merge):
    W = torch.zeros(4, 3)
    with pytest.raises(ValueError):
        merge.merged_weight(W, torch.zeros(2, 5), torch.zeros(4, 2), scale=1.0)  # in!=3


# ---------------------------------------------------------------------------
# (b) bf16-only guard
# ---------------------------------------------------------------------------

def test_b_guard_accepts_bf16_target(merge):
    keys = {"vision_tower.x.weight"}
    merge.assert_mergeable_target(keys, "vision_tower.x.weight")  # no raise


def test_b_guard_refuses_nvfp4_ct_target(merge):
    keys = {"vision_tower.x.weight_packed", "vision_tower.x.weight_scale",
            "vision_tower.x.weight_global_scale"}
    with pytest.raises(SystemExit) as e:
        merge.assert_mergeable_target(keys, "vision_tower.x.weight")
    msg = str(e.value)
    assert "nvfp4_ct" in msg and "merge_lora_into" in msg


def test_b_guard_refuses_nvfp4_modelopt_target(merge):
    keys = {"vision_tower.x.weight", "vision_tower.x.weight_scale",
            "vision_tower.x.weight_scale_2"}
    with pytest.raises(SystemExit) as e:
        merge.assert_mergeable_target(keys, "vision_tower.x.weight")
    assert "nvfp4_modelopt" in str(e.value)


def test_b_guard_refuses_fp8_target(merge):
    keys = {"vision_tower.x.weight", "vision_tower.x.weight_scale"}  # no scale_2 => fp8
    with pytest.raises(SystemExit) as e:
        merge.assert_mergeable_target(keys, "vision_tower.x.weight")
    assert "fp8" in str(e.value)


def test_b_guard_reports_absent_target(merge):
    with pytest.raises(SystemExit) as e:
        merge.assert_mergeable_target({"other.weight"}, "vision_tower.x.weight")
    assert "not present" in str(e.value)


# ---------------------------------------------------------------------------
# (c) shard rewrite: only the bf16 tensor changes; NVFP4 bytes stay identical;
#     shards without a merged tensor are copied byte-for-byte.
# ---------------------------------------------------------------------------

def _mixed_shard(path: Path):
    """A shard mixing a bf16 tower weight and an NVFP4 (ct) backbone module."""
    torch.manual_seed(1)
    tensors = {
        # bf16 vision tower weight (the merge target)
        "vision_tower.transformer.layers.0.attention.q_proj.weight":
            torch.randn(8, 6, dtype=torch.bfloat16),
        # NVFP4 compressed-tensors backbone module (must survive byte-identical)
        "language_model.model.layers.0.mlp.experts.0.gate_proj.weight_packed":
            torch.randint(0, 255, (8, 3), dtype=torch.uint8),
        "language_model.model.layers.0.mlp.experts.0.gate_proj.weight_scale":
            torch.randn(8, 1).to(torch.float8_e4m3fn),
        "language_model.model.layers.0.mlp.experts.0.gate_proj.weight_global_scale":
            torch.tensor([0.037], dtype=torch.float32),
    }
    save_file(tensors, str(path))
    return tensors


def test_c_rewrite_preserves_nvfp4_bytes_and_copies_others(merge, tmp_path):
    base = tmp_path / "base"
    out = tmp_path / "out"
    base.mkdir()
    out.mkdir()

    orig = _mixed_shard(base / "model-00001.safetensors")
    # A second shard with NO merged tensor: pure NVFP4 backbone, must be byte-copied.
    save_file(
        {"language_model.model.layers.1.mlp.experts.0.down_proj.weight_packed":
            torch.randint(0, 255, (8, 3), dtype=torch.uint8)},
        str(base / "model-00002.safetensors"),
    )

    tower_key = "vision_tower.transformer.layers.0.attention.q_proj.weight"
    weight_map = {k: "model-00001.safetensors" for k in orig}
    weight_map["language_model.model.layers.1.mlp.experts.0.down_proj.weight_packed"] = \
        "model-00002.safetensors"

    merged_tower = merge.merged_weight(
        orig[tower_key], torch.randn(2, 6, dtype=torch.bfloat16),
        torch.randn(8, 2, dtype=torch.bfloat16), scale=2.0,
    )
    summary = merge.rewrite_shards(base, out, weight_map, {tower_key: merged_tower})

    assert summary["n_shards_rewritten"] == 1 and summary["n_shards_copied"] == 1

    # Shard 2 (no merge) is byte-for-byte identical.
    assert _sha256(base / "model-00002.safetensors") == \
        _sha256(out / "model-00002.safetensors")

    # Shard 1 was rewritten: the bf16 tower weight is the merged tensor...
    from safetensors import safe_open
    with safe_open(out / "model-00001.safetensors", framework="pt") as sf:
        assert torch.equal(sf.get_tensor(tower_key), merged_tower)
        # ...and every NVFP4 backbone tensor is byte-identical to the original.
        for k in orig:
            if k == tower_key:
                continue
            a = sf.get_tensor(k)
            b = orig[k]
            assert a.dtype == b.dtype
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), \
                f"NVFP4 tensor {k} changed bytes across the rewrite"


def test_c_unplaced_replacement_raises(merge, tmp_path):
    base = tmp_path / "base"
    out = tmp_path / "out"
    base.mkdir()
    out.mkdir()
    _mixed_shard(base / "model-00001.safetensors")
    weight_map = {"vision_tower.transformer.layers.0.attention.q_proj.weight":
                  "model-00001.safetensors"}
    with pytest.raises(RuntimeError):
        merge.rewrite_shards(base, out, weight_map, {"not.in.map.weight": torch.zeros(2, 2)})


# ---------------------------------------------------------------------------
# (d) adapter-key -> base-key mapping over the vision module names
# ---------------------------------------------------------------------------

def _pairs(merge, family_name):
    from nvfp4_lora.families import FAMILIES
    return merge.vision_prefix_pairs(FAMILIES[family_name])


def test_d_mistral3_tower_key_maps(merge):
    pairs = _pairs(merge, "mistral3")
    akey = ("base_model.model.model.vision_tower.transformer.layers.3.attention."
            "q_proj.lora_A.weight")
    assert merge.adapter_key_to_base_key(akey, pairs) == \
        "vision_tower.transformer.layers.3.attention.q_proj.weight"
    assert merge.adapter_side(akey) == "A"


def test_d_mistral3_projector_key_maps(merge):
    pairs = _pairs(merge, "mistral3")
    akey = "base_model.model.model.multi_modal_projector.linear_1.lora_B.weight"
    assert merge.adapter_key_to_base_key(akey, pairs) == \
        "multi_modal_projector.linear_1.weight"
    assert merge.adapter_side(akey) == "B"


def test_d_llama4_tower_and_projector_keys_map(merge):
    pairs = _pairs(merge, "llama4")
    tower = "base_model.model.model.vision_model.model.layers.0.self_attn.o_proj.lora_A.weight"
    assert merge.adapter_key_to_base_key(tower, pairs) == \
        "vision_model.model.layers.0.self_attn.o_proj.weight"
    proj = "base_model.model.model.multi_modal_projector.linear_1.lora_B.weight"
    assert merge.adapter_key_to_base_key(proj, pairs) == \
        "multi_modal_projector.linear_1.weight"


def test_d_already_on_disk_prefix_passes_through(merge):
    pairs = _pairs(merge, "mistral3")
    akey = "base_model.model.vision_tower.transformer.layers.0.attention.k_proj.lora_A.weight"
    assert merge.adapter_key_to_base_key(akey, pairs) == \
        "vision_tower.transformer.layers.0.attention.k_proj.weight"


def test_d_text_backbone_key_is_rejected(merge):
    # A text-backbone adapter key must NOT map through the vision merge (wrong tool).
    pairs = _pairs(merge, "mistral3")
    akey = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
    with pytest.raises(ValueError) as e:
        merge.adapter_key_to_base_key(akey, pairs)
    assert "VISION adapter" in str(e.value)


def test_d_non_lora_key_raises(merge):
    pairs = _pairs(merge, "mistral3")
    with pytest.raises(ValueError):
        merge.adapter_key_to_base_key("base_model.model.model.vision_tower.x.weight", pairs)


# ---------------------------------------------------------------------------
# (d+) end-to-end merge over a tiny synthetic base + vision adapter (CPU)
# ---------------------------------------------------------------------------

def _mistral3_config(path: Path):
    (path / "config.json").write_text(json.dumps({
        "architectures": ["Mistral3ForConditionalGeneration"],
        "model_type": "mistral3",
    }))


def test_end_to_end_merge_bakes_delta_and_preserves_backbone(merge, tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    out = tmp_path / "merged"
    base.mkdir()
    adapter.mkdir()
    _mistral3_config(base)

    torch.manual_seed(2)
    tower_key = "vision_tower.transformer.layers.0.attention.q_proj.weight"
    W = torch.randn(8, 6, dtype=torch.bfloat16)
    backbone_key = "language_model.model.layers.0.mlp.experts.0.gate_proj.weight_packed"
    backbone = torch.randint(0, 255, (8, 3), dtype=torch.uint8)
    save_file({tower_key: W, backbone_key: backbone},
              str(base / "model-00001.safetensors"))
    (base / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 0},
        "weight_map": {tower_key: "model-00001.safetensors",
                       backbone_key: "model-00001.safetensors"},
    }))

    # A tiny vision adapter (native BF16LoRALinear save shape: base_model.model.<mem>).
    A = torch.randn(2, 6, dtype=torch.bfloat16)
    B = torch.randn(8, 2, dtype=torch.bfloat16)
    mem = "model.vision_tower.transformer.layers.0.attention.q_proj"
    save_file({f"base_model.model.{mem}.lora_A.weight": A,
               f"base_model.model.{mem}.lora_B.weight": B},
              str(adapter / "adapter_model.safetensors"))
    (adapter / "adapter_config.json").write_text(json.dumps({
        "r": 2, "lora_alpha": 4, "lora_dropout": 0.0,
        "target_modules": ["q_proj"],
    }))

    rc = merge.main([
        "--base-model-dir", str(base),
        "--adapter-dir", str(adapter),
        "--out-dir", str(out),
    ])
    assert rc == 0

    from safetensors import safe_open
    with safe_open(out / "model-00001.safetensors", framework="pt") as sf:
        got = sf.get_tensor(tower_key)
        expect = merge.merged_weight(W, A, B, scale=4 / 2)   # alpha/r = 2
        assert torch.equal(got, expect)
        # The NVFP4 backbone tensor is preserved byte-for-byte.
        assert torch.equal(sf.get_tensor(backbone_key).view(torch.uint8),
                           backbone.view(torch.uint8))

    # Index + config are copied unchanged (keys/shapes/dtypes preserved).
    assert _sha256(base / "model.safetensors.index.json") == \
        _sha256(out / "model.safetensors.index.json")
    assert _sha256(base / "config.json") == _sha256(out / "config.json")
    manifest = json.loads((out / "merge_manifest.json").read_text())
    assert manifest["merge_kind"] == "vision_bf16" and manifest["n_targets"] == 1
    assert manifest["merge_dtype"] == "bfloat16"


def test_end_to_end_refuses_quantized_target(merge, tmp_path):
    # An adapter whose target lands on an NVFP4 module must be refused end-to-end.
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    _mistral3_config(base)
    key = "vision_tower.transformer.layers.0.attention.q_proj"
    save_file({f"{key}.weight_packed": torch.randint(0, 255, (8, 3), dtype=torch.uint8),
               f"{key}.weight_scale": torch.randn(8, 1).to(torch.float8_e4m3fn),
               f"{key}.weight_global_scale": torch.tensor([0.03])},
              str(base / "model-00001.safetensors"))
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        f"{key}.weight_packed": "model-00001.safetensors",
        f"{key}.weight_scale": "model-00001.safetensors",
        f"{key}.weight_global_scale": "model-00001.safetensors",
    }}))
    mem = "model." + key
    save_file({f"base_model.model.{mem}.lora_A.weight": torch.randn(2, 6),
               f"base_model.model.{mem}.lora_B.weight": torch.randn(8, 2)},
              str(adapter / "adapter_model.safetensors"))
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))
    with pytest.raises(SystemExit):
        merge.main(["--base-model-dir", str(base), "--adapter-dir", str(adapter),
                    "--out-dir", str(tmp_path / "merged")])


def test_self_test_passes(merge):
    assert merge.self_test() == 0


# ---------------------------------------------------------------------------
# (e) VQA exact-match normalizer + eval row/image encoding
# ---------------------------------------------------------------------------

def test_e_normalizer_case_punct_article_insensitive(evalv):
    assert evalv.normalize_vqa("The X-ray.") == evalv.normalize_vqa("x ray")
    assert evalv.normalize_vqa("Yes!") == "yes"
    assert evalv.normalize_vqa("  A  Cat ") == "cat"
    assert evalv.normalize_vqa("an apple") == "apple"
    assert evalv.normalize_vqa("") == ""


def test_e_exact_match(evalv):
    assert evalv.vqa_exact_match("Yes.", "yes")
    assert evalv.vqa_exact_match("The left lung", "left lung")
    assert not evalv.vqa_exact_match("no", "yes")
    assert not evalv.vqa_exact_match("left", "right")


def test_e_extract_row_pulls_question_gold_and_images(evalv):
    row = {"messages": [
        {"role": "user", "content": [{"type": "image"},
                                     {"type": "text", "text": "What is shown?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a lung"}]}],
        "images": ["images/a.png"]}
    q, gold, image_parts = evalv.extract_row(row)
    assert q == "What is shown?" and gold == "a lung" and len(image_parts) == 1


def test_e_two_image_row_and_string_content(evalv):
    row = {"messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                     {"type": "text", "text": "match?"}]},
        {"role": "assistant", "content": "no"}],
        "images": ["a.png", "b.png"]}
    q, gold, image_parts = evalv.extract_row(row)
    assert q == "match?" and gold == "no" and len(image_parts) == 2


def test_e_image_data_url_and_message_shape(evalv, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfakebytes")
    url = evalv.image_data_url(img)
    assert url.startswith("data:image/png;base64,")
    import base64
    assert base64.b64decode(url.split(",", 1)[1]) == b"\x89PNG\r\n\x1a\nfakebytes"
    msgs = evalv.build_chat_messages("hello?", [url, url])
    assert msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert [c["type"] for c in content] == ["image_url", "image_url", "text"]
    assert content[0]["image_url"]["url"] == url
    assert content[-1]["text"] == "hello?"


def test_e_resolve_images_alignment_and_missing(evalv, tmp_path):
    (tmp_path / "a.png").write_bytes(b"1")
    parts = [{"type": "image"}]
    # count mismatch is row-numbered
    with pytest.raises(ValueError) as e:
        evalv.resolve_images(parts, ["a.png", "b.png"], tmp_path, 4)
    assert "row 4" in str(e.value)
    # missing file is row-numbered
    with pytest.raises(ValueError) as e:
        evalv.resolve_images(parts, ["missing.png"], tmp_path, 5)
    assert "row 5" in str(e.value) and "not found" in str(e.value)
    # happy path -> one data url
    urls = evalv.resolve_images(parts, ["a.png"], tmp_path, 0)
    assert len(urls) == 1 and urls[0].startswith("data:image/png;base64,")


def test_e_build_summary_em_and_delta(evalv):
    per = [
        {"em": {"base": True, "merged": True}, "pred": {"base": "yes", "merged": "yes"}},
        {"em": {"base": False, "merged": True}, "pred": {"base": "no", "merged": "yes"}},
        {"em": {"base": False, "merged": False}, "pred": {"base": "a", "merged": "b"}},
    ]
    summary = evalv.build_summary(per, ["base", "merged"])
    assert summary["n_paired"] == 3
    assert summary["exact_match"]["base"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary["exact_match"]["merged"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["em_delta_vs_base"]["merged"] == pytest.approx(1 / 3, abs=1e-4)
