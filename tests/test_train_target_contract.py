"""`--train-target {text,vision}` toggle contract matrix (no-model-load, CPU-only).

Mirrors the 22-case style of test_serve_contract / test_strict_load_and_meta: pure
registry / family_view / loader-inventory assertions over the synthetic mistral3 fixture
(config.json + model.safetensors.index.json, no weights) and the tiny generated
mm_vision image fixture. Nothing is loaded onto a GPU and no model is built.

The two halves of the toggle:
  T*  text mode is byte-for-byte today's behaviour (tower skipped/frozen/meta),
  V*  vision mode inverts EXACTLY (tower + projector load/materialize/train, text
      backbone frozen, vision scope selected),
  X*  the two scopes are disjoint (XOR: no module is ever in both),
  P*  projector-scope policy (in vision by default, never in text, never on meta),
  D*  the multimodal data path masks image + prompt tokens to -100,
  G*  the first-backward grad guard trips on the no_grad footgun.

Plus: the FULL existing suite stays green untouched -- that is the text-mode regression
contract.
"""
from __future__ import annotations

import json
import re

import pytest
import torch
import torch.nn as nn

from nvfp4_lora import families as F
from nvfp4_lora import loader as L
from nvfp4_lora import mm_data as M
from nvfp4_lora.families import adapter_key_to_base_prefix, family_view
from nvfp4_lora.linear import BF16LoRALinear

MISTRAL3 = "mistral3"
TEXT_ONLY_FAMILIES = ("qwen3", "llama", "glm4_moe", "nemotron_h", "qwen3_5_moe",
                      "qwen3_5_moe_text")
VISION_FAMILIES = ("mistral3", "mistral4", "llama4")


@pytest.fixture
def m3_dir(fixtures_dir):
    return fixtures_dir / "mistral3"


@pytest.fixture
def mm_dir(fixtures_dir):
    return fixtures_dir / "mm_vision"


def _entry(name):
    return F.FAMILIES[name]


# =========================================================================
# T -- text mode is unchanged
# =========================================================================

def test_T1_text_view_is_identity():
    m = _entry(MISTRAL3)
    assert family_view(m, "text") is m  # same object, zero behaviour change
    # The tower is skipped + frozen + meta-allowed, exactly as before.
    assert "vision_tower." in m["skip_st_prefixes"]
    assert "multi_modal_projector." in m["skip_st_prefixes"]
    assert m["freeze"] == ("vision_tower", "multi_modal_projector")
    assert "model.vision_tower." in m["meta_allowed_prefixes"]
    assert m["peft_scope"] == r"^model\.language_model\."


def test_T2_text_translator_skips_tower_and_projector():
    translate = F.make_family_translator(_entry(MISTRAL3))
    assert translate("vision_tower.transformer.layers.0.attention.q_proj.weight") is None
    assert translate("multi_modal_projector.linear_1.weight") is None
    # A text-backbone key still translates (not skipped).
    assert translate("language_model.model.layers.0.self_attn.q_a_proj.weight") == \
        "model.language_model.layers.0.self_attn.q_a_proj.weight"


def test_T3_text_scope_never_matches_a_tower_qproj():
    scope = re.compile(_entry(MISTRAL3)["peft_scope"])
    assert scope.search("model.vision_tower.transformer.layers.0.attention.q_proj") is None
    assert scope.search("model.language_model.layers.0.self_attn.q_a_proj") is not None


def test_T4_text_inventory_excludes_the_tower(m3_dir):
    # text mode (family=None) skips vision_tower.*, so a tower-only suffix inventories
    # to nothing -- proving the text path never reaches the tower.
    inv = L.build_target_inventory(m3_dir, ["q_proj"])
    assert inv["q_proj"]["counts"] == {}


@pytest.mark.parametrize("fam", TEXT_ONLY_FAMILIES)
def test_T5_text_only_families_identity_and_refuse_vision(fam):
    entry = _entry(fam)
    assert family_view(entry, "text") is entry
    assert not F.family_supports_vision(entry)
    with pytest.raises(SystemExit) as e:
        family_view(entry, "vision")
    assert "vision" in str(e.value)


def test_T6_family_config_without_vision_keys_refuses_vision(tmp_path):
    p = tmp_path / "family.json"
    p.write_text(json.dumps({
        "auto_class": "causal_lm", "expert_prefix": None,
        "peft_scope": r"^model\.layers\.", "freeze": [], "skip_st_prefixes": [],
        "st_to_model": None, "meta_allowed_prefixes": [], "moe_experts_class": None,
    }))
    fam = F.load_family_config(p)
    assert family_view(fam, "text") is fam                 # text identity preserved
    with pytest.raises(SystemExit):
        family_view(fam, "vision")                          # vision refused (no scope)


# =========================================================================
# V -- vision mode inverts exactly
# =========================================================================

def test_V1_vision_unskips_tower_and_projector():
    v = family_view(_entry(MISTRAL3), "vision")
    assert v["skip_st_prefixes"] == ()                      # tower + projector now LOAD
    translate = F.make_family_translator(v)
    assert translate("vision_tower.transformer.layers.0.attention.q_proj.weight") == \
        "model.vision_tower.transformer.layers.0.attention.q_proj.weight"
    assert translate("multi_modal_projector.linear_1.weight") == \
        "model.multi_modal_projector.linear_1.weight"


def test_V2_vision_removes_tower_from_meta_allowance():
    v = family_view(_entry(MISTRAL3), "vision")
    # assert_no_meta_tensors would now FAIL a meta tower/projector (nothing allow-listed).
    assert v["meta_allowed_prefixes"] == ()


def test_V3_vision_freezes_text_backbone_not_tower():
    v = family_view(_entry(MISTRAL3), "vision")
    assert v["freeze"] == ("language_model",)
    assert "vision_tower" not in v["freeze"]
    assert "multi_modal_projector" not in v["freeze"]


def test_V4_vision_scope_matches_tower_not_backbone():
    scope = re.compile(family_view(_entry(MISTRAL3), "vision")["peft_scope"])
    assert scope.search("model.vision_tower.transformer.layers.0.attention.q_proj")
    assert scope.search("model.language_model.layers.0.self_attn.q_proj") is None


def test_V5_projector_policy(m3_dir):
    m = _entry(MISTRAL3)
    inc = family_view(m, "vision", include_projector=True)
    exc = family_view(m, "vision", include_projector=False)
    # Default: projector IS a target.
    assert re.compile(inc["peft_scope"]).search("model.multi_modal_projector.linear_1")
    # --no-include-projector: NOT a target...
    assert re.compile(exc["peft_scope"]).search("model.multi_modal_projector.linear_1") is None
    # ...but still materialized + frozen, NEVER on meta (not skipped, not meta-allowed).
    for view in (inc, exc):
        assert "multi_modal_projector." not in view["skip_st_prefixes"]
        assert "model.multi_modal_projector." not in view["meta_allowed_prefixes"]


def test_V6_vision_inventory_counts_tower_and_projector(m3_dir):
    v = family_view(_entry(MISTRAL3), "vision")
    inv = L.build_target_inventory(m3_dir, ["q_proj", "linear_1", "q_a_proj"], family=v)
    # Tower attention q_proj: bf16, bucketed under its vision layer index.
    assert inv["q_proj"]["counts"] == {"bf16": 1}
    assert inv["q_proj"]["layers"] == {"bf16": [0]}
    # Projector linear_1: bf16, in its own (layer-less) bucket.
    assert inv["linear_1"]["counts"] == {"bf16": 1}
    # A TEXT-backbone suffix inventories to nothing in vision mode (restricted to the
    # tower/projector scope) -- zero text-backbone LoRA targets.
    assert inv["q_a_proj"]["counts"] == {}


@pytest.mark.parametrize("fam", VISION_FAMILIES)
def test_V7_all_vision_families_invert(fam):
    entry = _entry(fam)
    assert F.family_supports_vision(entry)
    v = family_view(entry, "vision")
    assert v["_train_target"] == "vision"
    # Every declared vision st-prefix is removed from skip + meta.
    for stp in entry["vision_st_prefixes"]:
        assert stp not in v["skip_st_prefixes"]
    for _st, mem in entry["vision_st_to_model"]:
        assert mem not in v["meta_allowed_prefixes"]
    # Freeze flips to the text backbone.
    assert v["freeze"] == entry["vision_freeze"]
    # Vision scope matches the tower, text scope does not.
    assert re.compile(v["peft_scope"]).search(entry["vision_st_to_model"][0][1] + "layers.0.self_attn.q_proj")


# =========================================================================
# X -- XOR disjointness
# =========================================================================

def test_X1_scopes_disjoint_over_checkpoint_inventory(m3_dir):
    m = _entry(MISTRAL3)
    text_scope = re.compile(m["peft_scope"])
    vision_scope = re.compile(family_view(m, "vision")["peft_scope"])
    wm = json.loads((m3_dir / "model.safetensors.index.json").read_text())["weight_map"]

    text_tr = F.make_family_translator(m)
    vision_tr = F.make_family_translator(family_view(m, "vision"))
    mem_names = set()
    for key in wm:
        for tr in (text_tr, vision_tr):
            mem = tr(key)
            if mem is not None:
                mem_names.add(mem.rsplit(".", 1)[0])  # module path (drop .weight/.scale)
    assert mem_names, "fixture produced no in-memory module names"
    both = [n for n in mem_names if text_scope.search(n) and vision_scope.search(n)]
    assert both == [], f"modules matched BOTH scopes (XOR violated): {both}"
    # And each scope actually matches SOMETHING (the test is not vacuous).
    assert any(text_scope.search(n) for n in mem_names)
    assert any(vision_scope.search(n) for n in mem_names)


def test_X2_vision_adapter_key_roundtrips():
    # A saved vision-tower LoRA key maps to its on-disk base prefix and back.
    mem_prefix, st_prefix = "model.vision_tower.", "vision_tower."
    module = "model.vision_tower.transformer.layers.0.attention.q_proj"
    akey = f"base_model.model.{module}.lora_B.weight"
    prefix, side = adapter_key_to_base_prefix(akey, mem_prefix, st_prefix)
    assert prefix == "vision_tower.transformer.layers.0.attention.q_proj"
    assert side == "B"
    # The projector round-trips the same way.
    pkey = "base_model.model.model.multi_modal_projector.linear_1.lora_A.weight"
    pprefix, pside = adapter_key_to_base_prefix(
        pkey, "model.multi_modal_projector.", "multi_modal_projector.")
    assert pprefix == "multi_modal_projector.linear_1" and pside == "A"


# =========================================================================
# D -- multimodal data path masking / validation
# =========================================================================

# A faithful mini Pixtral-like processor: [IMG]=10 expands to 3 image tokens closed by
# [IMG_END]=13; text words tokenize to distinct ids. No model/weights involved.
_IMG, _IMG_END = 10, 13


class _StubProcessor:
    def __init__(self):
        self.tokenizer = self
        self.unk_token_id = 0

    def convert_tokens_to_ids(self, t):
        return {"[IMG]": _IMG, "[IMG_END]": _IMG_END}.get(t, 0)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = ["<bos>"]
        for msg in messages:
            parts.append("[INST]" if msg["role"] == "user" else "[/INST]")
            content = msg["content"]
            if isinstance(content, list):
                for p in content:
                    parts.append("<IMG>" if p["type"] == "image" else p["text"])
            else:
                parts.append(content)
        if add_generation_prompt:
            parts.append("[/INST]")
        return " ".join(parts)

    def __call__(self, text=None, images=None, return_tensors=None):
        ids = []
        for tok in text.split():
            if tok == "<IMG>":
                ids += [_IMG, _IMG, _IMG, _IMG_END]
            elif tok == "<bos>":
                ids.append(1)
            elif tok == "[INST]":
                ids.append(2)
            elif tok == "[/INST]":
                ids.append(3)
            else:
                ids.append(100 + (abs(hash(tok)) % 50))
        input_ids = torch.tensor([ids])
        out = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        if images:
            out["pixel_values"] = torch.zeros(len(images), 3, 4, 4)
            out["image_sizes"] = torch.tensor([[4, 4]] * len(images))
        return out


def _collator(max_length=128):
    proc = _StubProcessor()
    ids = M.resolve_image_token_ids(proc)
    return M.MultimodalCollator(proc, image_token_ids=ids, max_length=max_length,
                                pad_token_id=0), ids


def test_dataset_loads_fixture_and_validates(mm_dir):
    ds = M.MultimodalJsonlDataset(str(mm_dir / "smoke.jsonl"))
    assert len(ds) == 10
    ex = ds[0]
    assert len(ex["images"]) == 1 and ex["images"][0].mode == "RGB"
    # The two-image row aligns to two PIL images.
    two = ds[9]
    assert M.count_image_parts(two["messages"]) == len(two["images"]) == 2


def test_D1_image_count_mismatch_is_row_numbered(tmp_path):
    from pathlib import Path
    bad = {"messages": [{"role": "user", "content": [{"type": "image"}]}], "images": []}
    with pytest.raises(ValueError) as e:
        M.resolve_image_paths(bad, 7, Path("."))
    assert "row 7" in str(e.value)


def test_D1b_missing_image_file_is_row_numbered(tmp_path):
    from pathlib import Path
    row = {"messages": [{"role": "user", "content": [{"type": "image"}]}],
           "images": ["nope.png"]}
    with pytest.raises(ValueError) as e:
        M.resolve_image_paths(row, 3, Path(tmp_path))
    assert "row 3" in str(e.value) and "not found" in str(e.value)


def test_D2_labels_mask_image_and_prompt_tokens():
    coll, img_ids = _collator()
    ex = {"messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "whatis"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "acat"}]}],
        "images": [object()]}  # one placeholder image; the stub does not decode it
    batch = coll([ex])
    ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    # Every image / control token is masked.
    for pos, tid in enumerate(ids):
        if tid in img_ids:
            assert labels[pos] == -100, f"image token at {pos} not masked"
    # Exactly the assistant answer span is supervised (everything else is -100).
    supervised = [p for p, l in enumerate(labels) if l != -100]
    assert supervised, "no supervised tokens"
    assert all(labels[p] == ids[p] for p in supervised)
    assert all(ids[p] not in img_ids for p in supervised)
    # pixel_values passed straight through.
    assert "pixel_values" in batch and batch["pixel_values"].shape[0] == 1


def test_D2b_masking_pure_function():
    labels = M.mask_labels([1, 10, 10, 13, 55, 66], prompt_len=4, image_token_ids=[10, 13])
    assert labels == [-100, -100, -100, -100, 55, 66]


def test_D3_over_max_length_is_hard_error_not_truncation():
    coll, _ = _collator(max_length=5)
    ex = {"messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "q"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a"}]}],
        "images": [object()]}
    with pytest.raises(ValueError) as e:
        coll([ex])
    assert "max_length" in str(e.value) and "truncat" in str(e.value)


# =========================================================================
# G -- first-backward gradient guard (the no_grad footgun)
# =========================================================================

class _TowerToLLM(nn.Module):
    """A minimal 'vision tower (LoRA) -> LLM head' stub for the grad guard.

    The LoRA lives on the 'tower' Linear. `sever=True` cuts the tower's contribution to
    the loss (h.detach()) -- the mechanical equivalent of wrapping the frozen-LLM forward
    in torch.no_grad / detaching hidden states between projector and LLM (the plan's
    footgun): the run still backpropagates (the LLM head keeps training), but the tower
    LoRA receives NO gradient, which is exactly the silent failure the guard detects
    (lora_B.grad is None after a successful backward). `sever=False` keeps the tower on
    the graph, so its lora_B gets a real gradient and the guard passes.
    """

    def __init__(self, sever: bool):
        super().__init__()
        self.sever = sever
        self.tower = BF16LoRALinear(4, 4, torch.zeros(4, 4), r=2, lora_alpha=4,
                                    dtype=torch.float32)
        self.llm = nn.Linear(4, 4, dtype=torch.float32)
        # A severed run still backpropagates through the LLM head (so backward does not
        # error); an intact run trains only the tower (LLM frozen), matching vision mode.
        for p in self.llm.parameters():
            p.requires_grad_(sever)

    def forward(self, x):
        h = self.tower(x)
        return self.llm(h.detach() if self.sever else h)


def test_G1_grad_guard_trips_on_severed_graph(train_mod):
    model = _TowerToLLM(sever=True)
    out = model(torch.randn(2, 4))
    out.sum().backward()
    with pytest.raises(SystemExit) as e:
        train_mod.assert_vision_grads_flow(model, lambda *a, **k: None)
    assert "severed autograd graph" in str(e.value)


def test_G1_grad_guard_passes_on_intact_graph(train_mod):
    model = _TowerToLLM(sever=False)
    # Nudge lora_B off zero so dL/dB is well-defined and non-zero on this stub.
    with torch.no_grad():
        model.tower.lora_B.add_(0.1)
    out = model(torch.randn(2, 4))
    out.sum().backward()
    events = []
    train_mod.assert_vision_grads_flow(model, lambda ev, **k: events.append((ev, k)))
    assert any(ev == "first_backward_grad_check" for ev, _ in events)


def test_freeze_all_then_enable_lora_enables_only_lora(train_mod):
    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.tower = BF16LoRALinear(4, 4, torch.zeros(4, 4), r=2, lora_alpha=4,
                                        dtype=torch.float32)
            self.other = nn.Linear(4, 4, dtype=torch.float32)  # trainable by default

    model = Wrap()
    assert any(p.requires_grad for p in model.other.parameters())  # starts trainable
    n = train_mod.freeze_all_then_enable_lora(model)
    assert n == 2  # lora_A + lora_B
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"tower.lora_A", "tower.lora_B"}
    assert all(not p.requires_grad for p in model.other.parameters())
