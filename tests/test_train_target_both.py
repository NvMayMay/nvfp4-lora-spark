"""`--train-target both` contract (no-model-load, CPU-only).

Covers the joint LLM + vision-tower LoRA mode and the review-hardened correctness fixes:
  B(view)  family_view("both") LOADS the tower like vision but KEEPS the registry freeze
           (not vision_freeze), carries the two scopes separately, pins peft_scope to text.
  B(inv)   the TEXT-suffix inventory re-excludes the tower (C6: no tower pollution of counts).
  B(wrap)  the two paired replace_bf16_targets passes wrap both halves with NO double-wrap,
           and freeze_all_then_enable_lora enables both A/B sets (C8).
  B(gate)  the both grad-gate is ASYMMETRIC -- ALL-nonzero on the vision half, >=1 on the
           text half (a MoE LLM routes only a subset per batch) (C4).
  B(data)  a text-only over-length row TRUNCATES (does not crash a run), and a batch mixing
           image + text-only rows is rejected (C7).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from nvfp4_lora import families as F
from nvfp4_lora import loader as L
from nvfp4_lora import mm_data as M
from nvfp4_lora.families import family_view
from nvfp4_lora.linear import BF16LoRALinear
from nvfp4_lora.loader import replace_bf16_targets

MISTRAL3 = "mistral3"


def _entry(name):
    return F.FAMILIES[name]


@pytest.fixture
def m3_dir(fixtures_dir):
    return fixtures_dir / "mistral3"


# =========================================================================
# family_view("both")
# =========================================================================

def test_both_loads_tower_like_vision():
    b = family_view(_entry(MISTRAL3), "both")
    assert b["_train_target"] == "both"
    # Tower + projector LOAD (un-skipped) and translate to their in-memory paths, same as vision.
    assert b["skip_st_prefixes"] == ()
    assert b["meta_allowed_prefixes"] == ()
    translate = F.make_family_translator(b)
    assert translate("vision_tower.transformer.layers.0.attention.q_proj.weight") == \
        "model.vision_tower.transformer.layers.0.attention.q_proj.weight"


def test_both_keeps_registry_freeze_not_vision_freeze():
    """C1: `both` must freeze the NON-trained towers (registry freeze), NOT the LLM."""
    entry = _entry(MISTRAL3)
    b = family_view(entry, "both")
    # KEEP the registry freeze (the towers) -- NOT vision_freeze (which freezes the LLM).
    assert b["freeze"] == entry["freeze"] == ("vision_tower", "multi_modal_projector")
    assert b["freeze"] != entry["vision_freeze"]        # vision_freeze == ("language_model",)
    assert "language_model" not in b["freeze"]          # the LLM is trainable in `both`


def test_both_carries_two_scopes_and_pins_peft_scope():
    """C4/finding4: the text and vision scopes are separate; peft_scope is pinned to text."""
    entry = _entry(MISTRAL3)
    b = family_view(entry, "both")
    import re
    # text scope matches the LLM backbone, NOT the tower.
    text_scope = re.compile(b["_text_peft_scope"])
    assert text_scope.search("model.language_model.layers.0.self_attn.q_proj")
    assert text_scope.search("model.vision_tower.transformer.layers.0.attention.q_proj") is None
    # vision scope matches the tower, NOT the LLM backbone.
    vis_scope = re.compile(b["_vision_peft_scope"])
    assert vis_scope.search("model.vision_tower.transformer.layers.0.attention.q_proj")
    assert vis_scope.search("model.language_model.layers.0.self_attn.q_proj") is None
    # peft_scope is pinned to the TEXT scope (so a stray consumer behaves text-like).
    assert b["peft_scope"] == entry["peft_scope"] == b["_text_peft_scope"]
    # projector scope present by default.
    assert b["_projector_scopes"]


def test_both_text_view_still_identity_and_unsupported_refuses():
    entry = _entry(MISTRAL3)
    assert family_view(entry, "text") is entry            # text unchanged
    text_only = _entry("qwen3")
    with pytest.raises(SystemExit) as e:
        family_view(text_only, "both")
    assert "both" in str(e.value)                          # refused, names the target


def test_both_rejects_bad_target_value():
    with pytest.raises(ValueError):
        family_view(_entry(MISTRAL3), "banana")


# =========================================================================
# C6 -- the TEXT inventory re-excludes the tower in `both`
# =========================================================================

def test_both_inventory_excludes_tower_from_text_counts(m3_dir):
    """`q_proj` exists ONLY in the tower here. vision mode counts it; text and `both` must not."""
    vis = family_view(_entry(MISTRAL3), "vision")
    both = family_view(_entry(MISTRAL3), "both")
    inv_vis = L.build_target_inventory(m3_dir, ["q_proj"], family=vis)
    inv_both = L.build_target_inventory(m3_dir, ["q_proj"], family=both)
    inv_text = L.build_target_inventory(m3_dir, ["q_proj"], family=_entry(MISTRAL3))
    # vision SEES the tower's q_proj; `both` and text EXCLUDE it (the C6 fix).
    assert inv_vis["q_proj"]["counts"] == {"bf16": 1}
    assert inv_both["q_proj"]["counts"] == {}
    assert inv_both["q_proj"]["counts"] == inv_text["q_proj"]["counts"]


# =========================================================================
# C8 -- two paired bf16 passes wrap both halves, no double-wrap
# =========================================================================

class _TinyBoth(nn.Module):
    """An LLM backbone (text scope) + a tower (vision scope) + a Sequential projector."""
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.attn = nn.Module()
        self.language_model.attn.q_proj = nn.Linear(4, 4, bias=False)
        self.vision_model = nn.Module()
        self.vision_model.blk = nn.Module()
        self.vision_model.blk.qkv = nn.Linear(4, 4, bias=False)
        self.mlp1 = nn.Sequential(nn.Linear(4, 8, bias=False), nn.GELU(),
                                  nn.Linear(8, 4, bias=False))


def test_both_paired_passes_wrap_both_halves_no_double_wrap(train_mod):
    m = _TinyBoth()
    # Pass A: text suffix under the TEXT scope, projector_scopes=() (projector belongs to B).
    n_a = replace_bf16_targets(m, ["q_proj"], r"^language_model\.", r=2, lora_alpha=4,
                               device=torch.device("cpu"), dtype=torch.float32,
                               projector_scopes=())
    # Pass B: tower suffix under the VISION scope + the path-scoped Sequential projector.
    n_b = replace_bf16_targets(m, ["qkv"], r"^vision_model\.", r=2, lora_alpha=4,
                               device=torch.device("cpu"), dtype=torch.float32,
                               projector_scopes=(r"^mlp1\.",))
    assert n_a == 1                                        # just the LLM q_proj
    assert n_b == 3                                        # tower qkv + mlp1.0 + mlp1.2
    wrapped = {n for n, mod in m.named_modules() if isinstance(mod, BF16LoRALinear)}
    assert wrapped == {"language_model.attn.q_proj", "vision_model.blk.qkv", "mlp1.0", "mlp1.2"}
    # Pass B did NOT re-wrap the pass-A module (BF16LoRALinear is not an nn.Linear).
    assert isinstance(m.language_model.attn.q_proj, BF16LoRALinear)

    # freeze/enable turns on exactly the LoRA A/B of BOTH halves, nothing else.
    n_enabled = train_mod.freeze_all_then_enable_lora(m)
    assert n_enabled == 8                                  # 4 wrapped Linears x (lora_A, lora_B)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert all(n.endswith(".lora_A") or n.endswith(".lora_B") for n in trainable)
    # Both halves are represented among the trainable params.
    assert any("language_model" in n for n in trainable)
    assert any("vision_model" in n or n.startswith("mlp1") for n in trainable)


# =========================================================================
# C4 -- asymmetric grad-gate (ALL vision / >=1 text)
# =========================================================================

class _BothGrad(nn.Module):
    def __init__(self, *, sever_vision=False, no_text_grad=False):
        super().__init__()
        self.tower = BF16LoRALinear(4, 4, torch.zeros(4, 4), r=2, lora_alpha=4, dtype=torch.float32)
        self.text_routed = BF16LoRALinear(4, 4, torch.zeros(4, 4), r=2, lora_alpha=4, dtype=torch.float32)
        # An "unrouted expert": a text LoRA that is NEVER in the graph this batch.
        self.text_idle = BF16LoRALinear(4, 4, torch.zeros(4, 4), r=2, lora_alpha=4, dtype=torch.float32)
        self.head = nn.Linear(4, 4, dtype=torch.float32)
        self.sever_vision = sever_vision
        self.no_text_grad = no_text_grad
        # Nudge lora_B off its zero init so dL/dB is well-defined + non-zero on this stub.
        with torch.no_grad():
            self.tower.lora_B.add_(0.1)
            self.text_routed.lora_B.add_(0.1)

    def forward(self, x):
        t = self.tower(x)
        if self.sever_vision:
            t = t.detach()                                 # vision graph severed
        txt = torch.zeros_like(x) if self.no_text_grad else self.text_routed(x)
        return self.head(t + txt)


def _is_vision(name):
    return name.startswith("tower")


def test_both_gate_passes_with_routed_subset(train_mod):
    """Vision half all-nonzero + only ONE text lora_B nonzero (the routed one) -> PASS."""
    m = _BothGrad()
    m(torch.randn(2, 4)).sum().backward()
    events = []
    train_mod.assert_vision_grads_flow(
        m, lambda ev, **k: events.append((ev, k)),
        train_target="both", is_vision_param=_is_vision)
    ev = dict(events)["both_first_image_grad_check"]
    assert ev["text_lora_B"] == 2 and ev["text_lora_B_with_grad"] == 1   # idle expert got none


def test_both_gate_trips_on_severed_vision(train_mod):
    m = _BothGrad(sever_vision=True)
    m(torch.randn(2, 4)).sum().backward()
    with pytest.raises(SystemExit) as e:
        train_mod.assert_vision_grads_flow(
            m, lambda *a, **k: None, train_target="both", is_vision_param=_is_vision)
    assert "severed vision" in str(e.value)


def test_both_gate_trips_when_no_text_grad(train_mod):
    m = _BothGrad(no_text_grad=True)
    m(torch.randn(2, 4)).sum().backward()
    with pytest.raises(SystemExit) as e:
        train_mod.assert_vision_grads_flow(
            m, lambda *a, **k: None, train_target="both", is_vision_param=_is_vision)
    assert "not training" in str(e.value)


# =========================================================================
# C7 -- text-only truncation + mixed-batch guard
# =========================================================================

class _LenProc:
    """Minimal processor: whitespace-tokenizes the rendered text; emits pixel_values iff images."""
    def __init__(self):
        self.tokenizer = self
        self.unk_token_id = 0

    def convert_tokens_to_ids(self, t):
        return 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return " ".join(m["content"] for m in messages if isinstance(m.get("content"), str))

    def __call__(self, text=None, images=None, return_tensors=None):
        n = len(text.split())
        input_ids = torch.arange(1, n + 1).unsqueeze(0)
        out = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        if images:
            out["pixel_values"] = torch.zeros(len(images), 3, 4, 4)
        return out


def _both_collator(max_length):
    return M.MultimodalCollator(_LenProc(), image_token_ids=[], max_length=max_length,
                                pad_token_id=0)


def test_text_only_overlength_row_truncates_not_raises():
    c = _both_collator(max_length=5)
    ex = {"messages": [{"role": "user", "content": "a b c d e f g h i j"}], "images": []}
    enc = c.encode_example(ex)                              # 10 tokens > 5, text-only
    assert int(enc["input_ids"].shape[-1]) == 5            # truncated, no raise
    assert "pixel_values" not in enc


def test_image_overlength_row_still_raises():
    c = _both_collator(max_length=5)
    ex = {"messages": [{"role": "user", "content": "a b c d e f g h i j"}], "images": ["img"]}
    with pytest.raises(ValueError) as e:
        c.encode_example(ex)                               # image row must NOT truncate
    assert "refusing to truncate" in str(e.value)


def test_mixed_image_text_batch_is_rejected():
    c = _both_collator(max_length=128)
    img = {"messages": [{"role": "user", "content": "hi"}], "images": ["img"]}
    txt = {"messages": [{"role": "user", "content": "hello there"}], "images": []}
    with pytest.raises(ValueError) as e:
        c([img, txt])
    assert "mixing image and text-only" in str(e.value)


# =========================================================================
# Text-only bypass (BOTH on a VLM whose forward mandates pixel_values)
# =========================================================================

class _StubLM(nn.Module):
    """A minimal stand-in for `model.language_model`: embeds ids, applies a head."""
    def __init__(self, vocab=10, d=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
        cfg = type("cfg", (), {})()
        cfg.vocab_size = vocab
        self.config = cfg

    def get_input_embeddings(self):
        return self.emb

    def forward(self, inputs_embeds=None, attention_mask=None, use_cache=None, return_dict=None):
        return type("out", (), {"logits": self.head(inputs_embeds)})()


def test_text_only_bypass_delegates_image_and_bypasses_text(train_mod):
    lm = _StubLM()
    sentinel = object()

    def _orig(**kw):
        return sentinel

    fwd = train_mod.build_text_only_bypass_forward(_orig, lm)

    # IMAGE batch (pixel_values present) -> the model's own forward, untouched.
    assert fwd(pixel_values=torch.zeros(1, 3, 4, 4), input_ids=torch.tensor([[1, 2]])) is sentinel

    # TEXT-only batch (no pixel_values) -> LLM path with a finite CE loss + LLM grad.
    ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 3, 4]])
    out = fwd(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert tuple(out.logits.shape) == (1, 4, 10)
    out.loss.backward()
    assert lm.head.weight.grad is not None    # the LLM adapter would receive gradient

    # A text batch with no supervised labels still returns (loss None), never crashes on no image.
    out2 = fwd(input_ids=ids, attention_mask=torch.ones_like(ids))
    assert out2.loss is None
