"""Phase-A hardening tests for the top-level-projector + frozen-backbone-scatter fixes
(the two silent-failure traps the reviewers flagged when onboarding NemotronH-Omni).

1. `test_projector_sequential_linears_actually_wrapped` -- proves the loader WRAPS a projector
   whose Linears are `nn.Sequential`-indexed (`mlp1.1`/`mlp1.3`, leaf names "1"/"3" that can't
   be `vision_target_suffixes`), via the path-based `projector_scopes`. The old code silently
   wrapped ZERO projector Linears; the grad-gate (which only checks params that exist) passed
   anyway, and merge produced a tower-only adapter.

2. `test_embedding_nonleaf_hook_legalizes_inplace_scatter` -- proves the vision-run embedding
   hook makes the model's OWN in-place image-feature scatter legal. A frozen embedding output
   is a leaf with requires_grad=False; the model reshapes it (a view) and writes grad-requiring
   image features in-place -> autograd forbids promoting a leaf via in-place. The hook makes the
   output a NON-leaf (o + a grad-requiring 0), so the scatter is legal and gradient reaches the
   image features (the tower LoRA path).
"""
import re

import pytest
import torch
import torch.nn as nn

from nvfp4_lora.loader import replace_bf16_targets, BF16LoRALinear

CPU = torch.device("cpu")


def test_projector_sequential_linears_actually_wrapped():
    # A top-level `mlp1` projector = Sequential(norm, Linear@idx1, act, Linear@idx3) -- exactly
    # nemotron_omni's shape -- plus a `vision_model` tower Linear named `qkv`.
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = nn.Module()
            self.vision_model.qkv = nn.Linear(8, 8, bias=False)
            self.mlp1 = nn.Sequential(
                nn.LayerNorm(8), nn.Linear(8, 16, bias=False), nn.GELU(), nn.Linear(16, 8, bias=False))

    m = Tiny()
    n = replace_bf16_targets(
        m, target_lora_suffixes=("qkv",), peft_scope=r"^vision_model\.",
        r=4, lora_alpha=8, device=CPU,
        projector_scopes=(r"^mlp1\.",),
    )
    wrapped = {name for name, mod in m.named_modules() if isinstance(mod, BF16LoRALinear)}
    # tower Linear matched by suffix; BOTH projector Linears matched by PATH (not suffix "1"/"3").
    assert "vision_model.qkv" in wrapped
    assert "mlp1.1" in wrapped and "mlp1.3" in wrapped, f"projector not wrapped: {wrapped}"
    assert n == 3
    # And they are trainable (lora_B is a real parameter that will receive grad).
    assert any(p.requires_grad for p in m.mlp1[1].parameters())


def test_embedding_nonleaf_hook_legalizes_inplace_scatter():
    """The model's forward pattern: inputs_embeds[selected] = inputs_embeds[selected]*0 + vit.

    The forbidden case arises when gradient-checkpointing's `make_inputs_require_grad` hook does
    `output.requires_grad_(True)` on the (frozen) embedding output -- an in-place promote that turns
    it into a LEAF that requires grad. Reshaping is then a view of that leaf, and the model's
    in-place image scatter on the view is forbidden. Our `mm_embed_grad_hook` instead returns
    `output + a grad-requiring 0` -> a NON-leaf, so the same scatter is legal.
    """
    emb = nn.Embedding(10, 4)
    emb.weight.requires_grad_(False)          # frozen backbone embedding
    ids = torch.tensor([[1, 2, 3]])
    selected = torch.tensor([True, False, True])

    def scatter(inputs_embeds):
        flat = inputs_embeds.reshape(-1, 4)   # a VIEW of the embedding output
        vit = torch.randn(int(selected.sum()), 4, requires_grad=True)  # image features (need grad)
        flat[selected] = flat[selected] * 0.0 + vit
        return flat, vit

    # GC-style hook (requires_grad_(True), in-place) -> embedding output is a LEAF that requires
    # grad -> the model's in-place scatter into its view is forbidden.
    def _gc_style_hook(_m, _i, o):
        o.requires_grad_(True)
        return o
    h_bad = emb.register_forward_hook(_gc_style_hook)
    try:
        with pytest.raises(RuntimeError, match="leaf Variable that requires grad"):
            scatter(emb(ids))
    finally:
        h_bad.remove()

    # Our fix: `output + a grad-requiring 0` -> a NON-leaf requiring grad -> scatter is legal and
    # gradient flows back to the image features (the tower LoRA path).
    h_fix = emb.register_forward_hook(
        lambda _m, _i, o: o + torch.zeros((), device=o.device, dtype=o.dtype).requires_grad_(True))
    try:
        flat, vit = scatter(emb(ids))
        flat.float().pow(2).sum().backward()
        assert vit.grad is not None and torch.isfinite(vit.grad).all() and vit.grad.abs().sum() > 0
    finally:
        h_fix.remove()
