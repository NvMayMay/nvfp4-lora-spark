"""cohere_tied_embedding_lora.py - make tied-embedding models runtime-LoRA-servable.

Cohere / Command-R / Command-A (and any `tie_word_embeddings` model) compute
logits THROUGH the input embedding: `commandr.py:compute_logits()` calls
`logits_processor(self.model.embed_tokens, hidden_states)`. With `--enable-lora`,
vLLM wraps `embed_tokens` in `VocabParallelEmbeddingWithLoRA`, which stores the
real layer under `self.base_layer` and does NOT delegate attribute access. The
logits path (`LogitsProcessor._get_logits`) then reads `lm_head.quant_method`,
`.weight`, `.shard_indices`, ... off the wrapper and raises:

    AttributeError: 'VocabParallelEmbeddingWithLoRA' object has no attribute 'quant_method'

...crashing EngineCore at profile_run. The embedding itself is bf16, so this is
NOT a quantization issue - it is the LoRA wrapper being incomplete for the
tied-embedding logits path.

Fix: give the wrapper a `__getattr__` that, for any attribute the wrapper itself
does not define, falls back to `base_layer`. Logits then compute as correct
tied-embedding logits (`F.linear` against the real embedding weight). Adapters
produced by this repo target only attention + MLP (q/k/v/o, gate/up/down) and
carry NO embedding-LoRA, so routing logits through `base_layer` loses no adapter
delta - the adapter's effect is fully present in `hidden_states` via the
transformer-layer LoRA. Untied-embedding models (e.g. Llama) never hit this path
and are unaffected.

Idempotent. Opt-in via `VLLM_PATCH_TIED_EMBED_LORA=1` (see sitecustomize.py).
"""

import torch.nn as nn


def apply_patch() -> bool:
    """Add base_layer attribute delegation to VocabParallelEmbeddingWithLoRA.

    Returns True if the delegation is in place (including if already applied).
    """
    from vllm.lora.layers import VocabParallelEmbeddingWithLoRA as _VE

    if getattr(_VE, "_nvfp4_tied_embed_delegation", False):
        return True

    def __getattr__(self, name):  # only invoked on a normal-lookup miss
        # Honor nn.Module's own resolution first (parameters/buffers/submodules).
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError:
            pass
        # Fall back to the wrapped base layer for anything the wrapper itself
        # does not define (quant_method, weight, shard_indices, org_vocab_size,
        # ...). Guard against recursion on `base_layer` itself.
        base = self.__dict__.get("_modules", {}).get("base_layer")
        if base is not None and name != "base_layer":
            return getattr(base, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r} "
            "(no base_layer delegation available)"
        )

    _VE.__getattr__ = __getattr__
    _VE._nvfp4_tied_embed_delegation = True
    return True
