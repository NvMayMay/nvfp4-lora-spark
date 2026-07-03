# Generic-family onboarding, end to end: fine-tune AND serve an unregistered NVFP4 model

Evidence that the repo's core claim - LoRA fine-tune ANY NVFP4 model and serve
that fine-tune with the LoRA applied at runtime - holds on a model family the
repo has NO registry entry for, via the generic-family fallback
(`nvfp4_lora/families.synthesize_generic_family`) plus the runtime-LoRA serve
path.

Target: **Command-A-Reasoning-08-2025-NVFP4** (`cohere2` / `Cohere2ForCausalLM`,
~111B, compressed-tensors `nvfp4-pack-quantized`, 64-layer dense). Unregistered:
no `cohere2` entry in `FAMILIES`.

## 1. Fine-tune (generic family) - PASS (2026-07-03, DGX Spark GB10 / sm_121)

`scripts/train_nvfp4_lora.py --allow-unverified-family` on the flat cohere2
checkpoint:
- `model_type='cohere2'` detected as unregistered -> synthesized generic family
  (UNVERIFIED banner shown, as designed).
- Target coverage: q/k/v/o + gate/up/down each resolve to **64 `nvfp4_ct`
  modules** (all 64 layers) - the CT-NVFP4 loader classified every target
  correctly on an arch it had never seen.
- `lora_mode=native`, 448 modules wrapped, 228.6M trainable params, optimizer
  ready, training steps produce sensible loss. Strict-load passed (no missing /
  unexpected keys, no cohere2-specific tensor tripped anything).

Note: the trainer's assistant-span mask needs `--max-length` large enough to
include the answer span; Command-A + Spider rows run ~1.4-2k tokens (long schema
+ the `<|START_RESPONSE|>` reasoning wrapper), so `--max-length 1024` drops every
row (0 supervised tokens) while `2048` keeps them. Not a family issue; a length
knob.

## 2. Serve base + runtime-LoRA - PASS (with tied-embedding patch)

`vllm serve <base> --enable-lora --lora-modules cmdA-lora=<adapter>` (vLLM 0.22.1,
host venv):
- NVFP4 base loads and serves: `FlashInferCutlassNvFp4LinearKernel` active on
  sm_121, arch resolves to `Cohere2ForCausalLM`, `supports_lora=True`.
- Bare `--enable-lora` first crashed at `profile_run -> compute_logits`:
  `AttributeError: 'VocabParallelEmbeddingWithLoRA' object has no attribute
  'quant_method'`. Cohere ties embeddings and computes logits THROUGH
  `embed_tokens`; vLLM's LoRA embedding wrapper does not delegate attributes to
  its `base_layer`. Fixed by the opt-in
  `serve/vllm_patches/cohere_tied_embedding_lora.py`
  (`VLLM_PATCH_TIED_EMBED_LORA=1`), after which the server reaches
  `Application startup complete`.

### Adapter-applied check (decisive metric = the forward pass)

Teacher-forced sum log-prob of the gold Spider SQL (8-token answer span), same
prompt, base vs the served adapter:

| served model | sum log-prob(gold SQL) |
|---|---|
| `cmdA-base` | -28.80 |
| `cmdA-lora` | **-17.38** |
| **delta (lora - base)** | **+11.42 nats** |

The two differ (adapter is genuinely applied at runtime, not silently a no-op)
AND the delta is directionally correct (the Spider-trained LoRA raises the gold
SQL's likelihood). Adapter here is a tiny 2-step run - quality is not the point;
the point is that the fine-tune is served and active.

## Takeaway

A new model family costs ZERO registry code to fine-tune (generic fallback) and,
for a tied-embedding family, ONE opt-in serve patch to run runtime-LoRA. The
fine-tune half is arch-agnostic; the serve half is arch-gated by vLLM, and the
one gate hit here (tied-embedding logits) is now handled in-repo.
