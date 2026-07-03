# vLLM patches for Nemotron-3-Super-NVFP4 on DGX Spark

Monkey-patches that improve vLLM 0.21.0 behavior when serving
`Nemotron-3-Super-120B-A12B-NVFP4` on a single DGX Spark (GB10, sm_121,
128 GB unified LPDDR5x).

## Files

### `marlin_repack_patch.py`
Replaces `vllm.model_executor.layers.quantization.utils.marlin_utils_fp4.prepare_nvfp4_moe_layer_for_marlin`
with a chunked + preallocated version. The upstream implementation builds a
list of 512 per-expert tensors and `torch.cat`s them, briefly holding both
the list and the cat output simultaneously. The patched version preallocates
the destination tensor once and writes per-expert results in-place, capping
per-call live memory.

Working set drops from ~9-12 GB per repack call to ~6 GB by avoiding the
upstream's `list + torch.cat` accumulation pattern, freeing ~3-6 GB during
Marlin compile for Super-120B. It does NOT fix Marlin's structural memory
pressure on Spark - see [../../docs/PERFORMANCE_ROADMAP.md](../../docs/PERFORMANCE_ROADMAP.md)
for the full backend story. Ship as a clean upstream PR titled e.g.
*"Avoid intermediate stacked-expert list during NVFP4 MoE Marlin preparation"*.

### `sitecustomize.py`
Python loads this automatically when present on `sys.path`. We use it to
ensure `marlin_repack_patch.apply_patch()` runs in BOTH the parent APIServer
process AND in vLLM's spawned `VLLM::EngineCore` subprocess (which uses
`multiprocessing.spawn` and does NOT inherit monkey-patches from the parent).

### `cohere_tied_embedding_lora.py`
Opt-in via `VLLM_PATCH_TIED_EMBED_LORA=1`. Makes **tied-embedding** models
(Cohere / Command-R / Command-A, `tie_word_embeddings`) serveable with
`--enable-lora`. Those models compute logits THROUGH the input embedding
(`commandr.py:compute_logits` -> `logits_processor(self.model.embed_tokens, ...)`).
With LoRA enabled vLLM wraps `embed_tokens` in `VocabParallelEmbeddingWithLoRA`,
which parks the real layer under `self.base_layer` and does not delegate
attribute access, so the logits path crashes with
`AttributeError: 'VocabParallelEmbeddingWithLoRA' object has no attribute 'quant_method'`.
The patch adds a `__getattr__` that falls back to `base_layer` for any attr the
wrapper does not define (`quant_method`, `weight`, `shard_indices`, ...), so
logits compute as correct tied-embedding logits. Repo adapters target only
attention + MLP (no embedding-LoRA), so no adapter delta is lost. Untied-embedding
models never take this path and are unaffected. Validated end to end on
Command-A (cohere2, 111B CT-NVFP4); see
[../../results/cross_arch/command_a_generic_serve/README.md](../../results/cross_arch/command_a_generic_serve/README.md).

## Usage

```bash
PYTHONPATH=/path/to/this/vllm_patches \
    vllm serve $MODEL_DIR \
        --moe-backend marlin \
        # ...your usual flags...
```

Python's `site` module will pick up `sitecustomize.py` from `PYTHONPATH`
and apply the patch in every spawned process. To verify it's active, look
for the stderr line `[sitecustomize pid=<pid>] applied marlin_repack_patch`
on each process.

The patch is idempotent - applying it twice is harmless.
