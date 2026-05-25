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
