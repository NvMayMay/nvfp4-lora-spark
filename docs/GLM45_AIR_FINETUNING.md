# GLM-4.5-Air (106B-A12B) NVFP4 LoRA: fine-tuning + serving runbook

End-to-end notes for fine-tuning GLM-4.5-Air in NVFP4 on a single GB10 DGX Spark
(130.7 GB unified memory) with the unified trainer, validated 2026-06-21/22.

## TL;DR recipe

```bash
# venv with transformers >=5.8 (has Glm4MoeForCausalLM) + liger-kernel
P=/path/to/.venvs/qwen-peft/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVFP4_EVAL_CACHE_GB=0   # don't build the 30 GB eval bf16 cache at long seq

$P -u scripts/train_nvfp4_lora.py \
  --model-dir /models/GLM-4.5-Air-106B-A12B-NVFP4 \
  --train-file train.chat.jsonl --val-file val.chat.jsonl \
  --output-dir adapters/glm45air_lora \
  --target-modules q_proj,k_proj,v_proj,o_proj \
  --max-length 8192 --fused-linear-ce \
  --epochs 2 --batch-size 1 --grad-accum 8 \
  --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
  --eval-every 50 --eval-subset 32 --checkpoint-every 50
```

Stop the co-tenant serve first (a vLLM serve holds ~112 GB UMA; GLM training needs ~74 GB).

## Model facts

- `model_type=glm4_moe`, arch `Glm4MoeForCausalLM`; 46 layers, hidden 4096, **vocab 151552**,
  128 routed experts (8/token) + 1 shared expert, `first_k_dense_replace=1` (layer 0 dense).
- NVFP4 checkpoint: compressed-tensors `nvfp4-pack-quantized`, ~58 GB / 13 shards.
- Attention has q/k/v biases (o_proj has none); attention projections are NVFP4-quantized
  (native LoRA path); router gate (`mlp.gate`) + norms + embed + lm_head are BF16.

## The one gotcha: experts are FUSED-3D in memory, not per-expert

The checkpoint stores experts per-expert (`model.layers.N.mlp.experts.E.{gate,up,down}_proj`),
but transformers materializes them as a **fused-3D block** `Glm4MoeNaiveMoe` (`gate_up_proj`
+ `down_proj` batched over experts) - structurally identical to `Mistral4NaiveMoe`. So this is
the fused-3D path, NOT the per-expert/nemotron path. A family entry with `moe_experts_class=None`
leaves 90 expert tensors (45 MoE layers x 2) stranded on the meta device and fails
`assert_no_meta_tensors`.

The working `nvfp4_lora/families.py` entry:

```python
"glm4_moe": {
    "auto_class": "causal_lm",
    "expert_prefix": ("model.", "model."),   # text-only: same backbone prefix on disk and in memory
    "peft_scope": r"^model\.layers\.",
    "freeze": (),
    "skip_st_prefixes": (),                   # no MTP/visual tensors in this checkpoint
    "st_to_model": None,                      # identity translation via the loader's dynamic heuristic
    "meta_allowed_prefixes": (),
    "moe_experts_class": "Glm4MoeNaiveMoe",   # fused-3D -> NVFP4Experts3D; split_gate_up_scales auto-probed
},
```

No new loader/experts code is needed: `replace_moe_experts_with_nvfp4_3d` +
`assemble_nvfp4_experts3d_batched` are generic, and `split_gate_up_scales` is auto-detected
from the shards (False for this checkpoint).

## `--fused-linear-ce` (FLCE)

At seq 8192 x vocab 151552 the full logits tensor + its fp32 upcast + grad is the largest
train-time spike (~10 GB). `--fused-linear-ce` binds Liger
`liger_kernel.transformers.model.glm4.lce_forward` onto the model: chunked logits, never the
full `seq x vocab` tensor. **Numerically exact** (dry-run loss matched the non-FLCE path to 3e-4).
Bind ONLY `lce_forward`; do NOT call `apply_liger_kernel_to_glm4` (it rewrites MoE MLPs into
dense SwiGLU and corrupts the NVFP4 experts).

8192 fits even without FLCE (~83 GB peak); FLCE drops it to ~74 GB and makes 16384 (~85 GB) easy.

## Context length: measure the data

Full context/memory matrix is in [CONTEXT_FIT_MATRIX.md](CONTEXT_FIT_MATRIX.md). For the ICH v4_1
corpus, token lengths are p50 4406 / p99 6748 / max 8485, so **8192 captures 99.7% of examples
fully** vs only 28% at 4096. 16384 fits but buys ~nothing for ~2x the per-step compute. Use
`--dry-run` to map memory before committing to a long run; it loads exactly like a real run,
does one forward+backward at `(batch, max_length)`, logs `cuda_max_allocated_gb`, and exits.

## Throughput note

This run is **dequant-bound, not token-bound**: dequantizing the 17.6k frozen NVFP4 modules per
micro-step dominates, so 8192 was only ~8% slower per update than 4096 (~270 s vs 249 s), not 2x.
The validated run did 2 epochs (191 updates) in 15.8 h, best val 0.8433
(curve 0.9029 -> 0.8579 -> 0.8460 -> 0.8433). Load is ~470 s (module replace + expert assembly).

## Serving (NVFP4 base + attention-only LoRA)

The adapter targets only dense attention (q/k/v/o), so it serves via the same
`attention_only_lora_cutlass_moe` patch used for Qwen NVFP4 MoE: the routed experts stay on the
CUTLASS NVFP4 kernel with LoRA disabled, and dense-attention LoRA is applied via punica. GLM is a
text-only causal LM with `model.layers.*` keys, so the Qwen multimodal key rewrite is a no-op here.
See `serve/run_glm45_air_nvfp4_dynamic_lora.sh`.
