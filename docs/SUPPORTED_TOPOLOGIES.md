# Supported NVFP4 checkpoint topologies (v1)

"Any NVFP4 model" is the goal; this document is the precise contract of what
the stack supports TODAY, so that a checkpoint outside the contract fails
early with a named assumption instead of producing a silently-partial run.
Run [`scripts/inspect_nvfp4_checkpoint.py`](../scripts/inspect_nvfp4_checkpoint.py)
on any checkpoint to evaluate it against this contract.

## Quantized linear storage

Two on-disk encodings of NVFP4 are supported, detected per module from the
safetensors index:

| | compressed-tensors (RedHatAI) | NVIDIA ModelOpt |
|---|---|---|
| packed weight | `<m>.weight_packed`, uint8 `(out, in/2)` | `<m>.weight`, uint8 `(out, in/2)` |
| group scale | `<m>.weight_scale`, fp8_e4m3fn `(out, in/16)` | `<m>.weight_scale`, fp8_e4m3fn `(out, in/16)` |
| per-tensor scale | `<m>.weight_global_scale`, fp32, stored as a divisor | `<m>.weight_scale_2`, fp32 |

Common requirements for both:

* group size 16, last-dimension grouping
* standard E2M1 nibble packing, low-nibble-first
* `in_features` even and divisible by 16

FP8 per-tensor modules (`.weight` in fp8_e4m3fn + scalar `.weight_scale`,
no `.weight_scale_2`) are recognized but NOT trainable: the loader dequantizes
them to frozen BF16. Targeting them is a hard error unless
`--allow-fp8-targets` is passed.

## Model families

Family knowledge (safetensors-to-in-memory key translation, PEFT scoping,
multimodal-tower skip lists, fused-MoE class names) lives in ONE registry:
[`nvfp4_lora/families.py`](../nvfp4_lora/families.py). Current entries:

| `model_type` | auto class | attention | routed experts | status |
|---|---|---|---|---|
| `qwen3_5_moe` / `qwen3_5_moe_text` | causal LM | NVFP4 q/k/v/o on full-attention layers (native LoRA); GatedDeltaNet linear-attention layers BF16 | per-expert CT keys, fused-3D in memory (`Qwen3_5MoeExperts`) | trained + merged + served end-to-end |
| `mistral3` / `mistral4` | image-text-to-text (vision tower frozen + unmaterialized) | MLA attention BF16 (PEFT LoRA) | per-expert CT keys, fused-3D in memory (`Mistral4NaiveMoe`) | trained end-to-end |
| `nemotron_h` (Nemotron-3 Nano/Super) | causal LM | BF16/FP8 (not LoRA-targeted) | per-expert ModelOpt keys, per-expert in memory (no fused-3D container; `st_to_model`/`expert_prefix`/`moe_experts_class` are None and the loader's dynamic prefix heuristic applies, since Nano materializes `backbone.*` but Super `model.*`) | unified trainer (validated: Nano dry-run + 3-step smoke); `train/*.py` remain the frozen v1.0 measurement-run path |

A `model_type` outside the registry is a hard error naming the registry file.

## Fused-3D MoE contract

For families using `NVFP4Experts3D`, the checkpoint must satisfy:

* routed experts stored as per-expert 2D tensors, in either key naming:
  compressed-tensors
  `<moe>.<i>.{gate_proj,up_proj,down_proj}.{weight_packed,weight_scale,weight_global_scale}`
  or ModelOpt
  `<moe>.<i>.{gate_proj,up_proj,down_proj}.{weight,weight_scale,weight_scale_2}`
  (the trainer probes the index and picks the format; ModelOpt naming is
  CPU-parity-tested but no public fused-MoE checkpoint ships it yet)
* the in-memory HF module class matches the registry's `moe_experts_class`
  and exposes `num_experts` / `hidden_dim` / `intermediate_dim`

Per-expert gate/up per-tensor scales may be equal or differ. Equal scales
(every public CT checkpoint validated so far: 12288/12288 experts on both
Qwen3.5-122B releases) use the fused gate_up fast path, one dequant + one
bmm per K-batch. Differing scales select split gate/up storage at load
(probed from the shards in about a second), which is exact but costs two
dequant+bmm pairs for the gate_up projection.

## Trainability rules (enforced before load)

For each `--target-modules` suffix, every matching module in the index is
classified individually by its runtime form: NVFP4 -> native LoRA; BF16 ->
PEFT-wrappable nn.Linear; FP8 per-tensor -> dequantized to a frozen BF16
nn.Linear, which PEFT can also wrap. A suffix with any NVFP4 modules is
"native"; a suffix with none (only BF16 and/or FP8) is "peft". The run
proceeds only when:

* every target suffix matches at least one module
* native and PEFT suffixes are not mixed in one run
* a native suffix is not also partly BF16 (those BF16 instances would not
  train natively; override `--allow-partial-targets`)
* a native suffix is not also partly FP8 (those stay frozen in a native run;
  override `--allow-fp8-targets`). FP8 under a PEFT suffix is fine — PEFT
  wraps the frozen BF16 Linear and trains it.

The verdict plus the full inventory is persisted to
`<output_dir>/target_coverage.json`.

## Loading rules (enforced at load)

* every on-disk tensor must map to a model path, except prefixes on the
  family's `skip_st_prefixes` allowlist (vision towers, projectors, MTP
  speculation heads)
* after loading, no parameter or buffer may remain on the meta device, except
  prefixes on the family's `meta_allowed_prefixes` allowlist (frozen
  multimodal towers)
* `--permissive-load` downgrades both to warnings, for bring-up only

## Known out-of-contract layouts

Checkpoints that will be rejected today, by design:

* fused-3D expert storage on disk (single `(num_experts, ...)` tensors
  rather than per-expert keys)
* NVFP4 with group size other than 16
* MoE module classes not named in the registry (`moe_experts_class`)
* `model_type` values without a registry entry

Each rejection message names the assumption that broke and where to add the
support. If you hit one with a public checkpoint, please open an issue with
the inspector's `--json` output attached.
