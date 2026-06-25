# Porting another NVFP4 family

Bring-your-own-NVFP4-checkpoint guide for the unified trainer
([`scripts/train_nvfp4_lora.py`](../scripts/train_nvfp4_lora.py)). The core
stack (dequant kernel, `NVFP4LoRALinear`, fused-3D MoE container) is
family-agnostic; porting a new model is a loader change, not a kernel change.
This walks the decision tree, the exact integration points, what the LoRA-mode
detection does to your adapter, and a validation ladder. The GB10
unified-memory failure signatures you will hit along the way are in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md#gb10-unified-memory-failure-signatures).

## 1. Read your checkpoint first

Start with the inspector; it answers everything in this section in one command
(seconds, no GPU, reads only `config.json` + the safetensors index):

```bash
python scripts/inspect_nvfp4_checkpoint.py /path/to/model \
    --target-modules q_proj,k_proj,v_proj,o_proj
```

It reports the quant format per module, per-suffix coverage with layer-level
gaps, the MoE topology and whether the fused-3D path supports it, and the
exact LoRA mechanism your target set would get (or the precise rejection
reason). Add `--deep` to also read the per-expert gate/up per-tensor scales
from the shards and report whether the fused gate_up fast path applies or
split storage will be selected, and `--json` for machine-readable output. The checkpoint-layout contract the inspector checks against is
[SUPPORTED_TOPOLOGIES.md](SUPPORTED_TOPOLOGIES.md).

The rest of this section explains what the inspector is looking at, for when
you need to make the call manually. Three properties of the checkpoint decide
how much work the port is. All three are answerable from `config.json` plus
`model.safetensors.index.json` without loading anything.

### Quant format: compressed-tensors vs ModelOpt

Look at `config.json` `quantization_config` (and the sibling
`hf_quant_config.json` if present):

- **compressed-tensors** (RedHatAI checkpoints): `quant_method` is
  `compressed-tensors`. On disk each quantized Linear stores
  `<module>.weight_packed` (uint8), `<module>.weight_scale` (fp8 group scale),
  and `<module>.weight_global_scale` (fp32 per-tensor scale).
- **NVIDIA ModelOpt**: `quant_method` is `modelopt`. On disk each quantized
  Linear stores `<module>.weight` (uint8), `<module>.weight_scale` (fp8 group
  scale), and `<module>.weight_scale_2` (fp32 per-tensor scale).

You do not have to trust `quantization_config`. The loader decides per module
from the index keys via `dequant.format_for_record`: `.weight_packed` present
means compressed-tensors, bare `.weight` (alongside a `.weight_scale`) means
ModelOpt. Both formats unpack with the same kernel; only the key names differ.

### Is attention quantized or BF16?

Grep the index (`model.safetensors.index.json`) for an attention projection:

```bash
python -c "import json,sys; wm=json.load(open(sys.argv[1]))['weight_map']; \
  print([k for k in wm if 'self_attn.q_proj' in k][:6])" \
  /path/to/model/model.safetensors.index.json
```

- If you see `...self_attn.q_proj.weight_packed` (or, for ModelOpt,
  `...self_attn.q_proj.weight` plus `...self_attn.q_proj.weight_scale`), then
  attention is NVFP4-quantized. LoRA on `q_proj,k_proj,v_proj,o_proj` will be
  **native** (baked into `NVFP4LoRALinear`).
- If you see only `...self_attn.q_proj.weight` with **no** `.weight_scale`
  sibling, attention is BF16. PEFT can wrap those targets directly.

This is exactly the split between the two community checkpoints:
Qwen3.5-122B has NVFP4 attention on its 12 full-attention layers (native LoRA),
while Mistral-Small-4 keeps MLA attention (`q_b_proj`, `kv_b_proj`, `o_proj`)
in BF16 (PEFT wrapping). See section 3 for what that means downstream.

### Is the MoE fused-3D or per-expert?

Check how routed experts appear in the index:

- **Per-expert** keys (`...mlp.experts.0.gate_proj.weight_packed`,
  `...experts.1.gate_proj...`, one set per expert) with an in-memory module
  class that holds a single fused tensor across all experts. This is the
  fused-3D case: the on-disk per-expert keys get assembled into the
  `(num_experts, ...)` buffers of `NVFP4Experts3D`. Both Qwen3.5 and
  Mistral-Small-4 are here, and both need a `family_class_names` entry
  (section 2c).
- If the experts are already plain NVFP4 Linears in the model graph (the
  Nemotron-3 routed-MoE layout, where each expert `up_proj`/`down_proj` is a
  normal `nn.Linear`), they are handled by `replace_nvfp4_modules` like any
  other NVFP4 Linear and need no fused-3D container.

The fingerprint is in the in-memory module class name, not the index: if the
HF model class fuses experts into one 3D parameter, you need the fused-3D path.

## 2. The integration point: one registry entry

All family knowledge lives in ONE place:
[`nvfp4_lora/families.py`](../nvfp4_lora/families.py). The trainer, the
loader's key translator, the fused-3D MoE replacer, the inspector and the
merge scripts all read the same entry, so adding a family is adding ONE dict
(plus tests). An unknown `model_type` raises with the list of known families.

```python
FAMILIES = {
    "your_model_type": {
        "auto_class": "causal_lm",          # or "image_text_to_text"
        "expert_prefix": ("model.", "model.language_model."),  # (in_memory, safetensors)
        "peft_scope": r"^model\.layers\.",  # regex anchoring PEFT targets to the text backbone
        "freeze": (),                        # submodules of model.model to freeze (towers)
        "skip_st_prefixes": ("model.visual.",),  # on-disk prefixes never loaded (towers, MTP)
        "st_to_model": (("model.language_model.", "model."),),  # key rewrite rules, first match wins
        "meta_allowed_prefixes": (),         # in-memory prefixes allowed to stay on meta
        "moe_experts_class": "YourFusedExpertsClass",  # fused-3D HF class name, or None
    },
}
```

- **`auto_class`**: `"causal_lm"` builds with `AutoModelForCausalLM`,
  `"image_text_to_text"` with `AutoModelForImageTextToText`. Use the latter
  for a multimodal wrapper (Mistral-Small-4's
  `Mistral3ForConditionalGeneration`); the text-only causal LM is still what
  gets trained, but the wrapper class controls how `from_config` builds the
  graph.
- **`expert_prefix`**: `(in_memory_prefix, safetensors_prefix)` for the
  routed-expert module paths. `load_model` uses this to map each in-memory
  `NVFP4Experts3D` module name to its on-disk key prefix before
  `assemble_nvfp4_experts3d_batched`. For Qwen3.5 the in-memory tree is
  `model.layers...` while on disk experts live under
  `model.language_model.layers...`, so the pair is
  `("model.", "model.language_model.")`.
- **`peft_scope`**: a regex prefix that anchors PEFT `target_modules` to the
  text backbone, so a bare suffix can never match a multimodal tower (whose
  weights may sit on meta). Only used when `lora_mode == "peft"`;
  `attach_peft_lora` composes it as
  `peft_scope + r".*\.(<suffixes>)$"`.
- **`freeze`**: submodule attribute names under `model.model` to set
  `requires_grad = False` after load (text-only training). Empty for a
  text-only checkpoint; `("vision_tower", "multi_modal_projector")` for a VLM.

Note the registry carries both the outer multimodal `model_type` and the
inner text-only one (`qwen3_5_moe` and `qwen3_5_moe_text`; `mistral3` and
`mistral4`). `AutoModelForCausalLM.from_config` instantiates the text-only
variant, whose `config.model_type` differs from the outer wrapper, so register
both keys with identical values.

### Key translation: `skip_st_prefixes` + `st_to_model`

The loader's `make_key_translator` builds the safetensors-to-in-memory
translation from the registry's `skip_st_prefixes` (return `None`, never
load) and `st_to_model` rewrite rules (ordered, first match wins, unmatched
keys pass through verbatim). You no longer write a code branch for a new
family; you declare the rules. What the rules must accomplish:

- **Rewrite the backbone prefix.** Strip the on-disk prefix and add the
  in-memory one. Qwen3.5 maps `model.language_model.layers.X.*` to
  `model.layers.X.*`. Mistral-Small-4 maps `language_model.model.layers.X.*`
  to `model.language_model.layers.X.*` and `language_model.lm_head.weight` to
  the top-level `lm_head.weight`.
- **Skip what is not part of the text model you are training.** Return `None`
  for:
  - **Vision towers**: Qwen3.5 returns `None` for `model.visual.*`;
    Mistral-Small-4 returns `None` for `vision_tower.*` and
    `multi_modal_projector.*`. These are not in the text-only graph, and
    `freeze` would not even reach them. Skipping keeps them out of the
    page-cache assembly entirely.
  - **MTP layers**: the Nemotron fallback returns `None` for `mtp.*`
    (Multi-Token Prediction speculation layers used only by vLLM speculative
    decoding, never trained). `load_non_nvfp4_weights` counts and reports the
    skipped tensors.
- **Pass everything else through.** Unmatched keys pass through verbatim,
  which handles keys like `lm_head.*` that share naming between disk and
  memory.

Loading is strict by default: an on-disk tensor whose translated path does not
exist in the model is a hard error naming the key (add the prefix to
`skip_st_prefixes` if it is genuinely not part of the training graph), and any
parameter left on the meta device after load fails unless its prefix is in
`meta_allowed_prefixes`. Use the trainer's `--permissive-load` only while
working out a new family's rules.

The Nemotron fallback at the bottom of `make_key_translator` is a heuristic
(`named_children()` scan for a child with `.layers`). It raises if it cannot
find a single backbone prefix or a `.layers` child, with the message telling
you to add a registry entry. Do not rely on it for a new non-Nemotron family.

### `moe_experts_class` (fused-3D only)

If section 1 told you the MoE is fused-3D, set `moe_experts_class` in the
registry entry to the **in-memory module class name** the HF model uses for
its fused expert block (the `module.__class__.__name__` that
`replace_moe_experts_with_nvfp4_3d` matches on). The replacement reads
`num_experts`, `hidden_dim`, and `intermediate_dim` off the old module and
swaps in `NVFP4Experts3D`. A family without the field raises
`replace_moe_experts_with_nvfp4_3d does not have a fused-3D MoE class...`.

`NVFP4Experts3D` expects the standard gate/up/down per-expert layout in either
compressed-tensors or ModelOpt key naming (the trainer probes the index and
selects the format). When gate and up share one per-tensor scale per expert,
they are stored fused along the output axis as `[gate, up]` (matching the
reference forward's `.chunk(2, dim=-1)`, the fast path); when the scales
differ, the trainer selects split gate/up storage automatically.
`assemble_nvfp4_experts3d_batched` (and the validating
`assemble_nvfp4_experts3d_from_safetensors_keys`) enforce the shapes; if your
checkpoint differs structurally (e.g. a w1/w3 naming), you will need to adapt
the assembler too.

If your routed experts are plain NVFP4 Linears in the graph (Nemotron layout),
skip this entirely; `replace_nvfp4_modules` handles them.

## 3. What LoRA-mode detection does to your adapter

The trainer does not let you choose the LoRA mechanism. `detect_lora_mode`
(backed by `loader.decide_lora_mode`) classifies EVERY module matching each
`--target-modules` suffix individually from the index: compressed-tensors
NVFP4 (`.weight_packed`), ModelOpt NVFP4 (`.weight` + `.weight_scale` +
`.weight_scale_2`), FP8 per-tensor (`.weight` + `.weight_scale` only), or
plain BF16:

- **All target modules NVFP4: `native`.** LoRA is baked into `NVFP4LoRALinear`
  at load (PEFT cannot wrap a packed NVFP4 weight). `r`, `alpha`, and
  `dropout` are passed straight into the module replacement.
- **All target modules BF16: `peft`.** Standard PEFT wrapping with the
  family-scoped `target_modules` regex from `peft_scope`. This is the
  BF16-attention recipe (Mistral-Small-4 MLA targets).
- **Anything else: hard error with the exact inventory.** Native and PEFT
  suffixes in one run, a suffix matching nothing, or a suffix quantized in some
  layers but BF16 in others (override: `--allow-partial-targets`) all fail
  before any weight is read. The full per-suffix inventory is written to
  `<output_dir>/target_coverage.json` on every run. (FP8-landing suffixes are
  no longer an error: FP8 trains natively via `FP8LoRALinear`.)

This choice changes nothing about the on-disk adapter you ship. Both paths
write `adapter_model.safetensors` with PEFT-style keys
(`base_model.model.<module>.lora_{A,B}.weight`) plus an `adapter_config.json`,
through the atomic save (`_save_adapter_atomic`). The native path constructs
those keys directly by walking `NVFP4LoRALinear` modules with `r > 0`; the PEFT
path goes through `get_peft_model_state_dict`. Either way the merge step
(`scripts/merge_lora_into_nvfp4.py`) and any downstream PEFT consumer see the
same format. The practical consequence of the mode is the load path and memory
profile, not the artifact.

One related note carried over from Nemotron: FP8 (not NVFP4) modules used to be
demoted to frozen BF16 at load time. In the unified trainer FP8 targets now
train natively via `FP8LoRALinear` (frozen FP8 base + bf16 LoRA), so an
FP8-landing suffix is a native run with no extra flag.

## 4. Validation ladder

Do not jump to a full run. The load stages and the first forward/backward are
where GB10 unified-memory bugs surface, and they surface fast.

### Step 0: dry-run preflight

`scripts/train_nvfp4_lora.py` takes `--dry-run`: it loads the model exactly as
a real run would (`load_model` + LoRA attach + gradient checkpointing +
optimizer state), runs one synthetic forward+backward at
`(batch_size, max_length)`, logs a memory reading, and exits without saving any
adapter. `--train-file` is not required in this mode. This is the cheapest way
to catch an out-of-memory: per the flag's own help, it surfaces an OOM in
roughly 12 minutes (one load plus one worst-case forward/backward) instead of
failing mid-run. Watch the `dry_run_ok` line for `post_load`, `post_backward`,
and `cuda_max_allocated_gb`.

```bash
python -u scripts/train_nvfp4_lora.py \
    --model-dir /path/to/your-NVFP4-checkpoint \
    --target-modules q_proj,k_proj,v_proj,o_proj \
    --max-length 2048 --output-dir /tmp/dryrun --dry-run
```

(The `--dry-run` flag is landing in the trainer in a parallel change; if your
checkout predates it, run the smoke below instead, which exercises the same
load path with three real steps.)

### Step 1: smoke (8 train examples, 3 steps)

```bash
python -u scripts/train_nvfp4_lora.py \
    --model-dir /path/to/your-NVFP4-checkpoint \
    --target-modules q_proj,k_proj,v_proj,o_proj \
    --max-train-examples 8 --max-val-examples 4 --max-steps 3 \
    --eval-every 0 --checkpoint-every 0 --output-dir /tmp/smoke
```

What healthy load-stage logs look like. The loader prints a `[load-mem]` line
after each stage (`memory_snapshot`): `post-meta-build`, `post-moe-replace`,
`post-linear-replace`, `post-expert-assembly`, `post-non-nvfp4-load`,
`post-workspaces`, `post-move-loop`. On a healthy load:

- `process_rss_gb` does **not** jump by a weight-sized amount at
  `post-moe-replace`. If it does (tens of GB), the fused expert container
  allocated on CPU; this is failure (a) in TROUBLESHOOTING.
- The `move-loop relocated NGB from CPU` WARNING does not fire (it triggers
  above 1 GB and means a stage placed weight-sized buffers on the wrong
  device).
- `dropped shard page cache: cuda_free X -> Y` shows `Y` substantially higher
  than `X` (tens of GB reclaimed). If `cuda_free` stays near 1-2 GB after the
  drop, that is failure (b).
- `cuda_free` after the load accounts for the model size; for a ~76 GB model on
  a 131 GB box you want roughly 50 GB free, not 1-2 GB.

Confirm `lora_attached` reports a non-zero `native_modules` (native mode) or a
non-zero `trainable` count, then watch three `train_step` lines land with a
finite loss.

### Step 2: full run

Only after the smoke is clean and the kernel ring stayed quiet
(`journalctl -k -f -g 'NVRM|Xid'`), launch the full run from a clean boot. Keep
the per-stage `[load-mem]` logs in your output; they are the audit trail for
any later OOM. For long-context configurations and the certified flag set, see
the README's long-context section.

## 5. Worked example: adding Qwen3.5 (retrospective)

Qwen3.5-122B-A10B is a compressed-tensors checkpoint with a hybrid backbone (36
GatedDeltaNet linear-attention layers, 12 full-attention layers with NVFP4
q/k/v/o) and fused-3D routed experts. The whole port is one registry entry in
[`nvfp4_lora/families.py`](../nvfp4_lora/families.py) (two keys, outer and
text-only, identical value):

```python
"qwen3_5_moe": {
    "auto_class": "causal_lm",
    "expert_prefix": ("model.", "model.language_model."),
    "peft_scope": r"^model\.layers\.",
    "freeze": (),
    "skip_st_prefixes": ("model.visual.",),
    "st_to_model": (("model.language_model.", "model."),),
    "meta_allowed_prefixes": (),
    "moe_experts_class": "Qwen3_5MoeExperts",
},
"qwen3_5_moe_text": { ... identical ... },  # config.model_type after AutoModelForCausalLM.from_config
```

Reading the entry top to bottom: `auto_class` is `causal_lm` (Qwen3.5 trains
as a plain causal LM, no tower to freeze, so `freeze` and
`meta_allowed_prefixes` are empty). `expert_prefix` reflects that experts live
under `model.language_model.layers...` on disk but `model.layers...` in
memory. `skip_st_prefixes` drops the vision tower (`model.visual.*`) from
text-only training, `st_to_model` rewrites the backbone prefix and lets
`lm_head.*` pass through, and `moe_experts_class` names the fused expert
block class the MoE replacer matches on.

**Mode detection result.** Targeting `q_proj,k_proj,v_proj,o_proj` resolves to
`native`: the 12 full-attention layers store those projections as NVFP4, so
they are detected as quantized and baked into `NVFP4LoRALinear`. PEFT is not
used. (The GatedDeltaNet layers have no standard q/k/v/o to target.)

**GB10 caveats for this family specifically.** Qwen3.5's GatedDeltaNet layers
need `flash-linear-attention` (pinned to 0.4.2) and `causal-conv1d`; without
them transformers silently falls back to a much slower torch path. The 0.5.0
backward-kernel crash and the NVFP4-attention eval-cache spike are both
documented as failure signatures (c) and (d) in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md#gb10-unified-memory-failure-signatures).
