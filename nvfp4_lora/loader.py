"""Loader: walk a Nemotron-3 NVFP4 checkpoint and replace target Linears with NVFP4LoRALinear.

1. Use `accelerate.init_empty_weights()` to construct the model architecture without
   allocating any weight memory.
2. Walk the model and replace each NVFP4-quantized Linear with our `NVFP4LoRALinear`:
   - target_modules (e.g., q_proj, v_proj) → NVFP4LoRALinear with r>0 (LoRA-trainable)
   - other NVFP4 modules → NVFP4LoRALinear with r=0 (frozen)
3. For non-NVFP4 modules (norms, embeddings, conv1d, Mamba state matrices, lm_head),
   load their on-disk bf16 weights normally into the meta-tensors via direct assignment.

This keeps Super-120B resident-memory near the 75 GB NVFP4 storage floor (no bf16 shadow
of any NVFP4-quantized weight).

Identifying NVFP4 vs non-NVFP4 modules:
- A module is NVFP4 iff it has `{module_name}.weight_scale` in the safetensors index.
- The exclude list in `hf_quant_config.json` is a sanity check, not authoritative.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import safetensors

from .linear import NVFP4LoRALinear


# --------------------------------------------------------------------------------------
# Inventory: which modules are NVFP4-quantized vs plain bf16
# --------------------------------------------------------------------------------------

def list_quantized_modules(model_dir: str | Path) -> set[str]:
    """Return the set of NVFP4-quantized module names by scanning the safetensors index.

    A module is NVFP4 iff it has a `.weight_scale` tensor (the per-group FP8 scales).
    Keys are returned VERBATIM from the safetensors index; the caller is responsible
    for translating to model-attribute paths via `make_key_translator`.
    """
    idx_path = Path(model_dir) / "model.safetensors.index.json"
    with open(idx_path) as f:
        idx = json.load(f)
    wm = idx["weight_map"]
    out = set()
    for key in wm.keys():
        if key.endswith(".weight_scale"):
            out.add(key[: -len(".weight_scale")])
    return out


def make_key_translator(model: nn.Module, model_dir: str | Path):
    """Build a function that maps a safetensors key to the corresponding model attribute path.

    Nemotron-3 family checkpoints use `backbone.X` in safetensors, but different model classes
    use different in-memory submodule names:
      - Nano-30B-A3B-NVFP4: `self.backbone = NemotronHModel(...)` → in-memory path `backbone.X`
      - Super-120B-A12B-NVFP4: `self.model = NemotronHModel(...)` → in-memory path `model.X`

    The translator also skips MTP (Multi-Token Prediction speculation) layers, which exist in
    safetensors as `mtp.X` but are a separate inference-only architecture vLLM uses for
    speculative decoding and that we never train.

    Returns (translate, skip_mtp_count) - translate(key) returns the model path, or None
    if the key should be skipped (e.g. MTP).
    """
    # Detect safetensors top-level prefix used by the sub-model
    idx = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())
    safetensors_prefixes = {k.split(".", 1)[0] for k in idx["weight_map"].keys()}
    # The submodel prefix is whichever appears in keys like "X.layers.0..."
    candidates = [p for p in safetensors_prefixes if p not in ("lm_head", "mtp")]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not determine safetensors submodel prefix; candidates={candidates}"
        )
    safetensors_prefix = candidates[0]

    # Detect model's in-memory submodel name (the one whose class has `.layers`)
    model_prefix = None
    for n, c in model.named_children():
        if hasattr(c, "layers"):
            model_prefix = n
            break
    if model_prefix is None:
        raise RuntimeError("Could not find a child module with `.layers` (NemotronHModel)")

    def translate(key: str) -> Optional[str]:
        if key.startswith("mtp."):
            return None  # skip MTP speculation layers
        if key.startswith(safetensors_prefix + "."):
            return model_prefix + "." + key[len(safetensors_prefix) + 1:]
        # Other keys (lm_head, etc.) pass through verbatim
        return key

    return translate, safetensors_prefix, model_prefix


def load_tensor(model_dir: str | Path, key: str, weight_map: dict) -> Optional[torch.Tensor]:
    """Load a single tensor from its shard. Returns None if key not present."""
    if key not in weight_map:
        return None
    shard_path = Path(model_dir) / weight_map[key]
    with safetensors.safe_open(str(shard_path), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


# --------------------------------------------------------------------------------------
# Module replacement
# --------------------------------------------------------------------------------------

def _get_parent(model: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    """Resolve 'a.b.c' → (model.a.b, 'c'). Used to swap a child module on its parent."""
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def replace_nvfp4_modules(
    model: nn.Module,
    model_dir: str | Path,
    target_lora_suffixes: Sequence[str],
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, int]:
    """Replace every NVFP4 nn.Linear in `model` with an `NVFP4LoRALinear` instance.

    - Modules whose last name component is in `target_lora_suffixes` get LoRA at the
      requested rank (trainable).
    - Other NVFP4 modules get r=0 (frozen-only) but still as NVFP4LoRALinear so the
      dequant happens through our path (not via uninitialized memory).

    Returns: dict with counts {"lora": N_lora, "frozen": N_frozen}.
    """
    model_dir = Path(model_dir)
    nvfp4_module_names_st = list_quantized_modules(model_dir)  # safetensors-side names
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    target_set = set(target_lora_suffixes)

    # Translate safetensors keys -> model attribute paths (handles Nano `backbone.` vs Super `model.`)
    translate, st_prefix, model_prefix = make_key_translator(model, model_dir)
    # Map model-side path -> safetensors-side name, so we can look up tensors via safetensors keys
    model_to_st: dict[str, str] = {}
    for st_name in nvfp4_module_names_st:
        m_name = translate(st_name)
        if m_name is not None:
            model_to_st[m_name] = st_name
    print(f"  detected prefix: safetensors='{st_prefix}.', model='{model_prefix}.'")

    counts = {"lora": 0, "frozen_nvfp4": 0, "frozen_fp8": 0}

    # Snapshot the list first; mutating the tree during iteration confuses named_modules
    quantized_paths: list[tuple[str, nn.Linear, str]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in model_to_st:
            continue
        quantized_paths.append((name, module, model_to_st[name]))

    for name, orig, st_name in quantized_paths:
        suffix = name.split(".")[-1]
        is_lora_target = suffix in target_set

        # Probe the weight dtype to decide quant format: NVFP4 (uint8 packed + fp8 group + fp32 per-tensor)
        # vs FP8 per-tensor (fp8 weight + fp32 scalar scale). Both share `.weight_scale` so we can't
        # tell from the safetensors index alone.
        w_tensor = load_tensor(model_dir, f"{st_name}.weight", wm)
        w_scale = load_tensor(model_dir, f"{st_name}.weight_scale", wm)
        bias = load_tensor(model_dir, f"{st_name}.bias", wm)
        if w_tensor is None or w_scale is None:
            raise RuntimeError(f"Missing quantization tensors for safetensors module {st_name}")

        in_features = orig.in_features
        out_features = orig.out_features
        parent, attr = _get_parent(model, name)

        if w_tensor.dtype == torch.uint8:
            # ---- NVFP4 path ----
            w_scale_2 = load_tensor(model_dir, f"{st_name}.weight_scale_2", wm)
            if w_scale_2 is None:
                raise RuntimeError(f"NVFP4 module {st_name} missing weight_scale_2")
            if w_tensor.shape[-1] * 2 != in_features:
                raise RuntimeError(
                    f"NVFP4 shape mismatch for {name}: weight is {tuple(w_tensor.shape)}, "
                    f"expected in_features={in_features} → packed dim {in_features // 2}"
                )
            new_mod = NVFP4LoRALinear(
                in_features=in_features,
                out_features=out_features,
                weight_uint8=w_tensor,
                weight_scale_fp8=w_scale,
                weight_scale_2_fp32=w_scale_2,
                group_size=16,
                bias=bias,
                r=(r if is_lora_target else 0),
                lora_alpha=(lora_alpha if is_lora_target else 0),
                lora_dropout=(lora_dropout if is_lora_target else 0.0),
                device=device,
                dtype=dtype,
            )
            setattr(parent, attr, new_mod)
            counts["lora" if is_lora_target else "frozen_nvfp4"] += 1
        elif w_tensor.dtype == torch.float8_e4m3fn:
            # ---- FP8 per-tensor path ----
            # weight is fp8_e4m3fn (out, in); weight_scale is a fp32 scalar.
            # No LoRA support for these in this loader (FP8 LoRA needs a different autograd path).
            # If suffix matched a LoRA target (e.g. shared_experts.up_proj in Super), demote to
            # frozen. Counted separately so the caller can see what was demoted.
            if is_lora_target:
                counts.setdefault("lora_demoted_fp8", 0)
                counts["lora_demoted_fp8"] += 1
            scale = float(w_scale.to(torch.float32).item())
            W = w_tensor.to(torch.float32).mul_(scale).to(dtype=dtype, device=device).contiguous()
            new_mod = nn.Linear(in_features, out_features, bias=(bias is not None), device=device, dtype=dtype)
            with torch.no_grad():
                new_mod.weight.copy_(W)
                if bias is not None:
                    new_mod.bias.copy_(bias.to(device=device, dtype=dtype))
            new_mod.weight.requires_grad_(False)
            if bias is not None:
                new_mod.bias.requires_grad_(False)
            setattr(parent, attr, new_mod)
            counts["frozen_fp8"] += 1
        else:
            raise RuntimeError(
                f"Unknown quantization format for {name}: weight dtype is {w_tensor.dtype}"
            )

    return counts


# --------------------------------------------------------------------------------------
# Non-NVFP4 weight loading (norms, embeddings, conv1d, Mamba SSM state, lm_head)
# --------------------------------------------------------------------------------------

def load_non_nvfp4_weights(
    model: nn.Module,
    model_dir: str | Path,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Load all on-disk tensors that are NOT part of an NVFP4 module's storage.

    Skips:
    - `.weight`, `.weight_scale`, `.weight_scale_2`, `.input_scale`, `.bias` of any module
      already replaced with NVFP4LoRALinear (those buffers were already set during replacement)
    - tensors that are themselves NVFP4 metadata (scale/scale_2/input_scale)

    Returns: number of tensors loaded.
    """
    model_dir = Path(model_dir)
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]

    # Translate safetensors keys -> model attribute paths
    translate, st_prefix, model_prefix = make_key_translator(model, model_dir)

    # Which modules in the model were swapped in by `replace_nvfp4_modules` (so their on-disk
    # NVFP4 / FP8 storage tensors must NOT be re-loaded here). Both NVFP4LoRALinear and the
    # frozen bf16 nn.Linear replacements set their weight at module-replacement time.
    replaced_module_names: set[str] = set()
    for name, m in model.named_modules():
        if isinstance(m, NVFP4LoRALinear):
            replaced_module_names.add(name)
    # Add FP8-replaced modules: detect via the safetensors index - anything whose key prefix
    # matches a quantized-module name AND has been swapped to a different class than expected.
    # Simpler: any module whose name appears in the quantized-module list has its weight
    # already handled (either by NVFP4LoRALinear above or by the FP8 dequant-to-bf16 path).
    nvfp4_module_names_st = list_quantized_modules(model_dir)
    for st_name in nvfp4_module_names_st:
        m_name = translate(st_name)
        if m_name is not None:
            replaced_module_names.add(m_name)
    nvfp4_replaced = replaced_module_names

    n_loaded = 0
    n_skipped_mtp = 0
    # Walk tensors in the safetensors index
    for key, shard_rel in wm.items():
        # Translate safetensors-side key to model-side path. None means "skip" (e.g. MTP).
        model_key = translate(key)
        if model_key is None:
            n_skipped_mtp += 1
            continue

        # Identify the module name for this tensor (everything before the last dot, on model side)
        module_name, tensor_attr = model_key.rsplit(".", 1)

        # Skip NVFP4 storage tensors for replaced modules (already loaded into NVFP4LoRALinear buffers)
        if module_name in nvfp4_replaced and tensor_attr in (
            "weight", "weight_scale", "weight_scale_2", "input_scale", "bias"
        ):
            continue

        # For non-NVFP4 modules: load the tensor and assign to the model
        shard_path = model_dir / shard_rel
        with safetensors.safe_open(str(shard_path), framework="pt", device="cpu") as f:
            tensor = f.get_tensor(key)

        # Navigate to the parent module and find the attribute (using model-side path)
        try:
            parent, attr = _get_parent(model, model_key)
        except AttributeError:
            print(f"  WARN: path not found in model: {model_key} (from safetensors key {key})")
            continue

        # Determine dtype to use:
        # - bf16 for general weights / norms / embeddings
        # - keep original dtype for special params (e.g., A_log, D, dt_bias for Mamba; they're often float32)
        target_dtype = dtype if tensor.dtype.is_floating_point and tensor.dtype != torch.float8_e4m3fn else tensor.dtype
        try:
            # Replace meta-tensor with real one
            t = tensor.to(device=device, dtype=target_dtype).contiguous()
            # nn.Parameter wrapping if the existing slot is a Parameter
            if isinstance(getattr(parent, attr, None), nn.Parameter):
                setattr(parent, attr, nn.Parameter(t, requires_grad=False))
            else:
                # Buffer or raw tensor
                # Use module method if available; otherwise direct setattr
                if hasattr(parent, "register_buffer") and attr in dict(parent.named_buffers(recurse=False)):
                    parent._buffers[attr] = t
                else:
                    setattr(parent, attr, t)
        except Exception as e:
            print(f"  WARN: failed to load {key}: {type(e).__name__}: {e}")
            continue

        n_loaded += 1
    if n_skipped_mtp > 0:
        print(f"  skipped {n_skipped_mtp} mtp.* tensors (Multi-Token Prediction; serve-only)")
    return n_loaded


# --------------------------------------------------------------------------------------
# Top-level entry
# --------------------------------------------------------------------------------------

def load_nemotron_with_nvfp4_lora(
    model_dir: str | Path,
    target_lora_suffixes: Sequence[str] = ("q_proj", "v_proj"),
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Load a Nemotron-3 NVFP4 checkpoint with NVFP4LoRALinear modules at target paths.

    Steps:
    1. Build model architecture via `init_empty_weights` (no allocations).
    2. Replace target NVFP4 Linears with `NVFP4LoRALinear` (LoRA-trainable on `target_lora_suffixes`,
       frozen elsewhere).
    3. Load all non-NVFP4 weights (embeddings, norms, Mamba state matrices, conv1d, lm_head).

    Returns the assembled model on `device` with the requested dtype.

    Caller is responsible for setting `model.train()` / `model.eval()` and for collecting
    `[p for p in model.parameters() if p.requires_grad]` for the optimizer.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    device = torch.device(device)
    model_dir = Path(model_dir)

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=dtype)

    print(f"=== loader: replacing NVFP4 Linears (LoRA targets: {list(target_lora_suffixes)}) ===")
    counts = replace_nvfp4_modules(
        model, model_dir, target_lora_suffixes,
        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        device=device, dtype=dtype,
    )
    msg = (
        f"  replaced: {counts['lora']} LoRA-target NVFP4 + {counts.get('frozen_nvfp4', 0)} "
        f"frozen-NVFP4 + {counts.get('frozen_fp8', 0)} frozen-FP8 modules"
    )
    if counts.get("lora_demoted_fp8", 0):
        msg += f" ({counts['lora_demoted_fp8']} LoRA targets demoted to frozen on FP8 path)"
    print(msg)
    if r > 0 and counts["lora"] == 0:
        raise SystemExit(
            "No NVFP4 LoRA target modules were installed. "
            f"Tried suffixes: {list(target_lora_suffixes)}. "
            "Check target_lora_suffixes for typos or unsupported target modules."
        )

    print("=== loader: loading non-NVFP4 weights (norms, embeddings, Mamba, conv1d, lm_head) ===")
    n = load_non_nvfp4_weights(model, model_dir, device=device, dtype=dtype)
    print(f"  loaded {n} non-NVFP4 tensors")

    return model
