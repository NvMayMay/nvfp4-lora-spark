#!/usr/bin/env python3
"""OpenAI-compatible server for Mistral-Small-4-119B (RedHatAI NVFP4-HF) + ICH LoRA.

============================================================================
DO NOT RUN UNTIL TRAINING COMPLETES (the 13h Qwen run owns the GPU until
~01:30). Loading this model allocates most of the 131 GB UMA.
============================================================================

Why this server exists: vLLM 0.22.1 cannot load the RH HF checkpoint (no
mistral4 text backbone; the checkpoint's text_config carries a stale
architectures field that recurses the Mistral3 wrapper), and request-time LoRA
over MLA kv_b_proj is unsound in vLLM anyway (decode uses load-time-absorbed
W_UK/W_UV copies). See docs/plans/SERVE_PATH_QWEN35_MISTRAL.md.

What it does instead:
  1. Loads the base exactly like the proven training path in
     scripts/train_mistral_rh_nvfp4_lora_ich_smoke.py (meta init via
     AutoModelForImageTextToText, NVFP4Experts3D MoE, BF16 attention via
     load_non_nvfp4_weights, Triton dequant workspaces).
  2. Attaches the PEFT adapter (q_b_proj / kv_b_proj / o_proj, all BF16) and,
     by default, merges it into the base weights in memory. The merge is the
     exact BF16 update W += (alpha/r) B @ A; no quantization is involved
     because the RH recipe leaves attention unquantized.
  3. Reuses the FastAPI app + endpoints from the existing house server
     Sandbox/serve_qwen3_6_35b_a3b_openai_transformers.py by module import,
     so /health, /v1/models, /v1/chat/completions and /v1/completions behave
     identically.

Known sharp edges:
  - peft 0.19.1 in the qwen-serve venv requires the in-place WeightConverter
    kwarg-filter patch (already applied; if adapter attach dies with
    "TypeError: WeightConverter.__init__() got an unexpected keyword argument
    'distributed_operation'", re-apply it per the memory note).
  - The vision tower stays on meta in this load path (text-only); the
    house server's model_device() helper is patched below to pin cuda:0.

Run (post-training):
    cd /home/veritan-spark-01/Veritan/Sandbox/repos/nvfp4-lora-spark
    /home/veritan-spark-01/Veritan/.venvs/qwen-serve/bin/python -u \
        serve/serve_mistral_rh_nvfp4_lora_openai.py \
        --adapter-path /home/veritan-spark-01/Veritan/Sandbox/adapters/mistral_small_4_119b_rh_nvfp4_lora_ich_v3_5

Expect a 10-20 minute load and low single-digit tok/s (single-stream
transformers MoE with per-forward NVFP4 dequant).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

HOUSE_SERVER = Path(
    "/home/veritan-spark-01/Veritan/Sandbox/serve_qwen3_6_35b_a3b_openai_transformers.py"
)
DEFAULT_MODEL_DIR = (
    "/home/veritan-spark-01/Veritan/Models/RedHatAI-Mistral-Small-4-119B-2603-NVFP4-HF"
)
DEFAULT_ADAPTER_DIR = (
    "/home/veritan-spark-01/Veritan/Sandbox/adapters/mistral_small_4_119b_rh_nvfp4_lora_ich_v3_5"
)
BASE_MODEL_ID = "mistral-small-4-119b-rh-nvfp4-transformers"


def import_house_server():
    spec = importlib.util.spec_from_file_location("house_openai_server", HOUSE_SERVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["house_openai_server"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve Mistral-Small-4 RH NVFP4 (+LoRA) via transformers.")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--adapter-path", default=DEFAULT_ADAPTER_DIR,
                   help="PEFT adapter dir; pass '' to serve the raw base.")
    p.add_argument("--adapter-id", default=None)
    p.add_argument("--merge-adapter", action=argparse.BooleanOptionalAction, default=True,
                   help="Merge the adapter into the BF16 attention weights after load "
                        "(exact for this recipe; faster inference). Default on.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_base(model_dir: Path, device, dtype):
    """Verbatim Phase 0.5 load path from train_mistral_rh_nvfp4_lora_ich_smoke.py."""
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForImageTextToText

    from nvfp4_lora.experts import (
        NVFP4Experts3D,
        assemble_nvfp4_experts3d_batched,
        replace_moe_experts_with_nvfp4_3d,
    )
    from nvfp4_lora.loader import (
        _assign_dequant_workspaces,
        load_non_nvfp4_weights,
        replace_nvfp4_modules,
    )

    print("[load] building model on meta...", flush=True)
    cfg = AutoConfig.from_pretrained(str(model_dir))
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(cfg)

    print("[load] replacing fused-3D MoE blocks with NVFP4Experts3D...", flush=True)
    family = getattr(model.config, "model_type", "mistral4")
    replace_moe_experts_with_nvfp4_3d(model, model_family=family)

    print("[load] replacing NVFP4 nn.Linear (shared experts; attention stays BF16)...", flush=True)
    replace_nvfp4_modules(
        model, model_dir,
        target_lora_suffixes=(),
        r=0, lora_alpha=0,
        device=device, dtype=dtype,
    )

    print("[load] assembling routed-expert NVFP4 buffers...", flush=True)
    idx_obj = json.loads((model_dir / "model.safetensors.index.json").read_text())
    wm = idx_obj["weight_map"]
    for name, module in model.named_modules():
        if isinstance(module, NVFP4Experts3D):
            assert name.startswith("model.language_model.")
            st_name = "language_model.model." + name[len("model.language_model."):]
            assemble_nvfp4_experts3d_batched(module, st_name, model_dir, wm)

    print("[load] loading BF16 attention + embeddings + norms + lm_head...", flush=True)
    load_non_nvfp4_weights(model, model_dir, device=device, dtype=dtype)

    print("[load] assigning NVFP4 dequant workspaces...", flush=True)
    _assign_dequant_workspaces(model, device=device, dtype=dtype)

    # Move stray CPU buffers/params (RoPE inv_freq etc.) to the device.
    # Vision tower params remain on meta on purpose (text-only serving).
    for mod in model.modules():
        for nm, buf in list(mod.named_buffers(recurse=False)):
            if buf.device.type == "cpu":
                mod._buffers[nm] = buf.to(device)
        for nm, par in list(mod.named_parameters(recurse=False)):
            if par.device.type == "cpu":
                mod._parameters[nm] = torch.nn.Parameter(
                    par.data.to(device), requires_grad=False
                )

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    if hasattr(model, "config"):
        try:
            model.config.use_cache = True
        except Exception:
            pass
    return model


def attach_adapter(model, adapter_path: str, merge: bool):
    from peft import PeftModel

    print(f"[adapter] attaching {adapter_path}", flush=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    if merge:
        print("[adapter] merging into BF16 attention (exact for this recipe)...", flush=True)
        model = model.merge_and_unload()
        model.eval()
        print("[adapter] merged.", flush=True)
    return model


def main() -> None:
    args = parse_args()

    house = import_house_server()  # also imports torch/fastapi/uvicorn
    # The house server's /health reports its own SERVER_MODEL_ID constant as
    # base_model; override it so health doesn't claim the qwen3.6 house model.
    house.SERVER_MODEL_ID = BASE_MODEL_ID
    import torch
    import uvicorn

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; refusing to load 119B on CPU.")
    device = torch.device(args.device)
    dtype = torch.bfloat16

    model_dir = Path(args.model_dir)
    t0 = time.time()
    model = load_base(model_dir, device, dtype)
    print(f"[load] base loaded in {time.time() - t0:.0f}s", flush=True)

    adapter_id = None
    if args.adapter_path:
        adapter_id = args.adapter_id or os.path.basename(os.path.normpath(args.adapter_path))
        model = attach_adapter(model, args.adapter_path, args.merge_adapter)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

    served_id = BASE_MODEL_ID if adapter_id is None else f"{BASE_MODEL_ID}+{adapter_id}"

    # Hand the loaded model to the house server's module state and pin the
    # generation device (next(model.parameters()) may be a meta vision param).
    house.state = house.RuntimeState(
        model_path=str(model_dir),
        tokenizer=tokenizer,
        model=model,
        model_kwargs=None,
        adapter_path=args.adapter_path or None,
        adapter_id=adapter_id,
        served_model_id=served_id,
    )
    house.model_device = lambda: device

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"[mem] cuda free {free_b / 2**30:.1f} GiB / total {total_b / 2**30:.1f} GiB", flush=True)

    print(f"[serve] {served_id} on {args.host}:{args.port}", flush=True)
    uvicorn.run(house.app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
