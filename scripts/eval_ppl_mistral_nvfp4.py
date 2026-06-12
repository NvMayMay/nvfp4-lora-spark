#!/usr/bin/env python3
"""Perplexity eval for Mistral-Small-4-119B-NVFP4 on wikitext-2 (validation split).

Reuses the Phase 0.5 acceptance-gate loading pipeline verbatim so we measure PPL of
the EXACT model that will ship — including the custom translator, NVFP4Experts3D
expert assembly, and CPU-tensor migration.

Compares against the published Mistral-Small-4 BF16 PPL when known (sanity bound;
expectation for NVFP4A16 weight-only on a 119B-class model is ≤3-5% PPL regression
vs BF16). A PPL > 10 on wikitext-2 indicates a quant bug.

Run:
    cd /home/veritan-spark-01/Veritan/Sandbox/repos/nvfp4-lora-spark
    /home/veritan-spark-01/Veritan/.venvs/qwen-serve/bin/python -u \\
        scripts/eval_ppl_mistral_nvfp4.py
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

from nvfp4_lora.experts import (
    NVFP4Experts3D,
    assemble_nvfp4_experts3d_batched,
    replace_moe_experts_with_nvfp4_3d,
)
from nvfp4_lora.loader import (
    _assign_dequant_workspaces,
    load_non_nvfp4_weights,
    make_key_translator,
    replace_nvfp4_modules,
)

MODEL_DIR = Path("/home/veritan-spark-01/Veritan/Models/Mistral-Small-4-119B-2603-NVFP4-HF")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 50          # 50 contiguous 512-token windows from wikitext-2 val
SEQ_LEN = 512
PUBLISHED_BF16_PPL_HINT = 4.5  # ballpark for 119B-class on wikitext-2; pure sanity bound


def load_model():
    """Load NVFP4 model using the same pipeline as Phase 0.5 acceptance."""
    import json
    print("Building model on meta…")
    cfg = AutoConfig.from_pretrained(str(MODEL_DIR))
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(cfg)

    print("Replacing MoE blocks with NVFP4Experts3D…")
    replace_moe_experts_with_nvfp4_3d(model, model_family="mistral4")

    print("Replacing nn.Linear NVFP4 modules…")
    replace_nvfp4_modules(
        model, MODEL_DIR,
        target_lora_suffixes=("q_b_proj", "kv_b_proj", "o_proj"),
        r=8, lora_alpha=16, device=DEVICE, dtype=torch.bfloat16,
    )

    print("Assembling routed-expert NVFP4 buffers…")
    idx = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    for name, module in model.named_modules():
        if isinstance(module, NVFP4Experts3D):
            assert name.startswith("model.language_model.")
            st_name = "language_model.model." + name[len("model.language_model."):]
            assemble_nvfp4_experts3d_batched(module, st_name, MODEL_DIR, wm)

    print("Loading non-NVFP4 weights…")
    load_non_nvfp4_weights(model, MODEL_DIR, device=DEVICE, dtype=torch.bfloat16)

    print("Assigning dequant workspaces…")
    _assign_dequant_workspaces(model, device=DEVICE, dtype=torch.bfloat16)

    # Migrate CPU-side buffers (RoPE inv_freq) + freeze vision (mirror Phase 0.5)
    n_moved = 0
    for mod in model.modules():
        for nm, buf in list(mod.named_buffers(recurse=False)):
            if buf.device.type == "cpu":
                mod._buffers[nm] = buf.to(DEVICE)
                n_moved += 1
        for nm, par in list(mod.named_parameters(recurse=False)):
            if par.device.type == "cpu":
                mod._parameters[nm] = torch.nn.Parameter(
                    par.data.to(DEVICE), requires_grad=par.requires_grad
                )
                n_moved += 1
    for attr in ("vision_tower", "multi_modal_projector"):
        sub = getattr(model.model, attr, None)
        if sub is not None:
            for p in sub.parameters():
                p.requires_grad = False
    print(f"  moved {n_moved} CPU tensors to {DEVICE}")
    return model


def load_wikitext_text() -> str:
    """Concatenate wikitext-2 validation split into one long string."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(row["text"] for row in ds if row["text"].strip())
    return text


@torch.no_grad()
def compute_ppl(model, tokenizer, text: str) -> tuple[float, int]:
    """Sliding-window PPL on the first N_SAMPLES * SEQ_LEN tokens."""
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    all_ids = enc.input_ids[0]
    total_tokens_needed = N_SAMPLES * SEQ_LEN
    if all_ids.numel() < total_tokens_needed:
        print(f"WARN: wikitext-2 val tokens ({all_ids.numel()}) < requested ({total_tokens_needed}); truncating")
        total_tokens_needed = (all_ids.numel() // SEQ_LEN) * SEQ_LEN
        actual_samples = total_tokens_needed // SEQ_LEN
    else:
        actual_samples = N_SAMPLES
    print(f"Evaluating on {actual_samples} non-overlapping {SEQ_LEN}-token windows = {total_tokens_needed} tokens")

    total_nll = 0.0
    total_tokens = 0
    model.train()  # skip eval-cache (same trick as Phase 0.5)
    t_start = time.time()
    for i in range(actual_samples):
        ids = all_ids[i * SEQ_LEN : (i + 1) * SEQ_LEN].unsqueeze(0).to(DEVICE)
        out = model(input_ids=ids)
        logits = out.logits          # (1, SEQ_LEN, vocab)
        # Standard LM loss: shift logits + labels by 1
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = ids[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        n_tok = shift_labels.numel()
        total_nll += loss.item()
        total_tokens += n_tok
        if (i + 1) % 5 == 0 or i == actual_samples - 1:
            running_ppl = math.exp(total_nll / total_tokens)
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{actual_samples}] running_ppl={running_ppl:.3f}  ({elapsed:.1f}s elapsed)")
    return math.exp(total_nll / total_tokens), total_tokens


def main() -> None:
    print(f"=== PPL eval: Mistral-Small-4-119B-NVFP4 on wikitext-2-val ===")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = load_model()
    print(f"\nLoad complete in {time.time()-t0:.1f}s\n")

    print("Loading wikitext-2 validation split…")
    text = load_wikitext_text()
    print(f"  text length: {len(text):,} chars\n")

    ppl, n_tok = compute_ppl(model, tokenizer, text)

    print(f"\n=== RESULT ===")
    print(f"PPL (NVFP4): {ppl:.3f}")
    print(f"Tokens evaluated: {n_tok}")
    print(f"Sanity hint (published BF16 119B-class on wikitext-2): ~{PUBLISHED_BF16_PPL_HINT}")
    if ppl > 10.0:
        print(f"⚠️  PPL > 10 — likely quant bug or eval mismatch")
    elif ppl > PUBLISHED_BF16_PPL_HINT * 1.10:
        print(f"⚠️  PPL > 10% above hint — investigate")
    else:
        print(f"✅ PPL within sanity bound for NVFP4A16 weight-only quant")


if __name__ == "__main__":
    main()
