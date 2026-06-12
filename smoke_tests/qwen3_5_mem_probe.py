#!/usr/bin/env python3
"""Memory probe v2 for the Qwen3.5-122B OOM during the first training step.

v1 finding: forward-only at 64 tokens passes; per-module deltas are tiny.
All four trainer OOM kills show anon-rss ~48 GB regardless of seq_len
(1024/2048) and kernel stack (with/without fla). The kill lands ~110 s after
optimizer_ready, i.e. plausibly during the first BACKWARD, which v1 never ran.

v2: staged forward+backward at 64 -> 512 -> 2048 tokens, logging process RSS,
cuda_free, and torch_allocated around every phase. Whatever stage dies, the
last logged line localizes the allocation (fwd vs bwd, and seq dependence).

Run:
    cd /home/veritan-spark-01/Veritan/Sandbox/repos/nvfp4-lora-spark
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/veritan-spark-01/Veritan/.venvs/qwen-serve/bin/python -u \
        smoke_tests/qwen3_5_mem_probe.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil
import torch
from pathlib import Path

MODEL_DIR = Path("/home/veritan-spark-01/Veritan/Models/RedHatAI-Qwen3.5-122B-A10B-NVFP4")
PROC = psutil.Process()


def snap(tag: str) -> None:
    torch.cuda.synchronize()
    free, _ = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    rss = PROC.memory_info().rss
    print(
        f"[probe] {tag}: rss={rss/1e9:.2f}GB torch_alloc={alloc/1e9:.2f}GB cuda_free={free/1e9:.2f}GB",
        flush=True,
    )


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_qwen3_5_122b_rh_nvfp4_lora_ich import load_qwen3_5_rh_nvfp4

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("[probe] loading model…", flush=True)
    model = load_qwen3_5_rh_nvfp4(
        MODEL_DIR, device, dtype,
        lora_target_suffixes=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    snap("post-load")

    for seq in (64, 512, 2048):
        ids = torch.randint(0, 1000, (1, seq), device=device)
        snap(f"seq{seq} pre-forward")
        out = model(input_ids=ids, labels=ids)
        snap(f"seq{seq} post-forward (loss={out.loss.item():.3f})")
        out.loss.backward()
        snap(f"seq{seq} post-backward")
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        del out
        torch.cuda.empty_cache()
        snap(f"seq{seq} post-cleanup")

    print("[probe] PASSED all stages (64/512/2048 fwd+bwd)", flush=True)


if __name__ == "__main__":
    main()
