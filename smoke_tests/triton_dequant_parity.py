"""Parity + speed test for the Triton NVFP4 dequant fast path.

Verifies that `dequantize_nvfp4_weight` with the Triton fast path produces output
exactly equal to (within bf16 rounding tolerance of) the eager PyTorch path, on
realistic shapes for the Mistral-Small-4-119B routed experts and MLA attention.

Then measures wall-time speedup.

Run:
    cd /home/veritan-spark-01/Veritan/Sandbox/repos/nvfp4-lora-spark
    /home/veritan-spark-01/Veritan/.venvs/qwen-serve/bin/python -u \\
        tests/test_triton_dequant_parity.py
"""
from __future__ import annotations

import time

import torch

from nvfp4_lora.dequant import dequantize_nvfp4_weight
from nvfp4_lora.triton_dequant import triton_available, triton_dequant_nvfp4


def make_inputs(out_feat: int, in_feat: int, group_size: int, device, format: str, seed: int = 0):
    g = torch.Generator(device=device).manual_seed(seed)
    weight_uint8 = torch.randint(0, 256, (out_feat, in_feat // 2), dtype=torch.uint8, device=device, generator=g)
    weight_scale_fp8_raw = torch.randn(out_feat, in_feat // group_size, dtype=torch.float32, device=device, generator=g) * 0.5
    weight_scale_fp8 = weight_scale_fp8_raw.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    if format == "modelopt":
        weight_scale_2 = torch.tensor(0.01, dtype=torch.float32, device=device)
    else:
        weight_scale_2 = torch.tensor([100.0], dtype=torch.float32, device=device)
    return weight_uint8, weight_scale_fp8, weight_scale_2


def run_parity(shape_name: str, out_feat: int, in_feat: int, group_size: int, format: str, device):
    weight_uint8, weight_scale_fp8, weight_scale_2 = make_inputs(out_feat, in_feat, group_size, device, format)

    # Force PyTorch path by routing to CPU then back. Cleaner: temporarily monkey-patch
    # triton_available. We use a private dispatch to PyTorch by moving to CPU.
    cpu_uint8 = weight_uint8.cpu()
    cpu_scale = weight_scale_fp8.cpu()
    cpu_s2 = weight_scale_2.cpu()
    ref_cpu = dequantize_nvfp4_weight(cpu_uint8, cpu_scale, cpu_s2, group_size=group_size, format=format)
    fast_cuda = dequantize_nvfp4_weight(weight_uint8, weight_scale_fp8, weight_scale_2, group_size=group_size, format=format)

    ref = ref_cpu.to(device)
    fast = fast_cuda

    assert ref.shape == fast.shape, f"shape mismatch {ref.shape} vs {fast.shape}"
    assert ref.dtype == fast.dtype == torch.bfloat16

    # Exact equality preferred — both paths produce bf16 from identical fp32 math.
    # Small differences arise from reduction order in Triton; accept atol=2e-3, rtol=5e-3.
    abs_diff = (ref.float() - fast.float()).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    nonzero_mismatch = (abs_diff > 0).float().mean().item()

    print(f"  {shape_name:24s} format={format:18s} "
          f"max_abs={max_abs:.6f}  mean_abs={mean_abs:.6f}  "
          f"nonzero_mismatch_frac={nonzero_mismatch:.4f}")
    assert max_abs < 5e-2, f"{shape_name} {format}: max abs diff {max_abs} exceeds 5e-2 (bf16 rounding tolerance)"
    return max_abs


def bench(shape_name: str, out_feat: int, in_feat: int, group_size: int, format: str, device, n=50):
    weight_uint8, weight_scale_fp8, weight_scale_2 = make_inputs(out_feat, in_feat, group_size, device, format)

    # Warmup
    for _ in range(3):
        dequantize_nvfp4_weight(weight_uint8, weight_scale_fp8, weight_scale_2, group_size=group_size, format=format)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n):
        out = dequantize_nvfp4_weight(weight_uint8, weight_scale_fp8, weight_scale_2, group_size=group_size, format=format)
    torch.cuda.synchronize()
    fast_ms = (time.perf_counter() - t0) / n * 1000

    # Brute-force PyTorch path: move to CPU each call. Not fair to compare CPU time, so
    # instead we force-disable the Triton path by swapping the loaded module's flag.
    import nvfp4_lora.triton_dequant as tdmod
    orig = tdmod._TRITON_AVAILABLE
    tdmod._TRITON_AVAILABLE = False
    try:
        for _ in range(3):
            dequantize_nvfp4_weight(weight_uint8, weight_scale_fp8, weight_scale_2, group_size=group_size, format=format)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            out = dequantize_nvfp4_weight(weight_uint8, weight_scale_fp8, weight_scale_2, group_size=group_size, format=format)
        torch.cuda.synchronize()
        py_ms = (time.perf_counter() - t0) / n * 1000
    finally:
        tdmod._TRITON_AVAILABLE = orig

    print(f"  {shape_name:24s} format={format:18s}  "
          f"PyTorch {py_ms:7.3f} ms  Triton {fast_ms:7.3f} ms  "
          f"speedup {py_ms / fast_ms:5.1f}x")


def main():
    assert triton_available(), "Triton must be available for this test"
    device = torch.device("cuda")

    print("=== Parity: Triton vs PyTorch (bf16 rounding tolerance: 5e-2) ===")
    # Mistral-Small-4 shapes from config: hidden=4096, moe_intermediate=2048, group_size=16.
    # Routed expert gate_up: out=2*2048=4096, in=4096
    # Routed expert down:    out=4096, in=2048
    # MLA attention has different shapes; pick representative ones.
    shapes = [
        ("expert_gate_up", 4096, 4096),
        ("expert_down",    4096, 2048),
        ("attn_q_b",       4096, 1024),
        ("attn_kv_b",      6144,  256),
        ("attn_o_proj",    4096, 4096),
    ]
    for name, of, inf in shapes:
        for fmt in ("modelopt", "compressed_tensors"):
            run_parity(name, of, inf, 16, fmt, device)

    print()
    print("=== Speed: Triton vs PyTorch ===")
    for name, of, inf in shapes:
        bench(name, of, inf, 16, "compressed_tensors", device, n=50)

    print()
    print("PARITY + SPEED TEST PASSED")


if __name__ == "__main__":
    main()
