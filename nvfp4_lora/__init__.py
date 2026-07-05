"""NVFP4 dequantization and LoRA training primitives for DGX Spark sm_121."""
from .dequant import dequantize_nvfp4_weight, NVFP4_E2M1_LUT
from .quantize import quantize_nvfp4_2d, quantize_nvfp4_3d_per_slice

__all__ = [
    "dequantize_nvfp4_weight",
    "NVFP4_E2M1_LUT",
    "quantize_nvfp4_2d",
    "quantize_nvfp4_3d_per_slice",
]
