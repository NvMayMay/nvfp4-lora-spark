"""NVFP4 dequantization and LoRA training primitives for DGX Spark sm_121."""
from .dequant import dequantize_nvfp4_weight, NVFP4_E2M1_LUT

__all__ = ["dequantize_nvfp4_weight", "NVFP4_E2M1_LUT"]
