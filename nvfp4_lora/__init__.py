"""nvfp4_lora - hand-rolled NVFP4 dequant + LoRA training primitives for DGX Spark sm_121.

See /path/to/research/agent_outputs/SYNTHESIS.md
for the architectural rationale (TE NVFP4 production path broken on sm_121, this
package implements the bf16-dequant fallback that both Opus and GPT-5.5 subagents
recommended).
"""
from .dequant import dequantize_nvfp4_weight, NVFP4_E2M1_LUT

__all__ = ["dequantize_nvfp4_weight", "NVFP4_E2M1_LUT"]
