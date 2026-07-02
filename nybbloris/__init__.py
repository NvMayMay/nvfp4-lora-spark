"""nybbloris -- LoRA fit and runtime-LoRA serve for NVFP4 models on consumer Blackwell.

The productized surface over the nvfp4_lora engine. v1 entry points:
  nybbloris inspect  -- pre-flight serve plan (binding + quant-liveness + engine req)
  nybbloris serve    -- pre-flight gate, then the dynamic-LoRA vLLM serve launcher
  nybbloris train    -- LoRA fine-tune (pass-through to the unified trainer)
"""
from .plan import render_plan, serve_plan

__version__ = "1.6.0"
__all__ = ["serve_plan", "render_plan", "__version__"]
