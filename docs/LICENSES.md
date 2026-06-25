# Licenses, provenance, and artifact policy

## Project license
Released under **Apache-2.0** (see `LICENSE`). Apache-2.0 is permissive and carries an
explicit patent grant, matching the core dependency stack.

## Runtime dependencies (and their licenses)
| Package | License |
|---|---|
| torch (PyTorch) | BSD-3-Clause |
| transformers | Apache-2.0 |
| peft | Apache-2.0 |
| vllm | Apache-2.0 |
| safetensors | Apache-2.0 |
| accelerate | Apache-2.0 |
| psutil | BSD-3-Clause |
| flash-linear-attention (GDN training only) | MIT (verify pinned 0.4.2) |

All permissive (Apache-2.0 / BSD / MIT); no copyleft in the runtime path. **PEFT is
vendored/patched** for the serving stack; the patch inherits PEFT's Apache-2.0 and is
documented where applied (`serve/`).

## Models (anchors + public example)
| Model | Role | License |
|---|---|---|
| nvidia/Qwen3.6-35B-A3B-NVFP4 | canonical anchor (MoE, FP8 attention) | Apache-2.0 (Qwen3); verify NVIDIA model-card terms |
| Qwen3-32B-NVFP4 | adjacent anchor (dense, flat, NVFP4 attention) | Apache-2.0 (Qwen3) |
| RedHatAI/Qwen3.5-122B-A10B-NVFP4 | showcased scale result (post-v1) | verify model-card license |

Base models are pulled from their original sources; nybbloris does **not** redistribute
base weights.

## Data
- **Public example: Spider** (`xlangai/spider`, text-to-SQL) -- the reproducible before/after
  result, scored deterministically with no DB execution (`scripts/prep_spider.py` +
  `scripts/eval_retention.py`).
- **Private clinical (ICH) data: NOT distributable, NOT part of any reproducibility claim.**
  It is internal evidence only; every published/reproducible number uses public data.

## Published-artifact policy
- Adapters shipped with the repo (e.g. the Spider demo adapter) are released under
  **Apache-2.0**, each with a **SHA256** of `adapter_model.safetensors` + `adapter_config.json`
  and the (base model, repo commit, exact command) that produced it.
- No clinical-data-derived adapter is ever published.

## To verify before v1 publish
- Exact license of `flash-linear-attention==0.4.2`.
- NVIDIA NVFP4 model-card terms (any restriction beyond the base Qwen3 Apache-2.0).
- RedHatAI 122B model-card license (only if 122B becomes a claimed path).
