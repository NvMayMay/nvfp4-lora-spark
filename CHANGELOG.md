# Changelog

All notable changes to nybbloris. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); releases are tagged on
[GitHub](https://github.com/NvMayMay/nvfp4-lora-spark/releases).

## [Unreleased]

### Added
- **Runtime-LoRA for a VLM's LLM half.** An in-tree vLLM plugin
  (`nvfp4_lora/vllm_plugins/nemotron_vl_lora.py`) gives the Nemotron-Omni wrapper
  (`NemotronH_Nano_VL_V2`) the `SupportsLoRA` contract it lacks upstream, so its LLM backbone
  serves a live adapter under `--enable-lora` without merging the 4-bit base. Pixtral /
  Mistral-Small VLMs already support this in stock vLLM. `scripts/export_llm_lora.py` splits
  the LLM half out of a `--train-target both` adapter. GPU-validated by a base-vs-adapter
  logprob delta. (#35)

### Changed
- README gains a table of contents, status badges, and a [documentation index](docs/README.md);
  the VLM serving story now documents the runtime-LoRA path for the LLM half.

## [1.9] - 2026-07-06
Fine-tune the LLM **and** the vision tower together (`--train-target both`): one run over a
mixed image+text dataset, dual LoRA scopes (native LLM + bf16 tower), validated end to end
(train → split → merge → serve → image+text inference) on Nemotron-Omni and Pixtral.

## [1.8] - 2026-07-05
Fine-tune the vision tower of an NVFP4 VLM (`--train-target vision`): freeze the 4-bit LLM
backbone, LoRA-train the bf16 tower + projector, merge and serve. +4.0 EM on vqa-rad with
Pixtral.

## [1.7] - 2026-07-03
Onboard any NVFP4 model. A generic family fallback trains and serves an unregistered flat
causal-LM, guarded by strict-load + target-coverage gates. Proven end to end on Command-A
(`cohere2`, 111B, no registry entry): trained and served runtime-LoRA with the adapter
provably applied.

## [1.6] - 2026-07-02
Capability contract and provenance: `nybbloris inspect` predicts binding from config + the
safetensors index (no weights), `serve --verify` proves the adapter changed the forward at
runtime, and a base-fingerprint manifest gate refuses a wrong-base adapter before launch.

## [1.5] - 2026-07-01
Qwen3.5-122B expert-LoRA serves, with the runtime apply-check confirming the adapter applies.

## [1.4] - 2026-06-28
Stronger Spider result, harness hardening, and dynamic LoRA hot-load.

## [1.3.0] - 2026-06-13
Strict loading: on-disk tensors that map to no path fail at load, and no parameter is left on
meta — trust the load, or fail before it.

## [1.3] - 2026-06-12
Multi-family training certified at 120B scale, dynamic LoRA serving, and a grouped MoE GEMM.

## [1.2] - 2026-06-08
Triton fused NVFP4 dequant kernel: ~20x on weight dequant, ~11x end to end.

## [1.0.1] - 2026-06-02
Larger context windows for fine-tuning.

## [1.0.0] - 2026-05-25
Initial release: LoRA fine-tuning on NVFP4 weights on a single DGX Spark (GB10).

[Unreleased]: https://github.com/NvMayMay/nvfp4-lora-spark/compare/v1.9...HEAD
[1.9]: https://github.com/NvMayMay/nvfp4-lora-spark/releases/tag/v1.9
[1.8]: https://github.com/NvMayMay/nvfp4-lora-spark/releases/tag/v1.8
[1.7]: https://github.com/NvMayMay/nvfp4-lora-spark/releases/tag/v1.7
[1.6]: https://github.com/NvMayMay/nvfp4-lora-spark/releases/tag/v1.6
[1.5]: https://github.com/NvMayMay/nvfp4-lora-spark/releases/tag/v1.5
[1.4]: https://github.com/NvMayMay/nvfp4-lora-spark/releases/tag/v1.4
[1.3.0]: https://github.com/NvMayMay/nvfp4-lora-spark/releases/tag/v1.3.0
[1.3]: https://github.com/NvMayMay/nvfp4-lora-spark/releases/tag/v1.3
[1.2]: https://github.com/NvMayMay/nvfp4-lora-spark/releases/tag/v1.2
[1.0.1]: https://github.com/NvMayMay/nvfp4-lora-spark/releases/tag/v1.0.1
[1.0.0]: https://github.com/NvMayMay/nvfp4-lora-spark/releases/tag/v1.0.0
