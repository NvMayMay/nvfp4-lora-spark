# CPU tests

CPU-only regression guard for the multi-family support surfaces, so the
newly-public Qwen3.5 / Mistral / Nemotron paths cannot silently regress. Scope is
strictly CPU-verifiable logic: family resolution (`resolve_family`, `FAMILIES`),
native-vs-PEFT LoRA detection (`detect_lora_mode`, `list_quantized_modules`),
safetensors key translation (`make_key_translator`), chat dataset
encoding/masking (`ChatJsonlDataset`, `collate_batch`), and the pure-torch NVFP4
dequant round-trip. No model weights are loaded and the GPU is never touched (an
autouse fixture fails any test that allocates CUDA memory).

Config/index fixtures under `fixtures/` are trimmed copies of the two public NVFP4
checkpoints. Real GPU smoke tests live in `smoke_tests/` and are not run here.
Run locally with `python -m pytest tests/ -q`.
