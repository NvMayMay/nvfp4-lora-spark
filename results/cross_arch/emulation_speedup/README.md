# Routed-only emulation dequant: GPU validation

Validation of `serve/vllm_patches/nvfp4_emulation_routed_dequant.py` (opt-in
`VLLM_PATCH_ROUTED_DEQUANT=1`). The emulation MoE backend is the only LoRA-capable NVFP4
MoE path on sm_121, but stock it dequantizes ALL experts to bf16 every forward. This patch
dequantizes only the routed experts per forward. It must be numerically EXACT and faster.

## Result: PASS (2026-07-02, DGX Spark GB10 / sm_121, vLLM 0.22.1)

- Base: `Nemotron-3-Nano-30B-A3B-NVFP4` (routed MoE), `--moe-backend emulation`, `--enforce-eager`,
  `--max-model-len 2048`, single stream.
- Adapter: an expert-LoRA (targets `up_proj`/`down_proj`, r=8), loaded as per-expert 2D
  (`is_3d_lora_weight=False`) -- no rekey needed (identity key resolution).
- Two arms: patch OFF vs patch ON. Decisive metric = prompt-echo logprobs (the forward pass),
  not generated text.

| check | result |
|---|---|
| LoRA fires (base != adapter), OFF | max\|delta\|=1.409 over 29 toks |
| LoRA fires (base != adapter), ON | max\|delta\|=1.409 (identical) |
| **parity** adapter OFF vs ON | **max\|delta\|=0 (bit-exact)** |
| **parity** base OFF vs ON | **max\|delta\|=0 (bit-exact)** |
| decode speedup (adapter) | 2.563 -> 12.859 tok/s = **5.02x** |
| decode speedup (base) | 2.656 -> 13.308 tok/s = 5.01x |
| runtime fallbacks to full dequant | 0 |

Bit-exact logprob parity (delta=0) subsumes the Spider-EM re-run in the patch's checklist:
identical per-token logprobs imply identical greedy decode by construction.

## Notes / honest scope

- Speedup scales with experts/routed ratio (E/k). Nemotron-Nano is A3B (small active set), so
  the ~5x here is a decode-dominated, single-stream figure. GLM-4.5-Air / the 120B (more experts)
  should differ; confirm per-model. The plan's 4-12x estimate brackets this.
- Concurrency (F2: raise `--max-num-seqs`) is an additional, orthogonal throughput lever not
  measured here.
- 120B (`Nemotron-3-Super-120B-A12B-NVFP4`) confirmation is a follow-up; the exact-parity result
  is architecture-general (same code path), the speedup magnitude is what varies.

## Reproduce

```bash
# arm OFF (no patch):
vllm serve <NVFP4-MoE> --moe-backend emulation --enable-lora --lora-modules canary=<adapter> \
  --served-model-name base --port 8001 --enforce-eager --max-model-len 2048
python scripts/validate_routed_dequant.py probe --arm off --out off.json
# arm ON (kill first), PYTHONPATH=serve/vllm_patches VLLM_PATCH_ROUTED_DEQUANT=1 vllm serve ... :
python scripts/validate_routed_dequant.py probe --arm on --out on.json
python scripts/validate_routed_dequant.py compare off.json on.json
```
