# Vision-LoRA, end to end: train on the frozen 4-bit backbone, merge, serve, measure

Public evidence that a **vision** LoRA (the Pixtral tower + projector of an NVFP4 VLM)
trains over a frozen 4-bit backbone, merges into the bf16 tower, serves live through vLLM,
and measurably improves a real task. The text analog is [../spider/](../spider/); this is
the vision analog.

## Result (2026-07-04, DGX Spark GB10 / sm_121, vLLM 0.22.1)

- Base: `Mistral-Small-3.2-24B-Instruct-2506-NVFP4` (dense NVFP4 backbone, **bf16** Pixtral
  vision tower + `multi_modal_projector`).
- Adapter: a `--train-target vision` LoRA (r=16), 50 modules = Pixtral `q_proj`/`v_proj`
  across all 24 tower layers + the 2 projector Linears, trained on `flaviagiammarino/vqa-rad`
  (radiology VQA, CC0). Best-checkpoint by val loss.
- Merged into the bf16 tower with `scripts/merge_vision_lora.py` (`W += (alpha/r)*B@A`, no
  dequant/requant); the NVFP4 backbone shards are preserved byte-for-byte.
- Served as a plain multimodal VLM via vLLM (`--tokenizer-mode auto`, no `--enable-lora`);
  eval by `scripts/eval_vision_retention.py` (normalized VQA exact-match, images sent as
  base64 to the OpenAI-compatible endpoint).

| vqa-rad val (n=451), served via vLLM | normalized exact-match |
|---|---|
| base VLM | 0.4501 |
| **+ vision LoRA (merged)** | **0.4900** |
| **delta** | **+0.0399 (+4.0 pts)** |

A 60-row spot check agreed (0.467 -> 0.517, +5.0). Raw per-row predictions:
`vqarad_base_n451.json`, `vqarad_merged_n451.json`.

## Scope / honesty

- **This is a deadline-capped viability + show-off run, not a tuned result.** The adapter
  is a ~half-epoch run (best-ckpt at ~update 60, before overfit); a full run would likely
  move the delta further. The point is that the served vision fine-tune demonstrably
  changes behavior for the better.
- **vLLM runtime-LoRA does not apply vision-tower adapters** (it targets the LLM backbone
  only), so unlike the text story there is no runtime-LoRA path for vision. The supported
  vision serve story is **merge-to-bf16-base** (this recipe): the LoRA is baked into the
  bf16 tower and the model is served as a normal multimodal VLM. See `docs/SERVING.md`.
- vLLM **does** serve the multimodal Mistral3-NVFP4 with image inputs on sm_121 (confirmed:
  a live image query returns coherent answers) via `--tokenizer-mode auto`; the mistral
  tokenizer path hit an image-token-count mismatch, so the HF tokenizer is used. A side
  effect is a `Ġ` space-marker leaking into multi-word answers; it affects base and merged
  equally (the delta is unaffected), and vqa-rad answers are mostly single-word.

## Reproduce

```bash
# 1. prepare the public dataset (downscales images to fit --max-length 2048)
python scripts/prepare_vision_dataset.py --out-dir data/vqa_rad --n 0 --val-n 451
# 2. train the vision adapter (frozen NVFP4 backbone; bf16 tower + projector LoRA)
python scripts/train_nvfp4_lora.py --train-target vision \
  --model-dir <Mistral-Small-3.2-24B-NVFP4> \
  --train-file data/vqa_rad/train.jsonl --val-file data/vqa_rad/val.jsonl \
  --vision-target-modules q_proj,v_proj,linear_1,linear_2 --epochs 2 --max-length 2048
# 3. merge the adapter into the bf16 tower (NVFP4 backbone untouched)
python scripts/merge_vision_lora.py --base-model-dir <base> \
  --adapter-dir <out>/best --out-dir <base>-vision-merged
# 4. serve base and merged (one at a time) and eval base vs merged
#    serve: serve/run_mistral24b_vision_merged.sh  (or vllm serve <dir> --tokenizer-mode auto)
python scripts/eval_vision_retention.py --base-url http://127.0.0.1:8000 \
  --dev-file data/vqa_rad/val.jsonl --models <served-name> --n 451 --out out.json
```
