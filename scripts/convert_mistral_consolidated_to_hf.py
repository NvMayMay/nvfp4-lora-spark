#!/usr/bin/env python3
"""Convert RedHatAI/Mistral-Small-4-119B-2603-NVFP4 (Mistral consolidated layout) ->
HuggingFace transformers layout, in-place renaming.

Why: the consolidated checkpoint is the canonical artifact for vLLM serving and embeds
RedHatAI's quantization parameters (NVFP4 weights for MoE, BF16 attention) directly.
By converting to HF naming without touching values, an adapter trained against the HF
copy is byte-portable to vLLM serving of the original consolidated checkpoint — no
re-quant, no calibration set, no per-tensor scale guessing.

Tensor values are NOT modified — only safetensors keys + a new HF-style `config.json`
and `model.safetensors.index.json` are written. Source shards are streamed; output
shards are 5 GB each.

Run:
    /home/veritan-spark-01/Veritan/.venvs/qwen-serve/bin/python -u \\
        scripts/convert_mistral_consolidated_to_hf.py
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("/home/veritan-spark-01/Veritan/Models/RedHatAI-Mistral-Small-4-119B-2603-NVFP4")
DST = Path("/home/veritan-spark-01/Veritan/Models/RedHatAI-Mistral-Small-4-119B-2603-NVFP4-HF")

MAX_SHARD_BYTES = 5 * 1024 ** 3


def rename_key(k: str) -> str | None:
    """Map a consolidated key -> HF key. Returns None to drop (none expected here)."""

    # --- top-level text backbone ---
    if k == "tok_embeddings.weight":
        return "language_model.model.embed_tokens.weight"
    if k == "output.weight":
        return "language_model.lm_head.weight"
    if k == "norm.weight":
        return "language_model.model.norm.weight"

    # --- per-layer text backbone ---
    m = re.match(r"layers\.(\d+)\.(.+)$", k)
    if m:
        L, rest = m.group(1), m.group(2)
        pfx = f"language_model.model.layers.{L}"

        # Norms
        if rest == "attention_norm.weight":
            return f"{pfx}.input_layernorm.weight"
        if rest == "ffn_norm.weight":
            return f"{pfx}.post_attention_layernorm.weight"

        # MLA attention block
        attn = {
            "attention.wq_a.weight":            f"{pfx}.self_attn.q_a_proj.weight",
            "attention.wq_b.weight":            f"{pfx}.self_attn.q_b_proj.weight",
            "attention.wkv_a_with_mqa.weight":  f"{pfx}.self_attn.kv_a_proj_with_mqa.weight",
            "attention.wkv_b.weight":           f"{pfx}.self_attn.kv_b_proj.weight",
            "attention.wo.weight":              f"{pfx}.self_attn.o_proj.weight",
            "attention.q_a_norm.weight":        f"{pfx}.self_attn.q_a_layernorm.weight",
            "attention.kv_a_norm.weight":       f"{pfx}.self_attn.kv_a_layernorm.weight",
        }
        if rest in attn:
            return attn[rest]

        # MoE router
        if rest == "gate.weight":
            return f"{pfx}.mlp.gate.weight"

        # Routed expert weights: experts.{e}.{w1|w2|w3}.{weight_packed|weight_scale|weight_global_scale|input_global_scale}
        em = re.match(r"experts\.(\d+)\.(w[123])\.(.+)$", rest)
        if em:
            e_idx, w_name, suffix = em.group(1), em.group(2), em.group(3)
            proj = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}[w_name]
            return f"{pfx}.mlp.experts.{e_idx}.{proj}.{suffix}"

        # Shared expert weights: shared_experts.{w1|w2|w3}.{...}
        sm = re.match(r"shared_experts\.(w[123])\.(.+)$", rest)
        if sm:
            w_name, suffix = sm.group(1), sm.group(2)
            proj = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}[w_name]
            return f"{pfx}.mlp.shared_experts.{proj}.{suffix}"

        raise ValueError(f"Unrecognised text-backbone key under layers.{L}: {rest!r}")

    # --- vision branch ---
    if k.startswith("vision_encoder."):
        # vision_encoder.ln_pre.weight -> vision_tower.ln_pre.weight
        # vision_encoder.patch_conv.weight -> vision_tower.patch_conv.weight
        # vision_encoder.transformer.layers.N.attention.{wq,wk,wv,wo}.weight -> ...transformer.layers.N.attention.{q,k,v,o}_proj.weight
        # vision_encoder.transformer.layers.N.feed_forward.{w1,w2,w3}.weight -> ...transformer.layers.N.feed_forward.{gate,down,up}_proj.weight (Pixtral convention varies; keep w1/w2/w3 to be safe)
        rest = k[len("vision_encoder."):]
        m2 = re.match(r"transformer\.layers\.(\d+)\.attention\.w([qkvo])\.weight$", rest)
        if m2:
            L, qkvo = m2.group(1), m2.group(2)
            mapname = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}[qkvo]
            return f"vision_tower.transformer.layers.{L}.attention.{mapname}.weight"
        m3 = re.match(r"transformer\.layers\.(\d+)\.feed_forward\.w(\d)\.weight$", rest)
        if m3:
            L, n = m3.group(1), m3.group(2)
            mapname = {"1": "gate_proj", "2": "down_proj", "3": "up_proj"}[n]
            return f"vision_tower.transformer.layers.{L}.feed_forward.{mapname}.weight"
        return f"vision_tower.{rest}"

    if k.startswith("vision_language_adapter."):
        rest = k[len("vision_language_adapter."):]
        mapname = {"w_in.weight": "linear_1.weight", "w_out.weight": "linear_2.weight"}
        if rest in mapname:
            return f"multi_modal_projector.{mapname[rest]}"
        return f"multi_modal_projector.{rest}"

    if k.startswith("patch_merger."):
        return f"multi_modal_projector.{k}"

    if k == "pre_mm_projector_norm.weight":
        return "multi_modal_projector.norm.weight"

    raise ValueError(f"Unhandled key: {k!r}")


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Source not found: {SRC}")
    DST.mkdir(parents=True, exist_ok=True)

    print("=== Convert RedHatAI Mistral consolidated -> HF naming (in place) ===")
    src_idx = json.loads((SRC / "consolidated.safetensors.index.json").read_text())["weight_map"]
    print(f"source keys: {len(src_idx)}")

    # Group source keys by source shard
    by_shard: dict[str, list[str]] = {}
    for k, sh in src_idx.items():
        by_shard.setdefault(sh, []).append(k)

    # Output buffer state
    out_buffer: dict[str, torch.Tensor] = {}
    out_size = 0
    out_shard_idx = 1
    out_template = "model-{idx:05d}.safetensors"
    weight_map_out: dict[str, str] = {}
    t0 = time.time()

    def flush():
        nonlocal out_buffer, out_size, out_shard_idx
        if not out_buffer:
            return
        name = out_template.format(idx=out_shard_idx)
        path = DST / name
        save_file(out_buffer, str(path), metadata={"format": "pt"})
        sz = sum(t.numel() * t.element_size() for t in out_buffer.values())
        print(f"  wrote {name}: {len(out_buffer)} keys, {sz/1e9:.2f} GB")
        for k in out_buffer:
            weight_map_out[k] = name
        out_buffer = {}
        out_size = 0
        out_shard_idx += 1

    def emit(new_key: str, tensor: torch.Tensor):
        nonlocal out_size
        sz = tensor.numel() * tensor.element_size()
        if out_size + sz > MAX_SHARD_BYTES and out_buffer:
            flush()
        out_buffer[new_key] = tensor.contiguous()
        out_size += sz

    for shard_name in sorted(by_shard):
        keys = sorted(by_shard[shard_name])
        print(f"--- source shard {shard_name} ({len(keys)} keys) ---")
        with safe_open(str(SRC / shard_name), framework="pt") as st:
            for k in keys:
                nk = rename_key(k)
                if nk is None:
                    continue
                emit(nk, st.get_tensor(k))
    flush()

    print(f"\nWriting model.safetensors.index.json…")
    total_size = sum((DST / shard).stat().st_size for shard in set(weight_map_out.values()))
    (DST / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_size}, "weight_map": weight_map_out},
        indent=2,
    ))

    # Build a HF-style config.json from params.json + RH's quantization_config.
    print("Writing config.json (HF-style, with RH's quant config)…")
    params = json.loads((SRC / "params.json").read_text())
    rh_qcfg = params.pop("quantization_config")

    # Translate ignore list to HF module names (which are what apply_quantization_config matches against)
    # RH ignore was Mistral-native pattern; in HF naming attention modules live under self_attn,
    # vision under vision_tower, projector under multi_modal_projector. The pattern below preserves
    # RH's intent: attention BF16, vision BF16, lm_head/embed_tokens BF16, MoE quantized.
    rh_qcfg["ignore"] = [
        "lm_head",
        "re:.*embed_tokens$",
        "re:.*gate$",                          # MoE router gate
        "re:.*self_attn.*",                    # all attention (q_a/q_b/kv_a/kv_b/o/norms) — BF16
        "re:.*vision_tower.*",                 # Pixtral vision encoder — BF16
        "re:.*multi_modal_projector.*",        # vision-text projector — BF16
        "re:.*input_layernorm$",
        "re:.*post_attention_layernorm$",
        "re:.*\\.norm$",
    ]

    # Build a minimal Mistral3ForConditionalGeneration HF config from RH's params.json + recipe
    text_cfg = {
        "architectures": ["Mistral3ForConditionalGeneration"],
        "model_type": "mistral4",
        "hidden_size": params["dim"],
        "intermediate_size": params["hidden_dim"],
        "moe_intermediate_size": params["moe"]["expert_hidden_dim"],
        "num_hidden_layers": params["n_layers"],
        "num_attention_heads": params["n_heads"],
        "num_key_value_heads": params["n_kv_heads"],
        "head_dim": params["head_dim"],
        "qk_nope_head_dim": params["qk_nope_head_dim"],
        "qk_rope_head_dim": params["qk_rope_head_dim"],
        "v_head_dim": params["v_head_dim"],
        "q_lora_rank": params["q_lora_rank"],
        "kv_lora_rank": params["kv_lora_rank"],
        "rms_norm_eps": params["norm_eps"],
        "vocab_size": params["vocab_size"],
        "max_position_embeddings": params["max_position_embeddings"],
        "rope_theta": params["rope_theta"],
        "first_k_dense_replace": params["moe"]["first_k_dense_replace"],
        "n_routed_experts": params["moe"]["num_experts"],
        "num_experts_per_tok": params["moe"]["num_experts_per_tok"],
        "n_shared_experts": params["moe"]["num_shared_experts"],
        "tie_word_embeddings": params.get("tied_embeddings", False),
        "hidden_act": "silu",
        "torch_dtype": "bfloat16",
    }

    vision_cfg = params["vision_encoder"]
    cfg = {
        "architectures": ["Mistral3ForConditionalGeneration"],
        "dtype": "bfloat16",
        "image_token_index": vision_cfg.get("image_token_id", 10),
        "model_type": "mistral3",
        "multimodal_projector_bias": False,
        "projector_hidden_act": "gelu",
        "spatial_merge_size": vision_cfg["spatial_merge_size"],
        "text_config": text_cfg,
        "vision_config": {
            "model_type": "pixtral",
            "hidden_size": vision_cfg["hidden_size"],
            "intermediate_size": vision_cfg["intermediate_size"],
            "num_hidden_layers": vision_cfg["num_hidden_layers"],
            "num_attention_heads": vision_cfg["num_attention_heads"],
            "num_channels": vision_cfg["num_channels"],
            "image_size": vision_cfg["image_size"],
            "patch_size": vision_cfg["patch_size"],
            "rope_theta": vision_cfg["rope_theta"],
        },
        "quantization_config": rh_qcfg,
    }
    (DST / "config.json").write_text(json.dumps(cfg, indent=2))

    # Copy tokenizer + chat_template files (RH consolidated has these too)
    print("Copying tokenizer + chat_template + processor files…")
    for f in SRC.iterdir():
        if f.is_file() and (
            f.suffix in (".jinja", ".txt", ".md") or
            f.name in ("processor_config.json",)
        ):
            shutil.copy2(f, DST / f.name)

    # Mistral consolidated typically doesn't ship a HF tokenizer.json — vLLM uses the
    # `tekken.json`-style native tokenizer. For HF transformers loading we need the HF
    # tokenizer files. If the original Mistral-Small-4 BF16 source has them, copy from there.
    bf16_src = Path("/home/veritan-spark-01/Veritan/Models/Mistral-Small-4-119B-2603-BF16-HF")
    if bf16_src.exists():
        for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                      "preprocessor_config.json", "generation_config.json"):
            src_f = bf16_src / fname
            if src_f.exists() and not (DST / fname).exists():
                shutil.copy2(src_f, DST / fname)
                print(f"  copied tokenizer file {fname} from BF16 source")

    elapsed = time.time() - t0
    print(f"\n=== Conversion complete in {elapsed/60:.1f} min ===")
    print(f"Output: {DST}")
    print(f"  Total size: {total_size/1e9:.2f} GB")
    print(f"  Output shards: {out_shard_idx - 1}")


if __name__ == "__main__":
    main()
