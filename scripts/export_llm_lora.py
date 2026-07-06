import argparse
import json
from pathlib import Path

try:
    from scripts.split_both_adapter import classify_keys, module_path_of
except ModuleNotFoundError:
    from split_both_adapter import classify_keys, module_path_of


def _load_both_config(adapter_dir: Path) -> dict:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"no adapter_config.json in {adapter_dir}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    both = cfg.get("both")
    if cfg.get("train_target") != "both" or not both:
        raise SystemExit(
            f"{cfg_path} is not a `--train-target both` adapter. "
            "This export only applies to unified both-adapters."
        )

    for req in ("vision_peft_scope", "vision_target_modules", "text_target_modules"):
        if req not in both:
            raise SystemExit(f"{cfg_path} `both` block missing required field {req!r}")

    return cfg


def _load_state(adapter_dir: Path) -> dict:
    from safetensors import safe_open

    files = sorted(adapter_dir.glob("adapter_model*.safetensors"))
    if not files:
        raise SystemExit(f"no adapter_model*.safetensors in {adapter_dir}")

    state = {}
    for adapter_file in files:
        with safe_open(adapter_file, framework="pt") as sf:
            for key in sf.keys():
                state[key] = sf.get_tensor(key)
    return state


def _is_attention_lora_key(adapter_key: str) -> bool:
    module_path = module_path_of(adapter_key)
    return module_path.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj"}


def _peft_text_config(cfg: dict) -> dict:
    both = cfg["both"]
    base_model = cfg.get("base_model_name_or_path", both.get("base_model_name_or_path"))
    return {
        "base_model_name_or_path": base_model,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": cfg["r"],
        "lora_alpha": cfg["lora_alpha"],
        "lora_dropout": cfg.get("lora_dropout", 0.0),
        "bias": "none",
        "target_modules": list(both["text_target_modules"]),
        "inference_mode": True,
        "fan_in_fan_out": False,
        "train_target": "text",
        "_export_from": "both",
        "_note": (
            "LLM-only export for vLLM runtime-LoRA. Vision/projector LoRA tensors "
            "were intentionally dropped; serve this with the tower-merged base."
        ),
    }


def export_llm_lora(
    adapter_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict:
    """Export only the LLM half of a unified both-adapter as a PEFT adapter dir."""

    from safetensors.torch import save_file

    adapter_dir = Path(adapter_dir)
    output_dir = Path(output_dir)

    cfg = _load_both_config(adapter_dir)
    both = cfg["both"]
    state = _load_state(adapter_dir)

    tower_keys, llm_keys = classify_keys(
        state.keys(),
        both["vision_peft_scope"],
        both.get("projector_scopes", ()),
    )

    if not llm_keys:
        raise SystemExit(
            "export refused: ZERO LLM keys remain after applying the both-adapter "
            "vision/projector scopes. This would serve the un-adapted language model."
        )

    attention_keys = [key for key in llm_keys if _is_attention_lora_key(key)]
    if not attention_keys:
        raise SystemExit(
            "export refused: ZERO LLM attention LoRA tensors remained "
            "(q_proj/k_proj/v_proj). This would likely serve an un-adapted base."
        )

    output_model = output_dir / "adapter_model.safetensors"
    output_config = output_dir / "adapter_config.json"
    if not overwrite and (output_model.exists() or output_config.exists()):
        raise SystemExit(
            f"{output_dir} already contains an exported adapter; pass --overwrite "
            "to replace it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: state[key].contiguous() for key in llm_keys},
        str(output_model),
    )
    output_config.write_text(
        json.dumps(_peft_text_config(cfg), indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "adapter_dir": str(adapter_dir),
        "output_dir": str(output_dir),
        "retained_llm_tensors": len(llm_keys),
        "retained_attention_tensors": len(attention_keys),
        "dropped_vision_projector_tensors": len(tower_keys),
        "adapter_model": str(output_model),
        "adapter_config": str(output_config),
    }
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the LLM half of a unified both-adapter for vLLM runtime-LoRA."
    )
    parser.add_argument("adapter_dir", help="Input train_target=both PEFT adapter directory")
    parser.add_argument("output_dir", help="Output PEFT adapter directory for the LLM half")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing adapter_model.safetensors / adapter_config.json",
    )
    args = parser.parse_args(argv)

    summary = export_llm_lora(
        args.adapter_dir,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
