#!/usr/bin/env python3
"""Forward parity probe: merged NVFP4 model vs base + runtime-LoRA reference.

Companion to validate_merge.py (structural) but at the BEHAVIOR level: does the
merged-and-re-quantized checkpoint produce the same next-token distributions as
the training-time reference (base NVFP4 + runtime NVFP4LoRALinear)? This is the
§8.4 "merge-vs-training-reference parity" check in miniature.

For each val example we teacher-force the full chat sequence through both models
and compare the next-token distributions at the assistant (supervised) positions:
  - top-1 argmax agreement
  - top-5 overlap
  - KL(reference || merged)
  - mean of the per-position max |Δ probability|

The ONLY difference between the two forward passes is the re-quantization of the
merged weight (reference = dequant(W) + s·BA applied at runtime; merged =
requant(dequant(W) + s·BA) applied as a plain weight), so these metrics isolate
the merge -> re-quant effect on model outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for nvfp4_lora
from nvfp4_lora.loader import load_nemotron_with_nvfp4_lora  # noqa: E402
from nvfp4_lora.linear import NVFP4LoRALinear  # noqa: E402


def load_adapter_native(model, adapter_dir: Path) -> int:
    """Load trained A/B into the model's NVFP4LoRALinear modules (native mode)."""
    from safetensors.torch import load_file
    sd = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    loaded = 0
    for name, mod in model.named_modules():
        if isinstance(mod, NVFP4LoRALinear) and mod.r > 0:
            k_a = f"base_model.model.{name}.lora_A.weight"
            k_b = f"base_model.model.{name}.lora_B.weight"
            if k_a in sd and k_b in sd:
                mod.lora_A.data.copy_(sd[k_a].to(mod.lora_A.device, mod.lora_A.dtype))
                mod.lora_B.data.copy_(sd[k_b].to(mod.lora_B.device, mod.lora_B.dtype))
                loaded += 1
    return loaded


def render_examples(tok, val_path: Path, n: int, max_length: int):
    """Render val messages to (input_ids, labels) with assistant-only supervision."""
    examples = []
    with open(val_path, encoding="utf-8") as f:
        for line in f:
            if len(examples) >= n:
                break
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line)["messages"]
            full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            ids = tok(full, add_special_tokens=False).input_ids
            labels = [-100] * len(ids)
            for i, m in enumerate(msgs):
                if m["role"] != "assistant":
                    continue
                prefix = tok(tok.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True),
                             add_special_tokens=False).input_ids
                through = tok(tok.apply_chat_template(msgs[:i + 1], tokenize=False, add_generation_prompt=False),
                              add_special_tokens=False).input_ids
                for p in range(len(prefix), min(len(through), len(ids))):
                    labels[p] = ids[p]
            ids = ids[:max_length]
            labels = labels[:max_length]
            if all(l == -100 for l in labels):
                continue
            examples.append((torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)))
    return examples


@torch.no_grad()
def forward_logits(model, ids, device):
    out = model(input_ids=ids.unsqueeze(0).to(device))
    return out.logits[0]  # [seq, vocab] (model dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-dir", required=True, type=Path)
    ap.add_argument("--merged-model-dir", required=True, type=Path)
    ap.add_argument("--adapter-dir", required=True, type=Path)
    ap.add_argument("--val-jsonl", required=True, type=Path)
    ap.add_argument("--n-examples", type=int, default=16)
    ap.add_argument("--target-suffixes", default="up_proj,down_proj")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    suffixes = tuple(s.strip() for s in args.target_suffixes.split(",") if s.strip())
    device = torch.device(args.device)

    tok = AutoTokenizer.from_pretrained(str(args.base_model_dir), use_fast=True, trust_remote_code=True)
    examples = render_examples(tok, args.val_jsonl, args.n_examples, args.max_length)
    print(f"[parity] {len(examples)} val examples")

    print("[parity] loading REFERENCE (base + runtime LoRA)...")
    ref = load_nemotron_with_nvfp4_lora(
        args.base_model_dir, target_lora_suffixes=suffixes,
        r=args.r, lora_alpha=args.alpha, device=args.device,
    )
    n_lora = load_adapter_native(ref, args.adapter_dir)
    n_lora_modules = sum(1 for m in ref.modules() if isinstance(m, NVFP4LoRALinear) and m.r > 0)
    print(f"[parity] adapter loaded into {n_lora}/{n_lora_modules} NVFP4LoRALinear modules "
          f"(untargeted slots keep B=0 = no-op)")
    ref.eval()

    print("[parity] loading MERGED (no runtime LoRA)...")
    merged = load_nemotron_with_nvfp4_lora(
        args.merged_model_dir, target_lora_suffixes=(), r=0, device=args.device,
    )
    merged.eval()

    print("[parity] loading BASE (no LoRA, no merge) for calibration...")
    base = load_nemotron_with_nvfp4_lora(
        args.base_model_dir, target_lora_suffixes=(), r=0, device=args.device,
    )
    base.eval()

    def compare(la, lb, pos):
        pa = la.index_select(0, pos).float()
        pb = lb.index_select(0, pos).float()
        top1 = (pa.argmax(-1) == pb.argmax(-1)).float().mean().item()
        ta = pa.topk(5, -1).indices
        tb = pb.topk(5, -1).indices
        top5 = (ta.unsqueeze(2) == tb.unsqueeze(1)).any(2).float().mean().item()
        pa_sm = F.softmax(pa, -1)
        pb_sm = F.softmax(pb, -1)
        kl = F.kl_div(F.log_softmax(pb, -1), pa_sm, reduction="batchmean").item()  # KL(a||b)
        maxpd = (pa_sm - pb_sm).abs().max(-1).values.mean().item()
        return {"top1": top1, "top5": top5, "kl": kl, "maxprobdiff": maxpd}

    # Three pairings calibrate the merge cost against the adapter's own effect:
    #   lora_effect    = base vs reference (how much the adapter changes outputs)
    #   merge_parity   = ref  vs merged    (the merge -> re-quant cost; should be << lora_effect)
    #   merged_vs_base = base vs merged    (did the merged model keep the adapter's signal?)
    pairs = {"lora_effect": [], "merge_parity": [], "merged_vs_base": []}
    n_pos = 0
    for idx, (ids, labels) in enumerate(examples):
        lr = forward_logits(ref, ids, device)
        lm = forward_logits(merged, ids, device)
        lb = forward_logits(base, ids, device)
        pos = (labels != -100).nonzero(as_tuple=True)[0]
        pos = pos[pos < lr.shape[0]].to(device)
        if len(pos) == 0:
            continue
        le = compare(lb, lr, pos)
        mp = compare(lr, lm, pos)
        mb = compare(lb, lm, pos)
        pairs["lora_effect"].append(le)
        pairs["merge_parity"].append(mp)
        pairs["merged_vs_base"].append(mb)
        n_pos += int(len(pos))
        print(f"  ex{idx}: pos={int(len(pos))} | lora_effect t1={le['top1']:.3f} KL={le['kl']:.3f}"
              f" | merge_parity t1={mp['top1']:.3f} KL={mp['kl']:.3f}"
              f" | merged_vs_base t1={mb['top1']:.3f} KL={mb['kl']:.3f}")

    def mean(xs, key):
        vals = [x[key] for x in xs]
        return (sum(vals) / len(vals)) if vals else float("nan")

    summary = {"n_examples": len(pairs["merge_parity"]), "n_assistant_positions": n_pos}
    for name, xs in pairs.items():
        summary[name] = {k: mean(xs, k) for k in ("top1", "top5", "kl", "maxprobdiff")}
    print("\n=== PARITY SUMMARY (assistant positions) ===")
    print(json.dumps(summary, indent=2))
    out = args.out or (args.merged_model_dir / "parity_vs_reference.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[parity] wrote {out}")


if __name__ == "__main__":
    main()
