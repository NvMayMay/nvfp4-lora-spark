#!/usr/bin/env python3
"""Eager runtime-LoRA probe on an NVFP4 base (sidesteps the vLLM loader).

Loads an NVFP4 checkpoint with NVFP4LoRALinear injected on the adapter's target
suffixes (the repo's own runtime-LoRA path, reading safetensors directly), then:

  1) BINDING: how many target modules receive adapter weights, under NAIVE prefix
     matching (base_model.model.<module_name>.*, what a careless loader does) vs
     robust SUFFIX matching (layers.N...). A gap = the silent-no-op re-key bug.
     Also breaks the bound modules down by kind (attention vs shared_expert), which
     answers whether the dogfood's 120 shared_expert targets are servable at all.

  2) BEHAVIOR (best-effort): greedy-generate base vs base+LoRA on a few val prompts
     and flag the fine-tuned style, confirming the delta is actually applied. The
     base habitually opens with a "Thinking Process:" chain-of-thought scaffold; the
     fine-tune suppresses it.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nvfp4_lora.loader import load_nemotron_with_nvfp4_lora  # noqa: E402
from nvfp4_lora.linear import NVFP4LoRALinear  # noqa: E402
from safetensors.torch import load_file  # noqa: E402


def suffix_of(s: str):
    m = re.search(r"(layers\..*)$", s)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-dir", required=True, type=Path)
    ap.add_argument("--adapter-dir", required=True, type=Path)
    ap.add_argument("--target-suffixes",
                    default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--r", type=int, default=128)
    ap.add_argument("--alpha", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--permissive", action="store_true",
                    help="strict=False: tolerate on-disk tensors absent from the inference graph (e.g. the mtp. speculation head)")
    ap.add_argument("--val-jsonl", type=Path, default=None)
    ap.add_argument("--n-gen", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    suffixes = tuple(s.strip() for s in args.target_suffixes.split(",") if s.strip())

    print(f"[eager] loading {args.base_model_dir.name} + NVFP4LoRALinear on {suffixes} (r={args.r})...")
    model = load_nemotron_with_nvfp4_lora(
        args.base_model_dir, target_lora_suffixes=suffixes,
        r=args.r, lora_alpha=args.alpha, device=args.device,
        strict=not args.permissive,
    )
    model.eval()

    lora_mods = [(n, m) for n, m in model.named_modules()
                 if isinstance(m, NVFP4LoRALinear) and m.r > 0]
    kind = collections.Counter()
    for n, _ in lora_mods:
        if "shared_expert" in n:
            kind["shared_expert"] += 1
        elif "self_attn" in n:
            kind["attention"] += 1
        else:
            kind["other"] += 1
    print(f"[eager] {len(lora_mods)} NVFP4LoRALinear (r>0) modules; by kind: {dict(kind)}")
    print("[eager] sample module name:", lora_mods[0][0] if lora_mods else None)

    sd = load_file(str(args.adapter_dir / "adapter_model.safetensors"))
    akeys = [k for k in sd if k.endswith(".lora_A.weight")]
    print(f"[eager] adapter has {len(akeys)} lora_A tensors; sample key:", akeys[0] if akeys else None)

    # adapter suffix index
    asuf: dict[str, dict[str, str]] = {}
    for k in sd:
        mm = re.search(r"(layers\..*)\.lora_([AB])\.weight$", k)
        if mm:
            asuf.setdefault(mm.group(1), {})[mm.group(2)] = k

    def naive_count() -> int:
        c = 0
        for n, _ in lora_mods:
            if (f"base_model.model.{n}.lora_A.weight" in sd
                    and f"base_model.model.{n}.lora_B.weight" in sd):
                c += 1
        return c

    def suffix_load(apply: bool) -> int:
        c = 0
        for n, m in lora_mods:
            s = suffix_of(n)
            if s and s in asuf and "A" in asuf[s] and "B" in asuf[s]:
                if apply:
                    m.lora_A.data.copy_(sd[asuf[s]["A"]].to(m.lora_A.device, m.lora_A.dtype))
                    m.lora_B.data.copy_(sd[asuf[s]["B"]].to(m.lora_B.device, m.lora_B.dtype))
                c += 1
        return c

    nc = naive_count()
    sc = suffix_load(apply=False)
    print(f"\n[BINDING] naive prefix-match: {nc}/{len(lora_mods)}  |  robust suffix-match: {sc}/{len(lora_mods)}")
    if nc == len(lora_mods):
        verdict = "naive binds fully (no re-key needed in this loader)"
    elif sc > nc:
        verdict = f"SILENT NO-OP without re-key (naive {nc}, correct {sc}); the binding contract bites"
    else:
        verdict = "no match either way (suffix mismatch)"
    print("[BINDING] verdict:", verdict)

    loaded = suffix_load(apply=True)
    print(f"[eager] adapter loaded into {loaded}/{len(lora_mods)} modules (suffix-match)")

    result = {
        "base": args.base_model_dir.name, "adapter": args.adapter_dir.name,
        "lora_modules": len(lora_mods), "by_kind": dict(kind),
        "naive_bind": nc, "suffix_bind": sc, "loaded": loaded, "verdict": verdict,
    }

    if args.val_jsonl:
        try:
            from transformers import AutoTokenizer, PreTrainedTokenizerFast
            try:
                tok = AutoTokenizer.from_pretrained(str(args.base_model_dir), trust_remote_code=True)
            except Exception:
                tok = PreTrainedTokenizerFast(tokenizer_file=str(args.base_model_dir / "tokenizer.json"))
                tcfg = json.load(open(args.base_model_dir / "tokenizer_config.json"))
                if tcfg.get("chat_template"):
                    tok.chat_template = tcfg["chat_template"]
            rows = [json.loads(l) for l in open(args.val_jsonl) if l.strip()]
            prompts = []
            for r in rows:
                pm = r["messages"][:-1]
                if sum(len(x["content"]) for x in pm) <= 13000:
                    prompts.append(pm)
                if len(prompts) >= args.n_gen:
                    break

            @torch.no_grad()
            def gen(messages):
                enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                              return_tensors="pt", return_dict=True)
                input_ids = enc["input_ids"].to(args.device)
                out = model.generate(input_ids=input_ids, max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
                return tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)

            def style(g):
                return "BASE-CoT" if g.lstrip().lower().startswith("thinking process") else "FT-report"

            ft = [gen(p) for p in prompts]                       # base + LoRA
            saved = [m.lora_B.data.clone() for _, m in lora_mods]
            for _, m in lora_mods:
                m.lora_B.data.zero_()                            # disable LoRA -> base
            base = [gen(p) for p in prompts]
            for (_, m), s in zip(lora_mods, saved):
                m.lora_B.data.copy_(s)

            result["gen"] = [
                {"base_style": style(b), "lora_style": style(f),
                 "base_head": b[:110], "lora_head": f[:110]}
                for b, f in zip(base, ft)
            ]
            print("\n[BEHAVIOR] base vs base+LoRA (style):")
            for i, g in enumerate(result["gen"]):
                print(f"  ex{i}: base={g['base_style']:9s} lora={g['lora_style']}")
                print(f"        base: {g['base_head']!r}")
                print(f"        lora: {g['lora_head']!r}")
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("[BEHAVIOR] skipped (error above); binding result still valid")

    print("\n=== EAGER PROBE RESULT ===")
    print(json.dumps({k: v for k, v in result.items() if k != "gen"}, indent=2))
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)
        print("[eager] wrote", args.out)


if __name__ == "__main__":
    main()
