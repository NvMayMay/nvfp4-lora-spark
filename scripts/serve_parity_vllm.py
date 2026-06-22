#!/usr/bin/env python
"""Serve-parity probe for runtime-LoRA in vLLM (the v1 serve thesis).

Greedy-decodes the same prompts through a running vLLM server under three served
model names and compares the generations:
  <base>    : NVFP4 base, no adapter
  ich_orig  : base + adapter with FLAT PEFT keys (base_model.model.model.layers.*)
              -> the attention_only_lora_cutlass_moe patch remaps them to the
                 language_model.* tree at load time
  ich_rekey : base + the SAME weights pre-rekeyed to language_model.* (the patch
              remap is then a no-op)

Decisive expectations if runtime-LoRA actually binds in vLLM:
  * ich_orig  == ich_rekey  (runtime re-key reproduces the offline re-key)
  * ich_orig  != base       (the adapter changes the output; not a silent no-op)
If the adapter were a silent no-op (the "swathes not applied" bug), ich_orig would
equal base. This is the Phase-1a 3-way calibration in the real serving engine.

Uses /v1/chat/completions so the server (which already holds a working tokenizer)
renders the chat template; comparison is on the returned strings. stdlib HTTP only.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def chat(base_url, model, messages, max_tokens, chat_template, timeout=600):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
    }
    if chat_template:
        body["chat_template"] = chat_template
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.load(r)
            return out["choices"][0]["message"]["content"], None
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:200]
            if attempt == 1:
                return None, f"HTTP {e.code}: {msg}"
            time.sleep(2)
        except Exception as e:  # noqa: BLE001
            if attempt == 1:
                return None, str(e)[:200]
            time.sleep(2)
    return None, "unreachable"


def char_lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def token_f1(pred, ref):
    """Whitespace-token F1 of a generation against the reference answer.

    A coarse but repeatable lexical-overlap proxy for "did this arm move toward the
    fine-tune target". Used for the base/full/shared-only retention comparison; the
    absolute value is crude, the RELATIVE ordering across arms is the signal.
    """
    from collections import Counter
    p, r = pred.split(), ref.split()
    if not p or not r:
        return 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--chat-template", default=None,
                    help="optional jinja file to force (e.g. the adapter's training template)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--max-prompt-chars", type=int, default=13000,
                    help="skip rows whose prompt messages exceed this (~3.3k tok) to stay in context")
    ap.add_argument("--models", nargs="+", required=True,
                    help="first must be the base served-model-name")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    chat_template = Path(args.chat_template).read_text() if args.chat_template else None

    rows = []
    with open(args.val_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    per_example, skipped = [], 0
    for row in rows:
        if len(per_example) >= args.n:
            break
        msgs = row["messages"]
        assert msgs[-1]["role"] == "assistant"
        prompt_msgs = msgs[:-1]
        reference = msgs[-1]["content"]
        nchars = sum(len(m["content"]) for m in prompt_msgs)
        if nchars > args.max_prompt_chars:
            skipped += 1
            continue
        gens, err = {}, None
        for m in args.models:
            g, e = chat(args.base_url, m, prompt_msgs, args.max_new_tokens, chat_template)
            if e:
                err = f"{m}: {e}"
                break
            gens[m] = g
        if err:
            print(f"  skip {row.get('item_id','')}: {err}")
            skipped += 1
            continue
        per_example.append({
            "item_id": row.get("item_id", str(len(per_example))),
            "prompt_chars": nchars,
            "generations": gens,
            "reference": reference,
            "reference_head": reference[:160],
        })
        print(f"[{len(per_example)}/{args.n}] {row.get('item_id','')} ({nchars} chars)")
    print(f"(used {len(per_example)}; skipped {skipped})")

    ms = args.models

    def pair_stats(a, b):
        exact, pref = 0, []
        for ex in per_example:
            ga, gb = ex["generations"][a], ex["generations"][b]
            if ga == gb:
                exact += 1
            m = max(len(ga), len(gb))
            pref.append(char_lcp(ga, gb) / m if m else 1.0)
        k = len(per_example) or 1
        return {
            "exact_match_frac": exact / k,
            "mean_char_prefix_agreement": sum(pref) / k,
        }

    pairs = {}
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            pairs[f"{ms[i]}__vs__{ms[j]}"] = pair_stats(ms[i], ms[j])

    # Per-model quality vs the ground-truth reference: does this arm's generation move
    # toward the fine-tune target? base < shared-only < full would mean attention LoRA
    # adds retention on top of shared-expert (the FP8-train decision input).
    def ref_stats(model):
        f1s, prefs = [], []
        for ex in per_example:
            g, ref = ex["generations"][model], ex["reference"]
            f1s.append(token_f1(g, ref))
            m = max(len(g), len(ref))
            prefs.append(char_lcp(g, ref) / m if m else 1.0)
        k = len(per_example) or 1
        return {"mean_token_f1_vs_ref": sum(f1s) / k,
                "mean_char_prefix_vs_ref": sum(prefs) / k}

    per_model_vs_reference = {m: ref_stats(m) for m in ms}

    summary = {
        "n": len(per_example),
        "skipped": skipped,
        "max_new_tokens": args.max_new_tokens,
        "models": ms,
        "pairs": pairs,
        "per_model_vs_reference": per_model_vs_reference,
    }
    Path(args.out).write_text(
        json.dumps({"summary": summary, "examples": per_example}, indent=2)
    )
    print("\n=== SERVE-PARITY SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
