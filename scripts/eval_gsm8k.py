#!/usr/bin/env python
"""GSM8K exact-match accuracy through a running vLLM server (base vs adapter).

The public before/after demo: greedy-decode GSM8K test questions under each served model
name, extract the final integer from the generation, exact-match it against the gold
(`#### N`), and report per-model accuracy. This is a real benchmark metric (not a token
overlap proxy) and is deterministic up to vLLM's greedy non-determinism. stdlib HTTP only.

Usage:
  python scripts/eval_gsm8k.py --test-file gsm8k/test.chat.jsonl \
      --models <base-served-name> <adapter-name> --n 200 --out gsm8k_eval.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


def chat(base_url, model, question, max_tokens, timeout=600, extra=None):
    body = {"model": model, "messages": [{"role": "user", "content": question}],
            "max_tokens": max_tokens, "temperature": 0.0, "seed": 0}
    if extra:
        body.update(extra)
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r and json.load(r)["choices"][0]["message"]["content"], None
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


def _last_int(s):
    nums = re.findall(r"-?\d[\d,]*", s or "")
    return int(nums[-1].replace(",", "")) if nums else None


def _gold_int(answer):
    m = re.search(r"####\s*(-?[\d,]+)", answer)
    return int(m.group(1).replace(",", "")) if m else _last_int(answer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--test-file", required=True, help="gsm8k test as chat jsonl ({messages:[user,assistant]})")
    ap.add_argument("--models", nargs="+", required=True, help="served model names (first = base)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--instruction", default="",
                    help="appended to each question; give the base a fair shot at the answer format "
                         "so the metric reflects capability, not format adherence")
    ap.add_argument("--no-think", action="store_true",
                    help="set chat_template_kwargs.enable_thinking=False (fair 0-shot vs a thinking base)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    extra = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_think else None
    rows = [json.loads(l) for l in open(args.test_file) if l.strip()][:args.n]
    correct = {m: 0 for m in args.models}
    used, per = 0, []
    for row in rows:
        msgs = row["messages"]
        question, gold = msgs[0]["content"], _gold_int(msgs[-1]["content"])
        if args.instruction:
            question = question + "\n\n" + args.instruction
        preds, err = {}, None
        for m in args.models:
            gen, e = chat(args.base_url, m, question, args.max_new_tokens, extra=extra)
            if e:
                err = f"{m}: {e}"
                break
            p = _last_int(gen)
            preds[m] = p
            if p is not None and p == gold:
                correct[m] += 1
        if err:
            print(f"  skip: {err}")
            continue
        used += 1
        per.append({"gold": gold, "pred": preds})
        if used % 20 == 0:
            print(f"[{used}/{len(rows)}]  running acc: " +
                  ", ".join(f"{m}={correct[m]/used:.3f}" for m in args.models))

    acc = {m: (correct[m] / used if used else 0.0) for m in args.models}
    summary = {"n": used, "models": args.models, "accuracy": acc,
               "lift_vs_first": {m: acc[m] - acc[args.models[0]] for m in args.models[1:]}}
    Path(args.out).write_text(json.dumps({"summary": summary, "per_example": per}, indent=2))
    print("\n=== GSM8K exact-match accuracy ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
