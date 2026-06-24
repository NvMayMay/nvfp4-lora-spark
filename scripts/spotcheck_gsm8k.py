#!/usr/bin/env python
"""Qualitative base-vs-adapter spot-check on GSM8K (a few problems, full text).

Unlike eval_gsm8k.py (which scores N problems for an accuracy number), this prints
the actual generations side by side so you can SEE what the adapter changed -
answer correctness AND reasoning/format style. stdlib HTTP only.

  python scripts/spotcheck_gsm8k.py --test-file gsm8k/test.chat.jsonl \
      --models base myft --n 6 --show-text 2
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path


def ask(base_url, model, q, max_tokens, extra=None):
    body = {"model": model, "messages": [{"role": "user", "content": q}],
            "max_tokens": max_tokens, "temperature": 0.0, "seed": 0}
    if extra:
        body.update(extra)
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def last_int(s):
    n = re.findall(r"-?\d[\d,]*", s or "")
    return n[-1].replace(",", "") if n else None


def gold_int(a):
    m = re.search(r"####\s*(-?[\d,]+)", a)
    return m.group(1).replace(",", "") if m else last_int(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--show-text", type=int, default=2, help="how many problems to print full generations for")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--instruction", default="",
                    help="appended to each question; gives the base a fair shot at the answer format")
    ap.add_argument("--no-think", action="store_true",
                    help="set chat_template_kwargs.enable_thinking=False (fair 0-shot vs a thinking base)")
    args = ap.parse_args()

    extra = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_think else None
    rows = [json.loads(l) for l in open(args.test_file) if l.strip()][:args.n]
    correct = {m: 0 for m in args.models}
    gens = []
    for i, row in enumerate(rows):
        q = row["messages"][0]["content"]
        if args.instruction:
            q = q + "\n\n" + args.instruction
        g = gold_int(row["messages"][-1]["content"])
        outs = {m: ask(args.base_url, m, q, args.max_new_tokens, extra) for m in args.models}
        gens.append((q, g, outs))
        marks = []
        for m in args.models:
            p = last_int(outs[m])
            ok = (p is not None and p == g)
            correct[m] += int(ok)
            marks.append(f"{m}={p} {'OK' if ok else 'X'}")
        print(f"Q{i+1} gold={g} | " + " | ".join(marks), flush=True)

    for i in range(min(args.show_text, len(gens))):
        q, g, outs = gens[i]
        print(f"\n===== Q{i+1} (gold={g}) =====")
        print("Q:", q.strip()[:300])
        for m in args.models:
            t = outs[m].strip()
            print(f"\n--- {m} ---\n{t[:600]}{' ...[trunc]' if len(t) > 600 else ''}")

    print("\ntally: " + ", ".join(f"{m} {correct[m]}/{len(rows)}" for m in args.models))


if __name__ == "__main__":
    main()
