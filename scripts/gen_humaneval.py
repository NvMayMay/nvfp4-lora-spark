#!/usr/bin/env python
"""Generate HumanEval(+) completions via a vLLM OpenAI endpoint -> EvalPlus samples.jsonl.

Greedy, thinking OFF by default (fair vs a Qwen3 thinking base - same fix as the GSM8K eval),
extracts the python code block. Score the output with EvalPlus:

  evalplus.sanitize --samples <out.jsonl>            # -> <out>-sanitized.jsonl
  evalplus.evaluate --dataset humaneval --samples <out>-sanitized.jsonl
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import threading
import urllib.request
from pathlib import Path


def chat(base_url, model, content, max_tokens, no_think, timeout=600):
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0.0, "seed": 0}
    if no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def extract_code(text, entry_point):
    """Prefer a fenced block that defines entry_point; else first fenced block; else raw."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text or "", re.DOTALL)
    for b in blocks:
        if f"def {entry_point}" in b:
            return b
    return blocks[0] if blocks else (text or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True, help="served model name (base / myft)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--workers", type=int, default=8, help="concurrent requests (server batches them)")
    ap.add_argument("--limit", type=int, default=0, help="first N problems only (0 = all 164)")
    args = ap.parse_args()

    from evalplus.data import get_human_eval_plus
    items = list(get_human_eval_plus().items())
    if args.limit:
        items = items[:args.limit]

    tmpl = ("Complete the following Python function. Reply with the complete function in a single "
            "```python code block and nothing else.\n\n```python\n{p}\n```")
    done = {"n": 0, "errs": 0}
    lock = threading.Lock()

    def work(item):
        tid, p = item
        try:
            gen = chat(args.base_url, args.model, tmpl.format(p=p["prompt"]),
                       args.max_new_tokens, args.no_think)
            err = False
        except Exception as e:  # noqa: BLE001
            print(f"  {tid} ERR {str(e)[:120]}", flush=True)
            gen, err = "", True
        with lock:
            done["n"] += 1
            done["errs"] += int(err)
            if done["n"] % 40 == 0:
                print(f"[{done['n']}/{len(items)}]", flush=True)
        return {"task_id": tid, "solution": extract_code(gen, p["entry_point"])}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        rows = list(ex.map(work, items))
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} samples ({done['errs']} gen errors) -> {args.out}")


if __name__ == "__main__":
    main()
