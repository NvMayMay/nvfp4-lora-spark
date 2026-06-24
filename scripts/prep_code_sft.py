#!/usr/bin/env python
"""Format a code-instruct dataset into chat jsonl for the trainer, with decontamination.

Default source: bigcode/self-oss-instruct-sc2-exec-filter-50k (StarCoder2 self-OSS-Instruct):
Python instruction->response pairs, execution-filtered, documented decontam vs HumanEval/MBPP,
StarCoder2-generated (no OpenAI-terms baggage). We add a belt-and-suspenders decontamination
pass against the actual HumanEval prompts before training, so the before/after number is clean.

Output rows: {"messages": [{"role":"user","content":...},{"role":"assistant","content":...}]}

  python scripts/prep_code_sft.py --out-dir /path/code_sft \
      --n 12000 --val-n 200 --decontam-file humaneval_prompts.txt
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _shingles(text, k=13):
    """Word k-grams, lowercased, for near-duplicate / contamination detection."""
    words = re.findall(r"\w+", (text or "").lower())
    return {" ".join(words[i:i + k]) for i in range(max(0, len(words) - k + 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bigcode/self-oss-instruct-sc2-exec-filter-50k")
    ap.add_argument("--split", default="train")
    ap.add_argument("--instruction-field", default="instruction")
    ap.add_argument("--response-field", default="response")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=12000, help="train examples to keep after filtering")
    ap.add_argument("--val-n", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=64)
    ap.add_argument("--max-chars", type=int, default=7000, help="instruction+response char cap (~2048 tok)")
    ap.add_argument("--require-code", action="store_true", default=True,
                    help="keep only responses containing a fenced code block")
    ap.add_argument("--streaming", action="store_true",
                    help="stream the dataset (for very large sources like OpenCodeInstruct)")
    ap.add_argument("--score-field", default="",
                    help="numeric quality field to filter on (e.g. average_test_score)")
    ap.add_argument("--min-score", type=float, default=1.0,
                    help="keep rows with score-field >= this (verified-correct = 1.0)")
    ap.add_argument("--decontam-file", default="",
                    help="text file of HumanEval prompts; drop train rows sharing a 13-gram with any")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset

    contam = set()
    if args.decontam_file and Path(args.decontam_file).exists():
        for line in Path(args.decontam_file).read_text().split("\n<<<DOC>>>\n"):
            contam |= _shingles(line)
        print(f"[decontam] {len(contam)} HumanEval 13-grams loaded", flush=True)

    if args.streaming:
        ds = load_dataset(args.dataset, split=args.split, streaming=True).shuffle(
            seed=args.seed, buffer_size=20000)
        print(f"[load] {args.dataset}:{args.split} (streaming)", flush=True)
    else:
        ds = load_dataset(args.dataset, split=args.split)
        print(f"[load] {args.dataset}:{args.split} = {len(ds)} rows", flush=True)
        ds = ds.shuffle(seed=args.seed)

    kept, dropped_len, dropped_code, dropped_contam, dropped_score = [], 0, 0, 0, 0
    for row in ds:
        if args.score_field:
            sc = row.get(args.score_field)
            try:
                if sc is None or float(sc) < args.min_score:
                    dropped_score += 1
                    continue
            except (TypeError, ValueError):
                dropped_score += 1
                continue
        instr = (row.get(args.instruction_field) or "").strip()
        resp = (row.get(args.response_field) or "").strip()
        if not instr or not resp:
            continue
        if not (args.min_chars <= len(instr) + len(resp) <= args.max_chars):
            dropped_len += 1
            continue
        if args.require_code and "```" not in resp:
            dropped_code += 1
            continue
        if contam and (_shingles(instr) & contam):
            dropped_contam += 1
            continue
        kept.append({"messages": [{"role": "user", "content": instr},
                                  {"role": "assistant", "content": resp}]})
        if len(kept) >= args.n + args.val_n:
            break

    val = kept[:args.val_n]
    train = kept[args.val_n:]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        p = out / f"code_sft.{name}.chat.jsonl"
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[write] {p}  ({len(rows)} rows)", flush=True)
    print(f"[filter] dropped: score={dropped_score} len={dropped_len} no-code={dropped_code} "
          f"contam={dropped_contam}", flush=True)
    if train:
        ex = train[0]["messages"]
        print("\n[sample] user:", ex[0]["content"][:200].replace("\n", " "))
        print("[sample] assistant:", ex[1]["content"][:200].replace("\n", " "))


if __name__ == "__main__":
    main()
