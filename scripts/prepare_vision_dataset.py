#!/usr/bin/env python
"""Prepare a small PUBLIC image+text VQA set as the vision-LoRA on-ramp dataset.

Public, deterministic before/after demo for `--train-target vision` (the vision analog
of scripts/prep_spider.py for the text retention demo). Source dataset:

    flaviagiammarino/vqa-rad  (HF, ~1.8k train / 451 test, CC0-1.0)

Why this one (evaluated against the alternatives in the plan's Open Q1):
  * Loads via `datasets` with NO gated / auth access, and the images are INLINE PIL
    objects in the parquet -- one `load_dataset` call, no side download, ~30s.
  * SHORT answers (mostly single-word, ~60% yes/no), so the before/after metric is a
    clean deterministic EXACT-MATCH (normalized) -- far more reproducible than caption
    BLEU (the plan prefers exact-match/ANLS over captioning).
  * CC0-1.0 (public domain) license -- safe to reference from an OSS repo's docs.
  * Radiology VQA is domain-ADJACENT for a clinical/regulatory shop (matches the
    doc/medical-image understanding use case) while being fully public -- the private
    synthetic doc-image set (plan Q1) is for the real internal run, this is the shippable
    public reproduction. `flaviagiammarino/vqa-rad` is the plan's named medical candidate.

Output (the repo's multimodal JSONL shape, scoping doc section B):

    <out>/train.jsonl, <out>/val.jsonl   -- one row per line:
      {"messages": [{"role":"user","content":[{"type":"image"},
                                              {"type":"text","text":<question+instruction>}]},
                    {"role":"assistant","content":[{"type":"text","text":<answer>}]}],
       "images": ["images/vqarad_train_000001.png"]}
    <out>/images/*.png                   -- the decoded RGB images (paths in the JSONL
                                            are RELATIVE to the JSONL's directory, which
                                            is how nvfp4_lora.mm_data resolves them)

Only a SMALL subset is written by default (a ~30-minute on-ramp, not the full set); the
full dataset is never committed. Deterministic: the first N rows of each split in order.

    python scripts/prepare_vision_dataset.py --out-dir data/vqa_rad --n 300 --val-n 60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

INSTRUCTION = (
    "Answer the question about the image with a short phrase, no explanation."
)


def _row(question: str, answer: str, image_rel: str) -> dict:
    """One multimodal chat row: image + question -> short answer."""
    prompt = f"{question.strip()}\n\n{INSTRUCTION}"
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": answer.strip()},
            ]},
        ],
        "images": [image_rel],
    }


def _downscale(img, max_edge: int):
    """Cap the longest edge to `max_edge` (preserve aspect), so a high-res scan does
    not blow past `--max-length` once Pixtral patchifies it (a 1000-px image is ~1300
    image tokens; the mm_data collator fail-loud-refuses to truncate a mid-image run).
    `max_edge=0` disables. No-op when the image already fits."""
    if not max_edge:
        return img
    w, h = img.size
    if max(w, h) <= max_edge:
        return img
    from PIL import Image
    scale = max_edge / max(w, h)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def _prepare_split(ds, split_tag: str, cap: int, out_dir: Path, images_dir: Path,
                   sample_shown: list, max_edge: int) -> int:
    n = min(cap, len(ds)) if cap else len(ds)
    rows = []
    for i in range(n):
        ex = ds[i]
        img = _downscale(ex["image"].convert("RGB"), max_edge)
        image_rel = f"images/vqarad_{split_tag}_{i:06d}.png"
        img.save(images_dir / f"vqarad_{split_tag}_{i:06d}.png")
        row = _row(ex["question"], str(ex["answer"]), image_rel)
        rows.append(row)
        if not sample_shown:
            print("\n" + "=" * 72)
            print(f"[sample] split={split_tag}  image={image_rel} size={img.size}")
            print("[USER]  ", row["messages"][0]["content"][1]["text"].replace("\n", " "))
            print("[ANSWER]", row["messages"][1]["content"][0]["text"])
            print("=" * 72 + "\n", flush=True)
            sample_shown.append(True)
    out_path = out_dir / f"{split_tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {out_path}  ({len(rows)} rows)", flush=True)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="flaviagiammarino/vqa-rad")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="test")
    ap.add_argument("--n", type=int, default=300, help="cap train rows (0 = all)")
    ap.add_argument("--val-n", type=int, default=60, help="cap val rows (0 = all)")
    ap.add_argument("--max-image-edge", type=int, default=768,
                    help="downscale longest image edge to this many px so examples fit "
                         "under --max-length 2048 (0 = keep native resolution)")
    args = ap.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    sample_shown: list = []
    for split, tag, cap in (
        (args.train_split, "train", args.n),
        (args.val_split, "val", args.val_n),
    ):
        ds = load_dataset(args.dataset, split=split)
        print(f"[load] {args.dataset}:{split} = {len(ds)} rows", flush=True)
        _prepare_split(ds, tag, cap, out_dir, images_dir, sample_shown, args.max_image_edge)

    print(
        f"\n[done] wrote {out_dir}/train.jsonl + {out_dir}/val.jsonl + {images_dir}/*.png\n"
        f"       train with: python scripts/train_nvfp4_lora.py --train-target vision \\\n"
        f"         --model-dir <Mistral-Small-3.2-24B-NVFP4> --train-file {out_dir}/train.jsonl \\\n"
        f"         --val-file {out_dir}/val.jsonl --vision-target-modules linear_1,linear_2 \\\n"
        f"         --max-length 2048 --batch-size 1 --checkpoint-every 50 --output-dir <out>",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
