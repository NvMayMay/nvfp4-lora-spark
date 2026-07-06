#!/usr/bin/env python
"""Deterministic before/after VQA retention eval through a served OpenAI endpoint.

Vision analog of scripts/eval_retention.py (which does text-to-SQL): scores a BASE
served-model-name against a MERGED served-model-name on the vqa-rad val set, by
generating a short answer for each {image, question} and scoring normalized
EXACT-MATCH against the gold answer. It talks to an OpenAI-compatible endpoint the
same way eval_retention does (base URL + served model names), except images are
sent as base64 data URLs in the chat message (vLLM's multimodal chat schema).

Metric: VQA-style normalized exact-match (EM). Lowercase, strip punctuation, drop
the articles a/an/the, collapse whitespace; a prediction is correct iff its
normalized form equals the normalized gold. Report per-model EM + delta vs base,
and dump per-row predictions (mirrors eval_retention.py's --out JSON shape:
{"summary": {...}, "per_example": [...]}).

Input rows are the nvfp4_lora.mm_data JSONL shape (one object per line):

  {"messages": [{"role": "user",
                 "content": [{"type": "image"},
                             {"type": "text", "text": "Is there consolidation?"}]},
                {"role": "assistant",
                 "content": [{"type": "text", "text": "yes"}]}],
   "images": ["images/vqarad_val_000001.png"]}

`images` paths resolve relative to the dev-file's directory (same rule as
nvfp4_lora.mm_data), and are inlined as base64 data URLs.

  python scripts/eval_vision_retention.py --dev-file data/vqa_rad/val.jsonl \
      --models mistral-base mistral-vision-merged --n 60 --out vqa_retention.json

The MERGED model is served by serve/run_mistral24b_vision_merged.sh (a vision-tower
adapter has no runtime-LoRA path -- see docs/SERVING.md section 6). This script is
serve-only: it does not train, merge, or load a model.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# HTTP helper (stdlib only, same shape as scripts/eval_retention.py)
# ---------------------------------------------------------------------------


def _post(base_url, path, body, timeout=600):
    req = urllib.request.Request(base_url.rstrip("/") + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r), None
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if attempt == 1:
                return None, f"HTTP {e.code}: {msg}"
            time.sleep(2)
        except Exception as e:  # noqa: BLE001
            if attempt == 1:
                return None, str(e)[:300]
            time.sleep(2)
    return None, "unreachable"


# ---------------------------------------------------------------------------
# VQA-style answer normalization + exact-match (pure, unit-testable).
# Mirrors the canonical VQA / SQuAD normalization: lowercase, strip punctuation,
# drop articles, collapse whitespace. No DB, no decoding -- deterministic.
# ---------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_vqa(ans: str) -> str:
    """Normalize an answer for VQA exact-match.

    lower -> strip punctuation -> drop the articles a/an/the -> collapse
    whitespace. "The X-ray." and "x ray" both normalize to "x ray"; "Yes!" -> "yes".
    """
    s = (ans or "").lower()
    s = _PUNCT_RE.sub(" ", s)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)


def vqa_exact_match(pred: str, gold: str) -> bool:
    """True iff the prediction normalizes to exactly the gold answer."""
    return normalize_vqa(pred) == normalize_vqa(gold)


# ---------------------------------------------------------------------------
# Row decoding + image encoding (pure, unit-testable).
# ---------------------------------------------------------------------------

def extract_row(row: dict) -> tuple[str, str, list]:
    """Pull (question_text, gold_answer, image_parts) from one mm_data row.

    `image_parts` is the ordered list of `{"type":"image"}` placeholders in the
    user turn (aligned 1:1 with the row's `images` sidecar). The question is the
    concatenation of the user turn's text parts; the gold is the assistant turn's
    text. Fail loud on a malformed row (mirrors mm_data's fail-fast validation).
    """
    msgs = row.get("messages")
    if not msgs:
        raise ValueError("row missing 'messages'")
    user = next((m for m in msgs if m.get("role") == "user"), None)
    asst = next((m for m in msgs if m.get("role") == "assistant"), None)
    if user is None or asst is None:
        raise ValueError("row needs both a user and an assistant turn")

    q_parts, image_parts = [], []
    ucontent = user.get("content")
    if isinstance(ucontent, list):
        for part in ucontent:
            if part.get("type") == "image":
                image_parts.append(part)
            elif part.get("type") == "text":
                q_parts.append(part.get("text", ""))
    else:
        q_parts.append(str(ucontent))

    acontent = asst.get("content")
    if isinstance(acontent, list):
        gold = " ".join(p.get("text", "") for p in acontent if p.get("type") == "text")
    else:
        gold = str(acontent)

    return "\n".join(q_parts).strip(), gold.strip(), image_parts


_MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                   ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"}


def image_data_url(path: Path) -> str:
    """Inline an image file as a base64 `data:` URL (vLLM multimodal chat schema)."""
    suffix = path.suffix.lower()
    mime = _MIME_BY_SUFFIX.get(suffix, "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_chat_messages(question: str, data_urls: list) -> list:
    """OpenAI-compatible chat messages: image_url parts + the question text.

    One user turn whose content is [image_url, image_url, ..., text]; each image is
    a `{"type":"image_url","image_url":{"url": <data-url>}}` part, matching how vLLM
    accepts images on /v1/chat/completions.
    """
    content = [{"type": "image_url", "image_url": {"url": u}} for u in data_urls]
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def resolve_images(image_parts: list, sidecar: list, base_dir: Path,
                   row_index: int) -> list:
    """Resolve the row's `images` sidecar to data URLs, aligned to the placeholders."""
    if len(image_parts) != len(sidecar):
        raise ValueError(
            f"row {row_index}: {len(image_parts)} image part(s) but {len(sidecar)} "
            f"path(s) in 'images' -- must align 1:1"
        )
    urls = []
    for rel in sidecar:
        p = Path(rel)
        if not p.is_absolute():
            p = base_dir / p
        if not p.exists():
            raise ValueError(f"row {row_index}: image file not found: {p}")
        urls.append(image_data_url(p))
    return urls


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def gen_answer(base_url, model, messages, max_tokens=32, timeout=600,
               chat_template_kwargs=None, strip_think=False):
    """Greedy short answer via /v1/chat/completions (temperature 0, seed 0).

    Returns (text, err). The server applies the model's own chat template and fuses
    the inlined images; we only ask for a short phrase.

    `chat_template_kwargs` (e.g. {"enable_thinking": false}) is passed to the server so a
    REASONING model (NemotronH-Omni) emits a direct answer instead of a <think> monologue;
    `strip_think` additionally removes any <think>...</think> that still leaks, so the EM is
    scored on the answer only. Both are no-ops (defaults) for non-reasoning models (Pixtral).
    """
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "seed": 0}
    if chat_template_kwargs:
        body["chat_template_kwargs"] = chat_template_kwargs
    resp, err = _post(base_url, "/v1/chat/completions", body, timeout)
    if err:
        return None, err
    text = resp["choices"][0]["message"].get("content") or ""
    if strip_think:
        text = _THINK_RE.sub("", text).strip()
    return text, None


# ---------------------------------------------------------------------------
# Summary (mirrors eval_retention: per-model EM, delta vs base, per-row preds)
# ---------------------------------------------------------------------------

def _round_or_none(v, ndigits=5):
    return round(v, ndigits) if v is not None else None


def build_summary(per_example: list, models: list) -> dict:
    """Per-model EM (over rows where EVERY model produced a paired answer) + delta."""
    base = models[0]
    paired = [rec for rec in per_example if all(m in rec.get("em", {}) for m in models)]
    summary = {"models": list(models), "n": len(per_example), "n_paired": len(paired)}
    em = {}
    for m in models:
        vals = [rec["em"][m] for rec in paired]
        em[m] = (sum(vals) / len(vals)) if vals else None
    summary["exact_match"] = {m: _round_or_none(em[m]) for m in models}
    summary["em_delta_vs_base"] = {
        m: _round_or_none((em[m] - em[base]) if em[m] is not None and em[base] is not None else None)
        for m in models[1:]
    }
    summary["empty_generations"] = {
        m: sum(1 for rec in per_example if m in rec.get("em", {})
               and not (rec.get("pred", {}).get(m) or "").strip())
        for m in models
    }
    warnings_out = []
    for m in models[1:]:
        if em.get(m) is not None and em.get(base) is not None and em[m] == em[base] and paired:
            warnings_out.append(
                f"merged model '{m}' produced IDENTICAL EM to base -- if predictions are "
                f"also identical the vision merge may be a no-op (check merge_stats.jsonl / "
                f"that the MERGED dir is being served, not the base).")
        if em.get(m) == 0.0 and summary["empty_generations"].get(m, 0) >= len(paired) and paired:
            warnings_out.append(
                f"exact-match for '{m}' is 0.0 but ALL generations were EMPTY -- a "
                f"measurement FAILURE (server/template/image-fuse issue), not a real score.")
    if warnings_out:
        summary["warnings"] = warnings_out
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--dev-file", required=True,
                    help="vqa-rad val as mm_data JSONL ({messages:[...], images:[...]})")
    ap.add_argument("--models", nargs="+", required=True,
                    help="served model names (first = base, rest = merged VLMs)")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chat-template-kwargs", default=None,
                    help='JSON passed as chat_template_kwargs to the server, e.g. '
                         '\'{"enable_thinking": false}\' to make a reasoning model (NemotronH-Omni) '
                         'answer directly instead of emitting a <think> monologue.')
    ap.add_argument("--strip-think", action="store_true",
                    help="Strip any <think>...</think> from the response before scoring EM "
                         "(belt-and-suspenders for reasoning models).")
    args = ap.parse_args()
    ctk = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None

    dev_path = Path(args.dev_file)
    base_dir = dev_path.parent
    rows = [json.loads(l) for l in open(dev_path) if l.strip()][:args.n]
    print(f"[load] {len(rows)} val rows from {dev_path}", flush=True)

    em_correct = {m: 0 for m in args.models}
    em_n = {m: 0 for m in args.models}
    em_skips = {m: 0 for m in args.models}
    em_empty = {m: 0 for m in args.models}
    per, used = [], 0

    for i, row in enumerate(rows):
        try:
            question, gold, image_parts = extract_row(row)
            data_urls = resolve_images(image_parts, row.get("images", []) or [], base_dir, i)
        except ValueError as e:
            print(f"  note: skipping row {i}: {e}", flush=True)
            continue
        messages = build_chat_messages(question, data_urls)

        rec = {"gold": gold, "question": question, "pred": {}, "em": {}}
        row_em = {}
        for m in args.models:
            text, err = gen_answer(args.base_url, m, messages, args.max_new_tokens,
                                   chat_template_kwargs=ctk, strip_think=args.strip_think)
            if err:
                row_em[m] = None
                em_skips[m] += 1
                if em_skips[m] == 1:
                    print(f"  note: generation failed for {m}: {err}", flush=True)
                continue
            pred = text.strip()
            if not pred:
                em_empty[m] += 1
                if em_empty[m] == 1:
                    print(f"  note: empty generation for {m}", flush=True)
            row_em[m] = vqa_exact_match(pred, gold)
            rec["pred"][m] = pred[:200]

        # Accumulate EM only on rows where EVERY model answered (paired before/after).
        if all(row_em.get(m) is not None for m in args.models):
            for m in args.models:
                rec["em"][m] = row_em[m]
                em_correct[m] += int(row_em[m])
                em_n[m] += 1

        if not rec["em"]:
            continue
        used += 1
        per.append(rec)
        if used % 20 == 0:
            msg = f"[{used}/{len(rows)}]  EM: " + ", ".join(
                f"{m}={em_correct[m]/em_n[m]:.3f}" for m in args.models if em_n[m])
            print(msg, flush=True)

    summary = build_summary(per, args.models)
    summary["skipped"] = em_skips
    summary["empty_generations"] = em_empty
    warnings_out = list(summary.pop("warnings", []))
    for m in args.models:
        if em_n[m] == 0:
            warnings_out.append(
                f"exact-match is 0/0 for '{m}': all {em_skips[m]} generations failed "
                f"(reported 0.0 is a FAILURE, not a real score). Check the served model "
                f"name, that the endpoint accepts images, and the image data URLs.")
    if warnings_out:
        summary["warnings"] = warnings_out

    Path(args.out).write_text(
        json.dumps({"summary": summary, "per_example": per}, indent=2, sort_keys=True)
    )
    print("\n=== VQA vision retention (base vs merged) ===")
    print(json.dumps(summary, indent=2))
    for w in warnings_out:
        print(f"WARNING: {w}", flush=True)
    print(f"\n[write] {args.out}")


if __name__ == "__main__":
    main()
