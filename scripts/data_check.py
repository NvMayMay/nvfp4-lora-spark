#!/usr/bin/env python
"""Training-data preflight -- the "data doctor" for chat-jsonl SFT datasets.

A quiet training run can still be silently broken by its DATA: a chat template that
puts the whole turn behind a role header the mask never reaches (0 supervised tokens
per row -> the loss is computed on nothing), a max_length that truncates away most of
the gold answers, or a tokenizer/template that has drifted from the one you thought you
were training with. This script renders and tokenizes a chat jsonl the same way the
trainer does (nvfp4_lora.chat_encode) and reports, before you burn a GPU-hour:

  * chat-template preview -- the first row rendered exactly as the trainer sees it, so a
    human can eyeball the role headers / thinking scaffold / EOS placement.
  * assistant-mask coverage -- fraction of rows with >0 SUPERVISED (assistant) tokens and
    a count of rows with an empty assistant span. A low coverage means the template/mask
    is not lining up and the loss is being computed on little or nothing.
  * truncation drop -- how many rows (and what fraction) exceed max_length, i.e. have
    their tail (usually the gold answer) cut off at this max_length.
  * token-length histogram -- bucketed full-example token lengths, so you can pick a
    max_length that keeps the answers rather than guessing.
  * tokenizer / template hashes -- so a data-check result is pinned to an exact tokenizer
    + chat template (drift is a silent-corruption source this project keeps hitting).

The GPU-independent logic (mask coverage, truncation counting, histogram bucketing) is
factored into pure functions that take token-id lists / simple inputs, so they can be
unit-tested WITHOUT loading a real tokenizer or model. The tokenizer-dependent rendering
lives behind a thin wrapper (`encode_rows`) that the CLI calls.

  python scripts/data_check.py --data spider/spider.train.chat.jsonl \
      --tokenizer /models/Qwen3.5-122B-NVFP4 --max-length 2048 --out data_check.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Make the `nvfp4_lora` package importable when run as a loose script from scripts/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Default histogram bucket upper-bounds (token counts). The final open bucket catches
# everything above the last edge. Chosen to bracket the common 512/1024/2048/4096 caps.
DEFAULT_BUCKETS = [128, 256, 512, 1024, 2048, 4096, 8192]


# ---------------------------------------------------------------------------
# Pure, tokenizer-free logic (unit-testable with plain lists of ints).
# ---------------------------------------------------------------------------


def n_supervised(labels) -> int:
    """Number of supervised (non-masked) tokens in a label list.

    The trainer masks non-assistant tokens to -100 (see nvfp4_lora.chat_encode); a
    supervised token is any label != -100.
    """
    return sum(1 for label in labels if label != -100)


def mask_coverage(rows_labels) -> dict:
    """Assistant-mask coverage over many rows' label lists.

    `rows_labels` is an iterable of per-row label lists. Returns the number of rows, the
    number with >0 supervised tokens, the number with an EMPTY assistant span (0
    supervised tokens -- the row the trainer learns nothing from), and the covered
    fraction. Empty input yields zeros with fraction 0.0 (never a ZeroDivisionError).
    """
    rows = list(rows_labels)
    n_rows = len(rows)
    supervised_counts = [n_supervised(labels) for labels in rows]
    n_covered = sum(1 for c in supervised_counts if c > 0)
    n_empty = n_rows - n_covered
    total_supervised = sum(supervised_counts)
    return {
        "n_rows": n_rows,
        "n_covered": n_covered,
        "n_empty_assistant_span": n_empty,
        "covered_fraction": (n_covered / n_rows) if n_rows else 0.0,
        "total_supervised_tokens": total_supervised,
        "mean_supervised_tokens": (total_supervised / n_rows) if n_rows else 0.0,
    }


def truncation_stats(token_lengths, max_length: int) -> dict:
    """How many rows would be truncated at `max_length`.

    `token_lengths` is an iterable of per-row FULL (pre-truncation) token counts. A row
    is "truncated" (its tail dropped) iff its length > max_length. Returns the count and
    fraction dropped. Empty input yields zeros with fraction 0.0.
    """
    lengths = list(token_lengths)
    n_rows = len(lengths)
    n_truncated = sum(1 for length in lengths if length > max_length)
    return {
        "max_length": max_length,
        "n_rows": n_rows,
        "n_truncated": n_truncated,
        "truncated_fraction": (n_truncated / n_rows) if n_rows else 0.0,
    }


def histogram(token_lengths, buckets=None) -> dict:
    """Bucket per-row token lengths into a coarse histogram.

    `buckets` is a sorted list of inclusive upper-bounds; a final open bucket "<edge>+"
    catches everything strictly above the last edge. Returns an ordered list of
    {"label", "lo", "hi", "count"} entries plus min/max/total for context. A length L
    falls in the first bucket whose upper-bound hi satisfies L <= hi.
    """
    lengths = list(token_lengths)
    edges = sorted(buckets if buckets is not None else DEFAULT_BUCKETS)
    entries = []
    lo = 0
    for hi in edges:
        entries.append({"label": f"{lo}-{hi}", "lo": lo, "hi": hi, "count": 0})
        lo = hi + 1
    # final open bucket
    last_edge = edges[-1] if edges else 0
    entries.append({"label": f"{last_edge + 1}+", "lo": last_edge + 1, "hi": None, "count": 0})

    for length in lengths:
        placed = False
        for entry in entries[:-1]:
            if length <= entry["hi"]:
                entry["count"] += 1
                placed = True
                break
        if not placed:
            entries[-1]["count"] += 1

    return {
        "buckets": entries,
        "n_rows": len(lengths),
        "min": min(lengths) if lengths else None,
        "max": max(lengths) if lengths else None,
        "total_tokens": sum(lengths),
    }


def _sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def tokenizer_hashes(tokenizer_dir) -> dict:
    """SHA-256 of the files that pin a tokenizer + chat template.

    Same file set as nybbloris.manifest._tokenizer_fingerprint, so a data-check result
    can be cross-referenced against an adapter manifest. Missing files are simply absent
    from the dict (not an error) -- e.g. a template baked into tokenizer_config.json has
    no separate chat_template.jinja.
    """
    tokenizer_dir = Path(tokenizer_dir)
    out = {}
    for fn in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "chat_template.jinja"):
        p = tokenizer_dir / fn
        if p.exists():
            out[fn] = _sha256_file(p)
    return out


# ---------------------------------------------------------------------------
# Thin tokenizer-dependent wrapper (not exercised by the CPU unit tests).
# ---------------------------------------------------------------------------


def load_rows(data_path, limit: int = 0):
    """Read a chat jsonl ({"messages": [...]}) into a list of message-lists.

    `limit > 0` caps the number of rows read. Rows without a "messages" key are skipped.
    """
    rows = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" not in obj:
                continue
            rows.append(obj["messages"])
            if limit and len(rows) >= limit:
                break
    return rows


def encode_rows(rows_messages, tokenizer, max_length: int):
    """Encode each row's messages via nvfp4_lora.chat_encode.encode_chat_example.

    Returns a list of the per-row encoded dicts (n_tokens, n_supervised, truncated,
    labels, ...). This is the single tokenizer-dependent boundary; everything the CLI
    reports downstream is derived from these dicts by the pure functions above.

    Note: encode_chat_example already truncates input_ids/labels to max_length, so its
    `truncated` flag (len(input_ids) > max_length pre-truncation) is the authoritative
    per-row truncation signal, and its returned labels are the post-truncation labels
    the trainer would actually supervise on.
    """
    from nvfp4_lora.chat_encode import encode_chat_example

    return [encode_chat_example(messages, tokenizer, max_length) for messages in rows_messages]


def build_report(encoded, rows_messages, tokenizer, max_length, buckets=None,
                 template_preview=None) -> dict:
    """Assemble the full data-check report from already-encoded rows (pure)."""
    labels = [e["labels"] for e in encoded]
    # Pre-truncation lengths for the histogram + truncation stats: encode reports the
    # POST-truncation n_tokens, so a truncated row would look capped. Recover the full
    # length from the truncated flag where possible; otherwise use n_tokens.
    full_lengths = []
    for e in encoded:
        # n_tokens is post-truncation (<= max_length). If truncated, the true length is
        # unknown from the encoded dict alone but is > max_length; use max_length + 1 as a
        # lower bound so the row lands in the ">max" territory of the histogram/truncation.
        if e.get("truncated"):
            full_lengths.append(max_length + 1)
        else:
            full_lengths.append(e["n_tokens"])

    report = {
        "data_check_version": 1,
        "max_length": max_length,
        "mask_coverage": mask_coverage(labels),
        "truncation": truncation_stats(full_lengths, max_length),
        "length_histogram": histogram(full_lengths, buckets),
        "empty_tokenization_rows": sum(
            1 for e in encoded if e.get("dropped_reason") == "empty_tokenization"),
    }
    if template_preview is not None:
        report["template_preview"] = template_preview
    return report


def _print_summary(report: dict) -> None:
    mc = report["mask_coverage"]
    tr = report["truncation"]
    hist = report["length_histogram"]
    print("=== data-check summary ===")
    print(f"rows: {mc['n_rows']}   max_length: {report['max_length']}")
    print(f"assistant-mask coverage: {mc['n_covered']}/{mc['n_rows']} "
          f"({mc['covered_fraction']:.1%})   empty-assistant-span rows: "
          f"{mc['n_empty_assistant_span']}")
    print(f"mean supervised tokens/row: {mc['mean_supervised_tokens']:.1f}")
    print(f"truncated at max_length: {tr['n_truncated']}/{tr['n_rows']} "
          f"({tr['truncated_fraction']:.1%})")
    print(f"token length: min={hist['min']} max={hist['max']}")
    print("length histogram:")
    for b in hist["buckets"]:
        print(f"  {b['label']:>12} : {b['count']}")
    if report.get("empty_tokenization_rows"):
        print(f"WARNING: {report['empty_tokenization_rows']} rows tokenized to nothing")
    if mc["n_rows"] and mc["covered_fraction"] < 0.99:
        print("WARNING: assistant-mask coverage < 99% -- some rows supervise 0 tokens; "
              "check the chat template / role masking.")
    if tr["n_rows"] and tr["truncated_fraction"] > 0.05:
        print(f"WARNING: {tr['truncated_fraction']:.1%} of rows truncated at "
              f"max_length={report['max_length']} -- gold answers may be cut off.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="chat jsonl ({'messages': [...]}) to check")
    ap.add_argument("--tokenizer", required=True,
                    help="tokenizer dir (HF) -- rendered/tokenized as the trainer would")
    ap.add_argument("--max-length", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap rows read (0 = all)")
    ap.add_argument("--buckets", default="",
                    help="comma-separated histogram bucket upper-bounds "
                         "(default: 128,256,512,1024,2048,4096,8192)")
    ap.add_argument("--out", default=None, help="write the report JSON here")
    args = ap.parse_args(argv)

    buckets = None
    if args.buckets.strip():
        buckets = sorted(int(x) for x in args.buckets.split(",") if x.strip())

    rows = load_rows(args.data, limit=args.limit)
    if not rows:
        print(f"[data-check] no rows with a 'messages' key in {args.data}", file=sys.stderr)
        return 2
    print(f"[load] {len(rows)} rows from {args.data}", flush=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Chat-template preview: render the first row exactly as the trainer sees it.
    from nvfp4_lora.chat_encode import _render  # thin wrapper over apply_chat_template
    template_preview = _render(tokenizer, rows[0], add_generation_prompt=False)

    encoded = encode_rows(rows, tokenizer, args.max_length)
    report = build_report(encoded, rows, tokenizer, args.max_length, buckets,
                          template_preview=template_preview)
    report["data_file"] = str(args.data)
    report["tokenizer_dir"] = str(args.tokenizer)
    report["tokenizer_hashes"] = tokenizer_hashes(args.tokenizer)

    print("\n----- chat-template preview (row 0) -----")
    print(template_preview)
    print("----- end preview -----\n")
    _print_summary(report)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\n[write] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
