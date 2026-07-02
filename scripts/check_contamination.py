#!/usr/bin/env python
"""Train<->eval contamination report for a chat-jsonl SFT dataset.

A retention/eval number is only credible if the eval set was not (partly) trained on.
This script measures overlap between a TRAIN chat jsonl and an EVAL chat jsonl three
ways and WARNs over disclosed thresholds:

  1. EXACT-QUESTION match -- normalized (whitespace/case-folded) equality of the eval
     "question" text against the set of train questions. This is the ground truth for
     "this exact item was in training"; unlike n-gram overlap it has NO false negatives
     from paraphrase because it is exact, and NO false positives. It is the primary
     signal; the n-gram measures below are a softer complement.
  2. N-GRAM OVERLAP at n=8 and n=13 (word n-grams over the normalized question). For each
     eval row we report the fraction of its n-grams that also appear anywhere in the
     train n-gram set; the row-level "contaminated" flag trips when that fraction exceeds
     a threshold (default 0.5). CAVEAT (disclosed): n-gram overlap has a HIGH FALSE-
     NEGATIVE rate on rephrasings -- a semantically identical but reworded question can
     share few 13-grams -- which is exactly why we ALSO do the exact-question match and
     do not rely on n-grams alone. 13-gram overlap is the GPT-3/dedup-lineage standard;
     8-gram is a looser, higher-recall complement.
  3. EXACT db_id / schema overlap -- when rows carry a `db_id` (Spider), the set of eval
     db_ids that also appear in train. For text-to-SQL, SCHEMA (db_id) overlap is expected
     and not itself contamination (Spider's dev dbs are disjoint from train by design, so
     any overlap is worth surfacing); we report it so a reviewer can judge.

The pure combinatorial logic (n-gram set build, overlap fraction, exact-match,
db overlap) is factored into small functions taking plain strings/lists, so it is
unit-testable without any dataset on disk.

  python scripts/check_contamination.py --train spider/spider.train.chat.jsonl \
      --eval spider/spider.dev.chat.jsonl --out contamination.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_NGRAMS = (8, 13)
# Row is flagged contaminated when the fraction of its n-grams seen in train exceeds this.
DEFAULT_ROW_THRESHOLD = 0.5
# Report-level WARN when the fraction of eval rows that are contaminated exceeds this.
DEFAULT_CORPUS_THRESHOLD = 0.02


# ---------------------------------------------------------------------------
# Pure combinatorial logic (unit-testable with plain strings/lists).
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip -- the canonical form for all comparisons."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _words(text: str) -> list[str]:
    """Word tokens of the normalized text (alphanumeric+underscore runs)."""
    return re.findall(r"\w+", normalize_text(text))


def ngram_set(text: str, n: int) -> set:
    """Set of word n-grams (as tuples) of `text`. Empty when fewer than n words."""
    words = _words(text)
    if len(words) < n:
        return set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def build_ngram_index(texts, n: int) -> set:
    """Union of the n-gram sets of many texts -- the train-side lookup set."""
    index = set()
    for text in texts:
        index |= ngram_set(text, n)
    return index


def overlap_fraction(text: str, train_index: set, n: int) -> float:
    """Fraction of `text`'s n-grams that appear in `train_index`.

    Returns 0.0 when `text` has no n-grams (too few words) -- a row we cannot judge by
    n-grams is treated as not-overlapping (the exact-match signal covers it).
    """
    grams = ngram_set(text, n)
    if not grams:
        return 0.0
    hit = sum(1 for g in grams if g in train_index)
    return hit / len(grams)


def exact_match_set(texts) -> set:
    """Set of normalized-exact forms of `texts` (for exact-question membership tests)."""
    return {normalize_text(t) for t in texts if normalize_text(t)}


def count_exact_matches(eval_texts, train_exact: set) -> int:
    """How many eval texts are byte-for-byte (post-normalization) in the train set."""
    return sum(1 for t in eval_texts if normalize_text(t) in train_exact)


def db_overlap(train_db_ids, eval_db_ids) -> dict:
    """Overlap between train and eval db_id sets (exact schema-identity overlap)."""
    train_set = {d for d in train_db_ids if d is not None}
    eval_set = {d for d in eval_db_ids if d is not None}
    shared = sorted(train_set & eval_set)
    return {
        "n_train_dbs": len(train_set),
        "n_eval_dbs": len(eval_set),
        "n_shared_dbs": len(shared),
        "shared_dbs": shared,
        "eval_db_overlap_fraction": (len(shared) / len(eval_set)) if eval_set else 0.0,
    }


def contamination_report(train_texts, eval_texts, *, ngrams=DEFAULT_NGRAMS,
                         row_threshold=DEFAULT_ROW_THRESHOLD,
                         train_db_ids=None, eval_db_ids=None) -> dict:
    """Full contamination report from plain lists (no I/O). Pure & unit-testable.

    `train_texts`/`eval_texts` are lists of question strings. `train_db_ids`/`eval_db_ids`
    are optional parallel lists (Spider); when both are provided a db-overlap block is
    added. Disclosed thresholds are echoed into the report.
    """
    n_eval = len(eval_texts)
    train_exact = exact_match_set(train_texts)
    n_exact = count_exact_matches(eval_texts, train_exact)

    indexes = {n: build_ngram_index(train_texts, n) for n in ngrams}
    ngram_out = {}
    per_row_flags = {n: 0 for n in ngrams}
    for n in ngrams:
        fractions = [overlap_fraction(t, indexes[n], n) for t in eval_texts]
        flagged = sum(1 for f in fractions if f > row_threshold)
        per_row_flags[n] = flagged
        mean_frac = (sum(fractions) / len(fractions)) if fractions else 0.0
        ngram_out[f"{n}gram"] = {
            "n": n,
            "row_threshold": row_threshold,
            "n_rows_over_threshold": flagged,
            "fraction_rows_over_threshold": (flagged / n_eval) if n_eval else 0.0,
            "mean_overlap_fraction": mean_frac,
        }

    report = {
        "contamination_check_version": 1,
        "n_train": len(train_texts),
        "n_eval": n_eval,
        "exact_question": {
            "n_exact_matches": n_exact,
            "fraction": (n_exact / n_eval) if n_eval else 0.0,
        },
        "ngram_overlap": ngram_out,
        "thresholds": {
            "row_ngram_overlap": row_threshold,
            "note": ("n-gram overlap has a HIGH false-negative rate on rephrasings; "
                     "the exact-question match is the primary, false-negative-free signal."),
        },
    }
    if train_db_ids is not None and eval_db_ids is not None:
        report["db_overlap"] = db_overlap(train_db_ids, eval_db_ids)
    return report


def add_warnings(report: dict, corpus_threshold=DEFAULT_CORPUS_THRESHOLD) -> list:
    """Attach human-readable WARN strings to `report` and return the list."""
    warnings = []
    n_eval = report.get("n_eval", 0)
    exact = report.get("exact_question", {})
    if exact.get("n_exact_matches", 0) > 0:
        warnings.append(
            f"{exact['n_exact_matches']} eval questions ({exact['fraction']:.2%}) are EXACT "
            f"matches of train questions -- direct contamination.")
    for key, block in (report.get("ngram_overlap") or {}).items():
        frac = block.get("fraction_rows_over_threshold", 0.0)
        if frac > corpus_threshold:
            warnings.append(
                f"{block['n_rows_over_threshold']} eval rows ({frac:.2%}) exceed the "
                f"{block['n']}-gram overlap threshold ({block['row_threshold']}) with train.")
    db = report.get("db_overlap")
    if db and db.get("n_shared_dbs", 0) > 0:
        warnings.append(
            f"{db['n_shared_dbs']} db_id(s) appear in BOTH train and eval "
            f"({db['eval_db_overlap_fraction']:.2%} of eval dbs): {db['shared_dbs']}")
    if warnings:
        report["warnings"] = warnings
    return warnings


# ---------------------------------------------------------------------------
# Thin I/O wrapper (not exercised by the CPU unit tests).
# ---------------------------------------------------------------------------


def _extract_question(messages, text_field=None) -> str:
    """Pull the question text from a chat row.

    Default: the first user message's content. `text_field` (dot-free top-level key) is
    honored elsewhere by the caller for non-chat rows.
    """
    for m in messages or []:
        if m.get("role") == "user":
            return m.get("content") or ""
    return (messages[0].get("content") if messages else "") or ""


def load_jsonl(path, text_field=None):
    """Read a chat/plain jsonl into (questions, db_ids) parallel lists.

    If `text_field` is given, each row's top-level `row[text_field]` is used as the
    question text (for non-chat datasets); otherwise the first user message is used.
    db_id is read from the top-level `db_id` when present (else None).
    """
    questions, db_ids = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if text_field and text_field in obj:
                q = obj.get(text_field) or ""
            else:
                q = _extract_question(obj.get("messages"))
            questions.append(q)
            db_ids.append(obj.get("db_id"))
    return questions, db_ids


def _print_summary(report: dict, warnings) -> None:
    print("=== contamination summary ===")
    print(f"train rows: {report['n_train']}   eval rows: {report['n_eval']}")
    exact = report["exact_question"]
    print(f"exact-question matches: {exact['n_exact_matches']} ({exact['fraction']:.2%})")
    for key, block in report["ngram_overlap"].items():
        print(f"{key} overlap: {block['n_rows_over_threshold']} rows over "
              f"{block['row_threshold']} ({block['fraction_rows_over_threshold']:.2%}); "
              f"mean row overlap {block['mean_overlap_fraction']:.3f}")
    db = report.get("db_overlap")
    if db:
        print(f"db_id overlap: {db['n_shared_dbs']} shared "
              f"({db['eval_db_overlap_fraction']:.2%} of eval dbs)")
    for w in warnings:
        print(f"WARNING: {w}")
    if not warnings:
        print("no contamination over thresholds.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", required=True, help="train chat/plain jsonl")
    ap.add_argument("--eval", required=True, help="eval chat/plain jsonl")
    ap.add_argument("--text-field", default="",
                    help="top-level key holding the question text (non-chat rows); "
                         "default = first user message content")
    ap.add_argument("--ngrams", default="8,13",
                    help="comma-separated n-gram sizes (default 8,13)")
    ap.add_argument("--row-threshold", type=float, default=DEFAULT_ROW_THRESHOLD,
                    help="per-row n-gram overlap fraction above which a row is flagged")
    ap.add_argument("--corpus-threshold", type=float, default=DEFAULT_CORPUS_THRESHOLD,
                    help="WARN when the fraction of flagged eval rows exceeds this")
    ap.add_argument("--out", default=None, help="write the report JSON here")
    args = ap.parse_args(argv)

    text_field = args.text_field or None
    ngrams = tuple(int(x) for x in args.ngrams.split(",") if x.strip())

    train_q, train_db = load_jsonl(args.train, text_field)
    eval_q, eval_db = load_jsonl(args.eval, text_field)
    if not eval_q:
        print(f"[contamination] no eval rows read from {args.eval}", file=sys.stderr)
        return 2
    print(f"[load] train={len(train_q)} eval={len(eval_q)} rows", flush=True)

    have_db = any(d is not None for d in train_db) and any(d is not None for d in eval_db)
    report = contamination_report(
        train_q, eval_q, ngrams=ngrams, row_threshold=args.row_threshold,
        train_db_ids=train_db if have_db else None,
        eval_db_ids=eval_db if have_db else None,
    )
    report["train_file"] = str(args.train)
    report["eval_file"] = str(args.eval)
    warnings = add_warnings(report, corpus_threshold=args.corpus_threshold)

    _print_summary(report, warnings)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\n[write] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
