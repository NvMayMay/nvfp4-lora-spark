#!/usr/bin/env python
"""Format Spider (text-to-SQL) into schema-grounded chat jsonl for the trainer.

Public, deterministic retention demo (project v1 hard-gate). Source: `xlangai/spider`
(HF) -- train (7000) + validation (1034). Each parquet row carries only `db_id`,
`question`, `query` (the gold SQL); it does NOT carry the DB schema. The schema is the
crux of the task -- without the table/column list the base hallucinates columns -- so we
join each row's `db_id` against Spider's per-DB schema and serialize it into the USER
message.

Schema source: `richardr1126/spider-schema` (HF dataset, `spider_schema_rows_v2.json`),
a flat list of 166 records: {db_id, "Schema (values (type))", "Primary Keys",
"Foreign Keys"}. Verified to cover 100% of the db_ids in both the train (140 dbs) and
validation (20 dbs) splits. This is the canonical Spider schema content (table : col
(type) , ... | next_table : ...) in a ready-to-serialize, human-readable form, so we use
it directly rather than re-parsing the original tables.json column_names_original arrays.

Output rows: {"messages": [{"role":"user","content":<schema+question prompt>},
                           {"role":"assistant","content":<gold SQL>}]}

  python scripts/prep_spider.py --out-dir /path/spider
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

INSTRUCTION = (
    "Return only the SQL query that answers the question. "
    "Output a single line of valid SQLite, no explanation, no markdown fences."
)


def _format_schema(rec):
    """Serialize one spider-schema record into a compact, readable schema block.

    rec["Schema (values (type))"] is already "table : col (type) , col (type) | table2 : ...".
    We expand the `|` table separator onto its own line so the column list per table is
    obvious to the model, and append PK/FK lines when present.
    """
    schema = (rec.get("Schema (values (type))") or "").strip()
    tables = [t.strip() for t in schema.split("|") if t.strip()]
    lines = ["Database schema:"]
    for t in tables:
        lines.append(f"  {t}")
    pk = (rec.get("Primary Keys") or "").strip()
    fk = (rec.get("Foreign Keys") or "").strip()
    if pk:
        lines.append("Primary keys: " + pk)
    if fk:
        lines.append("Foreign keys: " + fk)
    return "\n".join(lines)


def _build_prompt(schema_block, db_id, question):
    return (
        f"You are an expert SQLite assistant. Given a database schema and a question, "
        f"write a SQL query.\n\n"
        f"Database: {db_id}\n"
        f"{schema_block}\n\n"
        f"Question: {question.strip()}\n\n"
        f"{INSTRUCTION}"
    )


def _load_schemas(schema_file):
    """Return {db_id: formatted_schema_block}. Local json file or HF download."""
    if schema_file and Path(schema_file).exists():
        recs = json.load(open(schema_file))
    else:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("richardr1126/spider-schema",
                            "spider_schema_rows_v2.json", repo_type="dataset")
        recs = json.load(open(p))
    return {r["db_id"]: _format_schema(r) for r in recs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xlangai/spider")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--schema-file", default="",
                    help="local spider_schema_rows_v2.json; if absent, download from "
                         "richardr1126/spider-schema on HF")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--dev-split", default="validation")
    ap.add_argument("--n", type=int, default=0, help="cap train rows (0 = all)")
    ap.add_argument("--dev-n", type=int, default=0, help="cap dev rows (0 = all)")
    ap.add_argument("--limit", type=int, default=0,
                    help="dry-run: only process this many rows of each split, print, no write")
    args = ap.parse_args()

    from datasets import load_dataset

    schemas = _load_schemas(args.schema_file)
    print(f"[schema] {len(schemas)} db schemas loaded", flush=True)

    out = Path(args.out_dir)
    if not args.limit:
        out.mkdir(parents=True, exist_ok=True)

    sample_shown = False
    for split, fname, cap in (
        (args.train_split, "spider.train.chat.jsonl", args.n),
        (args.dev_split, "spider.dev.chat.jsonl", args.dev_n),
    ):
        ds = load_dataset(args.dataset, split=split)
        print(f"[load] {args.dataset}:{split} = {len(ds)} rows", flush=True)
        rows, missing = [], 0
        n_iter = args.limit if args.limit else (cap if cap else len(ds))
        for i in range(min(n_iter, len(ds))):
            row = ds[i]
            db_id = row["db_id"]
            sb = schemas.get(db_id)
            if sb is None:
                missing += 1
                continue
            user = _build_prompt(sb, db_id, row["question"])
            gold = row["query"].strip()
            rows.append({"db_id": db_id,
                         "messages": [{"role": "user", "content": user},
                                      {"role": "assistant", "content": gold}]})
            if not sample_shown:
                print("\n" + "=" * 72)
                print(f"[sample] db_id={db_id}  split={split}")
                print("-" * 72 + "\n[USER]\n" + user)
                print("-" * 72 + "\n[ASSISTANT]\n" + gold)
                print("=" * 72 + "\n", flush=True)
                sample_shown = True

        if missing:
            print(f"[warn] {split}: {missing} rows had no schema (skipped)", flush=True)

        if args.limit:
            print(f"[dry-run] {split}: would write {len(rows)} rows (not writing)", flush=True)
            continue

        p = out / fname
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[write] {p}  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
