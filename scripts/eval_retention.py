#!/usr/bin/env python
"""Deterministic before/after text-to-SQL retention eval through a served vLLM endpoint.

Public retention demo (project v1 hard-gate). Scores a base served-model-name against an
adapter served-model-name on Spider dev, two ways, both deterministic, neither touching a
database:

(a) TEACHER-FORCED NLL on the gold SQL (primary signal -- no decoding, fully deterministic).
    We hit /v1/completions with prompt = (schema+question prompt) + gold SQL, echo=True,
    max_tokens=0, logprobs=1. vLLM then returns choices[0].logprobs with three aligned
    arrays over the *echoed prompt tokens*:
        tokens[i]          -- the i-th prompt token's string
        token_logprobs[i]  -- log P(tokens[i] | tokens[<i]); [0] is None (BOS, no context)
        text_offset[i]     -- char offset where tokens[i] begins in the echoed text
    We isolate the gold-SQL span by char offset (text_offset[i] >= len(prompt_text)) and
    average -token_logprobs over exactly those tokens => mean per-token NLL on the gold SQL.
    Lower is better; a successful adapter should lower NLL vs base. (max_tokens=0 + echo is
    the portable vLLM way to get prompt-token logprobs; `prompt_logprobs` is the alternative
    but its per-token dicts are keyed by token-id and messier to span-align, so we use echo.)

(b) EXACT-SET-MATCH accuracy (greedy generation, no DB execution). Generate SQL greedily
    (temperature 0, seed 0), then score with a self-contained component-based set-match port
    of the canonical Spider evaluation: decompose each query into SELECT / WHERE / GROUP BY /
    HAVING / ORDER BY / LIMIT / keyword components and compare them as sets, normalizing
    aliases, whitespace and case and ignoring literal *values*. A prediction is correct iff
    every component set-matches. This mirrors Spider's exact-set-match (no values) without a DB.

  python scripts/eval_retention.py --dev-file spider/spider.dev.chat.jsonl \
      --models <base-served-name> <adapter-served-name> --n 200 --out spider_retention.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only, same shape as scripts/eval_gsm8k.py)
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


def _token_count(base_url, model, text, timeout=120):
    """Number of tokens `text` encodes to under the SERVER's tokenizer, via /tokenize.

    Tokenizer-agnostic: works for HF and mistral_common/tekken serves alike, since it
    asks the running server rather than re-tokenizing locally. Returns int or None.
    """
    resp, err = _post(base_url, "/tokenize", {"model": model, "prompt": text}, timeout)
    if err or not resp:
        return None
    if isinstance(resp.get("count"), int):
        return resp["count"]
    toks = resp.get("tokens")
    return len(toks) if isinstance(toks, list) else None


def gold_nll(base_url, model, prompt, gold, timeout=600):
    """Mean per-token NLL of `gold` given `prompt`, via echo completions.

    Returns (nll_per_token, n_gold_tokens) or (None, err).
    """
    full = prompt + gold
    body = {"model": model, "prompt": full, "max_tokens": 0,
            "echo": True, "logprobs": 1, "temperature": 0.0}
    resp, err = _post(base_url, "/v1/completions", body, timeout)
    if err:
        return None, err
    lp = resp["choices"][0].get("logprobs") or {}
    toks = lp.get("tokens") or []
    tlp = lp.get("token_logprobs") or []
    offs = lp.get("text_offset") or []
    if not toks or len(tlp) != len(toks):
        return None, "malformed logprobs (echo not honored?)"
    cut = len(prompt)
    # Primary: gold tokens = those whose char offset starts at/after the prompt boundary.
    # Requires reliable text_offset (correct for HF tokenizers).
    sel = []
    if len(offs) == len(toks):
        sel = [tlp[i] for i in range(len(toks)) if offs[i] >= cut and tlp[i] is not None]
        if not sel:
            sel = [tlp[i] for i in range(len(toks)) if offs[i] > cut and tlp[i] is not None]
    if not sel:
        # Fallback (tokenizer-agnostic): some serves (mistral_common/tekken) return
        # unreliable text_offset, so the char-offset select finds nothing. Ask the
        # server how many tokens the prompt vs the full string encode to, and take the
        # trailing (n_full - n_prompt) echoed tokens as the gold span.
        n_prompt = _token_count(base_url, model, prompt, timeout)
        n_full = _token_count(base_url, model, full, timeout)
        if n_prompt is not None and n_full is not None and n_full > n_prompt:
            k = n_full - n_prompt
            tail = [t for t in tlp[-k:] if t is not None]
            if tail:
                sel = tail
    if not sel:
        return None, "no gold tokens selected"
    nll = -sum(sel) / len(sel)
    return {"nll": nll, "n_tokens": len(sel)}, None


def gen_sql(base_url, model, prompt, max_tokens=256, timeout=600):
    """Greedy completion of the SQL. Returns (text, err)."""
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0, "seed": 0, "stop": ["\n\n", ";", "```"]}
    resp, err = _post(base_url, "/v1/completions", body, timeout)
    if err:
        return None, err
    return resp["choices"][0]["text"], None


# ---------------------------------------------------------------------------
# Self-contained Spider component-based exact-set-match (no values, no DB).
# Port of the canonical Spider eval idea: tokenize SQL, split into clause
# components, compare each as a set (ignoring literal values, column order,
# table aliases, case and whitespace). Robust enough for a public demo signal
# without pulling in the full grammar parser / sqlite execution.
# ---------------------------------------------------------------------------

_CLAUSE_KW = ["select", "from", "where", "group by", "having",
              "order by", "limit", "intersect", "union", "except"]
_KEYWORDS = {"distinct", "join", "on", "and", "or", "not", "in", "like",
             "between", "asc", "desc", "count", "sum", "avg", "min", "max",
             "as", "is", "null", "exists"}


def _strip_md(s):
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"```\s*$", "", s).strip()
    return s


def _normalize(sql):
    sql = _strip_md(sql)
    sql = sql.strip().rstrip(";").strip()
    # strip string/number literals to single placeholders => "ignore values"
    sql = re.sub(r"'[^']*'", "'V'", sql)
    sql = re.sub(r'"[^"]*"', "'V'", sql)
    sql = re.sub(r"\b\d+(\.\d+)?\b", "0", sql)
    sql = re.sub(r"\s+", " ", sql).strip().lower()
    # drop alias declarations ("table as t1" / "table t1") and alias-qualified
    # references ("t1.col" -> "col") so aliasing doesn't change the component sets,
    # matching canonical Spider's alias-insensitive comparison.
    sql = re.sub(r"\bas\s+[a-z_][a-z0-9_]*", "", sql)        # "head as t1" -> "head"
    sql = re.sub(r"\b[a-z_][a-z0-9_]*\.", "", sql)            # "t1.name" / "head.name" -> "name"
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql


def _split_clauses(sql):
    """Return {clause_name: clause_text} using outer-level keyword splits.

    Tracks parenthesis depth so subquery keywords don't fragment the top clause.
    """
    toks = re.findall(r"\(|\)|[a-z_][a-z0-9_]*|\.|\*|[<>=!]+|,", sql)
    # rebuild with explicit two-word clause keywords joined
    out, i = [], 0
    while i < len(toks):
        two = toks[i] + " " + toks[i + 1] if i + 1 < len(toks) else ""
        if two in ("group by", "order by"):
            out.append(two)
            i += 2
        else:
            out.append(toks[i])
            i += 1

    clauses, cur_name, cur, depth = {}, "select", [], 0
    for t in out:
        if t == "(":
            depth += 1
        elif t == ")":
            depth = max(0, depth - 1)
        if depth == 0 and t in _CLAUSE_KW:
            if cur:
                clauses[cur_name] = clauses.get(cur_name, []) + cur
            cur_name, cur = t, []
            continue
        cur.append(t)
    if cur:
        clauses[cur_name] = clauses.get(cur_name, []) + cur
    return clauses


def _component_sets(sql):
    """Map each Spider component to a frozenset of normalized tokens."""
    clauses = _split_clauses(_normalize(sql))
    comps = {}
    for name in _CLAUSE_KW:
        toks = clauses.get(name, [])
        # split SELECT / GROUP BY / ORDER BY on commas into element sets;
        # for everything else use a bag of meaningful tokens.
        cleaned = [t for t in toks if t not in (",",)]
        comps[name] = frozenset(cleaned)
    # keyword bag: which SQL keywords appear at all (captures join/distinct/agg/etc.)
    all_toks = re.findall(r"[a-z_][a-z0-9_]*", _normalize(sql))
    comps["keywords"] = frozenset(k for k in all_toks if k in _KEYWORDS)
    return comps


def exact_set_match(pred, gold):
    """True iff every Spider component set-matches between pred and gold."""
    p, g = _component_sets(pred), _component_sets(gold)
    for name in list(_CLAUSE_KW) + ["keywords"]:
        if p.get(name, frozenset()) != g.get(name, frozenset()):
            return False
    return True


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--dev-file", required=True,
                    help="spider dev as chat jsonl ({messages:[user,assistant]})")
    ap.add_argument("--models", nargs="+", required=True,
                    help="served model names (first = base, rest = adapters)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--no-nll", action="store_true", help="skip teacher-forced NLL")
    ap.add_argument("--no-em", action="store_true", help="skip exact-set-match generation")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.dev_file) if l.strip()][:args.n]
    print(f"[load] {len(rows)} dev rows", flush=True)

    nll_sum = {m: 0.0 for m in args.models}
    nll_tok = {m: 0 for m in args.models}
    nll_n = {m: 0 for m in args.models}
    em_correct = {m: 0 for m in args.models}
    em_n = {m: 0 for m in args.models}
    per, used = [], 0

    nll_skips = {m: 0 for m in args.models}
    em_skips = {m: 0 for m in args.models}
    for row in rows:
        msgs = row["messages"]
        prompt, gold = msgs[0]["content"], msgs[-1]["content"]
        # completions endpoint: append the assistant lead-in so generation continues as SQL.
        gen_prompt = prompt + "\n"
        rec = {"gold": gold, "nll": {}, "pred": {}, "em": {}}

        # Compute BOTH metrics for ALL models without aborting the row on one failure:
        # a model whose NLL is unmeasurable (e.g. a tokenizer whose echo logprobs the
        # harness cannot span-align) still contributes its EM, and one model failing
        # never silently drops the row for the others.
        row_nll, row_em = {}, {}
        for m in args.models:
            if not args.no_nll:
                r, e = gold_nll(args.base_url, m, gen_prompt, gold)
                row_nll[m] = None if e else r
                if e:
                    nll_skips[m] += 1
                    if nll_skips[m] == 1:
                        print(f"  note: NLL unavailable for {m}: {e}", flush=True)
            if not args.no_em:
                text, e = gen_sql(args.base_url, m, gen_prompt, args.max_new_tokens)
                if e:
                    row_em[m] = None
                    em_skips[m] += 1
                    if em_skips[m] == 1:
                        print(f"  note: generation failed for {m}: {e}", flush=True)
                else:
                    row_em[m] = exact_set_match(text, gold)
                    rec["pred"][m] = _strip_md(text).strip()[:300]

        # Accumulate NLL only on rows where EVERY model produced it (paired before/after).
        if not args.no_nll and all(row_nll.get(m) is not None for m in args.models):
            for m in args.models:
                rec["nll"][m] = round(row_nll[m]["nll"], 5)
                nll_sum[m] += row_nll[m]["nll"]
                nll_tok[m] += row_nll[m]["n_tokens"]
                nll_n[m] += 1
        # Accumulate EM only on rows where EVERY model produced it (paired).
        if not args.no_em and all(row_em.get(m) is not None for m in args.models):
            for m in args.models:
                rec["em"][m] = row_em[m]
                em_correct[m] += int(row_em[m])
                em_n[m] += 1

        if not rec["nll"] and not rec["em"]:
            continue  # nothing usable from this row for any model
        used += 1
        per.append(rec)
        if used % 20 == 0:
            msg = f"[{used}/{len(rows)}]"
            if not args.no_nll:
                msg += "  NLL: " + ", ".join(
                    f"{m}={nll_sum[m]/nll_n[m]:.4f}" for m in args.models if nll_n[m])
            if not args.no_em:
                msg += "  EM: " + ", ".join(
                    f"{m}={em_correct[m]/em_n[m]:.3f}" for m in args.models if em_n[m])
            print(msg, flush=True)

    base = args.models[0]
    summary = {"n": used, "models": args.models}
    # Skip accounting so a fully-failed metric is never mistaken for a real result.
    summary["skipped"] = {"nll": nll_skips, "em": em_skips}
    warnings_out = []
    if not args.no_nll:
        mean_nll = {m: (nll_sum[m] / nll_n[m] if nll_n[m] else None) for m in args.models}
        summary["mean_gold_nll"] = {m: (round(v, 5) if v is not None else None)
                                    for m, v in mean_nll.items()}
        summary["nll_delta_vs_base"] = {
            m: (round(mean_nll[m] - mean_nll[base], 5)
                if mean_nll[m] is not None and mean_nll[base] is not None else None)
            for m in args.models[1:]}  # negative = adapter improved (lower NLL)
        for m in args.models:
            if nll_n[m] == 0:
                warnings_out.append(
                    f"NLL is NULL for '{m}': all {nll_skips[m]} attempts failed. The served "
                    f"tokenizer is likely wrong for this model (tekken/mistral_common models "
                    f"must be served with --tokenizer-mode mistral). This is NOT a 0/None result.")
    if not args.no_em:
        acc = {m: (em_correct[m] / em_n[m] if em_n[m] else 0.0) for m in args.models}
        summary["exact_set_match"] = {m: round(acc[m], 5) for m in args.models}
        summary["em_delta_vs_base"] = {m: round(acc[m] - acc[base], 5)
                                       for m in args.models[1:]}  # positive = adapter better
        for m in args.models:
            if em_n[m] == 0:
                warnings_out.append(
                    f"exact-set-match is 0/0 for '{m}': all {em_skips[m]} generations failed "
                    f"(reported 0.0 is a FAILURE, not a real score).")
    # Detect a likely silent adapter no-op: an adapter whose NLL and EM are identical to base.
    if len(args.models) > 1 and used > 0:
        for m in args.models[1:]:
            same_nll = (not args.no_nll and nll_n[m] and nll_n[base]
                        and summary["mean_gold_nll"].get(m) == summary["mean_gold_nll"].get(base))
            same_em = (not args.no_em and em_n[m] and summary["exact_set_match"].get(m) == summary["exact_set_match"].get(base))
            if same_nll and same_em:
                warnings_out.append(
                    f"adapter '{m}' produced IDENTICAL NLL and EM to base -- likely a silent "
                    f"no-op (adapter not bound at serve). Check vLLM LoRA load / use serve --verify.")
    if warnings_out:
        summary["warnings"] = warnings_out

    Path(args.out).write_text(json.dumps({"summary": summary, "per_example": per}, indent=2))
    print("\n=== Spider text-to-SQL retention (base vs adapter) ===")
    print(json.dumps(summary, indent=2))
    for w in warnings_out:
        print(f"WARNING: {w}", flush=True)
    print(f"\n[write] {args.out}")


if __name__ == "__main__":
    main()
