"""CPU unit tests for the pure logic of scripts/data_check.py.

No tokenizer, no model, no CUDA: we exercise the tokenizer-free functions (mask
coverage, truncation counting, histogram bucketing) on plain lists of ints, following
the spec_from_file_location loader pattern from tests/test_serve_apply_check.py.
"""
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = REPO_ROOT / "scripts" / "data_check.py"
    spec = importlib.util.spec_from_file_location("data_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dc = _load()

M = -100  # the mask value used by nvfp4_lora.chat_encode


# --- n_supervised / mask_coverage -----------------------------------------


def test_n_supervised_counts_non_masked():
    assert dc.n_supervised([M, M, 5, 7, M]) == 2
    assert dc.n_supervised([M, M, M]) == 0
    assert dc.n_supervised([]) == 0


def test_mask_coverage_mixed_rows():
    rows = [
        [M, 1, 2],       # 2 supervised
        [M, M, M],       # empty assistant span
        [3, 4],          # 2 supervised
        [M],             # empty assistant span
    ]
    cov = dc.mask_coverage(rows)
    assert cov["n_rows"] == 4
    assert cov["n_covered"] == 2
    assert cov["n_empty_assistant_span"] == 2
    assert cov["covered_fraction"] == pytest.approx(0.5)
    assert cov["total_supervised_tokens"] == 4
    assert cov["mean_supervised_tokens"] == pytest.approx(1.0)


def test_mask_coverage_empty_is_safe():
    cov = dc.mask_coverage([])
    assert cov["n_rows"] == 0
    assert cov["covered_fraction"] == 0.0
    assert cov["mean_supervised_tokens"] == 0.0


def test_mask_coverage_all_empty_span():
    cov = dc.mask_coverage([[M, M], [M]])
    assert cov["n_covered"] == 0
    assert cov["n_empty_assistant_span"] == 2
    assert cov["covered_fraction"] == 0.0


# --- truncation_stats ------------------------------------------------------


def test_truncation_counts_strictly_over_max():
    stats = dc.truncation_stats([10, 100, 100, 101, 200], max_length=100)
    # only 101 and 200 exceed 100; 100 itself is NOT truncated (<=).
    assert stats["n_truncated"] == 2
    assert stats["n_rows"] == 5
    assert stats["truncated_fraction"] == pytest.approx(0.4)
    assert stats["max_length"] == 100


def test_truncation_empty_is_safe():
    stats = dc.truncation_stats([], max_length=50)
    assert stats["n_truncated"] == 0
    assert stats["truncated_fraction"] == 0.0


def test_truncation_none_over():
    stats = dc.truncation_stats([1, 2, 3], max_length=1000)
    assert stats["n_truncated"] == 0


# --- histogram -------------------------------------------------------------


def test_histogram_buckets_and_open_tail():
    lengths = [0, 1, 128, 129, 256, 5000, 9000]
    hist = dc.histogram(lengths, buckets=[128, 256, 512, 1024, 2048, 4096, 8192])
    counts = {b["label"]: b["count"] for b in hist["buckets"]}
    # 0,1,128 -> "0-128"; 129 -> "129-256" is bucket lo=129 hi=256 (label "129-256")
    assert counts["0-128"] == 3
    # the bucket after 128 has lo=129, hi=256
    assert counts["129-256"] == 2   # 129 and 256
    # 5000 -> "4097-8192"; 9000 -> open "8193+"
    assert counts["4097-8192"] == 1
    assert counts["8193+"] == 1
    assert hist["min"] == 0
    assert hist["max"] == 9000
    assert hist["n_rows"] == 7
    assert hist["total_tokens"] == sum(lengths)
    # every row is placed exactly once
    assert sum(b["count"] for b in hist["buckets"]) == len(lengths)


def test_histogram_empty():
    hist = dc.histogram([], buckets=[128, 256])
    assert hist["min"] is None
    assert hist["max"] is None
    assert sum(b["count"] for b in hist["buckets"]) == 0


def test_histogram_boundary_inclusive_upper():
    # a length exactly on an edge falls in that edge's bucket (L <= hi)
    hist = dc.histogram([128], buckets=[128, 256])
    counts = {b["label"]: b["count"] for b in hist["buckets"]}
    assert counts["0-128"] == 1


def test_histogram_default_buckets():
    hist = dc.histogram([100, 5000])
    # default buckets end at 8192; 5000 lands in "4097-8192"
    assert sum(b["count"] for b in hist["buckets"]) == 2


# --- build_report (pure assembly over encoded dicts) -----------------------


def test_build_report_uses_truncated_flag_for_length():
    # Two encoded rows: one truncated (full length unknown -> counted as > max),
    # one not. build_report must derive coverage + truncation + histogram.
    encoded = [
        {"n_tokens": 100, "n_supervised": 40, "truncated": True, "labels": [1] * 40 + [M] * 60,
         "dropped_reason": None},
        {"n_tokens": 50, "n_supervised": 0, "truncated": False, "labels": [M] * 50,
         "dropped_reason": "no_supervised_tokens"},
    ]
    report = dc.build_report(encoded, rows_messages=None, tokenizer=None, max_length=100,
                             buckets=[128, 256])
    assert report["max_length"] == 100
    # one truncated row
    assert report["truncation"]["n_truncated"] == 1
    # mask coverage: one row supervised, one empty
    assert report["mask_coverage"]["n_covered"] == 1
    assert report["mask_coverage"]["n_empty_assistant_span"] == 1
