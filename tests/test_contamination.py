"""CPU unit tests for the pure logic of scripts/check_contamination.py.

No dataset on disk, no CUDA: we exercise the combinatorial functions (n-gram set build,
overlap fraction, exact-match, db overlap) on plain strings/lists, using the
spec_from_file_location loader pattern from tests/test_serve_apply_check.py.
"""
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = REPO_ROOT / "scripts" / "check_contamination.py"
    spec = importlib.util.spec_from_file_location("check_contamination", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cc = _load()


# --- normalize / ngram_set -------------------------------------------------


def test_normalize_folds_case_and_whitespace():
    assert cc.normalize_text("  Hello   WORLD ") == "hello world"
    assert cc.normalize_text(None) == ""


def test_ngram_set_basic():
    grams = cc.ngram_set("the quick brown fox jumps", 3)
    assert ("the", "quick", "brown") in grams
    assert ("brown", "fox", "jumps") in grams
    assert len(grams) == 3  # 5 words -> 3 trigrams


def test_ngram_set_too_few_words_is_empty():
    assert cc.ngram_set("only three words here", 8) == set()
    assert cc.ngram_set("", 2) == set()


def test_ngram_set_is_case_insensitive():
    assert cc.ngram_set("The Quick", 2) == cc.ngram_set("the quick", 2)


# --- build_ngram_index / overlap_fraction ----------------------------------


def test_overlap_fraction_full_and_partial():
    train = ["alpha beta gamma delta epsilon"]
    index = cc.build_ngram_index(train, 2)
    # identical text -> all bigrams present
    assert cc.overlap_fraction("alpha beta gamma delta epsilon", index, 2) == pytest.approx(1.0)
    # half-shared: "alpha beta" is in the index, "zeta eta" is not
    frac = cc.overlap_fraction("alpha beta zeta eta", index, 2)
    # bigrams of eval: (alpha,beta)=hit, (beta,zeta)=miss, (zeta,eta)=miss -> 1/3
    assert frac == pytest.approx(1 / 3)


def test_overlap_fraction_no_ngrams_is_zero():
    index = cc.build_ngram_index(["a b c d e f g h i j"], 8)
    # eval too short to form an 8-gram -> 0.0, not a crash
    assert cc.overlap_fraction("short", index, 8) == 0.0


def test_overlap_fraction_no_shared():
    index = cc.build_ngram_index(["one two three"], 2)
    assert cc.overlap_fraction("four five six", index, 2) == 0.0


# --- exact match -----------------------------------------------------------


def test_exact_match_set_and_count():
    train = ["What is the capital?", "how MANY singers"]
    train_exact = cc.exact_match_set(train)
    # exact (normalized) match, including a case/whitespace variant
    evals = ["what is the capital?", "  How Many   Singers ", "unseen question"]
    assert cc.count_exact_matches(evals, train_exact) == 2


def test_exact_match_ignores_blank():
    train_exact = cc.exact_match_set(["", "   "])
    assert train_exact == set()
    assert cc.count_exact_matches(["anything"], train_exact) == 0


# --- db_overlap ------------------------------------------------------------


def test_db_overlap_counts_shared():
    out = cc.db_overlap(["a", "b", "c", None], ["b", "c", "d"])
    assert out["n_train_dbs"] == 3
    assert out["n_eval_dbs"] == 3
    assert out["n_shared_dbs"] == 2
    assert out["shared_dbs"] == ["b", "c"]
    assert out["eval_db_overlap_fraction"] == pytest.approx(2 / 3)


def test_db_overlap_disjoint():
    out = cc.db_overlap(["a", "b"], ["x", "y"])
    assert out["n_shared_dbs"] == 0
    assert out["eval_db_overlap_fraction"] == 0.0


def test_db_overlap_empty_eval_safe():
    out = cc.db_overlap(["a"], [])
    assert out["eval_db_overlap_fraction"] == 0.0


# --- contamination_report + warnings ---------------------------------------


def test_contamination_report_flags_exact_and_ngrams():
    train = ["show me all singers older than thirty who performed in two thousand ten"]
    # eval[0] is an exact duplicate; eval[1] is unrelated
    eval_texts = [
        "show me all singers older than thirty who performed in two thousand ten",
        "completely different unrelated harmless benign eval question about weather today now",
    ]
    report = cc.contamination_report(train, eval_texts, ngrams=(8, 13), row_threshold=0.5)
    assert report["n_train"] == 1
    assert report["n_eval"] == 2
    assert report["exact_question"]["n_exact_matches"] == 1
    # the duplicate row exceeds the 8-gram overlap threshold
    assert report["ngram_overlap"]["8gram"]["n_rows_over_threshold"] == 1
    warnings = cc.add_warnings(report, corpus_threshold=0.0)
    assert any("EXACT" in w for w in warnings)
    assert "warnings" in report


def test_contamination_report_clean_has_no_warnings():
    train = ["one two three four five six seven eight nine ten eleven twelve thirteen fourteen"]
    eval_texts = ["apple orange banana grape melon kiwi peach plum cherry lemon lime fig date pear"]
    report = cc.contamination_report(train, eval_texts, ngrams=(8, 13))
    assert report["exact_question"]["n_exact_matches"] == 0
    assert report["ngram_overlap"]["13gram"]["n_rows_over_threshold"] == 0
    warnings = cc.add_warnings(report, corpus_threshold=0.02)
    assert warnings == []
    assert "warnings" not in report


def test_contamination_report_db_overlap_included_and_warns():
    report = cc.contamination_report(
        ["q one two three"], ["q one two three"],
        ngrams=(2,), train_db_ids=["concert_singer"], eval_db_ids=["concert_singer"],
    )
    assert report["db_overlap"]["n_shared_dbs"] == 1
    warnings = cc.add_warnings(report, corpus_threshold=1.0)
    assert any("db_id" in w for w in warnings)


def test_contamination_report_empty_eval_safe():
    report = cc.contamination_report(["a b c"], [], ngrams=(2,))
    assert report["n_eval"] == 0
    assert report["exact_question"]["fraction"] == 0.0
    assert report["ngram_overlap"]["2gram"]["fraction_rows_over_threshold"] == 0.0
