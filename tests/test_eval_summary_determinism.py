from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_eval_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "eval_retention.py"
    spec = importlib.util.spec_from_file_location("eval_retention", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_summary_is_deterministic_and_ci_brackets_delta():
    mod = _load_eval_module()
    rows = [
        {"db_id": "db_b", "nll": {"base": 1.0, "ft": 0.8},
         "em": {"base": True, "ft": False}, "pred": {"base": "select 1", "ft": "select 2"}},
        {"db_id": "db_a", "nll": {"base": 2.0, "ft": 1.9},
         "em": {"base": True, "ft": True}, "pred": {"base": "select 1", "ft": "select 1"}},
        {"db_id": "db_a", "nll": {"base": 3.0, "ft": 2.8},
         "em": {"base": False, "ft": False}, "pred": {"base": "select 3", "ft": "select 4"}},
    ]

    s1 = mod.build_summary(rows, ["base", "ft"], no_nll=False, no_em=False, bootstrap_n=200)
    s2 = mod.build_summary(rows, ["base", "ft"], no_nll=False, no_em=False, bootstrap_n=200)

    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)
    nll_lo, nll_hi = s1["nll_delta_ci_vs_base"]["ft"]
    assert nll_lo <= s1["nll_delta_vs_base"]["ft"] <= nll_hi
    em_lo, em_hi = s1["em_delta_ci_vs_base"]["ft"]
    assert em_lo <= s1["em_delta_vs_base"]["ft"] <= em_hi


def test_build_summary_per_db_uses_paired_counts_and_sorted_db_keys():
    mod = _load_eval_module()
    rows = [
        {"db_id": "z", "nll": {"base": 1.0, "ft": 0.9}, "em": {"base": True, "ft": True}},
        {"db_id": "a", "nll": {"base": 2.0, "ft": 2.2}, "em": {"base": False, "ft": True}},
        {"db_id": "a", "nll": {"base": 3.0}, "em": {"base": True}},
    ]

    summary = mod.build_summary(rows, ["base", "ft"], no_nll=False, no_em=False, bootstrap_n=50)

    assert list(summary["per_db"]) == ["a", "z"]
    assert summary["per_db"]["a"]["n_nll"] == 1
    assert summary["per_db"]["a"]["n_em"] == 1
    assert summary["per_db"]["a"]["ft"]["em_delta"] == 1.0
    assert summary["per_db"]["z"]["ft"]["nll_delta"] == -0.1

    no_db = [{"nll": {"base": 1.0, "ft": 0.9}, "em": {"base": True, "ft": True}}]
    assert "per_db" not in mod.build_summary(
        no_db, ["base", "ft"], no_nll=False, no_em=False, bootstrap_n=10
    )


def test_metric_divergence_warns_and_empty_generations_suppress_it():
    mod = _load_eval_module()
    divergent = [
        {"nll": {"base": 1.0, "ft": 0.8}, "em": {"base": True, "ft": False},
         "pred": {"base": "select 1", "ft": "select 2"}},
        {"nll": {"base": 1.2, "ft": 1.0}, "em": {"base": True, "ft": False},
         "pred": {"base": "select 1", "ft": "select 2"}},
    ]
    empty = [
        {"nll": {"base": 1.0, "ft": 0.8}, "em": {"base": True, "ft": False},
         "pred": {"base": "select 1", "ft": ""}},
        {"nll": {"base": 1.2, "ft": 1.0}, "em": {"base": True, "ft": False},
         "pred": {"base": "select 1", "ft": ""}},
    ]

    s_div = mod.build_summary(divergent, ["base", "ft"], no_nll=False, no_em=False, bootstrap_n=50)
    s_empty = mod.build_summary(empty, ["base", "ft"], no_nll=False, no_em=False, bootstrap_n=50)

    assert any("divergent metrics" in w for w in s_div["warnings"])
    assert not any("divergent metrics" in w for w in s_empty.get("warnings", []))
