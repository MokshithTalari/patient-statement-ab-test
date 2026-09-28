"""Integrity tests: randomization, reproducibility, and report consistency."""
import json
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))
from generate_dataset import generate_cohort_data
from run_analysis import analyze

DATA = ROOT / "data" / "patient_billing_cohorts.csv"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


def test_dataset_reproduces_from_seed(df):
    regen = generate_cohort_data()
    pd.testing.assert_frame_equal(df, pd.read_csv(pd.io.common.StringIO(regen.to_csv(index=False))))


def test_no_patient_in_both_arms(df):
    assert df.groupby("patient_id").cohort_group.nunique().max() == 1


def test_randomization_valid(df):
    r = analyze(df)
    assert r["srm"]["p"] > 0.01, "sample ratio mismatch"
    assert r["balance_check"]["p"] > 0.01, "pre-treatment imbalance"


def test_published_results_match_data(df):
    published = json.loads((ROOT / "results.json").read_text())
    fresh = analyze(df)
    assert published["days"]["diff"] == pytest.approx(fresh["days"]["diff"])
    assert published["resolution_30d"]["diff_pp"] == pytest.approx(fresh["resolution_30d"]["diff_pp"])


def test_dashboard_embeds_current_results():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    assert (ROOT / "results.json").read_text(encoding="utf-8") in html


def test_guardrail_not_breached(df):
    assert analyze(df)["dispute"]["diff_pp"] <= 0.5
