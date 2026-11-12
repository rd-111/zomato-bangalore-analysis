import pandas as pd
import pytest

from src.eda import correlation_summary, location_summary, prepare_analytical_data, cuisine_summary


def sample_data():
    return pd.DataFrame({
        "rate": [4.0, 3.0, None, 5.0],
        "votes": [10, 20, 30, None],
        "approx_cost(for two people)": [100, None, 300, 200],
        "location": ["A", "A", "B", "B"],
        "online_order": ["Yes", "No", "Yes", "No"],
        "book_table": ["No", "No", "Yes", "Yes"],
        "cuisines": ["Thai", "Thai", "Cafe", "Cafe"],
    })


def test_prepare_data_retains_unrated_rows_for_non_rating_work():
    result = prepare_analytical_data(sample_data())
    assert result["subset_report"]["all_rows"] == 4
    assert result["subset_report"]["missing_rating_rows_retained_for_counts"] == 1
    assert result["subset_report"]["rating_complete_rows"] == 3
    assert result["subset_report"]["cost_and_rating_rows"] == 2


def test_location_and_cuisine_summaries_include_sample_counts_and_threshold_flag():
    data = sample_data()
    _, locations = location_summary(data, min_rating_count=2)
    _, cuisines = cuisine_summary(data, min_rating_count=2)
    assert locations.loc["A", "rating_count"] == 2
    assert bool(locations.loc["A", "eligible_for_ranking"])
    assert cuisines.loc["Thai", "rating_count"] == 2
    assert bool(cuisines.loc["Thai", "eligible_for_ranking"])


def test_correlation_summary_reports_pairwise_sample_size():
    result = correlation_summary(sample_data())
    cost = result.loc[result["feature_x"] == "approx_cost(for two people)"].iloc[0]
    assert cost["n"] == 2
    assert cost["pearson_r"] == pytest.approx(1.0)
