import numpy as np
import pandas as pd
import pytest

from src.clustering import (
    evaluate_k_values,
    fit_final_model,
    prepare_clustering_data,
    profile_clusters,
    scale_features,
)


def mini_data():
    return pd.DataFrame({
        "name": ["a", "b", "c", "d", "e", "missing"],
        "rate": [4.7, 4.4, 3.8, 3.4, 2.9, None],
        "votes": [200, 80, 30, 12, 3, 5],
        "approx_cost(for two people)": [1500, 1000, 600, 400, 250, 100],
    })


def test_prepare_clustering_data_drops_only_incomplete_model_rows_and_keeps_source_values():
    source = mini_data()
    prepared, report = prepare_clustering_data(source)
    assert len(prepared) == 5
    assert report["records_before_selection"] == 6
    assert report["records_after_selection"] == 5
    assert prepared["source_row_index"].tolist() == [0, 1, 2, 3, 4]
    assert prepared.loc[0, "votes"] == 200
    assert prepared.loc[0, "log_votes"] == pytest.approx(np.log1p(200))
    assert "log_votes" not in source.columns


def test_scale_and_k_evaluation_are_finite_and_reproducible():
    prepared, _ = prepare_clustering_data(mini_data())
    scaled, scaler = scale_features(prepared)
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-10)
    scores = evaluate_k_values(scaled, [2, 3], silhouette_sample_size=5)
    assert scores["k"].tolist() == [2, 3]
    assert scores["silhouette_sample_size"].tolist() == [5, 5]
    again = evaluate_k_values(scaled, [2, 3], silhouette_sample_size=5)
    assert np.allclose(scores["silhouette_score"], again["silhouette_score"])


def test_fit_and_profile_cluster_counts_sum_to_eligible_rows():
    prepared, _ = prepare_clustering_data(mini_data())
    labeled, _, _, _ = fit_final_model(prepared, n_clusters=2)
    profile = profile_clusters(labeled)
    assert labeled["cluster"].notna().all()
    assert int(profile["restaurant_count"].sum()) == len(prepared)
    assert profile["percentage"].sum() == pytest.approx(100.0)
    assert set(profile.columns) >= {"mean_rating", "median_votes", "median_cost"}


def test_invalid_or_missing_features_raise_clear_errors():
    data = mini_data().assign(votes=-1)
    with pytest.raises(ValueError, match="non-negative"):
        prepare_clustering_data(data)
