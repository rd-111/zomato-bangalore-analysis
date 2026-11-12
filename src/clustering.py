"""Reproducible K-Means restaurant segmentation helpers.

The model uses rating, log1p(votes), and approximate cost for two. Original
values are retained alongside labels for readable cluster profiles.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.eda import COST, RATING
from src.utils import FIGURES_DIR, PROCESSED_DATA_DIR

VOTES = "votes"
CLUSTER_FEATURES = (RATING, VOTES, COST)
MODEL_FEATURES = (RATING, "log_votes", COST)
RANDOM_STATE = 42


def load_cleaned_data(path: Path | None = None) -> pd.DataFrame:
    """Reuse the EDA loader so clustering starts from the established clean file."""
    from src.eda import load_cleaned_data as load_eda_cleaned_data
    return load_eda_cleaned_data(path)


def prepare_clustering_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the complete-case clustering subset and log-transformed model data.

    Original feature values remain in the returned dataframe. ``log_votes`` is
    an additional modeling feature and does not replace the original votes.
    The input dataframe is never modified.
    """
    missing_columns = [column for column in CLUSTER_FEATURES if column not in data.columns]
    if missing_columns:
        raise KeyError(f"Clustering input is missing columns: {missing_columns}")
    selected = data.loc[:, list(CLUSTER_FEATURES)].copy()
    selected.insert(0, "source_row_index", data.index)
    missing_before = selected.loc[:, list(CLUSTER_FEATURES)].isna().sum()
    for column in CLUSTER_FEATURES:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    complete = selected.dropna(subset=list(CLUSTER_FEATURES)).copy()
    # K-Means and log1p require finite values; fail loudly rather than dropping silently.
    model_base = complete.loc[:, list(CLUSTER_FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(model_base).all():
        raise ValueError("Clustering features contain infinite values; inspect source data before proceeding.")
    if (complete[VOTES] < 0).any():
        raise ValueError("Vote counts must be non-negative before applying log1p.")
    if (complete[COST] < 0).any():
        raise ValueError("Cost values must be non-negative before clustering.")
    complete["log_votes"] = np.log1p(complete[VOTES].astype(float))
    report = {
        "records_before_selection": int(len(data)),
        "records_after_selection": int(len(complete)),
        "records_excluded_for_missing_or_unparseable_features": int(len(data) - len(complete)),
        "missing_before_selection": {str(k): int(v) for k, v in missing_before.items()},
        "complete_feature_missing_after_selection": int(complete[list(CLUSTER_FEATURES)].isna().sum().sum()),
        "features_for_model": list(MODEL_FEATURES),
        "feature_summary_before_transform": complete[list(CLUSTER_FEATURES)].describe().to_dict(),
        "votes_skew_before_log": float(complete[VOTES].skew()),
        "votes_skew_after_log": float(complete["log_votes"].skew()),
        "cost_skew": float(complete[COST].skew()),
    }
    return complete, report


def scale_features(data: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Standardize rating, log votes, and cost; return matrix and fitted scaler."""
    missing = [column for column in MODEL_FEATURES if column not in data.columns]
    if missing:
        raise KeyError(f"Model feature columns missing: {missing}")
    features = data.loc[:, list(MODEL_FEATURES)].astype(float)
    if features.empty or features.isna().any().any() or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Model features must be non-empty, finite, and complete before scaling.")
    scaler = StandardScaler()
    return scaler.fit_transform(features), scaler


def evaluate_k_values(scaled_features: np.ndarray, k_values=range(2, 9), *,
                      random_state: int = RANDOM_STATE, silhouette_sample_size: int | None = 10000) -> pd.DataFrame:
    """Fit K-Means for K=2..8 and calculate inertia and silhouette scores.

    Silhouette uses at most ``silhouette_sample_size`` rows, selected by the
    metric's fixed random_state sampling. Inertia uses all eligible rows.
    """
    values = np.asarray(scaled_features, dtype=float)
    if values.ndim != 2 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("scaled_features must be a finite 2D array with at least three rows.")
    rows = []
    for k in k_values:
        if k < 2 or k >= len(values):
            raise ValueError(f"Each K must be >= 2 and less than sample size; got {k}.")
        model = KMeans(n_clusters=int(k), random_state=random_state, n_init=10)
        labels = model.fit_predict(values)
        sample_size = min(len(values), silhouette_sample_size or len(values))
        score = silhouette_score(values, labels, sample_size=sample_size if sample_size < len(values) else None,
                                 random_state=random_state)
        rows.append({"k": int(k), "inertia": float(model.inertia_),
                     "silhouette_score": float(score), "silhouette_sample_size": int(sample_size)})
    return pd.DataFrame(rows)


def choose_cluster_count(evaluation: pd.DataFrame, selected_k: int) -> int:
    """Validate and return a documented human-selected K from evaluation results."""
    available = set(evaluation["k"].astype(int))
    if selected_k not in available:
        raise ValueError(f"selected_k={selected_k} was not evaluated: {sorted(available)}")
    return selected_k


def fit_final_model(data: pd.DataFrame, n_clusters: int, *, random_state: int = RANDOM_STATE
                    ) -> tuple[pd.DataFrame, StandardScaler, KMeans, np.ndarray]:
    """Fit scaler and final K-Means, returning labeled rows and fitted objects."""
    scaled, scaler = scale_features(data)
    if n_clusters < 2 or n_clusters >= len(data):
        raise ValueError("n_clusters must be at least 2 and less than the eligible sample size.")
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(scaled)
    labeled = data.copy()
    labeled["cluster"] = labels.astype(int)
    return labeled, scaler, model, scaled


def profile_clusters(labeled_data: pd.DataFrame) -> pd.DataFrame:
    """Summarize cluster size and original feature mean/median for interpretation."""
    required = {"cluster", RATING, VOTES, COST}
    missing = required.difference(labeled_data.columns)
    if missing:
        raise KeyError(f"Labeled clustering data is missing columns: {sorted(missing)}")
    grouped = labeled_data.groupby("cluster").agg(
        restaurant_count=(RATING, "size"),
        mean_rating=(RATING, "mean"), median_rating=(RATING, "median"),
        mean_votes=(VOTES, "mean"), median_votes=(VOTES, "median"),
        mean_cost=(COST, "mean"), median_cost=(COST, "median"),
    )
    grouped["percentage"] = grouped["restaurant_count"] / len(labeled_data) * 100
    return grouped[["restaurant_count", "percentage", "mean_rating", "median_rating",
                    "mean_votes", "median_votes", "mean_cost", "median_cost"]].sort_index()


def name_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """Create concise relative profile names from actual cluster medians."""
    names = {}
    rating_mid = profile["median_rating"].median()
    votes_mid = profile["median_votes"].median()
    cost_order = profile["median_cost"].sort_values().index.tolist()
    for cluster, row in profile.iterrows():
        if cluster == cost_order[0]:
            price = "budget"
        elif cluster == cost_order[-1]:
            price = "premium"
        else:
            price = "mid-priced"
        rating = "high-rated" if row["median_rating"] >= rating_mid else "lower-rated"
        engagement = "high-engagement" if row["median_votes"] >= votes_mid else "lower-engagement"
        names[int(cluster)] = f"{price}, {rating}, {engagement}"
    return names


def pca_projection(scaled_features: np.ndarray) -> tuple[np.ndarray, PCA]:
    """Project standardized original features into two components for plotting only."""
    if scaled_features.ndim != 2 or scaled_features.shape[1] < 2:
        raise ValueError("PCA visualization requires at least two scaled features.")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    return pca.fit_transform(scaled_features), pca


def generate_clustering_figures(evaluation: pd.DataFrame, profile: pd.DataFrame,
                                labeled_data: pd.DataFrame, scaled_features: np.ndarray,
                                *, output_dir: Path = FIGURES_DIR) -> dict[str, Path]:
    """Generate seven clustering visualizations and return their file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    palette = sns.color_palette("colorblind", n_colors=len(profile))
    outputs: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=evaluation, x="k", y="inertia", marker="o", color="#277DA1", ax=ax)
    ax.set(title="K-Means elbow curve", xlabel="Number of clusters (K)", ylabel="Inertia (within-cluster sum of squares)")
    outputs["11_kmeans_elbow.png"] = _save(fig, "11_kmeans_elbow.png", output_dir)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=evaluation, x="k", y="silhouette_score", marker="o", color="#F3722C", ax=ax)
    ax.set(title="Silhouette score by number of clusters", xlabel="Number of clusters (K)", ylabel="Mean silhouette score", ylim=(-1, 1))
    outputs["12_silhouette_by_k.png"] = _save(fig, "12_silhouette_by_k.png", output_dir)

    plot_profile = profile.copy()
    plot_profile["cluster_name"] = [f"C{k}: {name_clusters(profile)[k]}" for k in plot_profile.index]
    for filename, column, title, xlabel, color in [
        ("13_restaurants_by_cluster.png", "restaurant_count", "Restaurants in each cluster", "Restaurant listings", "#277DA1"),
        ("14_average_rating_by_cluster.png", "mean_rating", "Average rating by cluster", "Mean rating (out of 5)", "#43AA8B"),
        ("15_median_cost_by_cluster.png", "median_cost", "Median listed cost by cluster", "Median approximate cost for two (₹)", "#F9C74F"),
        ("16_median_votes_by_cluster.png", "median_votes", "Median votes by cluster", "Median votes", "#9B5DE5"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=plot_profile.reset_index(), y="cluster_name", x=column, color=color, ax=ax)
        ax.set(title=title, xlabel=xlabel, ylabel="Cluster")
        outputs[filename] = _save(fig, filename, output_dir)

    projection, pca = pca_projection(scaled_features)
    pca_data = pd.DataFrame({"PC1": projection[:, 0], "PC2": projection[:, 1],
                             "cluster": labeled_data["cluster"].to_numpy()})
    # Fixed random sample keeps the PCA scatter legible without changing the model.
    pca_plot = pca_data.sample(min(12000, len(pca_data)), random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=pca_plot, x="PC1", y="PC2", hue="cluster", palette=palette,
                    alpha=.45, s=18, linewidth=0, ax=ax)
    variance = pca.explained_variance_ratio_.sum() * 100
    ax.set(title=f"Restaurant clusters projected to PCA (first two components explain {variance:.1f}%)",
           xlabel="Principal component 1", ylabel="Principal component 2")
    outputs["17_clusters_pca_2d.png"] = _save(fig, "17_clusters_pca_2d.png", output_dir)
    return outputs


def _save(fig, filename: str, output_dir: Path) -> Path:
    path = output_dir / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def run_clustering_pipeline(data: pd.DataFrame | None = None, *, selected_k: int,
                            input_path: Path | None = None, output_dir: Path = FIGURES_DIR,
                            silhouette_sample_size: int = 10000,
                            save_outputs: bool = True) -> dict[str, Any]:
    """Run preparation, K evaluation, final clustering, profiling, and plots.

    ``selected_k`` must be chosen after reviewing the returned evaluation table;
    no default cluster count is imposed.
    """
    source = load_cleaned_data(input_path) if data is None else data.copy()
    source = source.reset_index(drop=True)
    eligible, preparation = prepare_clustering_data(source)
    scaled_for_search, _ = scale_features(eligible)
    evaluation = evaluate_k_values(scaled_for_search, range(2, 9), silhouette_sample_size=silhouette_sample_size)
    selected_k = choose_cluster_count(evaluation, selected_k)
    feature_labeled, scaler, model, scaled = fit_final_model(eligible, selected_k)
    # Restore all original columns, preserving source row correspondence.
    source_positions = feature_labeled["source_row_index"].astype(int).to_numpy()
    labeled = source.iloc[source_positions].copy()
    labeled["cluster"] = feature_labeled["cluster"].to_numpy(dtype=int)
    profile = profile_clusters(labeled)
    labels = name_clusters(profile)
    figures = generate_clustering_figures(evaluation, profile, labeled, scaled, output_dir=output_dir)
    if save_outputs:
        processed_dir = PROCESSED_DATA_DIR
        processed_dir.mkdir(parents=True, exist_ok=True)
        labeled.to_csv(processed_dir / "zomato_clustered.csv", index=False)
        evaluation.to_csv(processed_dir / "clustering_k_evaluation.csv", index=False)
        profile.assign(cluster_name=pd.Series(labels)).to_csv(processed_dir / "cluster_profile.csv")
    return {"eligible_data": eligible, "preparation_report": preparation,
            "evaluation": evaluation, "selected_k": selected_k, "labeled_data": labeled,
            "profile": profile, "cluster_names": labels, "scaler": scaler,
            "model": model, "scaled_features": scaled, "figures": figures}
