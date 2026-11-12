"""Reusable EDA preparation, summaries, statistics, and visualizations.

All summaries retain their sample counts. Rating-specific functions use only
valid ratings; restaurant-count analyses use all rows. Functions do not silently
drop observations: each returned table includes the counts used.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import FIGURES_DIR, PROCESSED_DATA_DIR, ensure_project_directories

RATING = "rate"
COST = "approx_cost(for two people)"
MIN_LOCATION_RATINGS = 30
MIN_CUISINE_RATINGS = 50
PALETTE = {"Yes": "#2A9D8F", "No": "#E76F51"}

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 220, "axes.titleweight": "bold"})


def load_cleaned_data(path: Path | None = None) -> pd.DataFrame:
    """Load the cleaned CSV, preserving nulls and validated numeric columns."""
    dataset_path = Path(path) if path is not None else PROCESSED_DATA_DIR / "zomato_cleaned.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Cleaned dataset not found at {dataset_path}. Run the cleaning pipeline first.")
    try:
        data = pd.read_csv(dataset_path, low_memory=False)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not parse cleaned dataset at {dataset_path}: {exc}") from exc
    for column in (RATING, "votes", COST):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def prepare_analytical_data(data: pd.DataFrame) -> dict[str, Any]:
    """Create all-row and field-complete subsets with explicit exclusion counts."""
    required = {RATING, "votes", COST, "location", "online_order", "book_table", "cuisines"}
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise KeyError(f"Cleaned data is missing required columns: {missing_columns}")
    all_rows = data.copy()
    rated = all_rows.loc[all_rows[RATING].between(0, 5, inclusive="both")].copy()
    cost_rated = rated.dropna(subset=[COST]).copy()
    votes_rated = rated.dropna(subset=["votes"]).copy()
    report = {
        "all_rows": len(all_rows),
        "missing_rating_rows_retained_for_counts": int(all_rows[RATING].isna().sum()),
        "rating_complete_rows": len(rated),
        "rating_rows_excluded_from_rating_analyses": int(len(all_rows) - len(rated)),
        "cost_and_rating_rows": len(cost_rated),
        "rating_rows_missing_cost": int(rated[COST].isna().sum()),
        "votes_and_rating_rows": len(votes_rated),
        "rating_rows_missing_votes": int(rated["votes"].isna().sum()),
    }
    return {"all_rows": all_rows, "rated": rated, "cost_rated": cost_rated,
            "votes_rated": votes_rated, "subset_report": report}


def missing_value_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and percentages, sorted by missing count."""
    return pd.DataFrame({"missing_count": data.isna().sum(),
                         "missing_percent": data.isna().mean().mul(100)}).sort_values("missing_count", ascending=False)


def skewness_summary(data: pd.DataFrame, columns: tuple[str, ...] = ("votes", COST)) -> pd.DataFrame:
    """Summarize numeric range and skew; missing data is excluded per field only."""
    rows = []
    for column in columns:
        if column in data:
            values = pd.to_numeric(data[column], errors="coerce")
            rows.append({"column": column, "non_missing_n": int(values.notna().sum()),
                         "missing_n": int(values.isna().sum()), "min": values.min(),
                         "median": values.median(), "mean": values.mean(), "max": values.max(),
                         "skewness": values.skew()})
    return pd.DataFrame(rows).set_index("column")


def location_summary(data: pd.DataFrame, min_rating_count: int = MIN_LOCATION_RATINGS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all-row restaurant counts and rating means with sample counts."""
    counts = data.groupby("location", dropna=False).size().rename("restaurant_count").sort_values(ascending=False).to_frame()
    valid = data.dropna(subset=["location", RATING])
    ratings = valid.groupby("location")[RATING].agg(mean_rating="mean", median_rating="median", rating_count="count")
    ratings["restaurant_count"] = counts["restaurant_count"]
    ratings["eligible_for_ranking"] = ratings["rating_count"] >= min_rating_count
    return counts, ratings.sort_values(["eligible_for_ranking", "mean_rating"], ascending=[False, False])


def cuisine_summary(data: pd.DataFrame, min_rating_count: int = MIN_CUISINE_RATINGS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize exact listed cuisine combinations, retaining counts for context."""
    counts = data["cuisines"].value_counts(dropna=False).rename("restaurant_count").to_frame()
    valid = data.dropna(subset=["cuisines", RATING])
    ratings = valid.groupby("cuisines")[RATING].agg(mean_rating="mean", median_rating="median", rating_count="count")
    ratings["restaurant_count"] = counts["restaurant_count"]
    ratings["eligible_for_ranking"] = ratings["rating_count"] >= min_rating_count
    return counts, ratings.sort_values(["eligible_for_ranking", "mean_rating"], ascending=[False, False])


def group_rating_summary(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Mean, median, standard deviation, and n for ratings by a category."""
    valid = data.dropna(subset=[group_column, RATING])
    return valid.groupby(group_column)[RATING].agg(
        mean_rating="mean", median_rating="median", std_rating="std", rating_count="count"
    ).sort_values("mean_rating", ascending=False)


def correlation_summary(data: pd.DataFrame, pairs: tuple[tuple[str, str], ...] = ((COST, RATING), ("votes", RATING))) -> pd.DataFrame:
    """Calculate Pearson and Spearman correlation for pairwise-complete rows."""
    rows = []
    for x, y in pairs:
        valid = data[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        rows.append({"feature_x": x, "feature_y": y, "n": len(valid),
                     "pearson_r": valid[x].corr(valid[y], method="pearson"),
                     "spearman_rho": valid[x].corr(valid[y], method="spearman")})
    return pd.DataFrame(rows)


def correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Return pairwise Pearson correlations for numeric columns of interest."""
    columns = [column for column in (RATING, "votes", COST) if column in data]
    return data[columns].corr(method="pearson")


def save_figure(figure, filename: str, output_dir: Path = FIGURES_DIR) -> Path:
    """Save a Matplotlib figure as a high-resolution PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    figure.savefig(destination, bbox_inches="tight", dpi=220, facecolor="white")
    plt.close(figure)
    return destination


def _bar_chart(table: pd.DataFrame, value: str, title: str, xlabel: str,
               filename: str, *, color: str = "#277DA1", count_labels: bool = False) -> Path:
    plot = table.sort_values(value).copy()
    fig, ax = plt.subplots(figsize=(10, max(5, 0.42 * len(plot))))
    sns.barplot(data=plot.reset_index(), y=plot.index.name or plot.index.name, x=value, ax=ax, color=color)
    # The reset_index column is named after the original index, so use its first column robustly.
    category = plot.index.name or plot.reset_index().columns[0]
    ax.set(ylabel="", xlabel=xlabel, title=title)
    if count_labels:
        for i, (_, row) in enumerate(plot.iterrows()):
            ax.text(row[value] + 0.02, i, f"n={int(row['rating_count']):,}", va="center", fontsize=9)
    ax.grid(axis="y", visible=False)
    return save_figure(fig, filename)


def generate_eda_figures(data: pd.DataFrame, *, min_location_ratings: int = MIN_LOCATION_RATINGS,
                         min_cuisine_ratings: int = MIN_CUISINE_RATINGS,
                         output_dir: Path = FIGURES_DIR) -> dict[str, Path]:
    """Generate ten requested EDA charts and return their output paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rated = data.loc[data[RATING].between(0, 5, inclusive="both")].copy()
    locations, location_ratings = location_summary(data, min_location_ratings)
    cuisines, cuisine_ratings = cuisine_summary(data, min_cuisine_ratings)
    paths: dict[str, Path] = {}

    top = locations.head(10).sort_values("restaurant_count")
    fig, ax = plt.subplots(figsize=(10, 6))
    top_plot = top.reset_index()
    sns.barplot(x="restaurant_count", y="location", data=top_plot, ax=ax, color="#277DA1")
    ax.set(title="Bangalore locations with the most restaurant listings", xlabel="Restaurant listings", ylabel="Location")
    for y, value in enumerate(top["restaurant_count"]): ax.text(value + 25, y, f"{value:,}", va="center", fontsize=9)
    paths["01_top_locations.png"] = save_figure(fig, "01_top_locations.png", output_dir)

    eligible_locations = location_ratings.loc[location_ratings["rating_count"] >= min_location_ratings].nlargest(10, "mean_rating").sort_values("mean_rating")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="mean_rating", y="location", data=eligible_locations.reset_index(), ax=ax, color="#43AA8B")
    ax.set(title=f"Highest average-rated locations (at least {min_location_ratings} rated restaurants)", xlabel="Mean rating (out of 5)", ylabel="Location", xlim=(0, 5.6))
    for y, (_, row) in enumerate(eligible_locations.iterrows()): ax.text(row.mean_rating + .03, y, f"{row.mean_rating:.2f}  (n={int(row.rating_count):,})", va="center", fontsize=8)
    paths["02_best_rated_locations.png"] = save_figure(fig, "02_best_rated_locations.png", output_dir)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.histplot(rated[RATING], bins=np.arange(1.5, 5.05, .1), kde=True, color="#577590", ax=ax)
    ax.set(title="Distribution of available restaurant ratings", xlabel="Rating (out of 5)", ylabel="Restaurant listings")
    paths["03_rating_distribution.png"] = save_figure(fig, "03_rating_distribution.png", output_dir)

    for col, filename, title in [("online_order", "04_online_order_vs_rating.png", "Ratings by online ordering availability"),
                                  ("book_table", "05_table_booking_vs_rating.png", "Ratings by table booking availability")]:
        plotdata = rated.dropna(subset=[col, RATING])
        order = [value for value in ["No", "Yes"] if value in plotdata[col].unique()]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        sns.boxplot(data=plotdata, x=col, y=RATING, order=order, palette=PALETTE, hue=col, legend=False, ax=ax)
        sns.stripplot(data=plotdata.sample(min(900, len(plotdata)), random_state=42), x=col, y=RATING, order=order, color="#263238", alpha=.12, size=2, ax=ax)
        ax.set(title=title, xlabel=col.replace("_", " ").title(), ylabel="Rating (out of 5)", ylim=(1.5, 5.1))
        counts = plotdata[col].value_counts()
        ax.set_xticks(range(len(order)), [f"{v}\n(n={counts[v]:,})" for v in order])
        paths[filename] = save_figure(fig, filename, output_dir)

    cost_rated = rated.dropna(subset=[COST])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.scatterplot(data=cost_rated, x=COST, y=RATING, alpha=.18, s=15, color="#277DA1", ax=ax)
    cost_bins = pd.qcut(cost_rated[COST], q=20, duplicates="drop")
    cost_trend = cost_rated.groupby(cost_bins, observed=True).agg(x=(COST, "median"), y=(RATING, "median"))
    ax.plot(cost_trend["x"], cost_trend["y"], color="#E76F51", linewidth=2.5, label="Binned median rating")
    ax.legend(frameon=False)
    ax.set(title="Listed cost for two versus restaurant rating", xlabel="Approximate cost for two (₹)", ylabel="Rating (out of 5)", ylim=(1.5, 5.05))
    paths["06_cost_vs_rating.png"] = save_figure(fig, "06_cost_vs_rating.png", output_dir)

    corr = correlation_matrix(rated)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"label": "Pearson correlation"})
    ax.set_title("Correlation among rating, votes, and listed cost")
    paths["07_numeric_correlation.png"] = save_figure(fig, "07_numeric_correlation.png", output_dir)

    top_cuisines = cuisines.drop(index=np.nan, errors="ignore").head(10).sort_values("restaurant_count")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="restaurant_count", y="cuisines", data=top_cuisines.reset_index(), ax=ax, color="#F9C74F")
    ax.set(title="Most frequently listed cuisine combinations", xlabel="Restaurant listings", ylabel="Listed cuisine combination")
    paths["08_top_cuisines_by_count.png"] = save_figure(fig, "08_top_cuisines_by_count.png", output_dir)

    eligible_cuisines = cuisine_ratings.loc[cuisine_ratings["rating_count"] >= min_cuisine_ratings].nlargest(10, "mean_rating").sort_values("mean_rating")
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(x="mean_rating", y="cuisines", data=eligible_cuisines.reset_index(), ax=ax, color="#90BE6D")
    ax.set(title=f"Highest average-rated cuisine combinations (at least {min_cuisine_ratings} rated listings)", xlabel="Mean rating (out of 5)", ylabel="Listed cuisine combination", xlim=(0, 5.65))
    for y, (_, row) in enumerate(eligible_cuisines.iterrows()): ax.text(row.mean_rating + .03, y, f"{row.mean_rating:.2f} (n={int(row.rating_count):,})", va="center", fontsize=8)
    paths["09_top_rated_cuisines.png"] = save_figure(fig, "09_top_rated_cuisines.png", output_dir)

    votes_rated = rated.dropna(subset=["votes"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.scatterplot(data=votes_rated, x="votes", y=RATING, alpha=.16, s=15, color="#577590", ax=ax)
    vote_bins = pd.qcut(votes_rated["votes"], q=20, duplicates="drop")
    vote_trend = votes_rated.groupby(vote_bins, observed=True).agg(x=("votes", "median"), y=(RATING, "median"))
    ax.plot(vote_trend["x"], vote_trend["y"], color="#F94144", linewidth=2.5, marker="o", label="Binned median rating")
    ax.legend(frameon=False)
    ax.set_xscale("symlog", linthresh=1)
    ax.set(title="Customer votes versus restaurant rating", xlabel="Votes (symlog scale to show low and high counts)", ylabel="Rating (out of 5)", ylim=(1.5, 5.05))
    paths["10_votes_vs_rating.png"] = save_figure(fig, "10_votes_vs_rating.png", output_dir)
    return paths


def calculate_eda_results(data: pd.DataFrame, *, min_location_ratings: int = MIN_LOCATION_RATINGS,
                          min_cuisine_ratings: int = MIN_CUISINE_RATINGS) -> dict[str, Any]:
    """Compute actual descriptive statistics used in the notebook and report."""
    subsets = prepare_analytical_data(data)
    rated = subsets["rated"]
    means = {column: group_rating_summary(rated, column) for column in ("online_order", "book_table")}
    location_counts, location_ratings = location_summary(data, min_location_ratings)
    cuisine_counts, cuisine_ratings = cuisine_summary(data, min_cuisine_ratings)
    return {
        "subsets": subsets["subset_report"],
        "overall_rating": {"count": int(rated[RATING].count()), "mean": rated[RATING].mean(), "median": rated[RATING].median(), "std": rated[RATING].std()},
        "online_order_rating": means["online_order"],
        "book_table_rating": means["book_table"],
        "correlations": correlation_summary(rated),
        "location_counts": location_counts,
        "location_ratings": location_ratings,
        "cuisine_counts": cuisine_counts,
        "cuisine_ratings": cuisine_ratings,
        "skewness": skewness_summary(data),
    }
