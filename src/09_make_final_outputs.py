from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures" / "final"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def percentile_rank(series):
    return series.rank(method="average", pct=True) * 100


def make_decile(series):
    pct_rank = series.rank(method="first", pct=True)
    return np.ceil(pct_rank * 10).astype(int).clip(1, 10)


def save_histogram(series, title, xlabel, output_path, bins=50):
    plt.figure(figsize=(8, 5))
    plt.hist(series.dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of listings")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_scatter(df, x_col, y_col, title, xlabel, ylabel, output_path):
    plot_df = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()

    plt.figure(figsize=(8, 5))
    plt.scatter(plot_df[x_col], plot_df[y_col], alpha=0.35, s=14)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_bar_chart(df, x_col, y_col, title, xlabel, ylabel, output_path, rotate=True):
    plt.figure(figsize=(10, 5))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if rotate:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def label_candidate(row):
    if (
        row["final_overpricing_score"] >= 95
        and row["residual_signal_score"] >= 90
        and row["oof_percent_above_expected"] > 0
    ):
        return "strong candidate"

    if (
        row["final_overpricing_score"] >= 90
        and row["residual_signal_score"] >= 85
        and row["oof_percent_above_expected"] > 0
    ):
        return "moderate candidate"

    if row["final_overpricing_score"] >= 80 and row["oof_percent_above_expected"] > 0:
        return "watchlist"

    return "not flagged"


def main():
    input_path = PROCESSED_DIR / "cluster_density_scores.csv"
    require_file(input_path)

    df = pd.read_csv(input_path, low_memory=False)
    df["listing_id"] = df["listing_id"].astype(str)

    required_cols = [
        "listing_id",
        "actual_price",
        "oof_predicted_price",
        "oof_percent_above_expected",
        "residual_signal_score",
        "cluster_anomaly_score",
        "availability_rate",
        "reviews_last_365_days",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns from cluster_density_scores.csv: {missing}")

    df = df.copy()

    df["availability_signal_score"] = percentile_rank(df["availability_rate"])

    review_counts = pd.to_numeric(df["reviews_last_365_days"], errors="coerce").fillna(0)
    df["low_recent_review_score"] = 100 - percentile_rank(review_counts)

    df["demand_proxy_score"] = (
        0.60 * df["availability_signal_score"]
        + 0.40 * df["low_recent_review_score"]
    )

    df["final_overpricing_score"] = (
        0.60 * df["residual_signal_score"]
        + 0.25 * df["cluster_anomaly_score"]
        + 0.15 * df["demand_proxy_score"]
    )

    df["final_score_decile"] = make_decile(df["final_overpricing_score"])
    df["candidate_label"] = df.apply(label_candidate, axis=1)

    df["price_gap_dollars"] = df["actual_price"] - df["oof_predicted_price"]

    df["price_gap_label"] = "below or near expected"
    df.loc[df["oof_percent_above_expected"] >= 25, "price_gap_label"] = "25%+ above expected"
    df.loc[df["oof_percent_above_expected"] >= 50, "price_gap_label"] = "50%+ above expected"
    df.loc[df["oof_percent_above_expected"] >= 100, "price_gap_label"] = "100%+ above expected"

    output_path = PROCESSED_DIR / "final_overpricing_scores.csv"
    df.to_csv(output_path, index=False)

    top_cols = [
        "listing_id",
        "name",
        "room_type_model",
        "property_type",
        "neighbourhood_model",
        "capacity_bucket",
        "actual_price",
        "oof_predicted_price",
        "price_gap_dollars",
        "oof_percent_above_expected",
        "residual_signal_score",
        "cluster_anomaly_score",
        "demand_proxy_score",
        "final_overpricing_score",
        "candidate_label",
        "price_gap_label",
        "availability_rate",
        "reviews_last_365_days",
        "cluster",
    ]

    top_cols = [col for col in top_cols if col in df.columns]

    top_candidates = (
        df[df["candidate_label"] != "not flagged"]
        .sort_values("final_overpricing_score", ascending=False)
        .head(100)[top_cols]
        .copy()
    )

    top_candidates.to_csv(TABLES_DIR / "final_top_overpricing_candidates.csv", index=False)

    top_25 = (
        df.sort_values("final_overpricing_score", ascending=False)
        .head(25)[top_cols]
        .copy()
    )

    top_25.to_csv(TABLES_DIR / "final_top_25_overpricing_candidates.csv", index=False)

    score_summary = pd.DataFrame(
        {
            "metric": [
                "rows_scored",
                "mean_final_overpricing_score",
                "median_final_overpricing_score",
                "strong_candidates",
                "moderate_candidates",
                "watchlist_candidates",
                "not_flagged",
                "median_actual_price",
                "median_predicted_price",
                "median_percent_above_expected",
                "median_availability_rate",
                "median_reviews_last_365_days",
            ],
            "value": [
                len(df),
                round(df["final_overpricing_score"].mean(), 2),
                round(df["final_overpricing_score"].median(), 2),
                int((df["candidate_label"] == "strong candidate").sum()),
                int((df["candidate_label"] == "moderate candidate").sum()),
                int((df["candidate_label"] == "watchlist").sum()),
                int((df["candidate_label"] == "not flagged").sum()),
                round(df["actual_price"].median(), 2),
                round(df["oof_predicted_price"].median(), 2),
                round(df["oof_percent_above_expected"].median(), 2),
                round(df["availability_rate"].median(), 4),
                round(df["reviews_last_365_days"].median(), 2),
            ],
        }
    )

    score_summary.to_csv(TABLES_DIR / "final_score_summary.csv", index=False)

    label_summary = (
        df.groupby("candidate_label")
        .agg(
            listings=("listing_id", "count"),
            median_actual_price=("actual_price", "median"),
            median_predicted_price=("oof_predicted_price", "median"),
            median_percent_above_expected=("oof_percent_above_expected", "median"),
            median_availability_rate=("availability_rate", "median"),
            median_reviews_last_365_days=("reviews_last_365_days", "median"),
            median_final_overpricing_score=("final_overpricing_score", "median"),
        )
        .reset_index()
        .sort_values("median_final_overpricing_score", ascending=False)
    )

    label_summary.to_csv(TABLES_DIR / "final_candidate_label_summary.csv", index=False)

    decile_summary = (
        df.groupby("final_score_decile")
        .agg(
            listings=("listing_id", "count"),
            median_actual_price=("actual_price", "median"),
            median_predicted_price=("oof_predicted_price", "median"),
            median_percent_above_expected=("oof_percent_above_expected", "median"),
            median_availability_rate=("availability_rate", "median"),
            median_reviews_last_365_days=("reviews_last_365_days", "median"),
            median_residual_signal_score=("residual_signal_score", "median"),
            median_cluster_anomaly_score=("cluster_anomaly_score", "median"),
            median_demand_proxy_score=("demand_proxy_score", "median"),
            median_final_overpricing_score=("final_overpricing_score", "median"),
        )
        .reset_index()
        .sort_values("final_score_decile")
    )

    decile_summary.to_csv(TABLES_DIR / "final_score_decile_summary.csv", index=False)

    if "neighbourhood_model" in df.columns:
        neighborhood_summary = (
            df.groupby("neighbourhood_model")
            .agg(
                listings=("listing_id", "count"),
                strong_candidates=("candidate_label", lambda x: (x == "strong candidate").sum()),
                median_actual_price=("actual_price", "median"),
                median_final_overpricing_score=("final_overpricing_score", "median"),
                median_availability_rate=("availability_rate", "median"),
            )
            .reset_index()
        )

        neighborhood_summary["strong_candidate_rate"] = (
            neighborhood_summary["strong_candidates"] / neighborhood_summary["listings"]
        )

        neighborhood_summary = neighborhood_summary.sort_values(
            ["strong_candidates", "strong_candidate_rate"],
            ascending=False,
        )

        neighborhood_summary.to_csv(TABLES_DIR / "final_neighborhood_summary.csv", index=False)

    if "room_type_model" in df.columns:
        room_type_summary = (
            df.groupby("room_type_model")
            .agg(
                listings=("listing_id", "count"),
                strong_candidates=("candidate_label", lambda x: (x == "strong candidate").sum()),
                median_actual_price=("actual_price", "median"),
                median_final_overpricing_score=("final_overpricing_score", "median"),
                median_availability_rate=("availability_rate", "median"),
            )
            .reset_index()
            .sort_values("listings", ascending=False)
        )

        room_type_summary["strong_candidate_rate"] = (
            room_type_summary["strong_candidates"] / room_type_summary["listings"]
        )

        room_type_summary.to_csv(TABLES_DIR / "final_room_type_summary.csv", index=False)

    save_histogram(
        df["final_overpricing_score"],
        "Final Overpricing Score Distribution",
        "Final overpricing score",
        FIGURES_DIR / "final_overpricing_score_distribution.png",
    )

    save_scatter(
        df,
        "final_overpricing_score",
        "availability_rate",
        "Final Score vs. Availability Rate",
        "Final overpricing score",
        "Availability rate",
        FIGURES_DIR / "final_score_vs_availability.png",
    )

    save_scatter(
        df,
        "final_overpricing_score",
        "reviews_last_365_days",
        "Final Score vs. Recent Reviews",
        "Final overpricing score",
        "Reviews last 365 days",
        FIGURES_DIR / "final_score_vs_recent_reviews.png",
    )

    save_bar_chart(
        decile_summary,
        "final_score_decile",
        "median_availability_rate",
        "Median Availability Rate by Final Score Decile",
        "Final score decile",
        "Median availability rate",
        FIGURES_DIR / "decile_vs_availability.png",
        rotate=False,
    )

    save_bar_chart(
        decile_summary,
        "final_score_decile",
        "median_percent_above_expected",
        "Median Percent Above Expected by Final Score Decile",
        "Final score decile",
        "Median percent above expected",
        FIGURES_DIR / "decile_vs_percent_above_expected.png",
        rotate=False,
    )

    top_20_plot = top_25.head(20).copy()
    top_20_plot["short_id"] = top_20_plot["listing_id"].astype(str).str[-6:]

    save_bar_chart(
        top_20_plot.sort_values("final_overpricing_score", ascending=True),
        "short_id",
        "final_overpricing_score",
        "Top 20 Final Overpricing Candidates",
        "Listing ID ending",
        "Final overpricing score",
        FIGURES_DIR / "top_20_final_candidates.png",
        rotate=True,
    )

    print(f"\nSaved final scores to: {output_path}")

    print("\nFinal score summary:")
    print(score_summary.to_string(index=False))

    print("\nCandidate label summary:")
    print(label_summary.to_string(index=False))

    print("\nScore decile summary:")
    print(decile_summary.to_string(index=False))

    print("\nTop final candidates:")
    print(top_25.head(15).to_string(index=False))


if __name__ == "__main__":
    main()