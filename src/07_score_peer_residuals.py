from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures" / "model_eval"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def percentile_rank(series):
    return series.rank(method="average", pct=True) * 100


def robust_z_score(values):
    values = values.astype(float)
    median = values.median()
    mad = np.median(np.abs(values - median))

    if pd.isna(mad) or mad == 0:
        std = values.std(ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(np.zeros(len(values)), index=values.index)

        return (values - values.mean()) / std

    return 0.6745 * (values - median) / mad


def add_peer_residual_features(df, group_col, prefix, min_group_size=20):
    group_stats = (
        df.groupby(group_col, dropna=False)
        .agg(
            group_size=("listing_id", "count"),
            group_median_price=("actual_price", "median"),
            group_mean_price=("actual_price", "mean"),
            group_median_residual=("oof_log_residual", "median"),
            group_mean_residual=("oof_log_residual", "mean"),
            group_std_residual=("oof_log_residual", "std"),
        )
        .reset_index()
    )

    df = df.merge(group_stats, on=group_col, how="left")

    rename_map = {
        "group_size": f"{prefix}_size",
        "group_median_price": f"{prefix}_median_price",
        "group_mean_price": f"{prefix}_mean_price",
        "group_median_residual": f"{prefix}_median_residual",
        "group_mean_residual": f"{prefix}_mean_residual",
        "group_std_residual": f"{prefix}_std_residual",
    }

    df = df.rename(columns=rename_map)

    std_col = f"{prefix}_std_residual"
    mean_col = f"{prefix}_mean_residual"
    size_col = f"{prefix}_size"

    df[f"{prefix}_residual_z"] = (
        (df["oof_log_residual"] - df[mean_col]) / df[std_col]
    )

    too_small_or_bad = (
        (df[size_col] < min_group_size)
        | df[f"{prefix}_residual_z"].isna()
        | np.isinf(df[f"{prefix}_residual_z"])
    )

    df.loc[too_small_or_bad, f"{prefix}_residual_z"] = df.loc[
        too_small_or_bad, "global_residual_z"
    ]

    df[f"{prefix}_used_global_fallback"] = too_small_or_bad.astype(int)

    df[f"{prefix}_residual_percentile"] = (
        df.groupby(group_col, dropna=False)["oof_log_residual"]
        .rank(method="average", pct=True)
        * 100
    )

    df.loc[too_small_or_bad, f"{prefix}_residual_percentile"] = df.loc[
        too_small_or_bad, "global_residual_percentile"
    ]

    return df, group_stats


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


def main():
    input_path = PROCESSED_DIR / "model_predictions.csv"
    require_file(input_path)

    df = pd.read_csv(input_path, low_memory=False)
    df["listing_id"] = df["listing_id"].astype(str)

    required_cols = [
        "listing_id",
        "actual_price",
        "oof_predicted_price",
        "oof_log_residual",
        "oof_percent_above_expected",
        "simple_peer_group",
        "capacity_peer_group",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns from model_predictions.csv: {missing}")

    df = df.copy()

    df["global_residual_percentile"] = percentile_rank(df["oof_log_residual"])
    df["global_percent_above_expected_percentile"] = percentile_rank(
        df["oof_percent_above_expected"]
    )
    df["global_residual_z"] = robust_z_score(df["oof_log_residual"])

    df, simple_group_stats = add_peer_residual_features(
        df,
        group_col="simple_peer_group",
        prefix="simple_peer",
        min_group_size=20,
    )

    df, capacity_group_stats = add_peer_residual_features(
        df,
        group_col="capacity_peer_group",
        prefix="capacity_peer",
        min_group_size=20,
    )

    df["residual_signal_score"] = (
        0.40 * df["global_residual_percentile"]
        + 0.30 * df["simple_peer_residual_percentile"]
        + 0.30 * df["capacity_peer_residual_percentile"]
    )

    df["over_expected_flag"] = (df["oof_log_residual"] > 0).astype(int)
    df["high_residual_flag"] = (df["global_residual_percentile"] >= 90).astype(int)
    df["high_simple_peer_flag"] = (df["simple_peer_residual_percentile"] >= 90).astype(int)
    df["high_capacity_peer_flag"] = (df["capacity_peer_residual_percentile"] >= 90).astype(int)

    df["peer_residual_label"] = "not flagged"
    df.loc[
        (df["high_residual_flag"] == 1)
        & (df["high_simple_peer_flag"] == 1),
        "peer_residual_label",
    ] = "high residual and high within simple peer group"

    df.loc[
        (df["high_residual_flag"] == 1)
        & (df["high_capacity_peer_flag"] == 1),
        "peer_residual_label",
    ] = "high residual and high within capacity peer group"

    df.loc[
        (df["high_residual_flag"] == 1)
        & (df["high_simple_peer_flag"] == 1)
        & (df["high_capacity_peer_flag"] == 1),
        "peer_residual_label",
    ] = "high residual across both peer views"

    output_path = PROCESSED_DIR / "peer_residual_scores.csv"
    df.to_csv(output_path, index=False)

    simple_group_stats.to_csv(TABLES_DIR / "simple_peer_group_summary.csv", index=False)
    capacity_group_stats.to_csv(TABLES_DIR / "capacity_peer_group_summary.csv", index=False)

    scoring_summary = pd.DataFrame(
        {
            "metric": [
                "rows_scored",
                "simple_peer_groups",
                "capacity_peer_groups",
                "simple_peer_fallback_count",
                "capacity_peer_fallback_count",
                "listings_global_top_10_pct",
                "listings_simple_peer_top_10_pct",
                "listings_capacity_peer_top_10_pct",
                "median_residual_signal_score",
                "mean_residual_signal_score",
            ],
            "value": [
                len(df),
                df["simple_peer_group"].nunique(),
                df["capacity_peer_group"].nunique(),
                int(df["simple_peer_used_global_fallback"].sum()),
                int(df["capacity_peer_used_global_fallback"].sum()),
                int((df["global_residual_percentile"] >= 90).sum()),
                int((df["simple_peer_residual_percentile"] >= 90).sum()),
                int((df["capacity_peer_residual_percentile"] >= 90).sum()),
                round(df["residual_signal_score"].median(), 2),
                round(df["residual_signal_score"].mean(), 2),
            ],
        }
    )

    scoring_summary.to_csv(TABLES_DIR / "peer_residual_scoring_summary.csv", index=False)

    top_cols = [
        "listing_id",
        "name",
        "room_type_model",
        "neighbourhood_model",
        "capacity_bucket",
        "actual_price",
        "oof_predicted_price",
        "oof_percent_above_expected",
        "oof_log_residual",
        "global_residual_percentile",
        "simple_peer_residual_percentile",
        "capacity_peer_residual_percentile",
        "residual_signal_score",
        "availability_rate",
        "reviews_last_365_days",
        "peer_residual_label",
    ]

    top_cols = [col for col in top_cols if col in df.columns]

    top_residual = (
        df.sort_values("residual_signal_score", ascending=False)
        .head(50)[top_cols]
        .copy()
    )

    top_residual.to_csv(TABLES_DIR / "top_residual_candidates.csv", index=False)

    save_histogram(
        df["residual_signal_score"],
        "Residual Signal Score Distribution",
        "Residual signal score",
        FIGURES_DIR / "residual_signal_score_distribution.png",
    )

    save_histogram(
        df["simple_peer_residual_z"],
        "Simple Peer Residual Z-Score Distribution",
        "Peer residual z-score",
        FIGURES_DIR / "simple_peer_residual_z_distribution.png",
    )

    save_scatter(
        df,
        "oof_percent_above_expected",
        "availability_rate",
        "Percent Above Expected vs. Availability Rate",
        "Percent above expected price",
        "Availability rate",
        FIGURES_DIR / "percent_above_expected_vs_availability.png",
    )

    print(f"Saved peer residual scores to: {output_path}")

    print("\nPeer residual scoring summary:")
    print(scoring_summary.to_string(index=False))

    print("\nTop residual candidates:")
    print(top_residual.head(15).to_string(index=False))


if __name__ == "__main__":
    main()