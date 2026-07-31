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


def make_decile(series):
    pct_rank = series.rank(method="first", pct=True)
    return np.ceil(pct_rank * 10).astype(int).clip(1, 10)


def spearman_corr(df, x_col, y_col):
    temp = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()

    if len(temp) < 3:
        return np.nan

    return temp[x_col].rank().corr(temp[y_col].rank())


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


def save_bar_chart(df, x_col, y_col, title, xlabel, ylabel, output_path, rotate=False):
    plt.figure(figsize=(9, 5))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if rotate:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    scores_path = PROCESSED_DIR / "final_overpricing_scores.csv"
    model_comparison_path = TABLES_DIR / "model_comparison.csv"

    require_file(scores_path)
    require_file(model_comparison_path)

    df = pd.read_csv(scores_path, low_memory=False)
    model_comparison = pd.read_csv(model_comparison_path)

    print(f"Loaded final scores: {df.shape}")
    print(f"Loaded model comparison: {model_comparison.shape}")

    df["residual_score_decile"] = make_decile(df["residual_signal_score"])
    df["cluster_adjusted_score_decile"] = make_decile(df["cluster_adjusted_signal_score"])
    df["final_score_decile_check"] = make_decile(df["final_overpricing_score"])

    residual_decile_summary = (
        df.groupby("residual_score_decile")
        .agg(
            listings=("listing_id", "count"),
            median_actual_price=("actual_price", "median"),
            median_predicted_price=("oof_predicted_price", "median"),
            median_percent_above_expected=("oof_percent_above_expected", "median"),
            median_availability_rate=("availability_rate", "median"),
            median_reviews_last_365_days=("reviews_last_365_days", "median"),
            median_cluster_anomaly_score=("cluster_anomaly_score", "median"),
            median_final_overpricing_score=("final_overpricing_score", "median"),
        )
        .reset_index()
        .sort_values("residual_score_decile")
    )

    residual_decile_summary.to_csv(TABLES_DIR / "validation_residual_decile_summary.csv", index=False)

    cluster_adjusted_decile_summary = (
        df.groupby("cluster_adjusted_score_decile")
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
        .sort_values("cluster_adjusted_score_decile")
    )

    cluster_adjusted_decile_summary.to_csv(
        TABLES_DIR / "validation_cluster_adjusted_decile_summary.csv",
        index=False,
    )

    corr_rows = []

    pairs = [
        ("residual_signal_score", "availability_rate"),
        ("residual_signal_score", "reviews_last_365_days"),
        ("cluster_adjusted_signal_score", "availability_rate"),
        ("cluster_adjusted_signal_score", "reviews_last_365_days"),
        ("final_overpricing_score", "availability_rate"),
        ("final_overpricing_score", "reviews_last_365_days"),
        ("oof_percent_above_expected", "availability_rate"),
        ("oof_percent_above_expected", "reviews_last_365_days"),
    ]

    for x_col, y_col in pairs:
        if x_col in df.columns and y_col in df.columns:
            corr_rows.append(
                {
                    "x": x_col,
                    "y": y_col,
                    "spearman_correlation": spearman_corr(df, x_col, y_col),
                }
            )

    correlation_summary = pd.DataFrame(corr_rows)
    correlation_summary.to_csv(TABLES_DIR / "validation_spearman_correlations.csv", index=False)

    trained_models = model_comparison[model_comparison["model_type"] == "trained_model"].copy()
    baselines = model_comparison[model_comparison["model_type"] == "baseline"].copy()

    best_trained = trained_models.sort_values("rmse_log").iloc[0]
    best_baseline = baselines.sort_values("rmse_log").iloc[0]

    rmse_log_improvement_pct = (
        (best_baseline["rmse_log"] - best_trained["rmse_log"])
        / best_baseline["rmse_log"]
        * 100
    )

    mae_price_improvement_pct = (
        (best_baseline["mae_price"] - best_trained["mae_price"])
        / best_baseline["mae_price"]
        * 100
    )

    candidate_counts = df["candidate_label"].value_counts().to_dict()

    evaluation_summary = pd.DataFrame(
        {
            "metric": [
                "best_trained_model",
                "best_baseline_model",
                "best_trained_rmse_log",
                "best_baseline_rmse_log",
                "rmse_log_improvement_over_best_baseline_pct",
                "best_trained_mae_price",
                "best_baseline_mae_price",
                "mae_price_improvement_over_best_baseline_pct",
                "best_trained_r2_log",
                "strong_candidates",
                "moderate_candidates",
                "watchlist_candidates",
                "not_flagged",
                "median_final_score_top_decile",
                "median_availability_top_final_decile",
                "median_reviews_last_365_top_final_decile",
                "median_percent_above_expected_top_final_decile",
            ],
            "value": [
                best_trained["model"],
                best_baseline["model"],
                round(best_trained["rmse_log"], 4),
                round(best_baseline["rmse_log"], 4),
                round(rmse_log_improvement_pct, 2),
                round(best_trained["mae_price"], 2),
                round(best_baseline["mae_price"], 2),
                round(mae_price_improvement_pct, 2),
                round(best_trained["r2_log"], 4),
                int(candidate_counts.get("strong candidate", 0)),
                int(candidate_counts.get("moderate candidate", 0)),
                int(candidate_counts.get("watchlist", 0)),
                int(candidate_counts.get("not flagged", 0)),
                round(df[df["final_score_decile_check"] == 10]["final_overpricing_score"].median(), 2),
                round(df[df["final_score_decile_check"] == 10]["availability_rate"].median(), 4),
                round(df[df["final_score_decile_check"] == 10]["reviews_last_365_days"].median(), 2),
                round(df[df["final_score_decile_check"] == 10]["oof_percent_above_expected"].median(), 2),
            ],
        }
    )

    evaluation_summary.to_csv(TABLES_DIR / "validation_evaluation_summary.csv", index=False)

    top_overlap_summary = []

    cutoffs = [25, 50, 100, 250]

    for n in cutoffs:
        top_residual = set(
            df.sort_values("residual_signal_score", ascending=False)
            .head(n)["listing_id"]
            .astype(str)
        )

        top_cluster_adjusted = set(
            df.sort_values("cluster_adjusted_signal_score", ascending=False)
            .head(n)["listing_id"]
            .astype(str)
        )

        top_final = set(
            df.sort_values("final_overpricing_score", ascending=False)
            .head(n)["listing_id"]
            .astype(str)
        )

        top_overlap_summary.append(
            {
                "top_n": n,
                "overlap_final_with_residual": len(top_final & top_residual),
                "overlap_final_with_cluster_adjusted": len(top_final & top_cluster_adjusted),
                "overlap_residual_with_cluster_adjusted": len(top_residual & top_cluster_adjusted),
            }
        )

    top_overlap_summary = pd.DataFrame(top_overlap_summary)
    top_overlap_summary.to_csv(TABLES_DIR / "validation_top_rank_overlap.csv", index=False)

    save_scatter(
        df,
        "residual_signal_score",
        "availability_rate",
        "Residual Signal Score vs. Availability",
        "Residual signal score",
        "Availability rate",
        FIGURES_DIR / "validation_residual_score_vs_availability.png",
    )

    save_scatter(
        df,
        "residual_signal_score",
        "reviews_last_365_days",
        "Residual Signal Score vs. Recent Reviews",
        "Residual signal score",
        "Reviews last 365 days",
        FIGURES_DIR / "validation_residual_score_vs_reviews.png",
    )

    save_bar_chart(
        residual_decile_summary,
        "residual_score_decile",
        "median_availability_rate",
        "Median Availability by Residual Score Decile",
        "Residual score decile",
        "Median availability rate",
        FIGURES_DIR / "validation_residual_decile_availability.png",
    )

    save_bar_chart(
        residual_decile_summary,
        "residual_score_decile",
        "median_reviews_last_365_days",
        "Median Recent Reviews by Residual Score Decile",
        "Residual score decile",
        "Median reviews last 365 days",
        FIGURES_DIR / "validation_residual_decile_reviews.png",
    )

    print("\nEvaluation summary:")
    print(evaluation_summary.to_string(index=False))

    print("\nResidual decile summary:")
    print(residual_decile_summary.to_string(index=False))

    print("\nSpearman correlations:")
    print(correlation_summary.to_string(index=False))

    print("\nTop-rank overlap:")
    print(top_overlap_summary.to_string(index=False))


if __name__ == "__main__":
    main()