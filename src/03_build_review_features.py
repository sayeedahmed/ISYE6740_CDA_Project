from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw-data-files"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "tables"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def find_reviews_file():
    candidates = [
        RAW_DIR / "reviews.csv.gz",
        RAW_DIR / "reviews.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find reviews.csv.gz or reviews.csv in raw-data-files."
    )


def safe_ratio(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def main():
    reviews_path = find_reviews_file()
    print(f"Loading reviews file: {reviews_path}")

    header = pd.read_csv(reviews_path, compression="infer", nrows=0)
    available_columns = set(header.columns)

    required_cols = {"listing_id", "date"}
    missing_required = required_cols - available_columns
    if missing_required:
        raise ValueError(f"Reviews file is missing required columns: {missing_required}")

    optional_cols = ["id", "reviewer_id", "reviewer_name", "comments"]
    usecols = ["listing_id", "date"] + [
        col for col in optional_cols if col in available_columns
    ]

    reviews = pd.read_csv(
        reviews_path,
        compression="infer",
        usecols=usecols,
        low_memory=False,
    )

    print(f"Loaded reviews data with {reviews.shape[0]:,} rows and {reviews.shape[1]:,} columns.")

    reviews["listing_id"] = reviews["listing_id"].astype(str)
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")

    before = len(reviews)
    reviews = reviews[reviews["date"].notna()].copy()
    invalid_date_rows_removed = before - len(reviews)

    if reviews.empty:
        raise ValueError("Reviews file has no valid review dates after parsing.")

    analysis_date = reviews["date"].max()
    earliest_review_date = reviews["date"].min()

    print(f"Review date range: {earliest_review_date.date()} to {analysis_date.date()}")
    print(f"Removed rows with invalid dates: {invalid_date_rows_removed:,}")

    reviews["days_before_analysis"] = (analysis_date - reviews["date"]).dt.days

    if "comments" in reviews.columns:
        reviews["has_comment_text"] = (
            reviews["comments"].notna()
            & (reviews["comments"].astype(str).str.strip() != "")
        ).astype(int)
    else:
        reviews["has_comment_text"] = np.nan

    grouped = reviews.groupby("listing_id")

    features = grouped.agg(
        review_count_from_reviews_file=("date", "count"),
        first_review_date=("date", "min"),
        last_review_date=("date", "max"),
    )

    features["review_span_days"] = (
        features["last_review_date"] - features["first_review_date"]
    ).dt.days

    features["days_since_last_review"] = (
        analysis_date - features["last_review_date"]
    ).dt.days

    features["review_span_months"] = features["review_span_days"] / 30.4375
    features["review_span_months_safe"] = features["review_span_months"].clip(lower=1.0)
    features["monthly_review_rate_lifetime"] = (
        features["review_count_from_reviews_file"] / features["review_span_months_safe"]
    )

    for horizon in [30, 60, 90, 180, 365]:
        recent_counts = (
            reviews[reviews["days_before_analysis"] <= horizon]
            .groupby("listing_id")
            .size()
            .rename(f"reviews_last_{horizon}_days")
        )

        features = features.join(recent_counts)
        features[f"reviews_last_{horizon}_days"] = (
            features[f"reviews_last_{horizon}_days"].fillna(0).astype(int)
        )
        features[f"has_review_last_{horizon}_days"] = (
            features[f"reviews_last_{horizon}_days"] > 0
        ).astype(int)

    features["recent_review_share_365"] = safe_ratio(
        features["reviews_last_365_days"],
        features["review_count_from_reviews_file"],
    )

    features["recent_review_share_180"] = safe_ratio(
        features["reviews_last_180_days"],
        features["review_count_from_reviews_file"],
    )

    if "comments" in reviews.columns:
        comment_features = grouped.agg(
            review_comments_available=("has_comment_text", "sum")
        )
        features = features.join(comment_features)
        features["comment_available_rate"] = safe_ratio(
            features["review_comments_available"],
            features["review_count_from_reviews_file"],
        )

    for col in ["first_review_date", "last_review_date"]:
        features[col] = features[col].dt.strftime("%Y-%m-%d")

    features = features.reset_index()

    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan)

    output_path = PROCESSED_DIR / "review_features.csv"
    features.to_csv(output_path, index=False)

    data_summary = pd.DataFrame(
        {
            "metric": [
                "source_file",
                "raw_rows_loaded",
                "valid_review_rows",
                "invalid_date_rows_removed",
                "review_feature_rows",
                "unique_listing_ids_in_reviews",
                "earliest_review_date",
                "latest_review_date_used_as_analysis_date",
                "mean_reviews_per_reviewed_listing",
                "median_reviews_per_reviewed_listing",
                "mean_days_since_last_review",
                "median_days_since_last_review",
            ],
            "value": [
                str(reviews_path),
                len(reviews) + invalid_date_rows_removed,
                len(reviews),
                invalid_date_rows_removed,
                len(features),
                reviews["listing_id"].nunique(),
                str(earliest_review_date.date()),
                str(analysis_date.date()),
                round(features["review_count_from_reviews_file"].mean(), 2),
                round(features["review_count_from_reviews_file"].median(), 2),
                round(features["days_since_last_review"].mean(), 2),
                round(features["days_since_last_review"].median(), 2),
            ],
        }
    )

    data_summary.to_csv(TABLES_DIR / "review_data_summary.csv", index=False)

    missingness = (
        features.isna()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_fraction"})
    )
    missingness.to_csv(TABLES_DIR / "missingness_review_features.csv", index=False)

    print(f"Saved review features to: {output_path}")
    print(f"Review feature rows: {len(features):,}")

    preview_cols = [
        "listing_id",
        "review_count_from_reviews_file",
        "first_review_date",
        "last_review_date",
        "days_since_last_review",
        "reviews_last_90_days",
        "reviews_last_365_days",
        "monthly_review_rate_lifetime",
        "recent_review_share_365",
    ]
    preview_cols = [col for col in preview_cols if col in features.columns]

    print("\nReview feature preview:")
    print(features[preview_cols].head().to_string(index=False))

    print("\nReview data summary:")
    print(data_summary.to_string(index=False))


if __name__ == "__main__":
    main()
