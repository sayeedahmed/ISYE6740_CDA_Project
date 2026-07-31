from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "tables"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def clean_text_column(series):
    return (
        series.astype(str)
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )


def add_capacity_bucket(df):
    if "accommodates" not in df.columns:
        df["capacity_bucket"] = "unknown"
        return df

    df["capacity_bucket"] = pd.cut(
        df["accommodates"],
        bins=[0, 2, 4, 6, 16],
        labels=["1-2 guests", "3-4 guests", "5-6 guests", "7+ guests"],
        include_lowest=True,
    )

    df["capacity_bucket"] = df["capacity_bucket"].astype(str).replace("nan", "unknown")
    return df


def fill_review_features(df):
    review_count_cols = [
        "review_count_from_reviews_file",
        "reviews_last_30_days",
        "reviews_last_60_days",
        "reviews_last_90_days",
        "reviews_last_180_days",
        "reviews_last_365_days",
        "has_review_last_30_days",
        "has_review_last_60_days",
        "has_review_last_90_days",
        "has_review_last_180_days",
        "has_review_last_365_days",
    ]

    for col in review_count_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if "review_count_from_reviews_file" in df.columns:
        df["has_any_reviews"] = (df["review_count_from_reviews_file"] > 0).astype(int)
    else:
        df["has_any_reviews"] = 0

    if "days_since_last_review" in df.columns:
        max_observed = df["days_since_last_review"].max(skipna=True)
        if pd.isna(max_observed):
            max_observed = 9999
        df["days_since_last_review"] = df["days_since_last_review"].fillna(max_observed + 365)

    rate_cols = [
        "monthly_review_rate_lifetime",
        "recent_review_share_365",
        "recent_review_share_180",
        "comment_available_rate",
    ]

    for col in rate_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def fill_calendar_features(df):
    calendar_rate_cols = [
        "availability_rate",
        "unavailable_rate",
        "next_30_availability_rate",
        "next_60_availability_rate",
        "next_90_availability_rate",
        "next_180_availability_rate",
        "next_365_availability_rate",
    ]

    for col in calendar_rate_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    calendar_count_cols = [
        "calendar_days_total",
        "available_days",
        "unavailable_days",
        "next_30_calendar_days",
        "next_30_available_days",
        "next_60_calendar_days",
        "next_60_available_days",
        "next_90_calendar_days",
        "next_90_available_days",
        "next_180_calendar_days",
        "next_180_available_days",
        "next_365_calendar_days",
        "next_365_available_days",
    ]

    for col in calendar_count_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def main():
    listings_path = PROCESSED_DIR / "listings_clean.csv"
    calendar_path = PROCESSED_DIR / "calendar_features.csv"
    reviews_path = PROCESSED_DIR / "review_features.csv"

    for path in [listings_path, calendar_path, reviews_path]:
        require_file(path)

    listings = pd.read_csv(listings_path, low_memory=False)
    calendar = pd.read_csv(calendar_path, low_memory=False)
    reviews = pd.read_csv(reviews_path, low_memory=False)

    listings["listing_id"] = listings["listing_id"].astype(str)
    calendar["listing_id"] = calendar["listing_id"].astype(str)
    reviews["listing_id"] = reviews["listing_id"].astype(str)

    print(f"Loaded listings: {listings.shape}")
    print(f"Loaded calendar features: {calendar.shape}")
    print(f"Loaded review features: {reviews.shape}")

    base_rows = len(listings)

    df = listings.merge(
        calendar,
        on="listing_id",
        how="left",
        validate="one_to_one",
        indicator="calendar_merge_status",
    )

    df = df.merge(
        reviews,
        on="listing_id",
        how="left",
        validate="one_to_one",
        indicator="review_merge_status",
    )

    if len(df) != base_rows:
        raise ValueError(
            f"Row count changed after merges. Expected {base_rows}, got {len(df)}."
        )

    df["has_calendar_features"] = (df["calendar_merge_status"] == "both").astype(int)
    df["has_review_features"] = (df["review_merge_status"] == "both").astype(int)

    df = df.drop(columns=["calendar_merge_status", "review_merge_status"])

    df = fill_calendar_features(df)
    df = fill_review_features(df)
    df = add_capacity_bucket(df)

    text_cols = [
        "name",
        "room_type",
        "property_type",
        "neighbourhood_cleansed",
        "neighbourhood_group_cleansed",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = clean_text_column(df[col])

    if "neighbourhood_cleansed" in df.columns:
        df["neighbourhood_model"] = df["neighbourhood_cleansed"].fillna("Unknown")
    elif "neighbourhood" in df.columns:
        df["neighbourhood_model"] = df["neighbourhood"].fillna("Unknown")
    else:
        df["neighbourhood_model"] = "Unknown"

    if "room_type" in df.columns:
        df["room_type_model"] = df["room_type"].fillna("Unknown")
    else:
        df["room_type_model"] = "Unknown"

    df["simple_peer_group"] = (
        df["neighbourhood_model"].astype(str)
        + " | "
        + df["room_type_model"].astype(str)
    )

    df["capacity_peer_group"] = (
        df["neighbourhood_model"].astype(str)
        + " | "
        + df["room_type_model"].astype(str)
        + " | "
        + df["capacity_bucket"].astype(str)
    )

    if "price_num" not in df.columns or "log_price" not in df.columns:
        raise ValueError("Expected price_num and log_price from listings_clean.csv.")

    before = len(df)
    df = df[df["price_num"].notna() & df["log_price"].notna()].copy()
    removed_missing_target = before - len(df)

    output_path = PROCESSED_DIR / "modeling_table.csv"
    df.to_csv(output_path, index=False)

    merge_summary = pd.DataFrame(
        {
            "metric": [
                "listings_clean_rows",
                "calendar_feature_rows",
                "review_feature_rows",
                "modeling_table_rows",
                "listings_with_calendar_features",
                "listings_with_review_features",
                "listings_without_review_features",
                "rows_removed_missing_target",
                "modeling_table_columns",
            ],
            "value": [
                base_rows,
                len(calendar),
                len(reviews),
                len(df),
                int(df["has_calendar_features"].sum()),
                int(df["has_review_features"].sum()),
                int((df["has_review_features"] == 0).sum()),
                removed_missing_target,
                df.shape[1],
            ],
        }
    )

    merge_summary.to_csv(TABLES_DIR / "modeling_table_summary.csv", index=False)

    missingness = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_fraction"})
    )

    missingness.to_csv(TABLES_DIR / "missingness_modeling_table.csv", index=False)

    peer_summary = (
        df.groupby(["room_type_model", "capacity_bucket"], dropna=False)
        .agg(
            listings=("listing_id", "count"),
            median_price=("price_num", "median"),
            mean_price=("price_num", "mean"),
            median_availability_rate=("availability_rate", "median"),
            median_reviews_last_365_days=("reviews_last_365_days", "median"),
        )
        .reset_index()
        .sort_values(["room_type_model", "capacity_bucket"])
    )

    peer_summary.to_csv(TABLES_DIR / "modeling_table_peer_summary.csv", index=False)

    print(f"\nSaved modeling table to: {output_path}")
    print("\nModeling table summary:")
    print(merge_summary.to_string(index=False))

    preview_cols = [
        "listing_id",
        "name",
        "price_num",
        "log_price",
        "room_type_model",
        "neighbourhood_model",
        "capacity_bucket",
        "availability_rate",
        "reviews_last_365_days",
        "has_review_features",
        "simple_peer_group",
    ]

    preview_cols = [col for col in preview_cols if col in df.columns]

    print("\nPreview:")
    print(df[preview_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()