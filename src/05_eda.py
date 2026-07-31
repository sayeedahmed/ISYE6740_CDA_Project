from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures" / "eda"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def save_histogram(series, title, xlabel, output_path, bins=40):
    values = series.dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of listings")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_bar_chart(df, x_col, y_col, title, xlabel, ylabel, output_path, rotate=True):
    plot_df = df.copy()

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df[x_col].astype(str), plot_df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if rotate:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_scatter(df, x_col, y_col, title, xlabel, ylabel, output_path):
    plot_df = df[[x_col, y_col]].dropna()

    plt.figure(figsize=(8, 5))
    plt.scatter(plot_df[x_col], plot_df[y_col], alpha=0.35, s=14)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    input_path = PROCESSED_DIR / "modeling_table.csv"
    require_file(input_path)

    df = pd.read_csv(input_path, low_memory=False)

    print(f"Loaded modeling table: {df.shape}")

    if "price_num" not in df.columns or "log_price" not in df.columns:
        raise ValueError("Expected price_num and log_price in modeling_table.csv")

    summary = pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "unique_listings",
                "median_price",
                "mean_price",
                "min_price",
                "max_price",
                "median_availability_rate",
                "mean_availability_rate",
                "listings_with_reviews",
                "listings_without_reviews",
            ],
            "value": [
                len(df),
                df.shape[1],
                df["listing_id"].astype(str).nunique(),
                round(df["price_num"].median(), 2),
                round(df["price_num"].mean(), 2),
                round(df["price_num"].min(), 2),
                round(df["price_num"].max(), 2),
                round(df["availability_rate"].median(), 4) if "availability_rate" in df.columns else np.nan,
                round(df["availability_rate"].mean(), 4) if "availability_rate" in df.columns else np.nan,
                int(df["has_review_features"].sum()) if "has_review_features" in df.columns else np.nan,
                int((df["has_review_features"] == 0).sum()) if "has_review_features" in df.columns else np.nan,
            ],
        }
    )

    summary.to_csv(TABLES_DIR / "eda_overall_summary.csv", index=False)

    missingness = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_fraction"})
    )
    missingness.to_csv(TABLES_DIR / "eda_missingness.csv", index=False)

    if "room_type_model" in df.columns:
        room_summary = (
            df.groupby("room_type_model", dropna=False)
            .agg(
                listings=("listing_id", "count"),
                median_price=("price_num", "median"),
                mean_price=("price_num", "mean"),
                median_log_price=("log_price", "median"),
                median_availability_rate=("availability_rate", "median"),
                median_reviews_last_365_days=("reviews_last_365_days", "median"),
            )
            .reset_index()
            .sort_values("listings", ascending=False)
        )

        room_summary.to_csv(TABLES_DIR / "eda_room_type_summary.csv", index=False)

        save_bar_chart(
            room_summary,
            "room_type_model",
            "median_price",
            "Median Price by Room Type",
            "Room type",
            "Median nightly price ($)",
            FIGURES_DIR / "median_price_by_room_type.png",
        )

    if "neighbourhood_model" in df.columns:
        neighborhood_summary = (
            df.groupby("neighbourhood_model", dropna=False)
            .agg(
                listings=("listing_id", "count"),
                median_price=("price_num", "median"),
                mean_price=("price_num", "mean"),
                median_availability_rate=("availability_rate", "median"),
                median_reviews_last_365_days=("reviews_last_365_days", "median"),
            )
            .reset_index()
        )

        neighborhood_summary.to_csv(TABLES_DIR / "eda_neighborhood_summary.csv", index=False)

        top_neighborhoods_count = (
            neighborhood_summary.sort_values("listings", ascending=False)
            .head(15)
            .copy()
        )

        top_neighborhoods_price = (
            neighborhood_summary[neighborhood_summary["listings"] >= 30]
            .sort_values("median_price", ascending=False)
            .head(15)
            .copy()
        )

        top_neighborhoods_count.to_csv(TABLES_DIR / "eda_top_neighborhoods_by_count.csv", index=False)
        top_neighborhoods_price.to_csv(TABLES_DIR / "eda_top_neighborhoods_by_median_price.csv", index=False)

        save_bar_chart(
            top_neighborhoods_count,
            "neighbourhood_model",
            "listings",
            "Top Neighborhoods by Listing Count",
            "Neighborhood",
            "Number of listings",
            FIGURES_DIR / "top_neighborhoods_by_count.png",
        )

        save_bar_chart(
            top_neighborhoods_price,
            "neighbourhood_model",
            "median_price",
            "Top Neighborhoods by Median Price",
            "Neighborhood",
            "Median nightly price ($)",
            FIGURES_DIR / "top_neighborhoods_by_median_price.png",
        )

    numeric_cols = [
        "price_num",
        "log_price",
        "accommodates",
        "bedrooms",
        "beds",
        "bathrooms_num",
        "minimum_nights",
        "maximum_nights",
        "availability_rate",
        "next_30_availability_rate",
        "next_90_availability_rate",
        "reviews_last_365_days",
        "review_scores_rating",
        "review_scores_cleanliness",
        "review_scores_location",
        "review_scores_value",
        "reviews_per_month",
        "amenity_count",
    ]

    numeric_cols = [col for col in numeric_cols if col in df.columns]

    corr = df[numeric_cols].corr(numeric_only=True)
    corr.to_csv(TABLES_DIR / "eda_numeric_correlation.csv")

    price_corr = (
        corr["log_price"]
        .drop(labels=["log_price"], errors="ignore")
        .sort_values(key=lambda x: x.abs(), ascending=False)
        .reset_index()
        .rename(columns={"index": "feature", "log_price": "correlation_with_log_price"})
    )

    price_corr.to_csv(TABLES_DIR / "eda_log_price_correlations.csv", index=False)

    save_histogram(
        df["price_num"],
        "Nightly Price Distribution",
        "Nightly price ($)",
        FIGURES_DIR / "price_distribution.png",
        bins=50,
    )

    save_histogram(
        df["log_price"],
        "Log Nightly Price Distribution",
        "log(price)",
        FIGURES_DIR / "log_price_distribution.png",
        bins=50,
    )

    if "availability_rate" in df.columns:
        save_histogram(
            df["availability_rate"],
            "Future Availability Rate Distribution",
            "Availability rate",
            FIGURES_DIR / "availability_rate_distribution.png",
            bins=40,
        )

    if "reviews_last_365_days" in df.columns:
        save_histogram(
            df["reviews_last_365_days"],
            "Reviews in Last 365 Days",
            "Reviews last 365 days",
            FIGURES_DIR / "reviews_last_365_days_distribution.png",
            bins=50,
        )

    if "accommodates" in df.columns:
        save_scatter(
            df,
            "accommodates",
            "price_num",
            "Nightly Price vs. Guest Capacity",
            "Accommodates",
            "Nightly price ($)",
            FIGURES_DIR / "price_vs_accommodates.png",
        )

    if {"longitude", "latitude"}.issubset(df.columns):
        map_df = df[["longitude", "latitude", "price_num"]].dropna().copy()

        plt.figure(figsize=(8, 7))
        plt.scatter(map_df["longitude"], map_df["latitude"], alpha=0.35, s=10)
        plt.title("Austin Airbnb Listing Locations")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "listing_location_scatter.png", dpi=200)
        plt.close()

    print("\nSaved EDA tables to:")
    print(TABLES_DIR)

    print("\nSaved EDA figures to:")
    print(FIGURES_DIR)

    print("\nOverall summary:")
    print(summary.to_string(index=False))

    if "room_type_model" in df.columns:
        print("\nRoom type summary:")
        print(room_summary.to_string(index=False))

    print("\nTop log-price correlations:")
    print(price_corr.head(12).to_string(index=False))


if __name__ == "__main__":
    main()