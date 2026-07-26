from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR_CANDIDATES = [
    ROOT / "raw-data-files",
    ROOT / "data" / "raw-data-files",
]

RAW_DIR = next((p for p in RAW_DIR_CANDIDATES if p.exists()), None)
if RAW_DIR is None:
    raise FileNotFoundError(
        "Could not find raw-data-files folder. Expected either "
        "'raw-data-files/' or 'data/raw-data-files/' under the project root."
    )

PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "tables"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def parse_money(value):
    if pd.isna(value):
        return np.nan

    value = str(value).replace("$", "").replace(",", "").strip()

    try:
        return float(value)
    except ValueError:
        return np.nan


def find_calendar_file():
    candidates = [
        RAW_DIR / "calendar.csv.gz",
        RAW_DIR / "calendar.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find calendar.csv.gz or calendar.csv in raw-data-files."
    )


def safe_ratio(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def main():
    calendar_path = find_calendar_file()
    print(f"Loading calendar file: {calendar_path}")

    header = pd.read_csv(calendar_path, compression="infer", nrows=0)
    available_columns = set(header.columns)

    required_cols = {"listing_id", "date", "available"}
    missing_required = required_cols - available_columns
    if missing_required:
        raise ValueError(f"Calendar file is missing required columns: {missing_required}")

    optional_cols = ["price", "adjusted_price", "minimum_nights", "maximum_nights"]
    usecols = ["listing_id", "date", "available"] + [
        col for col in optional_cols if col in available_columns
    ]

    calendar = pd.read_csv(
        calendar_path,
        compression="infer",
        usecols=usecols,
        low_memory=False,
    )

    print(f"Loaded calendar data with {calendar.shape[0]:,} rows and {calendar.shape[1]:,} columns.")

    calendar["listing_id"] = calendar["listing_id"].astype(str)
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")

    before = len(calendar)
    calendar = calendar[calendar["date"].notna()].copy()
    removed_bad_dates = before - len(calendar)

    calendar["available_flag"] = (
        calendar["available"].astype(str).str.lower().str.strip() == "t"
    ).astype(int)

    if "price" in calendar.columns:
        calendar["calendar_price_num"] = calendar["price"].apply(parse_money)
    else:
        calendar["calendar_price_num"] = np.nan

    if "adjusted_price" in calendar.columns:
        calendar["adjusted_calendar_price_num"] = calendar["adjusted_price"].apply(parse_money)
        calendar["calendar_price_num"] = calendar["calendar_price_num"].fillna(
            calendar["adjusted_calendar_price_num"]
        )

    for col in ["minimum_nights", "maximum_nights"]:
        if col in calendar.columns:
            calendar[col + "_num"] = pd.to_numeric(calendar[col], errors="coerce")

    calendar_start_date = calendar["date"].min()
    calendar_end_date = calendar["date"].max()

    calendar["days_from_start"] = (calendar["date"] - calendar_start_date).dt.days
    calendar["is_weekend"] = calendar["date"].dt.dayofweek.isin([5, 6]).astype(int)

    print(f"Calendar date range: {calendar_start_date.date()} to {calendar_end_date.date()}")
    print(f"Removed rows with invalid dates: {removed_bad_dates:,}")

    grouped = calendar.groupby("listing_id")

    features = grouped.agg(
        calendar_days_total=("date", "count"),
        available_days=("available_flag", "sum"),
        mean_calendar_price=("calendar_price_num", "mean"),
        median_calendar_price=("calendar_price_num", "median"),
        std_calendar_price=("calendar_price_num", "std"),
        min_calendar_price=("calendar_price_num", "min"),
        max_calendar_price=("calendar_price_num", "max"),
    )

    features["unavailable_days"] = (
        features["calendar_days_total"] - features["available_days"]
    )

    features["availability_rate"] = safe_ratio(
        features["available_days"], features["calendar_days_total"]
    )

    features["unavailable_rate"] = safe_ratio(
        features["unavailable_days"], features["calendar_days_total"]
    )

    features["price_variation_ratio"] = safe_ratio(
        features["std_calendar_price"], features["mean_calendar_price"]
    )

    features["calendar_price_range"] = (
        features["max_calendar_price"] - features["min_calendar_price"]
    )

    price_quantiles = (
        calendar.groupby("listing_id")["calendar_price_num"]
        .quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: "calendar_price_q25", 0.75: "calendar_price_q75"})
    )

    features = features.join(price_quantiles)
    features["calendar_price_iqr"] = (
        features["calendar_price_q75"] - features["calendar_price_q25"]
    )

    weekend_median = (
        calendar[calendar["is_weekend"] == 1]
        .groupby("listing_id")["calendar_price_num"]
        .median()
        .rename("weekend_median_price")
    )

    weekday_median = (
        calendar[calendar["is_weekend"] == 0]
        .groupby("listing_id")["calendar_price_num"]
        .median()
        .rename("weekday_median_price")
    )

    features = features.join(weekend_median).join(weekday_median)

    features["weekend_price_premium"] = (
        features["weekend_median_price"] - features["weekday_median_price"]
    )

    features["weekend_price_premium_ratio"] = safe_ratio(
        features["weekend_price_premium"], features["weekday_median_price"]
    )

    for horizon in [30, 60, 90, 180, 365]:
        horizon_df = calendar[
            (calendar["days_from_start"] >= 0)
            & (calendar["days_from_start"] < horizon)
        ].copy()

        horizon_grouped = horizon_df.groupby("listing_id")

        horizon_features = horizon_grouped.agg(
            **{
                f"next_{horizon}_calendar_days": ("date", "count"),
                f"next_{horizon}_available_days": ("available_flag", "sum"),
                f"next_{horizon}_median_price": ("calendar_price_num", "median"),
                f"next_{horizon}_mean_price": ("calendar_price_num", "mean"),
            }
        )

        horizon_features[f"next_{horizon}_availability_rate"] = safe_ratio(
            horizon_features[f"next_{horizon}_available_days"],
            horizon_features[f"next_{horizon}_calendar_days"],
        )

        features = features.join(horizon_features)

    if "minimum_nights_num" in calendar.columns:
        min_nights_features = grouped.agg(
            calendar_minimum_nights_median=("minimum_nights_num", "median"),
            calendar_minimum_nights_mean=("minimum_nights_num", "mean"),
        )
        features = features.join(min_nights_features)

    if "maximum_nights_num" in calendar.columns:
        max_nights_features = grouped.agg(
            calendar_maximum_nights_median=("maximum_nights_num", "median"),
            calendar_maximum_nights_mean=("maximum_nights_num", "mean"),
        )
        features = features.join(max_nights_features)

    features = features.reset_index()

    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Standardize a few missing values caused by no price variation.
    for col in [
        "std_calendar_price",
        "price_variation_ratio",
        "calendar_price_range",
        "calendar_price_iqr",
        "weekend_price_premium",
        "weekend_price_premium_ratio",
    ]:
        if col in features.columns:
            features[col] = features[col].fillna(0)

    output_path = PROCESSED_DIR / "calendar_features.csv"
    features.to_csv(output_path, index=False)

    data_summary = pd.DataFrame(
        {
            "metric": [
                "source_file",
                "raw_rows_loaded",
                "calendar_feature_rows",
                "calendar_start_date",
                "calendar_end_date",
                "invalid_date_rows_removed",
                "unique_listing_ids_in_calendar",
                "mean_availability_rate",
                "median_availability_rate",
                "mean_calendar_price",
                "median_calendar_price",
            ],
            "value": [
                str(calendar_path),
                len(calendar),
                len(features),
                str(calendar_start_date.date()),
                str(calendar_end_date.date()),
                removed_bad_dates,
                calendar["listing_id"].nunique(),
                round(features["availability_rate"].mean(), 4),
                round(features["availability_rate"].median(), 4),
                round(features["mean_calendar_price"].mean(), 2),
                round(features["median_calendar_price"].median(), 2),
            ],
        }
    )

    data_summary.to_csv(TABLES_DIR / "calendar_data_summary.csv", index=False)

    missingness = (
        features.isna()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_fraction"})
    )
    missingness.to_csv(TABLES_DIR / "missingness_calendar_features.csv", index=False)

    print(f"Saved calendar features to: {output_path}")
    print(f"Calendar feature rows: {len(features):,}")
    print("\nCalendar feature preview:")
    preview_cols = [
        "listing_id",
        "calendar_days_total",
        "availability_rate",
        "next_30_availability_rate",
        "next_90_availability_rate",
        "median_calendar_price",
        "price_variation_ratio",
        "weekend_price_premium",
    ]
    preview_cols = [col for col in preview_cols if col in features.columns]
    print(features[preview_cols].head().to_string(index=False))

    print("\nCalendar data summary:")
    print(data_summary.to_string(index=False))


if __name__ == "__main__":
    main()