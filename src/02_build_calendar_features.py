from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw-data-files"
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

    has_calendar_price = "price" in available_columns or "adjusted_price" in available_columns

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
    invalid_date_rows_removed = before - len(calendar)

    calendar["available_flag"] = (
        calendar["available"].astype(str).str.lower().str.strip() == "t"
    ).astype(int)

    if has_calendar_price:
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

    print(f"Calendar date range: {calendar_start_date.date()} to {calendar_end_date.date()}")
    print(f"Removed rows with invalid dates: {invalid_date_rows_removed:,}")

    grouped = calendar.groupby("listing_id")

    features = grouped.agg(
        calendar_days_total=("date", "count"),
        available_days=("available_flag", "sum"),
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

    if has_calendar_price:
        price_features = grouped.agg(
            mean_calendar_price=("calendar_price_num", "mean"),
            median_calendar_price=("calendar_price_num", "median"),
            std_calendar_price=("calendar_price_num", "std"),
            min_calendar_price=("calendar_price_num", "min"),
            max_calendar_price=("calendar_price_num", "max"),
        )

        price_features["price_variation_ratio"] = safe_ratio(
            price_features["std_calendar_price"],
            price_features["mean_calendar_price"],
        )

        price_features["calendar_price_range"] = (
            price_features["max_calendar_price"] - price_features["min_calendar_price"]
        )

        features = features.join(price_features)

    features = features.reset_index()

    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan)

    output_path = PROCESSED_DIR / "calendar_features.csv"
    features.to_csv(output_path, index=False)

    summary_metrics = [
        "source_file",
        "raw_rows_loaded",
        "calendar_feature_rows",
        "calendar_start_date",
        "calendar_end_date",
        "invalid_date_rows_removed",
        "unique_listing_ids_in_calendar",
        "calendar_price_columns_available",
        "mean_availability_rate",
        "median_availability_rate",
    ]

    summary_values = [
        str(calendar_path),
        len(calendar),
        len(features),
        str(calendar_start_date.date()),
        str(calendar_end_date.date()),
        invalid_date_rows_removed,
        calendar["listing_id"].nunique(),
        has_calendar_price,
        round(features["availability_rate"].mean(), 4),
        round(features["availability_rate"].median(), 4),
    ]

    if has_calendar_price and "median_calendar_price" in features.columns:
        summary_metrics.extend(["mean_calendar_price", "median_calendar_price"])
        summary_values.extend(
            [
                round(features["mean_calendar_price"].mean(), 2),
                round(features["median_calendar_price"].median(), 2),
            ]
        )

    data_summary = pd.DataFrame(
        {
            "metric": summary_metrics,
            "value": summary_values,
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

    preview_cols = [
        "listing_id",
        "calendar_days_total",
        "availability_rate",
        "next_30_availability_rate",
        "next_90_availability_rate",
        "next_180_availability_rate",
        "next_365_availability_rate",
        "calendar_minimum_nights_median",
        "calendar_maximum_nights_median",
    ]

    if has_calendar_price:
        preview_cols.extend(
            [
                "median_calendar_price",
                "price_variation_ratio",
                "calendar_price_range",
            ]
        )

    preview_cols = [col for col in preview_cols if col in features.columns]

    print("\nCalendar feature preview:")
    print(features[preview_cols].head().to_string(index=False))

    print("\nCalendar data summary:")
    print(data_summary.to_string(index=False))

    if not has_calendar_price:
        print(
            "\nNote: This calendar file does not include price or adjusted_price columns. "
            "Calendar features were limited to availability and stay-rule variables."
        )


if __name__ == "__main__":
    main()
