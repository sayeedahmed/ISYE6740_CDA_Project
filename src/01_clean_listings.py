from pathlib import Path
import ast
import re
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


def parse_percent(value):
    if pd.isna(value):
        return np.nan
    value = str(value).replace("%", "").strip()
    try:
        return float(value) / 100.0
    except ValueError:
        return np.nan


def parse_bool(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    if value in {"t", "true", "yes", "1"}:
        return 1
    if value in {"f", "false", "no", "0"}:
        return 0
    return np.nan


def parse_bathrooms_text(value):
    if pd.isna(value):
        return np.nan

    text = str(value).lower().strip()

    if "half" in text:
        return 0.5

    match = re.search(r"(\d+(\.\d+)?)", text)
    if match:
        return float(match.group(1))

    return np.nan


def parse_amenities(value):
    if pd.isna(value):
        return []

    text = str(value)

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).lower().strip() for x in parsed]
    except Exception:
        pass

    text = text.replace("[", "").replace("]", "").replace('"', "")
    return [x.lower().strip() for x in text.split(",") if x.strip()]


def has_amenity(amenities, keywords):
    joined = " | ".join(amenities)
    return int(any(keyword in joined for keyword in keywords))


def add_amenity_features(df):
    if "amenities" in df.columns:
        amenities_list = df["amenities"].apply(parse_amenities)
    else:
        amenities_list = pd.Series([[] for _ in range(len(df))], index=df.index)

    df["amenity_count"] = amenities_list.apply(len)
    df["has_wifi"] = amenities_list.apply(lambda x: has_amenity(x, ["wifi", "internet"]))
    df["has_kitchen"] = amenities_list.apply(lambda x: has_amenity(x, ["kitchen"]))
    df["has_parking"] = amenities_list.apply(lambda x: has_amenity(x, ["parking", "garage"]))
    df["has_pool"] = amenities_list.apply(lambda x: has_amenity(x, ["pool"]))
    df["has_hot_tub"] = amenities_list.apply(lambda x: has_amenity(x, ["hot tub", "jacuzzi"]))
    df["has_washer"] = amenities_list.apply(lambda x: has_amenity(x, ["washer"]))
    df["has_dryer"] = amenities_list.apply(lambda x: has_amenity(x, ["dryer"]))
    df["has_air_conditioning"] = amenities_list.apply(lambda x: has_amenity(x, ["air conditioning", "ac"]))
    df["has_dedicated_workspace"] = amenities_list.apply(lambda x: has_amenity(x, ["dedicated workspace", "workspace"]))

    return df


def add_cleaning_step(summary, step, before, after):
    summary.append(
        {
            "step": step,
            "rows_before": before,
            "rows_after": after,
            "rows_removed": before - after,
        }
    )


def main():
    input_candidates = [
        RAW_DIR / "listings.csv.gz",
        RAW_DIR / "listings.csv",
    ]

    input_path = next((p for p in input_candidates if p.exists()), None)

    if input_path is None:
        raise FileNotFoundError(
            "Could not find listings.csv.gz or listings.csv in raw-data-files."
        )

    df = pd.read_csv(input_path, compression="infer", low_memory=False)
    
    cleaning_summary = []

    original_rows = len(df)
    print(f"Loaded listings.csv with {original_rows:,} rows and {df.shape[1]:,} columns.")

    if "id" not in df.columns:
        raise ValueError("Expected listings.csv to contain an 'id' column.")

    df = df.rename(columns={"id": "listing_id"})
    df["listing_id"] = df["listing_id"].astype(str)

    before = len(df)
    df = df.drop_duplicates(subset=["listing_id"])
    add_cleaning_step(cleaning_summary, "drop duplicate listing_id records", before, len(df))

    if "price" not in df.columns:
        raise ValueError("Expected listings.csv to contain a 'price' column.")

    df["price_num"] = df["price"].apply(parse_money)

    before = len(df)
    df = df[df["price_num"].notna() & (df["price_num"] > 0)].copy()
    add_cleaning_step(cleaning_summary, "remove missing, zero, or negative prices", before, len(df))

    price_upper = df["price_num"].quantile(0.995)

    before = len(df)
    df = df[df["price_num"] <= price_upper].copy()
    add_cleaning_step(cleaning_summary, f"remove prices above 99.5th percentile (${price_upper:,.2f})", before, len(df))

    df["log_price"] = np.log(df["price_num"])

    numeric_cols = [
        "accommodates",
        "bedrooms",
        "beds",
        "minimum_nights",
        "maximum_nights",
        "number_of_reviews",
        "number_of_reviews_ltm",
        "number_of_reviews_l30d",
        "review_scores_rating",
        "review_scores_accuracy",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value",
        "reviews_per_month",
        "host_listings_count",
        "host_total_listings_count",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "bathrooms" in df.columns:
        df["bathrooms_num"] = pd.to_numeric(df["bathrooms"], errors="coerce")
    elif "bathrooms_text" in df.columns:
        df["bathrooms_num"] = df["bathrooms_text"].apply(parse_bathrooms_text)
    else:
        df["bathrooms_num"] = np.nan

    if "bathrooms_text" in df.columns:
        df["bathrooms_from_text"] = df["bathrooms_text"].apply(parse_bathrooms_text)
        df["bathrooms_num"] = df["bathrooms_num"].fillna(df["bathrooms_from_text"])

    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in df.columns:
            df[col + "_num"] = df[col].apply(parse_percent)

    for col in ["host_is_superhost", "host_has_profile_pic", "host_identity_verified", "instant_bookable"]:
        if col in df.columns:
            df[col + "_num"] = df[col].apply(parse_bool)

    if {"latitude", "longitude"}.issubset(df.columns):
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

        before = len(df)
        df = df[
            df["latitude"].between(29.0, 31.5)
            & df["longitude"].between(-99.5, -96.5)
        ].copy()
        add_cleaning_step(cleaning_summary, "remove invalid or out-of-region coordinates", before, len(df))

    if "accommodates" in df.columns:
        before = len(df)
        df = df[df["accommodates"].between(1, 16) | df["accommodates"].isna()].copy()
        add_cleaning_step(cleaning_summary, "remove implausible accommodates values", before, len(df))

    if "bedrooms" in df.columns:
        before = len(df)
        df = df[(df["bedrooms"] <= 10) | df["bedrooms"].isna()].copy()
        add_cleaning_step(cleaning_summary, "remove implausible bedroom values", before, len(df))

    if "beds" in df.columns:
        before = len(df)
        df = df[(df["beds"] <= 20) | df["beds"].isna()].copy()
        add_cleaning_step(cleaning_summary, "remove implausible bed values", before, len(df))

    before = len(df)
    df = df[(df["bathrooms_num"] <= 10) | df["bathrooms_num"].isna()].copy()
    add_cleaning_step(cleaning_summary, "remove implausible bathroom values", before, len(df))

    df = add_amenity_features(df)

    output_path = PROCESSED_DIR / "listings_clean.csv"
    df.to_csv(output_path, index=False)

    cleaning_summary_df = pd.DataFrame(cleaning_summary)
    cleaning_summary_df.to_csv(TABLES_DIR / "cleaning_summary_listings.csv", index=False)

    missingness = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_fraction"})
    )
    missingness.to_csv(TABLES_DIR / "missingness_listings.csv", index=False)

    data_summary = pd.DataFrame(
        {
            "metric": [
                "raw_rows",
                "clean_rows",
                "raw_columns",
                "clean_columns",
                "min_price",
                "median_price",
                "mean_price",
                "max_price",
                "price_upper_filter_99_5_pct",
            ],
            "value": [
                original_rows,
                len(df),
                "see raw file",
                df.shape[1],
                round(df["price_num"].min(), 2),
                round(df["price_num"].median(), 2),
                round(df["price_num"].mean(), 2),
                round(df["price_num"].max(), 2),
                round(price_upper, 2),
            ],
        }
    )
    data_summary.to_csv(TABLES_DIR / "data_summary_listings.csv", index=False)

    print(f"Saved cleaned listings to: {output_path}")
    print(f"Rows after cleaning: {len(df):,}")
    print("\nCleaning summary:")
    print(cleaning_summary_df.to_string(index=False))
    print("\nPrice summary after cleaning:")
    print(df["price_num"].describe())


if __name__ == "__main__":
    main()