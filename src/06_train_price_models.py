from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures" / "model_eval"
MODELS_DIR = ROOT / "models"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_predictions(model_name, y_true_log, y_pred_log):
    y_true_price = np.exp(y_true_log)
    y_pred_price = np.exp(y_pred_log)

    return {
        "model": model_name,
        "mae_log": mean_absolute_error(y_true_log, y_pred_log),
        "rmse_log": rmse(y_true_log, y_pred_log),
        "r2_log": r2_score(y_true_log, y_pred_log),
        "mae_price": mean_absolute_error(y_true_price, y_pred_price),
        "rmse_price": rmse(y_true_price, y_pred_price),
        "r2_price": r2_score(y_true_price, y_pred_price),
        "mape_price_pct": safe_mape(y_true_price, y_pred_price),
    }


def group_median_prediction(train_df, test_df, group_cols, fallback_value):
    lookup = (
        train_df.groupby(group_cols, dropna=False)["log_price"]
        .median()
        .reset_index()
        .rename(columns={"log_price": "group_median_log_price"})
    )

    pred_df = test_df[group_cols].merge(lookup, on=group_cols, how="left")
    return pred_df["group_median_log_price"].fillna(fallback_value).to_numpy()


def add_simple_transforms(df):
    transforms = {
        "minimum_nights": "minimum_nights_log1p",
        "maximum_nights": "maximum_nights_log1p",
        "number_of_reviews": "number_of_reviews_log1p",
        "number_of_reviews_ltm": "number_of_reviews_ltm_log1p",
        "number_of_reviews_l30d": "number_of_reviews_l30d_log1p",
        "reviews_last_365_days": "reviews_last_365_days_log1p",
        "review_count_from_reviews_file": "review_count_from_reviews_file_log1p",
        "host_listings_count": "host_listings_count_log1p",
        "host_total_listings_count": "host_total_listings_count_log1p",
    }

    for old_col, new_col in transforms.items():
        if old_col in df.columns:
            values = pd.to_numeric(df[old_col], errors="coerce").clip(lower=0)
            df[new_col] = np.log1p(values)

    return df


def build_feature_lists(df):
    numeric_candidates = [
        "latitude",
        "longitude",
        "accommodates",
        "bedrooms",
        "beds",
        "bathrooms_num",
        "minimum_nights_log1p",
        "maximum_nights_log1p",
        "number_of_reviews_log1p",
        "number_of_reviews_ltm_log1p",
        "number_of_reviews_l30d_log1p",
        "review_scores_rating",
        "review_scores_accuracy",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value",
        "reviews_per_month",
        "reviews_last_365_days_log1p",
        "review_count_from_reviews_file_log1p",
        "days_since_last_review",
        "monthly_review_rate_lifetime",
        "recent_review_share_365",
        "availability_rate",
        "unavailable_rate",
        "next_30_availability_rate",
        "next_60_availability_rate",
        "next_90_availability_rate",
        "next_180_availability_rate",
        "next_365_availability_rate",
        "calendar_minimum_nights_median",
        "calendar_minimum_nights_mean",
        "calendar_maximum_nights_median",
        "calendar_maximum_nights_mean",
        "amenity_count",
        "has_wifi",
        "has_kitchen",
        "has_parking",
        "has_pool",
        "has_hot_tub",
        "has_washer",
        "has_dryer",
        "has_air_conditioning",
        "has_dedicated_workspace",
        "host_response_rate_num",
        "host_acceptance_rate_num",
        "host_is_superhost_num",
        "host_has_profile_pic_num",
        "host_identity_verified_num",
        "instant_bookable_num",
        "host_listings_count_log1p",
        "host_total_listings_count_log1p",
        "has_any_reviews",
        "has_review_features",
    ]

    categorical_candidates = [
        "room_type_model",
        "property_type",
        "neighbourhood_model",
        "capacity_bucket",
    ]

    numeric_features = [col for col in numeric_candidates if col in df.columns]
    categorical_features = [col for col in categorical_candidates if col in df.columns]

    return numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def plot_actual_vs_predicted(results_df, output_path):
    plot_df = results_df[results_df["split"] == "test"].copy()

    plt.figure(figsize=(7, 6))
    plt.scatter(plot_df["actual_price"], plot_df["predicted_price_test_model"], alpha=0.35, s=16)

    max_val = np.nanpercentile(
        np.concatenate(
            [
                plot_df["actual_price"].to_numpy(),
                plot_df["predicted_price_test_model"].to_numpy(),
            ]
        ),
        98,
    )

    plt.plot([0, max_val], [0, max_val])
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.title("Actual vs Predicted Nightly Price")
    plt.xlabel("Actual nightly price ($)")
    plt.ylabel("Predicted nightly price ($)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_residuals(results_df, output_path):
    plt.figure(figsize=(8, 5))
    plt.hist(results_df["oof_log_residual"].dropna(), bins=50)
    plt.title("Out-of-Fold Log Residual Distribution")
    plt.xlabel("Actual log(price) - predicted log(price)")
    plt.ylabel("Number of listings")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_feature_importance(importance_df, output_path):
    top = importance_df.head(15).copy()
    top = top.sort_values("importance_mean", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance_mean"])
    plt.title("Top Model Features by Permutation Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    input_path = PROCESSED_DIR / "modeling_table.csv"
    require_file(input_path)

    df = pd.read_csv(input_path, low_memory=False)
    df = add_simple_transforms(df)

    if "log_price" not in df.columns or "price_num" not in df.columns:
        raise ValueError("Expected log_price and price_num in modeling_table.csv")

    df = df[df["log_price"].notna() & df["price_num"].notna()].copy()
    df["listing_id"] = df["listing_id"].astype(str)

    numeric_features, categorical_features = build_feature_lists(df)

    print(f"Loaded modeling table: {df.shape}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    print("\nCategorical features used:")
    print(categorical_features)

    feature_cols = numeric_features + categorical_features
    X = df[feature_cols].copy()
    y = df["log_price"].copy()

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.20,
        random_state=42,
        stratify=df["room_type_model"] if "room_type_model" in df.columns else None,
    )

    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    global_median = y_train.median()

    results = []

    baseline_predictions = {
        "Overall median": np.full(len(y_test), global_median),
    }

    if "neighbourhood_model" in df.columns:
        baseline_predictions["Neighborhood median"] = group_median_prediction(
            train_df,
            test_df,
            ["neighbourhood_model"],
            global_median,
        )

    if "room_type_model" in df.columns:
        baseline_predictions["Room type median"] = group_median_prediction(
            train_df,
            test_df,
            ["room_type_model"],
            global_median,
        )

    if {"neighbourhood_model", "room_type_model"}.issubset(df.columns):
        baseline_predictions["Neighborhood + room type median"] = group_median_prediction(
            train_df,
            test_df,
            ["neighbourhood_model", "room_type_model"],
            global_median,
        )

    if {"neighbourhood_model", "room_type_model", "capacity_bucket"}.issubset(df.columns):
        baseline_predictions["Neighborhood + room type + capacity median"] = group_median_prediction(
            train_df,
            test_df,
            ["neighbourhood_model", "room_type_model", "capacity_bucket"],
            global_median,
        )

    for name, pred in baseline_predictions.items():
        row = evaluate_predictions(name, y_test, pred)
        row["model_type"] = "baseline"
        row["best_params"] = ""
        results.append(row)

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    model_specs = {
        "Linear regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge regression": GridSearchCV(
            estimator=Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", Ridge(random_state=42)),
                ]
            ),
            param_grid={"model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]},
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        ),
        "Lasso regression": GridSearchCV(
            estimator=Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", Lasso(max_iter=20000, random_state=42)),
                ]
            ),
            param_grid={"model__alpha": [0.0005, 0.001, 0.005, 0.01, 0.05]},
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        ),
        "Random forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient boosting": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=400,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        l2_regularization=0.01,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    trained_models = {}

    for name, estimator in model_specs.items():
        print(f"\nTraining {name}...")
        estimator.fit(X_train, y_train)

        if isinstance(estimator, GridSearchCV):
            fitted_model = estimator.best_estimator_
            best_params = estimator.best_params_
        else:
            fitted_model = estimator
            best_params = ""

        pred_log = fitted_model.predict(X_test)

        row = evaluate_predictions(name, y_test, pred_log)
        row["model_type"] = "trained_model"
        row["best_params"] = str(best_params)
        results.append(row)

        trained_models[name] = fitted_model

        print(
            f"{name}: RMSE log = {row['rmse_log']:.4f}, "
            f"MAE price = ${row['mae_price']:.2f}, "
            f"R2 log = {row['r2_log']:.4f}"
        )

    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values(["rmse_log", "mae_price"]).reset_index(drop=True)
    comparison.to_csv(TABLES_DIR / "model_comparison.csv", index=False)

    trained_comparison = comparison[comparison["model_type"] == "trained_model"].copy()
    best_model_name = trained_comparison.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    print("\nModel comparison:")
    print(comparison.to_string(index=False))

    print(f"\nSelected fair-price model: {best_model_name}")

    selected_summary = pd.DataFrame(
        {
            "metric": [
                "selected_model",
                "selection_metric",
                "numeric_features",
                "categorical_features",
                "train_rows",
                "test_rows",
            ],
            "value": [
                best_model_name,
                "lowest test RMSE on log(price) among trained models",
                len(numeric_features),
                len(categorical_features),
                len(X_train),
                len(X_test),
            ],
        }
    )

    selected_summary.to_csv(TABLES_DIR / "selected_model_summary.csv", index=False)

    test_pred_log = best_model.predict(X_test)

    test_results = pd.DataFrame(
        {
            "listing_id": df.loc[test_idx, "listing_id"].to_numpy(),
            "split": "test",
            "actual_log_price": y_test.to_numpy(),
            "actual_price": np.exp(y_test.to_numpy()),
            "predicted_log_price_test_model": test_pred_log,
            "predicted_price_test_model": np.exp(test_pred_log),
        },
        index=test_idx,
    )

    train_results = pd.DataFrame(
        {
            "listing_id": df.loc[train_idx, "listing_id"].to_numpy(),
            "split": "train",
            "actual_log_price": y_train.to_numpy(),
            "actual_price": np.exp(y_train.to_numpy()),
            "predicted_log_price_test_model": np.nan,
            "predicted_price_test_model": np.nan,
        },
        index=train_idx,
    )

    split_results = pd.concat([train_results, test_results], axis=0).sort_index()

    print("\nBuilding out-of-fold predictions for residual scoring...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred_log = cross_val_predict(
        clone(best_model),
        X,
        y,
        cv=cv,
        n_jobs=-1,
    )

    final_model = clone(best_model)
    final_model.fit(X, y)

    full_pred_log = final_model.predict(X)

    predictions = df[
        [
            "listing_id",
            "name",
            "room_type_model",
            "property_type",
            "neighbourhood_model",
            "capacity_bucket",
            "simple_peer_group",
            "capacity_peer_group",
            "price_num",
            "log_price",
            "availability_rate",
            "next_90_availability_rate",
            "reviews_last_365_days",
            "has_review_features",
        ]
    ].copy()

    predictions["actual_price"] = predictions["price_num"]
    predictions["actual_log_price"] = predictions["log_price"]
    predictions["oof_predicted_log_price"] = oof_pred_log
    predictions["oof_predicted_price"] = np.exp(oof_pred_log)
    predictions["oof_log_residual"] = predictions["actual_log_price"] - predictions["oof_predicted_log_price"]
    predictions["oof_price_ratio"] = predictions["actual_price"] / predictions["oof_predicted_price"]
    predictions["oof_percent_above_expected"] = (predictions["oof_price_ratio"] - 1) * 100

    predictions["full_model_predicted_log_price"] = full_pred_log
    predictions["full_model_predicted_price"] = np.exp(full_pred_log)
    predictions["full_model_log_residual"] = predictions["actual_log_price"] - predictions["full_model_predicted_log_price"]

    predictions = predictions.merge(
        split_results[
            [
                "listing_id",
                "split",
                "predicted_log_price_test_model",
                "predicted_price_test_model",
            ]
        ],
        on="listing_id",
        how="left",
    )

    predictions_path = PROCESSED_DIR / "model_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    print(f"\nSaved predictions to: {predictions_path}")

    joblib.dump(final_model, MODELS_DIR / "best_price_model.joblib")

    print("\nCalculating permutation importance...")
    importance = permutation_importance(
        final_model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    importance_df.to_csv(TABLES_DIR / "feature_importance_permutation.csv", index=False)

    plot_actual_vs_predicted(
        predictions,
        FIGURES_DIR / "actual_vs_predicted_price.png",
    )

    plot_residuals(
        predictions,
        FIGURES_DIR / "oof_log_residual_distribution.png",
    )

    plot_feature_importance(
        importance_df,
        FIGURES_DIR / "top_feature_importance.png",
    )

    print("\nTop feature importance:")
    print(importance_df.head(15).to_string(index=False))

    print("\nDone. Main outputs:")
    print(TABLES_DIR / "model_comparison.csv")
    print(TABLES_DIR / "selected_model_summary.csv")
    print(TABLES_DIR / "feature_importance_permutation.csv")
    print(PROCESSED_DIR / "model_predictions.csv")
    print(MODELS_DIR / "best_price_model.joblib")


if __name__ == "__main__":
    main()