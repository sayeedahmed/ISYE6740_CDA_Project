from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures" / "clustering"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def percentile_rank(series):
    return series.rank(method="average", pct=True) * 100


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

    new_cols = {}

    for old_col, new_col in transforms.items():
        if old_col in df.columns:
            values = pd.to_numeric(df[old_col], errors="coerce").clip(lower=0)
            new_cols[new_col] = np.log1p(values)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df.copy()


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


def choose_kmeans_model(pca_features, k_values):
    rows = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(pca_features)

        score = silhouette_score(pca_features, labels)

        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": score,
            }
        )

    selection = pd.DataFrame(rows)

    # A simple rule: choose the best silhouette score, but do not go too small.
    selected_k = int(selection.sort_values("silhouette_score", ascending=False).iloc[0]["k"])

    final_model = KMeans(n_clusters=selected_k, random_state=42, n_init=30)
    final_labels = final_model.fit_predict(pca_features)

    return final_model, final_labels, selection, selected_k


def cluster_distance_to_center(pca_features, labels, centers):
    distances = np.zeros(len(labels))

    for cluster_id in np.unique(labels):
        idx = labels == cluster_id
        center = centers[cluster_id]
        distances[idx] = np.linalg.norm(pca_features[idx] - center, axis=1)

    return distances


def save_bar_chart(df, x_col, y_col, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(8, 5))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_scatter(df, x_col, y_col, color_col, title, xlabel, ylabel, output_path):
    plot_df = df[[x_col, y_col, color_col]].dropna().copy()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        plot_df[x_col],
        plot_df[y_col],
        c=plot_df[color_col],
        alpha=0.45,
        s=14,
    )
    plt.colorbar(scatter, label=color_col)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_histogram(series, title, xlabel, output_path, bins=50):
    plt.figure(figsize=(8, 5))
    plt.hist(series.dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of listings")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    modeling_path = PROCESSED_DIR / "modeling_table.csv"
    residual_path = PROCESSED_DIR / "peer_residual_scores.csv"

    require_file(modeling_path)
    require_file(residual_path)

    modeling = pd.read_csv(modeling_path, low_memory=False)
    residuals = pd.read_csv(residual_path, low_memory=False)

    modeling["listing_id"] = modeling["listing_id"].astype(str)
    residuals["listing_id"] = residuals["listing_id"].astype(str)

    modeling = add_simple_transforms(modeling)

    numeric_features, categorical_features = build_feature_lists(modeling)
    feature_cols = numeric_features + categorical_features

    print(f"Loaded modeling table: {modeling.shape}")
    print(f"Loaded residual scores: {residuals.shape}")
    print(f"Numeric features used for clustering: {len(numeric_features)}")
    print(f"Categorical features used for clustering: {len(categorical_features)}")

    X = modeling[feature_cols].copy()

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_processed = preprocessor.fit_transform(X)

    max_components = min(20, X_processed.shape[1])
    pca_full = PCA(n_components=max_components, random_state=42)
    X_pca_full = pca_full.fit_transform(X_processed)

    explained = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(max_components)],
            "explained_variance_ratio": pca_full.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca_full.explained_variance_ratio_),
        }
    )

    explained.to_csv(TABLES_DIR / "pca_explained_variance.csv", index=False)

    # Keep enough components to capture structure without making clustering too noisy.
    n_components = int(
        np.searchsorted(explained["cumulative_explained_variance"].to_numpy(), 0.80) + 1
    )
    n_components = max(5, min(n_components, 15, max_components))

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_processed)

    k_values = [4, 5, 6, 7, 8, 9, 10, 12]
    kmeans, labels, k_selection, selected_k = choose_kmeans_model(X_pca, k_values)

    k_selection.to_csv(TABLES_DIR / "kmeans_selection.csv", index=False)

    distances = cluster_distance_to_center(X_pca, labels, kmeans.cluster_centers_)

    cluster_df = modeling[
        [
            "listing_id",
            "name",
            "price_num",
            "log_price",
            "room_type_model",
            "property_type",
            "neighbourhood_model",
            "capacity_bucket",
            "availability_rate",
            "reviews_last_365_days",
        ]
    ].copy()

    cluster_df["cluster"] = labels
    cluster_df["cluster_distance"] = distances
    cluster_df["cluster_distance_percentile"] = percentile_rank(cluster_df["cluster_distance"])

    for i in range(n_components):
        cluster_df[f"pca_{i + 1}"] = X_pca[:, i]

    gmm = GaussianMixture(
        n_components=selected_k,
        covariance_type="full",
        random_state=42,
        reg_covar=1e-4,
    )

    gmm.fit(X_pca)
    log_density = gmm.score_samples(X_pca)

    cluster_df["gmm_log_density"] = log_density
    cluster_df["gmm_anomaly_score"] = -cluster_df["gmm_log_density"]
    cluster_df["gmm_anomaly_percentile"] = percentile_rank(cluster_df["gmm_anomaly_score"])

    cluster_df["cluster_anomaly_score"] = (
        0.50 * cluster_df["cluster_distance_percentile"]
        + 0.50 * cluster_df["gmm_anomaly_percentile"]
    )

    residual_keep_cols = [
        "listing_id",
        "actual_price",
        "oof_predicted_price",
        "oof_percent_above_expected",
        "oof_log_residual",
        "global_residual_percentile",
        "simple_peer_residual_percentile",
        "capacity_peer_residual_percentile",
        "residual_signal_score",
        "peer_residual_label",
    ]

    residual_keep_cols = [col for col in residual_keep_cols if col in residuals.columns]

    cluster_df = cluster_df.merge(
        residuals[residual_keep_cols],
        on="listing_id",
        how="left",
        validate="one_to_one",
    )

    cluster_df["cluster_adjusted_signal_score"] = (
        0.70 * cluster_df["residual_signal_score"]
        + 0.30 * cluster_df["cluster_anomaly_score"]
    )

    output_path = PROCESSED_DIR / "cluster_density_scores.csv"
    cluster_df.to_csv(output_path, index=False)

    cluster_summary = (
        cluster_df.groupby("cluster")
        .agg(
            listings=("listing_id", "count"),
            median_price=("price_num", "median"),
            mean_price=("price_num", "mean"),
            median_availability_rate=("availability_rate", "median"),
            median_reviews_last_365_days=("reviews_last_365_days", "median"),
            median_residual_signal_score=("residual_signal_score", "median"),
            median_cluster_anomaly_score=("cluster_anomaly_score", "median"),
            median_cluster_adjusted_signal_score=("cluster_adjusted_signal_score", "median"),
        )
        .reset_index()
        .sort_values("cluster")
    )

    cluster_summary.to_csv(TABLES_DIR / "cluster_summary.csv", index=False)

    run_summary = pd.DataFrame(
        {
            "metric": [
                "rows_clustered",
                "input_features",
                "numeric_features",
                "categorical_features",
                "pca_components_used",
                "pca_cumulative_variance_used",
                "selected_kmeans_k",
                "selected_kmeans_silhouette",
                "median_cluster_anomaly_score",
                "mean_cluster_anomaly_score",
            ],
            "value": [
                len(cluster_df),
                len(feature_cols),
                len(numeric_features),
                len(categorical_features),
                n_components,
                round(float(np.sum(pca.explained_variance_ratio_)), 4),
                selected_k,
                round(float(k_selection.loc[k_selection["k"] == selected_k, "silhouette_score"].iloc[0]), 4),
                round(float(cluster_df["cluster_anomaly_score"].median()), 2),
                round(float(cluster_df["cluster_anomaly_score"].mean()), 2),
            ],
        }
    )

    run_summary.to_csv(TABLES_DIR / "clustering_run_summary.csv", index=False)

    top_cols = [
        "listing_id",
        "name",
        "room_type_model",
        "neighbourhood_model",
        "capacity_bucket",
        "actual_price",
        "oof_predicted_price",
        "oof_percent_above_expected",
        "residual_signal_score",
        "cluster",
        "cluster_distance_percentile",
        "gmm_anomaly_percentile",
        "cluster_anomaly_score",
        "cluster_adjusted_signal_score",
        "availability_rate",
        "reviews_last_365_days",
    ]

    top_cols = [col for col in top_cols if col in cluster_df.columns]

    top_cluster_adjusted = (
        cluster_df.sort_values("cluster_adjusted_signal_score", ascending=False)
        .head(50)[top_cols]
        .copy()
    )

    top_cluster_adjusted.to_csv(TABLES_DIR / "top_cluster_adjusted_candidates.csv", index=False)

    save_bar_chart(
        k_selection,
        "k",
        "silhouette_score",
        "K-Means Silhouette Score by k",
        "Number of clusters",
        "Silhouette score",
        FIGURES_DIR / "kmeans_silhouette_by_k.png",
    )

    save_bar_chart(
        explained.head(15),
        "component",
        "explained_variance_ratio",
        "PCA Explained Variance",
        "Principal component",
        "Explained variance ratio",
        FIGURES_DIR / "pca_explained_variance.png",
    )

    save_scatter(
        cluster_df,
        "pca_1",
        "pca_2",
        "cluster",
        "Listings in PCA Space by Cluster",
        "PC1",
        "PC2",
        FIGURES_DIR / "pca_cluster_scatter.png",
    )

    save_scatter(
        cluster_df,
        "pca_1",
        "pca_2",
        "cluster_adjusted_signal_score",
        "Cluster-Adjusted Overpricing Signal in PCA Space",
        "PC1",
        "PC2",
        FIGURES_DIR / "pca_cluster_adjusted_signal.png",
    )

    save_histogram(
        cluster_df["cluster_anomaly_score"],
        "Cluster Anomaly Score Distribution",
        "Cluster anomaly score",
        FIGURES_DIR / "cluster_anomaly_score_distribution.png",
    )

    print(f"\nSaved cluster and density scores to: {output_path}")

    print("\nClustering run summary:")
    print(run_summary.to_string(index=False))

    print("\nK selection:")
    print(k_selection.to_string(index=False))

    print("\nCluster summary:")
    print(cluster_summary.to_string(index=False))

    print("\nTop cluster-adjusted candidates:")
    print(top_cluster_adjusted.head(15).to_string(index=False))


if __name__ == "__main__":
    main()