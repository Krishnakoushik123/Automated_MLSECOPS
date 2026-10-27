"""
tests/unit/test_unsupervised.py
--------------------------------
Unit tests for src/training/unsupervised.py

Tests cover:
  - Input validation (valid, invalid, edge cases)
  - KMeans clustering (metrics, supervised comparison)
  - DBSCAN clustering (including single-cluster edge case)
  - IsolationForest anomaly detection (with and without ground truth)
  - PCA analysis (explained variance, n_components_for_95pct)
  - train_unsupervised() unified interface
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification

from src.training.unsupervised import (
    UNSUPERVISED_ALGORITHMS,
    get_unsupervised_task,
    train_clustering,
    train_anomaly_detection,
    run_pca_analysis,
    train_unsupervised,
    validate_unsupervised_inputs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blob_data():
    """3-cluster blob dataset, 300 samples, 10 features."""
    X, y = make_blobs(n_samples=300, centers=3, n_features=10, random_state=42)
    return X, y


@pytest.fixture
def anomaly_data():
    """Imbalanced binary classification dataset suitable for anomaly detection."""
    X, y = make_classification(n_samples=400, n_features=8, weights=[0.9, 0.1], random_state=42)
    return X, y


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

def test_get_unsupervised_task_known():
    assert get_unsupervised_task("kmeans") == "clustering"
    assert get_unsupervised_task("dbscan") == "clustering"
    assert get_unsupervised_task("isolation_forest") == "anomaly_detection"
    assert get_unsupervised_task("pca_analysis") == "dimensionality_reduction"


def test_get_unsupervised_task_unknown():
    assert get_unsupervised_task("random_forest") == "unknown"
    assert get_unsupervised_task("") == "unknown"


def test_algorithm_map_contains_expected():
    assert "kmeans" in UNSUPERVISED_ALGORITHMS
    assert "dbscan" in UNSUPERVISED_ALGORITHMS
    assert "isolation_forest" in UNSUPERVISED_ALGORITHMS
    assert "pca_analysis" in UNSUPERVISED_ALGORITHMS


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_validate_valid_kmeans(blob_data):
    X, _ = blob_data
    valid, errors = validate_unsupervised_inputs(X, "kmeans", {"n_clusters": 3})
    assert valid, f"Expected valid, got errors: {errors}"


def test_validate_kmeans_n_clusters_too_small(blob_data):
    X, _ = blob_data
    valid, errors = validate_unsupervised_inputs(X, "kmeans", {"n_clusters": 1})
    assert not valid
    assert any("n_clusters" in e for e in errors)


def test_validate_kmeans_n_clusters_too_large():
    X = np.random.randn(10, 4)
    valid, errors = validate_unsupervised_inputs(X, "kmeans", {"n_clusters": 15})
    assert not valid
    assert any("n_clusters" in e for e in errors)


def test_validate_unknown_algorithm(blob_data):
    X, _ = blob_data
    valid, errors = validate_unsupervised_inputs(X, "xgboost", {})
    assert not valid
    assert any("Unknown" in e for e in errors)


def test_validate_non_finite_values(blob_data):
    X, _ = blob_data
    X_bad = X.copy()
    X_bad[0, 0] = np.inf
    valid, errors = validate_unsupervised_inputs(X_bad, "kmeans", {"n_clusters": 3})
    assert not valid
    assert any("non-finite" in e for e in errors)


def test_validate_empty_array():
    X = np.zeros((0, 5))
    valid, errors = validate_unsupervised_inputs(X, "kmeans", {})
    assert not valid


def test_validate_1d_array():
    X = np.random.randn(100)
    valid, errors = validate_unsupervised_inputs(X, "kmeans", {})
    assert not valid
    assert any("2D" in e for e in errors)


def test_validate_pca_n_components_too_large(blob_data):
    X, _ = blob_data
    valid, errors = validate_unsupervised_inputs(X, "pca_analysis", {"n_components": 1000})
    assert not valid


# ---------------------------------------------------------------------------
# Clustering — KMeans
# ---------------------------------------------------------------------------

def test_kmeans_basic(blob_data):
    X, y = blob_data
    result = train_clustering(X, "kmeans", {"n_clusters": 3}, seed=42)
    assert result["algorithm"] == "kmeans"
    metrics = result["metrics"]
    assert metrics["n_clusters_found"] == 3
    assert "silhouette_score" in metrics
    assert metrics["silhouette_score"] is not None
    assert -1.0 <= metrics["silhouette_score"] <= 1.0


def test_kmeans_with_ground_truth(blob_data):
    X, y = blob_data
    result = train_clustering(X, "kmeans", {"n_clusters": 3}, y_true=y, seed=42)
    metrics = result["metrics"]
    assert "supervised_f1" in metrics
    assert 0.0 <= metrics["supervised_f1"] <= 1.0
    assert "supervised_accuracy" in metrics


def test_kmeans_labels_length(blob_data):
    X, y = blob_data
    result = train_clustering(X, "kmeans", {"n_clusters": 3}, seed=42)
    assert len(result["metrics"]["cluster_labels"]) == len(X)


def test_kmeans_davies_bouldin(blob_data):
    X, _ = blob_data
    result = train_clustering(X, "kmeans", {"n_clusters": 3}, seed=42)
    assert result["metrics"]["davies_bouldin_score"] is not None
    assert result["metrics"]["davies_bouldin_score"] >= 0.0


# ---------------------------------------------------------------------------
# Clustering — DBSCAN
# ---------------------------------------------------------------------------

def test_dbscan_basic(blob_data):
    X, _ = blob_data
    result = train_clustering(X, "dbscan", {"eps": 1.5, "min_samples": 5}, seed=42)
    assert result["algorithm"] == "dbscan"
    metrics = result["metrics"]
    assert "n_clusters_found" in metrics
    assert "n_noise_points" in metrics
    assert metrics["n_noise_points"] >= 0


def test_dbscan_insufficient_clusters_warning():
    """DBSCAN with very tight eps may produce <=1 cluster — check graceful handling."""
    X = np.random.randn(50, 4)
    result = train_clustering(X, "dbscan", {"eps": 0.001, "min_samples": 100}, seed=42)
    metrics = result["metrics"]
    # Either all noise or 1 cluster — metrics should degrade gracefully, not crash
    assert "n_clusters_found" in metrics
    if metrics["n_clusters_found"] < 2:
        assert metrics["silhouette_score"] is None
        assert "warning" in metrics


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------

def test_isolation_forest_basic(anomaly_data):
    X, y = anomaly_data
    result = train_anomaly_detection(X, "isolation_forest", {"contamination": 0.1}, seed=42)
    assert result["algorithm"] == "isolation_forest"
    metrics = result["metrics"]
    assert 0.0 <= metrics["anomaly_rate"] <= 1.0
    assert metrics["n_total_samples"] == len(X)
    assert metrics["n_anomalies_detected"] >= 0
    assert len(metrics["anomaly_predictions"]) == len(X)


def test_isolation_forest_with_binary_ground_truth(anomaly_data):
    X, y = anomaly_data
    result = train_anomaly_detection(X, "isolation_forest", {"contamination": 0.1}, y_true=y, seed=42)
    metrics = result["metrics"]
    assert "supervised_f1" in metrics
    assert 0.0 <= metrics["supervised_f1"] <= 1.0
    assert "supervised_roc_auc" in metrics


def test_isolation_forest_auto_contamination(anomaly_data):
    X, _ = anomaly_data
    result = train_anomaly_detection(X, "isolation_forest", {"contamination": "auto"}, seed=42)
    assert result["metrics"]["n_total_samples"] == len(X)


def test_isolation_forest_training_time_positive(anomaly_data):
    X, _ = anomaly_data
    result = train_anomaly_detection(X, "isolation_forest", {"contamination": 0.1}, seed=42)
    assert result["metrics"]["training_time_seconds"] > 0


# ---------------------------------------------------------------------------
# PCA Analysis
# ---------------------------------------------------------------------------

def test_pca_analysis_basic(blob_data):
    X, _ = blob_data
    result = run_pca_analysis(X, {"n_components": 5}, seed=42)
    assert result["algorithm"] == "pca_analysis"
    metrics = result["metrics"]
    assert metrics["n_components"] == 5
    assert len(metrics["explained_variance_ratio"]) == 5
    assert 0.0 < metrics["total_explained_variance"] <= 1.0


def test_pca_cumulative_variance(blob_data):
    X, _ = blob_data
    result = run_pca_analysis(X, {"n_components": X.shape[1]}, seed=42)
    metrics = result["metrics"]
    cevr = metrics["cumulative_explained_variance"]
    # Cumulative should be non-decreasing and end near 1.0
    for i in range(1, len(cevr)):
        assert cevr[i] >= cevr[i-1] - 1e-9
    assert cevr[-1] > 0.9


def test_pca_95pct_components(blob_data):
    X, _ = blob_data
    n_total = X.shape[1]
    result = run_pca_analysis(X, {"n_components": n_total}, seed=42)
    n95 = result["metrics"]["n_components_for_95pct_variance"]
    assert 1 <= n95 <= n_total


def test_pca_transformed_shape(blob_data):
    X, _ = blob_data
    result = run_pca_analysis(X, {"n_components": 3}, seed=42)
    Xt = result["X_transformed"]
    assert Xt.shape == (len(X), 3)


# ---------------------------------------------------------------------------
# Unified train_unsupervised interface
# ---------------------------------------------------------------------------

def test_train_unsupervised_kmeans(blob_data, tmp_path):
    X, _ = blob_data
    result = train_unsupervised("kmeans", X, hyperparams={"n_clusters": 3},
                                 models_dir=tmp_path, seed=42)
    assert result["algorithm"] == "kmeans"
    assert result["task_category"] == "clustering"
    assert result["model_path"] is not None
    assert "silhouette_score" in result["metrics"]


def test_train_unsupervised_isolation_forest(anomaly_data, tmp_path):
    X, y = anomaly_data
    result = train_unsupervised("isolation_forest", X, y_true=y,
                                 hyperparams={"contamination": 0.1},
                                 models_dir=tmp_path, seed=42)
    assert result["task_category"] == "anomaly_detection"
    assert "supervised_f1" in result["metrics"]


def test_train_unsupervised_pca(blob_data, tmp_path):
    X, _ = blob_data
    result = train_unsupervised("pca_analysis", X, hyperparams={"n_components": 4},
                                 models_dir=tmp_path, seed=42)
    assert result["task_category"] == "dimensionality_reduction"
    assert result["metrics"]["n_components"] == 4


def test_train_unsupervised_invalid_algorithm(blob_data):
    X, _ = blob_data
    with pytest.raises(ValueError, match="Unknown unsupervised algorithm"):
        train_unsupervised("random_forest", X, seed=42)


def test_train_unsupervised_no_model_dir(blob_data):
    """Should work without models_dir (no persistence)."""
    X, _ = blob_data
    result = train_unsupervised("kmeans", X, hyperparams={"n_clusters": 2}, seed=42)
    assert result["model_path"] is None
