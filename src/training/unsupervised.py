"""
src/training/unsupervised.py
-----------------------------
Unsupervised and semi-supervised ML task support:
  - Clustering   : KMeans, DBSCAN
  - Anomaly Det. : IsolationForest
  - Dim. Reduction: PCA (standalone, for analysis)

All functions return a standardised result dict compatible with the existing
experiment_schema so they integrate seamlessly with the model registry, API,
and monitoring interfaces.

Run standalone:
    python -m src.training.unsupervised
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    confusion_matrix,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Algorithm registry for unsupervised tasks
# ---------------------------------------------------------------------------

UNSUPERVISED_ALGORITHMS: Dict[str, str] = {
    "kmeans": "clustering",
    "dbscan": "clustering",
    "isolation_forest": "anomaly_detection",
    "pca_analysis": "dimensionality_reduction",
}


def get_unsupervised_task(algorithm: str) -> str:
    return UNSUPERVISED_ALGORITHMS.get(algorithm, "unknown")


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def train_clustering(
    X: np.ndarray,
    algorithm: str = "kmeans",
    hyperparams: Optional[Dict[str, Any]] = None,
    y_true: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    if hyperparams is None:
        hyperparams = {}

    t_start = time.perf_counter()

    if algorithm == "kmeans":
        n_clusters = hyperparams.get("n_clusters", 2)
        model = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=hyperparams.get("n_init", 10),
            max_iter=hyperparams.get("max_iter", 300),
        )
        labels = model.fit_predict(X)
    elif algorithm == "dbscan":
        model = DBSCAN(
            eps=hyperparams.get("eps", 0.5),
            min_samples=hyperparams.get("min_samples", 5),
            n_jobs=-1,
        )
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm!r}")

    training_time = time.perf_counter() - t_start

    unique_labels = np.unique(labels[labels != -1])
    n_clusters_found = len(unique_labels)

    metrics: Dict[str, Any] = {
        "training_time_seconds": training_time,
        "n_clusters_found": n_clusters_found,
        "n_noise_points": int(np.sum(labels == -1)),
    }

    valid_mask = labels != -1
    if n_clusters_found >= 2 and np.sum(valid_mask) >= n_clusters_found:
        try:
            metrics["silhouette_score"] = float(silhouette_score(X[valid_mask], labels[valid_mask]))
        except Exception as e:
            logger.warning("Silhouette score failed: %s", e)
            metrics["silhouette_score"] = None
        try:
            metrics["davies_bouldin_score"] = float(davies_bouldin_score(X[valid_mask], labels[valid_mask]))
        except Exception as e:
            logger.warning("Davies-Bouldin score failed: %s", e)
            metrics["davies_bouldin_score"] = None
        try:
            metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X[valid_mask], labels[valid_mask]))
        except Exception as e:
            logger.warning("Calinski-Harabasz score failed: %s", e)
            metrics["calinski_harabasz_score"] = None
    else:
        metrics.update({
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "calinski_harabasz_score": None,
            "warning": "Insufficient clusters for metric computation. Adjust algorithm parameters.",
        })

    if y_true is not None:
        sup_metrics = _compute_supervised_from_cluster_labels(labels, y_true, n_clusters_found)
        metrics.update(sup_metrics)

    metrics["cluster_labels"] = labels.tolist()

    return {"algorithm": algorithm, "model": model, "metrics": metrics, "labels": labels}


def _compute_supervised_from_cluster_labels(
    cluster_labels: np.ndarray,
    y_true: np.ndarray,
    n_clusters_found: int,
) -> Dict[str, Any]:
    sup: Dict[str, Any] = {}
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

        valid_mask = cluster_labels != -1
        if np.sum(valid_mask) == 0 or n_clusters_found < 2:
            sup["supervised_metrics_note"] = "Not enough valid cluster assignments."
            return sup

        c_labels = cluster_labels[valid_mask]
        t_labels = y_true[valid_mask]
        unique_true = np.unique(t_labels)

        mapping: Dict[int, Any] = {}
        for cluster_id in np.unique(c_labels):
            mask_c = c_labels == cluster_id
            values, counts = np.unique(t_labels[mask_c], return_counts=True)
            mapping[cluster_id] = values[np.argmax(counts)]

        y_mapped = np.array([mapping[c] for c in c_labels])
        avg = "binary" if len(unique_true) <= 2 else "weighted"
        sup.update({
            "supervised_accuracy": float(accuracy_score(t_labels, y_mapped)),
            "supervised_f1": float(f1_score(t_labels, y_mapped, average=avg, zero_division=0)),
            "supervised_precision": float(precision_score(t_labels, y_mapped, average=avg, zero_division=0)),
            "supervised_recall": float(recall_score(t_labels, y_mapped, average=avg, zero_division=0)),
            "supervised_confusion_matrix": confusion_matrix(t_labels, y_mapped).tolist(),
            "supervised_metrics_note": "Cluster-to-class majority-vote mapping. Illustrative only.",
        })
    except Exception as e:
        logger.warning("Supervised metrics from clustering failed: %s", e)
        sup["supervised_metrics_note"] = f"Could not compute: {e}"
    return sup


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------

def train_anomaly_detection(
    X: np.ndarray,
    algorithm: str = "isolation_forest",
    hyperparams: Optional[Dict[str, Any]] = None,
    y_true: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    if hyperparams is None:
        hyperparams = {}

    t_start = time.perf_counter()

    if algorithm == "isolation_forest":
        contamination = hyperparams.get("contamination", "auto")
        if contamination != "auto":
            contamination = float(contamination)
        model = IsolationForest(
            n_estimators=hyperparams.get("n_estimators", 100),
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X)
        raw_preds = model.predict(X)
        binary_preds = np.where(raw_preds == -1, 1, 0)
        scores = model.decision_function(X)
    else:
        raise ValueError(f"Unknown anomaly detection algorithm: {algorithm!r}")

    training_time = time.perf_counter() - t_start

    n_anomalies = int(np.sum(binary_preds == 1))
    n_total = len(X)
    anomaly_rate = n_anomalies / n_total if n_total > 0 else 0.0

    metrics: Dict[str, Any] = {
        "training_time_seconds": training_time,
        "n_anomalies_detected": n_anomalies,
        "n_total_samples": n_total,
        "anomaly_rate": round(anomaly_rate, 4),
        "mean_anomaly_score": float(np.mean(scores)),
        "std_anomaly_score": float(np.std(scores)),
    }

    if y_true is not None:
        unique_true = np.unique(y_true)
        if len(unique_true) == 2:
            try:
                from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
                metrics.update({
                    "supervised_accuracy": float(accuracy_score(y_true, binary_preds)),
                    "supervised_f1": float(f1_score(y_true, binary_preds, zero_division=0)),
                    "supervised_precision": float(precision_score(y_true, binary_preds, zero_division=0)),
                    "supervised_recall": float(recall_score(y_true, binary_preds, zero_division=0)),
                    "supervised_roc_auc": float(roc_auc_score(y_true, -scores)),
                    "supervised_confusion_matrix": confusion_matrix(y_true, binary_preds).tolist(),
                    "supervised_metrics_note": "Ground-truth binary labels provided. Anomaly=1, Normal=0.",
                })
            except Exception as e:
                logger.warning("Supervised anomaly metrics failed: %s", e)
                metrics["supervised_metrics_note"] = f"Could not compute: {e}"
        else:
            metrics["supervised_metrics_note"] = "Labels not binary; supervised metrics skipped."

    metrics["anomaly_predictions"] = binary_preds.tolist()

    return {"algorithm": algorithm, "model": model, "metrics": metrics, "predictions": binary_preds, "scores": scores}


# ---------------------------------------------------------------------------
# Dimensionality Reduction (PCA Analysis)
# ---------------------------------------------------------------------------

def run_pca_analysis(
    X: np.ndarray,
    hyperparams: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    if hyperparams is None:
        hyperparams = {}

    n_components = hyperparams.get("n_components", min(X.shape[1], min(X.shape[0], 10)))
    n_components = min(int(n_components), X.shape[1], X.shape[0])

    t_start = time.perf_counter()
    model = PCA(n_components=n_components, random_state=seed)
    X_transformed = model.fit_transform(X)
    training_time = time.perf_counter() - t_start

    evr = model.explained_variance_ratio_
    cumulative_evr = np.cumsum(evr)
    n_for_95 = int(np.searchsorted(cumulative_evr, 0.95)) + 1

    metrics: Dict[str, Any] = {
        "training_time_seconds": training_time,
        "n_components": n_components,
        "explained_variance_ratio": evr.tolist(),
        "cumulative_explained_variance": cumulative_evr.tolist(),
        "total_explained_variance": float(np.sum(evr)),
        "n_components_for_95pct_variance": min(n_for_95, n_components),
        "singular_values": model.singular_values_.tolist(),
    }

    return {"algorithm": "pca_analysis", "model": model, "metrics": metrics, "X_transformed": X_transformed}


# ---------------------------------------------------------------------------
# Unified unsupervised trainer
# ---------------------------------------------------------------------------

def train_unsupervised(
    algorithm: str,
    X: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
    models_dir: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Unified interface for unsupervised model training."""
    if hyperparams is None:
        hyperparams = {}

    task_category = get_unsupervised_task(algorithm)
    if task_category == "unknown":
        raise ValueError(f"Unknown unsupervised algorithm: {algorithm!r}. Supported: {list(UNSUPERVISED_ALGORITHMS.keys())}")

    logger.info("Training unsupervised algorithm: %s (task=%s)", algorithm, task_category)

    if task_category == "clustering":
        result = train_clustering(X, algorithm, hyperparams, y_true, seed)
    elif task_category == "anomaly_detection":
        result = train_anomaly_detection(X, algorithm, hyperparams, y_true, seed)
    elif task_category == "dimensionality_reduction":
        result = run_pca_analysis(X, hyperparams, seed)
    else:
        raise ValueError(f"Unhandled task category: {task_category}")

    model = result.pop("model", None)
    metrics = result.get("metrics", {})

    model_path = None
    if model is not None and models_dir is not None:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{algorithm}.joblib"
        joblib.dump(model, model_path)
        logger.info("Saved unsupervised model -> %s", model_path)

    return {
        "algorithm": algorithm,
        "task_category": task_category,
        "model_path": str(model_path) if model_path else None,
        "metrics": metrics,
        "hyperparams": hyperparams,
    }


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_unsupervised_inputs(
    X: np.ndarray,
    algorithm: str,
    hyperparams: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if X is None or len(X) == 0:
        errors.append("Feature array X must not be empty.")
        return False, errors
    if X.ndim != 2:
        errors.append(f"Feature array must be 2D, got shape {X.shape}.")
        return False, errors
    if X.shape[0] < 2:
        errors.append(f"At least 2 samples required, got {X.shape[0]}.")
        return False, errors
    if algorithm not in UNSUPERVISED_ALGORITHMS:
        errors.append(f"Unknown algorithm '{algorithm}'. Supported: {list(UNSUPERVISED_ALGORITHMS.keys())}")
        return False, errors
    if hyperparams is None:
        hyperparams = {}
    if algorithm == "kmeans":
        n_clusters = hyperparams.get("n_clusters", 2)
        if int(n_clusters) < 2:
            errors.append("KMeans: n_clusters must be >= 2.")
        if int(n_clusters) >= X.shape[0]:
            errors.append(f"KMeans: n_clusters ({n_clusters}) must be < n_samples ({X.shape[0]}).")
    elif algorithm == "pca_analysis":
        n_components = hyperparams.get("n_components", X.shape[1])
        max_comp = min(X.shape[0], X.shape[1])
        if int(n_components) < 1 or int(n_components) > max_comp:
            errors.append(f"PCA: n_components must be between 1 and {max_comp}, got {n_components}.")
    if not np.isfinite(X).all():
        n_inf = int(np.sum(~np.isfinite(X)))
        errors.append(f"Feature array contains {n_inf} non-finite value(s). Apply imputation first.")

    return len(errors) == 0, errors
