"""
tests/unit/test_image_pipeline.py
----------------------------------
Unit tests for src/image/pipeline.py

Tests cover:
  - Dependency checking
  - Image classifier training on synthetic embeddings
  - Image clustering on synthetic embeddings
  - Image anomaly detection on synthetic embeddings
  - Error handling (insufficient classes, invalid tasks)
  - End-to-end run_image_pipeline with a synthetic valid image ZIP archive
"""

from __future__ import annotations

import io
import zipfile
import numpy as np
import pytest
from PIL import Image

from src.image.pipeline import (
    check_image_dependencies,
    train_image_classifier,
    train_image_clustering,
    train_image_anomaly_detection,
    run_image_pipeline,
)


def _make_dummy_image_zip(n_samples_per_class: int = 10) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for cls in ["cats", "dogs"]:
            for i in range(n_samples_per_class):
                # Simple 32x32 image with random colors
                color = (255, 0, 0) if cls == "cats" else (0, 0, 255)
                img = Image.new("RGB", (32, 32), color=color)
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="JPEG")
                zf.writestr(f"dataset/{cls}/img_{i:02d}.jpg", img_bytes.getvalue())
    return buf.getvalue()


HAS_IMAGE_DEPS, MISSING_IMAGE_DEPS = check_image_dependencies()
requires_image_deps = pytest.mark.skipif(
    not HAS_IMAGE_DEPS,
    reason=f"Optional image dependencies not installed: {MISSING_IMAGE_DEPS}",
)


def test_check_image_dependencies():
    ok, missing = check_image_dependencies()
    assert isinstance(ok, bool)
    assert isinstance(missing, list)


def test_train_image_classifier_synthetic():
    rng = np.random.default_rng(42)
    # 40 samples, 16 features (or 576 embedding dim), 2 classes
    X = np.vstack([rng.normal(1.0, 0.5, (20, 16)), rng.normal(-1.0, 0.5, (20, 16))])
    y = np.array([0] * 20 + [1] * 20)

    result = train_image_classifier(X, y, hyperparams={"C": 1.0}, seed=42)
    assert result["algorithm"] == "image_classification"
    assert "metrics" in result
    assert result["metrics"]["accuracy"] > 0.8
    assert "f1" in result["metrics"]
    assert "precision" in result["metrics"]
    assert "recall" in result["metrics"]


def test_train_image_classifier_single_class_raises():
    X = np.random.randn(20, 8)
    y = np.zeros(20, dtype=int)
    with pytest.raises(ValueError, match="at least 2 classes"):
        train_image_classifier(X, y)


def test_train_image_clustering_synthetic():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 16))
    result = train_image_clustering(X, algorithm="kmeans", hyperparams={"n_clusters": 3}, seed=42)
    assert result["task_category"] == "image_clustering"
    assert "metrics" in result
    assert "silhouette_score" in result["metrics"]
    assert "calinski_harabasz_score" in result["metrics"]


def test_train_image_anomaly_detection_synthetic():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 16))
    result = train_image_anomaly_detection(X, hyperparams={"contamination": 0.1}, seed=42)
    assert result["task_category"] == "image_anomaly_detection"
    assert "metrics" in result
    assert "anomaly_rate" in result["metrics"]
    assert "n_anomalies_detected" in result["metrics"]


def test_run_image_pipeline_invalid_task():
    zip_bytes = _make_dummy_image_zip(2)
    result = run_image_pipeline(zip_bytes, task="unsupported_task")
    assert result["status"] == "error"
    if not HAS_IMAGE_DEPS:
        assert "Missing image dependencies" in result["error"]
    else:
        assert "Unknown image task" in result["error"]


def test_run_image_pipeline_missing_deps_handled(tmp_path):
    """When dependencies are missing, run_image_pipeline returns an informative error dictionary."""
    from unittest import mock
    with mock.patch("src.image.pipeline.check_image_dependencies", return_value=(False, ["torch", "torchvision"])):
        zip_bytes = _make_dummy_image_zip(2)
        result = run_image_pipeline(
            zip_bytes=zip_bytes,
            task="image_clustering",
            models_dir=tmp_path / "models",
        )
        assert result["status"] == "error"
        assert "Missing image dependencies" in result["error"]


@requires_image_deps
def test_run_image_pipeline_clustering_success(tmp_path):
    zip_bytes = _make_dummy_image_zip(5)
    result = run_image_pipeline(
        zip_bytes=zip_bytes,
        task="image_clustering",
        hyperparams={"n_clusters": 2},
        models_dir=tmp_path / "models",
        seed=42,
    )
    assert result["status"] == "success"
    assert "metrics" in result
    assert result["metrics"]["n_images_processed"] == 10
    assert result["model_path"] is not None


@requires_image_deps
def test_run_image_pipeline_anomaly_detection_success(tmp_path):
    zip_bytes = _make_dummy_image_zip(5)
    result = run_image_pipeline(
        zip_bytes=zip_bytes,
        task="image_anomaly_detection",
        hyperparams={"contamination": 0.1},
        models_dir=tmp_path / "models",
        seed=42,
    )
    assert result["status"] == "success"
    assert "metrics" in result
    assert result["metrics"]["n_images_processed"] == 10
