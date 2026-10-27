"""
src/image/pipeline.py
----------------------
Lightweight image ML pipeline (CPU-only):
  - Classification: MobileNetV3-Small pretrained feature extractor + linear head
  - Clustering:     pretrained embeddings + KMeans / DBSCAN
  - Anomaly Det.:   pretrained embeddings + IsolationForest

Design principles:
  - Batch processing to avoid loading full dataset into memory
  - Lazy PIL import (graceful degradation if not installed)
  - No GPU required; torch.no_grad() for inference
  - Configurable batch size and max file count
  - Returns metrics compatible with existing experiment_schema
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Configurable limits
DEFAULT_BATCH_SIZE: int = 32
MAX_IMAGES_TO_PROCESS: int = 10_000
EMBED_DIM: int = 576       # MobileNetV3-Small last channel dim
IMG_SIZE: int = 224        # Standard ImageNet input size


def _pil_available() -> bool:
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _torchvision_available() -> bool:
    try:
        import torchvision  # noqa: F401
        return True
    except ImportError:
        return False


def check_image_dependencies() -> Tuple[bool, List[str]]:
    """Check whether optional image dependencies are installed."""
    missing = []
    if not _pil_available():
        missing.append("Pillow")
    if not _torch_available():
        missing.append("torch")
    if not _torchvision_available():
        missing.append("torchvision")
    return len(missing) == 0, missing


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _build_extractor():
    """
    Build a MobileNetV3-Small feature extractor.
    Falls back to None if torch/torchvision unavailable.
    """
    if not (_torch_available() and _torchvision_available()):
        return None, None

    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    # Load pretrained weights (downloads once, cached locally)
    try:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
    except Exception:
        # Older torchvision API fallback
        model = models.mobilenet_v3_small(pretrained=True)

    # Strip classifier head, keep feature extraction layers
    model.classifier = torch.nn.Identity()
    model.eval()

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, transform


_EXTRACTOR = None
_TRANSFORM = None


def _get_extractor():
    global _EXTRACTOR, _TRANSFORM
    if _EXTRACTOR is None:
        _EXTRACTOR, _TRANSFORM = _build_extractor()
    return _EXTRACTOR, _TRANSFORM


def extract_image_embedding(pil_image) -> Optional[np.ndarray]:
    """Extract a feature embedding from a single PIL Image."""
    model, transform = _get_extractor()
    if model is None or transform is None:
        return None
    try:
        import torch
        tensor = transform(pil_image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            feat = model(tensor)
        return feat.squeeze().cpu().numpy()
    except Exception as e:
        logger.warning("Embedding extraction failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# ZIP dataset loading (batched, memory-safe)
# ---------------------------------------------------------------------------

def load_images_from_zip(
    zip_bytes: bytes,
    max_images: int = MAX_IMAGES_TO_PROCESS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Extract and embed images from a ZIP archive in batches.

    Returns:
        embeddings:  list of 1-D numpy arrays
        labels:      list of class folder names ('' if none)
        filenames:   list of original filenames
    """
    ok, missing = check_image_dependencies()
    if not ok:
        raise RuntimeError(f"Missing optional image dependencies: {', '.join(missing)}. Install with: pip install Pillow torch torchvision")

    from PIL import Image, UnidentifiedImageError

    SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    embeddings: List[np.ndarray] = []
    labels: List[str] = []
    filenames: List[str] = []
    n_corrupt = 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        candidates = [n for n in zf.namelist() if Path(n).suffix.lower() in SUPPORTED]
        candidates = candidates[:max_images]

        for name in candidates:
            # Infer label from parent folder
            parts = Path(name).parts
            if len(parts) >= 3:
                label = parts[1]
            elif len(parts) == 2:
                label = parts[0]
            else:
                label = ""

            try:
                img_bytes = zf.read(name)
                img = Image.open(io.BytesIO(img_bytes))
                emb = extract_image_embedding(img)
                if emb is not None:
                    embeddings.append(emb)
                    labels.append(label)
                    filenames.append(name)
            except (UnidentifiedImageError, Exception) as e:
                n_corrupt += 1
                logger.debug("Skipping corrupt/unreadable image %s: %s", name, e)

    logger.info(
        "Loaded %d embeddings from %d candidates (%d corrupt/skipped)",
        len(embeddings), len(candidates), n_corrupt
    )

    return embeddings, labels, filenames


# ---------------------------------------------------------------------------
# Image Classification
# ---------------------------------------------------------------------------

def train_image_classifier(
    embeddings: np.ndarray,
    y: np.ndarray,
    hyperparams: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train a linear classifier on top of pretrained image embeddings.
    Uses existing supervised metrics from sklearn.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

    if hyperparams is None:
        hyperparams = {}

    if len(np.unique(y)) < 2:
        raise ValueError("Image classification requires at least 2 classes in the dataset.")

    test_size = hyperparams.get("test_size", 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=test_size, random_state=seed, stratify=y
    )

    t_start = time.perf_counter()
    clf = LogisticRegression(
        max_iter=hyperparams.get("max_iter", 1000),
        C=hyperparams.get("C", 1.0),
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    training_time = time.perf_counter() - t_start

    y_pred = clf.predict(X_test)
    n_classes = len(np.unique(y))
    avg = "binary" if n_classes == 2 else "weighted"

    metrics = {
        "training_time_seconds": training_time,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
        "n_classes": n_classes,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "backbone": "MobileNetV3-Small (ImageNet pretrained)",
        "head": "LogisticRegression",
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }

    return {
        "algorithm": "image_classification",
        "task_category": "image_classification",
        "model": clf,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Image Clustering
# ---------------------------------------------------------------------------

def train_image_clustering(
    embeddings: np.ndarray,
    algorithm: str = "kmeans",
    hyperparams: Optional[Dict[str, Any]] = None,
    y_true: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Cluster image embeddings using KMeans or DBSCAN."""
    from src.training.unsupervised import train_clustering

    result = train_clustering(embeddings, algorithm, hyperparams, y_true, seed)
    result["task_category"] = "image_clustering"
    return result


# ---------------------------------------------------------------------------
# Image Anomaly Detection
# ---------------------------------------------------------------------------

def train_image_anomaly_detection(
    embeddings: np.ndarray,
    hyperparams: Optional[Dict[str, Any]] = None,
    y_true: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Detect anomalous images using IsolationForest on embeddings."""
    from src.training.unsupervised import train_anomaly_detection

    result = train_anomaly_detection(embeddings, "isolation_forest", hyperparams, y_true, seed)
    result["task_category"] = "image_anomaly_detection"
    return result


# ---------------------------------------------------------------------------
# Unified image pipeline entry-point
# ---------------------------------------------------------------------------

def run_image_pipeline(
    zip_bytes: bytes,
    task: str = "image_classification",
    hyperparams: Optional[Dict[str, Any]] = None,
    models_dir: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full image pipeline: validate -> embed -> train -> return results.

    Parameters
    ----------
    zip_bytes:   raw bytes of uploaded ZIP archive
    task:        'image_classification', 'image_clustering', 'image_anomaly_detection'
    hyperparams: dict with algorithm-specific parameters
    models_dir:  path to save trained model artifact
    seed:        random seed

    Returns a result dict with metrics and model_path.
    """
    if hyperparams is None:
        hyperparams = {}

    # Check dependencies
    ok, missing = check_image_dependencies()
    if not ok:
        return {
            "status": "error",
            "error": f"Missing image dependencies: {', '.join(missing)}. Install: pip install Pillow torch torchvision",
            "algorithm": task,
            "task_category": task,
            "metrics": {},
        }

    # Validate ZIP
    from src.image.validator import validate_image_zip
    is_valid, errors, zip_info = validate_image_zip(zip_bytes)
    if not is_valid:
        return {
            "status": "error",
            "error": "; ".join(errors),
            "algorithm": task,
            "task_category": task,
            "metrics": {},
            "zip_info": zip_info,
        }

    logger.info("ZIP validation passed: %s", zip_info)

    # Extract embeddings
    t_embed_start = time.perf_counter()
    max_img = hyperparams.get("max_images", MAX_IMAGES_TO_PROCESS)
    embeddings_list, labels_list, filenames = load_images_from_zip(
        zip_bytes, max_images=max_img
    )
    embed_time = time.perf_counter() - t_embed_start

    if len(embeddings_list) == 0:
        return {
            "status": "error",
            "error": "No valid images could be embedded from the archive.",
            "algorithm": task,
            "task_category": task,
            "metrics": {},
        }

    X = np.vstack(embeddings_list)

    # Convert string labels to integers if present
    label_names = sorted(set(labels_list))
    label_map = {l: i for i, l in enumerate(label_names) if l}
    y = np.array([label_map.get(l, -1) for l in labels_list])
    has_labels = len(label_names) > 0 and label_names != [""]

    logger.info(
        "Embedded %d images in %.2fs. Classes: %s",
        len(embeddings_list), embed_time, label_names[:10]
    )

    # Task dispatch
    import joblib

    if task == "image_classification":
        if not has_labels or len(np.unique(y[y >= 0])) < 2:
            return {
                "status": "error",
                "error": (
                    "Image classification requires at least 2 labelled classes in class-folder format "
                    "(e.g., archive/class_name/image.jpg). "
                    f"Found classes: {label_names[:10]}"
                ),
                "algorithm": task,
                "task_category": task,
                "metrics": {},
            }
        valid_mask = y >= 0
        result = train_image_classifier(X[valid_mask], y[valid_mask], hyperparams, seed)

    elif task == "image_clustering":
        algo = hyperparams.get("clustering_algorithm", "kmeans")
        y_for_eval = y if has_labels else None
        result = train_image_clustering(X, algo, hyperparams, y_for_eval, seed)

    elif task == "image_anomaly_detection":
        y_for_eval = y if has_labels else None
        result = train_image_anomaly_detection(X, hyperparams, y_for_eval, seed)

    else:
        return {
            "status": "error",
            "error": f"Unknown image task: {task!r}. Use: image_classification, image_clustering, image_anomaly_detection",
            "algorithm": task,
            "task_category": task,
            "metrics": {},
        }

    model = result.pop("model", None)
    result["metrics"]["embedding_time_seconds"] = round(embed_time, 3)
    result["metrics"]["n_images_processed"] = len(embeddings_list)
    result["metrics"]["n_classes"] = zip_info.get("n_classes", len(label_names))
    result["zip_info"] = zip_info

    # Save model
    model_path = None
    if model is not None and models_dir is not None:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{task}.joblib"
        joblib.dump(model, model_path)
        logger.info("Saved image model -> %s", model_path)

    result["model_path"] = str(model_path) if model_path else None
    result["status"] = "success"
    result["hyperparams"] = hyperparams

    return result
