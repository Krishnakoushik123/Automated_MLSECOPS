"""
src/image/validator.py
-----------------------
Secure image dataset validation:
  - ZIP archive safety (no traversal, size limits, decompression ratio)
  - Image format validation (JPEG, PNG, BMP, TIFF, WEBP)
  - Class-folder structure or CSV manifest support
  - Configurable resource limits (file count, size, dimensions)
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default limits
MAX_UPLOAD_BYTES: int = 500 * 1024 * 1024      # 500 MB
MAX_DECOMPRESSION_RATIO: float = 50.0           # max 50x expansion
MAX_FILE_COUNT: int = 50_000
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
MAX_IMAGE_DIM: int = 10_000                     # pixels per side


def validate_image_zip(
    zip_bytes: bytes,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
    max_file_count: int = MAX_FILE_COUNT,
    max_decompression_ratio: float = MAX_DECOMPRESSION_RATIO,
) -> Tuple[bool, List[str], Dict]:
    """
    Validate an uploaded ZIP archive containing image data.

    Returns (is_valid, errors, info_dict).
    """
    errors: List[str] = []
    info: Dict = {}

    # Size check
    if len(zip_bytes) > max_upload_bytes:
        errors.append(
            f"Upload exceeds size limit: {len(zip_bytes) / 1024**2:.1f} MB > "
            f"{max_upload_bytes / 1024**2:.0f} MB."
        )
        return False, errors, info

    if len(zip_bytes) < 4:
        errors.append("File appears empty or too small to be a valid ZIP.")
        return False, errors, info

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        errors.append("File is not a valid ZIP archive.")
        return False, errors, info

    all_names = zf.namelist()

    # File count check
    if len(all_names) > max_file_count:
        errors.append(f"Archive contains too many files: {len(all_names)} > {max_file_count}.")
        return False, errors, info

    # Archive traversal check (path injection)
    safe_names = []
    for name in all_names:
        norm = Path(name).as_posix()
        if ".." in norm or norm.startswith("/"):
            errors.append(f"Unsafe archive path detected: {name!r}. Rejecting archive.")
            return False, errors, info
        safe_names.append(name)

    # Decompression ratio check (decompression bomb prevention)
    total_compressed = len(zip_bytes)
    total_uncompressed = sum(info_item.file_size for info_item in zf.infolist())
    ratio = total_uncompressed / max(total_compressed, 1)
    if ratio > max_decompression_ratio:
        errors.append(
            f"Decompression ratio too high: {ratio:.1f}x > {max_decompression_ratio:.0f}x limit. "
            "Possible decompression bomb."
        )
        return False, errors, info

    # Identify image files
    image_names = [n for n in safe_names if Path(n).suffix.lower() in SUPPORTED_IMAGE_EXTS]
    if len(image_names) == 0:
        errors.append(
            f"No supported image files found in archive. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_IMAGE_EXTS))}"
        )
        return False, errors, info

    info["total_files"] = len(safe_names)
    info["image_files"] = len(image_names)
    info["total_uncompressed_mb"] = round(total_uncompressed / 1024**2, 2)
    info["decompression_ratio"] = round(ratio, 2)

    # Detect structure: class-folder layout
    class_dirs: Dict[str, List[str]] = {}
    for name in image_names:
        parts = Path(name).parts
        if len(parts) >= 2:
            # parts[0] could be the archive root, parts[1] might be the class
            # Handle: images/<class>/<file> OR <class>/<file>
            class_candidate = parts[1] if len(parts) >= 3 else parts[0]
            class_dirs.setdefault(class_candidate, []).append(name)
        else:
            class_dirs.setdefault("__root__", []).append(name)

    # Remove __root__ as a class (unlabelled)
    unlabelled = class_dirs.pop("__root__", [])
    info["class_folders"] = {k: len(v) for k, v in class_dirs.items()}
    info["unlabelled_images"] = len(unlabelled)
    info["n_classes"] = len(class_dirs)
    info["image_names_sample"] = image_names[:5]

    return True, errors, info


def validate_single_image_bytes(image_bytes: bytes, name: str = "image") -> Tuple[bool, str]:
    """
    Validate a single image from raw bytes.
    Returns (is_valid, error_message).
    """
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # detects corrupt files without fully decoding
        return True, ""
    except ImportError:
        # PIL not installed — skip pixel validation, only check header
        if len(image_bytes) < 8:
            return False, f"{name}: file too small to be a valid image."
        return True, ""
    except Exception as e:
        return False, f"{name}: {e}"


def validate_image_dimensions(image_bytes: bytes, name: str = "image") -> Tuple[bool, str, Optional[Tuple[int, int]]]:
    """
    Check image dimensions are within safe bounds.
    Returns (is_valid, error_message, (width, height) or None).
    """
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        if w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM:
            return False, f"{name}: dimensions {w}x{h} exceed max {MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}.", (w, h)
        return True, "", (w, h)
    except ImportError:
        return True, "", None
    except Exception as e:
        return False, f"{name}: cannot read dimensions: {e}", None
