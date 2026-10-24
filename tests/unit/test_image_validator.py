"""
tests/unit/test_image_validator.py
------------------------------------
Unit tests for src/image/validator.py

Tests cover:
  - Valid ZIP archive with class-folder images
  - Archive traversal detection (path injection)
  - Decompression bomb detection
  - No supported images in archive
  - File size limit enforcement
  - Empty / corrupt ZIP detection
"""

import io
import zipfile
import pytest

from src.image.validator import validate_image_zip


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_zip(entries: dict) -> bytes:
    """
    Create an in-memory ZIP.
    entries: {archive_path: content_bytes}
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


TINY_JPEG = bytes([
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10,  # JPEG SOI + APP0 marker
    0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x01,
    0x00, 0x00, 0xFF, 0xD9,              # EOI
])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_valid_zip_class_folder_structure():
    """Standard class-folder layout should pass validation."""
    entries = {
        "dataset/cat/img1.jpg": TINY_JPEG,
        "dataset/cat/img2.jpg": TINY_JPEG,
        "dataset/dog/img1.jpg": TINY_JPEG,
    }
    is_valid, errors, info = validate_image_zip(_make_zip(entries))
    assert is_valid, f"Expected valid, got errors: {errors}"
    assert info["image_files"] == 3
    assert info["n_classes"] >= 1


def test_archive_traversal_rejected():
    """ZIP entries with ../ paths should be rejected."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../etc/passwd.jpg", b"x")
    is_valid, errors, info = validate_image_zip(buf.getvalue())
    assert not is_valid
    assert any("Unsafe" in e or "traversal" in e.lower() for e in errors)


def test_no_images_in_zip():
    """ZIP with no image files should be rejected."""
    entries = {"data.csv": b"a,b,c\n1,2,3\n", "readme.txt": b"hello"}
    is_valid, errors, info = validate_image_zip(_make_zip(entries))
    assert not is_valid
    assert any("No supported image" in e for e in errors)


def test_empty_file_rejected():
    """Empty bytes should fail."""
    is_valid, errors, info = validate_image_zip(b"")
    assert not is_valid


def test_not_a_zip():
    """Random bytes that aren't a ZIP should be detected."""
    is_valid, errors, info = validate_image_zip(b"PK not a real zip" + b"\x00" * 100)
    assert not is_valid


def test_size_limit_enforced():
    """Upload exceeding max_upload_bytes should be rejected."""
    entries = {"cat/img1.jpg": TINY_JPEG}
    zb = _make_zip(entries)
    # Set max to 1 byte
    is_valid, errors, info = validate_image_zip(zb, max_upload_bytes=1)
    assert not is_valid
    assert any("size limit" in e for e in errors)


def test_file_count_limit_enforced():
    """More than max_file_count entries should be rejected."""
    entries = {f"cat/img{i}.jpg": TINY_JPEG for i in range(10)}
    zb = _make_zip(entries)
    is_valid, errors, info = validate_image_zip(zb, max_file_count=5)
    assert not is_valid
    assert any("too many files" in e.lower() for e in errors)


def test_decompression_bomb_rejected():
    """Excessive decompression ratio should be rejected."""
    # Create highly compressible content
    compressible_content = b"A" * 100_000
    entries = {"cat/img.png": compressible_content}
    zb = _make_zip(entries)
    # Set limit to 1x (i.e., any expansion triggers it)
    is_valid, errors, info = validate_image_zip(zb, max_decompression_ratio=0.01)
    assert not is_valid
    assert any("ratio" in e.lower() or "bomb" in e.lower() for e in errors)


def test_info_fields_populated():
    """Info dict should have expected fields on success."""
    entries = {
        "archive/cats/a.jpg": TINY_JPEG,
        "archive/dogs/b.png": TINY_JPEG * 2,
    }
    is_valid, errors, info = validate_image_zip(_make_zip(entries))
    assert is_valid
    assert "total_files" in info
    assert "image_files" in info
    assert "n_classes" in info
    assert "decompression_ratio" in info
    assert "image_names_sample" in info


def test_mixed_file_types():
    """ZIPs with some non-image files should succeed if enough images exist."""
    entries = {
        "class_a/img1.jpg": TINY_JPEG,
        "class_a/metadata.txt": b"description",
        "class_b/img1.png": TINY_JPEG,
    }
    is_valid, errors, info = validate_image_zip(_make_zip(entries))
    assert is_valid
    assert info["image_files"] == 2
    assert info["total_files"] == 3
