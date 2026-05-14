"""
EXIF metadata extraction.

Reads DateTimeOriginal, camera make/model, and GPS from image files.
Augments Photo objects with exif_date, camera_make/model if DigiKam
didn't already have them (DigiKam often stores camera from its own scan).
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path

import exifread

_DATE_TAGS = [
    "EXIF DateTimeOriginal",
    "EXIF DateTimeDigitized",
    "Image DateTime",
]

_MAKE_TAG  = "Image Make"
_MODEL_TAG = "Image Model"


def _parse_exif_dt(raw: str) -> datetime | None:
    raw = raw.strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def extract_exif(path: Path) -> dict:
    """
    Return a dict with keys:
      date        — datetime | None  (DateTimeOriginal preferred)
      make        — str | None
      model       — str | None
    Returns empty dict on any read/parse error.
    """
    try:
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, details=False, stop_tag="GPS GPSAltitude")
    except Exception:
        return {}

    result: dict = {}

    for tag in _DATE_TAGS:
        if tag in tags:
            dt = _parse_exif_dt(str(tags[tag]))
            if dt:
                result["date"] = dt
                break

    if _MAKE_TAG in tags:
        result["make"] = str(tags[_MAKE_TAG]).strip()
    if _MODEL_TAG in tags:
        result["model"] = str(tags[_MODEL_TAG]).strip()

    return result


def enrich_photos_with_exif(photos, *, quiet: bool = False) -> None:
    """
    Fill exif_date / camera_make / camera_model fields in-place for every
    Photo whose full_path exists on disk.  Skips photos that already have all
    three fields set (avoids redundant I/O on re-runs).
    """
    from ..db.models import Photo

    for photo in photos:
        if photo.exif_date and photo.camera_make and photo.camera_model:
            continue
        if not photo.full_path.exists():
            continue
        data = extract_exif(photo.full_path)
        if not photo.exif_date:
            photo.exif_date = data.get("date")
        if not photo.camera_make:
            photo.camera_make = data.get("make")
        if not photo.camera_model:
            photo.camera_model = data.get("model")
