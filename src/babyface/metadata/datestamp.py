"""
Datestamp anomaly detection and correction.

Common problems in infant photo collections:
  1. Camera clock reset to 2000-01-01 or 1970-01-01 (epoch)
  2. Camera clock set to wrong year (common after battery swap)
  3. Scanned / imported photos with the scan/import date instead of original
  4. Timezone errors that shift dates by ±1 day

Detection strategy:
  - Collect all available dates per photo (EXIF, DigiKam, folder, filename)
  - Flag when any two sources disagree by > DISAGREEMENT_DAYS
  - Flag epoch-reset dates (before SANE_EARLIEST)
  - Flag future dates (after SANE_LATEST)
  - Flag EXIF dates that contradict the folder date by > FOLDER_DRIFT_DAYS

Correction strategy:
  - Epoch/future: use folder date
  - Disagreement: use median of clean neighbour EXIF dates in same album,
    then fall back to folder date
"""
from __future__ import annotations
from datetime import datetime, timedelta, date
from collections import defaultdict
from ..db.models import Photo

# Sanity bounds — adjust if your collection predates digital cameras further
SANE_EARLIEST   = datetime(1990, 1, 1)
SANE_LATEST     = datetime(2030, 1, 1)
DISAGREEMENT_DAYS   = 90   # flag if any two sources differ by more than this
FOLDER_DRIFT_DAYS   = 30   # flag moderate EXIF ↔ folder drift (softer signal)


def _available_dates(photo: Photo) -> list[tuple[str, datetime]]:
    sources: list[tuple[str, datetime]] = []
    if photo.exif_date:
        sources.append(("exif", photo.exif_date))
    if photo.digikam_date:
        sources.append(("digikam", photo.digikam_date))
    if photo.filename_date:
        sources.append(("filename", photo.filename_date))
    if photo.folder_date:
        d = photo.folder_date
        sources.append(("folder", datetime(d.year, d.month, d.day)))
    return sources


def detect_anomalies(photo: Photo) -> list[str]:
    """Return a list of flag strings describing datestamp problems (empty = clean)."""
    flags: list[str] = []
    sources = _available_dates(photo)

    if not sources:
        return ["no_date"]

    for src, dt in sources:
        if src == "exif":
            if dt < SANE_EARLIEST:
                flags.append(f"epoch_reset:exif={dt.date()}")
            elif dt > SANE_LATEST:
                flags.append(f"future:exif={dt.date()}")

    # Cross-source disagreement
    for i, (src_a, dt_a) in enumerate(sources):
        for src_b, dt_b in sources[i + 1:]:
            delta = abs((dt_a - dt_b).days)
            if delta > DISAGREEMENT_DAYS:
                flags.append(f"disagree:{src_a}↔{src_b}={delta}d")

    # Softer EXIF ↔ folder drift (only when not already caught above)
    if photo.exif_date and photo.folder_date:
        fd = datetime(photo.folder_date.year, photo.folder_date.month, photo.folder_date.day)
        drift = abs((photo.exif_date - fd).days)
        if FOLDER_DRIFT_DAYS < drift <= DISAGREEMENT_DAYS:
            flags.append(f"folder_drift:{drift}d")

    return flags


def _folder_dt(photo: Photo) -> datetime | None:
    if photo.folder_date:
        d = photo.folder_date
        return datetime(d.year, d.month, d.day)
    return None


def estimate_corrected_date(photo: Photo, album_photos: list[Photo]) -> datetime | None:
    """
    Best-effort corrected date.  Priority:
      1. Unflagged EXIF date (already correct)
      2. Median EXIF date from clean album neighbours (same camera model preferred)
      3. Folder date
      4. DigiKam date
    """
    if not photo.datestamp_flags:
        return photo.exif_date or photo.digikam_date or _folder_dt(photo)

    # Collect clean neighbour EXIF dates
    clean_dates: list[datetime] = []
    for nb in album_photos:
        if nb.id == photo.id or nb.datestamp_flags:
            continue
        if nb.exif_date is None:
            continue
        # Prefer same camera model if we know it
        if photo.camera_model and nb.camera_model and nb.camera_model != photo.camera_model:
            continue
        clean_dates.append(nb.exif_date)

    if clean_dates:
        clean_dates.sort()
        return clean_dates[len(clean_dates) // 2]

    return _folder_dt(photo) or photo.digikam_date


def annotate_photos(photos: list[Photo]) -> None:
    """
    Detect anomalies and fill corrected_date for all photos in-place.
    Groups photos by album so neighbours are drawn from the same shoot.
    """
    for photo in photos:
        photo.datestamp_flags = detect_anomalies(photo)

    by_album: dict[str, list[Photo]] = defaultdict(list)
    for photo in photos:
        by_album[photo.album_path].append(photo)

    for photo in photos:
        photo.corrected_date = estimate_corrected_date(photo, by_album[photo.album_path])
