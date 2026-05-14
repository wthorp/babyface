"""
Core data models shared across the pipeline.

DigiKam schema key points:
  - Images.status=1 means "available" (not trashed/hidden)
  - ImageTagProperties.property='autodetectedFace' → XML rect in pixels
  - Tags with pid=4 are People tags; tag.name = person's display name
  - FaceMatrices.embedding = 128-dim float32 stored as 512-byte BLOB

Synofoto (PostgreSQL) mirrors this with:
  - unit.takentime = Unix timestamp
  - face.bounding_box = JSON {top_left:{x,y}, bottom_right:{x,y}} (normalized 0-1)
  - person.name = display name
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
import numpy as np


@dataclass
class FaceRegion:
    tag_id: int
    person_name: str | None
    # Pixel coords from DigiKam ImageTagProperties <rect .../>
    x: int
    y: int
    width: int
    height: int
    # DINOv2 patch-mean embedding set during extraction step
    dino_embedding: np.ndarray | None = None


@dataclass
class Photo:
    id: int
    filename: str
    album_path: str          # e.g. /2007-03-10  or  /2009-08-23 new in Germany
    full_path: Path          # photo_root + album_path + filename
    file_size: int | None
    unique_hash: str | None  # used to look up thumbnails in thumbnails-digikam.db

    # Date sources — populated progressively through the pipeline
    exif_date: datetime | None = None
    digikam_date: datetime | None = None   # ImageInformation.creationDate
    folder_date: date | None = None        # parsed from album_path prefix
    filename_date: datetime | None = None  # parsed from filename patterns

    # Camera
    camera_make: str | None = None
    camera_model: str | None = None

    # Dimensions
    width: int | None = None
    height: int | None = None

    faces: list[FaceRegion] = field(default_factory=list)

    # Datestamp validation outputs
    datestamp_flags: list[str] = field(default_factory=list)
    corrected_date: datetime | None = None


@dataclass
class Identity:
    """A known person from recognition.db / DigiKam face engine."""
    id: int
    name: str
    # Each entry is a 128-dim float32 vector (one per training face crop)
    embeddings: list[np.ndarray] = field(default_factory=list)

    @property
    def centroid(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        return np.stack(self.embeddings).mean(axis=0)
