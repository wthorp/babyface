"""
Reads photos, face regions, and identities from the DigiKam SQLite databases.

Database files (expected next to src/ in project root):
  digikam4.db       — main metadata (Images, Albums, Tags, ImageTagProperties, …)
  recognition.db    — face embeddings (Identities, FaceMatrices)
  thumbnails-digikam.db — JPEG thumbnail BLOBs keyed by (uniqueHash, fileSize)
"""
from __future__ import annotations
import re
import sqlite3
import struct
from datetime import datetime, date
from pathlib import Path
import numpy as np

from .models import FaceRegion, Identity, Photo

# Register HEIC/HEIF support if pillow-heif is installed
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# Default paths relative to the project root (two levels above this file)
_ROOT = Path(__file__).parent.parent.parent.parent
DIGIKAM_DB    = _ROOT / "digikam4.db"
RECOGNITION_DB = _ROOT / "recognition.db"
THUMBNAILS_DB  = _ROOT / "thumbnails-digikam.db"

# Module-level connection cache — avoids opening a new SQLite connection per face region
_thumbnail_conn_cache: dict[Path, sqlite3.Connection] = {}


def _open_db(db_path: "Path | str") -> sqlite3.Connection:
    """Open a SQLite DB, using immutable URI mode for read-only filesystems."""
    p = Path(db_path)
    try:
        conn = sqlite3.connect(p)
        conn.execute("SELECT 1")
        return conn
    except sqlite3.OperationalError:
        conn = sqlite3.connect(
            f"file:{p.resolve().as_posix()}?immutable=1", uri=True
        )
        return conn


def _thumbnail_conn(db_path: Path) -> sqlite3.Connection:
    if db_path not in _thumbnail_conn_cache:
        _thumbnail_conn_cache[db_path] = _open_db(db_path)
    return _thumbnail_conn_cache[db_path]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_rect(xml: str) -> tuple[int, int, int, int] | None:
    """Parse DigiKam's <rect x="…" y="…" width="…" height="…"/> face rect."""
    m = re.fullmatch(
        r'\s*<rect x="(\d+)" y="(\d+)" width="(\d+)" height="(\d+)"/>\s*',
        xml,
    )
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))) if m else None


def _parse_folder_date(album_path: str) -> date | None:
    """Extract leading YYYY-MM-DD from paths like '/2007-03-10' or '/2009-08-23 new in Germany'."""
    m = re.match(r"^/(\d{4}-\d{2}-\d{2})", album_path)
    if m:
        try:
            return date.fromisoformat(m.group(1))
        except ValueError:
            pass
    return None


def _parse_filename_date(name: str) -> datetime | None:
    """Parse dates embedded in filenames: '2010-10-25 19.08.44.png', '20110219_165529.jpg', etc."""
    patterns = [
        (r"(\d{4})-(\d{2})-(\d{2})[ _T](\d{2})[.:](\d{2})[.:](\d{2})", True),
        (r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", True),
        (r"(\d{4})-(\d{2})-(\d{2})", False),
    ]
    for pattern, has_time in patterns:
        m = re.search(pattern, name)
        if m:
            try:
                g = m.groups()
                y, mo, d_ = int(g[0]), int(g[1]), int(g[2])
                if has_time:
                    return datetime(y, mo, d_, int(g[3]), int(g[4]), int(g[5]))
                return datetime(y, mo, d_)
            except ValueError:
                continue
    return None


def _parse_digikam_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    raw = raw.split(".")[0].replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_photos(
    db_path: Path = DIGIKAM_DB,
    photo_root: Path | None = None,
) -> list[Photo]:
    """
    Load all available (status=1) photos with their face regions.

    photo_root overrides the AlbumRoots specificPath stored in the DB.
    Pass photo_root=Path('Z:/photos') if you have the NAS mounted, or
    leave as None to fall back to thumbnails for image data.
    """
    conn = _open_db(db_path)
    conn.row_factory = sqlite3.Row

    # Resolve photo root from DB if not overridden
    if photo_root is None:
        row = conn.execute("SELECT specificPath FROM AlbumRoots LIMIT 1").fetchone()
        photo_root = Path(row["specificPath"]) if row else Path("/")

    # Build tag-id → person name map (Tags under pid=4 are People)
    tag_names: dict[int, str] = {
        row["id"]: row["name"]
        for row in conn.execute("SELECT id, name FROM Tags WHERE pid = 4")
    }

    # Load face rects: imageid → [FaceRegion, …]
    faces_by_image: dict[int, list[FaceRegion]] = {}
    for row in conn.execute(
        "SELECT imageid, tagid, value FROM ImageTagProperties WHERE property = 'autodetectedFace'"
    ):
        rect = _parse_rect(row["value"])
        if rect:
            faces_by_image.setdefault(row["imageid"], []).append(
                FaceRegion(
                    tag_id=row["tagid"],
                    person_name=tag_names.get(row["tagid"]),
                    x=rect[0], y=rect[1], width=rect[2], height=rect[3],
                )
            )

    photos: list[Photo] = []
    query = """
        SELECT
            i.id, i.name, i.fileSize, i.uniqueHash,
            a.relativePath  AS album_path,
            ii.creationDate, ii.width, ii.height,
            im.make, im.model
        FROM Images i
        JOIN  Albums          a  ON i.album   = a.id
        LEFT JOIN ImageInformation ii ON i.id = ii.imageid
        LEFT JOIN ImageMetadata    im ON i.id = im.imageid
        WHERE i.status = 1
        ORDER BY a.relativePath, i.name
    """
    for row in conn.execute(query):
        album_path: str = row["album_path"]
        photos.append(Photo(
            id           = row["id"],
            filename     = row["name"],
            album_path   = album_path,
            full_path    = photo_root / album_path.lstrip("/") / row["name"],
            file_size    = row["fileSize"],
            unique_hash  = row["uniqueHash"],
            digikam_date = _parse_digikam_dt(row["creationDate"]),
            folder_date  = _parse_folder_date(album_path),
            filename_date= _parse_filename_date(row["name"]),
            camera_make  = row["make"],
            camera_model = row["model"],
            width        = row["width"],
            height       = row["height"],
            faces        = faces_by_image.get(row["id"], []),
        ))

    conn.close()
    return photos


def load_identities(db_path: Path = RECOGNITION_DB) -> list[Identity]:
    """
    Load known people + their 128-dim face embeddings from recognition.db.

    FaceMatrices.embedding is a raw BLOB of float32 values (512 bytes = 128 floats).
    """
    conn = _open_db(db_path)
    conn.row_factory = sqlite3.Row

    # Prefer fullName; fall back to name when fullName is absent
    names: dict[int, str] = {}
    for row in conn.execute(
        "SELECT id, value, attribute FROM IdentityAttributes"
        " WHERE attribute IN ('fullName', 'name')"
    ):
        if row["attribute"] == "fullName" or row["id"] not in names:
            names[row["id"]] = row["value"]

    embs_by_id: dict[int, list[np.ndarray]] = {}
    for row in conn.execute("SELECT identity, embedding FROM FaceMatrices"):
        blob: bytes = row["embedding"]
        n = len(blob) // 4   # float32 = 4 bytes
        vec = np.frombuffer(blob, dtype=np.float32, count=n).copy()
        embs_by_id.setdefault(row["identity"], []).append(vec)

    identities = [
        Identity(id=id_, name=name, embeddings=embs_by_id.get(id_, []))
        for id_, name in sorted(names.items())
    ]
    conn.close()
    return identities


def load_thumbnail(
    unique_hash: str,
    file_size: int,
    db_path: Path = THUMBNAILS_DB,
    conn: sqlite3.Connection | None = None,
) -> bytes | None:
    """
    Retrieve a JPEG thumbnail BLOB from thumbnails-digikam.db.

    DigiKam stores PGF or JPEG2000 thumbnails; for JPEG (type=2) the
    data is a raw JPEG byte string ready for Pillow.
    Returns None if not found.

    Pass an existing `conn` to reuse a connection across many calls;
    otherwise the module-level cached connection for db_path is used.
    """
    c = conn if conn is not None else _thumbnail_conn(db_path)
    row = c.execute(
        """SELECT t.data, t.type
           FROM Thumbnails t
           JOIN UniqueHashes u ON u.thumbId = t.id
           WHERE u.uniqueHash = ? AND u.fileSize = ?
           LIMIT 1""",
        (unique_hash, file_size),
    ).fetchone()
    if row is None:
        return None
    # type 2 = JPEG, type 1 = PGF (not easily decodable without libpgf)
    return bytes(row[0]) if row[1] == 2 else None


def open_photo_image(photo: Photo, thumbnails_db: Path = THUMBNAILS_DB):
    """
    Return a PIL Image for a photo, preferring full-res but falling back
    to the local thumbnail BLOB when the network share is unavailable.
    """
    from PIL import Image
    import io

    if photo.full_path.exists():
        try:
            return Image.open(photo.full_path)
        except Exception:
            pass

    if photo.unique_hash and photo.file_size:
        blob = load_thumbnail(photo.unique_hash, photo.file_size, thumbnails_db)
        if blob:
            try:
                return Image.open(io.BytesIO(blob))
            except Exception:
                pass

    return None
