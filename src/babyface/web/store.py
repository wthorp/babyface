"""
Data access layer for the labeling web UI.

Loads predictions6.json (or a configurable file), maintains a corrections
SQLite DB, builds a thumbnail index against thumbnails-digikam.db, and
clusters Unknown faces via HDBSCAN using the DINOv2 embedding cache.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import re

import numpy as np


@dataclass
class DigiKamFace:
    photo_id: int
    tag_id: int
    name: str
    filename: str
    album_path: str
    bbox: list[int]   # [x, y, w, h]

    @property
    def date_approx(self) -> str:
        m = re.match(r"/(\d{4}-\d{2}-\d{2})", self.album_path)
        return m.group(1) if m else self.album_path.lstrip("/")[:10]


@dataclass
class FacePred:
    photo_id: int
    face_idx: int
    filename: str
    album_path: str
    bbox: list[int]        # [x, y, w, h]
    original_tag: str
    predicted_identity: Optional[str]
    score: float
    ambiguous: bool
    flags: list[str]
    cluster_id: Optional[int] = None
    corrected: bool = False

    @property
    def key(self) -> tuple[int, int]:
        return (self.photo_id, self.face_idx)

    @property
    def is_unknown(self) -> bool:
        pi = self.predicted_identity
        return not pi or pi.strip().lower() in ("unknown", "")

    @property
    def date_approx(self) -> str:
        """Best-effort date from album_path like /2009-08-23 ..."""
        m = re.match(r"/(\d{4}-\d{2}-\d{2})", self.album_path)
        return m.group(1) if m else self.album_path.lstrip("/")[:10]


_PALETTE = [
    "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
    "#1abc9c","#e67e22","#e91e63","#00bcd4","#8bc34a",
]

def _identity_color(name: str) -> str:
    if not name:
        return "#95a5a6"
    return _PALETTE[hash(name) % len(_PALETTE)]


class DataStore:
    def __init__(
        self,
        repo_dir: Path,
        predictions_file: str = "predictions6.json",
        photo_root: Optional[Path] = None,
    ):
        self.repo_dir = repo_dir
        self.photo_root = photo_root
        self._lock = threading.Lock()

        self.predictions: dict[tuple[int, int], FacePred] = {}
        self._thumb_index: dict[tuple, int] = {}   # (filename_upper, x,y,w,h) → thumb_id
        self._thumb_conn: Optional[sqlite3.Connection] = None

        self.retrain_log: list[str] = []
        self.retrain_running: bool = False
        self.retrain_exit_code: Optional[int] = None

        self._load(predictions_file)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, predictions_file: str) -> None:
        path = self.repo_dir / predictions_file
        with open(path) as f:
            raw = json.load(f)
        for rec in raw:
            p = FacePred(
                photo_id=rec["photo_id"],
                face_idx=rec["face_idx"],
                filename=rec["filename"],
                album_path=rec["album_path"],
                bbox=rec["bbox"],
                original_tag=rec.get("original_tag", ""),
                predicted_identity=rec.get("predicted_identity"),
                score=rec.get("score", 0.0),
                ambiguous=bool(rec.get("ambiguous", False)),
                flags=rec.get("flags", []),
            )
            self.predictions[p.key] = p

        self._init_corrections_db()
        self._apply_corrections()
        self._build_thumb_index()
        self._cluster_unknowns()

    def _init_corrections_db(self) -> None:
        con = self._corrections_connect()
        con.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                photo_id    INTEGER NOT NULL,
                face_idx    INTEGER NOT NULL,
                identity    TEXT,
                corrected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (photo_id, face_idx)
            )
        """)
        con.commit()
        con.close()

    def _corrections_connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.repo_dir / "corrections.db"))

    def _apply_corrections(self) -> None:
        con = self._corrections_connect()
        for row in con.execute("SELECT photo_id, face_idx, identity FROM corrections"):
            pid, fidx, ident = row
            key = (pid, fidx)
            if key in self.predictions:
                self.predictions[key].predicted_identity = ident or None
                self.predictions[key].ambiguous = False
                self.predictions[key].corrected = True
        con.close()

    # ------------------------------------------------------------------
    # Corrections
    # ------------------------------------------------------------------

    def save_correction(self, photo_id: int, face_idx: int, identity: str) -> None:
        ident = identity.strip() if identity else None
        if ident and ident.lower() == "unknown":
            ident = None
        con = self._corrections_connect()
        con.execute(
            "INSERT OR REPLACE INTO corrections (photo_id, face_idx, identity) VALUES (?,?,?)",
            (photo_id, face_idx, ident),
        )
        con.commit()
        con.close()
        with self._lock:
            key = (photo_id, face_idx)
            if key in self.predictions:
                self.predictions[key].predicted_identity = ident
                self.predictions[key].ambiguous = False
                self.predictions[key].corrected = True

    def get_corrections_count(self) -> int:
        con = self._corrections_connect()
        n = con.execute("SELECT count(*) FROM corrections").fetchone()[0]
        con.close()
        return n

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        preds = list(self.predictions.values())
        assigned   = sum(1 for p in preds if not p.is_unknown and not p.ambiguous)
        ambiguous  = sum(1 for p in preds if p.ambiguous)
        unknown    = sum(1 for p in preds if p.is_unknown and not p.ambiguous)
        corrected  = sum(1 for p in preds if p.corrected)
        identities = sorted({p.predicted_identity for p in preds
                              if not p.is_unknown and not p.ambiguous})
        clusters   = sorted({p.cluster_id for p in preds
                              if p.is_unknown and p.cluster_id is not None and p.cluster_id >= 0})
        return dict(
            total=len(preds),
            assigned=assigned,
            ambiguous=ambiguous,
            unknown=unknown,
            corrected=corrected,
            n_identities=len(identities),
            n_clusters=len(clusters),
        )

    # ------------------------------------------------------------------
    # Identity grids
    # ------------------------------------------------------------------

    def identity_names(self) -> list[tuple[str, int, str]]:
        """Returns [(name, count, color)] sorted by count desc."""
        counts: dict[str, int] = {}
        for p in self.predictions.values():
            if not p.is_unknown and not p.ambiguous:
                counts[p.predicted_identity] = counts.get(p.predicted_identity, 0) + 1
        return sorted(
            [(name, cnt, _identity_color(name)) for name, cnt in counts.items()],
            key=lambda x: -x[1],
        )

    def faces_for_identity(
        self, name: str, page: int = 0, page_size: int = 50
    ) -> tuple[list[FacePred], int]:
        faces = sorted(
            [p for p in self.predictions.values() if p.predicted_identity == name],
            key=lambda p: (-p.score, p.album_path),
        )
        total = len(faces)
        start = page * page_size
        return faces[start : start + page_size], total

    # ------------------------------------------------------------------
    # Ambiguous queue
    # ------------------------------------------------------------------

    def ambiguous_faces(
        self, page: int = 0, page_size: int = 50
    ) -> tuple[list[FacePred], int]:
        faces = sorted(
            [p for p in self.predictions.values() if p.ambiguous],
            key=lambda p: p.score,   # show least confident first
        )
        total = len(faces)
        start = page * page_size
        return faces[start : start + page_size], total

    # ------------------------------------------------------------------
    # Unknown clusters
    # ------------------------------------------------------------------

    def clusters(self, page: int = 0, page_size: int = 30) -> tuple[list[dict], int]:
        """Returns list of cluster dicts {cluster_id, count, representative_face}."""
        by_cluster: dict[int, list[FacePred]] = {}
        for p in self.predictions.values():
            if p.is_unknown:
                cid = p.cluster_id if p.cluster_id is not None else -1
                by_cluster.setdefault(cid, []).append(p)

        result = []
        for cid, faces in sorted(by_cluster.items(), key=lambda kv: -len(kv[1])):
            if cid == -1:
                continue   # noise faces shown separately
            rep = max(faces, key=lambda p: p.score)
            result.append({"cluster_id": cid, "count": len(faces), "representative": rep})

        noise = by_cluster.get(-1, [])
        if noise:
            result.append({"cluster_id": -1, "count": len(noise), "representative": noise[0]})

        total = len(result)
        start = page * page_size
        return result[start : start + page_size], total

    def faces_in_cluster(self, cluster_id: int) -> list[FacePred]:
        return [p for p in self.predictions.values()
                if p.is_unknown and p.cluster_id == cluster_id]

    def assign_cluster(self, cluster_id: int, identity: str) -> int:
        """Assign all faces in a cluster to the given identity. Returns count."""
        faces = self.faces_in_cluster(cluster_id)
        for p in faces:
            self.save_correction(p.photo_id, p.face_idx, identity)
        return len(faces)

    # ------------------------------------------------------------------
    # Thumbnails / face crops
    # ------------------------------------------------------------------

    def _build_thumb_index(self) -> None:
        thumb_path = self.repo_dir / "thumbnails-digikam.db"
        if not thumb_path.exists():
            return
        self._thumb_conn = sqlite3.connect(str(thumb_path), check_same_thread=False)
        # Parse: detail:///path?rect=x,y-WxH
        _rect_re = re.compile(r"\?rect=(\d+),(\d+)-(\d+)x(\d+)$")
        for row in self._thumb_conn.execute(
            "SELECT identifier, thumbId FROM CustomIdentifiers"
        ):
            ident, tid = row
            m = _rect_re.search(ident)
            if not m:
                continue
            fname = ident.split("/")[-1].split("?")[0].upper()
            x, y, w, h = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            self._thumb_index[(fname, x, y, w, h)] = tid

    def get_face_crop_bytes(
        self, photo_id: int, face_idx: int
    ) -> Optional[tuple[bytes, str]]:
        """Returns (image_bytes, mime_type) or None."""
        pred = self.predictions.get((photo_id, face_idx))
        if pred is None:
            return None

        # Try real photo first (if photo_root is set)
        if self.photo_root:
            try:
                from PIL import Image, ImageOps
                import io
                photo_path = self.photo_root / pred.album_path.lstrip("/") / pred.filename
                if photo_path.exists():
                    # DigiKam stores bbox in display-space (after EXIF rotation)
                    img = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
                    x, y, w, h = pred.bbox
                    pad = int(max(w, h) * 0.2)
                    crop = img.crop((max(0, x-pad), max(0, y-pad),
                                     min(img.width, x+w+pad), min(img.height, y+h+pad)))
                    buf = io.BytesIO()
                    crop.save(buf, format="JPEG", quality=85)
                    return buf.getvalue(), "image/jpeg"
            except Exception:
                pass

        return None  # no crop available

    def identity_color(self, name: str) -> str:
        return _identity_color(name or "")

    # ------------------------------------------------------------------
    # DigiKam ground-truth faces
    # ------------------------------------------------------------------

    def _load_digikam_faces(self) -> None:
        """Lazy-load tagged faces from digikam4.db into self._digikam_faces."""
        db_path = self.repo_dir / "digikam4.db"
        if not db_path.exists():
            self._digikam_faces: list[DigiKamFace] = []
            return
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        tag_names = {
            row["id"]: row["name"]
            for row in con.execute("SELECT id, name FROM Tags WHERE pid = 4")
        }
        import re as _re
        _rect_re = _re.compile(
            r'<rect x="(\d+)" y="(\d+)" width="(\d+)" height="(\d+)"/>'
        )
        faces: list[DigiKamFace] = []
        for row in con.execute("""
            SELECT itp.imageid, itp.tagid, itp.value,
                   i.name AS filename, a.relativePath AS album_path
            FROM ImageTagProperties itp
            JOIN Images  i ON i.id = itp.imageid
            JOIN Albums  a ON a.id = i.album
            WHERE itp.property = 'autodetectedFace'
              AND itp.tagid IN ({})
        """.format(",".join(str(k) for k in tag_names) if tag_names else "NULL")):
            m = _rect_re.search(row["value"])
            if not m:
                continue
            x, y, w, h = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            person = tag_names.get(row["tagid"], "Unknown")
            faces.append(DigiKamFace(
                photo_id=row["imageid"],
                tag_id=row["tagid"],
                name=person,
                filename=row["filename"],
                album_path=row["album_path"],
                bbox=[x, y, w, h],
            ))
        con.close()
        self._digikam_faces = faces

    @property
    def digikam_faces(self) -> list[DigiKamFace]:
        if not hasattr(self, "_digikam_faces"):
            self._load_digikam_faces()
        return self._digikam_faces

    def digikam_identity_names(self) -> list[tuple[str, int, str]]:
        counts: dict[str, int] = {}
        for f in self.digikam_faces:
            counts[f.name] = counts.get(f.name, 0) + 1
        return sorted(
            [(n, c, _identity_color(n)) for n, c in counts.items()],
            key=lambda x: -x[1],
        )

    def digikam_faces_for_identity(
        self, name: str, page: int = 0, page_size: int = 50
    ) -> tuple[list[DigiKamFace], int]:
        faces = sorted(
            [f for f in self.digikam_faces if f.name == name],
            key=lambda f: f.date_approx,
        )
        total = len(faces)
        start = page * page_size
        return faces[start : start + page_size], total

    def get_digikam_crop_bytes(
        self, photo_id: int, tag_id: int
    ) -> Optional[tuple[bytes, str]]:
        face = next(
            (f for f in self.digikam_faces if f.photo_id == photo_id and f.tag_id == tag_id),
            None,
        )
        if face is None or not self.photo_root:
            return None
        try:
            from PIL import Image, ImageOps
            import io
            photo_path = self.photo_root / face.album_path.lstrip("/") / face.filename
            if not photo_path.exists():
                return None
            # DigiKam stores bbox in display-space (after EXIF rotation)
            img = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
            x, y, w, h = face.bbox
            pad = int(max(w, h) * 0.2)
            crop = img.crop((max(0, x - pad), max(0, y - pad),
                             min(img.width, x + w + pad), min(img.height, y + h + pad)))
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=85)
            return buf.getvalue(), "image/jpeg"
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _cluster_unknowns(self) -> None:
        unknown_keys = [
            k for k, p in self.predictions.items() if p.is_unknown
        ]
        if not unknown_keys:
            return

        emb_file = self.repo_dir / "embeddings" / "emb_dinov2_vit_base_patch14_dinov2.lvd142m.pt"
        if not emb_file.exists():
            return

        try:
            import torch
            blob = torch.load(emb_file, weights_only=True)
            cache = blob["embeddings"] if isinstance(blob, dict) and "embeddings" in blob else blob
            valid = [(k, cache[k].numpy().astype(np.float32))
                     for k in unknown_keys if k in cache]
            if len(valid) < 10:
                return

            mat = np.stack([v for _, v in valid])
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat = mat / np.where(norms > 0, norms, 1)

            # Cosine distance = 1 - cosine_sim; use euclidean on L2-normed vecs (equiv)
            from sklearn.cluster import HDBSCAN
            clust = HDBSCAN(min_cluster_size=5, min_samples=3, metric="euclidean",
                            cluster_selection_epsilon=0.3)
            labels = clust.fit_predict(mat)

            for (k, _), label in zip(valid, labels):
                self.predictions[k].cluster_id = int(label)
        except Exception:
            pass   # clustering is best-effort; UI still works without it
