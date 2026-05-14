"""
Identity clustering: assign infant photos to individuals.

Pipeline:
  1. Build [N, D+1] feature matrix: D-dim L2-normalised DINOv2 embedding
     + 1 scaled temporal feature (corrected date in days / temporal_sigma).
  2. When identity_centroids are provided, inject them as labeled anchor rows:
     - Semi-supervised path (requires the standalone `hdbscan` package):
       label array passed directly to hdbscan.HDBSCAN.fit — labeled rows act
       as hard must-link constraints so known identities seed their own clusters.
     - Pre-seeding fallback (sklearn HDBSCAN, always available):
       centroid rows appended to the feature matrix as extra unlabeled points.
       Because centroids sit at the dense core of each identity's faces they
       pull nearby points into the same cluster; seed cluster assignments are
       read back to name clusters without post-hoc cosine matching.
  3. Match each cluster to a name: DigiKam majority-vote tags first, then
     seed-point cluster assignments for any still-unnamed clusters.

Note on embedding spaces:
  DigiKam's 128-dim embeddings (OpenCV DNN) and our 768-dim DINOv2 embeddings
  live in different spaces and cannot be compared directly.  Identity centroids
  passed to this module must already be in DINOv2 space (produced by
  build_identity_dino_centroids).
"""
from __future__ import annotations
import re
import sqlite3
import warnings
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import HDBSCAN

from ..db.models import Photo, FaceRegion, Identity
from ..db.digikam import open_photo_image
from ..embeddings.kernels import pairwise_cosine, fused_similarity
from ..embeddings.extract import crop_face

# Semi-supervised HDBSCAN from the standalone `hdbscan` package (McInnes et al.)
# sklearn's HDBSCAN does not support semi-supervised mode.
# Install: uv add hdbscan  (requires a C compiler; not available in all envs)
try:
    import hdbscan as _hdbscan_pkg  # type: ignore[import-untyped]
    _HDBSCAN_PKG: ModuleType | None = _hdbscan_pkg
except ImportError:
    _HDBSCAN_PKG = None

_EPOCH = datetime(2000, 1, 1)


@dataclass
class ClusteredFace:
    photo_id:       int
    photo_path:     str
    face_idx:       int
    cluster_id:     int    # -1 = noise / unassigned
    person_name:    str | None   # from DigiKam tag if present
    predicted_name: str | None   # from cluster→identity matching
    confidence:     float        # 0–1; 0 for noise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_to_days(dt: datetime | None) -> float:
    if dt is None:
        return 0.0
    return (dt - _EPOCH).total_seconds() / 86400.0


def _build_feature_matrix(
    embeddings: dict[tuple[int, int], torch.Tensor],
    photo_map:  dict[int, Photo],
    temporal_weight: float = 0.15,
    temporal_sigma:  float = 30.0,
) -> tuple[np.ndarray, list[tuple[int, int]], float]:
    """
    Returns (features [N, D+1], ordered_keys, days_mean).

    The final column is (days - days_mean) / temporal_sigma * temporal_weight.
    days_mean is returned so centroid seeds can be normalised into the same
    temporal space without recomputing the mean over the full dataset.
    """
    keys = sorted(embeddings.keys())
    embs = torch.stack([embeddings[k] for k in keys])        # [N, D]
    embs = F.normalize(embs.float(), dim=-1)

    days = torch.tensor(
        [_date_to_days(photo_map[pid].corrected_date if pid in photo_map else None)
         for pid, _ in keys],
        dtype=torch.float32,
    )
    days_mean = float(days.mean().item())
    if temporal_sigma > 0:
        days_scaled = (days - days_mean) / temporal_sigma
        features = torch.cat([embs, days_scaled.unsqueeze(1) * temporal_weight], dim=1)
    else:
        features = embs  # pure visual, no temporal component
    return features.cpu().numpy(), keys, days_mean


def _centroid_seed_rows(
    identity_centroids: dict[int, torch.Tensor],
    known_identities:   list[Identity],
    photo_map:          dict[int, Photo],
    days_mean:          float,
    temporal_sigma:     float,
    temporal_weight:    float,
) -> tuple[np.ndarray, list[int]]:
    """
    Build one feature-matrix row per identity centroid, normalised into the
    same space as _build_feature_matrix.

    Temporal component: mean corrected_date of all labeled faces for that
    identity (falls back to days_mean when no labeled faces are in photo_map).

    Returns (seed_features [K, D+1], seed_identity_ids [K]).
    """
    ident_name_map = {i.id: i.name for i in known_identities}

    # Accumulate days per person name from labeled faces in the working set
    name_to_days: dict[str, list[float]] = {}
    for photo in photo_map.values():
        d = _date_to_days(photo.corrected_date)
        for face in photo.faces:
            if face.person_name:
                name_to_days.setdefault(face.person_name, []).append(d)

    rows: list[np.ndarray] = []
    ids:  list[int]        = []
    for ident_id, centroid in sorted(identity_centroids.items()):
        name      = ident_name_map.get(ident_id, "")
        days_list = name_to_days.get(name, [])
        mean_days = float(np.mean(days_list)) if days_list else days_mean
        emb = F.normalize(centroid.float().unsqueeze(0), dim=-1).squeeze(0).cpu().numpy()
        if temporal_sigma > 0:
            t_feat = (mean_days - days_mean) / temporal_sigma * temporal_weight
            rows.append(np.append(emb, t_feat))
        else:
            rows.append(emb)
        ids.append(ident_id)

    if not rows:
        return np.empty((0, 0)), []
    return np.stack(rows), ids


def build_identity_dino_centroids(
    identities: list[Identity],
    extractor,                   # EmbeddingExtractor
    digikam_db,                  # Path to digikam4.db
    thumbnails_db,               # Path to thumbnails-digikam.db (or None)
    max_per_identity: int = 20,
    photo_root: "Path | None" = None,
) -> dict[int, torch.Tensor]:
    """
    For each known identity, retrieve up to max_per_identity face crops
    from the database and embed them with DINOv2 to build a centroid in the
    DINOv2 space.

    photo_root: if provided, constructs real file paths so full-res images are
    used instead of (or before) falling back to thumbnails.

    Returns {identity_id: dino_centroid_768dim}.
    """
    from ..db.digikam import _open_db
    conn = _open_db(Path(digikam_db))
    conn.row_factory = sqlite3.Row

    # Resolve AlbumRoot from DB when photo_root not supplied
    if photo_root is None:
        row0 = conn.execute("SELECT specificPath FROM AlbumRoots LIMIT 1").fetchone()
        photo_root = Path(row0["specificPath"]) if row0 else Path("/")

    identity_centroids: dict[int, torch.Tensor] = {}

    for ident in identities:
        rows = conn.execute(
            """SELECT itp.imageid, itp.value, i.fileSize, i.uniqueHash,
                      a.relativePath, i.name
               FROM ImageTagProperties itp
               JOIN Tags t ON itp.tagid = t.id
               JOIN Images i ON itp.imageid = i.id
               JOIN Albums a ON i.album = a.id
               WHERE itp.property = 'autodetectedFace'
                 AND t.name = ?
               LIMIT ?""",
            (ident.name, max_per_identity),
        ).fetchall()

        crops: list[Image.Image] = []
        for row in rows:
            m = re.fullmatch(
                r'\s*<rect x="(\d+)" y="(\d+)" width="(\d+)" height="(\d+)"/>\s*',
                row["value"],
            )
            if not m:
                continue
            rect = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
            album_rel: str = row["relativePath"]
            full = photo_root / album_rel.lstrip("/") / row["name"]
            dummy = Photo(
                id=row["imageid"], filename=row["name"],
                album_path=album_rel,
                full_path=full,
                file_size=row["fileSize"], unique_hash=row["uniqueHash"],
            )
            img = open_photo_image(dummy, thumbnails_db)
            if img is None:
                continue
            crop = crop_face(img.convert("RGB"), rect)
            if crop is None:
                continue
            crops.append(crop)

        if not crops:
            continue

        with torch.no_grad():
            batch_emb = extractor.embed_batch(crops)
            centroid = F.normalize(batch_emb.mean(0), dim=-1)
        identity_centroids[ident.id] = centroid

    conn.close()
    return identity_centroids


# ---------------------------------------------------------------------------
# Nearest-centroid assignment (no HDBSCAN)
# ---------------------------------------------------------------------------

def run_nearest_centroid_pipeline(
    photos:             list[Photo],
    embeddings:         dict[tuple[int, int], torch.Tensor],
    known_identities:   list[Identity],
    identity_centroids: dict[int, torch.Tensor],
    min_similarity:     float = 0.5,
) -> list[ClusteredFace]:
    """
    Assign each face to the nearest identity centroid by cosine similarity.
    Faces below min_similarity to every centroid are left as noise (cluster -1).
    Returns 1 cluster per identity that has at least one assigned face.
    """
    if not embeddings or not identity_centroids:
        return []

    photo_map   = {p.id: p for p in photos}
    keys        = sorted(embeddings.keys())
    embs        = F.normalize(torch.stack([embeddings[k] for k in keys]).float(), dim=-1)

    ident_ids   = sorted(identity_centroids.keys())
    centroids   = F.normalize(torch.stack([identity_centroids[i] for i in ident_ids]).float(), dim=-1)
    ident_name  = {i.id: i.name for i in known_identities}

    # sim [N, K]
    sim = torch.mm(embs.cpu(), centroids.cpu().T)
    best_sim, best_idx = sim.max(dim=1)

    # identity_id → dense cluster_id
    used_ids = sorted({ident_ids[int(best_idx[n])] for n in range(len(keys)) if float(best_sim[n]) >= min_similarity})
    id_to_cid = {ident_id: cid for cid, ident_id in enumerate(used_ids)}

    results: list[ClusteredFace] = []
    for n, (photo_id, face_idx) in enumerate(keys):
        photo = photo_map.get(photo_id)
        if photo is None or face_idx >= len(photo.faces):
            continue
        face   = photo.faces[face_idx]
        sim_v  = float(best_sim[n])
        if sim_v >= min_similarity:
            ident_id = ident_ids[int(best_idx[n])]
            cid      = id_to_cid[ident_id]
            name     = ident_name.get(ident_id)
        else:
            cid, name, sim_v = -1, None, 0.0
        results.append(ClusteredFace(
            photo_id       = photo_id,
            photo_path     = str(photo.full_path),
            face_idx       = face_idx,
            cluster_id     = cid,
            person_name    = face.person_name,
            predicted_name = name,
            confidence     = sim_v,
        ))
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_clustering_pipeline(
    photos:             list[Photo],
    embeddings:         dict[tuple[int, int], torch.Tensor],
    known_identities:   list[Identity],
    identity_centroids: dict[int, torch.Tensor] | None = None,
    temporal_sigma:     float = 30.0,
    temporal_weight:    float = 0.15,
    min_cluster_size:   int   = 3,
    min_samples:        int   = 2,
    pca_dims:           int   = 0,
    cluster_epsilon:    float = 0.0,
) -> list[ClusteredFace]:
    """
    Full clustering pipeline.

    identity_centroids — optional {id: dino_centroid} from
    build_identity_dino_centroids.  When supplied, centroids are injected
    into the feature matrix as labeled anchors so HDBSCAN can use them as
    seeds for known-identity clusters.  Naming priority:
      1. DigiKam majority-vote tag (direct human labels)
      2. Seed-point cluster assignment (centroid landed in that cluster)

    Returns a ClusteredFace per embedded face region.
    """
    if len(embeddings) < 2:
        return []

    photo_map  = {p.id: p for p in photos}
    features, keys, days_mean = _build_feature_matrix(
        embeddings, photo_map, temporal_weight, temporal_sigma,
    )
    n_data = len(keys)

    # ------------------------------------------------------------------ #
    # Optional PCA reduction (dramatically speeds up HDBSCAN in high-D)  #
    # ------------------------------------------------------------------ #
    pca_model = None
    if pca_dims > 0 and pca_dims < features.shape[1]:
        from sklearn.decomposition import PCA
        pca_model = PCA(n_components=pca_dims, random_state=42)
        features = pca_model.fit_transform(features)

    # ------------------------------------------------------------------ #
    # HDBSCAN — with or without identity anchor seeds                     #
    # ------------------------------------------------------------------ #
    cluster_from_seed: dict[int, str] = {}   # cluster_id → identity name
    ident_name_map = {i.id: i.name for i in known_identities}

    if identity_centroids:
        seed_features, seed_ids = _centroid_seed_rows(
            identity_centroids, known_identities, photo_map,
            days_mean, temporal_sigma, temporal_weight,
        )
        # Project seeds into PCA space if PCA was applied to data
        if pca_model is not None and seed_features.shape[0] > 0:
            seed_features = pca_model.transform(seed_features)
    else:
        seed_features, seed_ids = np.empty((0, 0)), []

    if seed_ids:
        combined = np.vstack([features, seed_features])  # [N+K, D+1]

        if _HDBSCAN_PKG is not None:
            # Semi-supervised: labeled rows (seeds) act as must-link constraints.
            # Label values: identity_id for seeds, -1 for unlabeled data.
            y = np.full(len(combined), -1, dtype=np.intp)
            # Re-map identity IDs to a compact 0-based label space because
            # hdbscan requires labels to be small non-negative integers.
            for i in range(len(seed_ids)):
                y[n_data + i] = i

            ss_clusterer = _HDBSCAN_PKG.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
                cluster_selection_epsilon=cluster_epsilon,
            )
            try:
                all_labels: np.ndarray = ss_clusterer.fit_predict(combined, y)
            except TypeError:
                # Installed version doesn't accept y — fall through to sklearn
                all_labels = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric="euclidean",
                    cluster_selection_method="eom",
                    cluster_selection_epsilon=cluster_epsilon,
                ).fit_predict(combined)
        else:
            # Pre-seeding fallback: seeds are extra unlabeled data points.
            # Their position at each identity's centroid pulls nearby faces
            # into the same cluster; we read back their cluster assignment
            # below to name clusters without a separate cosine-matching pass.
            all_labels = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
                cluster_selection_epsilon=cluster_epsilon,
            ).fit_predict(combined)

        labels     = all_labels[:n_data]    # labels for real face rows
        seed_labels = all_labels[n_data:]   # labels for seed rows

        # Map cluster_id → identity name via seed assignments.
        # If two centroids land in the same cluster the assignment is
        # ambiguous — discard it and let DigiKam tags decide instead.
        collided: set[int] = set()
        for ident_id, cid in zip(seed_ids, seed_labels.tolist()):
            if cid >= 0:
                if cid in cluster_from_seed:
                    collided.add(cid)
                    warnings.warn(
                        f"Two identity centroids landed in cluster {cid} "
                        f"({cluster_from_seed[cid]!r} and "
                        f"{ident_name_map.get(ident_id, '')!r}); "
                        "skipping seed-based name for that cluster.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    cluster_from_seed[cid] = ident_name_map.get(ident_id, "")
        for cid in collided:
            cluster_from_seed.pop(cid, None)
    else:
        labels = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            cluster_selection_epsilon=cluster_epsilon,
        ).fit_predict(features)

    # ------------------------------------------------------------------ #
    # Cluster naming: DigiKam tags > seed assignments                     #
    # ------------------------------------------------------------------ #

    # 1. Majority-vote DigiKam tag per cluster
    cluster_tag: dict[int, list[str]] = {}
    for idx, (photo_id, face_idx) in enumerate(keys):
        cid = int(labels[idx])
        if cid == -1:
            continue
        photo = photo_map.get(photo_id)
        if photo and face_idx < len(photo.faces):
            name = photo.faces[face_idx].person_name
            if name:
                cluster_tag.setdefault(cid, []).append(name)

    cluster_name: dict[int, str | None] = {
        cid: Counter(names).most_common(1)[0][0]
        for cid, names in cluster_tag.items()
        if names
    }

    # 2. Fill gaps from seed-point cluster assignments
    for cid, name in cluster_from_seed.items():
        if cid not in cluster_name and name:
            cluster_name[cid] = name

    # ------------------------------------------------------------------ #
    # Assemble results (real faces only — seeds are not returned)         #
    # ------------------------------------------------------------------ #
    results: list[ClusteredFace] = []
    for idx, (photo_id, face_idx) in enumerate(keys):
        photo = photo_map.get(photo_id)
        if photo is None or face_idx >= len(photo.faces):
            continue
        face = photo.faces[face_idx]
        cid  = int(labels[idx])
        results.append(ClusteredFace(
            photo_id       = photo_id,
            photo_path     = str(photo.full_path),
            face_idx       = face_idx,
            cluster_id     = cid,
            person_name    = face.person_name,
            predicted_name = cluster_name.get(cid) if cid >= 0 else None,
            confidence     = 1.0 if cid >= 0 else 0.0,
        ))

    return results
