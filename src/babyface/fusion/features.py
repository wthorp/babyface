"""
Per-(face, candidate-identity) feature extraction for the fusion model.

Each row scored by the GBM is a *pair*: one detected face and one candidate
identity.  The label (training only) is 1 if that identity is the face's true
DigiKam tag, else 0.  At inference we emit the same rows for untagged faces
(label NaN) and let the model rank candidates.

Features per row
----------------
  sim_<backbone>   cosine(face embedding, identity's centroid) for each model
                   (DINOv2, ArcFace, SigLIP scene).  Missing when the backbone
                   produced no embedding for that face (e.g. ArcFace found no
                   face) — left as NaN, which LightGBM consumes natively.  The
                   tree learns the "babyness gate" implicitly: when arcface is
                   missing/low it leans on dino, no explicit age model needed.
  time_prox        exp(-gap² / 2σ²) to the identity's nearest known photo date
  in_active_range  1 if the photo date falls within the identity's tagged span
  camera_p         P(this camera | identity) from the identity's tagged photos
  camera_seen      1 if the identity was ever tagged on this camera model
  geo_km           min great-circle km to the identity's geotagged photos
                   (NaN when either side lacks GPS — only ~10% have it)
  id_log_prior     log count of the identity's tagged faces (base rate)
  album_person_match  1 if the candidate's first-name token appears as a whole word
                   in this photo's album path (e.g. "lisa" in "Lisa's 30th Bday").
                   Conservative: skips kinship titles (grandpa/uncle/…), requires
                   ≥3 chars. ~7% of photos are in person-named albums.

Centroids, date spans, camera histograms and GPS points are all built from the
*train* split only (passed in via `train_records`) so evaluation stays honest.

Backbone embedding maps are keyed (photo_id, face_idx) for face/patch models
and (photo_id, -1) for scene models (one per image, shared by its faces); the
`scene_backbones` set tells the builder which use the (photo_id, -1) key.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
# torch not imported: only used in string type hints (PEP 563) and via .numpy()
# on tensors passed in by the caller. Keeping this module torch-free lets the
# CLI import lightgbm before torch to avoid the macOS OpenMP segfault.

_EPOCH = datetime(2000, 1, 1)
_UNKNOWN_TAGS = {"unknown", "unidentified", ""}   # treated as the reject class
N_SUBCENTROIDS = 3   # k-means clusters per identity per face backbone

# Album-name person matching
_TITLE_PREFIXES = frozenset({"grandpa", "grandma", "uncle", "aunt", "great", "baby"})
_RE_NONWORD = re.compile(r'[^a-z0-9]+')


def _album_search_token(name: str) -> str:
    """First meaningful word (>=3 chars, not a kinship title) from an identity name."""
    words = [w for w in name.lower().split() if len(w) >= 3]
    for w in words:
        if w not in _TITLE_PREFIXES:
            return w
    return words[-1] if words else ""


def _album_word_set(album_path: str) -> frozenset[str]:
    """Lowercase word tokens from an album path, length >= 3."""
    return frozenset(w for w in _RE_NONWORD.split(album_path.lower()) if len(w) >= 3)


def _days(dt: datetime | None) -> float | None:
    return None if dt is None else (dt - _EPOCH).total_seconds() / 86400.0


def _photo_days(photo) -> float | None:
    return _days(photo.corrected_date or photo.digikam_date)


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    (lat1, lon1), (lat2, lon2) = a, b
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


# ---------------------------------------------------------------------------
# Per-identity model (built from the train split only)
# ---------------------------------------------------------------------------

@dataclass
class IdentityModel:
    name: str
    centroids: dict[str, np.ndarray] = field(default_factory=dict)  # backbone_id → unit vec
    days: np.ndarray = field(default_factory=lambda: np.empty(0))   # train photo days
    camera_counts: dict[str, int] = field(default_factory=dict)
    n_camera_obs: int = 0
    gps: list[tuple[float, float]] = field(default_factory=list)
    albums: set[str] = field(default_factory=set)                   # train album_paths
    album_face_counts: dict[str, int] = field(default_factory=dict) # album_path → face count
    scene_embs: dict[str, np.ndarray] = field(default_factory=dict)    # backbone_id → [N, D] for scene models
    sub_centroids: dict[str, np.ndarray] = field(default_factory=dict) # backbone_id → [k, D] k-means clusters
    n_faces: int = 0


def build_identity_models(
    train_records: list[tuple[int, int, str]],          # (photo_id, face_idx, person_name)
    embeddings_by_backbone: dict[str, dict[tuple[int, int], torch.Tensor]],
    scene_backbones: set[str],
    photo_map: dict[int, "object"],                     # photo_id → Photo
    gps_map: dict[int, tuple[float, float]] | None = None,
    quality_weights: dict[tuple[int, int], float] | None = None,  # (photo_id, face_idx) → score
) -> dict[str, IdentityModel]:
    """Aggregate each known identity's centroids, date span, cameras and GPS."""
    gps_map = gps_map or {}
    models: dict[str, IdentityModel] = {}
    # backbone_id → {name → list[np.ndarray]}
    embs: dict[str, dict[str, list[np.ndarray]]] = {b: {} for b in embeddings_by_backbone}
    # backbone_id → {name → list[float]} — parallel quality weights for non-scene backbones
    qws: dict[str, dict[str, list[float]]] = {b: {} for b in embeddings_by_backbone}

    seen_scene: set[tuple[str, int, str]] = set()   # deduplicate (name, photo_id, backbone)

    for pid, fidx, name in train_records:
        if not name or name.strip().lower() in _UNKNOWN_TAGS:
            continue
        m = models.setdefault(name, IdentityModel(name=name))
        m.n_faces += 1
        photo = photo_map.get(pid)
        if photo is not None:
            d = _photo_days(photo)
            if d is not None:
                m.days = np.append(m.days, d)
            if photo.camera_model:
                m.camera_counts[photo.camera_model] = m.camera_counts.get(photo.camera_model, 0) + 1
                m.n_camera_obs += 1
            m.albums.add(photo.album_path)
            m.album_face_counts[photo.album_path] = m.album_face_counts.get(photo.album_path, 0) + 1
        if pid in gps_map:
            m.gps.append(gps_map[pid])
        for b, emap in embeddings_by_backbone.items():
            key = (pid, -1) if b in scene_backbones else (pid, fidx)
            # For scene backbones deduplicate by photo so the same image isn't
            # added multiple times when an identity has several faces in one photo.
            if b in scene_backbones:
                sk = (name, pid, b)
                if sk in seen_scene:
                    continue
                seen_scene.add(sk)
            t = emap.get(key)
            if t is not None:
                embs[b].setdefault(name, []).append(_l2(t.numpy().astype(np.float32)))
                if quality_weights is not None and b not in scene_backbones:
                    qws[b].setdefault(name, []).append(quality_weights.get((pid, fidx), 1.0))

    for b, by_name in embs.items():
        for name, vecs in by_name.items():
            if name not in models or not vecs:
                continue
            mat = np.stack(vecs)                               # [N, D] normalised
            # Quality-weighted centroid for face backbones; unweighted for scene backbones.
            q_list = qws.get(b, {}).get(name) if (quality_weights is not None and b not in scene_backbones) else None
            if q_list and len(q_list) == len(vecs):
                q_arr = np.array(q_list, dtype=np.float32)
                q_arr /= q_arr.sum()
                centroid = np.average(mat, axis=0, weights=q_arr)
            else:
                centroid = np.mean(mat, axis=0)
                q_arr = None
            models[name].centroids[b] = _l2(centroid)
            if b in scene_backbones:
                models[name].scene_embs[b] = mat               # kept for max-sim feature
            elif len(vecs) >= 2 * N_SUBCENTROIDS:
                from sklearn.cluster import KMeans              # lazy — avoids top-level sklearn dep
                km = KMeans(n_clusters=N_SUBCENTROIDS, n_init=5, random_state=42, verbose=0)
                km.fit(mat, sample_weight=q_arr)
                models[name].sub_centroids[b] = np.stack([_l2(c) for c in km.cluster_centers_])
    return models


# ---------------------------------------------------------------------------
# Centroid index for fast candidate shortlisting
# ---------------------------------------------------------------------------

@dataclass
class _CentroidIndex:
    backbones: list[str]
    names: dict[str, list[str]]              # backbone → identity names (matrix row order)
    mats: dict[str, np.ndarray]              # backbone → [n_ids, D] unit centroids


def _build_index(models: dict[str, IdentityModel], backbones: list[str]) -> _CentroidIndex:
    names: dict[str, list[str]] = {}
    mats: dict[str, np.ndarray] = {}
    for b in backbones:
        ns = [n for n, m in models.items() if b in m.centroids]
        names[b] = ns
        mats[b] = np.stack([models[n].centroids[b] for n in ns]) if ns else np.empty((0, 1))
    return _CentroidIndex(backbones, names, mats)


# ---------------------------------------------------------------------------
# Feature table
# ---------------------------------------------------------------------------

@dataclass
class FeatureTable:
    X: np.ndarray                     # [n_rows, n_feats] float32 (NaN allowed)
    y: np.ndarray                     # [n_rows] float: 1.0 / 0.0, or NaN if unlabeled
    feature_names: list[str]
    rows: list[tuple[int, int, str]]  # (photo_id, face_idx, candidate_name)
    group_ids: np.ndarray             # [n_rows] int: same id = rows for one face
    session_linked: np.ndarray        # [n_rows] bool: candidate shares an album with this face


def build_feature_table(
    face_records: list[tuple[int, int, str | None]],   # (photo_id, face_idx, true_name|None)
    identity_models: dict[str, IdentityModel],
    embeddings_by_backbone: dict[str, dict[tuple[int, int], torch.Tensor]],
    scene_backbones: set[str],
    photo_map: dict[int, "object"],
    gps_map: dict[int, tuple[float, float]] | None = None,
    top_k: int = 15,
    time_sigma_days: float = 45.0,
) -> FeatureTable:
    """
    Build scored rows for each face: its top-K candidate identities (by best
    embedding similarity), always including the true identity for tagged faces
    so positives are never dropped.  Untagged faces yield label-NaN rows.
    Faces tagged Unknown yield all-negative rows (the reject class).
    """
    gps_map = gps_map or {}
    backbones = sorted(embeddings_by_backbone)
    index = _build_index(identity_models, backbones)
    total_faces = sum(m.n_faces for m in identity_models.values()) or 1

    feat_names = (
        [f"sim_{b}" for b in backbones]
        + ["time_prox", "in_active_range", "camera_p", "camera_seen", "geo_km", "id_log_prior",
           "in_same_album", "n_same_album_faces", "album_person_match"]
    )

    X_rows: list[list[float]] = []
    y_rows: list[float] = []
    row_keys: list[tuple[int, int, str]] = []
    groups: list[int] = []
    linked: list[bool] = []

    for gid, (pid, fidx, true_name) in enumerate(face_records):
        photo = photo_map.get(pid)
        # Per-backbone normalised embedding for this face
        face_vecs: dict[str, np.ndarray] = {}
        for b in backbones:
            key = (pid, -1) if b in scene_backbones else (pid, fidx)
            t = embeddings_by_backbone[b].get(key)
            if t is not None:
                face_vecs[b] = _l2(t.numpy().astype(np.float32))

        # Rank identities by best similarity across the backbones we have
        best: dict[str, float] = {}
        for b, fv in face_vecs.items():
            mat, ns = index.mats[b], index.names[b]
            if not ns:
                continue
            sims = mat @ fv
            for name, s in zip(ns, sims):
                if s > best.get(name, -2.0):
                    best[name] = float(s)

        candidates = [n for n, _ in sorted(best.items(), key=lambda kv: -kv[1])[:top_k]]
        is_unknown = bool(true_name) and true_name.strip().lower() in _UNKNOWN_TAGS
        if true_name and not is_unknown and true_name in identity_models and true_name not in candidates:
            candidates.append(true_name)              # never drop a positive
        if not candidates:
            continue

        pdays = _photo_days(photo) if photo else None
        pcam = photo.camera_model if photo else None
        pgps = gps_map.get(pid)
        alb_words = _album_word_set(photo.album_path) if photo else frozenset()

        for name in candidates:
            m = identity_models[name]
            # Sim features: face backbones use centroid dot-product; scene backbones
            # use max cosine to any individual training photo (better than centroid
            # average for background-context signals that vary across locations).
            sim_feats: list[float] = []
            for b in backbones:
                if b not in face_vecs:
                    sim_feats.append(float("nan"))
                elif b in scene_backbones and b in m.scene_embs and m.scene_embs[b].shape[0] > 0:
                    # Scene backbone: max cosine to any individual training photo embedding.
                    sim_feats.append(float(np.max(m.scene_embs[b] @ face_vecs[b])))
                elif b in m.sub_centroids:
                    # Face backbone with sub-centroids: max cosine to any k-means cluster.
                    sim_feats.append(float(np.max(m.sub_centroids[b] @ face_vecs[b])))
                elif b in m.centroids:
                    sim_feats.append(float(m.centroids[b] @ face_vecs[b]))
                else:
                    sim_feats.append(float("nan"))
            # Temporal
            if pdays is not None and m.days.size:
                gap = float(np.min(np.abs(m.days - pdays)))
                time_prox = math.exp(-(gap * gap) / (2 * time_sigma_days ** 2))
                in_range = 1.0 if m.days.min() <= pdays <= m.days.max() else 0.0
            else:
                time_prox, in_range = float("nan"), float("nan")
            # Camera
            if pcam and m.n_camera_obs:
                camera_p = m.camera_counts.get(pcam, 0) / m.n_camera_obs
                camera_seen = 1.0 if pcam in m.camera_counts else 0.0
            else:
                camera_p, camera_seen = float("nan"), float("nan")
            # Geo
            geo_km = (min(_haversine_km(pgps, g) for g in m.gps)
                      if (pgps and m.gps) else float("nan"))
            # Album membership: how many of this identity's faces are in this album
            if photo is not None:
                in_same_album = 1.0 if photo.album_path in m.albums else 0.0
                n_same_album = float(m.album_face_counts.get(photo.album_path, 0))
            else:
                in_same_album, n_same_album = 0.0, 0.0
            # Album person-name match: candidate's first meaningful name word in album path
            token = _album_search_token(m.name)
            album_person_match = 1.0 if token and token in alb_words else 0.0

            X_rows.append(sim_feats + [
                time_prox, in_range, camera_p, camera_seen, geo_km,
                math.log1p(m.n_faces),
                in_same_album, n_same_album, album_person_match,
            ])
            if true_name is None:
                y_rows.append(float("nan"))           # inference row
            else:
                y_rows.append(1.0 if name == true_name else 0.0)
            row_keys.append((pid, fidx, name))
            groups.append(gid)
            linked.append(bool(photo) and photo.album_path in m.albums)

    return FeatureTable(
        X=np.asarray(X_rows, dtype=np.float32).reshape(-1, len(feat_names)),
        y=np.asarray(y_rows, dtype=np.float32),
        feature_names=feat_names,
        rows=row_keys,
        group_ids=np.asarray(groups, dtype=np.int64),
        session_linked=np.asarray(linked, dtype=bool),
    )
