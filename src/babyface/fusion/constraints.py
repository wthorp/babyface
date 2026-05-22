"""
Constraint reconciliation on top of the per-(face, identity) GBM scores.

The fusion model scores each face/identity pair independently.  Two structural
facts it can't enforce on its own get applied here:

  1. Mutual exclusion — a person appears at most once per photo.  We solve a
     per-photo assignment (Hungarian) over the scores: identities are exclusive
     columns; every face also has its own non-exclusive "Unknown" slot priced
     at the reject threshold, so faces below threshold (or crowded out of a
     contested identity) fall through to Unknown instead of forcing a bad pick.

  2. Spatiotemporal coherence — "Eric isn't in Germany for one photo."  For a
     proposed (face → identity), we test the photo's date / camera / GPS against
     that identity's *established track* with robust outlier stats and attach
     human-readable flags.  By default this is advisory (flags for review); set
     coherence_lambda > 0 to also soft-down-weight incoherent pairs before the
     assignment, nudging it toward a more plausible identity.

Coherence overlaps with features the GBM already saw (time_prox, camera_seen,
geo_km), so down-weighting is opt-in to avoid double-counting; the *flags* are
always useful because they explain, in words, why an assignment is suspect.

scipy is a core dependency; LightGBM is imported lazily by score_faces.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .features import build_feature_table, _haversine_km, _photo_days, _UNKNOWN_TAGS


# ---------------------------------------------------------------------------
# Scoring untagged faces with a trained FusionModel
# ---------------------------------------------------------------------------

def score_faces(
    model,                                  # fusion.FusionModel
    face_records: list[tuple[int, int, str | None]],
    embeddings_by_backbone: dict,
    photo_map: dict,
    gps_map: dict | None = None,
    top_k: int = 15,
) -> dict[tuple[int, int], dict[str, float]]:
    """Return {(photo_id, face_idx): {identity: P(correct)}} for the candidates."""
    import lightgbm  # noqa: F401  (import before torch already happened upstream)
    ft = build_feature_table(face_records, model.identity_models,
                             embeddings_by_backbone, model.scene_backbones,
                             photo_map, gps_map, top_k=top_k)
    if ft.X.size == 0:
        return {}
    # Align columns to the model's training feature order.
    if ft.feature_names != model.feature_names:
        idx = [ft.feature_names.index(n) for n in model.feature_names]
        X = ft.X[:, idx]
    else:
        X = ft.X
    proba = model.booster.predict_proba(X)[:, 1]
    out: dict[tuple[int, int], dict[str, float]] = {}
    for (pid, fidx, name), p in zip(ft.rows, proba):
        out.setdefault((pid, fidx), {})[name] = float(p)
    return out


# ---------------------------------------------------------------------------
# Spatiotemporal coherence
# ---------------------------------------------------------------------------

def coherence_flags(
    photo, identity_model,
    pgps: tuple[float, float] | None = None,
    time_z: float = 4.0,
    geo_km_max: float = 500.0,
    min_camera_obs: int = 10,
) -> list[str]:
    """
    Human-readable reasons a (face → identity) assignment looks anomalous
    against the identity's established track.  Empty list = coherent.
    """
    flags: list[str] = []
    m = identity_model
    pday = _photo_days(photo) if photo else None

    # Temporal: robust z-score (median / MAD) AND outside the tagged span.
    if pday is not None and m.days.size >= 5:
        med = float(np.median(m.days))
        mad = float(np.median(np.abs(m.days - med))) or 1.0
        z = abs(pday - med) / (1.4826 * mad)
        outside = not (m.days.min() <= pday <= m.days.max())
        if z > time_z and outside:
            yrs = abs(pday - med) / 365.0
            flags.append(f"date {yrs:.1f}y from {m.name}'s usual range (z={z:.1f})")

    # Geo: far from every known location for this identity.
    if pgps and m.gps:
        dmin = min(_haversine_km(pgps, g) for g in m.gps)
        if dmin > geo_km_max:
            flags.append(f"{dmin:.0f}km from {m.name}'s known locations")

    # Camera: a device never seen for a well-observed identity.
    if photo and photo.camera_model and m.n_camera_obs >= min_camera_obs \
            and photo.camera_model not in m.camera_counts:
        flags.append(f"camera {photo.camera_model!r} never used for {m.name}")

    return flags


# ---------------------------------------------------------------------------
# Per-photo mutual-exclusion assignment
# ---------------------------------------------------------------------------

@dataclass
class Assignment:
    photo_id: int
    face_idx: int
    identity: str | None          # None = Unknown / rejected
    score: float
    flags: list[str] = field(default_factory=list)
    ambiguous: bool = False       # top-2 within ambiguous_margin of top-1


def resolve_photo(
    photo,
    face_scores: dict[int, dict[str, float]],   # face_idx → {identity: prob}
    identity_models: dict,
    pgps: tuple[float, float] | None = None,
    reject_threshold: float = 0.5,
    coherence_lambda: float = 0.0,
    ambiguous_margin: float = 0.0,
) -> list[Assignment]:
    """
    Assign identities to the faces of one photo with no identity used twice.

    Builds a cost matrix [F, I + F]: the first I columns are the photo's
    candidate identities (exclusive); the last F columns are per-face Unknown
    slots priced at reject_threshold (face i may only use column I+i).  A
    max-weight matching gives each face exactly one column.
    """
    face_idxs = sorted(face_scores)
    F = len(face_idxs)
    if F == 0:
        return []
    identities = sorted({ident for s in face_scores.values() for ident in s
                         if ident.strip().lower() not in _UNKNOWN_TAGS})
    I = len(identities)
    id_col = {name: j for j, name in enumerate(identities)}

    NEG = -1e9
    M = np.full((F, I + F), NEG, dtype=np.float64)
    flags_cache: dict[tuple[int, str], list[str]] = {}

    for fi, fidx in enumerate(face_idxs):
        for ident, p in face_scores[fidx].items():
            if ident.strip().lower() in _UNKNOWN_TAGS:
                continue
            score = p
            if coherence_lambda > 0 and ident in identity_models:
                fl = coherence_flags(photo, identity_models[ident], pgps)
                flags_cache[(fidx, ident)] = fl
                if fl:
                    score *= max(0.0, 1.0 - coherence_lambda * (len(fl) / 3.0))
            M[fi, id_col[ident]] = score
        M[fi, I + fi] = reject_threshold       # this face's Unknown escape

    rows, cols = linear_sum_assignment(M, maximize=True)

    out: list[Assignment] = []
    for fi, col in zip(rows, cols):
        fidx = face_idxs[fi]
        if col < I and M[fi, col] > NEG / 2:
            ident = identities[col]
            score = float(face_scores[fidx].get(ident, float("nan")))
            fl = flags_cache.get((fidx, ident))
            if fl is None and ident in identity_models:
                fl = coherence_flags(photo, identity_models[ident], pgps)
            out.append(Assignment(photo.id, fidx, ident, score, fl or []))
        else:
            best = max(face_scores[fidx].values()) if face_scores[fidx] else 0.0
            out.append(Assignment(photo.id, fidx, None, float(best), []))

    if ambiguous_margin > 0.0:
        for a in out:
            if a.identity is None:
                continue
            fscores = face_scores.get(a.face_idx, {})
            assigned_score = fscores.get(a.identity, 0.0)
            rivals = [s for n, s in fscores.items()
                      if n != a.identity and n.strip().lower() not in _UNKNOWN_TAGS]
            if rivals and assigned_score - max(rivals) < ambiguous_margin:
                a.ambiguous = True

    return out


def reconcile(
    model,                                  # fusion.FusionModel
    face_records: list[tuple[int, int, str | None]],
    embeddings_by_backbone: dict,
    photo_map: dict,
    gps_map: dict | None = None,
    reject_threshold: float = 0.5,
    coherence_lambda: float = 0.0,
    ambiguous_margin: float = 0.0,
    top_k: int = 15,
) -> list[Assignment]:
    """Score faces, then resolve mutual exclusion + coherence per photo."""
    gps_map = gps_map or {}
    scores = score_faces(model, face_records, embeddings_by_backbone,
                         photo_map, gps_map, top_k=top_k)
    by_photo: dict[int, dict[int, dict[str, float]]] = {}
    for (pid, fidx), s in scores.items():
        by_photo.setdefault(pid, {})[fidx] = s

    results: list[Assignment] = []
    for pid, face_scores in by_photo.items():
        photo = photo_map.get(pid)
        if photo is None:
            continue
        results.extend(resolve_photo(
            photo, face_scores, model.identity_models,
            pgps=gps_map.get(pid), reject_threshold=reject_threshold,
            coherence_lambda=coherence_lambda, ambiguous_margin=ambiguous_margin,
        ))
    return results


def write_predictions(assignments: list[Assignment], photo_map: dict, path) -> int:
    """
    Write reviewable predictions to `path` (.json or .csv). Nothing is written
    back to DigiKam — this is an export for human review/import. Each record
    carries the photo, face region, original tag, prediction, score and flags.
    Returns the number of records written.
    """
    import csv
    import json
    from pathlib import Path

    path = Path(path)
    recs = []
    for a in assignments:
        p = photo_map.get(a.photo_id)
        face = p.faces[a.face_idx] if (p and a.face_idx < len(p.faces)) else None
        recs.append({
            "photo_id": a.photo_id,
            "filename": getattr(p, "filename", ""),
            "album_path": getattr(p, "album_path", ""),
            "face_idx": a.face_idx,
            "bbox": [face.x, face.y, face.width, face.height] if face else None,
            "original_tag": (face.person_name if face else None),
            "predicted_identity": a.identity,
            "score": round(a.score, 4),
            "ambiguous": a.ambiguous,
            "flags": a.flags,
        })

    if path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["photo_id", "filename", "album_path", "face_idx", "bbox",
                        "original_tag", "predicted_identity", "score", "ambiguous", "flags"])
            for r in recs:
                w.writerow([r["photo_id"], r["filename"], r["album_path"], r["face_idx"],
                            json.dumps(r["bbox"]), r["original_tag"] or "",
                            r["predicted_identity"] or "", r["score"], r["ambiguous"],
                            "; ".join(r["flags"])])
    else:
        path.write_text(json.dumps(recs, indent=2), encoding="utf-8")
    return len(recs)
