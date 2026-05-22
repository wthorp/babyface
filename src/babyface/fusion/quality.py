"""
Face crop quality scoring via Laplacian variance (sharpness proxy).

Used to weight training photo contributions to identity centroids:
sharper, cleaner crops pull the centroid closer to their embedding.

Score is Laplacian variance computed as the mean of 2nd-difference
variances along each axis — no scipy dependency required.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _lap_var(gray: np.ndarray) -> float:
    """Laplacian variance via 2nd-difference approximation (no scipy needed)."""
    d2r = np.diff(gray.astype(np.float32), 2, axis=0)
    d2c = np.diff(gray.astype(np.float32), 2, axis=1)
    return float((d2r.var() + d2c.var()) / 2.0)


def compute_quality_scores(
    tagged_faces: list[tuple[int, int, str]],
    photo_map: dict,
) -> dict[tuple[int, int], float]:
    """
    Compute Laplacian variance for every (photo_id, face_idx) in tagged_faces.
    Photo.full_path must already be resolved (pass photo_root to load_photos).
    Returns raw (unnormalized) scores; faces that can't be loaded are absent.
    """
    from PIL import Image
    from ..embeddings.extract import crop_face

    # Group by photo to load each image once
    by_photo: dict[int, list[int]] = {}
    for pid, fidx, _ in tagged_faces:
        by_photo.setdefault(pid, []).append(fidx)

    scores: dict[tuple[int, int], float] = {}
    n = 0
    for pid, fidxs in by_photo.items():
        photo = photo_map.get(pid)
        if photo is None:
            continue
        try:
            if not photo.full_path.exists():
                continue
            img = Image.open(photo.full_path).convert("RGB")
        except Exception:
            continue
        for fidx in fidxs:
            if fidx >= len(photo.faces):
                continue
            f = photo.faces[fidx]
            crop = crop_face(img, (f.x, f.y, f.width, f.height))
            if crop is None:
                continue
            gray = np.array(crop.convert("L"))
            scores[(pid, fidx)] = _lap_var(gray)
            n += 1
        img.close()
        if n % 5000 == 0 and n > 0:
            print(f"  quality: {n:,} faces scored ...", flush=True)

    return scores


def normalize_scores(
    scores: dict[tuple[int, int], float],
    percentile_clip: float = 99.0,
) -> dict[tuple[int, int], float]:
    """Clip at the given percentile and scale to [epsilon, 1]."""
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=np.float32)
    top = float(np.percentile(vals, percentile_clip))
    if top <= 0:
        return {k: 1.0 for k in scores}
    eps = 0.05   # floor so even blurry crops get some weight
    return {k: float(max(eps, min(v, top) / top)) for k, v in scores.items()}


def save_quality_cache(scores: dict[tuple[int, int], float], path: Path) -> None:
    data = {f"{k[0]},{k[1]}": v for k, v in scores.items()}
    path.write_text(json.dumps(data), encoding="utf-8")
    print(f"  quality cache: {len(scores):,} scores written to {path}")


def load_quality_cache(path: Path) -> dict[tuple[int, int], float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {(int(k.split(",")[0]), int(k.split(",")[1])): float(v) for k, v in data.items()}
