"""
HTML gallery export for cluster results.

Generates a self-contained HTML file — no external dependencies — with face
crops grouped by cluster, largest clusters first, noise at the bottom.

Each face crop is base64-encoded inline so the file opens anywhere without
needing access to the photo library or thumbnails DB.

Usage:
    from babyface.export.html import export_html
    export_html(results, photos_by_id, thumbnails_db=Path("thumbnails-digikam.db"), output=Path("clusters.html"))
"""
from __future__ import annotations

import base64
import io
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from ..cluster.identity import ClusteredFace
from ..db.digikam import open_photo_image
from ..db.models import Photo
from ..embeddings.extract import crop_face

# Max pixel size for face crop thumbnails in the gallery
_THUMB_SIZE = 140


def _crop_to_b64(photo: Photo, face_idx: int, thumbnails_db: Path | None) -> str | None:
    """
    Open the photo, crop the face region, downscale, and return a base64 JPEG
    data-URL string, or None if the image can't be loaded.
    """
    try:
        img = open_photo_image(photo, thumbnails_db) if thumbnails_db else (
            Image.open(photo.full_path) if photo.full_path.exists() else None
        )
        if img is None:
            return None
        face = photo.faces[face_idx]
        crop = crop_face(img.convert("RGB"), (face.x, face.y, face.width, face.height))
        if crop is None:
            return None
        # Downscale to thumbnail size, preserving aspect ratio
        crop.thumbnail((_THUMB_SIZE, _THUMB_SIZE), Image.LANCZOS)
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def export_html(
    results: list[ClusteredFace],
    photos_by_id: dict[int, Photo],
    output: Path,
    thumbnails_db: Path | None = None,
) -> int:
    """
    Write a self-contained HTML gallery to `output`.

    Returns the number of face crops successfully rendered.
    """
    # Group by cluster_id
    by_cluster: dict[int, list[ClusteredFace]] = defaultdict(list)
    for r in results:
        by_cluster[r.cluster_id].append(r)

    # Sort: real clusters largest-first, noise last
    cluster_ids = sorted(
        [cid for cid in by_cluster if cid >= 0],
        key=lambda cid: -len(by_cluster[cid]),
    )
    if -1 in by_cluster:
        cluster_ids.append(-1)

    rendered = 0
    cluster_html_parts: list[str] = []

    for cid in cluster_ids:
        faces = by_cluster[cid]
        predicted = next((f.predicted_name for f in faces if f.predicted_name), None)
        is_noise = cid == -1

        label = "Noise / unassigned" if is_noise else (
            f"Cluster {cid}" + (f" — {predicted}" if predicted else "")
        )
        badge_class = "badge-noise" if is_noise else ("badge-named" if predicted else "badge-unknown")

        thumb_parts: list[str] = []
        for f in faces:
            photo = photos_by_id.get(f.photo_id)
            if photo is None or f.face_idx >= len(photo.faces):
                continue
            data_url = _crop_to_b64(photo, f.face_idx, thumbnails_db)
            if data_url is None:
                continue
            rendered += 1
            dk_name = photo.faces[f.face_idx].person_name or ""
            conf_pct = f"{f.confidence:.0%}"
            date_str = (photo.digikam_date or photo.folder_date or "")
            date_str = str(date_str)[:10] if date_str else ""
            tooltip = " | ".join(filter(None, [dk_name, f"conf {conf_pct}", date_str]))
            thumb_parts.append(
                f'<div class="thumb" title="{_esc(tooltip)}">'
                f'<img src="{data_url}" loading="lazy">'
                f'{"<div class=label>" + _esc(dk_name) + "</div>" if dk_name else ""}'
                f'</div>'
            )

        if not thumb_parts:
            continue  # skip clusters with no renderable crops

        open_attr = "" if is_noise else " open"
        cluster_html_parts.append(f"""
        <details{open_attr} class="cluster">
          <summary>
            <span class="cluster-label">{_esc(label)}</span>
            <span class="badge {badge_class}">{len(faces)} faces</span>
          </summary>
          <div class="grid">
            {"".join(thumb_parts)}
          </div>
        </details>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>babyface — cluster gallery</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: system-ui, sans-serif;
    background: #111;
    color: #eee;
    margin: 0;
    padding: 1rem 1.5rem 3rem;
  }}
  h1 {{ font-size: 1.3rem; color: #aaa; font-weight: 400; margin: 0 0 1.5rem; }}
  .stats {{ font-size: 0.85rem; color: #666; margin-bottom: 2rem; }}

  .cluster {{
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    margin-bottom: 1rem;
    overflow: hidden;
  }}
  .cluster summary {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.65rem 1rem;
    cursor: pointer;
    background: #1a1a1a;
    user-select: none;
    list-style: none;
  }}
  .cluster summary::-webkit-details-marker {{ display: none; }}
  .cluster summary::before {{
    content: "▶";
    font-size: 0.7rem;
    color: #555;
    transition: transform 0.15s;
    flex-shrink: 0;
  }}
  .cluster[open] > summary::before {{ transform: rotate(90deg); }}

  .cluster-label {{ font-size: 0.95rem; flex: 1; }}
  .badge {{
    font-size: 0.75rem;
    padding: 0.2em 0.6em;
    border-radius: 999px;
    font-weight: 600;
    flex-shrink: 0;
  }}
  .badge-named   {{ background: #1a3a5c; color: #7ec8e3; }}
  .badge-unknown {{ background: #2a2a1a; color: #b0a060; }}
  .badge-noise   {{ background: #2a1a1a; color: #b06060; }}

  .grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding: 10px;
    background: #161616;
  }}
  .thumb {{
    position: relative;
    border-radius: 4px;
    overflow: hidden;
    background: #222;
    flex-shrink: 0;
  }}
  .thumb img {{
    display: block;
    width: {_THUMB_SIZE}px;
    height: {_THUMB_SIZE}px;
    object-fit: cover;
  }}
  .thumb .label {{
    position: absolute;
    bottom: 0; left: 0; right: 0;
    background: rgba(0,0,0,0.65);
    color: #fff;
    font-size: 0.65rem;
    padding: 2px 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
</style>
</head>
<body>
<h1>babyface — cluster gallery</h1>
<div class="stats">
  {len(cluster_ids) - (1 if -1 in by_cluster else 0)} clusters &nbsp;·&nbsp;
  {len(results)} faces &nbsp;·&nbsp;
  {rendered} thumbnails rendered
</div>
{"".join(cluster_html_parts)}
</body>
</html>"""

    output.write_text(html, encoding="utf-8")
    return rendered


def _esc(s: str) -> str:
    """Minimal HTML escaping for attribute values and text content."""
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )


def export_review_html(
    assignments: list,                     # list[fusion.constraints.Assignment]
    photos_by_id: dict[int, Photo],
    output: Path,
    thumbnails_db: Path | None = None,
    unknown_clusters: list[ClusteredFace] | None = None,
) -> int:
    """
    Review gallery for the fusion labeler.  Faces are grouped by *predicted*
    identity (largest first, Unknown/rejected last), with special sections up
    top for coherence-flagged faces and ambiguous near-ties, and an Unknown
    Clusters section at the bottom (HDBSCAN groups within the Unknown pool).

    Returns the number of crops rendered.
    """
    flagged   = [a for a in assignments if a.flags]
    ambiguous = [a for a in assignments if a.ambiguous]
    by_ident: dict[str, list] = defaultdict(list)
    for a in assignments:
        if a.ambiguous:
            continue   # ambiguous faces go in their own section, not per-identity
        by_ident[a.identity or "Unknown / rejected"].append(a)

    order = sorted((k for k in by_ident if k != "Unknown / rejected"),
                   key=lambda k: -len(by_ident[k]))
    if "Unknown / rejected" in by_ident:
        order.append("Unknown / rejected")

    rendered = 0

    def _thumb(a) -> str | None:
        nonlocal rendered
        photo = photos_by_id.get(a.photo_id)
        if photo is None or a.face_idx >= len(photo.faces):
            return None
        data_url = _crop_to_b64(photo, a.face_idx, thumbnails_db)
        if data_url is None:
            return None
        rendered += 1
        dk = photo.faces[a.face_idx].person_name or ""
        pred = a.identity or "—"
        date_str = str(photo.digikam_date or photo.folder_date or "")[:10]
        tip = " | ".join(filter(None, [
            f"pred {pred} ({a.score:.0%})",
            f"tag {dk}" if dk else "",
            date_str, *a.flags,
        ]))
        warn = '<div class="warn">⚠</div>' if a.flags else ""
        # Mismatch between an existing human tag and the prediction is worth a marker.
        mism = " mismatch" if (dk and a.identity and dk != a.identity) else ""
        return (
            f'<div class="thumb{mism}" title="{_esc(tip)}">'
            f'<img src="{data_url}" loading="lazy">{warn}'
            f'<div class="label">{_esc(pred)} {a.score:.0%}</div>'
            f'</div>'
        )

    sections: list[str] = []
    if flagged:
        parts = [t for t in (_thumb(a) for a in flagged) if t]
        if parts:
            sections.append(f"""
        <details open class="cluster review">
          <summary><span class="cluster-label">⚠ Needs review — coherence flags</span>
          <span class="badge badge-noise">{len(flagged)} faces</span></summary>
          <div class="grid">{"".join(parts)}</div>
        </details>""")

    if ambiguous:
        parts = [t for t in (_thumb(a) for a in ambiguous) if t]
        if parts:
            sections.append(f"""
        <details open class="cluster ambig">
          <summary><span class="cluster-label">❓ Ambiguous — top-2 near-tie, manual resolution needed</span>
          <span class="badge badge-ambig">{len(ambiguous)} faces</span></summary>
          <div class="grid">{"".join(parts)}</div>
        </details>""")

    for name in order:
        items = by_ident[name]
        parts = [t for t in (_thumb(a) for a in items) if t]
        if not parts:
            continue
        is_unk = name == "Unknown / rejected"
        badge = "badge-noise" if is_unk else "badge-named"
        sections.append(f"""
        <details class="cluster">
          <summary><span class="cluster-label">{_esc(name)}</span>
          <span class="badge {badge}">{len(items)} faces</span></summary>
          <div class="grid">{"".join(parts)}</div>
        </details>""")

    if unknown_clusters:
        by_uc: dict[int, list[ClusteredFace]] = defaultdict(list)
        for cf in unknown_clusters:
            if cf.cluster_id >= 0:
                by_uc[cf.cluster_id].append(cf)
        for cid in sorted(by_uc, key=lambda c: -len(by_uc[c])):
            cfaces = by_uc[cid]
            cthumbs: list[str] = []
            for cf in cfaces:
                photo = photos_by_id.get(cf.photo_id)
                if photo is None or cf.face_idx >= len(photo.faces):
                    continue
                data_url = _crop_to_b64(photo, cf.face_idx, thumbnails_db)
                if data_url is None:
                    continue
                rendered += 1
                dk = photo.faces[cf.face_idx].person_name or ""
                date_str = str(photo.digikam_date or photo.folder_date or "")[:10]
                tip = " | ".join(filter(None, [
                    f"unknown group {cid}", f"conf {cf.confidence:.0%}", dk, date_str,
                ]))
                cthumbs.append(
                    f'<div class="thumb" title="{_esc(tip)}">'
                    f'<img src="{data_url}" loading="lazy">'
                    f'{"<div class=label>" + _esc(dk) + "</div>" if dk else ""}'
                    f'</div>'
                )
            if cthumbs:
                sections.append(f"""
        <details class="cluster unk-cluster">
          <summary><span class="cluster-label">Unknown group {cid}</span>
          <span class="badge badge-unknown">{len(cfaces)} faces</span></summary>
          <div class="grid">{"".join(cthumbs)}</div>
        </details>""")

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>babyface — review gallery</title><style>
  *,*::before,*::after{{box-sizing:border-box}}
  body{{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:1rem 1.5rem 3rem}}
  h1{{font-size:1.3rem;color:#aaa;font-weight:400;margin:0 0 .5rem}}
  .stats{{font-size:.85rem;color:#666;margin-bottom:1.5rem}}
  .cluster{{border:1px solid #2a2a2a;border-radius:8px;margin-bottom:1rem;overflow:hidden}}
  .cluster.review{{border-color:#7a3a3a}}
  .cluster summary{{display:flex;align-items:center;gap:.75rem;padding:.65rem 1rem;cursor:pointer;background:#1a1a1a;user-select:none;list-style:none}}
  .cluster summary::-webkit-details-marker{{display:none}}
  .cluster summary::before{{content:"▶";font-size:.7rem;color:#555;transition:transform .15s}}
  .cluster[open]>summary::before{{transform:rotate(90deg)}}
  .cluster-label{{font-size:.95rem;flex:1}}
  .badge{{font-size:.75rem;padding:.2em .6em;border-radius:999px;font-weight:600}}
  .badge-named{{background:#1a3a5c;color:#7ec8e3}} .badge-noise{{background:#2a1a1a;color:#b06060}} .badge-ambig{{background:#2a1a3a;color:#c07ee3}} .badge-unknown{{background:#2a2a1a;color:#b0a060}}
  .grid{{display:flex;flex-wrap:wrap;gap:6px;padding:10px;background:#161616}}
  .thumb{{position:relative;border-radius:4px;overflow:hidden;background:#222}}
  .thumb.mismatch{{outline:2px solid #d08010}}
  .thumb img{{display:block;width:{_THUMB_SIZE}px;height:{_THUMB_SIZE}px;object-fit:cover}}
  .thumb .warn{{position:absolute;top:2px;right:2px;background:rgba(180,40,40,.9);color:#fff;font-size:.7rem;padding:0 4px;border-radius:3px}}
  .thumb .label{{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.65);color:#fff;font-size:.65rem;padding:2px 4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
</style></head><body>
<h1>babyface — review gallery</h1>
<div class="stats">{len(assignments)} faces labeled · {len(ambiguous)} ambiguous · {len(flagged)} flagged · {rendered} thumbnails · hover a face for details</div>
{"".join(sections)}
</body></html>"""
    output.write_text(html, encoding="utf-8")
    return rendered
