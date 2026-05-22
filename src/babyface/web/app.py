"""
FastAPI labeling UI for babyface predictions.

Routes
------
GET  /                      stats + identity list
GET  /identity/{name}       paginated face grid for one identity
GET  /ambiguous             ambiguous queue (top-2 near-tie faces)
GET  /clusters              unknown face clusters
GET  /cluster/{id}          all faces in a cluster
GET  /face/{pid}/{fidx}     serve face crop (real JPEG or SVG placeholder)
POST /correct               apply a single face correction (HTMX)
POST /cluster/assign        assign all faces in cluster to identity (HTMX)
POST /retrain               kick off background retrain
GET  /retrain/status        HTMX poll – retrain progress
"""
from __future__ import annotations

import asyncio
import html
import io
import math
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .store import DataStore, _identity_color

_HERE = Path(__file__).parent
_REPO: Optional[Path] = None
_STORE: Optional[DataStore] = None


def create_app(repo_dir: Path, predictions_file: str = "predictions6.json",
               photo_root: Optional[Path] = None) -> FastAPI:
    global _REPO, _STORE
    _REPO = repo_dir
    _STORE = DataStore(repo_dir, predictions_file=predictions_file, photo_root=photo_root)

    app = FastAPI(title="babyface labeling UI")
    app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")
    templates = Jinja2Templates(directory=str(_HERE / "templates"))

    # -- helpers ----------------------------------------------------------

    def tmpl(name: str, req: Request, **ctx) -> HTMLResponse:
        ctx["store"] = _STORE
        ctx["stats"] = _STORE.stats()
        return templates.TemplateResponse(request=req, name=name, context=ctx)

    def page_range(total: int, page: int, page_size: int) -> dict:
        n_pages = max(1, math.ceil(total / page_size))
        return dict(page=page, n_pages=n_pages, total=total, page_size=page_size,
                    has_prev=page > 0, has_next=page < n_pages - 1)

    # -- routes -----------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index(req: Request):
        return tmpl("index.html", req, identity_names=_STORE.identity_names())

    @app.get("/identity/{name}", response_class=HTMLResponse)
    async def identity(req: Request, name: str, page: int = 0):
        faces, total = _STORE.faces_for_identity(name, page=page)
        return tmpl("identity.html", req,
                    identity=name,
                    color=_identity_color(name),
                    faces=faces,
                    **page_range(total, page, 50),
                    all_identities=_STORE.identity_names())

    @app.get("/ambiguous", response_class=HTMLResponse)
    async def ambiguous(req: Request, page: int = 0):
        faces, total = _STORE.ambiguous_faces(page=page)
        return tmpl("ambiguous.html", req,
                    faces=faces,
                    **page_range(total, page, 50),
                    all_identities=_STORE.identity_names())

    @app.get("/clusters", response_class=HTMLResponse)
    async def clusters(req: Request, page: int = 0):
        cluster_list, total = _STORE.clusters(page=page)
        return tmpl("clusters.html", req,
                    cluster_list=cluster_list,
                    **page_range(total, page, 30),
                    all_identities=_STORE.identity_names())

    @app.get("/cluster/{cluster_id}", response_class=HTMLResponse)
    async def cluster_detail(req: Request, cluster_id: int, page: int = 0):
        faces = _STORE.faces_in_cluster(cluster_id)
        page_size = 50
        total = len(faces)
        start = page * page_size
        return tmpl("cluster_detail.html", req,
                    cluster_id=cluster_id,
                    faces=faces[start:start+page_size],
                    **page_range(total, page, page_size),
                    all_identities=_STORE.identity_names())

    @app.get("/face/{photo_id}/{face_idx}")
    async def face_crop(photo_id: int, face_idx: int):
        result = _STORE.get_face_crop_bytes(photo_id, face_idx)
        if result:
            data, mime = result
            return Response(content=data, media_type=mime)
        return Response(content=_placeholder_svg(photo_id, face_idx),
                        media_type="image/svg+xml")

    @app.post("/correct")
    async def correct(
        req: Request,
        photo_id: int = Form(...),
        face_idx: int = Form(...),
        identity: str = Form(...),
        return_to: str = Form(default=""),
    ):
        _STORE.save_correction(photo_id, face_idx, identity)
        pred = _STORE.predictions.get((photo_id, face_idx))
        # Return updated card fragment for HTMX
        name = pred.predicted_identity or "Unknown"
        color = _identity_color(name)
        frag = _face_card_html(photo_id, face_idx, pred, color,
                               _STORE.identity_names(), return_to)
        return HTMLResponse(frag)

    @app.post("/cluster/assign")
    async def cluster_assign(
        req: Request,
        cluster_id: int = Form(...),
        identity: str = Form(...),
    ):
        n = _STORE.assign_cluster(cluster_id, identity)
        return HTMLResponse(
            f'<div class="toast">Assigned {n} faces to <strong>{html.escape(identity)}</strong></div>'
        )

    @app.post("/retrain")
    async def retrain(req: Request):
        if _STORE.retrain_running:
            return HTMLResponse('<div class="retrain-status">Already running…</div>')
        _STORE.retrain_running = True
        _STORE.retrain_log = ["Starting retrain…"]
        _STORE.retrain_exit_code = None
        asyncio.get_event_loop().run_in_executor(None, _run_retrain)
        return HTMLResponse('<div class="retrain-status" hx-get="/retrain/status" '
                            'hx-trigger="every 3s" hx-swap="outerHTML">Starting…</div>')

    @app.get("/retrain/status", response_class=HTMLResponse)
    async def retrain_status(req: Request):
        logs = "\n".join(_STORE.retrain_log[-20:])
        if _STORE.retrain_running:
            return HTMLResponse(
                f'<div class="retrain-status" hx-get="/retrain/status" '
                f'hx-trigger="every 3s" hx-swap="outerHTML">'
                f'<pre>{html.escape(logs)}</pre></div>'
            )
        code = _STORE.retrain_exit_code
        cls = "retrain-done" if code == 0 else "retrain-error"
        return HTMLResponse(
            f'<div class="retrain-status {cls}"><pre>{html.escape(logs)}</pre>'
            f'<p>{"Done ✓" if code == 0 else f"Failed (exit {code})"}</p></div>'
        )

    return app


# ---------------------------------------------------------------------------
# Retrain worker
# ---------------------------------------------------------------------------

def _run_retrain() -> None:
    assert _STORE is not None and _REPO is not None
    venv_py = _REPO / ".venv" / "bin" / "python"
    torch_lib = _get_torch_lib()
    import os
    env = os.environ.copy()
    if torch_lib:
        env["LD_LIBRARY_PATH"] = torch_lib + (":" + env["LD_LIBRARY_PATH"]
                                               if env.get("LD_LIBRARY_PATH") else "")

    # Export corrections as pseudo-labels file
    corrections_pseudo = _REPO / "corrections_pseudo.json"
    _export_corrections_as_pseudo(corrections_pseudo)

    cmd = [
        str(venv_py), "-m", "babyface.cli", "label",
        "--db", str(_REPO / "digikam4.db"),
        "--photo-root", "/data/photos/photos",
        "--cache-dir", str(_REPO / "embeddings"),
        "--min-face-px", "48",
        "--ambiguous-margin", "0.10",
        "--pseudo-labels", str(corrections_pseudo),
        "--quality-cache", str(_REPO / "quality_cache.json"),
        "--export-html", str(_REPO / "review_retrain.html"),
        "--writeback", str(_REPO / "predictions_retrain.json"),
    ]
    _STORE.retrain_log.append("$ " + " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(_REPO), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in proc.stdout:
            _STORE.retrain_log.append(line.rstrip())
        proc.wait()
        _STORE.retrain_exit_code = proc.returncode
    except Exception as exc:
        _STORE.retrain_log.append(f"ERROR: {exc}")
        _STORE.retrain_exit_code = 1
    finally:
        _STORE.retrain_running = False


def _get_torch_lib() -> Optional[str]:
    try:
        import torch, pathlib
        return str(pathlib.Path(torch.__file__).parent / "lib")
    except Exception:
        return None


def _export_corrections_as_pseudo(out_path: Path) -> None:
    """Write corrections as a pseudo-label JSON the CLI can consume."""
    import sqlite3, json
    records = []
    con = sqlite3.connect(str(_REPO / "corrections.db"))
    for row in con.execute("SELECT photo_id, face_idx, identity FROM corrections WHERE identity IS NOT NULL"):
        pid, fidx, ident = row
        pred = _STORE.predictions.get((pid, fidx))
        records.append({
            "photo_id": pid,
            "face_idx": fidx,
            "filename": pred.filename if pred else "",
            "album_path": pred.album_path if pred else "",
            "face_idx": fidx,
            "bbox": pred.bbox if pred else [0, 0, 0, 0],
            "original_tag": "Unknown",
            "predicted_identity": ident,
            "score": 1.0,
            "ambiguous": False,
            "flags": [],
        })
    con.close()
    out_path.write_text(json.dumps(records), encoding="utf-8")


# ---------------------------------------------------------------------------
# Face crop placeholder
# ---------------------------------------------------------------------------

def _placeholder_svg(photo_id: int, face_idx: int) -> bytes:
    pred = _STORE.predictions.get((photo_id, face_idx)) if _STORE else None
    name = (pred.predicted_identity or "?") if pred else "?"
    initial = name[0].upper() if name else "?"
    color = _identity_color(name)
    score = f"{pred.score:.0%}" if pred else ""
    svg = textwrap.dedent(f"""\
        <svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">
          <rect width="120" height="120" rx="8" fill="{color}" opacity="0.25"/>
          <text x="60" y="68" text-anchor="middle" font-family="sans-serif"
                font-size="42" font-weight="bold" fill="{color}">{html.escape(initial)}</text>
          <text x="60" y="108" text-anchor="middle" font-family="sans-serif"
                font-size="12" fill="#666">{html.escape(score)}</text>
        </svg>
    """)
    return svg.encode()


# ---------------------------------------------------------------------------
# HTMX partial: face card
# ---------------------------------------------------------------------------

def _face_card_html(photo_id: int, face_idx: int, pred, color: str,
                    all_identities: list, return_to: str) -> str:
    ident = pred.predicted_identity or "Unknown"
    opts = "".join(
        f'<option value="{html.escape(n)}" {"selected" if n == ident else ""}>{html.escape(n)}</option>'
        for n, _, _ in all_identities
    )
    badge = "corrected" if pred.corrected else ""
    return f"""
<div class="face-card {badge}" id="face-{photo_id}-{face_idx}">
  <img src="/face/{photo_id}/{face_idx}" loading="lazy" width="120" height="120"
       alt="{html.escape(ident)}">
  <div class="face-meta">
    <div class="face-name" style="color:{color}">{html.escape(ident)}</div>
    <div class="face-score">{pred.score:.0%}</div>
    <div class="face-date">{html.escape(pred.date_approx)}</div>
  </div>
  <form hx-post="/correct" hx-target="#face-{photo_id}-{face_idx}" hx-swap="outerHTML">
    <input type="hidden" name="photo_id" value="{photo_id}">
    <input type="hidden" name="face_idx" value="{face_idx}">
    <input type="hidden" name="return_to" value="{html.escape(return_to)}">
    <select name="identity" onchange="this.form.requestSubmit()">
      <option value="Unknown">Unknown</option>
      {opts}
    </select>
  </form>
</div>"""
