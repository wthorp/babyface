"""
CLI entry points for the babyface pipeline.

Commands:
  babyface inspect-db        — summarise what's in digikam4.db + recognition.db
  babyface detect-datestamps — scan photos for anomalous timestamps
  babyface cluster           — extract DINOv2 embeddings and cluster by identity
"""
from __future__ import annotations
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

_DEFAULT_DK  = Path("digikam4.db")
_DEFAULT_REC = Path("recognition.db")
_DEFAULT_TH  = Path("thumbnails-digikam.db")


@click.group()
def main():
    """Infant photo identity clustering tool."""
    pass


# ---------------------------------------------------------------------------
# inspect-db
# ---------------------------------------------------------------------------

@main.command("inspect-db")
@click.option("--db",             default=_DEFAULT_DK,  type=Path, show_default=True)
@click.option("--recognition-db", default=_DEFAULT_REC, type=Path, show_default=True)
@click.option("--photo-root",     default=None,         type=Path)
def inspect_db(db: Path, recognition_db: Path, photo_root: Path | None):
    """Print a summary of the DigiKam and recognition databases."""
    from .db.digikam import load_photos, load_identities
    from .embeddings.kernels import helion_available

    console.print(f"[bold]Loading[/bold] {db} …")
    photos     = load_photos(db, photo_root)
    identities = load_identities(recognition_db)

    t = Table(title="Database Summary")
    t.add_column("Metric",  style="cyan")
    t.add_column("Value",   style="green", justify="right")
    t.add_row("Total photos",           f"{len(photos):,}")
    t.add_row("Photos with faces",      f"{sum(1 for p in photos if p.faces):,}")
    t.add_row("Total face regions",     f"{sum(len(p.faces) for p in photos):,}")
    t.add_row("Tagged face regions",    f"{sum(1 for p in photos for f in p.faces if f.person_name):,}")
    t.add_row("Albums",                 f"{len({p.album_path for p in photos}):,}")
    t.add_row("Known identities",       f"{len(identities):,}")
    t.add_row("Total face embeddings",  f"{sum(len(i.embeddings) for i in identities):,}")
    t.add_row("Helion GPU kernels",     "[green]available[/green]" if helion_available() else "[yellow]CPU fallback[/yellow]")
    console.print(t)

    # Top 10 most-photographed people
    counts: dict[str, int] = {}
    for p in photos:
        for f in p.faces:
            if f.person_name:
                counts[f.person_name] = counts.get(f.person_name, 0) + 1
    if counts:
        top = Table(title="Top 10 Tagged People")
        top.add_column("Name", style="cyan")
        top.add_column("Face regions", justify="right", style="green")
        for name, cnt in sorted(counts.items(), key=lambda x: -x[1])[:10]:
            top.add_row(name, f"{cnt:,}")
        console.print(top)


# ---------------------------------------------------------------------------
# detect-datestamps
# ---------------------------------------------------------------------------

@main.command("detect-datestamps")
@click.option("--db",          default=_DEFAULT_DK, type=Path, show_default=True)
@click.option("--photo-root",  type=Path, default=None,
              help="Mount point for photo files (enables EXIF reading).")
@click.option("--limit",       default=0, type=int,
              help="Process only first N photos (0 = all).")
@click.option("--show-clean",  is_flag=True, default=False,
              help="Also list photos with no anomalies.")
def detect_datestamps(db: Path, photo_root: Path | None, limit: int, show_clean: bool):
    """Detect photos with suspicious or inconsistent datestamps."""
    from .db.digikam import load_photos
    from .metadata.exif import enrich_photos_with_exif
    from .metadata.datestamp import annotate_photos

    photos = load_photos(db, photo_root)
    if limit:
        photos = photos[:limit]

    if photo_root:
        console.print(f"[bold]Reading EXIF from {len(photos)} photos …[/bold]")
        enrich_photos_with_exif(photos)
    else:
        console.print("[yellow]No --photo-root supplied; skipping EXIF read (using DigiKam dates only).[/yellow]")

    annotate_photos(photos)

    flagged = [p for p in photos if p.datestamp_flags]
    shown   = photos if show_clean else flagged

    t = Table(title=f"Datestamp Anomalies  ({len(flagged)}/{len(photos)} flagged)")
    t.add_column("File",          max_width=30)
    t.add_column("Album",         max_width=30)
    t.add_column("EXIF date",     style="cyan")
    t.add_column("Folder date",   style="cyan")
    t.add_column("Flags",         style="red")
    t.add_column("Corrected",     style="green")
    for p in shown[:100]:
        t.add_row(
            p.filename,
            p.album_path,
            str(p.exif_date.date()) if p.exif_date else "—",
            str(p.folder_date)      if p.folder_date else "—",
            "; ".join(p.datestamp_flags) or "clean",
            str(p.corrected_date.date()) if p.corrected_date else "—",
        )
    console.print(t)
    if len(shown) > 100:
        console.print(f"[dim](showing first 100 of {len(shown)})[/dim]")


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------

@main.command("cluster")
@click.option("--db",             default=_DEFAULT_DK,  type=Path, show_default=True)
@click.option("--recognition-db", default=_DEFAULT_REC, type=Path, show_default=True)
@click.option("--thumbnails-db",  default=_DEFAULT_TH,  type=Path, show_default=True)
@click.option("--photo-root",     default=None, type=Path,
              help="If mounted, full-res photos are used; otherwise thumbnails.")
@click.option("--device",         default="cpu",  show_default=True,
              help="'cuda', 'mps', or 'cpu'.")
@click.option("--limit",          default=500, type=int, show_default=True,
              help="Limit number of photos embedded (use 0 for all).")
@click.option("--min-cluster",    default=3,   type=int, show_default=True)
@click.option("--temporal-sigma", default=30.0, type=float, show_default=True,
              help="Temporal window in days for fused similarity.")
@click.option("--semi-supervised", is_flag=True, default=False,
              help="Build DINOv2 identity centroids from recognition.db and use them as HDBSCAN seeds.")
@click.option("--max-seeds",      default=20, type=int, show_default=True,
              help="Max face crops per identity used to build DINOv2 centroids (--semi-supervised only).")
@click.option("--save-embeddings", default=None, type=Path,
              help="Save computed embeddings to this .pt file after extraction.")
@click.option("--load-embeddings", default=None, type=Path,
              help="Load embeddings from this .pt file instead of running DINOv2 (skips extraction).")
@click.option("--pca-dims", default=0, type=int, show_default=True,
              help="Reduce embeddings to this many PCA dimensions before clustering (0=off).")
@click.option("--cluster-epsilon", default=0.0, type=float, show_default=True,
              help="HDBSCAN cluster_selection_epsilon: merge clusters within this distance (0=off).")
@click.option("--nearest-centroid", is_flag=True, default=False,
              help="Skip HDBSCAN; assign each face to nearest identity centroid (requires --semi-supervised).")
@click.option("--min-similarity", default=0.5, type=float, show_default=True,
              help="Cosine similarity threshold for --nearest-centroid assignment (0–1).")
@click.option("--baby-names", required=True, multiple=True, metavar="NAME",
              help="Name to report a per-person cluster breakdown for. Repeat for multiple: "
                   "--baby-names Alice --baby-names Bob.")
@click.option("--export-html", default=None, type=Path, metavar="PATH",
              help="Write a self-contained HTML gallery of face crops grouped by cluster to PATH.")
def cluster(
    db, recognition_db, thumbnails_db, photo_root,
    device, limit, min_cluster, temporal_sigma,
    semi_supervised, max_seeds, save_embeddings, load_embeddings, pca_dims, cluster_epsilon,
    nearest_centroid, min_similarity, baby_names, export_html,
):
    """Extract DINOv2 patch embeddings and cluster faces by identity."""
    from .db.digikam import load_photos, load_identities
    from .metadata.datestamp import annotate_photos
    from .embeddings.extract import EmbeddingExtractor
    from .cluster.identity import run_clustering_pipeline, build_identity_dino_centroids, run_nearest_centroid_pipeline

    console.print("[bold]Loading database …[/bold]")
    photos     = load_photos(db, photo_root)
    if limit:
        photos = photos[:limit]
    identities = load_identities(recognition_db)
    annotate_photos(photos)

    if load_embeddings and load_embeddings.exists():
        console.print(f"[bold]Loading embeddings from[/bold] {load_embeddings} …")
        import torch as _torch
        saved = _torch.load(load_embeddings, weights_only=True)
        embeddings = saved["embeddings"]
        console.print(f"  → {len(embeddings)} face embeddings loaded")
        # Still need an extractor to build identity centroids when requested
        identity_centroids = None
        if semi_supervised:
            if not recognition_db.exists():
                console.print(f"[red]--semi-supervised: recognition-db {recognition_db} not found.[/red]")
                return
            extractor = EmbeddingExtractor(device=device)
            th_db_for_seeds = thumbnails_db if thumbnails_db.exists() else None
            console.print(
                f"[bold]Building identity centroids[/bold] from {len(identities)} identities "
                f"(max {max_seeds} crops each) …"
            )
            identity_centroids = build_identity_dino_centroids(
                identities, extractor, db, th_db_for_seeds,
                max_per_identity=max_seeds,
                photo_root=photo_root,
            )
            console.print(f"  → {len(identity_centroids)} identity centroids built")
            if identity_centroids:
                id_name = {i.id: i.name for i in identities}
                named = [id_name.get(k, str(k)) for k in sorted(identity_centroids)]
                console.print(f"  Seeded: {', '.join(named[:20])}"
                              + (f" … (+{len(named)-20} more)" if len(named) > 20 else ""))
        else:
            extractor = None
    else:
        console.print(f"[bold]Building DINOv2 extractor[/bold] (device={device}) …")
        extractor = EmbeddingExtractor(device=device)

        # Build identity centroids before embedding photos (reuses extractor)
        identity_centroids = None
        if semi_supervised:
            if not recognition_db.exists():
                console.print(f"[red]--semi-supervised: recognition-db {recognition_db} not found.[/red]")
                return
            th_db_for_seeds = thumbnails_db if thumbnails_db.exists() else None
            console.print(
                f"[bold]Building identity centroids[/bold] from {len(identities)} identities "
                f"(max {max_seeds} crops each) …"
            )
            identity_centroids = build_identity_dino_centroids(
                identities, extractor, db, th_db_for_seeds,
                max_per_identity=max_seeds,
                photo_root=photo_root,
            )
            console.print(f"  → {len(identity_centroids)} identity centroids built")
            if identity_centroids:
                id_name = {i.id: i.name for i in identities}
                named = [id_name.get(k, str(k)) for k in sorted(identity_centroids)]
                console.print(f"  Seeded: {', '.join(named[:20])}"
                              + (f" … (+{len(named)-20} more)" if len(named) > 20 else ""))

        console.print(f"[bold]Embedding faces across {len(photos)} photos …[/bold]")
        th_db = thumbnails_db if thumbnails_db.exists() else None
        embeddings = extractor.embed_all_faces(photos, thumbnails_db=th_db)
        console.print(f"  → {len(embeddings)} face embeddings extracted")

        if not embeddings:
            console.print("[red]No embeddings produced — check photo-root / thumbnails-db path.[/red]")
            return

        if save_embeddings:
            import torch as _torch
            console.print(f"[bold]Saving embeddings to[/bold] {save_embeddings} …")
            _torch.save({"embeddings": embeddings}, save_embeddings)
            console.print(f"  → saved {len(embeddings)} embeddings")

    if nearest_centroid:
        if not identity_centroids:
            console.print("[red]--nearest-centroid requires --semi-supervised to build identity centroids.[/red]")
            return
        console.print(f"[bold]Assigning faces to nearest centroid[/bold] (min_similarity={min_similarity}) …")
        results = run_nearest_centroid_pipeline(
            photos, embeddings, identities,
            identity_centroids=identity_centroids,
            min_similarity=min_similarity,
        )
    else:
        results = run_clustering_pipeline(
            photos, embeddings, identities,
            identity_centroids=identity_centroids,
            temporal_sigma=temporal_sigma,
            min_cluster_size=min_cluster,
            pca_dims=pca_dims,
            cluster_epsilon=cluster_epsilon,
        )

    from collections import Counter, defaultdict
    cluster_ids = [r.cluster_id for r in results]
    counts = Counter(cluster_ids)

    t = Table(title=f"Clustering Results  ({len(results)} faces, {len(counts)-1} clusters + noise)")
    t.add_column("Cluster", justify="right")
    t.add_column("Faces",   justify="right", style="green")
    t.add_column("Assigned name", style="cyan")
    t.add_row("noise (-1)", str(counts.get(-1, 0)), "—")
    for cid in sorted(k for k in counts if k >= 0):
        name = next(
            (r.predicted_name for r in results if r.cluster_id == cid and r.predicted_name),
            "unknown",
        )
        t.add_row(str(cid), str(counts[cid]), name or "?")
    console.print(t)

    # Per-person cluster-size breakdown for target identities
    target_names = list(baby_names)
    by_name: dict[str, list[int]] = defaultdict(list)
    for r in results:
        if r.predicted_name in target_names:
            by_name[r.predicted_name].append(counts[r.cluster_id])
    for name in target_names:
        face_counts = sorted(set(by_name[name]), reverse=True)
        n_clusters = len(set(
            r.cluster_id for r in results if r.predicted_name == name and r.cluster_id >= 0
        ))
        total = sum(1 for r in results if r.predicted_name == name and r.cluster_id >= 0)
        console.print(
            f"[cyan]{name}[/cyan]: {n_clusters} clusters, {total} faces — "
            f"top sizes: {face_counts[:10]}"
        )

    # HTML gallery export
    if export_html:
        from .export.html import export_html as _export_html
        console.print(f"[bold]Exporting HTML gallery to[/bold] {export_html} …")
        photos_by_id = {p.id: p for p in photos}
        th_db = thumbnails_db if thumbnails_db.exists() else None
        n_rendered = _export_html(
            results,
            photos_by_id,
            output=export_html,
            thumbnails_db=th_db,
        )
        console.print(f"  → [green]{n_rendered}[/green] face crops written — open [link={export_html.resolve().as_uri()}]{export_html}[/link]")


# ---------------------------------------------------------------------------
# embed — compute & cache per-backbone embeddings (slow; run once per model)
# ---------------------------------------------------------------------------

@main.command("embed")
@click.option("--db",            default=_DEFAULT_DK, type=Path, show_default=True)
@click.option("--thumbnails-db", default=_DEFAULT_TH, type=Path, show_default=True)
@click.option("--photo-root",    default=None, type=Path,
              help="If mounted, full-res photos are used; otherwise thumbnails.")
@click.option("--cache-dir",     default=Path("embeddings"), type=Path, show_default=True,
              help="Directory for per-model embedding caches (emb_<id>.pt).")
@click.option("--backbones", "-b", multiple=True, default=("dinov2",), show_default=True,
              help="Backbone keys to embed. Repeatable. Known: dinov2, arcface, siglip.")
@click.option("--device",        default=None, show_default=True, help="cpu, cuda, mps, or auto-detected if omitted.")
@click.option("--limit",         default=0, type=int, show_default=True,
              help="Embed only the first N photos (0 = all).")
def embed(db, thumbnails_db, photo_root, cache_dir, backbones, device, limit):
    """Compute embeddings for one or more backbones and cache them to disk."""
    from .db.digikam import load_photos
    from .embeddings import registry as reg

    cache_dir.mkdir(parents=True, exist_ok=True)
    console.print("[bold]Loading database …[/bold]")
    photos = load_photos(db, photo_root)
    if limit:
        photos = photos[:limit]
    th_db = thumbnails_db if thumbnails_db.exists() else None
    console.print(f"  → {len(photos):,} photos "
                  f"({sum(len(p.faces) for p in photos):,} face regions)")

    for key in backbones:
        if key not in reg.REGISTRY:
            console.print(f"[red]Unknown backbone {key!r}; known: {sorted(reg.REGISTRY)}[/red]")
            continue
        console.print(f"[bold]Loading backbone[/bold] {key} (device={device or 'auto'}) …")
        try:
            backbone = reg.get_backbone(key, device=device)
        except ImportError as e:
            console.print(f"[red]{e}[/red]")
            continue
        console.print(f"[bold]Embedding with[/bold] {backbone.id} "
                      f"(input={backbone.input}, dim={backbone.dim}) …")
        out = reg.cache_path(cache_dir, backbone.id)
        ckpt = out.with_suffix(".ckpt.pt")  # streams to disk; auto-resumed on restart
        embs = reg.embed_photos(backbone, photos, thumbnails_db=th_db,
                                checkpoint_path=ckpt)
        reg.save_embeddings(out, backbone, embs)
        if ckpt.exists():
            ckpt.unlink()  # clean up checkpoint after successful save
        console.print(f"  → {len(embs):,} embeddings saved to [green]{out}[/green]")


# ---------------------------------------------------------------------------
# label — fuse cached embeddings + EXIF, evaluate, and label the library
# ---------------------------------------------------------------------------

@main.command("label")
@click.option("--db",            default=_DEFAULT_DK, type=Path, show_default=True)
@click.option("--thumbnails-db", default=_DEFAULT_TH, type=Path, show_default=True)
@click.option("--photo-root",    default=None, type=Path)
@click.option("--cache-dir",     default=Path("embeddings"), type=Path, show_default=True,
              help="Directory holding emb_<id>.pt caches written by `embed`.")
@click.option("--reject-threshold", default=0.5, type=float, show_default=True,
              help="Min fused score to assign an identity (else Unknown).")
@click.option("--coherence-lambda", default=0.0, type=float, show_default=True,
              help="0 = coherence flags advisory only; >0 soft-down-weights incoherent pairs.")
@click.option("--ambiguous-margin", default=0.10, type=float, show_default=True,
              help="Flag as ambiguous when top-2 score is within this many pp of top-1 (0=off).")
@click.option("--siglip-certainty", default=0.0, type=float, show_default=True,
              help="SigLIP cascade: when SigLIP top-1 margin >= this, use SigLIP scores directly "
                   "(bypasses DINOv2/GBM for clear-cut cases). Suggested: 0.05.")
@click.option("--min-face-px",   default=48, type=int, show_default=True,
              help="Skip faces whose width or height is below this many pixels (0=all).")
@click.option("--top-k",         default=15, type=int, show_default=True,
              help="Candidate identities scored per face.")
@click.option("--folds",         default=5, type=int, show_default=True,
              help="K for cross-fitting / out-of-fold evaluation.")
@click.option("--eval/--no-eval", default=True, show_default=True,
              help="Run the out-of-fold leaderboard before labeling.")
@click.option("--audit-tagged", is_flag=True, default=False,
              help="Also re-predict already-tagged faces to surface likely mislabels.")
@click.option("--pseudo-labels",          default=None, type=Path,
              help="Predictions JSON from a prior run. High-confidence assignments become training data.")
@click.option("--pseudo-label-min-score", default=0.75, type=float, show_default=True,
              help="Min score threshold for accepting a pseudo-label (used when 'ambiguous' field absent).")
@click.option("--quality-cache",  default=None, type=Path,
              help="Path to quality-score JSON. Computed on first run (~5-10 min), cached for reuse.")
@click.option("--export-html",   default=None, type=Path, help="Write a review gallery to PATH.")
@click.option("--writeback",     default=None, type=Path, help="Write predictions to PATH (.json/.csv).")
def label(db, thumbnails_db, photo_root, cache_dir, reject_threshold, coherence_lambda,
          ambiguous_margin, siglip_certainty, min_face_px, top_k, folds, eval, audit_tagged,
          pseudo_labels, pseudo_label_min_score, quality_cache, export_html, writeback):
    """Late-fusion identity labeling over cached embeddings + EXIF."""
    import lightgbm  # noqa: F401,E402 — must import before torch (macOS OpenMP)
    import torch

    from .db.digikam import load_photos, load_gps
    from .metadata.datestamp import annotate_photos
    from .fusion.features import _UNKNOWN_TAGS
    from .fusion.fusion import crossfit_design, evaluate, train_final
    from .fusion.constraints import reconcile, write_predictions

    console.print("[bold]Loading database …[/bold]")
    photos = load_photos(db, photo_root)
    annotate_photos(photos)
    photo_map = {p.id: p for p in photos}
    gps_map = load_gps(db)
    console.print(f"  → {len(photos):,} photos, {len(gps_map):,} geotagged")

    # Load every per-model embedding cache present.
    embeddings_by_backbone: dict[str, dict] = {}
    scene_backbones: set[str] = set()
    caches = sorted(cache_dir.glob("emb_*.pt"))
    if not caches:
        console.print(f"[red]No emb_*.pt caches in {cache_dir}. Run `babyface embed` first.[/red]")
        return
    for path in caches:
        blob = torch.load(path, weights_only=True)
        bid = blob["backbone_id"]
        embeddings_by_backbone[bid] = blob["embeddings"]
        if blob.get("input") == "whole_image":
            scene_backbones.add(bid)
        console.print(f"  loaded {len(blob['embeddings']):,} from {path.name}  ({bid})")

    # Partition faces by tag status, optionally filtering below min_face_px.
    def _is_unknown(n): return (not n) or n.strip().lower() in _UNKNOWN_TAGS
    known, unknown, untagged = [], [], []
    n_filtered_size = 0
    for p in photos:
        for i, f in enumerate(p.faces):
            if min_face_px > 0 and (f.width < min_face_px or f.height < min_face_px):
                n_filtered_size += 1
                continue
            if f.person_name is None:
                untagged.append((p.id, i, None))
            elif _is_unknown(f.person_name):
                unknown.append((p.id, i, f.person_name))
            else:
                known.append((p.id, i, f.person_name))
    pseudo_known: list[tuple[int, int, str]] = []
    if pseudo_labels and pseudo_labels.exists():
        import json as _json
        raw = _json.loads(pseudo_labels.read_text())
        known_keys = {(pid, fidx) for pid, fidx, _ in known}
        for rec in raw:
            ident = rec.get("predicted_identity")
            if not ident or _is_unknown(ident):
                continue
            if (rec["photo_id"], rec["face_idx"]) in known_keys:
                continue  # already a DigiKam-labeled face
            if rec.get("flags"):
                continue  # skip coherence-flagged predictions
            # Use the explicit ambiguous field when available; fall back to score threshold.
            if "ambiguous" in rec:
                if rec["ambiguous"]:
                    continue
            elif rec["score"] < pseudo_label_min_score:
                continue
            pseudo_known.append((rec["photo_id"], rec["face_idx"], ident))
        console.print(f"  pseudo-labels: {len(pseudo_known):,} added from {pseudo_labels.name}")
    train_faces = known + unknown + pseudo_known
    size_note = f"  filtered {n_filtered_size:,} faces < {min_face_px}px\n" if min_face_px > 0 else ""
    console.print(f"{size_note}  faces — known: {len(known):,}  unknown: {len(unknown):,}  untagged: {len(untagged):,}")

    # Quality-weighted centroids: compute or load Laplacian-variance scores.
    quality_weights = None
    if quality_cache:
        from .fusion.quality import (compute_quality_scores, normalize_scores,
                                     save_quality_cache, load_quality_cache)
        if quality_cache.exists():
            quality_weights = load_quality_cache(quality_cache)
            console.print(f"  quality cache: loaded {len(quality_weights):,} scores from {quality_cache.name}")
        else:
            console.print("[bold]Computing face quality scores (Laplacian variance) …[/bold]")
            raw_scores = compute_quality_scores(train_faces, photo_map)
            quality_weights = normalize_scores(raw_scores)
            save_quality_cache(quality_weights, quality_cache)

    # Out-of-fold evaluation / leaderboard.
    if eval:
        console.print("[bold]Cross-fitting + evaluating …[/bold]")
        design = crossfit_design(train_faces, embeddings_by_backbone, scene_backbones,
                                 photo_map, gps_map, k=folds, top_k=top_k,
                                 quality_weights=quality_weights)
        report = evaluate(design, reject_threshold=reject_threshold,
                         siglip_certainty=siglip_certainty)
        _print_leaderboard(report)

    # Train final model and label.
    console.print("[bold]Training final fusion model …[/bold]")
    model = train_final(train_faces, embeddings_by_backbone, scene_backbones,
                        photo_map, gps_map, k=folds, top_k=top_k,
                        quality_weights=quality_weights,
                        siglip_certainty=siglip_certainty)

    targets = untagged + unknown + (known if audit_tagged else [])
    console.print(f"[bold]Labeling {len(targets):,} faces[/bold] "
                  f"(reject<{reject_threshold}, ambiguous_margin={ambiguous_margin}) …")
    assignments = reconcile(model, targets, embeddings_by_backbone, photo_map, gps_map,
                            reject_threshold=reject_threshold,
                            coherence_lambda=coherence_lambda,
                            ambiguous_margin=ambiguous_margin, top_k=top_k)
    n_named = sum(1 for a in assignments if a.identity and not a.ambiguous)
    n_ambig  = sum(1 for a in assignments if a.ambiguous)
    n_unk    = sum(1 for a in assignments if not a.identity)
    n_flagged = sum(1 for a in assignments if a.flags)
    console.print(f"  → {n_named:,} assigned, {n_ambig:,} ambiguous, "
                  f"{n_unk:,} Unknown, {n_flagged:,} flagged")

    # Cluster the Unknown faces by DINOv2 similarity to surface unnamed repeating people.
    unknown_clusters = []
    if export_html:
        dinov2_key = next((b for b in embeddings_by_backbone if b.startswith("dinov2")), None)
        unk_face_keys = [(a.photo_id, a.face_idx) for a in assignments if not a.identity]
        if dinov2_key and unk_face_keys:
            from .cluster.identity import run_clustering_pipeline
            unk_embs = {k: embeddings_by_backbone[dinov2_key][k]
                        for k in unk_face_keys if k in embeddings_by_backbone[dinov2_key]}
            if unk_embs:
                console.print(f"[bold]Clustering {len(unk_embs):,} Unknown faces …[/bold]")
                unknown_clusters = run_clustering_pipeline(
                    photos, unk_embs, known_identities=[],
                    min_cluster_size=3, temporal_sigma=30.0,
                )
                n_unk_clusters = len({r.cluster_id for r in unknown_clusters if r.cluster_id >= 0})
                n_unk_noise    = sum(1 for r in unknown_clusters if r.cluster_id == -1)
                console.print(f"  → {n_unk_clusters:,} Unknown clusters, {n_unk_noise:,} noise")

    if writeback:
        n = write_predictions(assignments, photo_map, writeback)
        console.print(f"  → {n:,} predictions written to [green]{writeback}[/green]")

    if export_html:
        from .export.html import export_review_html
        th_db = thumbnails_db if thumbnails_db.exists() else None
        n = export_review_html(assignments, photo_map, export_html,
                               thumbnails_db=th_db, unknown_clusters=unknown_clusters)
        console.print(f"  → [green]{n:,}[/green] crops — open "
                      f"[link={export_html.resolve().as_uri()}]{export_html}[/link]")


def _print_leaderboard(report) -> None:
    """Render the per-model vs fused out-of-fold leaderboard."""
    def pct(x):
        return "—" if x != x else f"{x:.1%}"     # x!=x catches NaN

    t = Table(title=f"Out-of-fold top-1 accuracy  ({report.n_faces:,} faces)")
    t.add_column("Model", style="cyan")
    t.add_column("All", justify="right")
    t.add_column("Session-linked", justify="right", style="green")
    t.add_column("Isolated", justify="right", style="yellow")
    for name, r in sorted(report.per_model.items(), key=lambda kv: -(kv[1]["top1"]["all"] or 0)):
        t.add_row(name, pct(r["top1"]["all"]), pct(r["top1"]["linked"]), pct(r["top1"]["isolated"]))
    f = report.fused["top1"]
    t.add_row("[bold]FUSED (GBM)[/bold]",
              f"[bold]{pct(f['all'])}[/bold]", f"[bold]{pct(f['linked'])}[/bold]",
              f"[bold]{pct(f['isolated'])}[/bold]")
    console.print(t)

    rej = report.fused.get("reject", {})
    if rej:
        console.print(f"[dim]Reject accuracy on Unknown faces — "
                      f"all {pct(rej.get('all'))}, isolated {pct(rej.get('isolated'))}[/dim]")
    if report.shap_importance:
        top = ", ".join(f"{n} {v:.3f}" for n, v in report.shap_importance[:6])
        console.print(f"[dim]SHAP importance: {top}[/dim]")


@main.command()
@click.option("--predictions", default="predictions6.json", show_default=True,
              help="Predictions JSON file to serve (relative to repo dir).")
@click.option("--photo-root", type=Path, default=None,
              help="Photo root for serving real face crops (optional).")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8080, show_default=True)
@click.option("--reload", is_flag=True, default=False, help="Dev auto-reload.")
def web(predictions: str, photo_root, host: str, port: int, reload: bool):
    """Launch the interactive labeling web UI."""
    import uvicorn
    from .web.app import create_app
    repo_dir = Path(__file__).parent.parent.parent  # .../babyface-new
    app = create_app(repo_dir, predictions_file=predictions, photo_root=photo_root)
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
