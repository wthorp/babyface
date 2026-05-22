"""
OOF failure analysis for the fusion model.

Answers:
  1. OOF accuracy excluding ambiguous predictions (top-1 - top-2 < 0.10)
  2. Error breakdown by face size, top identity, error type (wrong / rejected)
"""
from __future__ import annotations

import sys
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO / "src"))

from babyface.db.digikam import load_photos
from babyface.db.models import Photo
from babyface.fusion.features import _UNKNOWN_TAGS
from babyface.fusion.fusion import crossfit_design, _lgbm_params

DB         = REPO / "digikam4.db"
CACHE_DIR  = REPO / "embeddings"
MIN_FACE   = 48
AMB_MARGIN = 0.10
REJECT_TH  = 0.5


def _is_unk(n: str) -> bool:
    return (not n) or n.strip().lower() in _UNKNOWN_TAGS


print("Loading database ...")
photos = load_photos(DB)
photo_map: dict[int, Photo] = {p.id: p for p in photos}

face_size: dict[tuple[int, int], tuple[int, int]] = {}
for p in photos:
    for i, f in enumerate(p.faces):
        face_size[(p.id, i)] = (f.width, f.height)

print("Loading embeddings ...")
embeddings_by_backbone: dict = {}
scene_backbones: set[str] = set()
for path in sorted(CACHE_DIR.glob("emb_*.pt")):
    blob = torch.load(path, weights_only=True)
    bid = blob["backbone_id"]
    embeddings_by_backbone[bid] = blob["embeddings"]
    if blob.get("input") == "whole_image":
        scene_backbones.add(bid)
    print(f"  loaded {len(blob['embeddings']):,} from {path.name}  ({bid})")

gps_map: dict[int, tuple[float, float]] = {}
with sqlite3.connect(DB) as cx:
    for row in cx.execute(
        "SELECT imageid, latitudeNumber, longitudeNumber FROM ImagePositions "
        "WHERE latitudeNumber IS NOT NULL"
    ):
        gps_map[row[0]] = (row[1], row[2])

print("Building tagged faces list ...")
tagged_faces: list[tuple[int, int, str]] = []
for p in photos:
    for i, f in enumerate(p.faces):
        if not f.person_name or _is_unk(f.person_name):
            continue
        w, h = f.width, f.height
        if w < MIN_FACE or h < MIN_FACE:
            continue
        tagged_faces.append((p.id, i, f.person_name))
print(f"  {len(tagged_faces):,} tagged faces for crossfit")

print("Cross-fitting (10-15 min) ...")
d = crossfit_design(tagged_faces, embeddings_by_backbone, scene_backbones,
                    photo_map, gps_map, k=5, seed=42, top_k=15)

print("Running OOF LightGBM ...")
import lightgbm as lgb
oof = np.full(len(d.y), np.nan)
for kk in np.unique(d.fold):
    tr, te = d.fold != kk, d.fold == kk
    clf = lgb.LGBMClassifier(**_lgbm_params())
    clf.fit(d.X[tr], d.y[tr])
    oof[te] = clf.predict_proba(d.X[te])[:, 1]

print("Analysing failures ...")

by_face: dict[int, list[int]] = {}
for r in range(len(d.y)):
    by_face.setdefault(int(d.face_id[r]), []).append(r)


def size_bucket(w: int, h: int) -> str:
    m = min(w, h)
    if m < 64:  return "tiny(<64)"
    if m < 96:  return "small(64-96)"
    if m < 140: return "medium(96-140)"
    return "large(>=140)"


results = []
for fi, rows in by_face.items():
    tn = d.face_true[fi]
    if _is_unk(tn):
        continue
    sub = np.array(rows)
    sc = oof[sub]
    order = np.argsort(-sc)
    top1_row   = int(sub[order[0]])
    top1_score = float(sc[order[0]])
    top2_score = float(sc[order[1]]) if len(order) > 1 else 0.0
    top1_name  = d.cand_names[top1_row]

    if top1_score < REJECT_TH:
        err_type = "rejected"
        correct  = False
    elif top1_name == tn:
        err_type = "correct"
        correct  = True
    else:
        err_type = "wrong_identity"
        correct  = False

    ambiguous = (top1_score >= REJECT_TH) and (top1_score - top2_score < AMB_MARGIN)
    pid, fidx, _ = tagged_faces[fi]
    w, h = face_size.get((pid, fidx), (0, 0))
    linked = d.face_linked.get(fi, False)

    results.append({
        "fi": fi, "true": tn, "pred": top1_name,
        "top1": top1_score, "top2": top2_score,
        "correct": correct, "err_type": err_type,
        "ambiguous": ambiguous, "linked": linked,
        "size_bucket": size_bucket(w, h), "min_dim": min(w, h),
    })

total   = len(results)
n_corr  = sum(r["correct"] for r in results)
n_ambig = sum(r["ambiguous"] for r in results)

non_ambig   = [r for r in results if not r["ambiguous"]]
na_correct  = sum(r["correct"] for r in non_ambig)
acc_all     = n_corr / total
acc_excl    = na_correct / len(non_ambig) if non_ambig else float("nan")

print(f"\n{'='*60}")
print("Q1 -- Accuracy (excl. ambiguous predictions)")
print(f"{'='*60}")
print(f"  All faces:       {acc_all:.1%}  ({n_corr}/{total})")
print(f"  Excl. ambiguous: {acc_excl:.1%}  ({na_correct}/{len(non_ambig)})")
print(f"  Ambiguous faces: {n_ambig} ({n_ambig/total:.1%} of known faces in OOF)")

print(f"\n{'='*60}")
print("Q2a -- Error rate by face size bucket")
print(f"{'='*60}")
by_size: dict[str, list] = defaultdict(list)
for r in results:
    by_size[r["size_bucket"]].append(r)
for bucket in ["tiny(<64)", "small(64-96)", "medium(96-140)", "large(>=140)"]:
    rr = by_size.get(bucket, [])
    if not rr:
        continue
    acc = sum(r["correct"] for r in rr) / len(rr)
    print(f"  {bucket:<22}  n={len(rr):5,}  acc={acc:.1%}  err={1-acc:.1%}")

print(f"\n{'='*60}")
print("Q2b -- Top-20 identities by error count")
print(f"{'='*60}")
by_ident: dict[str, list] = defaultdict(list)
for r in results:
    by_ident[r["true"]].append(r)
ident_stats = sorted(
    [(name, len(rr), sum(not r["correct"] for r in rr)) for name, rr in by_ident.items()],
    key=lambda x: -x[2],
)
print(f"  {'Identity':<28} {'n':>6}  {'errors':>7}  {'acc':>6}")
print(f"  {'-'*28} {'----':>6}  {'------':>7}  {'---':>6}")
for name, n, n_err in ident_stats[:20]:
    print(f"  {name:<28} {n:>6,}  {n_err:>7,}  {1-n_err/n:>6.1%}")

print(f"\n{'='*60}")
n_err_total = total - n_corr
print(f"Q2c -- Error type (all errors = {n_err_total:,})")
print(f"{'='*60}")
type_counts: dict[str, int] = defaultdict(int)
for r in results:
    if not r["correct"]:
        type_counts[r["err_type"]] += 1
for k, v in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {k:<22}  {v:,}  ({v/n_err_total:.1%} of errors)")

print("\n  Split by session-linkage:")
for linked in [True, False]:
    label = "session-linked" if linked else "isolated"
    sub   = [r for r in results if r["linked"] == linked]
    n_sub = len(sub)
    if not n_sub:
        continue
    n_sub_err = sum(not r["correct"] for r in sub)
    acc_sub   = 1 - n_sub_err / n_sub
    tc: dict[str, int] = defaultdict(int)
    for r in sub:
        if not r["correct"]:
            tc[r["err_type"]] += 1
    bd = ", ".join(f"{k}={v}" for k, v in sorted(tc.items(), key=lambda x: -x[1]))
    print(f"  {label:<20} n={n_sub:5,}  acc={acc_sub:.1%}  errors: {bd}")

print(f"\n{'='*60}")
print("Q2d -- Error rate by size x linkage")
print(f"{'='*60}")
print(f"  {'size bucket':<22} {'linked acc':>12} {'isolated acc':>13}")
for bucket in ["tiny(<64)", "small(64-96)", "medium(96-140)", "large(>=140)"]:
    lr = [r for r in by_size.get(bucket, []) if r["linked"]]
    ir = [r for r in by_size.get(bucket, []) if not r["linked"]]
    la = sum(r["correct"] for r in lr) / len(lr) if lr else float("nan")
    ia = sum(r["correct"] for r in ir) / len(ir) if ir else float("nan")
    print(f"  {bucket:<22} {la:>11.1%}  {ia:>11.1%}")

print("\nDone.")
