"""
Retrain the run12 fusion model and save weights to weights_run12/.
Mirrors run12.sh label args exactly.
"""
import json, sys
from pathlib import Path
import joblib, torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from babyface.db.digikam import load_photos, load_gps
from babyface.metadata.datestamp import annotate_photos
from babyface.fusion.features import _UNKNOWN_TAGS
from babyface.fusion.fusion import train_final

REPO = Path(__file__).parent
DB   = REPO / "digikam4.db"
CACHE = REPO / "embeddings"
PHOTO_ROOT = Path("/data/photos/photos")
PSEUDO = REPO / "predictions11.json"
QUALITY = REPO / "quality_cache.json"
OUT  = REPO / "weights_run12"
OUT.mkdir(exist_ok=True)

MIN_FACE_PX = 48
SIGLIP_CERTAINTY = 0.05
TOP_K = 15
FOLDS = 5

print("Loading database …")
photos = load_photos(DB, PHOTO_ROOT)
annotate_photos(photos)
photo_map = {p.id: p for p in photos}
gps_map = load_gps(DB)
print(f"  {len(photos):,} photos, {len(gps_map):,} geotagged")

print("Loading embeddings …")
embeddings_by_backbone: dict[str, dict] = {}
scene_backbones: set[str] = set()
for path in sorted(CACHE.glob("emb_*.pt")):
    blob = torch.load(path, weights_only=True)
    bid = blob["backbone_id"]
    embeddings_by_backbone[bid] = blob["embeddings"]
    if blob.get("input") == "whole_image":
        scene_backbones.add(bid)
    print(f"  loaded {len(blob['embeddings']):,} from {path.name}")

def _is_unknown(n): return (not n) or n.strip().lower() in _UNKNOWN_TAGS

known, unknown, untagged = [], [], []
for p in photos:
    for i, f in enumerate(p.faces):
        if f.width < MIN_FACE_PX or f.height < MIN_FACE_PX:
            continue
        if f.person_name is None:
            untagged.append((p.id, i, None))
        elif _is_unknown(f.person_name):
            unknown.append((p.id, i, f.person_name))
        else:
            known.append((p.id, i, f.person_name))

pseudo_known = []
if PSEUDO.exists():
    raw = json.loads(PSEUDO.read_text())
    known_keys = {(pid, fidx) for pid, fidx, _ in known}
    for rec in raw:
        ident = rec.get("predicted_identity")
        if not ident or _is_unknown(ident):
            continue
        if (rec["photo_id"], rec["face_idx"]) in known_keys:
            continue
        if rec.get("flags"):
            continue
        if "ambiguous" in rec:
            if rec["ambiguous"]:
                continue
        elif rec["score"] < 0.5:
            continue
        pseudo_known.append((rec["photo_id"], rec["face_idx"], ident))
    print(f"  pseudo-labels: {len(pseudo_known):,} from {PSEUDO.name}")

train_faces = known + unknown + pseudo_known
print(f"  train faces: {len(train_faces):,}")

quality_weights = None
if QUALITY.exists() and QUALITY.stat().st_size > 2:
    from babyface.fusion.quality import load_quality_cache
    quality_weights = load_quality_cache(QUALITY)
    print(f"  quality: {len(quality_weights):,} scores loaded")

print("Training final fusion model …")
model = train_final(train_faces, embeddings_by_backbone, scene_backbones,
                    photo_map, gps_map, k=FOLDS, top_k=TOP_K,
                    quality_weights=quality_weights,
                    siglip_certainty=SIGLIP_CERTAINTY)

print("Saving weights …")
joblib.dump(model.booster, OUT / "gbm_booster.pkl")
joblib.dump(model.identity_models, OUT / "identity_models.pkl")
joblib.dump({
    "feature_names": model.feature_names,
    "scene_backbones": model.scene_backbones,
    "siglip_certainty": model.siglip_certainty,
}, OUT / "fusion_meta.pkl")

sizes = {p.name: p.stat().st_size for p in OUT.glob("*.pkl")}
print("Done:", {k: f"{v/1024:.1f} KB" for k, v in sizes.items()})
