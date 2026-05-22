"""
Gradient-boosted late fusion + honest evaluation.

The fusion model is a LightGBM binary classifier over the per-(face, identity)
feature rows from features.py: it learns P(this identity is correct for this
face) by weighing the per-model similarities against the EXIF context.  To
identify a face we score all its candidate rows and take the argmax (subject to
a reject threshold for "no known person").

Honesty has two independent leakage risks, handled by two layers of K-fold:

  1. Centroid self-leakage — a face must not be compared to a centroid it
     helped build, or its own similarity is inflated.  We K-fold the tagged
     faces and, for each fold, build identity centroids/EXIF stats from the
     *other* folds.  Same-album siblings stay in the training folds, so the
     session signal is preserved; only the face itself is held out.

  2. GBM overfitting — the classifier is evaluated out-of-fold on the same
     folds, so reported accuracy is on faces whose rows never trained it.

For deployment we then refit identity models on *all* tagged faces and train
the GBM on all cross-fitted rows (matching the inference distribution, where a
face is never in its own centroid).

LightGBM and SHAP are imported lazily (optional `fusion` extra), so importing
this module costs nothing until you actually train.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
# NB: torch is intentionally NOT imported here (only used in string type hints).
# On macOS, importing torch's OpenMP runtime before LightGBM's segfaults; keeping
# this module torch-free lets the CLI import lightgbm first as the mitigation.

from .features import build_identity_models, build_feature_table, FeatureTable, _UNKNOWN_TAGS


# ---------------------------------------------------------------------------
# Cross-fitted feature design over all tagged faces
# ---------------------------------------------------------------------------

@dataclass
class CrossfitDesign:
    """Leakage-free feature rows for every tagged face, plus eval bookkeeping."""
    X: np.ndarray                 # [n_rows, n_feats]
    y: np.ndarray                 # [n_rows] 1.0/0.0
    feature_names: list[str]
    cand_names: list[str]         # candidate identity per row
    face_id: np.ndarray           # [n_rows] global face index (argmax grouping)
    fold: np.ndarray              # [n_rows] fold each row's face belongs to
    session_linked: np.ndarray    # [n_rows] bool
    face_true: dict[int, str]     # global face index → true tag (may be Unknown)
    face_linked: dict[int, bool]  # global face index → is this face session-linked


def crossfit_design(
    tagged_faces: list[tuple[int, int, str]],   # (photo_id, face_idx, true_name)
    embeddings_by_backbone: dict[str, dict[tuple[int, int], torch.Tensor]],
    scene_backbones: set[str],
    photo_map: dict[int, "object"],
    gps_map: dict[int, tuple[float, float]] | None = None,
    k: int = 5,
    seed: int = 42,
    top_k: int = 15,
    quality_weights: dict | None = None,
) -> CrossfitDesign:
    rng = np.random.default_rng(seed)
    fold_of = rng.integers(0, k, size=len(tagged_faces))
    face_index = {(pid, fidx): i for i, (pid, fidx, _) in enumerate(tagged_faces)}
    face_true = {i: name for i, (_, _, name) in enumerate(tagged_faces)}

    Xs, ys, names, fids, folds, links = [], [], [], [], [], []
    feat_names: list[str] = []

    for kk in range(k):
        train_recs = [tagged_faces[i] for i in range(len(tagged_faces)) if fold_of[i] != kk]
        test_recs  = [tagged_faces[i] for i in range(len(tagged_faces)) if fold_of[i] == kk]
        if not test_recs:
            continue
        models = build_identity_models(train_recs, embeddings_by_backbone,
                                       scene_backbones, photo_map, gps_map,
                                       quality_weights=quality_weights)
        ft = build_feature_table(test_recs, models, embeddings_by_backbone,
                                 scene_backbones, photo_map, gps_map, top_k=top_k)
        if ft.X.size == 0:
            continue
        feat_names = ft.feature_names
        Xs.append(ft.X)
        ys.append(ft.y)
        names.extend(n for _, _, n in ft.rows)
        fids.extend(face_index[(pid, fidx)] for pid, fidx, _ in ft.rows)
        folds.extend([kk] * len(ft.rows))
        links.extend(ft.session_linked.tolist())

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    cand = names
    face_id = np.asarray(fids, dtype=np.int64)
    fold = np.asarray(folds, dtype=np.int64)
    linked = np.asarray(links, dtype=bool)

    # Face-level linkage: known faces use the true candidate's link flag;
    # Unknown faces use "any candidate's album was seen in training".
    face_linked: dict[int, bool] = {}
    for r in range(len(cand)):
        fi = int(face_id[r])
        tn = face_true[fi]
        is_unk = (not tn) or tn.strip().lower() in _UNKNOWN_TAGS
        if is_unk:
            face_linked[fi] = face_linked.get(fi, False) or bool(linked[r])
        elif cand[r] == tn:
            face_linked[fi] = bool(linked[r])
    return CrossfitDesign(X, y, feat_names, cand, face_id, fold, linked,
                          face_true, face_linked)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _is_unknown(name: str) -> bool:
    return (not name) or name.strip().lower() in _UNKNOWN_TAGS


def _per_face_accuracy(
    score: np.ndarray,            # [n_rows] higher = more likely correct
    d: CrossfitDesign,
    mask: np.ndarray,             # [n_rows] which rows participate (e.g. test fold)
    reject_threshold: float | None = None,
) -> dict[str, dict[str, float]]:
    """
    Top-1 accuracy grouped per face, stratified all / session-linked / isolated.
    Known faces: argmax candidate must equal the true tag.  Unknown faces (only
    scored when reject_threshold is set): correct iff max score < threshold.
    """
    # Gather rows per face within the mask
    by_face: dict[int, list[int]] = {}
    for r in np.nonzero(mask)[0]:
        by_face.setdefault(int(d.face_id[r]), []).append(int(r))

    strata = {"all": [0, 0], "linked": [0, 0], "isolated": [0, 0]}     # [correct, total]
    rej = {"all": [0, 0], "linked": [0, 0], "isolated": [0, 0]}

    for fi, rows in by_face.items():
        tn = d.face_true[fi]
        unk = _is_unknown(tn)
        sub = np.array(rows)
        sc = score[sub]
        best = int(sub[int(np.argmax(sc))])
        strat = "linked" if d.face_linked.get(fi, False) else "isolated"

        if unk:
            if reject_threshold is None:
                continue
            ok = float(np.max(sc)) < reject_threshold
            for s in ("all", strat):
                rej[s][1] += 1
                rej[s][0] += int(ok)
        else:
            if tn not in {d.cand_names[r] for r in rows}:
                ok = False                       # true identity wasn't a candidate
            else:
                ok = d.cand_names[best] == tn
            for s in ("all", strat):
                strata[s][1] += 1
                strata[s][0] += int(ok)

    out = {s: (c / t if t else float("nan")) for s, (c, t) in strata.items()}
    out_n = {f"n_{s}": t for s, (_, t) in strata.items()}
    result = {"top1": out, **{k: v for k, v in out_n.items()}}
    if reject_threshold is not None:
        result["reject"] = {s: (c / t if t else float("nan")) for s, (c, t) in rej.items()}
    return result


# ---------------------------------------------------------------------------
# Evaluation: per-model baselines + fused, out-of-fold
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    fused: dict
    per_model: dict[str, dict]
    shap_importance: list[tuple[str, float]] = field(default_factory=list)
    n_faces: int = 0
    n_rows: int = 0


def _lgbm_params() -> dict:
    return dict(objective="binary", n_estimators=300, learning_rate=0.05,
                num_leaves=31, min_child_samples=40, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0, is_unbalance=True,
                verbose=-1, n_jobs=-1)


def evaluate(d: CrossfitDesign, reject_threshold: float = 0.5) -> EvalReport:
    """Out-of-fold fused accuracy + single-model baselines, stratified."""
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("evaluate() needs LightGBM. Install: uv add lightgbm") from e

    # ---- fused: OOF LightGBM ----
    oof = np.full(len(d.y), np.nan)
    for kk in np.unique(d.fold):
        tr, te = d.fold != kk, d.fold == kk
        clf = lgb.LGBMClassifier(**_lgbm_params())
        clf.fit(d.X[tr], d.y[tr])
        oof[te] = clf.predict_proba(d.X[te])[:, 1]
    fused = _per_face_accuracy(oof, d, mask=np.ones(len(d.y), bool),
                               reject_threshold=reject_threshold)

    # ---- per-model baselines: rank by a single similarity feature ----
    per_model: dict[str, dict] = {}
    sim_cols = [(i, n) for i, n in enumerate(d.feature_names) if n.startswith("sim_")]
    for col, name in sim_cols:
        s = np.nan_to_num(d.X[:, col], nan=-np.inf)
        per_model[name] = _per_face_accuracy(s, d, mask=np.ones(len(d.y), bool),
                                              reject_threshold=None)

    n_faces = len(set(d.face_id.tolist()))
    report = EvalReport(fused=fused, per_model=per_model,
                        n_faces=n_faces, n_rows=len(d.y))

    # ---- SHAP global importance (optional) ----
    try:
        import shap
        clf = lgb.LGBMClassifier(**_lgbm_params()).fit(d.X, d.y)
        expl = shap.TreeExplainer(clf)
        sv = expl.shap_values(d.X[:min(5000, len(d.X))])
        sv = sv[1] if isinstance(sv, list) else sv      # binary → class-1 contributions
        imp = np.abs(sv).mean(axis=0)
        report.shap_importance = sorted(zip(d.feature_names, imp.tolist()),
                                        key=lambda kv: -kv[1])
    except ImportError:
        pass
    return report


# ---------------------------------------------------------------------------
# Deployment: train final model on all tagged faces
# ---------------------------------------------------------------------------

@dataclass
class FusionModel:
    booster: "object"             # fitted LGBMClassifier
    identity_models: dict         # built from ALL tagged faces
    feature_names: list[str]
    scene_backbones: set[str]


def train_final(
    tagged_faces: list[tuple[int, int, str]],
    embeddings_by_backbone: dict[str, dict[tuple[int, int], torch.Tensor]],
    scene_backbones: set[str],
    photo_map: dict[int, "object"],
    gps_map: dict[int, tuple[float, float]] | None = None,
    k: int = 5,
    seed: int = 42,
    top_k: int = 15,
    quality_weights: dict | None = None,
) -> FusionModel:
    """Cross-fit rows (for an inference-matched training distribution), fit the
    GBM on all of them, and refit identity models on every tagged face."""
    import lightgbm as lgb
    d = crossfit_design(tagged_faces, embeddings_by_backbone, scene_backbones,
                        photo_map, gps_map, k=k, seed=seed, top_k=top_k,
                        quality_weights=quality_weights)
    clf = lgb.LGBMClassifier(**_lgbm_params()).fit(d.X, d.y)
    full_models = build_identity_models(tagged_faces, embeddings_by_backbone,
                                        scene_backbones, photo_map, gps_map,
                                        quality_weights=quality_weights)
    return FusionModel(clf, full_models, d.feature_names, scene_backbones)
