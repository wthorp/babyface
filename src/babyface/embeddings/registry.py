"""
Pluggable embedding backbones for multi-signal late fusion.

The original tool used a single model (DINOv2 on face crops).  The labeling
pipeline fuses *several* complementary views of each photo:

  - patch  : DINOv2 on the face crop — low-level texture (the baby signal).
  - face   : an adult face recogniser (ArcFace / InsightFace) — strong on
             adults, weak on infants; included as both a workhorse and a
             control to see where it breaks down.
  - scene  : a whole-IMAGE encoder (SigLIP) — captures background / context
             ("non-face context"); two faces in the same room/outing share it.

Each backbone conforms to the `Backbone` interface so the feature-extraction
layer can request embeddings without caring how they're produced.  Backbones
declare what pixels they consume (`input`): a tight face crop vs. the whole
image.  Heavy/optional dependencies (insightface, transformers) are imported
lazily inside each backbone's loader so importing this module never forces
them — mirroring how kernels.py treats Helion and identity.py treats hdbscan.

Per-model embedding caches are kept in separate files (one per backbone id) so
you can compute, store, and A/B models independently without re-running the
others.  Caches are keyed by (photo_id, face_idx) for face/patch backbones and
(photo_id, -1) for scene backbones (one embedding per image, shared by every
face in it).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
from PIL import Image, ImageOps

InputKind = Literal["face_crop", "whole_image"]
BackboneKind = Literal["patch", "face", "scene"]


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class Backbone(abc.ABC):
    """
    A model that turns PIL images into a [N, D] embedding tensor.

    Subclasses set `id`, `kind`, `input`, and `dim`, and implement `embed`.
    `embed` must return one row per input image, in order.  Rows for images
    the backbone could not process (e.g. a face crop with no detectable face
    for an alignment-dependent face model) are filled with NaN so callers can
    drop them with `~torch.isnan(rows).any(dim=1)`.
    """

    id: str
    kind: BackboneKind
    input: InputKind
    dim: int

    @abc.abstractmethod
    def embed(self, images: list[Image.Image], batch_size: int = 32) -> torch.Tensor:
        """Return [len(images), dim] (un-normalised; callers L2-normalise)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# DINOv2 — patch-mean on the face crop (wraps the existing extractor)
# ---------------------------------------------------------------------------

class DinoV2Backbone(Backbone):
    """
    DINOv2 patch-mean over the face crop.  Composes the existing
    EmbeddingExtractor so the proven extraction path is reused verbatim.
    """

    kind: BackboneKind = "patch"
    input: InputKind = "face_crop"

    def __init__(self, device: str | torch.device | None = None,
                 model_name: str = "vit_base_patch14_dinov2.lvd142m"):
        from .extract import EmbeddingExtractor  # local import: heavy (timm)
        self.id = f"dinov2:{model_name}"
        self._extractor = EmbeddingExtractor(device=device)
        # vit_base = 768; resolve from the loaded model so larger variants work.
        self.dim = int(self._extractor.model.num_features)

    def embed(self, images: list[Image.Image], batch_size: int = 32) -> torch.Tensor:
        if not images:
            return torch.empty(0, self.dim)
        return self._extractor.embed_batch(images, batch_size=batch_size).cpu()


# ---------------------------------------------------------------------------
# SigLIP — whole-image scene/context encoder
# ---------------------------------------------------------------------------

class SiglipSceneBackbone(Backbone):
    """
    Whole-image embedding from a SigLIP vision-language encoder (HuggingFace
    transformers).  Used to capture background/context shared by faces in the
    same photo or outing.  Consumes the WHOLE image, not the face crop.
    """

    kind: BackboneKind = "scene"
    input: InputKind = "whole_image"

    def __init__(self, device: str | torch.device | None = None,
                 model_name: str = "google/siglip2-base-patch16-224"):
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "SiglipSceneBackbone needs `transformers`. "
                "Install with: uv add transformers"
            ) from e
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.id = f"siglip:{model_name}"
        self._model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self._processor = AutoProcessor.from_pretrained(model_name)
        self.dim = int(self._model.config.vision_config.hidden_size)

    @torch.inference_mode()
    def embed(self, images: list[Image.Image], batch_size: int = 32) -> torch.Tensor:
        if not images:
            return torch.empty(0, self.dim)
        out: list[torch.Tensor] = []
        for start in range(0, len(images), batch_size):
            chunk = [im.convert("RGB") for im in images[start:start + batch_size]]
            inputs = self._processor(images=chunk, return_tensors="pt").to(self.device)
            feats = self._model.get_image_features(**inputs)  # [B, D] or ModelOutput in transformers 5+
            if not isinstance(feats, torch.Tensor):
                feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats[0]
            out.append(feats.float().cpu())
        return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# ArcFace (InsightFace) — adult face recogniser on the face crop
# ---------------------------------------------------------------------------

class ArcFaceBackbone(Backbone):
    """
    ArcFace embedding via InsightFace's buffalo_l pack.

    Caveat that matters: ArcFace expects a 5-point-landmark-aligned face, which
    InsightFace produces by *running its own detector* on the input.  We feed
    it the padded DigiKam crop and let it detect+align inside that crop.  When
    no face is detected (common for infants, profiles, motion blur) the row is
    returned as NaN — this is itself informative: it's part of why face
    recognisers underperform on babies, which the fusion model can learn from.
    """

    kind: BackboneKind = "face"
    input: InputKind = "face_crop"
    dim = 512  # ArcFace embedding size (buffalo_l)

    def __init__(self, device: str | torch.device | None = None,
                 model_name: str = "buffalo_l"):
        try:
            import insightface  # noqa: F401
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "ArcFaceBackbone needs `insightface` + `onnxruntime`. "
                "Install with: uv add insightface onnxruntime"
            ) from e
        self.id = f"arcface:{model_name}"
        dev = str(device) if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "cuda" in dev else ["CPUExecutionProvider"])
        self._app = FaceAnalysis(name=model_name, providers=providers)
        self._app.prepare(ctx_id=0 if "cuda" in dev else -1)

    def embed(self, images: list[Image.Image], batch_size: int = 32) -> torch.Tensor:
        # InsightFace operates per-image on BGR numpy arrays; no batching API.
        import numpy as np
        rows: list[torch.Tensor] = []
        for im in images:
            arr = np.asarray(im.convert("RGB"))[:, :, ::-1]  # RGB→BGR
            faces = self._app.get(arr)
            if not faces:
                rows.append(torch.full((self.dim,), float("nan")))
                continue
            # Largest detected face within the crop wins.
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            rows.append(torch.from_numpy(face.normed_embedding.copy()).float())
        return torch.stack(rows) if rows else torch.empty(0, self.dim)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BackboneSpec:
    """A recommended candidate model and how to build it."""
    key: str                       # short CLI-friendly handle, e.g. "dinov2"
    factory: Callable[..., Backbone]
    note: str

# Curated starting slate (the models we agreed to benchmark). `key` is what a
# user types; the factory builds it on demand so unused models cost nothing.
REGISTRY: dict[str, BackboneSpec] = {
    "dinov2":  BackboneSpec("dinov2",  DinoV2Backbone,
                            "DINOv2 patch-mean on face crop — baby texture signal"),
    "arcface": BackboneSpec("arcface", ArcFaceBackbone,
                            "ArcFace/InsightFace on face crop — adult workhorse + baby control"),
    "siglip":  BackboneSpec("siglip",  SiglipSceneBackbone,
                            "SigLIP on whole image — scene/context signal"),
}


def available_backbones() -> dict[str, str]:
    """{key: note} for every registered candidate (does not load them)."""
    return {k: s.note for k, s in REGISTRY.items()}


def get_backbone(key: str, device: str | torch.device | None = None, **kwargs) -> Backbone:
    """Instantiate a backbone by key. Heavy deps load here, not at import."""
    if key not in REGISTRY:
        raise KeyError(f"Unknown backbone {key!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[key].factory(device=device, **kwargs)


# ---------------------------------------------------------------------------
# Per-model embedding cache
# ---------------------------------------------------------------------------

def cache_path(cache_dir: Path, backbone_id: str) -> Path:
    """One .pt per backbone; ':' and '/' in ids are filename-sanitised."""
    safe = backbone_id.replace("/", "__").replace(":", "_")
    return Path(cache_dir) / f"emb_{safe}.pt"


def save_embeddings(path: Path, backbone: Backbone,
                    embeddings: dict[tuple[int, int], torch.Tensor]) -> None:
    """Persist a backbone's embeddings with enough metadata to validate on load."""
    torch.save({
        "backbone_id": backbone.id,
        "kind": backbone.kind,
        "input": backbone.input,
        "dim": backbone.dim,
        "embeddings": embeddings,
    }, path)


def load_embeddings(path: Path) -> dict[tuple[int, int], torch.Tensor]:
    """Load a per-model cache, returning the {(photo_id, face_idx): tensor} map."""
    blob = torch.load(path, weights_only=True)
    return blob["embeddings"]


# ---------------------------------------------------------------------------
# Backbone-agnostic embedding driver
# ---------------------------------------------------------------------------

def embed_photos(
    backbone: Backbone,
    photos: list,                       # list[db.models.Photo]
    thumbnails_db: Path | None = None,
    batch_size: int = 32,
    chunk_size: int = 500,
    progress_every: int = 5000,
    checkpoint_path: Path | None = None,
) -> dict[tuple[int, int], torch.Tensor]:
    """
    Run a backbone over `photos`, dispatching on backbone.input:

      face_crop   → one embedding per face region, key (photo_id, face_idx).
      whole_image → one embedding per photo,        key (photo_id, -1),
                    shared by every face in that image (scene context).

    Returns {(photo_id, face_idx_or_-1): embedding}.  NaN rows produced by a
    backbone (e.g. ArcFace finding no face) are dropped here, so absence of a
    key means "this backbone had no signal for that face" — which downstream
    feature extraction treats as a missing feature, not a zero.

    For whole_image backbones, chunk_size is capped at batch_size to keep peak
    memory bounded — full-res photos can be 36 MB+ decompressed, so we never
    accumulate more than one model batch worth of images at a time.

    If checkpoint_path is given, embeddings are streamed to disk every
    progress_every photos. On restart the checkpoint is loaded automatically,
    and already-embedded photo IDs are skipped — so a crash loses at most one
    progress_every window of work.
    """
    from ..db.digikam import open_photo_image
    from .extract import crop_face

    # Whole-image backbones load full-res photos; cap accumulation to one
    # model batch so peak RAM ≈ batch_size × max_resized_image, not chunk_size.
    effective_chunk = batch_size if backbone.input == "whole_image" else chunk_size

    # Resume from checkpoint if available.
    result: dict[tuple[int, int], torch.Tensor] = {}
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            ckpt = torch.load(checkpoint_path, weights_only=True)
            result = ckpt.get("embeddings", {})
            print(f"  [checkpoint] resumed {len(result)} embeddings from {checkpoint_path}",
                  flush=True)
        except Exception as e:
            print(f"  [checkpoint] could not load {checkpoint_path}: {e} — starting fresh",
                  flush=True)

    already_done: set[int] = {k[0] for k in result}  # photo ids already embedded
    done = 0

    for chunk_start in range(0, len(photos), effective_chunk):
        chunk = photos[chunk_start:chunk_start + effective_chunk]
        keys: list[tuple[int, int]] = []
        imgs: list[Image.Image] = []

        for photo in chunk:
            if photo.id in already_done:
                continue
            try:
                img = open_photo_image(photo, thumbnails_db) if thumbnails_db else (
                    Image.open(photo.full_path) if photo.full_path.exists() else None)
                if img is None:
                    continue
                img = ImageOps.exif_transpose(img)
                img_rgb = img.convert("RGB")
                del img  # free the original; img_rgb may still be full-res
            except Exception:
                continue

            if backbone.input == "whole_image":
                # Resize to 256px — SigLIP only needs 224px, so this loses nothing.
                img_rgb.thumbnail((256, 256), Image.LANCZOS)
                keys.append((photo.id, -1))
                imgs.append(img_rgb)
            else:  # face_crop
                for idx, face in enumerate(photo.faces):
                    crop = crop_face(img_rgb, (face.x, face.y, face.width, face.height))
                    if crop is None:
                        continue
                    keys.append((photo.id, idx))
                    imgs.append(crop)

        if imgs:
            embs = backbone.embed(imgs, batch_size=batch_size)
            ok = ~torch.isnan(embs).any(dim=1)
            for i, key in enumerate(keys):
                if bool(ok[i]):
                    result[key] = embs[i]
            del imgs  # release resized images after embedding

        done += len(chunk)
        if done % progress_every == 0 or chunk_start + effective_chunk >= len(photos):
            print(f"  [{backbone.id}: {done}/{len(photos)} photos, {len(result)} embedded]",
                  flush=True)
            if checkpoint_path:
                torch.save({"backbone_id": backbone.id, "kind": backbone.kind,
                            "input": backbone.input, "dim": backbone.dim,
                            "embeddings": result}, checkpoint_path)

    return result
