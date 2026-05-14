"""
Patch-level face embedding extraction using DINOv2.

Why DINOv2 for infants?
  Standard face recognisers (FaceNet, ArcFace) are trained on adult faces
  with strong shape/beard/wrinkle cues.  Infant faces look nearly identical
  to these models because those cues don't exist yet.  DINOv2's patch tokens
  capture low-level texture patterns (skin tone variation, ear shape, hairline
  geometry) that differ between infants and remain stable across growth spurts.
  We use the mean of all patch tokens rather than just the CLS token to get a
  richer spatial summary of the face crop.

Model: vit_base_patch14_dinov2.lvd142m from timm (~330MB on first use).
Embedding dim: 768 per patch, returned as a single mean-pooled 768-dim vector.
"""
from __future__ import annotations
import io
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Register HEIC/HEIF support if available
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

_MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"
_FACE_PADDING = 0.3   # fractional padding added around bounding box


def _load_model(device: torch.device):
    model = timm.create_model(_MODEL_NAME, pretrained=True, num_classes=0)
    model = model.to(device).eval()
    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg)
    return model, transform


def crop_face(img: Image.Image, rect: tuple[int, int, int, int], padding: float = _FACE_PADDING) -> Image.Image | None:
    """
    Crop a face region with contextual padding.
    rect = (x, y, width, height) in pixels (DigiKam's <rect> format).
    Returns None for degenerate crops (< 4×4 px after clamping).
    """
    x, y, w, h = rect
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(img.width,  x + w + pad_x)
    y1 = min(img.height, y + h + pad_y)
    if x0 >= x1 or y0 >= y1:
        return None
    crop = img.crop((x0, y0, x1, y1))
    if crop.width < 4 or crop.height < 4:
        return None
    return crop


class EmbeddingExtractor:
    """
    Wraps a DINOv2 ViT-B/14 to produce 768-dim patch-mean embeddings.

    Usage:
        extractor = EmbeddingExtractor()          # auto-selects CUDA if available
        emb = extractor.embed_face(photo_path, (x, y, w, h))   # torch.Tensor [768]
        embs = extractor.embed_batch([img1, img2])              # torch.Tensor [N, 768]
    """

    def __init__(self, device: str | torch.device | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, self.transform = _load_model(self.device)

    @torch.inference_mode()
    def embed_patch_mean(self, img: Image.Image) -> torch.Tensor:
        """768-dim mean of all patch tokens for a single PIL image."""
        x = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        # get_intermediate_layers returns list of [1, num_patches, D] tensors
        patch_tokens = self.model.get_intermediate_layers(x, n=1)[0]  # [1, P, D]
        return patch_tokens.squeeze(0).mean(0)  # [D]

    @torch.inference_mode()
    def embed_batch(self, crops: list[Image.Image], batch_size: int = 32) -> torch.Tensor:
        """
        Embed a list of PIL face crops; returns [N, D] tensor.
        Processes in mini-batches to avoid OOM on large collections.
        """
        results: list[torch.Tensor] = []
        for start in range(0, len(crops), batch_size):
            chunk = crops[start : start + batch_size]
            tensors = torch.stack([
                self.transform(c.convert("RGB")) for c in chunk
            ]).to(self.device)
            patch_tokens = self.model.get_intermediate_layers(tensors, n=1)[0]  # [B, P, D]
            results.append(patch_tokens.mean(1))   # [B, D]
        return torch.cat(results, dim=0)

    def embed_face(
        self,
        photo: "Photo",  # db.models.Photo
        face_idx: int,
        thumbnails_db: Path | None = None,
    ) -> torch.Tensor | None:
        """
        Embed a single face region.  Opens full photo if available on disk,
        otherwise falls back to the local thumbnail BLOB.
        Returns None on any error (missing file, decode failure, etc.).
        """
        from ..db.digikam import open_photo_image
        from ..db.models import Photo

        try:
            img = open_photo_image(photo, thumbnails_db) if thumbnails_db else (
                Image.open(photo.full_path) if photo.full_path.exists() else None
            )
            if img is None:
                return None
            face = photo.faces[face_idx]
            crop = crop_face(img, (face.x, face.y, face.width, face.height))
            if crop is None:
                return None
            return self.embed_patch_mean(crop)
        except Exception:
            return None

    def embed_all_faces(
        self,
        photos: list["Photo"],
        thumbnails_db: Path | None = None,
        batch_size: int = 32,
        chunk_size: int = 500,
        progress_every: int = 5000,
    ) -> dict[tuple[int, int], torch.Tensor]:
        """
        Extract DINOv2 embeddings for every face region across all photos.

        Returns {(photo_id, face_idx): embedding_tensor}.
        Processes photos in chunks to avoid collecting all crops in RAM at once.
        """
        import sys
        from ..db.digikam import open_photo_image

        result: dict[tuple[int, int], torch.Tensor] = {}
        photos_done = 0
        faces_done = 0

        for chunk_start in range(0, len(photos), chunk_size):
            chunk = photos[chunk_start : chunk_start + chunk_size]

            keys: list[tuple[int, int]] = []
            crops: list[Image.Image] = []

            for photo in chunk:
                if not photo.faces:
                    continue
                try:
                    img = open_photo_image(photo, thumbnails_db) if thumbnails_db else (
                        Image.open(photo.full_path) if photo.full_path.exists() else None
                    )
                    if img is None:
                        continue
                    img_rgb = img.convert("RGB")
                    for idx, face in enumerate(photo.faces):
                        crop = crop_face(img_rgb, (face.x, face.y, face.width, face.height))
                        if crop is None:
                            continue
                        keys.append((photo.id, idx))
                        crops.append(crop)
                except Exception:
                    continue

            if crops:
                embeddings = self.embed_batch(crops, batch_size=batch_size)
                for i, key in enumerate(keys):
                    result[key] = embeddings[i]
                faces_done += len(crops)

            photos_done += len(chunk)
            if photos_done % progress_every == 0 or chunk_start + chunk_size >= len(photos):
                print(f"  [{photos_done}/{len(photos)} photos, {faces_done} faces embedded]",
                      flush=True)

        return result
