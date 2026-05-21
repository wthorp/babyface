# babyface

A tool for clustering infant photos by identity — built to solve a specific problem during a photo library migration.

## Background

I've been migrating my photo library from DigiKam to [Immich](https://immich.app/). DigiKam is great for a lot of things, and its face tagging is precise — you can draw exact bounding boxes and name them. But there's a catch: it doesn't scale. You have to manually tag every single image, and it's a real slog when you have thousands of photos.

The bigger problem is that neither DigiKam's face engine nor most off-the-shelf face recognition tools work well on babies. Adult face recognition relies on features like bone structure, facial hair, and wrinkles — none of which exist on an infant. Two babies from the same family can look nearly identical to a model trained on adult faces, while the same baby at 3 months versus 9 months can look completely different.

`babyface` takes a different approach: instead of using a face recognition model, it uses a vision model (DINOv2) that looks at low-level texture patterns — skin tone, ear shape, hairline geometry — that differ between individuals and stay relatively stable as a baby grows. It also factors in when a photo was taken, since photos from the same time period are more likely to be the same child.

The goal is to pre-cluster a library so that you're confirming identities rather than tagging from scratch.

## What it does

- **`inspect-db`** — summarises your DigiKam databases: how many photos, face regions, tagged identities, and known embeddings.
- **`detect-datestamps`** — scans your library for photos with suspicious or inconsistent timestamps (common with baby photos that include scans, imports, or cameras that had their clocks reset).
- **`cluster`** — extracts embeddings for every face region across your library and groups them into clusters. Each cluster gets a predicted name if there are enough DigiKam-tagged faces to vote on it.
- **`embed`** — computes face/scene embeddings for one or more vision models and caches them to disk (one file per model), so models can be A/B-compared without re-running.
- **`label`** — fuses those embeddings with EXIF context (time, camera, GPS) using a gradient-boosted model to predict an identity for every face. Evaluates which models work best, then exports a review gallery and a predictions file.

### Single model vs. multi-model fusion

The original `cluster` command uses one model (DINOv2) and groups faces. The newer
`embed` + `label` commands take a different, **transductive** approach aimed at labeling
*this* library as accurately as possible:

- **Several complementary signals**, fused: DINOv2 on the face crop (low-level texture —
  the baby signal), an adult face recogniser (ArcFace/InsightFace), and a whole-image
  scene encoder (SigLIP) for background/context — plus EXIF time, camera, and GPS.
- **Late fusion with a gradient-boosted tree** that learns how much to trust each signal,
  with [SHAP](https://github.com/shap/shap) explaining *which* signal drove each decision.
- **Honest evaluation**: faces are scored against centroids built from *other* faces
  (K-fold cross-fitting, no self-leakage), and accuracy is reported separately for
  faces that share a photo session with a tagged face vs. **isolated** faces — the hard,
  error-prone cases.
- **Constraint reconciliation**: a person can't appear twice in one photo (resolved by
  optimal assignment), and assignments that contradict an identity's established
  time/camera/location track are flagged for review (e.g. "this is the only photo of
  Eric ever taken 6000 km away — probably a misclassification").

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (for dependency management)
- Your DigiKam database files: `digikam4.db`, `recognition.db`, and optionally `thumbnails-digikam.db`
- A GPU helps significantly for large libraries, but CPU works too

## Setup

```bash
./run.sh --help
```

`run.sh` handles installing `uv` if needed, syncing dependencies, and running the tool. On first run it will download the DINOv2 model weights (~330 MB).

## Usage

### Inspect your databases

```bash
./run.sh inspect-db --db /path/to/digikam4.db --recognition-db /path/to/recognition.db
```

### Check for datestamp anomalies

```bash
./run.sh detect-datestamps --db /path/to/digikam4.db --photo-root /path/to/photos
```

### Cluster faces by identity

```bash
./run.sh cluster \
  --db /path/to/digikam4.db \
  --recognition-db /path/to/recognition.db \
  --thumbnails-db /path/to/thumbnails-digikam.db \
  --baby-names Alice \
  --baby-names Bob
```

`--baby-names` is required and can be repeated for each person you want a per-cluster breakdown for. Everything else has sensible defaults.

### Key cluster options

| Flag | Default | What it does |
|---|---|---|
| `--limit` | 500 | Max photos to embed (0 = all) |
| `--device` | cpu | `cpu`, `cuda`, or `mps` |
| `--semi-supervised` | off | Use existing DigiKam face tags to seed clusters |
| `--save-embeddings` | — | Cache embeddings to a file for faster re-runs |
| `--load-embeddings` | — | Load cached embeddings instead of re-running the model |
| `--pca-dims` | 0 | Reduce embedding dimensions before clustering (speeds things up) |

Run `./run.sh cluster --help` for the full list.

### Multi-model fusion labeling

The fusion pipeline needs extra dependencies (the boosting model and the optional
backbones). Install them with:

```bash
uv sync --extra fusion
```

Then embed once per model, and label as many times as you like:

```bash
# 1. Compute & cache embeddings (slow; one file per model under embeddings/)
./run.sh embed --backbones dinov2 --backbones arcface --backbones siglip \
  --device cuda --photo-root /path/to/photos

# 2. Fuse + evaluate + label (fast; iterate on thresholds and outputs)
./run.sh label \
  --reject-threshold 0.5 \
  --export-html review.html \
  --writeback predictions.json
```

`run2.sh` wraps both steps with sensible defaults (it embeds only if the cache is
missing, then labels). Set `FORCE_EMBED=1` to re-embed, or `DEVICE=cuda` to use a GPU.

**Outputs:**

- A **leaderboard** table — per-model vs. fused top-1 accuracy, split into session-linked
  and isolated faces. This is the "which model is best?" answer.
- A **review gallery** (`--export-html`) — faces grouped by predicted identity, with a
  "Needs review" section collecting every coherence-flagged face, and a marker where a
  prediction disagrees with an existing DigiKam tag.
- A **predictions file** (`--writeback`, `.json` or `.csv`) — every face with its
  predicted identity, confidence, and flags, for review or import. Nothing is written
  back to DigiKam.

| Flag | Default | What it does |
|---|---|---|
| `--backbones` | `dinov2` | Model(s) to embed (`embed`). Repeat for several. |
| `--reject-threshold` | 0.5 | Min fused score to assign an identity; below this → Unknown |
| `--coherence-lambda` | 0.0 | 0 = flags are advisory; >0 down-weights incoherent assignments |
| `--folds` | 5 | K for cross-fitting / out-of-fold evaluation |
| `--audit-tagged` | off | Also re-predict already-tagged faces to surface likely mislabels |
| `--no-eval` | — | Skip the leaderboard and go straight to labeling |

> **macOS note:** PyTorch and LightGBM ship conflicting OpenMP runtimes; importing
> torch first can crash. The `label` command and `run2.sh` import LightGBM first to
> avoid this. Linux/CUDA machines are unaffected.

## Notes

- The databases are read-only — nothing is written back to DigiKam.
- GPU kernels (via [Helion](https://github.com/pytorch/helion)) are used automatically on Linux with a CUDA GPU. On macOS or CPU-only machines the tool falls back to standard PyTorch with identical results.
- The `--save-embeddings` / `--load-embeddings` flags are useful for iterating on clustering parameters without re-running the model each time.
