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

## Notes

- The databases are read-only — nothing is written back to DigiKam.
- GPU kernels (via [Helion](https://github.com/pytorch/helion)) are used automatically on Linux with a CUDA GPU. On macOS or CPU-only machines the tool falls back to standard PyTorch with identical results.
- The `--save-embeddings` / `--load-embeddings` flags are useful for iterating on clustering parameters without re-running the model each time.
