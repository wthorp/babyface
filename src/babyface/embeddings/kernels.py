"""
Helion GPU kernels for face embedding similarity.

Helion (https://github.com/pytorch/helion) is a Python-embedded DSL that
compiles tiled loops into autotuned Triton kernels.  It requires:
  - Linux
  - CUDA or ROCm GPU
  - PyTorch >= 2.9 and Triton >= 3.5

When those conditions are not met (e.g. macOS dev machine, CPU-only CI)
HELION_AVAILABLE is False and every public function falls back to plain
PyTorch, preserving identical numerical results.

Two kernels are provided:

  pairwise_cosine(a, b) → [N, M]
    Computes the full cosine similarity matrix between two sets of
    L2-normalised embeddings.  Structurally a tiled matmul; used when
    matching DINOv2 face embeddings against known-identity centroids.

  fused_similarity(emb_a, emb_b, t_a, t_b, temporal_sigma) → [N, M]
    Computes cosine_sim × temporal_proximity in a single kernel pass,
    avoiding a second memory round-trip over the large [N, M] output.
    temporal_proximity = exp(-(Δt)² / (2σ²)) where Δt is in days.
    Used to build the affinity matrix for HDBSCAN clustering.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

try:
    import helion
    import helion.language as hl
    _HELION_AVAILABLE = True
except (ImportError, RuntimeError):
    _HELION_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helion kernels (compiled only when _HELION_AVAILABLE == True)
# ---------------------------------------------------------------------------

if _HELION_AVAILABLE:

    @helion.kernel()
    def _cosine_kernel(
        a: torch.Tensor,    # [N, D]  L2-normalised
        b: torch.Tensor,    # [M, D]  L2-normalised
        out: torch.Tensor,  # [N, M]  output — cosine similarities
    ) -> torch.Tensor:
        """
        Tiled matmul over L2-normalised embeddings.

        For unit-norm vectors: a @ b.T == cosine_similarity(a, b).
        Helion's autotuner picks block sizes and loop orders for the
        target hardware automatically.
        """
        n, d = a.size()
        m, _ = b.size()
        for tile_n, tile_m in hl.tile([n, m]):
            acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
            for tile_d in hl.tile(d):
                acc = torch.addmm(acc, a[tile_n, tile_d], b[tile_m, tile_d].T)
            out[tile_n, tile_m] = acc
        return out

    @helion.kernel()
    def _fused_similarity_kernel(
        emb_a: torch.Tensor,       # [N, D]  L2-normalised
        emb_b: torch.Tensor,       # [M, D]  L2-normalised
        t_a:   torch.Tensor,       # [N]     corrected date in days since epoch
        t_b:   torch.Tensor,       # [M]     corrected date in days since epoch
        sigma2: torch.Tensor,      # []      scalar: 2 * temporal_sigma^2
        out:   torch.Tensor,       # [N, M]
    ) -> torch.Tensor:
        """
        Fused kernel: cosine_sim(emb_i, emb_j) × exp(-(t_i - t_j)² / sigma2).

        By computing both terms inside the same tiled loop we write [N, M]
        only once, saving the extra bandwidth of a second kernel that would
        read the cosine result and multiply by the temporal weight.
        """
        n, d = emb_a.size()
        m, _ = emb_b.size()
        for tile_n, tile_m in hl.tile([n, m]):
            # Embedding cosine similarity (matmul of normalised vectors)
            cos = hl.zeros([tile_n, tile_m], dtype=torch.float32)
            for tile_d in hl.tile(d):
                cos = torch.addmm(cos, emb_a[tile_n, tile_d], emb_b[tile_m, tile_d].T)
            # Temporal Gaussian: exp(-Δt² / sigma2)
            dt = t_a[tile_n].unsqueeze(1) - t_b[tile_m].unsqueeze(0)  # [bn, bm]
            temporal = torch.exp(-(dt * dt) / sigma2)
            out[tile_n, tile_m] = cos * temporal
        return out


# ---------------------------------------------------------------------------
# Public API  (dispatch to Helion or PyTorch fallback)
# ---------------------------------------------------------------------------

def pairwise_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute [N, M] cosine similarity matrix.

    Inputs do NOT need to be pre-normalised; this function normalises them.
    Uses the Helion kernel when running on a CUDA device, otherwise falls
    back to a plain PyTorch matmul.
    """
    a_n = F.normalize(a.float(), dim=-1)
    b_n = F.normalize(b.float(), dim=-1)
    if _HELION_AVAILABLE and a.is_cuda:
        out = torch.empty(a_n.size(0), b_n.size(0), dtype=torch.float32, device=a.device)
        return _cosine_kernel(a_n, b_n, out)
    return a_n @ b_n.T


def fused_similarity(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    timestamps_a: torch.Tensor,
    timestamps_b: torch.Tensor,
    temporal_sigma: float = 30.0,
) -> torch.Tensor:
    """
    Compute [N, M] combined similarity: cosine_sim × temporal_proximity.

    temporal_sigma — characteristic time scale in days (default 30).
      A value of 30 means photos taken on the same day score ≈1.0 on the
      temporal component; photos 60 days apart score ≈0.14.

    Uses the fused Helion kernel on CUDA; falls back to two-step PyTorch.
    """
    emb_a_n = F.normalize(emb_a.float(), dim=-1)
    emb_b_n = F.normalize(emb_b.float(), dim=-1)
    t_a = timestamps_a.float().to(emb_a.device)
    t_b = timestamps_b.float().to(emb_b.device)
    sigma2 = torch.tensor(2.0 * temporal_sigma ** 2, dtype=torch.float32, device=emb_a.device)

    if _HELION_AVAILABLE and emb_a.is_cuda:
        out = torch.empty(emb_a_n.size(0), emb_b_n.size(0), dtype=torch.float32, device=emb_a.device)
        return _fused_similarity_kernel(emb_a_n, emb_b_n, t_a, t_b, sigma2, out)

    # CPU / non-Helion fallback — same math, two passes
    cos = emb_a_n @ emb_b_n.T                               # [N, M]
    dt  = t_a.unsqueeze(1) - t_b.unsqueeze(0)               # [N, M]
    return cos * torch.exp(-(dt * dt) / sigma2)


def helion_available() -> bool:
    return _HELION_AVAILABLE
