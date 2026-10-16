"""
Löwdin Symmetric Orthogonalization Module
-----------------------------------------
Provides numerically stable implementations of Löwdin Symmetric Orthogonalization (LSO),
matrix square roots, inverse square roots, multi-factor risk profile distances, and
strategy uniqueness indices for quantitative portfolio analysis.
"""

from __future__ import annotations

import numpy as np
import scipy.spatial.distance as ssd


def compute_lowdin_matrices(
    correlation_matrix: np.ndarray,
    eps: float = 1e-10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Löwdin symmetric orthogonalization transformation matrices:
      - C^(1/2)  : Symmetric square root (order-invariant shock generator)
      - C^(-1/2) : Symmetric inverse square root (factor whitening / decorrelation)
      - eigenvalues: Sorted eigenvalue spectrum

    Guarantees:
      - Symmetry: M = M.T
      - Order-invariance: Permuting inputs permutes rows/cols equivariantly.
      - PSD robustness: Eigenvalues below `eps` are clipped.
    """
    corr = np.asarray(correlation_matrix, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("Input correlation matrix must be square (N x N).")

    # Enforce strict symmetry
    corr_sym = (corr + corr.T) / 2.0

    # Symmetric eigendecomposition (dsyevd)
    eigvals, eigvecs = np.linalg.eigh(corr_sym)

    # Inline eigenvalue clipping to ensure positive semi-definiteness
    eigvals_clipped = np.maximum(eigvals, eps)

    # Construct C^(1/2) = V * diag(sqrt(Lambda)) * V^T
    sqrt_diag = np.sqrt(eigvals_clipped)
    c_half = eigvecs @ np.diag(sqrt_diag) @ eigvecs.T

    # Construct C^(-1/2) = V * diag(1 / sqrt(Lambda)) * V^T
    inv_sqrt_diag = 1.0 / sqrt_diag
    c_inv_half = eigvecs @ np.diag(inv_sqrt_diag) @ eigvecs.T

    # Symmetrize outputs to mitigate floating point precision noise
    c_half = (c_half + c_half.T) / 2.0
    c_inv_half = (c_inv_half + c_inv_half.T) / 2.0

    return c_half, c_inv_half, eigvals


def compute_strategy_uniqueness(correlation_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the Löwdin 'Strategy Uniqueness' Index:
    In Löwdin orthogonalization, the diagonal elements (C^(1/2))_ii represent the self-overlap
    and autonomy of each asset's risk profile.

    Returns:
      1D NumPy array of scores in [0, 1]. A score near 1.0 indicates pure alpha / true autonomy,
      while lower values indicate collinearity and shared factor variance.
    """
    c_half, _, _ = compute_lowdin_matrices(correlation_matrix)
    diag_vals = np.diag(c_half)
    return np.clip(diag_vals, 0.0, 1.0)


def compute_lowdin_profile_distance(correlation_matrix: np.ndarray) -> np.ndarray:
    """
    Computes a condensed Euclidean distance matrix based on the Löwdin orthogonal
    risk profiles (rows of C^(1/2)).

    Strategies that load onto identical underlying risk factors will have zero distance,
    while orthogonal strategies will have maximum distance.

    Returns:
      1D condensed distance array suitable for `scipy.cluster.hierarchy.linkage`.
    """
    c_half, _, _ = compute_lowdin_matrices(correlation_matrix)
    # Pairwise Euclidean distance between rows of C^(1/2)
    dist_matrix = ssd.squareform(ssd.pdist(c_half, metric="euclidean"))
    np.fill_diagonal(dist_matrix, 0.0)
    return ssd.squareform(dist_matrix)
