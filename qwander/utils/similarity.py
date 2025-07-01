"""
Class for comparing node embedding matrices via structural similarity metrics.
"""

from typing import Dict

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances


class EmbeddingComparator:
    """
    Compare two embedding matrices using a battery of similarity and divergence metrics.
    """

    @staticmethod
    def pearson_correlation(A: np.ndarray,
                            B: np.ndarray) -> float:
        """
        Pearson correlation with handling for constant input arrays.

        Returns 1.0 if arrays are constant and equal, 0.0 if constant and not equal.
        """
        A, B = np.asarray(A), np.asarray(B)
        if A.ndim != 1 or B.ndim != 1:
            raise ValueError(f"Inputs must be 1D; got {A.shape}, {B.shape}")
        if A.shape[0] != B.shape[0]:
            raise ValueError(f"Input lengths must match; got {A.shape[0]} and {B.shape[0]}.")

        stdA, stdB = A.std(), B.std()
        if stdA == 0 or stdB == 0:
            return 1.0 if np.allclose(A, B) else 0.0

        return float(np.corrcoef(A, B)[0, 1])

    @staticmethod
    def spearman_correlation(A: np.ndarray,
                             B: np.ndarray) -> float:
        """
        Spearman rank correlation of two 1D arrays.
        """
        A, B = np.asarray(A), np.asarray(B)
        if A.ndim != 1 or B.ndim != 1:
            raise ValueError(f"Inputs must be 1D; got {A.shape}, {B.shape}")
        if A.shape[0] != B.shape[0]:
            raise ValueError(f"Input lengths must match; got {A.shape[0]} and {B.shape[0]}.")

        rho, _ = spearmanr(A, B)

        return float(rho)

    def compare(self,
                matrix_a: np.ndarray,
                matrix_b: np.ndarray) -> Dict[str, float]:
        """
        Compare two embedding matrices using a set of structural similarity metrics.

        Parameters
        ----------
        matrix_a, matrix_b : array_like, shape (n_nodes, d)
            Embedding matrices to compare. Must have the same shape.

        Returns
        -------
        Dict[str, float]
            {
                "distance_correlation":
                    Pearson correlation of pairwise Euclidean distances,
                "cosine_correlation":
                      Pearson correlation of pairwise cosine similarities,
                "spearman_distance_correlation":
                  Spearman correlation of Euclidean distances,
                "spearman_cosine_correlation":
                    Spearman correlation of cosine similarities,
            }
        """
        A = np.asarray(matrix_a, dtype=float)
        B = np.asarray(matrix_b, dtype=float)

        # Validate input
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError(f"Both inputs must be 2D (got shapes {A.shape}, {B.shape}).")
        if A.shape != B.shape:
            raise ValueError(f"Shapes must match (got {A.shape}, {B.shape}).")
        n, _ = A.shape
        if n < 2:
            return dict(
                distance_correlation=1.0,
                cosine_correlation=1.0,
                spearman_distance_correlation=1.0,
                spearman_cosine_correlation=1.0,
            )

        # Pairwise Euclidean distances
        dist_a = pairwise_distances(A, metric="euclidean")
        dist_b = pairwise_distances(B, metric="euclidean")

        # Pairwise cosine similarities
        norm_a = np.linalg.norm(A, axis=1, keepdims=True)
        norm_b = np.linalg.norm(B, axis=1, keepdims=True)
        normed_a = A / np.where(norm_a == 0, 1.0, norm_a)
        normed_b = B / np.where(norm_b == 0, 1.0, norm_b)
        cos_a = normed_a @ normed_a.T
        cos_b = normed_b @ normed_b.T

        # Flatten upper triangle indices (i < j)
        idx = np.triu_indices(n, k=1)
        dA_flat = dist_a[idx]
        dB_flat = dist_b[idx]
        cA_flat = cos_a[idx]
        cB_flat = cos_b[idx]

        # Metrics
        pearson_dist = self.pearson_correlation(dA_flat, dB_flat)
        pearson_cos = self.pearson_correlation(cA_flat, cB_flat)
        spearman_dist = self.spearman_correlation(dA_flat, dB_flat)
        spearman_cos = self.spearman_correlation(cA_flat, cB_flat)

        return {
            "distance_correlation": pearson_dist,
            "cosine_correlation": pearson_cos,
            "spearman_distance_correlation": spearman_dist,
            "spearman_cosine_correlation": spearman_cos,
        }
