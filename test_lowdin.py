"""
Unit tests for Löwdin Symmetric Orthogonalization
"""

import unittest
import numpy as np
import scipy.spatial.distance as ssd

from lowdin_orthogonalization import (
    compute_lowdin_matrices,
    compute_strategy_uniqueness,
    compute_lowdin_profile_distance
)

class TestLowdinOrthogonalization(unittest.TestCase):
    def setUp(self):
        # 4x4 realistic correlation matrix
        self.C = np.array([
            [1.00, 0.65, 0.20, -0.15],
            [0.65, 1.00, 0.40, -0.05],
            [0.20, 0.40, 1.00,  0.30],
            [-0.15, -0.05, 0.30, 1.00]
        ])

    def test_symmetry(self):
        c_half, c_inv_half, _ = compute_lowdin_matrices(self.C)
        np.testing.assert_allclose(c_half, c_half.T, atol=1e-12)
        np.testing.assert_allclose(c_inv_half, c_inv_half.T, atol=1e-12)

    def test_reconstruction(self):
        # C^(1/2) * C^(1/2) should equal C
        c_half, _, _ = compute_lowdin_matrices(self.C)
        reconstructed_C = c_half @ c_half
        np.testing.assert_allclose(reconstructed_C, self.C, atol=1e-10)

    def test_inverse_identity(self):
        # C^(1/2) * C^(-1/2) should equal I
        c_half, c_inv_half, _ = compute_lowdin_matrices(self.C)
        identity = c_half @ c_inv_half
        np.testing.assert_allclose(identity, np.eye(self.C.shape[0]), atol=1e-10)

    def test_order_invariance_permutation(self):
        # Permuting rows and columns should exactly permute C^(1/2) without distortion
        perm = [2, 0, 3, 1]
        P = np.eye(4)[perm]
        C_perm = P @ self.C @ P.T
        
        c_half_orig, _, _ = compute_lowdin_matrices(self.C)
        c_half_perm, _, _ = compute_lowdin_matrices(C_perm)
        
        # Expected permuted c_half
        expected_perm = P @ c_half_orig @ P.T
        np.testing.assert_allclose(c_half_perm, expected_perm, atol=1e-10)

    def test_non_positive_definite_handling(self):
        # Create rank-deficient / non-PSD matrix
        v = np.array([[1.0], [1.0], [1.0]])
        # Highly collinear matrix with negative eigenvalue perturbed
        C_singular = np.ones((3, 3))
        C_singular[0, 1] = C_singular[1, 0] = 1.05 # invalid correlation
        
        c_half, c_inv_half, eigvals = compute_lowdin_matrices(C_singular, eps=1e-8)
        self.assertTrue(np.all(np.isfinite(c_half)))
        self.assertTrue(np.all(np.isfinite(c_inv_half)))

    def test_uniqueness_scores(self):
        scores = compute_strategy_uniqueness(self.C)
        self.assertEqual(len(scores), 4)
        # Scores should be bounded in [0, 1]
        self.assertTrue(np.all(scores >= 0.0) and np.all(scores <= 1.0))

    def test_profile_distance(self):
        dist = compute_lowdin_profile_distance(self.C)
        # For N=4, condensed distance length is 4*(4-1)/2 = 6
        self.assertEqual(len(dist), 6)
        self.assertTrue(np.all(dist >= 0.0))

if __name__ == "__main__":
    unittest.main()
