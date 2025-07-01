"""
Unit tests for qw_embed.utils.coins module.
"""

import unittest

import numpy as np

from qwander.qwander.utils.coins import (
    hadamard_coin,
    grover_coin,
    fourier_coin,
    identity_coin,
    random_coin
)


class CoinTestMixin:
    """
    Utility helpers reused across all coin-operator tests.
    """

    @staticmethod
    def is_unitary(mat: np.ndarray, atol: float = 1e-8) -> bool:
        """
        Check unitarity:  ‖U U† − I‖ ≤ atol.
        """
        return np.allclose(mat @ mat.conj().T, np.eye(mat.shape[0]), atol=atol)

    def assert_is_complex128(self, mat: np.ndarray) -> None:
        """
        Assert that the matrix dtype is exactly complex128.
        """
        self.assertEqual(mat.dtype, np.complex128, "dtype must be complex128")


class TestHadamardCoin(unittest.TestCase, CoinTestMixin):
    """
    Tests for the 2×2 Hadamard coin.
    """

    def test_shape_and_values(self) -> None:
        """
        Matrix must be 2×2 and match the analytical Hadamard form.
        """
        H = hadamard_coin()
        expected = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        self.assertEqual(H.shape, (2, 2))
        np.testing.assert_allclose(H, expected, atol=1e-8)

    def test_unitarity(self) -> None:
        """
        H · H† should equal the identity matrix.
        """
        H = hadamard_coin()
        self.assertTrue(self.is_unitary(H))

    def test_dtype_is_complex128(self) -> None:
        """
        Return value must use complex128 for consistency across coins.
        """
        H = hadamard_coin()
        self.assert_is_complex128(H)

    def test_return_is_fresh_instance(self) -> None:
        """
        Each call should return an independent array (no shared state).
        """
        H1 = hadamard_coin()
        H2 = hadamard_coin()

        # Mutate H1 and ensure H2 is unchanged
        H1[0, 0] += 1.0
        self.assertFalse(np.allclose(H1, H2))


class TestGroverCoin(unittest.TestCase, CoinTestMixin):
    """
    Tests for the variable-dimension Grover diffusion coin.
    """

    def test_invalid_dimension_raises(self) -> None:
        """
        d must be positive; d ≤ 0 should raise ValueError.
        """
        for d in (-1, 0):
            with self.subTest(d=d):
                with self.assertRaises(ValueError):
                    grover_coin(d)

    def test_shape_and_values(self) -> None:
        """
        Matrix definition should equal 2·J/d − I for d ≥ 1.
        """
        for d in (1, 2, 3, 5):
            with self.subTest(d=d):
                G = grover_coin(d)
                self.assertEqual(G.shape, (d, d))

                if d == 1:
                    expected = np.array([[1.0]])
                else:
                    J = np.ones((d, d))
                    expected = 2 * J / d - np.eye(d)

                np.testing.assert_allclose(G, expected, atol=1e-8)

    def test_unitarity(self) -> None:
        """
        Grover coin must be unitary for all tested d.
        """
        for d in (1, 2, 3, 5):
            with self.subTest(d=d):
                self.assertTrue(self.is_unitary(grover_coin(d)))

    def test_dtype_is_complex128(self) -> None:
        """
        Return dtype should be complex128.
        """
        for d in (1, 2, 4):
            with self.subTest(d=d):
                self.assert_is_complex128(grover_coin(d))

    def test_return_is_fresh_instance(self) -> None:
        """
        Each call returns an independent array (no shared mutable state).
        """
        d = 3
        G1 = grover_coin(d)
        G2 = grover_coin(d)
        G1[0, 0] += 1.0
        self.assertFalse(np.allclose(G1, G2, atol=1e-12))


class TestFourierCoin(unittest.TestCase, CoinTestMixin):
    """
    Tests for the variable-dimension discrete Fourier coin.
    """

    def test_invalid_dimension_raises(self) -> None:
        """
        d must be positive; non-positive d should raise ValueError.
        """
        for d in (-2, 0):
            with self.subTest(d=d):
                with self.assertRaises(ValueError):
                    fourier_coin(d)

    def test_shape_and_magnitude(self) -> None:
        """
        Matrix must be d×d and each entry must have magnitude 1/√d.
        """
        for d in (1, 2, 3, 4, 5):
            with self.subTest(d=d):
                F = fourier_coin(d)
                self.assertEqual(F.shape, (d, d))
                np.testing.assert_allclose(np.abs(F), np.ones((d, d)) / np.sqrt(d), atol=1e-8)

    def test_unitarity(self) -> None:
        """
        Fourier coin must be unitary.
        """
        for d in (1, 2, 3, 4, 5):
            with self.subTest(d=d):
                self.assertTrue(self.is_unitary(fourier_coin(d)))

    def test_dtype_is_complex128(self) -> None:
        """
        Return dtype should be complex128.
        """
        self.assert_is_complex128(fourier_coin(3))

    def test_return_is_fresh_instance(self) -> None:
        """
        Each call returns a fresh array (no shared mutable state).
        """
        d = 4
        F1 = fourier_coin(d)
        F2 = fourier_coin(d)
        F1[0, 0] += 1.0
        self.assertFalse(np.allclose(F1, F2, atol=1e-12))


class TestIdentityCoin(unittest.TestCase, CoinTestMixin):
    """
    Tests for the variable-dimension identity coin.
    """

    def test_invalid_dimension_raises(self) -> None:
        """
        d must be positive; non-positive d should raise ValueError.
        """
        for d in (-3, 0):
            with self.subTest(d=d):
                with self.assertRaises(ValueError):
                    identity_coin(d)

    def test_shape_and_entries(self) -> None:
        """
        Matrix must equal the d×d identity.
        """
        for d in (1, 2, 5):
            with self.subTest(d=d):
                I = identity_coin(d)
                self.assertEqual(I.shape, (d, d))
                np.testing.assert_allclose(I, np.eye(d), atol=1e-8)

    def test_unitarity(self) -> None:
        """
        Identity coin must be unitary (trivial).
        """
        for d in (1, 2, 5):
            with self.subTest(d=d):
                self.assertTrue(self.is_unitary(identity_coin(d)))

    def test_dtype_is_complex128(self) -> None:
        """
        Return dtype should be complex128.
        """
        self.assert_is_complex128(identity_coin(2))

    def test_return_is_fresh_instance(self) -> None:
        """
        Each call returns a fresh array (no shared mutable state).
        """
        d = 3
        I1 = identity_coin(d)
        I2 = identity_coin(d)
        I1[0, 0] += 1.0
        self.assertFalse(np.allclose(I1, I2, atol=1e-12))


class TestRandomCoin(unittest.TestCase, CoinTestMixin):
    """
    Tests for the Haar-random unitary coin.
    """

    def test_invalid_dimension_raises(self) -> None:
        """
        d must be positive; d ≤ 0 should raise ValueError.
        """
        for d in (-5, 0):
            with self.subTest(d=d):
                with self.assertRaises(ValueError):
                    random_coin(d)

    def test_shape(self) -> None:
        """
        Matrix must be d×d for multiple dimensions.
        """
        for d in (1, 2, 3, 6):
            with self.subTest(d=d):
                U = random_coin(d)
                self.assertEqual(U.shape, (d, d))

    def test_unitarity(self) -> None:
        """
        Random coin must be unitary.
        """
        for d in (1, 2, 3, 6):
            with self.subTest(d=d):
                self.assertTrue(self.is_unitary(random_coin(d)))

    def test_dtype_is_complex128(self) -> None:
        """
        Return dtype should be complex128.
        """
        self.assert_is_complex128(random_coin(4))

    def test_reproducible_rng(self) -> None:
        """
        Same RNG seed → identical matrix; different seed → likely different.
        """
        d = 4
        rng_a1 = np.random.default_rng(123)
        rng_a2 = np.random.default_rng(123)
        rng_b = np.random.default_rng(124)

        Ua1 = random_coin(d, rng=rng_a1)
        Ua2 = random_coin(d, rng=rng_a2)
        Ub = random_coin(d, rng=rng_b)

        # Identical seeds must yield identical matrices
        np.testing.assert_allclose(Ua1, Ua2, atol=1e-12)

        # Different seed should almost certainly differ
        self.assertFalse(np.allclose(Ua1, Ub, atol=1e-12))

    def test_return_is_fresh_instance(self) -> None:
        """
        Each call without rng returns an independent array (no shared state).
        """
        U1 = random_coin(3)
        U2 = random_coin(3)
        # Modify U1 and ensure U2 is unaffected
        U1[0, 0] += 1.0
        self.assertFalse(np.allclose(U1, U2, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
