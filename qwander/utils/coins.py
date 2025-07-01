"""
Quantum-walk coin operators for `qwander`.

Each function returns a unitary NumPy array of shape (d, d) with dtype complex128.
These coins dictate how a discrete-time quantum walk redistributes amplitude among
outgoing edges of a graph vertex.

Available coins
---------------
- hadamard_coin(): 2×2 Hadamard coin (d must be 2)
- grover_coin(d): d×d Grover diffusion coin (d ≥ 1)
- fourier_coin(d): d×d discrete Fourier coin (d ≥ 1)
- identity_coin(d): d×d identity coin (d ≥ 1)
- random_coin(d[, rng]): d×d Haar-random unitary coin (d ≥ 1)
"""

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "hadamard_coin",
    "grover_coin",
    "fourier_coin",
    "identity_coin",
    "random_coin",
]


def hadamard_coin() -> NDArray[np.complex128]:
    """
    Return the 2×2 Hadamard coin.

    Returns
    -------
    NDArray[np.complex128]
        (1/√2) * [[1,  1], [1, -1]]
    """
    return (
        np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
        / np.sqrt(2.0, dtype=np.complex128)
    )


def grover_coin(d: int) -> NDArray[np.complex128]:
    """
    Return a Grover diffusion coin of size d×d.

    Entry (j, k) equals 2/d for j ≠ k, and (2/d)−1 on the diagonal.

    Parameters
    ----------
    d: int
        Positive coin dimension.

    Returns
    -------
    NDArray[np.complex128]
        Unitary Grover matrix.

    Raises
    ------
    TypeError
        If d is not an integer.
    ValueError
        If d < 1.
    """
    if not isinstance(d, int):
        raise TypeError(f"Dimension 'd' must be int, got {type(d).__name__}.")
    if d < 1:
        raise ValueError(f"Grover coin requires d ≥ 1 (got {d}).")

    m = 2.0 / d
    return (
         m * np.ones((d, d), dtype=np.complex128) - np.eye(d, dtype=np.complex128)
    )


def fourier_coin(d: int) -> NDArray[np.complex128]:
    """
    Return a discrete Fourier coin of size d × d.

    Entry (j, k) equals ω**(j·k) / √d where ω = exp(2πi / d).

    Parameters
    ----------
    d: int
        Positive coin dimension.

    Returns
    -------
    NDArray[np.complex128]
        Unitary Fourier matrix.

    Raises
    ------
    TypeError
        If d is not an integer.
    ValueError
        If d < 1.
    """
    if not isinstance(d, int):
        raise TypeError(f"Dimension 'd' must be int, got {type(d).__name__}.")
    if d < 1:
        raise ValueError(f"Fourier coin requires d ≥ 1 (got {d}).")

    omega = np.exp(2j * np.pi / d, dtype=np.complex128)
    jk = np.outer(np.arange(d), np.arange(d))

    return omega**jk / np.sqrt(d, dtype=np.complex128)


def identity_coin(d: int) -> NDArray[np.complex128]:
    """
    Return the d × d identity coin.

    Parameters
    ----------
    d: int
        Positive coin dimension.

    Returns
    -------
    NDArray[np.complex128]
        Identity matrix.

    Raises
    ------
    TypeError
        If d is not an integer.
    ValueError
        If d < 1.
    """
    if not isinstance(d, int):
        raise TypeError(f"Dimension 'd' must be int, got {type(d).__name__}.")
    if d < 1:
        raise ValueError(f"Identity coin requires d ≥ 1 (got {d}).")

    return np.eye(d, dtype=np.complex128)


def random_coin(d: int,
                rng: np.random.Generator | None = None) -> NDArray[np.complex128]:
    """
    Return a Haar-random unitary coin of size d × d.

    A complex Ginibre matrix is QR-decomposed and its Q factor is adjusted
    so the diagonal of R has unit modulus, guaranteeing Haar measure.

    Parameters
    ----------
    d: int
        Positive coin dimension.
    rng: numpy.random.Generator, optional
        RNG to use.  If None np.random.default_rng() is used.

    Returns
    -------
    NDArray[np.complex128]
        Haar-distributed unitary matrix.

    Raises
    ------
    TypeError
        If d is not an integer.
    ValueError
        If d < 1.
    """
    if not isinstance(d, int):
        raise TypeError(f"Dimension 'd' must be int, got {type(d).__name__}.")
    if d < 1:
        raise ValueError(f"Random coin requires d ≥ 1 (got {d}).")

    rng = rng or np.random.default_rng()
    z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    q, r = np.linalg.qr(z)

    # Column-wise phase correction so Q is truly unitary
    phases = np.exp(-1j * np.angle(np.diag(r)))
    q = q * phases[None, :]

    return q.astype(np.complex128, copy=False)
