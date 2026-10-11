"""Log-Cholesky coordinates for SPD matrices (Lin 2019), reference implementation.

``phi(S) = (strict lower part of L, log diag L)`` with ``S = L L^T``. The log-Cholesky
distance is the Euclidean distance between these coordinate vectors, so the space is
flat: Fréchet means are coordinate means and geodesic transports are translations.
Coordinate order: strict lower part in row-major order (``np.tril_indices(m, -1)``),
followed by the m log-diagonal entries.
"""

from __future__ import annotations

import numpy as np


def spd_to_coords(S: np.ndarray) -> np.ndarray:
    """(…, m, m) SPD matrices -> (…, m(m+1)/2) log-Cholesky coordinates."""
    S = np.asarray(S, dtype=float)
    m = S.shape[-1]
    L = np.linalg.cholesky(S)
    r, c = np.tril_indices(m, -1)
    diag = np.log(np.diagonal(L, axis1=-2, axis2=-1))
    return np.concatenate([L[..., r, c], diag], axis=-1)


def coords_to_cholesky(z: np.ndarray, m: int) -> np.ndarray:
    """Coordinates -> lower-triangular Cholesky factor(s) L."""
    z = np.asarray(z, dtype=float)
    r, c = np.tril_indices(m, -1)
    k = r.shape[0]
    L = np.zeros(z.shape[:-1] + (m, m))
    L[..., r, c] = z[..., :k]
    idx = np.arange(m)
    L[..., idx, idx] = np.exp(z[..., k:])
    return L


def coords_to_spd(z: np.ndarray, m: int) -> np.ndarray:
    """Coordinates -> SPD matrices ``L L^T``."""
    L = coords_to_cholesky(z, m)
    return L @ np.swapaxes(L, -1, -2)


def logchol_dist_sq(S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """Squared log-Cholesky distance."""
    d = spd_to_coords(S1) - spd_to_coords(S2)
    return np.sum(d * d, axis=-1)
