"""Backend selection: the Rust core (``traceot._core``) or the numpy reference.

Both backends expose the same functions with the same signatures:

``quantile_grid, w2_sq, pava, pava_rows, collapse_knots, transport_apply,
local_linear_weights, global_frechet_weights, adopt_w2_fit, adopt_w2_predict,
adopt_w2_component, additive_euclid_fit, additive_euclid_predict,
spd_to_coords, coords_to_spd``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from . import reference as _ref

try:  # pragma: no cover - exercised implicitly
    from . import _core

    HAVE_RUST = True
except ImportError:  # pragma: no cover
    _core = None
    HAVE_RUST = False

_NAMES = (
    "quantile_grid",
    "pava",
    "pava_rows",
    "collapse_knots",
    "transport_apply",
    "local_linear_weights",
    "global_frechet_weights",
    "adopt_w2_fit",
    "adopt_w2_predict",
    "adopt_w2_component",
    "additive_euclid_fit",
    "additive_euclid_predict",
    "spd_to_coords",
    "coords_to_spd",
)


def _f(a, ndim):
    a = np.ascontiguousarray(a, dtype=float)
    if a.ndim != ndim:
        raise ValueError(f"expected a {ndim}-D array, got shape {a.shape}")
    return a


def _rust_namespace():
    c = _core

    def w2_sq(q1, q2):
        q1, q2 = np.broadcast_arrays(np.asarray(q1, float), np.asarray(q2, float))
        shape = q1.shape[:-1]
        a = np.ascontiguousarray(q1.reshape(-1, q1.shape[-1]))
        b = np.ascontiguousarray(q2.reshape(-1, q2.shape[-1]))
        return c.w2_sq_rows(a, b).reshape(shape)

    def pava(y, lower=None, upper=None):
        return c.pava(_f(y, 1), lower, upper)

    def pava_rows(Y, lower=None, upper=None):
        return c.pava_rows(_f(np.atleast_2d(Y), 2), lower, upper)

    def collapse_knots(a, b):
        return c.collapse_knots(_f(a, 1), _f(b, 1))

    def transport_apply(a, b, y):
        return c.transport_apply(_f(a, 1), _f(b, 1), _f(np.atleast_1d(y), 1))

    def local_linear_weights(x_train, x_eval, h, kernel="gaussian"):
        return c.local_linear_weights(
            _f(np.ravel(x_train), 1), _f(np.ravel(x_eval), 1), float(h), kernel
        )

    def global_frechet_weights(X_train, X_eval):
        return c.global_frechet_weights(_f(np.atleast_2d(X_train), 2), _f(np.atleast_2d(X_eval), 2))

    def adopt_w2_fit(X, Y, h, kernel="gaussian", tol=1e-10, max_iter=200, lower=None, upper=None):
        X, Y = _f(np.atleast_2d(X), 2), _f(np.atleast_2d(Y), 2)
        h = _ref.smoother.check_bandwidths(h, X.shape[1])
        return c.adopt_w2_fit(X, Y, h, kernel, float(tol), int(max_iter), lower, upper)

    def adopt_w2_predict(X_train, P, M, mu, h, X_new, kernel="gaussian", lower=None, upper=None):
        X_train = _f(np.atleast_2d(X_train), 2)
        h = _ref.smoother.check_bandwidths(h, X_train.shape[1])
        return c.adopt_w2_predict(
            X_train,
            _f(P, 3),
            _f(M, 2),
            _f(mu, 1),
            h,
            _f(np.atleast_2d(X_new), 2),
            kernel,
            lower,
            upper,
        )

    def adopt_w2_component(X_train, P_j, h_j, x_eval, kernel="gaussian", lower=None, upper=None):
        return c.adopt_w2_component(
            _f(np.ravel(X_train), 1),
            _f(P_j, 2),
            float(h_j),
            _f(np.ravel(x_eval), 1),
            kernel,
            lower,
            upper,
        )

    def additive_euclid_fit(X, Z, h, kernel="gaussian", tol=1e-10, max_iter=200):
        X, Z = _f(np.atleast_2d(X), 2), _f(np.atleast_2d(Z), 2)
        h = _ref.smoother.check_bandwidths(h, X.shape[1])
        return c.additive_euclid_fit(X, Z, h, kernel, float(tol), int(max_iter))

    def additive_euclid_predict(X_train, P, M, mu, h, X_new, kernel="gaussian"):
        X_train = _f(np.atleast_2d(X_train), 2)
        h = _ref.smoother.check_bandwidths(h, X_train.shape[1])
        return c.additive_euclid_predict(
            X_train, _f(P, 3), _f(M, 2), _f(mu, 1), h, _f(np.atleast_2d(X_new), 2), kernel
        )

    def spd_to_coords(S):
        S = np.asarray(S, float)
        flat = S.reshape((-1,) + S.shape[-2:])
        return c.spd_to_coords(_f(flat, 3)).reshape(S.shape[:-2] + (-1,))

    def coords_to_spd(z, m):
        z = np.asarray(z, float)
        flat = z.reshape(-1, z.shape[-1])
        return c.coords_to_spd(_f(flat, 2), int(m)).reshape(z.shape[:-1] + (m, m))

    ns = dict(locals())
    ns["quantile_grid"] = c.quantile_grid
    ns["name"] = "rust"
    return SimpleNamespace(**{k: ns[k] for k in (*_NAMES, "w2_sq", "name")})


def _reference_namespace():
    ns = {k: getattr(_ref, k) for k in _NAMES}
    ns["w2_sq"] = _ref.w2_sq
    ns["name"] = "reference"
    return SimpleNamespace(**ns)


REFERENCE = _reference_namespace()
RUST = _rust_namespace() if HAVE_RUST else None


def get_backend(name: str = "auto"):
    """Return the namespace for ``"rust"``, ``"reference"`` or ``"auto"`` (Rust if built)."""
    if name == "auto":
        return RUST if HAVE_RUST else REFERENCE
    if name == "reference":
        return REFERENCE
    if name == "rust":
        if not HAVE_RUST:
            raise ImportError(
                "traceot._core is not built; run `maturin develop` or use 'reference'"
            )
        return RUST
    raise ValueError(f"unknown backend {name!r}")
