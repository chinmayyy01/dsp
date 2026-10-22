"""Time-domain adaptive filters: LMS, NLMS and RLS.

All three solve the same problem: given a reference signal ``x`` and a desired
signal ``d``, find filter taps ``w`` that predict ``d`` from ``x``.  The
prediction ``y = w . x`` is subtracted, and the residual ``e = d - y`` is both
the output and the error that drives adaptation.

The inner loops are compiled with Numba when it is installed.  On a 0.1-CPU
free hosting instance the pure-Python loop takes minutes for a few seconds of
audio; the compiled kernel brings that back to a second.  Without Numba the
code still runs, just slowly.
"""

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator


DEFAULT_ORDER = 32
DEFAULT_BLOCK = 256


def _taps(reference: np.ndarray, order: int) -> np.ndarray:
    padded = np.concatenate([np.zeros(order - 1), np.asarray(reference, dtype=np.float64)])
    return np.ascontiguousarray(
        np.lib.stride_tricks.sliding_window_view(padded, order)[:, ::-1]
    )


def _check_freeze(freeze: np.ndarray | None, n: int) -> np.ndarray:
    if freeze is None:
        return np.zeros(n, dtype=np.bool_)
    freeze = np.asarray(freeze, dtype=np.bool_)
    if freeze.shape[0] != n:
        raise ValueError("freeze mask must have one entry per sample")
    return freeze


def _mse_curve(error: np.ndarray, block: int) -> np.ndarray:
    n_blocks = len(error) // block
    if n_blocks == 0:
        return np.array([float(np.mean(error ** 2))]) if len(error) else np.zeros(1)
    trimmed = error[:n_blocks * block].reshape(n_blocks, block)
    return np.mean(trimmed ** 2, axis=1)


@njit(cache=True)
def _lms_kernel(x_mat, desired, mu, freeze):
    n = desired.shape[0]
    order = x_mat.shape[1]
    w = np.zeros(order)
    y = np.zeros(n)
    e = np.zeros(n)

    for i in range(n):
        acc = 0.0
        for j in range(order):
            acc += w[j] * x_mat[i, j]
        y[i] = acc
        e[i] = desired[i] - acc
        if not freeze[i]:
            for j in range(order):
                w[j] += mu * e[i] * x_mat[i, j]

    return w, y, e


@njit(cache=True)
def _nlms_kernel(x_mat, desired, mu, freeze, eps):
    n = desired.shape[0]
    order = x_mat.shape[1]
    w = np.zeros(order)
    y = np.zeros(n)
    e = np.zeros(n)

    for i in range(n):
        acc = 0.0
        energy = 0.0
        for j in range(order):
            value = x_mat[i, j]
            acc += w[j] * value
            energy += value * value
        y[i] = acc
        e[i] = desired[i] - acc
        if not freeze[i]:
            step = mu / (energy + eps)
            for j in range(order):
                w[j] += step * e[i] * x_mat[i, j]

    return w, y, e


@njit(cache=True)
def _rls_kernel(x_mat, desired, lam, delta, freeze):
    n = desired.shape[0]
    order = x_mat.shape[1]
    w = np.zeros(order)
    p_inv = np.eye(order) / delta
    y = np.zeros(n)
    e = np.zeros(n)

    for i in range(n):
        x = x_mat[i]
        px = p_inv @ x
        denominator = lam + x @ px
        gain = px / denominator

        acc = 0.0
        for j in range(order):
            acc += w[j] * x[j]
        y[i] = acc
        e[i] = desired[i] - acc

        if not freeze[i]:
            for j in range(order):
                w[j] += gain[j] * e[i]
            for row in range(order):
                for col in range(order):
                    p_inv[row, col] = (p_inv[row, col] - gain[row] * px[col]) / lam
            for row in range(order):
                for col in range(order):
                    p_inv[row, col] = 0.5 * (p_inv[row, col] + p_inv[col, row])

    return w, y, e


def lms(reference: np.ndarray, desired: np.ndarray, mu: float = 0.01,
        order: int = DEFAULT_ORDER, block: int = DEFAULT_BLOCK,
        freeze: np.ndarray | None = None) -> dict:
    """w(n+1) = w(n) + mu * e(n) * x(n).

    ``freeze`` is an optional per-sample mask; where it is True the weights
    are held constant (used by a double-talk detector).
    """
    x_mat = _taps(reference, order)
    d = np.asarray(desired, dtype=np.float64)
    mask = _check_freeze(freeze, len(d))
    w, y, e = _lms_kernel(x_mat, d, float(mu), mask)
    return {"output": e, "prediction": y, "error": e, "weights": w,
            "mse": _mse_curve(e, block)}


def nlms(reference: np.ndarray, desired: np.ndarray, mu: float = 0.5,
         order: int = DEFAULT_ORDER, block: int = DEFAULT_BLOCK,
         eps: float = 1e-8, freeze: np.ndarray | None = None) -> dict:
    """w(n+1) = w(n) + mu * e(n) * x(n) / (||x(n)||^2 + eps)."""
    x_mat = _taps(reference, order)
    d = np.asarray(desired, dtype=np.float64)
    mask = _check_freeze(freeze, len(d))
    w, y, e = _nlms_kernel(x_mat, d, float(mu), mask, float(eps))
    return {"output": e, "prediction": y, "error": e, "weights": w,
            "mse": _mse_curve(e, block)}


def rls(reference: np.ndarray, desired: np.ndarray, lam: float = 0.99,
        order: int = DEFAULT_ORDER, block: int = DEFAULT_BLOCK,
        delta: float = 1.0, freeze: np.ndarray | None = None) -> dict:
    """Recursive least squares with forgetting factor ``lam``.

    k(n) = P x / (lam + x^T P x)
    w(n+1) = w(n) + k(n) e(n)
    P(n+1) = (P(n) - k(n) x^T P(n)) / lam
    """
    x_mat = _taps(reference, order)
    d = np.asarray(desired, dtype=np.float64)
    mask = _check_freeze(freeze, len(d))
    w, y, e = _rls_kernel(x_mat, d, float(lam), float(delta), mask)
    return {"output": e, "prediction": y, "error": e, "weights": w,
            "mse": _mse_curve(e, block)}
