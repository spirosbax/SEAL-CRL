"""
Common math helpers shared across estimators and evaluations.
"""

import numpy as np
from sklearn.metrics import r2_score
from typing import Optional


def r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² score with safety checks.
    Returns NaN when variance is zero or computation fails.
    """
    if np.std(y_true) == 0:
        return np.nan
    try:
        return r2_score(y_true, y_pred)
    except Exception:
        return np.nan


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically-stable sigmoid.
    """
    z_clip = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


def add_row_jitter(A: np.ndarray, rng: np.random.Generator, noise: float = 1e-6) -> np.ndarray:
    """
    Add small jitter to zero-variance rows to avoid numerical issues.
    """
    A2 = A.astype(float, copy=True)
    row_std = A2.std(axis=1, ddof=0)
    idx = np.where(row_std == 0.0)[0]
    if idx.size:
        A2[idx] += rng.normal(0.0, noise, size=(idx.size, A2.shape[1]))
    return A2


# Backwards-compatibility alias
def add_jitter_if_constant(Y: np.ndarray, rng: np.random.Generator, noise: float = 1e-6) -> np.ndarray:
    return add_row_jitter(Y, rng, noise)


