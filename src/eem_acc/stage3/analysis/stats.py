"""S3.5 test statistics (spec sections 4, 6, 7): exact one-sided paired sign-flip test, the non-inferiority shift,
untied counts, t intervals, the minimum detectable difference and a seeded stream bootstrap."""
from __future__ import annotations

import math

import numpy as np

TOL = 1e-12
MAX_N = 20


def _pattern_means(x: np.ndarray, chunk: int = 1 << 15):
    """Yield mean(e * x) for all 2^n sign patterns e, in chunks (bit k of the pattern index = sign of x_k), so
    n = 20 needs no 2^20 x 20 array (code review L13)."""
    n = len(x)
    bits = np.arange(n)
    for start in range(0, 1 << n, chunk):
        k = np.arange(start, min(start + chunk, 1 << n), dtype=np.int64)
        signs = 1.0 - 2.0 * ((k[:, None] >> bits) & 1)
        yield (signs * x).mean(axis=1)


def _check(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or len(x) > MAX_N:
        raise ValueError(f"need 1..{MAX_N} paired differences, got {len(x)}")
    return x


def sign_flip_one_sided(x) -> float:
    """Exact one-sided paired sign-flip test of H1: mean(x) > 0. p = share of the 2^n sign patterns e with
    mean(e * x) >= mean(x) (tolerance 1e-12). Zeros are ties: they change no pattern's mean, so the smallest
    attainable p is 2^-(untied n)."""
    x = _check(x)
    obs = x.mean()
    hits = sum(int(np.sum(m >= obs - TOL)) for m in _pattern_means(x))
    return hits / float(1 << len(x))


def sign_flip_two_sided(x) -> float:
    x = _check(x)
    obs = abs(x.mean())
    hits = sum(int(np.sum(np.abs(m) >= obs - TOL)) for m in _pattern_means(x))
    return hits / float(1 << len(x))


def untied(x) -> int:
    return int(np.sum(np.abs(np.asarray(x, dtype=float)) > TOL))


def directional(d, h1: str, shift: float = 0.0) -> np.ndarray:
    """The tested values for H1 'greater' (mean(d + shift) > 0) or 'less' (mean(d - shift) < 0, tested as
    mean(shift - d) > 0)."""
    d = np.asarray(d, dtype=float)
    if h1 == "greater":
        return d + shift
    if h1 == "less":
        return shift - d
    raise ValueError(h1)


def t_ci(d, level: float = 0.95) -> tuple[float, float]:
    from scipy import stats
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2:
        return (math.nan, math.nan)
    q = stats.t.ppf(1 - (1 - level) / 2, n - 1)
    h = q * d.std(ddof=1) / math.sqrt(n)
    return (float(d.mean() - h), float(d.mean() + h))


def mdd(d, alpha_one_sided: float, power: float = 0.8) -> float:
    """Minimum detectable difference at the observed SD: (t_{1-a, n-1} + t_{power, n-1}) * SD / sqrt(n)."""
    from scipy import stats
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2:
        return math.nan
    sd = d.std(ddof=1)
    return float((stats.t.ppf(1 - alpha_one_sided, n - 1) + stats.t.ppf(power, n - 1)) * sd / math.sqrt(n))


def bootstrap_ratio(num, den, n_boot: int = 10_000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Ratio of sums sum(num)/sum(den) over streams, with a stream-level percentile bootstrap interval."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    if len(num) != len(den) or len(num) == 0:
        raise ValueError("need paired, non-empty per-stream sums")
    est = float(num.sum() / den.sum()) if den.sum() > 0 else math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    ds = den[idx].sum(axis=1)
    rs = np.where(ds > 0, num[idx].sum(axis=1) / np.where(ds > 0, ds, 1), np.nan)
    rs = rs[~np.isnan(rs)]
    if len(rs) == 0:                                   # every resample has a zero denominator (code review L14)
        return est, math.nan, math.nan
    a = (1 - level) / 2
    return est, float(np.quantile(rs, a)), float(np.quantile(rs, 1 - a))
