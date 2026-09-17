"""Spectral predictors listed in manuscript Table 1."""
from __future__ import annotations

import numpy as np


def _safe_ratio(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan),
        where=np.abs(denominator) > 1e-12,
    )


def spectral_indices(red, green, red_edge, nir) -> dict[str, np.ndarray]:
    """Return NDVI, LCI, GNDVI, OSAVI, and NDRE from reflectance arrays."""
    red = np.asarray(red, dtype=np.float64)
    green = np.asarray(green, dtype=np.float64)
    red_edge = np.asarray(red_edge, dtype=np.float64)
    nir = np.asarray(nir, dtype=np.float64)
    return {
        "NDVI": _safe_ratio(nir - red, nir + red),
        "LCI": _safe_ratio(nir - red_edge, nir + red),
        "GNDVI": _safe_ratio(nir - green, nir + green),
        "OSAVI": _safe_ratio(nir - red, nir + red + 0.16),
        "NDRE": _safe_ratio(nir - red_edge, nir + red_edge),
    }


def canopy_coverage(ndvi, threshold: float = 0.45) -> float:
    """Canopy-pixel fraction after the reference NDVI threshold is applied."""
    values = np.asarray(ndvi, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.count_nonzero(values[finite] >= threshold) / np.count_nonzero(finite))


def plot_spectral_features(red, green, red_edge, nir, ndvi_threshold: float = 0.45) -> dict[str, float]:
    """Aggregate the six multispectral inputs for one plant/plot observation."""
    indices = spectral_indices(red, green, red_edge, nir)
    vegetation = np.isfinite(indices["NDVI"]) & (indices["NDVI"] >= ndvi_threshold)
    output = {"canopy_coverage": canopy_coverage(indices["NDVI"], ndvi_threshold)}
    for name, values in indices.items():
        values = np.asarray(values, dtype=np.float64)
        output[name] = float(np.nanmean(values[vegetation])) if np.any(vegetation) else float("nan")
    return output
