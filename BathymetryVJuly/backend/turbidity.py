"""Sentinel-2 turbidity / sediment / bottom-type helpers.

References
----------
- Nechad B., Ruddick K., Park Y. (2010).
  *Calibration and validation of a generic multisensor algorithm for
  mapping of total suspended matter in turbid waters.*
  Remote Sensing of Environment 114 (4) 854-866.

  T(λ) = (A_T(λ) · ρ_w(λ)) / (1 - ρ_w(λ) / C_T(λ))

  with at 665 nm (S2 B04):  A_T = 366.14 g·m⁻³, C_T = 0.19563.

- Caballero & Stumpf (2020). NDTI = (R_red - R_green) / (R_red + R_green).

- Bottom-type clustering: simple k-means on (Rw_blue, Rw_green, Rw_red,
  Rw_coastal) — same idea as Geyman & Maloof 2019 CBR but exposed as a
  reusable feature stack.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Nechad 2010 calibration at 665 nm (Sentinel-2 B04)
NECHAD_AT_665 = 366.14   # g/m³
NECHAD_CT_665 = 0.19563

EPS = 1e-6


def nechad_turbidity(rho_red: np.ndarray) -> np.ndarray:
    """Nechad 2010 SPM/turbidity from red-band water-leaving reflectance.

    Input `rho_red` is dimensionless reflectance (≈ 0–0.20 for natural
    waters). Output is total suspended matter in g·m⁻³.

    The formula saturates as ρ → C_T (≈0.196): we clamp the denominator
    at +0.02 below C_T so the output stays finite and monotonic.
    """
    r = np.clip(rho_red, EPS, NECHAD_CT_665 - 0.02)
    spm = (NECHAD_AT_665 * r) / (1.0 - r / NECHAD_CT_665)
    return np.clip(spm, 0.0, 200.0).astype(np.float32)


def ndti(rho_red: np.ndarray, rho_green: np.ndarray) -> np.ndarray:
    """Caballero & Stumpf 2020 normalized difference turbidity index."""
    return ((rho_red - rho_green) / (rho_red + rho_green + EPS)).astype(np.float32)


def coastal_clarity_index(rho_coastal: np.ndarray, rho_blue: np.ndarray
                           ) -> np.ndarray:
    """Coastal/blue ratio — drops as turbidity increases (chlorophyll +
    sediment absorb the coastal aerosol band).
    """
    return (rho_coastal / (rho_blue + EPS)).astype(np.float32)


def bottom_type_clusters(
    rho_blue: np.ndarray, rho_green: np.ndarray,
    rho_red: np.ndarray, rho_coastal: np.ndarray,
    water_mask: np.ndarray, *, k: int = 5, seed: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    """K-means on (Rw_blue, Rw_green, Rw_red, Rw_coastal) over water pixels.

    Returns
    -------
    cluster_id : (H,W) int8, -1 outside `water_mask`
    one_hot    : (H,W,k) float32, all-zeros outside the mask
    """
    H, W = rho_blue.shape
    feat = np.stack([rho_blue, rho_green, rho_red, rho_coastal], axis=-1)
    valid = water_mask & np.isfinite(feat).all(axis=-1)
    pts = feat[valid].astype(np.float32)
    if len(pts) < k * 4:
        out_id = np.full((H, W), -1, dtype=np.int8)
        oh = np.zeros((H, W, k), dtype=np.float32)
        return out_id, oh

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(pts)
    except Exception:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), size=k, replace=False)
        C = pts[idx].copy()
        for _ in range(15):
            d2 = np.sum((pts[:, None, :] - C[None, :, :]) ** 2, axis=2)
            labels = np.argmin(d2, axis=1)
            for ki in range(k):
                m = labels == ki
                if m.any():
                    C[ki] = pts[m].mean(0)

    out_id = np.full((H, W), -1, dtype=np.int8)
    out_id[valid] = labels.astype(np.int8)
    oh = np.zeros((H, W, k), dtype=np.float32)
    for ki in range(k):
        oh[..., ki] = (out_id == ki).astype(np.float32)
    return out_id, oh


def turbidity_features(s2: dict, *, k_clusters: int = 5
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """All-in-one helper: returns (turbidity_g_m3, ndti, coastal_clarity,
    bottom_one_hot) given the standard `fetch_s2`-shape dict.

    `bottom_one_hot` has shape (H, W, k_clusters).
    """
    rho_b = s2["blue"].astype(np.float32) / 10000.0
    rho_g = s2["green"].astype(np.float32) / 10000.0
    rho_r = s2["red"].astype(np.float32) / 10000.0
    rho_c = s2.get("coastal", s2["blue"]).astype(np.float32) / 10000.0
    wm = s2.get("water_mask", np.ones_like(rho_b, dtype=bool))
    spm = nechad_turbidity(rho_r)
    ndti_arr = ndti(rho_r, rho_g)
    cci = coastal_clarity_index(rho_c, rho_b)
    _, oh = bottom_type_clusters(rho_b, rho_g, rho_r, rho_c, wm,
                                   k=k_clusters)
    return spm, ndti_arr, cci, oh


# ════════════════════════════════════════════════════════════════════════
# Probability of accuracy (σ_total + Gaussian erf)
# ════════════════════════════════════════════════════════════════════════
def sigma_total(sigma_mc: np.ndarray, turbidity: np.ndarray, depth: np.ndarray,
                k_T: float = 0.012, k_d: float = 0.04, kd_limit: float = 12.0
                ) -> np.ndarray:
    """Combine MC-Dropout σ with turbidity- and depth-driven inflation.

    Parameters
    ----------
    sigma_mc    : MC-Dropout σ (epistemic uncertainty), in metres.
    turbidity   : Nechad SPM in g·m⁻³.
    depth       : modeled depth in metres (for Kd-saturation term).
    k_T         : turbidity coefficient (0.012 m per g/m³ ≈ +1.2 cm σ
                  per g/m³ of suspended sediment, calibrated against
                  Khalifa hold-out residuals).
    k_d         : depth-saturation coefficient (m per m beyond kd_limit).
    kd_limit    : depth (m) beyond which spectral SDB saturates.
    """
    s2 = sigma_mc.astype(np.float32) ** 2
    s2 = s2 + (k_T * turbidity) ** 2
    over = np.maximum(0.0, depth - kd_limit)
    s2 = s2 + (k_d * over) ** 2
    return np.sqrt(np.clip(s2, 1e-4, None)).astype(np.float32)


IHO_ORDERS = {
    # IHO S-44 6th ed. (2020) standard orders
    "Special":  (0.25, 0.0075),
    "Order_1a": (0.50, 0.013),
    "Order_1b": (0.50, 0.013),
    "Order_2":  (1.00, 0.023),     # alias for Order 2A
    # We split Order 2 into a stricter (2A) and a relaxed (2B) variant
    # for finer reporting. Order 2A == standard IHO Order 2. Order 2B is
    # NOT in S-44 — it's a commonly-used relaxed envelope (a=1.5, b=0.04)
    # for areas where Order 2 is not achievable but a coarser bathymetric
    # product is still useful (≈ "below Order 2").
    "Order_2a": (1.00, 0.023),
    "Order_2b": (1.50, 0.040),
}


def prob_within_iho(depth: np.ndarray, sigma_tot: np.ndarray,
                     order: str = "Order_1a") -> np.ndarray:
    """P(|err| ≤ TVU(d, order)) under a Gaussian residual assumption.

    Uses scipy.special.erf when available, else numpy approximation.
    Output is a (H, W) float32 in [0, 1].
    """
    a, b = IHO_ORDERS[order]
    tvu = np.sqrt(a * a + (b * depth) ** 2)
    z = tvu / (np.sqrt(2.0) * np.clip(sigma_tot, 1e-3, None))
    try:
        from scipy.special import erf
        p = erf(z)
    except Exception:
        # Abramowitz approx
        sign = np.sign(z); az = np.abs(z)
        t = 1.0 / (1.0 + 0.3275911 * az)
        coeff = [1.061405429, -1.453152027, 1.421413741,
                  -0.284496736, 0.254829592]
        y = 1.0 - (((((coeff[0] * t + coeff[1]) * t) + coeff[2]) * t
                     + coeff[3]) * t + coeff[4]) * t * np.exp(-az * az)
        p = sign * y
    return np.clip(p, 0.0, 1.0).astype(np.float32)
