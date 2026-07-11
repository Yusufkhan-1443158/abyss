# INTEGRATION: Wraps / supersedes the per_bin reporting inside evaluate() in
# backend/sdb_cnn_baseline.py.  After calling evaluate(), pass its test-set
# pred/truth arrays to stratified_report() for a richer, publication-ready table
# covering the four IHO S-44 depth bands (0-2, 2-5, 5-10, 10+ m).
#
# Primary integration points:
#   from backend.sdb_optim.stratified_metrics import stratified_report, error_distribution_plots
#   report, df, md_table = stratified_report(pred, truth)
#   error_distribution_plots(pred, truth, out_path=Path("results/error_plots"))
#
# The depth-band boundaries (0,2,5,10,inf) map to the IHO Order Special / 1a / 1b /
# lower-order CATZOC zones used in the evaluation contract.
"""
Depth-stratified error metrics for Satellite-Derived Bathymetry validation.

Depth strata are aligned with IHO S-44 (6th ed.) accuracy requirements:
  0–2 m   : IHO Special Order / Order 1a (CATZOC A1/A2), critical for harbour approaches
  2–5 m   : transitional / IHO Order 1a
  5–10 m  : IHO Order 1b / CATZOC B
  10 m+   : IHO Order 2 / CATZOC C/D, open coastal waters

Reporting four stratified metrics:
  RMSE  (m) — square root of mean squared error, depth-noise dominated
  MAE   (m) — mean absolute error, less sensitive to outliers
  R²        — coefficient of determination, variance explained
  MBE   (m) — mean bias error (positive = overprediction), systematic offset

Honesty contract:
  - n (sample count) is always reported alongside each stratum metric.
  - Bands with n < 30 are flagged with a WARNING: metrics are unreliable at small n.
  - R² can be negative (worse than predicting the mean); that is reported as-is.
  - No metric is hidden, substituted, or smoothed.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "StratumResult",
    "stratified_report",
    "error_distribution_plots",
    "DEPTH_BINS_DEFAULT",
]

# Default depth strata: (lo_m, hi_m) — hi=inf means no upper bound
DEPTH_BINS_DEFAULT: Tuple[Tuple[float, float], ...] = (
    (0.0, 2.0),
    (2.0, 5.0),
    (5.0, 10.0),
    (10.0, float("inf")),
)

# Sample-count threshold below which a stratum is flagged unreliable
_N_WARN = 30


# ─────────────────────────────────────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────────────────────────────────────

class StratumResult:
    """Metrics for a single depth stratum.

    Attributes
    ----------
    lo, hi   : depth-bin boundaries (m)
    n        : sample count
    rmse     : root-mean-squared error (m)
    mae      : mean absolute error (m)
    r2       : coefficient of determination (may be negative)
    mbe      : mean bias error, pred − truth (m)
    low_n    : True if n < 30 (metrics unreliable)
    """

    __slots__ = ("lo", "hi", "n", "rmse", "mae", "r2", "mbe", "low_n")

    def __init__(
        self,
        lo: float,
        hi: float,
        n: int,
        rmse: float,
        mae: float,
        r2: float,
        mbe: float,
    ) -> None:
        self.lo   = lo
        self.hi   = hi
        self.n    = n
        self.rmse = rmse
        self.mae  = mae
        self.r2   = r2
        self.mbe  = mbe
        self.low_n = n < _N_WARN

    @property
    def label(self) -> str:
        hi_str = "∞" if self.hi == float("inf") else f"{self.hi:.0f}"
        return f"{self.lo:.0f}–{hi_str} m"

    def __repr__(self) -> str:  # pragma: no cover
        flag = " [LOW-N]" if self.low_n else ""
        return (
            f"StratumResult({self.label}, n={self.n}{flag}, "
            f"RMSE={self.rmse:.3f} m, MAE={self.mae:.3f} m, "
            f"R²={self.r2:.3f}, MBE={self.mbe:+.3f} m)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_stratum(
    pred_s: np.ndarray,
    truth_s: np.ndarray,
    lo: float,
    hi: float,
) -> StratumResult:
    """Compute metrics for a single stratum; handles n=0 gracefully."""
    n = len(truth_s)
    if n == 0:
        return StratumResult(lo, hi, 0, float("nan"), float("nan"),
                             float("nan"), float("nan"))

    err = pred_s - truth_s
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    mbe  = float(np.mean(err))

    ss_tot = float(np.sum((truth_s - truth_s.mean()) ** 2))
    ss_res = float(np.sum(err ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))

    return StratumResult(lo, hi, n, rmse, mae, r2, mbe)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def stratified_report(
    pred: Union[np.ndarray, Sequence[float]],
    truth: Union[np.ndarray, Sequence[float]],
    depth_bins: Tuple[Tuple[float, float], ...] = DEPTH_BINS_DEFAULT,
    print_table: bool = True,
    latex: bool = False,
) -> Tuple[List[StratumResult], pd.DataFrame, str]:
    """Compute depth-stratified RMSE / MAE / R² / MBE and render a tidy table.

    Parameters
    ----------
    pred        : (N,) predicted depths (m, positive-down)
    truth       : (N,) in-situ / chart truth depths (m, positive-down)
    depth_bins  : sequence of (lo_m, hi_m) tuples.  Default = IHO-aligned 4 bands.
    print_table : if True, print the Markdown table to stdout
    latex       : if True, also return a LaTeX tabular string

    Returns
    -------
    strata  : list of StratumResult (one per stratum + overall)
    df      : pandas DataFrame, one row per stratum (+ "Overall")
    md_str  : Markdown string of the table

    Notes
    -----
    - n is always shown.
    - Bands with n < 30 are flagged in the table with [*] and a warning is printed.
    - R² < 0 (model worse than mean) is shown as-is.
    """
    pred  = np.asarray(pred,  dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    if pred.shape != truth.shape or pred.ndim != 1:
        raise ValueError(
            f"pred and truth must be 1-D arrays of the same length; "
            f"got pred={pred.shape}, truth={truth.shape}"
        )

    # Strip non-finite values (NaN depth = missing coverage)
    ok = np.isfinite(pred) & np.isfinite(truth)
    n_dropped = int((~ok).sum())
    if n_dropped > 0:
        warnings.warn(
            f"stratified_report: dropped {n_dropped} non-finite pairs.",
            RuntimeWarning,
            stacklevel=2,
        )
    pred  = pred[ok]
    truth = truth[ok]

    strata: List[StratumResult] = []
    for lo, hi in depth_bins:
        mask = (truth >= lo) & (truth < hi)
        sr = _compute_stratum(pred[mask], truth[mask], lo, hi)
        strata.append(sr)
        if sr.low_n and sr.n > 0:
            warnings.warn(
                f"stratified_report: stratum {sr.label} has n={sr.n} < {_N_WARN}; "
                "metrics unreliable — do NOT report without flagging.",
                UserWarning,
                stacklevel=2,
            )

    # Overall row
    overall = _compute_stratum(pred, truth, -float("inf"), float("inf"))
    # Give it a nice label via patching (lo/hi not used for label lookup after creation)
    overall_row = StratumResult(0.0, float("inf"), overall.n,
                                overall.rmse, overall.mae, overall.r2, overall.mbe)

    # Build DataFrame
    rows: List[Dict[str, Any]] = []
    for sr in strata:
        flag = " [*]" if sr.low_n and sr.n > 0 else ""
        rows.append({
            "Stratum": sr.label + flag,
            "n": sr.n,
            "RMSE (m)": f"{sr.rmse:.3f}" if sr.n > 0 else "—",
            "MAE (m)":  f"{sr.mae:.3f}"  if sr.n > 0 else "—",
            "R²":       f"{sr.r2:.3f}"   if sr.n > 0 else "—",
            "MBE (m)":  f"{sr.mbe:+.3f}" if sr.n > 0 else "—",
        })
    # Overall
    flag_all = " [*]" if overall_row.low_n and overall_row.n > 0 else ""
    rows.append({
        "Stratum": "Overall" + flag_all,
        "n": overall_row.n,
        "RMSE (m)": f"{overall_row.rmse:.3f}" if overall_row.n > 0 else "—",
        "MAE (m)":  f"{overall_row.mae:.3f}"  if overall_row.n > 0 else "—",
        "R²":       f"{overall_row.r2:.3f}"   if overall_row.n > 0 else "—",
        "MBE (m)":  f"{overall_row.mbe:+.3f}" if overall_row.n > 0 else "—",
    })

    df = pd.DataFrame(rows)

    # Markdown table
    md_str = df.to_markdown(index=False)
    if any(sr.low_n and sr.n > 0 for sr in strata):
        md_str += "\n\n[*] n < 30: metrics unreliable, flagged per honesty contract."

    if print_table:
        print("\n" + md_str + "\n")

    all_strata = strata + [overall_row]
    return all_strata, df, md_str


# ─────────────────────────────────────────────────────────────────────────────
# Publication-quality error plots
# ─────────────────────────────────────────────────────────────────────────────

def error_distribution_plots(
    pred: Union[np.ndarray, Sequence[float]],
    truth: Union[np.ndarray, Sequence[float]],
    out_path: Union[str, Path],
    depth_bins: Tuple[Tuple[float, float], ...] = DEPTH_BINS_DEFAULT,
    dpi: int = 300,
    file_stem: str = "sdb_error",
) -> List[Path]:
    """Generate and save publication-ready error-distribution plots.

    Produces three figures:
      1. Residual histogram (overall + per depth band, stacked)
      2. Residual-vs-depth scatter with per-band mean bias (MBE) overlaid
      3. Predicted-vs-true 1:1 scatter with identity line + R² annotation

    Parameters
    ----------
    pred       : (N,) predicted depths (m)
    truth      : (N,) true depths (m)
    out_path   : directory where PNGs are saved (created if absent)
    depth_bins : depth strata for colouring (default IHO-aligned)
    dpi        : output resolution (default 300 for publication)
    file_stem  : filename prefix for the three output files

    Returns
    -------
    paths : list of three Path objects to the saved PNGs
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive, safe for server/CI
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    pred  = np.asarray(pred,  dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    ok = np.isfinite(pred) & np.isfinite(truth)
    pred  = pred[ok]
    truth = truth[ok]
    residuals = pred - truth

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Publication style: serif font, minimal spines
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": dpi,
    })

    # Colour cycle per band (4 bands)
    band_colours = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    band_labels  = []
    for lo, hi in depth_bins:
        hi_str = "∞" if hi == float("inf") else f"{hi:.0f}"
        band_labels.append(f"{lo:.0f}–{hi_str} m")

    saved: List[Path] = []

    # ── Figure 1: Residual histogram ─────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.hist(residuals, bins=60, color="#455A64", alpha=0.75, edgecolor="white",
             linewidth=0.3, label=f"All (n={len(residuals)})")
    for (lo, hi), col, lbl in zip(depth_bins, band_colours, band_labels):
        mask = (truth >= lo) & (truth < hi)
        if mask.sum() >= 3:
            ax1.hist(residuals[mask], bins=40, color=col, alpha=0.55,
                     edgecolor="none", label=f"{lbl} (n={mask.sum()})")
    ax1.axvline(0.0, color="black", linewidth=1.0, linestyle="--", label="Zero bias")
    ax1.axvline(float(np.mean(residuals)), color="red", linewidth=1.2,
                linestyle="-", alpha=0.7,
                label=f"MBE = {np.mean(residuals):+.3f} m")
    ax1.set_xlabel("Residual  (pred − truth)  [m]")
    ax1.set_ylabel("Count")
    ax1.set_title("Depth residual distribution by stratum")
    ax1.legend(fontsize=8, frameon=False)
    fig1.tight_layout()
    p1 = out_path / f"{file_stem}_residual_hist.png"
    fig1.savefig(p1, dpi=dpi, bbox_inches="tight")
    plt.close(fig1)
    saved.append(p1)

    # ── Figure 2: Residual vs depth scatter ──────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.scatter(truth, residuals, s=5, alpha=0.30, color="#607D8B",
                edgecolors="none", rasterized=True, label="_nolegend_")
    ax2.axhline(0.0, color="black", linewidth=1.0, linestyle="--")

    # Per-band MBE marker with error bar (± 1 std)
    for (lo, hi), col, lbl in zip(depth_bins, band_colours, band_labels):
        mask = (truth >= lo) & (truth < hi)
        if mask.sum() >= 3:
            mid = (lo + min(hi, truth.max() + 1)) / 2.0
            mbe = float(np.mean(residuals[mask]))
            std = float(np.std(residuals[mask]))
            n_s  = int(mask.sum())
            ax2.errorbar(mid, mbe, yerr=std, fmt="o", color=col,
                         markersize=8, capsize=4, linewidth=1.5,
                         label=f"{lbl}  MBE={mbe:+.2f} m (n={n_s})")

    ax2.set_xlabel("True depth  [m]")
    ax2.set_ylabel("Residual  (pred − truth)  [m]")
    ax2.set_title("Residual vs depth (per-band bias ± 1σ)")
    ax2.legend(fontsize=8, frameon=False)
    fig2.tight_layout()
    p2 = out_path / f"{file_stem}_residual_vs_depth.png"
    fig2.savefig(p2, dpi=dpi, bbox_inches="tight")
    plt.close(fig2)
    saved.append(p2)

    # ── Figure 3: Predicted vs true 1:1 scatter ───────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    # Colour by depth band
    for (lo, hi), col, lbl in zip(depth_bins, band_colours, band_labels):
        mask = (truth >= lo) & (truth < hi)
        if mask.sum() >= 1:
            ax3.scatter(truth[mask], pred[mask], s=5, alpha=0.35, color=col,
                        edgecolors="none", rasterized=True, label=lbl)

    # 1:1 line over full range
    mn = float(min(truth.min(), pred.min()))
    mx = float(max(truth.max(), pred.max()))
    ax3.plot([mn, mx], [mn, mx], color="black", linewidth=1.2,
             linestyle="--", label="1:1 line")

    # R² annotation
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    ss_res = float(np.sum(residuals ** 2))
    r2_all = 1.0 - ss_res / max(ss_tot, 1e-12)
    rmse_all = float(np.sqrt(np.mean(residuals ** 2)))
    ax3.annotate(
        f"Overall R² = {r2_all:.3f}\nRMSE = {rmse_all:.3f} m\nn = {len(truth)}",
        xy=(0.05, 0.92), xycoords="axes fraction",
        fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.7),
    )
    ax3.set_xlabel("True depth  [m]")
    ax3.set_ylabel("Predicted depth  [m]")
    ax3.set_title("Predicted vs. true depth (1:1 scatter)")
    ax3.set_aspect("equal", adjustable="box")
    ax3.legend(fontsize=8, frameon=False)
    fig3.tight_layout()
    p3 = out_path / f"{file_stem}_pred_vs_true.png"
    fig3.savefig(p3, dpi=dpi, bbox_inches="tight")
    plt.close(fig3)
    saved.append(p3)

    return saved
