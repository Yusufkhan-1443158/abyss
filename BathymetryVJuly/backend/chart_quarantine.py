"""Chart-sounding quarantine filter (IHO request #1, item 1.1).

i-Boating raster tiles encode TWO things at the publisher's deepest
contour value (here ``MAX_DEPTH_CAP_M = 25.0`` m):

1.  *Soundings* whose true depth happens to be ~25 m.
2.  *Limit-of-data fill* — every pixel deeper than the chart's
    publication cap is rendered at 25 m.  These are NOT soundings;
    IHO S-4 §B-413 calls them "limit-of-data" indications.

Likewise, **contour vertices** appear as long runs of *identical*
depth values strung along the contour line.  When OCR'd they become
clusters of 5-20+ identical depths within a few-tens-of-metres
neighbourhood.  These are also not independent soundings — they are
one contour, sampled many times.

This module provides ``filter_chart_quarantine`` which removes both
artefact classes from a soundings DataFrame and reports counters.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


DEFAULT_CAP_M = 25.0
DEFAULT_CAP_TOL_M = 0.05
DEFAULT_CONTOUR_RADIUS_M = 100.0
DEFAULT_CONTOUR_MIN_CLUSTER = 5
DEFAULT_CONTOUR_DEPTH_TOL_M = 0.05


def filter_chart_quarantine(
    df: pd.DataFrame,
    cap_m: float = DEFAULT_CAP_M,
    cap_tol_m: float = DEFAULT_CAP_TOL_M,
    contour_radius_m: float = DEFAULT_CONTOUR_RADIUS_M,
    contour_min_cluster: int = DEFAULT_CONTOUR_MIN_CLUSTER,
    contour_depth_tol_m: float = DEFAULT_CONTOUR_DEPTH_TOL_M,
) -> Tuple[pd.DataFrame, int, int]:
    """Drop chart-cap fill and contour-vertex duplicates from soundings.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have ``lat``, ``lon``, ``depth`` columns (any other
        columns are preserved on the surviving rows).
    cap_m : float
        Site's chart cap (i-Boating tile cap is 25 m).  Any point
        whose depth equals ``cap_m`` within ``±cap_tol_m`` is dropped
        (``n_chart_cap_dropped``).
    contour_radius_m : float
        Search radius (metres) for the contour-cluster test.
    contour_min_cluster : int
        Minimum cluster size (the point itself + neighbours of the
        same depth within ``contour_radius_m``) to be flagged as a
        contour-vertex duplicate (``n_chart_contour_dropped``).
    contour_depth_tol_m : float
        Two depths are considered "identical" for the cluster test if
        they agree within this tolerance (default 5 cm — i-Boating
        depths are reported to 0.1 m).

    Returns
    -------
    (filtered_df, n_chart_cap_dropped, n_chart_contour_dropped)
    """
    if df is None or len(df) == 0:
        return df, 0, 0

    df = df.reset_index(drop=True)
    depth = df["depth"].values.astype(np.float64)

    # ---- (a) chart-cap fill --------------------------------------------
    cap_mask = np.abs(depth - cap_m) <= cap_tol_m
    n_chart_cap_dropped = int(cap_mask.sum())

    # ---- (b) contour-vertex clusters -----------------------------------
    # On the points that survived (a), look for clusters of ≥
    # ``contour_min_cluster`` points sharing the same depth (within
    # ``contour_depth_tol_m``) inside a ``contour_radius_m`` ball.
    keep_mask = ~cap_mask
    contour_mask = np.zeros(len(df), dtype=bool)

    if keep_mask.sum() >= contour_min_cluster:
        try:
            from sklearn.neighbors import BallTree

            sub_idx = np.where(keep_mask)[0]
            lats = df["lat"].values[sub_idx].astype(np.float64)
            lons = df["lon"].values[sub_idx].astype(np.float64)
            sub_depth = depth[sub_idx]

            xy = np.deg2rad(np.column_stack([lats, lons]))
            tree = BallTree(xy, metric="haversine")
            R_earth = 6371000.0
            radius_rad = contour_radius_m / R_earth
            neighbours = tree.query_radius(xy, r=radius_rad)

            for i, nb in enumerate(neighbours):
                # cluster = neighbours (incl. self) that share this point's depth
                same = nb[np.abs(sub_depth[nb] - sub_depth[i]) <= contour_depth_tol_m]
                if len(same) >= contour_min_cluster:
                    contour_mask[sub_idx[same]] = True
        except ImportError:
            # No sklearn available — skip contour test, return only cap drops
            pass

    n_chart_contour_dropped = int(contour_mask.sum())

    drop_all = cap_mask | contour_mask
    out = df.loc[~drop_all].reset_index(drop=True)
    return out, n_chart_cap_dropped, n_chart_contour_dropped
