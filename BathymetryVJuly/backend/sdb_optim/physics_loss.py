# INTEGRATION: Drop-in physics-informed wrapper around any depth-regression loss.
# Replace HeteroNLLLoss (backend/sdb_cnn_baseline.py) with PhysicsInformedLoss
# to add radiative-transfer regularisation during training.
#
# Primary integration point:
#   from backend.sdb_optim.physics_loss import PhysicsInformedLoss
#   from backend.sdb_cnn_baseline import HeteroNLLLoss
#   criterion = PhysicsInformedLoss(
#       base_loss=HeteroNLLLoss(),
#       lambda_stumpf=0.10,
#       lambda_lyzenga=0.05,
#       lambda_monotonic=0.10,
#   )
#   loss = criterion(mu, logsigma, target, stumpf_ratio=batch_ratio)
#
# All three physics terms default to 0.0 so the module is a no-op unless
# the caller sets non-zero lambdas — backward-compatible with existing loops.
"""
Physics-informed loss module for Satellite-Derived Bathymetry (SDB).

Physics background
------------------
The Beer-Lambert law governs underwater light propagation in shallow, optically-shallow
water.  For a Lambertian-reflecting seabed at depth z:

    Rw_i ≈ R_deep_i + (A_i - R_deep_i) * exp(-2 * Kd_i * z)

where Rw_i is the water-leaving reflectance in band i, Kd_i is the diffuse
attenuation coefficient, R_deep_i is the deep-water limit, and A_i is the seabed
albedo.  Two classical linearisations exploit this:

Stumpf (2003)
    z ∝ ln(Rw_blue) / ln(Rw_green)   (Stumpf, Holderied & Sinclair 2003, L&O)

This log-ratio cancels seabed albedo variation to first order when Kd_blue > Kd_green
(the typical case for clear tropical/UAE waters).  The relationship between z and the
Stumpf ratio is affine: z = m1 * ratio + m0.  If a trained network predicts depth
values that are *uncorrelated* with the Stumpf ratio, or that contradict its sign,
the network has learned something physically inconsistent.  The Stumpf penalty term
enforces correlation between predicted depths and Stumpf ratios by penalising the
squared deviation between the predicted-depth ranking and the ratio-derived ranking.

Lyzenga (1985)
    X_i = ln(Rw_i - R_deep_i)
    z ≈ a_0 + a_1*X_blue + a_2*X_green (+ a_3*X_red)

Lyzenga features are the reference input to the classical two-band log-linear SDB
model (Lyzenga 1985 RSE; Lyzenga et al. 2006 IJP).  The network should not assign
depths that are wildly inconsistent with this well-characterised linear relationship.
The Lyzenga regulariser measures the MSE between predicted depths and the Lyzenga
linear prediction (fit on the current batch as a reference), penalising inconsistency.

Monotonicity / water-column physics
    In clear, spectrally monotone water the blue channel attenuates faster than green
    (Kd_blue > Kd_green).  At fixed seabed albedo:
        ∂z/∂(–ln Rw_blue) > 0  and  ∂z/∂(–ln Rw_green) > 0
    Depth increases with both blue and green attenuation.  The monotonicity penalty
    flags predictions that are inconsistent with this expectation: when a sample has
    high blue attenuation relative to its neighbours but the model predicts a
    *shallower* depth, that violates water-column physics.

References
----------
Stumpf, R.P., Holderied, K. & Sinclair, M. (2003). Determination of water depth
  with high-resolution satellite imagery over variable bottom types.
  Limnology and Oceanography, 48(1part2), 547-556.  doi:10.4319/lo.2003.48.1_part_2.0547

Lyzenga, D.R. (1985). Shallow-water bathymetry using combined lidar and passive
  multispectral scanner data. International Journal of Remote Sensing, 6(1), 115-125.

Lyzenga, D.R., Malinas, N.P. & Tanis, F.J. (2006). Multispectral bathymetry using
  a simple physically based algorithm. IEEE Transactions on Geoscience and Remote Sensing,
  44(8), 2251-2259.  doi:10.1109/TGRS.2006.872909
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

L = logging.getLogger(__name__)


class PhysicsInformedLoss(nn.Module):
    """Supervised depth loss augmented with radiative-transfer physics regularisers.

    Composes an existing base loss (e.g. HeteroNLLLoss or SILogRankLoss from
    backend/sdb_cnn_baseline.py) with three optional physics terms:

    1. Stumpf consistency (``lambda_stumpf``): affine/correlation penalty tying
       predicted depth to the batch Stumpf log-ratio.  Implemented as the MSE
       between the predicted depths and their affine regression on the ratio,
       i.e. it penalises *lack of correlation* rather than the mean-offset
       (which is handled by the base loss).

    2. Lyzenga batch regulariser (``lambda_lyzenga``): per-batch OLS fit of the
       Lyzenga linear model z = a0 + a1*X_blue + a2*X_green on the predicted
       depths; the residual MSE is the penalty.  This nudges the network toward
       predictions that are consistent with the batch Lyzenga surface even when
       no external Lyzenga coefficients are supplied.

    3. Monotonicity penalty (``lambda_monotonic``): for each pair of pixels in the
       batch, if pixel A has strictly higher blue attenuation than pixel B
       (i.e. lower Rw_blue → deeper expected), the predicted depth of A should be
       ≥ that of B.  Violations are penalised with a hinge loss.  This encodes
       the fundamental water-column physics: depth increases with attenuation.

    All three lambdas default to 0.0 so the module is a strict no-op unless
    enabled, preserving backward-compatibility with existing training loops.

    Parameters
    ----------
    base_loss : nn.Module
        The supervised depth loss to augment.  Must accept ``(mu, lv, target)``
        for HeteroNLLLoss, or ``(mu, target)`` for simpler losses.  The
        ``forward`` method here calls it with all three positional arguments
        and falls back to two if that raises TypeError.
    lambda_stumpf : float
        Weight for the Stumpf affine-consistency term.  Typical range 0.05–0.20.
    lambda_lyzenga : float
        Weight for the Lyzenga batch-OLS regulariser.  Typical range 0.03–0.10.
    lambda_monotonic : float
        Weight for the depth-vs-attenuation monotonicity hinge.  Typical range
        0.05–0.20.  Uses at most ``max_monotonic_pairs`` random pairs per batch.
    max_monotonic_pairs : int
        Maximum number of random pixel pairs to check for monotonicity.
        Kept small (256) so the penalty adds <5 % training-time overhead.
    monotonic_margin_m : float
        Hinge margin in metres.  Pairs whose ground-truth depths differ by less
        than this threshold are not penalised (noisy / ambiguous ordering).

    Examples
    --------
    >>> from backend.sdb_cnn_baseline import HeteroNLLLoss
    >>> from backend.sdb_optim.physics_loss import PhysicsInformedLoss
    >>> crit = PhysicsInformedLoss(HeteroNLLLoss(), lambda_stumpf=0.10,
    ...                            lambda_monotonic=0.10)
    >>> mu = torch.rand(32); lv = torch.zeros(32); target = torch.rand(32) * 15
    >>> ratio = torch.rand(32) * 0.5 + 1.0   # Stumpf ratios
    >>> ln_blue = -torch.rand(32); ln_green = -torch.rand(32) * 0.7
    >>> loss = crit(mu, lv, target, stumpf_ratio=ratio,
    ...            ln_blue=ln_blue, ln_green=ln_green)
    """

    def __init__(
        self,
        base_loss: nn.Module,
        lambda_stumpf: float = 0.0,
        lambda_lyzenga: float = 0.0,
        lambda_monotonic: float = 0.0,
        max_monotonic_pairs: int = 256,
        monotonic_margin_m: float = 0.3,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.lambda_stumpf = float(lambda_stumpf)
        self.lambda_lyzenga = float(lambda_lyzenga)
        self.lambda_monotonic = float(lambda_monotonic)
        self.max_monotonic_pairs = int(max_monotonic_pairs)
        self.monotonic_margin_m = float(monotonic_margin_m)

    # ------------------------------------------------------------------
    # Stumpf consistency term
    # ------------------------------------------------------------------
    @staticmethod
    def _stumpf_affine_residual(
        pred_depth: torch.Tensor,
        stumpf_ratio: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between pred_depth and its OLS fit on stumpf_ratio.

        The penalty is zero when pred_depth is a perfectly affine function of the
        Stumpf log-ratio (i.e. the physics is satisfied up to the allowed linear
        transformation).  It grows as predictions become uncorrelated with the
        ratio, regardless of sign/scale.

        Parameters
        ----------
        pred_depth : (N,) predicted depth
        stumpf_ratio : (N,) ln(1000*Rw_blue)/ln(1000*Rw_green)

        Returns
        -------
        scalar MSE residual
        """
        n = pred_depth.shape[0]
        if n < 4:
            return pred_depth.new_zeros(())

        # OLS: fit y = a + b*x  (affine regression)
        x = stumpf_ratio.double()
        y = pred_depth.double()
        x_bar = x.mean()
        y_bar = y.mean()
        x_c = x - x_bar
        y_c = y - y_bar
        denom = (x_c * x_c).sum() + 1e-8
        b = (x_c * y_c).sum() / denom
        a = y_bar - b * x_bar
        y_hat = a + b * x
        residual_mse = ((y - y_hat) ** 2).mean()
        return residual_mse.float()

    # ------------------------------------------------------------------
    # Lyzenga batch regulariser
    # ------------------------------------------------------------------
    @staticmethod
    def _lyzenga_batch_residual(
        pred_depth: torch.Tensor,
        ln_blue: torch.Tensor,
        ln_green: torch.Tensor,
        ln_red: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Batch-OLS Lyzenga regulariser.

        Fits z = a0 + a1*X_blue + a2*X_green [+ a3*X_red] on predicted depths,
        returns the OLS residual MSE.  The penalty measures how well the batch
        predictions are explained by the Lyzenga linear surface — consistent
        predictions should have low residual.

        Parameters
        ----------
        pred_depth : (N,) predicted depth (detached from grad for the OLS fit,
                     but gradient flows through the residual computation)
        ln_blue : (N,) log-attenuation of blue band
        ln_green : (N,) log-attenuation of green band
        ln_red : optional (N,) log-attenuation of red band

        Returns
        -------
        scalar MSE residual w.r.t. the batch Lyzenga surface
        """
        n = pred_depth.shape[0]
        if n < 6:
            return pred_depth.new_zeros(())

        # Build design matrix [1, X_b, X_g (, X_r)]
        ones = torch.ones(n, 1, dtype=torch.float64, device=pred_depth.device)
        cols = [ones,
                ln_blue.double().unsqueeze(1),
                ln_green.double().unsqueeze(1)]
        if ln_red is not None:
            cols.append(ln_red.double().unsqueeze(1))
        X = torch.cat(cols, dim=1)   # (N, 2/3/4)
        y = pred_depth.double()      # (N,)

        # OLS via normal equations: coef = (X^T X)^{-1} X^T y
        XtX = X.t().mm(X) + 1e-6 * torch.eye(X.shape[1], dtype=torch.float64,
                                               device=X.device)
        Xty = X.t().mv(y)
        try:
            coef = torch.linalg.solve(XtX, Xty)
        except RuntimeError:
            coef = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze(1)

        y_hat = X.mv(coef).float()
        residual_mse = F.mse_loss(pred_depth, y_hat.detach())
        return residual_mse

    # ------------------------------------------------------------------
    # Monotonicity hinge
    # ------------------------------------------------------------------
    def _monotonicity_penalty(
        self,
        pred_depth: torch.Tensor,
        ln_blue: torch.Tensor,
    ) -> torch.Tensor:
        """Depth-vs-attenuation monotonicity hinge.

        Water-column physics (Beer-Lambert): higher blue attenuation (more
        negative ln_blue, or equivalently lower Rw_blue) indicates deeper water
        at constant seabed albedo.  For each random pair (i, j):
            if  ln_blue[i] < ln_blue[j]  (i.e. i has higher attenuation)
            then  pred_depth[i] >= pred_depth[j]  (i should be deeper)
        Violations are penalised with a hinge:
            penalty_ij = max(0, pred_depth[j] - pred_depth[i] + margin)

        This is a soft constraint and naturally handles: (a) bottom-albedo
        variability (which breaks strict monotonicity), (b) noise in reflectance.
        The margin (default 0.3 m) allows for small violations.

        Parameters
        ----------
        pred_depth : (N,) predicted depths in metres
        ln_blue : (N,) log of blue reflectance (or deglinted blue radiance)
                  More negative → higher attenuation → deeper expected

        Returns
        -------
        scalar hinge penalty
        """
        n = pred_depth.shape[0]
        if n < 4:
            return pred_depth.new_zeros(())

        n_pairs = min(self.max_monotonic_pairs, n * (n - 1) // 2)
        if n_pairs <= 0:
            return pred_depth.new_zeros(())

        # Sample random pairs
        idx_i = torch.randint(0, n, (n_pairs,), device=pred_depth.device)
        idx_j = torch.randint(0, n, (n_pairs,), device=pred_depth.device)
        # Ensure i != j
        same = idx_i == idx_j
        idx_j[same] = (idx_j[same] + 1) % n

        # Attenuation difference: negative ln_blue → higher attenuation
        att_i = -ln_blue[idx_i]  # higher value = higher attenuation
        att_j = -ln_blue[idx_j]

        # If att_i > att_j (i has more attenuation), i should be deeper
        i_deeper_expected = att_i > att_j
        if not i_deeper_expected.any():
            return pred_depth.new_zeros(())

        d_i = pred_depth[idx_i[i_deeper_expected]]
        d_j = pred_depth[idx_j[i_deeper_expected]]

        # Hinge: penalise when d_i < d_j (i should be deeper but isn't)
        margin = self.monotonic_margin_m
        penalty = F.relu(d_j - d_i + margin)
        return penalty.mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        mu: torch.Tensor,
        logsigma: torch.Tensor,
        target: torch.Tensor,
        stumpf_ratio: Optional[torch.Tensor] = None,
        ln_blue: Optional[torch.Tensor] = None,
        ln_green: Optional[torch.Tensor] = None,
        ln_red: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute combined physics-informed loss.

        Parameters
        ----------
        mu : (N,) or (B,1,H,W) predicted depth mean
        logsigma : (N,) or (B,1,H,W) predicted log(sigma)
        target : (N,) or (B,1,H,W) ground-truth depth
        stumpf_ratio : optional (N,) Stumpf log-ratio ln(1000*B2)/ln(1000*B3).
            If None, the Stumpf term is skipped regardless of lambda_stumpf.
        ln_blue : optional (N,) log(blue reflectance). Required for both the
            Lyzenga and monotonicity terms.
        ln_green : optional (N,) log(green reflectance).
        ln_red : optional (N,) log(red reflectance).
        weights : optional per-sample weights forwarded to the base loss.

        Returns
        -------
        Scalar total loss tensor (with gradient).
        """
        # ---- base supervised loss ----------------------------------------
        # Support both HeteroNLLLoss(mu, lv, target [, weights]) and
        # simpler losses(mu, target).
        try:
            if weights is not None:
                base = self.base_loss(mu, logsigma, target, weights)
            else:
                base = self.base_loss(mu, logsigma, target)
        except TypeError:
            base = self.base_loss(mu, target)

        total = base

        # Flatten to 1-D for the physics terms (handles both patch-CNN and U-Net)
        mu_flat = mu.reshape(-1)
        target_flat = target.reshape(-1)

        # Only operate on finite, positive predictions
        valid = torch.isfinite(mu_flat) & torch.isfinite(target_flat) & (target_flat > 0)
        if valid.sum() < 4:
            return total

        mu_v = mu_flat[valid]

        # ---- Stumpf consistency ------------------------------------------
        if self.lambda_stumpf > 0.0 and stumpf_ratio is not None:
            ratio_v = stumpf_ratio.reshape(-1)[valid]
            if torch.isfinite(ratio_v).all():
                st_loss = self._stumpf_affine_residual(mu_v, ratio_v)
                total = total + self.lambda_stumpf * st_loss

        # ---- Lyzenga batch regulariser -----------------------------------
        if self.lambda_lyzenga > 0.0 and ln_blue is not None and ln_green is not None:
            lb_v = ln_blue.reshape(-1)[valid]
            lg_v = ln_green.reshape(-1)[valid]
            lr_v = ln_red.reshape(-1)[valid] if ln_red is not None else None
            fin = torch.isfinite(lb_v) & torch.isfinite(lg_v)
            if lr_v is not None:
                fin = fin & torch.isfinite(lr_v)
            if fin.sum() >= 6:
                lyz_loss = self._lyzenga_batch_residual(
                    mu_v[fin], lb_v[fin], lg_v[fin],
                    lr_v[fin] if lr_v is not None else None,
                )
                total = total + self.lambda_lyzenga * lyz_loss

        # ---- Monotonicity hinge -----------------------------------------
        if self.lambda_monotonic > 0.0 and ln_blue is not None:
            lb_v = ln_blue.reshape(-1)[valid]
            fin = torch.isfinite(lb_v)
            if fin.sum() >= 4:
                mono_loss = self._monotonicity_penalty(mu_v[fin], lb_v[fin])
                total = total + self.lambda_monotonic * mono_loss

        return total
