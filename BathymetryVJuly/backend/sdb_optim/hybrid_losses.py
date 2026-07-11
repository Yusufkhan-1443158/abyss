# INTEGRATION: Drop-in augmentation of any depth-regression training loop.
# Replace or supplement HeteroNLLLoss from backend/sdb_cnn_baseline.py with
# HybridStructuralLoss to add structural supervision.  The module is a
# torch.nn.Module with the same (pred, target) call signature as HeteroNLLLoss
# when used in pointwise-only mode (lambda_ssim=0, lambda_grad=0).
#
# Primary integration point:
#   from backend.sdb_optim.hybrid_losses import HybridStructuralLoss
#   criterion = HybridStructuralLoss(base_loss, lambda_ssim=0.15, lambda_grad=0.10)
#   # In training loop: loss = criterion(pred_raster, true_raster)
#   # Rasters must be 4-D: (B, 1, H, W)
#
# Works with any pointwise base (HeteroNLLLoss, nn.L1Loss, nn.MSELoss) passed
# as `base`.  The SSIM and gradient terms are computed on the raw (not
# log-transformed) predicted vs true depth rasters.
"""
Hybrid structural loss combining pointwise depth supervision with structural
image-space terms that preserve underwater morphology.

Why structural terms matter
---------------------------
Pointwise RMSE treats every pixel independently; a model minimising it tends to
predict the conditional mean at each pixel, which blurs sharp bathymetric features:
channel edges, reef crests, sand-wave crests, tidal scour pits.  These boundaries
carry critical navigation and sediment-transport information.

Two complementary structural priors are added:

1. Differentiable windowed SSIM (Wang et al. 2004): measures luminance + contrast +
   structural similarity in small windows.  SSIM ~1 forces the *spatial organisation*
   of the predicted depth to match the truth — it is insensitive to global brightness
   but sensitive to local structure.  This prevents depth-plateau artefacts where
   a smooth prediction satisfies RMSE but erases, e.g., a 2 m-deep channel.

   Reference: Z. Wang, A.C. Bovik, H.R. Sheikh, E.P. Simoncelli (2004).
   "Image quality assessment: from error visibility to structural similarity."
   IEEE TIP, 13(4), 600-612.  doi:10.1109/TIP.2003.819861

2. Sobel gradient-matching (first-order depth gradient): the L1 difference of
   horizontal and vertical gradients (Sobel kernels) between predicted and true
   depth.  This penalises a prediction that predicts the right depth values but
   with incorrect slope — i.e., it smooths over channel banks, reef edges, or
   sandy ridges.  Gradient matching is the "sharpness" complement of SSIM's
   "structure" term.

   Reference: Eigen, D. & Fergus, R. (2015). "Predicting depth, surface normals
   and semantic labels with a common multi-scale convolutional architecture."
   ICCV 2015.  doi:10.1109/ICCV.2015.304

Structural terms matter most in shallow (<5 m) domains where:
  - Dune and ripple fields exist (fine-scale depth variation)
  - Channel and shoal boundaries are navigation-critical
  - ICESat-2 transects are sparse → regression-to-mean is the failure mode

The composite loss is:
  L = L_base + lambda_ssim * (1 - SSIM) + lambda_grad * L_grad
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["HybridStructuralLoss", "ssim_loss", "gradient_loss"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """1-D Gaussian kernel, used to build 2-D separable window for SSIM.

    Parameters
    ----------
    window_size : kernel side length (odd; typical 11)
    sigma       : Gaussian spread (typical 1.5)
    device      : target device

    Returns
    -------
    kernel : (window_size, window_size) normalised Gaussian
    """
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    window_2d = g.unsqueeze(1) * g.unsqueeze(0)   # outer product → (W, W)
    return window_2d


def _ssim_map(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    c1: float = 0.01 ** 2,
    c2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Per-pixel SSIM map for 4-D inputs (B, C, H, W).

    The constants c1, c2 use the standard Wang et al. 2004 defaults with
    data_range=1 (normalised depth rasters).  If your depths are not in [0,1]
    pass c1=(k1*L)^2, c2=(k2*L)^2 where L is the data range.

    Returns
    -------
    ssim_map : (B, C, H', W') tensor of local SSIM values in [−1, 1].
    """
    B, C, H, W = pred.shape
    win = _gaussian_kernel(window_size, sigma, pred.device)
    # Expand to (out_ch, in_ch/groups, kH, kW) for depthwise conv
    win = win.unsqueeze(0).unsqueeze(0).expand(C, 1, window_size, window_size)
    pad = window_size // 2

    def _filt(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, win, padding=pad, groups=C)

    mu1 = _filt(pred)
    mu2 = _filt(target)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _filt(pred * pred) - mu1_sq
    sigma2_sq = _filt(target * target) - mu2_sq
    sigma12   = _filt(pred * target) - mu1_mu2

    # Clamp variance to zero (numerical safety)
    sigma1_sq = torch.clamp(sigma1_sq, 0.0)
    sigma2_sq = torch.clamp(sigma2_sq, 0.0)

    numerator   = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_val    = numerator / (denominator + 1e-8)
    return ssim_val


def _sobel_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sobel gradient in x and y for a (B, C, H, W) tensor.

    Returns (grad_x, grad_y) each shape (B, C, H, W), padded same size.
    Sobel filters approximate the first derivative at each pixel.
    """
    B, C, H, W = x.shape
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3).expand(C, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3).expand(C, 1, 3, 3)

    gx = F.conv2d(x, sobel_x, padding=1, groups=C)
    gy = F.conv2d(x, sobel_y, padding=1, groups=C)
    return gx, gy


# ─────────────────────────────────────────────────────────────────────────────
# Public loss functions (usable standalone)
# ─────────────────────────────────────────────────────────────────────────────

def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: Optional[float] = None,
) -> torch.Tensor:
    """SSIM-based loss: 1 − mean(SSIM(pred, target)).

    Parameters
    ----------
    pred        : (B, 1, H, W) predicted depth raster (float32)
    target      : (B, 1, H, W) true depth raster
    window_size : Gaussian window side length (default 11, same as Wang 2004)
    sigma       : Gaussian spread (default 1.5)
    data_range  : if given, rescale constants c1/c2 for the range.
                  Default: assume rasters pre-normalised to [0, 1].

    Returns
    -------
    loss : scalar tensor in [0, 2]  (0 = identical, 2 = anti-correlated)
    """
    if pred.ndim != 4:
        raise ValueError(f"ssim_loss expects 4-D input (B,C,H,W), got {pred.shape}")

    c_scale = 1.0 if data_range is None else (1.0 / data_range)
    c1 = (0.01 * c_scale) ** 2
    c2 = (0.03 * c_scale) ** 2

    s_map = _ssim_map(pred, target, window_size=window_size, sigma=sigma, c1=c1, c2=c2)
    return 1.0 - s_map.mean()


def gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """L1 Sobel gradient matching loss.

    Penalises differences in horizontal and vertical depth gradients between
    prediction and truth.  Encourages the model to preserve channel walls,
    reef crest slopes, and sand-wave gradients that RMSE alone blurs.

    Parameters
    ----------
    pred   : (B, 1, H, W) predicted depth raster
    target : (B, 1, H, W) true depth raster

    Returns
    -------
    loss : scalar, mean absolute gradient error (same units as pred/target)
    """
    if pred.ndim != 4:
        raise ValueError(f"gradient_loss expects 4-D input (B,C,H,W), got {pred.shape}")

    pred_gx, pred_gy = _sobel_gradients(pred)
    true_gx, true_gy = _sobel_gradients(target)

    loss_x = torch.abs(pred_gx - true_gx).mean()
    loss_y = torch.abs(pred_gy - true_gy).mean()
    return (loss_x + loss_y) * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Primary module
# ─────────────────────────────────────────────────────────────────────────────

class HybridStructuralLoss(nn.Module):
    """Combined depth loss: pointwise + SSIM + gradient-matching.

    Combines:
      1. A pointwise base loss (e.g. HeteroNLLLoss, nn.L1Loss, nn.MSELoss).
         The base receives the same arguments passed to forward().
      2. Differentiable windowed SSIM term — preserves spatial structure and
         prevents the depth-plateau artefact common when minimising RMSE alone.
      3. Sobel gradient-matching term — preserves sharp bathymetric transitions
         (channel banks, reef crests) that SSIM can miss at coarse windows.

    Parameters
    ----------
    base         : nn.Module implementing the pointwise depth loss.
                   Called as `base(*args, **kwargs)` in forward().
                   Should return a scalar loss tensor.
    lambda_ssim  : weight on SSIM term (0 = off; typical 0.05–0.20).
    lambda_grad  : weight on gradient term (0 = off; typical 0.05–0.15).
    window_size  : Gaussian window for SSIM (default 11).
    sigma_ssim   : Gaussian sigma for SSIM window (default 1.5).
    data_range   : Depth data range for SSIM constant scaling (default None → [0,1]).
                   Pass MAX_DEPTH_M (e.g. 25.0) if depths are in metres.

    Usage
    -----
    Patch-level (1-D) mode: pass pred_raster=None; structural terms are skipped.
    Raster mode (U-Net training): pass pred_raster and true_raster as 4-D tensors
    via the `pred_raster` / `true_raster` kwargs.

    Example
    -------
    >>> from backend.sdb_cnn_baseline import HeteroNLLLoss
    >>> base = HeteroNLLLoss()
    >>> criterion = HybridStructuralLoss(base, lambda_ssim=0.15, lambda_grad=0.10)
    >>> # Raster U-Net training:
    >>> loss = criterion(mu, lv, target_1d,
    ...                  pred_raster=pred_4d, true_raster=true_4d)
    >>> # Patch CNN (pointwise only):
    >>> loss = criterion(mu, lv, target_1d)
    """

    def __init__(
        self,
        base: nn.Module,
        lambda_ssim: float = 0.15,
        lambda_grad: float = 0.10,
        window_size: int = 11,
        sigma_ssim: float = 1.5,
        data_range: Optional[float] = None,
    ) -> None:
        super().__init__()
        if lambda_ssim < 0 or lambda_grad < 0:
            raise ValueError("lambda_ssim and lambda_grad must be >= 0")
        self.base = base
        self.lambda_ssim = lambda_ssim
        self.lambda_grad = lambda_grad
        self.window_size = window_size
        self.sigma_ssim = sigma_ssim
        self.data_range = data_range

    def forward(
        self,
        *args,
        pred_raster: Optional[torch.Tensor] = None,
        true_raster: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute hybrid loss.

        Parameters
        ----------
        *args, **kwargs : forwarded to self.base (pointwise term)
        pred_raster : (B, 1, H, W) optional predicted depth raster for structural terms
        true_raster : (B, 1, H, W) optional true depth raster for structural terms

        Returns
        -------
        loss_total : scalar tensor
        """
        loss = self.base(*args, **kwargs)

        if pred_raster is not None and true_raster is not None:
            if self.lambda_ssim > 0.0:
                l_ssim = ssim_loss(
                    pred_raster, true_raster,
                    window_size=self.window_size,
                    sigma=self.sigma_ssim,
                    data_range=self.data_range,
                )
                loss = loss + self.lambda_ssim * l_ssim

            if self.lambda_grad > 0.0:
                l_grad = gradient_loss(pred_raster, true_raster)
                loss = loss + self.lambda_grad * l_grad

        return loss
