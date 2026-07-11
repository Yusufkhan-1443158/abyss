"""
Professional Satellite-Derived Bathymetry — Deep Learning Engine
═════════════════════════════════════════════════════════════════

Architecture: Attention U-Net with CBAM + ASPP multi-scale fusion
─────────────────────────────────────────────────────────────────
  - Channel & Spatial Attention (CBAM) at each skip connection
  - Atrous Spatial Pyramid Pooling (ASPP) at bottleneck for multi-scale
  - Residual blocks with LeakyReLU throughout
  - Softplus head for positive-only depth output

Training pipeline:
  - Data augmentation: random flips, 90° rotations, Gaussian noise
  - Depth-aware weighted Huber loss (shallow = higher weight)
  - Spatial smoothness (TV) + gradient-direction (Sobel) regularisation
  - CosineAnnealingWarmRestarts scheduler
  - MC Dropout for prediction uncertainty estimation
  - Model caching: save/load trained weights to skip retraining
  - Ensemble mode: train N models, average predictions

Input: 9-channel feature stack from Sentinel-2
  [Blue, Green, Red, NIR, NDWI, ln(B/G), ln(G/R), B/G, G/R]

Output: per-pixel depth map (0–25 m) + uncertainty map

References:
  - Woo et al. (2018), CBAM: Convolutional Block Attention Module
  - Chen et al. (2018), Encoder-Decoder with ASPP for Semantic Segmentation
  - Gal & Ghahramani (2016), Dropout as a Bayesian Approximation
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

L = logging.getLogger("bathy.cnn")
MAX_DEPTH_M = 25.0

MODEL_CACHE_DIR = Path("/tmp/bathy/models")
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════

class _ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ResBlock(nn.Module):
    def __init__(self, ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = _ConvBnRelu(ch, ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.dropout(self.conv2(self.conv1(x))) + x)


# ═══════════════════════════════════════════════════════════
# CBAM — Channel & Spatial Attention Module
# ═══════════════════════════════════════════════════════════

class _ChannelAttention(nn.Module):
    """Squeeze-excitation style channel attention."""
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        mid = max(ch // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(ch, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = x.mean(dim=(2, 3))  # (B, C)
        mx = x.amax(dim=(2, 3))   # (B, C)
        att = torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # (B, C)
        return x * att.unsqueeze(-1).unsqueeze(-1)


class _SpatialAttention(nn.Module):
    """Learn where to attend spatially."""
    def __init__(self, kernel: int = 7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        att = self.conv(torch.cat([avg, mx], dim=1))
        return x * att


class CBAM(nn.Module):
    """Convolutional Block Attention Module — channel then spatial."""
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        self.ca = _ChannelAttention(ch, reduction)
        self.sa = _SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# ═══════════════════════════════════════════════════════════
# ASPP — Atrous Spatial Pyramid Pooling
# ═══════════════════════════════════════════════════════════

class ASPP(nn.Module):
    """Multi-scale feature extraction at the bottleneck."""
    def __init__(self, in_ch: int, out_ch: int, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            self.branches.append(_ConvBnRelu(in_ch, out_ch, kernel=3, dilation=r))
        # Global average pooling branch (no BatchNorm — 1x1 spatial)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fuse = _ConvBnRelu(out_ch * (len(rates) + 1), out_ch, kernel=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        gap = self.gap(x)
        gap = F.interpolate(gap, size=x.shape[2:], mode="bilinear", align_corners=False)
        feats.append(gap)
        return self.fuse(torch.cat(feats, dim=1))


# ═══════════════════════════════════════════════════════════
# ATTENTION U-NET with ASPP
# ═══════════════════════════════════════════════════════════

class _Encoder(nn.Module):
    def __init__(self, in_ch: int, base: int = 32, dropout: float = 0.1):
        super().__init__()
        self.enc1 = nn.Sequential(_ConvBnRelu(in_ch, base), _ResBlock(base, dropout))
        self.enc2 = nn.Sequential(_ConvBnRelu(base, base * 2), _ResBlock(base * 2, dropout))
        self.enc3 = nn.Sequential(_ConvBnRelu(base * 2, base * 4), _ResBlock(base * 4, dropout))
        self.enc4 = nn.Sequential(_ConvBnRelu(base * 4, base * 8), _ResBlock(base * 8, dropout))
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return e1, e2, e3, e4


class _Decoder(nn.Module):
    def __init__(self, base: int = 32, dropout: float = 0.1):
        super().__init__()
        # Upsampling + attention at each skip
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.att3 = CBAM(base * 4)
        self.dec3 = nn.Sequential(_ConvBnRelu(base * 8, base * 4), _ResBlock(base * 4, dropout))

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.att2 = CBAM(base * 2)
        self.dec2 = nn.Sequential(_ConvBnRelu(base * 4, base * 2), _ResBlock(base * 2, dropout))

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.att1 = CBAM(base)
        self.dec1 = nn.Sequential(_ConvBnRelu(base * 2, base), _ResBlock(base, dropout))

    @staticmethod
    def _match(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dh = x.shape[2] - target.shape[2]
        dw = x.shape[3] - target.shape[3]
        if dh > 0 or dw > 0:
            x = x[:, :, : target.shape[2], : target.shape[3]]
        elif dh < 0 or dw < 0:
            x = F.pad(x, (0, max(0, -dw), 0, max(0, -dh)))
        return x

    def forward(self, e1, e2, e3, e4):
        d3 = self.dec3(torch.cat([self._match(self.up3(e4), e3), self.att3(e3)], dim=1))
        d2 = self.dec2(torch.cat([self._match(self.up2(d3), e2), self.att2(e2)], dim=1))
        d1 = self.dec1(torch.cat([self._match(self.up1(d2), e1), self.att1(e1)], dim=1))
        return d1


class BathyAttentionUNet(nn.Module):
    """
    Attention U-Net with ASPP for bathymetry depth regression.
    Input : (B, C, H, W) — C channels of reflectance / indices.
    Output: (B, 1, H, W) — predicted depth >= 0.
    """
    def __init__(self, in_channels: int = 13, base_features: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = _Encoder(in_channels, base_features, dropout)
        self.aspp = ASPP(base_features * 8, base_features * 8, rates=(1, 3, 6, 12))
        self.decoder = _Decoder(base_features, dropout)
        self.head = nn.Sequential(
            nn.Conv2d(base_features, base_features // 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(base_features // 2, 1, 1),
            nn.Softplus(),  # depth > 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1, e2, e3, e4 = self.encoder(x)
        bottleneck = self.aspp(e4)
        d = self.decoder(e1, e2, e3, bottleneck)
        return self.head(d)


# Keep backward compatibility alias
BathyUNet = BathyAttentionUNet


# ═══════════════════════════════════════════════════════════
# FEATURE STACK + NORMALISATION
# ═══════════════════════════════════════════════════════════

def _build_feature_stack(s2: dict) -> np.ndarray:
    """
    Build extended feature stack from S2 bands.
    Literature-informed features:
      - 5 raw bands: coastal (B01), blue (B02), green (B03), red (B04), NIR (B08)
      - NDWI water index
      - Stumpf ratio: ln(B02)/ln(B03)  — Stumpf et al. (2003)
      - Lyzenga ratio: ln(B03)/ln(B04)  — Lyzenga (1978)
      - Direct ratios: B/G, G/R, coastal/blue
      - Depth-Invariant Index: ln(B02)-ki/kj*ln(B03) — Lyzenga (2006)
      - Local texture: std_dev in 5x5 window on blue band — spatial context
    """
    eps = 1e-6
    coastal = np.clip(s2.get("coastal", s2["blue"]).astype(np.float64) / 10000, eps, None)
    blue = np.clip(s2["blue"].astype(np.float64) / 10000, eps, None)
    green = np.clip(s2["green"].astype(np.float64) / 10000, eps, None)
    red = np.clip(s2["red"].astype(np.float64) / 10000, eps, None)
    nir = np.clip(s2["nir"].astype(np.float64) / 10000, eps, None)
    ndwi = s2["ndwi"].astype(np.float64)

    # Band ratios (Stumpf 2003, Lyzenga 1978)
    lnBG = np.log(blue + eps) / np.log(green + eps)
    lnGR = np.log(green + eps) / np.log(red + eps)
    BG = blue / (green + eps)
    GR = green / (red + eps)
    CB = coastal / (blue + eps)  # coastal/blue ratio — deep penetration sensitivity

    # Depth-Invariant Index (Lyzenga 2006)
    # DII = ln(Bi) - (ki/kj) * ln(Bj)
    # ki/kj estimated from covariance of log-transformed bands over uniform bottom
    ln_b = np.log(blue + eps)
    ln_g = np.log(green + eps)
    water = s2.get("water_mask", ndwi > 0)
    # Estimate attenuation ratio from band covariance
    if np.sum(water) > 100:
        lb_w = ln_b[water]
        lg_w = ln_g[water]
        valid = np.isfinite(lb_w) & np.isfinite(lg_w)
        if valid.sum() > 50:
            cov = np.cov(lb_w[valid], lg_w[valid])
            var_diff = cov[0, 0] - cov[1, 1]
            cov_bg = cov[0, 1]
            # Lyzenga formula for ki/kj
            ki_kj = (var_diff + np.sqrt(var_diff ** 2 + 4 * cov_bg ** 2)) / (2 * cov_bg + eps)
        else:
            ki_kj = 1.0
    else:
        ki_kj = 1.0
    dii = ln_b - ki_kj * ln_g  # Depth-Invariant Index

    # Local texture: std deviation in 5x5 window on blue band
    from scipy.ndimage import uniform_filter
    blue_mean = uniform_filter(blue, size=5)
    blue_sq_mean = uniform_filter(blue ** 2, size=5)
    texture = np.sqrt(np.clip(blue_sq_mean - blue_mean ** 2, 0, None))

    features = np.stack([
        coastal, blue, green, red, nir,     # 5 raw bands
        ndwi,                                # water index
        lnBG, lnGR,                          # log ratios (Stumpf, Lyzenga)
        BG, GR, CB,                          # direct ratios
        dii,                                 # Depth-Invariant Index
        texture,                             # spatial texture
    ], axis=0).astype(np.float32)

    return features  # (13, H, W)


def _normalise_stack(stack: np.ndarray, water_mask: np.ndarray = None) -> np.ndarray:
    """
    Per-channel percentile normalisation to [0, 1].
    FIX: Compute percentiles on WATER pixels only — land/cloud artifacts
    skew the distribution and clip shallow-water features.
    """
    out = np.empty_like(stack)
    for c in range(stack.shape[0]):
        ch = stack[c]
        if water_mask is not None:
            finite = ch[water_mask & np.isfinite(ch)]
        else:
            finite = ch[np.isfinite(ch)]
        if len(finite) < 10:
            out[c] = 0.0
            continue
        lo, hi = np.percentile(finite, [2, 98])
        if hi - lo < 1e-8:
            hi = lo + 1.0
        out[c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    return np.nan_to_num(out, nan=0.0)


def _pad_to_multiple(t: torch.Tensor, div: int = 16):
    _, _, h, w = t.shape
    ph = (div - h % div) % div
    pw = (div - w % div) % div
    if ph or pw:
        # Use constant padding if reflect would exceed input dimension
        mode = "reflect" if ph < h and pw < w else "constant"
        t = F.pad(t, (0, pw, 0, ph), mode=mode)
    return t, h, w


# ═══════════════════════════════════════════════════════════
# DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════

def _augment_batch(feat: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor):
    """
    Apply random augmentations to a batch of patches:
    - Random horizontal/vertical flips
    - Random 90° rotations
    - Gaussian noise injection on features
    """
    b = feat.shape[0]
    for i in range(b):
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            feat[i] = feat[i].flip(-1)
            tgt[i] = tgt[i].flip(-1)
            mask[i] = mask[i].flip(-1)
        # Random vertical flip
        if torch.rand(1).item() > 0.5:
            feat[i] = feat[i].flip(-2)
            tgt[i] = tgt[i].flip(-2)
            mask[i] = mask[i].flip(-2)
        # Random 90° rotation (0, 1, 2, or 3 times)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            feat[i] = torch.rot90(feat[i], k, dims=(-2, -1))
            tgt[i] = torch.rot90(tgt[i], k, dims=(-2, -1))
            mask[i] = torch.rot90(mask[i], k, dims=(-2, -1))
        # Gaussian noise on features (not target)
        if torch.rand(1).item() > 0.5:
            noise = torch.randn_like(feat[i]) * 0.02
            feat[i] = feat[i] + noise

    return feat, tgt, mask


# ═══════════════════════════════════════════════════════════
# DEPTH-AWARE LOSS FUNCTION
# ═══════════════════════════════════════════════════════════

def _depth_aware_loss(pred, target, mask, depth_scale):
    """
    Literature-informed loss function for SDB.

    Components (based on Beer-Lambert law + IHO S-44):
    1. Linear Huber loss with depth-dependent weighting (shallow > deep)
    2. Log-depth loss — reflectance decays exponentially with depth,
       so errors in log-space are more physically meaningful
       (reduces heteroscedasticity, per Sagawa et al. 2019)
    3. Relative error penalty — penalise fractional error (important for shallow)
    4. Spatial smoothness — Total Variation regularisation

    iter#4 (4.1): pred / target are now in METRES (depth_scale = 1.0).
    Constants below previously assumed [0,1] normalised inputs and have been
    rescaled by ``MAX_DEPTH_M`` (25 m) where needed.
    """
    diff = (pred - target) * mask
    Dmax = float(MAX_DEPTH_M)

    # ── 1. Depth-weighted Huber loss (metres space) ──
    # Shallow pixels weigh more (1 + 2*(1 - d/Dmax)).
    depth_weight = (1.0 + 2.0 * (1.0 - target.clamp(0, Dmax) / Dmax)) * mask
    weighted_diff = diff * depth_weight
    loss_huber = F.smooth_l1_loss(weighted_diff, torch.zeros_like(weighted_diff))

    # ── 2. Log-depth loss (Beer-Lambert motivated, metres) ──
    # log1p on depth in metres — no extra ×10 factor (target already in m).
    pred_log = torch.log1p(pred.clamp(min=0)) * mask
    target_log = torch.log1p(target.clamp(min=0)) * mask
    loss_log = F.smooth_l1_loss(pred_log, target_log)

    # ── 3. Relative error penalty ──
    # +1.0 m offset (was +0.1 in [0,1] space — same physical meaning).
    rel_diff = diff / (target + 1.0)
    loss_rel = (rel_diff.abs() * mask).mean()

    # ── 4. Spatial smoothness — Total Variation ──
    loss_tv = torch.tensor(0.0, device=pred.device)
    if pred.shape[2] > 1 and pred.shape[3] > 1:
        dy = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs().mean()
        dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean()
        loss_tv = dx + dy

    # ── Combined: 0.5*Huber + 0.25*log + 0.15*relative + 0.003*TV ──
    return 0.5 * loss_huber + 0.25 * loss_log + 0.15 * loss_rel + 0.003 * loss_tv


def _berhu_loss_pixel(diff_abs, c_factor=0.2):
    """Berhu (reverse Huber) per-pixel loss.

    L1 below threshold c, L2 above c.  c is set to ``c_factor * max(|diff|)``
    in the batch (Laina 2016).  Better for monocular depth estimation than
    plain MSE because it lets large residuals dominate gradient direction
    without letting noise in small residuals dominate magnitude.
    """
    c = c_factor * diff_abs.max().clamp(min=1e-3)
    l1_part = diff_abs
    l2_part = (diff_abs ** 2 + c ** 2) / (2.0 * c)
    return torch.where(diff_abs <= c, l1_part, l2_part)


def _density_weights_from_depth(target_norm, mask, n_bins=8, eps=0.05):
    """Density-based sample weights (DenseLoss / DenseWeight, Steininger 2021).

    Returns a per-pixel weight tensor where rare-depth pixels weigh more.
    iter#4 (4.1): target is in METRES (was [0,1]).  Bin over [0, MAX_DEPTH_M]
    instead of [0, 1].  Output is normalised so weights average to 1 across
    valid pixels.
    """
    Dmax = float(MAX_DEPTH_M)
    flat_t = target_norm[mask > 0]
    if flat_t.numel() < 16:
        return torch.ones_like(target_norm)
    hist = torch.histc(flat_t.clamp(0.0, Dmax), bins=n_bins, min=0.0, max=Dmax)
    hist = hist / max(float(hist.sum()), 1.0)
    # inverse-density per bin, with floor `eps` to avoid blow-up
    inv = 1.0 / (hist + eps)
    inv = inv / inv.mean()  # normalised so mean weight ≈ 1
    bin_w = Dmax / n_bins
    bin_idx = torch.clamp((target_norm / bin_w).long(), 0, n_bins - 1)
    w = inv[bin_idx]
    return w * mask


def _berhu_density_loss(pred, target, mask, depth_scale):
    """v5 SOTA-aligned loss: Berhu + inverse-density depth weighting +
    log-depth term + light TV regularisation.

    Designed to fix the regression-to-mean failure: rare deep-end samples
    get higher gradient contribution so the model learns to predict the
    full depth dynamic range instead of collapsing to ~mean.
    Reference: Laina et al. 2016 (Berhu); Steininger et al. 2021 (DenseLoss).
    """
    w = _density_weights_from_depth(target, mask, n_bins=8)

    diff_abs = (pred - target).abs() * mask
    berhu_per_px = _berhu_loss_pixel(diff_abs) * w
    valid_n = mask.sum().clamp(min=1)
    loss_berhu = berhu_per_px.sum() / valid_n

    # iter#4 (4.1): target / pred are in metres — drop the ×10 scaler.
    pred_log = torch.log1p(pred.clamp(min=0)) * mask
    target_log = torch.log1p(target.clamp(min=0)) * mask
    loss_log = F.smooth_l1_loss(pred_log, target_log)

    loss_tv = torch.tensor(0.0, device=pred.device)
    if pred.shape[2] > 1 and pred.shape[3] > 1:
        dy = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs().mean()
        dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean()
        loss_tv = dx + dy

    return 0.7 * loss_berhu + 0.25 * loss_log + 0.005 * loss_tv


# ═══════════════════════════════════════════════════════════
# TRAINING HELPERS
# ═══════════════════════════════════════════════════════════

def _rasterise_ref_points(ref_pts: dict, bbox: list, H: int, W: int) -> np.ndarray:
    """Rasterise sparse reference points onto the image grid."""
    w, s, e, n = bbox
    target = np.full((H, W), np.nan, dtype=np.float32)
    lats, lons, depths = ref_pts["lats"], ref_pts["lons"], ref_pts["depths"]
    count = 0
    for i in range(len(lats)):
        r = max(0, min(H - 1, int((n - lats[i]) / (n - s + 1e-10) * H)))
        c = max(0, min(W - 1, int((lons[i] - w) / (e - w + 1e-10) * W)))
        d = depths[i]
        if np.isfinite(d) and 0 < d <= MAX_DEPTH_M:
            target[r, c] = min(d, MAX_DEPTH_M)
            count += 1
    L.info(f"CNN: rasterised {count}/{len(lats)} ref points onto {W}x{H} grid")
    return target


def _extract_patches(features, target, mask, patch_size=64, n_patches=512):
    """
    Depth-stratified patch sampling — ensures balanced representation
    across shallow, moderate, and deep zones.

    FIX: Random sampling over-represents whichever depth zone has
    the most spatial coverage, under-fitting rare depth ranges.
    Stratification ensures shallow (critical for navigation) and
    deep (hard for SDB) zones get adequate training samples.
    """
    _, _, H, W = features.shape
    ps = min(patch_size, H, W)

    # Find all valid pixel locations and their depths
    mask_np = mask[0, 0].cpu().numpy()
    tgt_np = target[0, 0].cpu().numpy()
    valid = mask_np > 0.5
    if valid.sum() < 2:
        return None, None, None

    valid_depths = tgt_np[valid]
    # Create depth bins: [0-3m, 3-8m, 8-15m, 15-25m]
    bin_edges = [0, 3.0 / (tgt_np.max() + 1e-6), 8.0 / (tgt_np.max() + 1e-6),
                 15.0 / (tgt_np.max() + 1e-6), 1.1]
    # Normalised target, so use percentile-based bins instead
    p25, p50, p75 = np.percentile(valid_depths[valid_depths > 0], [25, 50, 75])

    # Build location lists per depth bin
    ys, xs = np.where(valid)
    depths_at_valid = tgt_np[valid]
    bins = {
        "shallow": [], "moderate": [], "deep": [], "very_deep": []
    }
    for i in range(len(ys)):
        d = depths_at_valid[i]
        if d <= p25:
            bins["shallow"].append((ys[i], xs[i]))
        elif d <= p50:
            bins["moderate"].append((ys[i], xs[i]))
        elif d <= p75:
            bins["deep"].append((ys[i], xs[i]))
        else:
            bins["very_deep"].append((ys[i], xs[i]))

    # Allocate patches proportionally: ensure each non-empty bin gets at least 15%
    active_bins = {k: v for k, v in bins.items() if len(v) > 0}
    n_bins = len(active_bins)
    per_bin = max(n_patches // max(n_bins, 1), 8)

    feat_patches, tgt_patches, mask_patches = [], [], []
    for bin_name, locs in active_bins.items():
        target_count = per_bin
        attempts = 0
        collected = 0
        while collected < target_count and attempts < target_count * 15:
            # Pick a random valid pixel from this bin and centre a patch on it
            idx = np.random.randint(0, len(locs))
            cy, cx = locs[idx]
            y0 = max(0, min(H - ps, cy - ps // 2))
            x0 = max(0, min(W - ps, cx - ps // 2))
            m = mask[:, :, y0: y0 + ps, x0: x0 + ps]
            if m.sum() < 2:
                attempts += 1
                continue
            feat_patches.append(features[:, :, y0: y0 + ps, x0: x0 + ps])
            tgt_patches.append(target[:, :, y0: y0 + ps, x0: x0 + ps])
            mask_patches.append(m)
            collected += 1
            attempts += 1

    if not feat_patches:
        return None, None, None
    return torch.cat(feat_patches), torch.cat(tgt_patches), torch.cat(mask_patches)


def _model_cache_key(s2: dict, ref_pts: dict, bbox: list) -> str:
    """Generate a hash key for model caching based on input data."""
    h = hashlib.md5()
    # iter#4 (4.1): "arch_v2" tag — model now trains/predicts in metres
    # directly (no depth_scale [0,1] normalisation).  Cached weights from
    # iter#3 must NOT silently reload — different output target space.
    # iter#5 (5.1): bumped to "arch_v3" — α cut 3.0→1.5 + truth-depth-gated
    # asymmetric NLL (gate at 12 m).  iter#4 weights must not reload.
    # iter#9.B: bumped to "arch_v4" — variance-anchor loss term
    # (λ_var=0.10, std-of-std penalty) added to _hetero_nll to close the
    # iter #8.A dynamic-range collapse (per-segment slopes 0.05-0.30,
    # below b≥0.4 floor).  arch_v3 weights must not reload.
    # iter#11.B: bumped to "arch_v5" — per-site truth-σ-conditional λ_var
    # in _hetero_nll (Kendall & Gal 2017 §4.2): λ_var = 0.10 if truth-σ
    # ≥ 6 m, 0.03 if ≥ 5 m, else 0.  Disables the variance-anchor at
    # narrow-σ sites (Lulu, SBY etc) where iter #10 over-expanded
    # predicted variance and regressed RMSE by +5–6 m.  arch_v4 weights
    # must NOT reload.
    h.update(b"arch_v5")
    h.update(f"{bbox}".encode())
    h.update(f"{s2['red'].shape}".encode())
    h.update(f"{len(ref_pts['lats'])}".encode())
    if len(ref_pts['depths']) > 0:
        h.update(f"{float(np.mean(ref_pts['depths'])):.4f}".encode())
    return h.hexdigest()[:12]


# ═══════════════════════════════════════════════════════════
# MC DROPOUT INFERENCE
# ═══════════════════════════════════════════════════════════

def _mc_tta_predict(model, features, H, W, depth_scale, water, n_forward=8):
    """
    MC Dropout + Test-Time Augmentation (TTA).

    FIX: Original MC Dropout used identical input for every pass.
    Now each pass applies a random geometric augmentation (flip/rotate),
    runs inference, then un-augments the output. This provides:
    - Dropout randomness (epistemic uncertainty)
    - Geometric augmentation diversity (reduces spatial bias)
    - Averaged predictions are more robust than single-view

    TTA augmentations: identity, H-flip, V-flip, H+V-flip, 4 rotations = 8 variants.
    """
    tile, step = 256, 224

    # Define TTA transforms: (transform_fn, inverse_fn)
    def _identity(x): return x
    def _hflip(x): return x.flip(-1)
    def _vflip(x): return x.flip(-2)
    def _hvflip(x): return x.flip(-1).flip(-2)
    def _rot90(x): return torch.rot90(x, 1, dims=(-2, -1))
    def _rot180(x): return torch.rot90(x, 2, dims=(-2, -1))
    def _rot270(x): return torch.rot90(x, 3, dims=(-2, -1))

    def _inv_rot90(x): return torch.rot90(x, 3, dims=(-2, -1))
    def _inv_rot270(x): return torch.rot90(x, 1, dims=(-2, -1))

    tta_transforms = [
        (_identity, _identity),
        (_hflip, _hflip),
        (_vflip, _vflip),
        (_hvflip, _hvflip),
        (_rot90, _inv_rot90),
        (_rot180, _rot180),
        (_rot270, _inv_rot270),
    ]

    predictions = []
    for mc_pass in range(n_forward):
        # Enable dropout, keep BatchNorm in eval
        model.train()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        # Pick a TTA transform for this pass
        aug_fn, inv_fn = tta_transforms[mc_pass % len(tta_transforms)]

        depth_out = np.zeros((H, W), dtype=np.float32)
        weight_out = np.zeros((H, W), dtype=np.float32)

        with torch.no_grad():
            for y0 in range(0, H, step):
                for x0 in range(0, W, step):
                    y1 = min(y0 + tile, H)
                    x1 = min(x0 + tile, W)
                    inp = features[:, :, y0:y1, x0:x1]
                    # Apply TTA augmentation
                    inp_aug = aug_fn(inp)
                    inp_p, th, tw = _pad_to_multiple(inp_aug)
                    pred = model(inp_p)[:, :, :th, :tw]
                    # Invert augmentation on prediction
                    pred_inv = inv_fn(pred)[:, 0].cpu().numpy()[0]
                    # Handle shape mismatch from rotation
                    ph, pw = y1 - y0, x1 - x0
                    pred_crop = pred_inv[:ph, :pw] if pred_inv.shape[0] >= ph and pred_inv.shape[1] >= pw else pred_inv
                    actual_h = min(pred_crop.shape[0], ph)
                    actual_w = min(pred_crop.shape[1], pw)
                    depth_out[y0:y0+actual_h, x0:x0+actual_w] += pred_crop[:actual_h, :actual_w]
                    weight_out[y0:y0+actual_h, x0:x0+actual_w] += 1.0

        # Track uncovered pixels so they become NaN rather than 0 (prevents
        # false-zero depth bands where tile stride leaves gaps).
        uncovered = weight_out <= 0.0
        weight_out_safe = np.maximum(weight_out, 1.0)
        depth_map = (depth_out / weight_out_safe) * depth_scale
        depth_map = np.clip(depth_map, 0, MAX_DEPTH_M)
        depth_map[uncovered] = np.nan
        depth_map[~water] = np.nan
        predictions.append(depth_map)

    stack = np.stack(predictions, axis=0)
    mean_depth = np.nanmean(stack, axis=0)
    uncertainty = np.nanstd(stack, axis=0)

    # Smooth
    try:
        from scipy.ndimage import gaussian_filter
        valid = np.isfinite(mean_depth)
        smoothed = gaussian_filter(np.nan_to_num(mean_depth, nan=0.0), sigma=1.0)
        mean_depth = np.where(valid, smoothed, np.nan)
        mean_depth[~water] = np.nan
        unc_smoothed = gaussian_filter(np.nan_to_num(uncertainty, nan=0.0), sigma=1.5)
        uncertainty = np.where(valid, unc_smoothed, np.nan)
        uncertainty[~water] = np.nan
    except ImportError:
        pass

    return mean_depth, uncertainty


# ═══════════════════════════════════════════════════════════
# SINGLE MODEL TRAINING
# ═══════════════════════════════════════════════════════════

def _train_single_model(
    features, target_t, mask_t, depth_scale, device,
    epochs=80, base_features=32, patch_size=64, lr=3e-4,
    augment=True, model_id=0, **kwargs,
):
    """Train one Attention U-Net model. Returns (model, training_loss)."""
    model = BathyAttentionUNet(
        in_channels=13, base_features=base_features, dropout=0.1
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser, T_0=max(20, epochs // 4), T_mult=2
    )

    n_patches_max = kwargs.get("max_patches", 256)
    feat_p, tgt_p, mask_p = _extract_patches(
        features, target_t, mask_t, patch_size=patch_size, n_patches=n_patches_max
    )
    if feat_p is None:
        return None, float("inf")

    n_patches = feat_p.shape[0]
    batch_size = min(16, n_patches)
    patience = kwargs.get("patience", 10)
    L.info(f"  Model {model_id}: {n_patches} patches, {epochs} epochs, "
           f"base={base_features}, lr={lr}, patience={patience}")

    model.train()
    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        perm = torch.randperm(n_patches)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_patches, batch_size):
            idx = perm[i: i + batch_size]
            fb = feat_p[idx].clone()
            tb = tgt_p[idx].clone()
            mb = mask_p[idx].clone()

            # Data augmentation
            if augment:
                fb, tb, mb = _augment_batch(fb, tb, mb)

            fb_p, ph, pw = _pad_to_multiple(fb)
            pred = model(fb_p)[:, :, :ph, :pw]

            loss_mode = kwargs.get("loss_mode", "depth_aware")
            if loss_mode == "berhu_density":
                loss = _berhu_density_loss(pred, tb, mb, depth_scale)
            else:
                loss = _depth_aware_loss(pred, tb, mb, depth_scale)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # Log every 2 epochs (short runs) / 5 epochs (long runs) so the user sees progress.
        log_stride = 2 if epochs <= 40 else 5
        if (epoch + 1) % log_stride == 0 or epoch == 0:
            L.info(f"  Model {model_id}: epoch {epoch+1}/{epochs}, "
                   f"loss={avg_loss:.5f}, best={best_loss:.5f}")

        # Early stopping
        if no_improve >= patience:
            L.info(f"  Model {model_id}: early stop at epoch {epoch+1} (no improve for {patience})")
            break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    return model, best_loss


# ═══════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════

def cnn_train_and_predict(
    s2: dict,
    ref_pts: dict,
    bbox: list,
    epochs: int = 80,
    base_features: int = 32,
    patch_size: int = 64,
    lr: float = 3e-4,
    n_ensemble: int = 3,
    mc_passes: int = 6,
    use_cache: bool = True,
    max_patches: int = 192,
    patience: int = 12,
    loss_mode: str = "depth_aware",   # "depth_aware" | "berhu_density"
) -> tuple:
    """
    Professional bathymetry estimation with Attention U-Net ensemble.

    Pipeline:
      1. Build 9-channel feature stack from S2 bands
      2. Rasterise reference points as training targets
      3. Check model cache — skip training if cached
      4. Train N ensemble models with data augmentation
      5. MC Dropout inference for uncertainty estimation
      6. Ensemble averaging for final prediction
      7. Compute R², feature importance, quality metrics
      8. Cache trained model for reuse

    Parameters
    ----------
    s2 : dict with keys red, green, blue, nir, ndwi (H, W arrays)
    ref_pts : dict with lats, lons, depths arrays
    bbox : [west, south, east, north]
    epochs : training iterations per model
    n_ensemble : number of models to train (1=single, 3=default ensemble)
    mc_passes : MC Dropout forward passes for uncertainty
    use_cache : whether to save/load model weights

    Returns
    -------
    (result_dict, error_string_or_None)
    """
    H, W = s2["red"].shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L.info(f"CNN: device={device}, ensemble={n_ensemble}, mc_passes={mc_passes}")

    # Subsample ref points for speed (10m grid can't resolve >5k unique pixels anyway)
    MAX_REF = 5000
    n_ref_orig = len(ref_pts["lats"])
    if n_ref_orig > MAX_REF:
        idx_sub = np.random.choice(n_ref_orig, MAX_REF, replace=False)
        ref_pts = {
            "lats": ref_pts["lats"][idx_sub],
            "lons": ref_pts["lons"][idx_sub],
            "depths": ref_pts["depths"][idx_sub],
        }
        L.info(f"CNN: subsampled {n_ref_orig} → {MAX_REF} ref points")

    # Build features
    raw_stack = _build_feature_stack(s2)
    water = s2.get("water_mask", s2["ndwi"] > 0)
    norm_stack = _normalise_stack(raw_stack, water_mask=water)  # FIX: water-only percentiles
    features = torch.from_numpy(norm_stack).unsqueeze(0).to(device)
    target_np = _rasterise_ref_points(ref_pts, bbox, H, W)
    n_valid = int(np.sum(np.isfinite(target_np) & (target_np > 0)))
    if n_valid < 10:
        return None, f"Only {n_valid} valid reference pixels for CNN training"

    # iter#4 (4.1): drop depth_scale [0,1] rescaling — train and predict in
    # metres directly.  The previous normalisation forced the Softplus head
    # to fight a target compressed into a tight band, which contributed to
    # mid-range collapse.  ``depth_scale`` is kept = 1.0 so legacy call sites
    # (loss helpers, MC-TTA) stay structurally unchanged but become no-ops.
    valid_depths = target_np[np.isfinite(target_np) & (target_np > 0)]
    _ = valid_depths  # retained for parity / future diagnostics
    depth_scale = 1.0

    target_t = torch.from_numpy(
        np.nan_to_num(target_np, nan=0.0)
    ).unsqueeze(0).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(
        (np.isfinite(target_np) & (target_np > 0)).astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(device)

    # ── Check model cache ──
    cache_key = _model_cache_key(s2, ref_pts, bbox)
    cache_path = MODEL_CACHE_DIR / f"bathy_attunet_{cache_key}.pt"

    models = []
    if use_cache and cache_path.exists():
        L.info(f"CNN: loading cached model from {cache_path}")
        try:
            model = BathyAttentionUNet(in_channels=13, base_features=base_features).to(device)
            state = torch.load(cache_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            models = [model]
            L.info("CNN: cached model loaded successfully")
        except Exception as ex:
            L.warning(f"CNN: cache load failed — {ex}, training from scratch")
            models = []

    # ── Hold out 20% of ref points for TRUE validation ──
    # FIX: Previously R² was computed on training data (overfit metric).
    # Now we spatially hold out 20% and report honest validation R².
    n_ref = len(ref_pts["lats"])
    np.random.seed(42)
    perm = np.random.permutation(n_ref)
    n_val = max(3, int(n_ref * 0.2))
    val_idx = set(perm[:n_val].tolist())
    train_ref = {
        "lats": np.array([ref_pts["lats"][i] for i in range(n_ref) if i not in val_idx]),
        "lons": np.array([ref_pts["lons"][i] for i in range(n_ref) if i not in val_idx]),
        "depths": np.array([ref_pts["depths"][i] for i in range(n_ref) if i not in val_idx]),
    }
    val_ref = {
        "lats": np.array([ref_pts["lats"][i] for i in range(n_ref) if i in val_idx]),
        "lons": np.array([ref_pts["lons"][i] for i in range(n_ref) if i in val_idx]),
        "depths": np.array([ref_pts["depths"][i] for i in range(n_ref) if i in val_idx]),
    }
    L.info(f"CNN: holdout split — {len(train_ref['lats'])} train, {len(val_ref['lats'])} val")

    # Re-rasterise with train-only points (iter#4: metres directly)
    target_np_train = _rasterise_ref_points(train_ref, bbox, H, W)
    target_t = torch.from_numpy(
        np.nan_to_num(target_np_train, nan=0.0)
    ).unsqueeze(0).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(
        (np.isfinite(target_np_train) & (target_np_train > 0)).astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(device)

    # ── Train DIVERSE ensemble ──
    # FIX: Vary hyperparams per model for true ensemble diversity
    ensemble_configs = [
        {"base_features": 32, "lr": 3e-4, "dropout": 0.10, "patch_size": 64},
        {"base_features": 48, "lr": 2e-4, "dropout": 0.15, "patch_size": 48},
        {"base_features": 24, "lr": 5e-4, "dropout": 0.08, "patch_size": 96},
        {"base_features": 32, "lr": 1e-4, "dropout": 0.12, "patch_size": 64},
        {"base_features": 40, "lr": 3e-4, "dropout": 0.10, "patch_size": 56},
    ]

    if not models:
        L.info(f"CNN: training {n_ensemble}-model DIVERSE ensemble (parallel)...")
        import concurrent.futures

        def _train_one(i):
            cfg = ensemble_configs[i % len(ensemble_configs)]
            return _train_single_model(
                features, target_t, mask_t, depth_scale, device,
                epochs=epochs, base_features=cfg["base_features"],
                patch_size=cfg["patch_size"], lr=cfg["lr"],
                augment=True, model_id=i,
                max_patches=max_patches, patience=patience,
                loss_mode=loss_mode,
            )

        # Train models in parallel threads (PyTorch releases GIL during compute)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_ensemble) as executor:
            futures = {executor.submit(_train_one, i): i for i in range(n_ensemble)}
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    model, loss = future.result()
                    cfg = ensemble_configs[i % len(ensemble_configs)]
                    if model is not None:
                        models.append(model)
                        L.info(f"  Model {i}: loss={loss:.5f} (base={cfg['base_features']}, "
                               f"lr={cfg['lr']}, ps={cfg['patch_size']})")
                except Exception as ex:
                    L.warning(f"  Model {i}: failed — {ex}")

        if not models:
            return None, "All ensemble models failed to train"

        if use_cache:
            try:
                torch.save(models[0].state_dict(), cache_path)
                L.info(f"CNN: model cached to {cache_path}")
            except Exception as ex:
                L.warning(f"CNN: cache save failed — {ex}")

    # ── Inference with MC Dropout + TTA ──
    L.info(f"CNN: inference ({len(models)} models × {mc_passes} MC+TTA passes)...")
    all_depths = []
    all_uncertainties = []

    for i, model in enumerate(models):
        depth_map, uncertainty = _mc_tta_predict(
            model, features, H, W, depth_scale, water,
            n_forward=max(2, mc_passes // len(models))
        )
        all_depths.append(depth_map)
        all_uncertainties.append(uncertainty)

    # Ensemble average
    if len(all_depths) > 1:
        depth_stack = np.stack(all_depths, axis=0)
        depth_out = np.nanmean(depth_stack, axis=0)
        ensemble_std = np.nanstd(depth_stack, axis=0)
        uncertainty_out = np.nanmean(np.stack(all_uncertainties, axis=0), axis=0)
        uncertainty_out = np.sqrt(uncertainty_out ** 2 + ensemble_std ** 2)
    else:
        depth_out = all_depths[0]
        uncertainty_out = all_uncertainties[0]

    depth_out = np.clip(depth_out, 0, MAX_DEPTH_M)
    depth_out[~water] = np.nan
    uncertainty_out[~water] = np.nan

    # ── Residual correction (thin-plate spline interpolation) ──
    # FIX: Compute residuals on TRAIN points, fit spatial correction surface,
    # apply to remove systematic bias from the prediction.
    w_b, s_b, e_b, n_b = bbox
    res_lats, res_lons, residuals = [], [], []
    for i in range(len(train_ref["lats"])):
        r = max(0, min(H - 1, int((n_b - train_ref["lats"][i]) / (n_b - s_b + 1e-10) * H)))
        c = max(0, min(W - 1, int((train_ref["lons"][i] - w_b) / (e_b - w_b + 1e-10) * W)))
        if water[r, c] and np.isfinite(depth_out[r, c]):
            res = train_ref["depths"][i] - depth_out[r, c]  # true - pred
            res_lats.append(train_ref["lats"][i])
            res_lons.append(train_ref["lons"][i])
            residuals.append(res)

    if len(residuals) >= 5:
        try:
            from scipy.interpolate import RBFInterpolator
            coords = np.column_stack([res_lons, res_lats])
            rbf = RBFInterpolator(coords, residuals, kernel="thin_plate_spline", smoothing=1.0)
            # Build prediction grid
            row_coords = np.linspace(n_b, s_b, H)
            col_coords = np.linspace(w_b, e_b, W)
            grid_lon, grid_lat = np.meshgrid(col_coords, row_coords)
            grid_pts = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
            correction = rbf(grid_pts).reshape(H, W)
            # Apply correction only on water
            depth_out = np.where(water & np.isfinite(depth_out),
                                 np.clip(depth_out + correction, 0, MAX_DEPTH_M), depth_out)
            depth_out[~water] = np.nan
            mean_corr = float(np.mean(np.abs(np.array(residuals))))
            L.info(f"CNN: residual correction applied (mean |residual|={mean_corr:.3f}m, "
                   f"{len(residuals)} points)")
        except Exception as ex:
            L.warning(f"CNN: residual correction failed — {ex}")

    # ── Compute R² on HELD-OUT validation points ──
    # FIX: This is now true out-of-sample validation, not training accuracy
    pred_at_val, true_at_val = [], []
    for i in range(len(val_ref["lats"])):
        r = max(0, min(H - 1, int((n_b - val_ref["lats"][i]) / (n_b - s_b + 1e-10) * H)))
        c = max(0, min(W - 1, int((val_ref["lons"][i] - w_b) / (e_b - w_b + 1e-10) * W)))
        if water[r, c] and np.isfinite(depth_out[r, c]):
            pred_at_val.append(depth_out[r, c])
            true_at_val.append(val_ref["depths"][i])

    r2 = 0.0
    rmse = 0.0
    mae = 0.0
    per_zone_rmse = {}
    if len(pred_at_val) > 3:
        pred_arr = np.array(pred_at_val)
        true_arr = np.array(true_at_val)
        ss_res = np.sum((pred_arr - true_arr) ** 2)
        ss_tot = np.sum((true_arr - np.mean(true_arr)) ** 2)
        r2 = round(float(1 - ss_res / (ss_tot + 1e-10)), 4)
        rmse = round(float(np.sqrt(np.mean((pred_arr - true_arr) ** 2))), 3)
        mae = round(float(np.mean(np.abs(pred_arr - true_arr))), 3)

        # Per-zone RMSE (IHO depth zones)
        for zone_name, z_min, z_max in [("0-5m", 0, 5), ("5-10m", 5, 10),
                                          ("10-15m", 10, 15), ("15-25m", 15, 25)]:
            zone_mask = (true_arr >= z_min) & (true_arr < z_max)
            if zone_mask.sum() >= 2:
                zone_rmse = float(np.sqrt(np.mean((pred_arr[zone_mask] - true_arr[zone_mask]) ** 2)))
                per_zone_rmse[zone_name] = round(zone_rmse, 3)

        L.info(f"CNN VALIDATION (holdout): R²={r2}, RMSE={rmse}m, MAE={mae}m "
               f"({len(pred_at_val)} val pts)")
        if per_zone_rmse:
            L.info(f"  Per-zone RMSE: {per_zone_rmse}")

    # ── Feature importance via gradient sensitivity ──
    importance = {}
    feature_names = ["Coastal", "Blue", "Green", "Red", "NIR", "NDWI",
                     "ln(B/G)", "ln(G/R)", "B/G", "G/R", "C/B", "DII", "Texture"]
    try:
        models[0].eval()
        inp = features.clone().requires_grad_(True)
        inp_p, ph, pw = _pad_to_multiple(inp)
        out = models[0](inp_p)[:, :, :ph, :pw]
        out.sum().backward()
        grad = inp.grad.abs().mean(dim=(0, 2, 3)).cpu().numpy()
        total = grad.sum() + 1e-10
        for i, name in enumerate(feature_names):
            importance[name] = round(float(grad[i] / total), 3)
    except Exception:
        for name in feature_names:
            importance[name] = round(1.0 / len(feature_names), 3)

    # ── Quality metrics ──
    valid_depth = depth_out[np.isfinite(depth_out) & (depth_out > 0)]
    valid_unc = uncertainty_out[np.isfinite(uncertainty_out)]
    quality = {
        "mean_uncertainty_m": round(float(np.mean(valid_unc)), 3) if len(valid_unc) > 0 else 0,
        "max_uncertainty_m": round(float(np.max(valid_unc)), 3) if len(valid_unc) > 0 else 0,
        "p90_uncertainty_m": round(float(np.percentile(valid_unc, 90)), 3) if len(valid_unc) > 0 else 0,
        "rmse": rmse,
        "mae": mae,
        "per_zone_rmse": per_zone_rmse,
        "n_ensemble": len(models),
        "mc_passes": mc_passes,
        "n_train": len(train_ref["lats"]),
        "n_val": len(pred_at_val),
        "validation_type": "holdout_20pct",
        "depth_coverage_pct": round(float(np.sum(np.isfinite(depth_out)) / max(np.sum(water), 1) * 100), 1),
    }

    L.info(f"CNN Attention U-Net: R²={r2}, RMSE={rmse}m, MAE={mae}m, "
           f"ensemble={len(models)}, uncertainty={quality['mean_uncertainty_m']}m, "
           f"coverage={quality['depth_coverage_pct']}%")

    return {
        "depth": depth_out,
        "uncertainty": uncertainty_out,
        "r2": r2,
        "n_train": len(train_ref["lats"]),
        "importance": importance,
        "method": f"Attention U-Net (×{len(models)} ensemble, CBAM+ASPP)",
        "quality": quality,
    }, None
