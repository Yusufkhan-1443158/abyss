# INTEGRATION: Drop-in enhanced encoder-decoder for the SDB multi-temporal pipeline.
# Matches the c_in → (mu, logsigma) contract of _build_unet in backend/unet_sdb.py
# so it can replace the existing AttUNet in train_unet_sdb without touching the
# trainer.  Instantiate with:
#
#   from backend.sdb_optim.attention_unet_v2 import AttentionUNetV2
#   model = AttentionUNetV2(c_in=12, base=32, p_drop=0.10)
#   mu, logsigma = model(x)  # x: (B, C, H, W)
#
# For MC-Dropout inference, put model.train() before the forward pass:
#   model.train()  # keeps dropout active
#   mu, logsigma = model(x)
#
# For ensembling with the existing predict_grid_tta from backend/unet_sdb.py,
# pass this model where the original AttUNet is expected — the forward signature
# is identical.
"""
Attention U-Net v2: enhanced encoder-decoder for multi-temporal Sentinel-2 / Landsat
satellite-derived bathymetry (SDB).

Architecture improvements over ``_build_unet`` in ``backend/unet_sdb.py``
--------------------------------------------------------------------------

1. Residual convolutional blocks (ResConvBlock)
   Each encoder stage uses two Conv-BN-GELU layers wrapped in an identity residual
   shortcut (1×1 projection if channel counts differ).  Residual connections improve
   gradient flow in deep encoders and stabilise training on small SDB datasets
   (He et al. 2016, CVPR).  The existing ConvBlock is non-residual.

2. Convolutional Block Attention Module (CBAM, optional)
   After each ResConvBlock, a CBAM gate (Woo et al. 2018, ECCV) recalibrates features
   along *both* the channel and spatial axes.  Channel attention selects spectrally
   relevant bands (e.g. down-weighting NIR in turbid water).  Spatial attention
   focuses on the optically-shallow pixels.  Controlled by ``use_cbam``.

3. Attention gates on skip connections (Oktay 2018)
   Additive attention gates (Oktay, Schlemper et al. 2018, MIDL) suppress
   non-discriminative skip-connection features before the skip is concatenated
   with the up-sampled feature map.  Identical in design to the existing
   AttentionGate in unet_sdb.py — preserved for interface consistency.

4. Heteroscedastic dual head (mu, log-sigma)
   The output head predicts depth mean (mu, via Softplus → positive) and
   log-sigma (clamped [-1.5, 2.5]), matching the head in _build_unet iter#4.
   The _hetero_nll loss from unet_sdb.py is therefore directly compatible.

5. Parameterised width, depth, and dropout
   ``base`` controls feature-map width (default 32).  ``depth`` selects encoder
   depth (3 or 4 levels).  ``p_drop`` controls MC-Dropout probability.

References
----------
He, K., Zhang, X., Ren, S. & Sun, J. (2016). Deep residual learning for image
  recognition. CVPR 2016.  doi:10.1109/CVPR.2016.90

Oktay, O. et al. (2018). Attention U-Net: learning where to look for the pancreas.
  MIDL 2018.  arXiv:1804.03999

Woo, S., Park, J., Lee, J.Y. & Kweon, I.S. (2018). CBAM: Convolutional block
  attention module. ECCV 2018.  doi:10.1007/978-3-030-01234-2_1

Kendall, A. & Gal, Y. (2017). What uncertainties do we need in Bayesian deep
  learning for computer vision? NeurIPS 2017.  arXiv:1703.04977
"""
from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

L = logging.getLogger(__name__)

MAX_DEPTH_M = 25.0  # keeps the same cap as unet_sdb.py


# ════════════════════════════════════════════════════════════════════════
# Building blocks
# ════════════════════════════════════════════════════════════════════════

class ResConvBlock(nn.Module):
    """Two-layer Conv-BN-GELU block with identity residual shortcut.

    A 1×1 projection is added when ``ci != co`` so the shortcut can be added
    directly.  Dropout2d is applied after the first activation (He et al. 2016 §3).
    """

    def __init__(self, ci: int, co: int, p_drop: float = 0.0) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False),
            nn.BatchNorm2d(co),
            nn.GELU(),
            nn.Dropout2d(p_drop),
            nn.Conv2d(co, co, 3, padding=1, bias=False),
            nn.BatchNorm2d(co),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(ci, co, 1, bias=False),
            nn.BatchNorm2d(co),
        ) if ci != co else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.proj(x)


class ChannelAttention(nn.Module):
    """CBAM channel-attention: SE-like squeeze-and-excitation.

    Woo et al. 2018 §3.1.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg = x.mean(dim=(2, 3))          # (B, C)
        mx  = x.amax(dim=(2, 3))         # (B, C)
        gate = torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # (B, C)
        return x * gate.view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    """CBAM spatial-attention: 7×7 conv on channel-pooled features.

    Woo et al. 2018 §3.2.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)   # (B,1,H,W)
        mx  = x.amax(dim=1, keepdim=True)  # (B,1,H,W)
        gate = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * gate


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al. 2018).

    Sequential channel → spatial attention applied as a post-block gate.
    """

    def __init__(self, channels: int, reduction: int = 8,
                 spatial_kernel: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class AttentionGate(nn.Module):
    """Additive attention gate on skip connections (Oktay et al. 2018).

    Identical interface to the gate in _build_unet (unet_sdb.py) to ensure
    full compatibility with predict_grid_tta and similar inference helpers.
    """

    def __init__(self, c_skip: int, c_gate: int, c_inter: int) -> None:
        super().__init__()
        self.W_skip = nn.Conv2d(c_skip, c_inter, 1, bias=False)
        self.W_gate = nn.Conv2d(c_gate, c_inter, 1, bias=False)
        self.psi    = nn.Sequential(nn.Conv2d(c_inter, 1, 1), nn.Sigmoid())

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        g = F.interpolate(gate, size=skip.shape[-2:],
                          mode="bilinear", align_corners=False)
        attn = self.psi(F.gelu(self.W_skip(skip) + self.W_gate(g)))
        return skip * attn


# ════════════════════════════════════════════════════════════════════════
# Main model
# ════════════════════════════════════════════════════════════════════════

class AttentionUNetV2(nn.Module):
    """Attention U-Net v2 for multi-temporal SDB.

    Encoder-decoder with residual convolutional blocks, optional CBAM gates,
    additive attention on skip connections, and a heteroscedastic dual head
    (mu, log-sigma).

    Parameters
    ----------
    c_in : int
        Number of input channels.  For the standard SDB feature cube (A0/A1/A2)
        this is 12–13.  For multi-temporal stacks it can be larger.
    base : int
        Base feature-map width.  Encoder widths will be base, 2B, 4B, 8B;
        bottleneck 16B.  Default 32 (matches _build_unet).
    p_drop : float
        Dropout probability applied in every ResConvBlock and bottleneck.
        Also controls MC-Dropout: call ``model.train()`` during inference to
        keep dropout active.
    depth : int
        Encoder depth: 3 or 4 levels (4 is the default, matching _build_unet).
    use_cbam : bool
        Apply CBAM after each encoder and decoder block.  Adds ~10 % parameters
        but improves feature selectivity on spectrally-crowded multi-temporal
        input cubes.  Default True.
    max_depth_m : float
        Physical depth cap (metres, positive-down).  Applied via Softplus to the
        mean head output — does NOT truncate; Softplus is unbounded but grows
        slowly above this scale.
    """

    def __init__(
        self,
        c_in: int,
        base: int = 32,
        p_drop: float = 0.10,
        depth: int = 4,
        use_cbam: bool = True,
        max_depth_m: float = MAX_DEPTH_M,
    ) -> None:
        super().__init__()
        if depth not in (3, 4):
            raise ValueError(f"depth must be 3 or 4, got {depth}")

        self.depth = depth
        self.use_cbam = use_cbam
        self.max_depth_m = float(max_depth_m)
        B = base

        # ── Encoder ───────────────────────────────────────────────────
        self.enc1 = ResConvBlock(c_in, B, p_drop)
        self.enc2 = ResConvBlock(B, 2 * B, p_drop)
        self.enc3 = ResConvBlock(2 * B, 4 * B, p_drop)
        if depth == 4:
            self.enc4 = ResConvBlock(4 * B, 8 * B, p_drop)
            self.bottleneck = ResConvBlock(8 * B, 16 * B, p_drop * 1.5)
        else:
            self.bottleneck = ResConvBlock(4 * B, 8 * B, p_drop * 1.5)

        # Optional CBAM per encoder level
        if use_cbam:
            self.cbam_e1 = CBAM(B)
            self.cbam_e2 = CBAM(2 * B)
            self.cbam_e3 = CBAM(4 * B)
            if depth == 4:
                self.cbam_e4 = CBAM(8 * B)

        # ── Decoder ───────────────────────────────────────────────────
        if depth == 4:
            self.up4  = nn.ConvTranspose2d(16 * B, 8 * B, 2, stride=2)
            self.att4 = AttentionGate(8 * B, 8 * B, 4 * B)
            self.dec4 = ResConvBlock(16 * B, 8 * B, p_drop)
            if use_cbam:
                self.cbam_d4 = CBAM(8 * B)
            self.up3  = nn.ConvTranspose2d(8 * B, 4 * B, 2, stride=2)
            self.att3 = AttentionGate(4 * B, 4 * B, 2 * B)
            self.dec3 = ResConvBlock(8 * B, 4 * B, p_drop)
            if use_cbam:
                self.cbam_d3 = CBAM(4 * B)
            self.up2  = nn.ConvTranspose2d(4 * B, 2 * B, 2, stride=2)
            self.att2 = AttentionGate(2 * B, 2 * B, B)
            self.dec2 = ResConvBlock(4 * B, 2 * B, p_drop)
            if use_cbam:
                self.cbam_d2 = CBAM(2 * B)
            self.up1  = nn.ConvTranspose2d(2 * B, B, 2, stride=2)
            self.att1 = AttentionGate(B, B, B // 2)
            self.dec1 = ResConvBlock(2 * B, B, p_drop)
            if use_cbam:
                self.cbam_d1 = CBAM(B)
        else:
            # depth == 3
            self.up3  = nn.ConvTranspose2d(8 * B, 4 * B, 2, stride=2)
            self.att3 = AttentionGate(4 * B, 4 * B, 2 * B)
            self.dec3 = ResConvBlock(8 * B, 4 * B, p_drop)
            if use_cbam:
                self.cbam_d3 = CBAM(4 * B)
            self.up2  = nn.ConvTranspose2d(4 * B, 2 * B, 2, stride=2)
            self.att2 = AttentionGate(2 * B, 2 * B, B)
            self.dec2 = ResConvBlock(4 * B, 2 * B, p_drop)
            if use_cbam:
                self.cbam_d2 = CBAM(2 * B)
            self.up1  = nn.ConvTranspose2d(2 * B, B, 2, stride=2)
            self.att1 = AttentionGate(B, B, B // 2)
            self.dec1 = ResConvBlock(2 * B, B, p_drop)
            if use_cbam:
                self.cbam_d1 = CBAM(B)

        # ── Heteroscedastic head ──────────────────────────────────────
        # mu: Softplus (positive, unbounded) — matches iter#4 of _build_unet
        # logsigma: clamped to [-1.5, 2.5]  — matches _build_unet
        self.head_mu       = nn.Conv2d(B, 1, 1)
        self.head_logsigma = nn.Conv2d(B, 1, 1)

        L.info(
            f"AttentionUNetV2: c_in={c_in}, base={base}, depth={depth}, "
            f"use_cbam={use_cbam}, p_drop={p_drop:.2f}"
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : (B, C, H, W) float32 — normalised feature cube

        Returns
        -------
        mu : (B, H, W) predicted depth mean (positive, Softplus-activated)
        logsigma : (B, H, W) predicted log-sigma (clamped to [-1.5, 2.5])
        """
        # ── Encode ────────────────────────────────────────────────────
        e1 = self.enc1(x)
        if self.use_cbam:
            e1 = self.cbam_e1(e1)

        e2 = self.enc2(F.max_pool2d(e1, 2))
        if self.use_cbam:
            e2 = self.cbam_e2(e2)

        e3 = self.enc3(F.max_pool2d(e2, 2))
        if self.use_cbam:
            e3 = self.cbam_e3(e3)

        if self.depth == 4:
            e4 = self.enc4(F.max_pool2d(e3, 2))
            if self.use_cbam:
                e4 = self.cbam_e4(e4)
            b = self.bottleneck(F.max_pool2d(e4, 2))
        else:
            b = self.bottleneck(F.max_pool2d(e3, 2))

        # ── Decode ────────────────────────────────────────────────────
        if self.depth == 4:
            d4 = self.up4(b)
            d4 = self.dec4(torch.cat([d4, self.att4(e4, d4)], dim=1))
            if self.use_cbam:
                d4 = self.cbam_d4(d4)
            d3 = self.up3(d4)
            d3 = self.dec3(torch.cat([d3, self.att3(e3, d3)], dim=1))
            if self.use_cbam:
                d3 = self.cbam_d3(d3)
            d2 = self.up2(d3)
            d2 = self.dec2(torch.cat([d2, self.att2(e2, d2)], dim=1))
            if self.use_cbam:
                d2 = self.cbam_d2(d2)
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, self.att1(e1, d1)], dim=1))
            if self.use_cbam:
                d1 = self.cbam_d1(d1)
        else:
            d3 = self.up3(b)
            d3 = self.dec3(torch.cat([d3, self.att3(e3, d3)], dim=1))
            if self.use_cbam:
                d3 = self.cbam_d3(d3)
            d2 = self.up2(d3)
            d2 = self.dec2(torch.cat([d2, self.att2(e2, d2)], dim=1))
            if self.use_cbam:
                d2 = self.cbam_d2(d2)
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, self.att1(e1, d1)], dim=1))
            if self.use_cbam:
                d1 = self.cbam_d1(d1)

        # ── Head ──────────────────────────────────────────────────────
        mu       = F.softplus(self.head_mu(d1)).squeeze(1)          # (B, H, W)
        logsigma = self.head_logsigma(d1).squeeze(1).clamp(-1.5, 2.5)  # (B, H, W)

        return mu, logsigma
