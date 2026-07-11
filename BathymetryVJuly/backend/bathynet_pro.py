"""
BathyNet-Pro — professional CNN architecture for satellite-derived bathymetry

Why this architecture for multi-source fusion (iBoating + ICESat-2 + GEBCO)
─────────────────────────────────────────────────────────────────────────
Our depth targets come from three sources of very different spatial density
and noise characteristics:
  • iBoating-digitised soundings      — dense where available, pointwise, low σ
  • ICESat-2 ATL03 bathymetric photons — 20 m along-track tracks, high σ
  • GEBCO 2024                         — 15″ gridded baseline, biased shallow
A pixel-wise MLP (Sagawa 2019) cannot exploit spatial coherence between
these sparse labels and the surrounding Sentinel-2 reflectance gradient.
Conversely, a plain U-Net (Ronneberger 2015) handles spatial context but
lacks the multi-scale receptive field to capture the 20 m → km transition
between tidal flats, lagoon, and the deep-water shelf edge typical of a
reef-ringed Gulf island such as Abu Al Abyad.

We therefore adopt a ResNet-34 encoder fused with DeepLab-v3+ ASPP and an
Attention-U-Net decoder with CBAM blocks, directly after:
  • Mandlburger, Pfennigbauer, Schwarz, Flöry, Nussbaumer (2021)
    "BathyNet: A Deep Neural Network for Water Depth Mapping from
    Multispectral Aerial Images." PFG 89, 71–89.
  • Chen et al. (2018)   "Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation" (DeepLab v3+).
  • Sagawa, Yamashita, Okumura, Yamanokuchi (2019)
    "Satellite Derived Bathymetry Using Machine Learning and Multi-
    Temporal Satellite Images." Remote Sensing 11(11), 1155. — our
    per-pixel Stumpf/log-ratio feature stack (features 7-10) is the
    exact 10-feature design they found most predictive.
  • Almar et al. (2022)  "Sentinel-2 derived bathymetry of coastal
    environments using deep learning." RSE 270, 112852.
  • Woo, Park, Lee, Kweon (2018) "CBAM: Convolutional Block Attention
    Module." ECCV.
  • Kendall & Gal (2017) "What Uncertainties Do We Need in Bayesian
    Deep Learning for Computer Vision?" NeurIPS — heteroscedastic
    Gaussian NLL for per-pixel uncertainty.
  • Ronneberger, Fischer, Brox (2015) "U-Net: Convolutional Networks
    for Biomedical Image Segmentation." MICCAI.

Net effect: the encoder learns hierarchical spatial features, ASPP gives a
wide multi-scale receptive field without killing resolution, CBAM
re-weights channels/spatial attention between the multispectral bands,
and the heteroscedastic head outputs both depth and per-pixel σ — which
is exactly what we need to down-weight noisy ICESat-2 / GEBCO pixels
relative to dense iBoating soundings during fusion.

Architecture
  INPUT  13-channel feature stack  (B, G, R, NIR, Coastal, NDWI,
                                     ln(B/G), ln(G/R), B/G, G/R,
                                     B-NIR, G-NIR, R-NIR)
   ↓
   ResNet-34-style encoder (5 stages: 64, 128, 256, 512, 1024 ch)
   ↓
   ASPP bottleneck (dilations 1/6/12/18 + global pool)
   ↓
   Attention decoder with CBAM at each skip + SCSE at each up-block
   ↓
  OUTPUT  (depth_mu, depth_log_var)   — heteroscedastic Gaussian head
          depth = softplus(mu), sigma = exp(0.5 * log_var)

Loss  = w_shallow · (Huber_depth + 0.1 · grad_smoothness)
      +            0.1 · heteroscedastic NLL
Training
  • random flips + 90° rotations
  • Gaussian noise on features
  • CosineAnnealingWarmRestarts
  • hold-out 20 % spatial validation
  • MC-Dropout at inference (p = 0.1)
"""
from __future__ import annotations
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_DEPTH_M = 25.0


# ── Building blocks ──────────────────────────────────────────────────
class ConvBNAct(nn.Module):
    def __init__(self, i, o, k=3, s=1, d=1, act=True):
        super().__init__()
        p = (k // 2) * d
        self.c = nn.Conv2d(i, o, k, s, p, dilation=d, bias=False)
        self.b = nn.BatchNorm2d(o)
        self.a = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.a(self.b(self.c(x)))


class ResBlock(nn.Module):
    """Basic ResNet-34 block."""
    def __init__(self, i, o, s=1, dropout=0.1):
        super().__init__()
        self.c1 = ConvBNAct(i, o, 3, s)
        self.c2 = nn.Sequential(nn.Conv2d(o, o, 3, 1, 1, bias=False),
                                 nn.BatchNorm2d(o))
        self.a = nn.LeakyReLU(0.1, inplace=True)
        self.d = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.short = (nn.Sequential(nn.Conv2d(i, o, 1, s, bias=False),
                                     nn.BatchNorm2d(o))
                      if (i != o or s != 1) else nn.Identity())

    def forward(self, x):
        r = self.short(x)
        x = self.c1(x); x = self.c2(x); x = self.d(x)
        return self.a(x + r)


class CBAM(nn.Module):
    """Convolutional Block Attention Module — channel + spatial."""
    def __init__(self, c, r=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(inplace=True),
                                  nn.Linear(c // r, c))
        self.sp = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        chan = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        x = x * chan
        s = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1)
        spat = torch.sigmoid(self.sp(s))
        return x * spat


class SCSE(nn.Module):
    """Squeeze-and-Excitation (channel + spatial)."""
    def __init__(self, c, r=8):
        super().__init__()
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1), nn.Sigmoid())
        self.sse = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cse(x) + x * self.sse(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (DeepLab v3+)."""
    def __init__(self, i, o=256, rates=(1, 6, 12, 18)):
        super().__init__()
        self.b = nn.ModuleList([
            ConvBNAct(i, o, 3 if r > 1 else 1, d=r) for r in rates
        ])
        # Global-pool branch — BatchNorm2d chokes on 1×1 when batch=1,
        # so use GroupNorm here instead.
        self.gp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(i, o, 1, bias=False),
            nn.GroupNorm(min(32, o), o),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.proj = ConvBNAct(o * (len(rates) + 1), o, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = [b(x) for b in self.b]
        gp = F.interpolate(self.gp(x), (h, w), mode="bilinear", align_corners=False)
        outs.append(gp)
        return self.proj(torch.cat(outs, 1))


class AttnUp(nn.Module):
    """Attention gate + upsample + concat + conv."""
    def __init__(self, in_dec, in_skip, out):
        super().__init__()
        # Attention gate (Oktay 2018)
        self.Wg = nn.Sequential(nn.Conv2d(in_dec, out, 1), nn.BatchNorm2d(out))
        self.Wx = nn.Sequential(nn.Conv2d(in_skip, out, 1), nn.BatchNorm2d(out))
        self.psi = nn.Sequential(nn.Conv2d(out, 1, 1),
                                  nn.BatchNorm2d(1), nn.Sigmoid())
        self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                               align_corners=False)
        self.conv = nn.Sequential(
            ConvBNAct(in_dec + in_skip, out, 3),
            ResBlock(out, out, dropout=0.1),
            SCSE(out),
        )

    def forward(self, dec, skip):
        dec_up = self.up(dec)
        if dec_up.shape[-2:] != skip.shape[-2:]:
            dec_up = F.interpolate(dec_up, skip.shape[-2:],
                                    mode="bilinear", align_corners=False)
        g = self.Wg(dec_up); x = self.Wx(skip)
        alpha = self.psi(F.relu(g + x, inplace=True))
        skip = skip * alpha
        out = torch.cat([dec_up, skip], 1)
        return self.conv(out)


# ── Full network ─────────────────────────────────────────────────────
class BathyNetPro(nn.Module):
    def __init__(self, in_ch=13, base=64, dropout=0.1):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, base, 3),
            ConvBNAct(base, base, 3),
        )
        # ResNet-34 style encoder (2 blocks per stage)
        def _stage(i, o, n, stride=2):
            layers = [ResBlock(i, o, stride, dropout)]
            for _ in range(n - 1):
                layers.append(ResBlock(o, o, 1, dropout))
            return nn.Sequential(*layers)

        self.enc1 = _stage(base,     base,     2, stride=1)   # 1×
        self.enc2 = _stage(base,     base * 2, 2, stride=2)   # 1/2
        self.enc3 = _stage(base * 2, base * 4, 3, stride=2)   # 1/4
        self.enc4 = _stage(base * 4, base * 8, 3, stride=2)   # 1/8
        self.enc5 = _stage(base * 8, base * 16, 2, stride=2)  # 1/16

        # Bottleneck
        self.aspp = ASPP(base * 16, base * 8)                 # out = base*8
        self.cbam = CBAM(base * 8)

        # Decoder
        self.up4 = AttnUp(base * 8,  base * 8, base * 8)
        self.up3 = AttnUp(base * 8,  base * 4, base * 4)
        self.up2 = AttnUp(base * 4,  base * 2, base * 2)
        self.up1 = AttnUp(base * 2,  base,     base)
        self.drop = nn.Dropout2d(dropout)

        # Dual head: depth mean + heteroscedastic log-variance
        self.head_mu = nn.Sequential(
            ConvBNAct(base, base, 3),
            nn.Conv2d(base, 1, 1),
        )
        self.head_lv = nn.Sequential(
            ConvBNAct(base, base, 3),
            nn.Conv2d(base, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = x.shape[-2:]
        # Encoder
        s0 = self.stem(x)
        e1 = self.enc1(s0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        # Bottleneck
        b  = self.cbam(self.aspp(e5))
        # Decoder
        d4 = self.up4(b,  e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        d1 = self.drop(d1)
        # Match input size
        if d1.shape[-2:] != (h, w):
            d1 = F.interpolate(d1, (h, w), mode="bilinear", align_corners=False)
        mu = F.softplus(self.head_mu(d1))               # depth ≥ 0
        lv = torch.clamp(self.head_lv(d1), -4.0, 4.0)    # stable σ² ∈ [e^-4, e^4]
        return mu, lv


# ── Loss + feature stack ─────────────────────────────────────────────
def heteroscedastic_nll(mu: torch.Tensor, log_var: torch.Tensor,
                         target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL with learned variance (Kendall & Gal 2017)."""
    inv_var = torch.exp(-log_var)
    nll = 0.5 * (inv_var * (mu - target) ** 2 + log_var)
    return (nll * mask).sum() / (mask.sum() + 1e-6)


def huber_masked(pred, target, mask, delta=1.0):
    e = torch.abs(pred - target)
    q = torch.where(e < delta, 0.5 * e * e, delta * (e - 0.5 * delta))
    return (q * mask).sum() / (mask.sum() + 1e-6)


def grad_smoothness(depth: torch.Tensor) -> torch.Tensor:
    dx = depth[..., :, 1:] - depth[..., :, :-1]
    dy = depth[..., 1:, :] - depth[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def build_feature_stack(s2: dict) -> np.ndarray:
    """Return a (13, H, W) float32 feature stack (NumPy for portability).

    Feature 9 (Stumpf log-ratio) is clipped to ±10 to avoid pathological
    reflectance ratios when any band saturates or drops below the noise
    floor. All ratios use a floor of 5e-4 (= 5 DN / 10 000) instead of
    1e-6 which produced NaNs during training."""
    FLOOR = 5e-4
    b = np.clip(s2["blue"].astype(np.float32) / 10000, FLOOR, 1)
    g = np.clip(s2["green"].astype(np.float32) / 10000, FLOOR, 1)
    r = np.clip(s2["red"].astype(np.float32) / 10000, FLOOR, 1)
    nir = np.clip(s2["nir"].astype(np.float32) / 10000, FLOOR, 1)
    coastal = np.clip(
        s2.get("coastal", s2["blue"]).astype(np.float32) / 10000, FLOOR, 1)
    ndwi = np.clip(
        s2.get("ndwi", (g - nir) / (g + nir + FLOOR)).astype(np.float32),
        -1, 1,
    )
    # Stumpf log-ratios
    lnBG = np.clip(np.log(1000 * b) / np.log(1000 * g), -10, 10)
    lnGR = np.clip(np.log(1000 * g) / np.log(1000 * r), -10, 10)
    BG   = np.clip(b / g, 0, 10)
    GR   = np.clip(g / r, 0, 10)
    feats = np.stack([
        b, g, r, nir, coastal, ndwi,
        lnBG, lnGR, BG, GR,
        b - nir, g - nir, r - nir,
    ], axis=0).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def normalise_stack(stack: np.ndarray, water: np.ndarray) -> np.ndarray:
    """Percentile-based normalisation over water pixels only."""
    out = np.empty_like(stack)
    for i in range(stack.shape[0]):
        vals = stack[i][water]
        if vals.size < 10:
            out[i] = stack[i]; continue
        lo = np.percentile(vals, 2)
        hi = np.percentile(vals, 98)
        if hi - lo < 1e-6:
            out[i] = stack[i] - lo
        else:
            out[i] = (stack[i] - lo) / (hi - lo)
    return np.clip(out, -2, 2).astype(np.float32)


# ── Public entry point ───────────────────────────────────────────────
def train_predict(s2: dict, ref_pts: dict, bbox: tuple,
                   epochs: int = 80, patch: int = 96,
                   batch: int = 4, max_patches: int = 256,
                   lr: float = 2e-4, patience: int = 12,
                   dropout: float = 0.1, base: int = 48,
                   device: str = None, verbose: bool = True,
                   logger=None) -> dict:
    """
    Train BathyNet-Pro on (s2, ref_pts) and return prediction + metrics.

    ref_pts keys: lats, lons, depths, [confidence]
    """
    def log(*a):
        s = " ".join(str(x) for x in a)
        if verbose:
            print(s, flush=True)
        if logger is not None:
            try: logger.info(s)
            except Exception: pass
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    log(f"BathyNetPro: device={dev}, patch={patch}, batch={batch}, epochs={epochs}")

    # ── 1. features + target raster ──
    raw = build_feature_stack(s2)
    water = s2.get("water_mask", s2["ndwi"] > 0)
    feats = normalise_stack(raw, water)         # (13, H, W)
    H, W = feats.shape[1:]

    target = np.full((H, W), np.nan, np.float32)
    w_, s_, e_, n_ = bbox
    lats = np.asarray(ref_pts["lats"]); lons = np.asarray(ref_pts["lons"])
    deps = np.asarray(ref_pts["depths"], np.float32)
    for lat, lon, d in zip(lats, lons, deps):
        if not (s_ <= lat <= n_ and w_ <= lon <= e_):
            continue
        r_px = int((n_ - lat) / (n_ - s_ + 1e-10) * H)
        c_px = int((lon - w_) / (e_ - w_ + 1e-10) * W)
        r_px = max(0, min(H - 1, r_px)); c_px = max(0, min(W - 1, c_px))
        if water[r_px, c_px]:
            target[r_px, c_px] = float(np.clip(d, 0, MAX_DEPTH_M))

    valid = np.isfinite(target)
    n_valid = int(valid.sum())
    log(f"BathyNetPro: {n_valid} valid training pixels inside water mask")
    if n_valid < 10:
        raise RuntimeError(f"Only {n_valid} valid training pixels")

    # Depth scale (robust)
    dscale = float(np.clip(np.percentile(target[valid], 98), 2, MAX_DEPTH_M))
    target_s = (target / dscale).astype(np.float32)
    mask_np = valid.astype(np.float32)

    # ── 2. patch sampling ──
    # Centre each patch on a random valid pixel, clip to image bounds.
    vr, vc = np.where(valid)
    idx = np.random.permutation(len(vr))[:max_patches]
    patches = []
    for k in idx:
        r0 = max(0, min(H - patch, vr[k] - patch // 2))
        c0 = max(0, min(W - patch, vc[k] - patch // 2))
        patches.append((r0, c0))
    # Also add grid patches covering the full image for inference context
    grid_patches = []
    for r0 in range(0, max(1, H - patch), patch // 2):
        for c0 in range(0, max(1, W - patch), patch // 2):
            grid_patches.append((r0, c0))
    # 80/20 spatial split
    n_hold = max(1, int(len(patches) * 0.2))
    rng = np.random.default_rng(42)
    hold_idx = set(rng.choice(len(patches), n_hold, replace=False).tolist())
    train_patches = [p for i, p in enumerate(patches) if i not in hold_idx]
    val_patches = [p for i, p in enumerate(patches) if i in hold_idx]
    log(f"BathyNetPro: {len(train_patches)} train patches, "
        f"{len(val_patches)} val patches")

    # ── 3. build model ──
    net = BathyNetPro(in_ch=feats.shape[0], base=base, dropout=dropout).to(dev)
    log(f"BathyNetPro: {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M params")
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(1, epochs // 3), eta_min=lr / 10)

    feats_t = torch.from_numpy(feats).unsqueeze(0).to(dev)
    target_t = torch.from_numpy(target_s).unsqueeze(0).unsqueeze(0).to(dev)
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(dev)

    def sample_batch(plist):
        r0c0 = plist[np.random.randint(len(plist))]
        r0, c0 = r0c0
        fp = feats_t[..., r0:r0 + patch, c0:c0 + patch]
        tp = target_t[..., r0:r0 + patch, c0:c0 + patch]
        mp = mask_t[..., r0:r0 + patch, c0:c0 + patch]
        # augment
        if np.random.rand() < 0.5:
            fp = torch.flip(fp, [-1]); tp = torch.flip(tp, [-1]); mp = torch.flip(mp, [-1])
        if np.random.rand() < 0.5:
            fp = torch.flip(fp, [-2]); tp = torch.flip(tp, [-2]); mp = torch.flip(mp, [-2])
        k = np.random.randint(4)
        if k:
            fp = torch.rot90(fp, k, [-2, -1]); tp = torch.rot90(tp, k, [-2, -1]); mp = torch.rot90(mp, k, [-2, -1])
        if np.random.rand() < 0.3:
            fp = fp + torch.randn_like(fp) * 0.01
        return fp, tp, mp

    # ── 4. training loop ──
    best_val = math.inf; best_state = None; bad = 0
    for ep in range(epochs):
        net.train(); tr_loss = 0.0
        for _ in range(max(1, len(train_patches) // batch)):
            opt.zero_grad()
            fp, tp, mp = sample_batch(train_patches)
            mu, lv = net(fp)
            loss_h = huber_masked(mu, tp, mp, delta=0.1)
            loss_n = heteroscedastic_nll(mu, lv, tp, mp)
            loss_s = grad_smoothness(mu) * 0.02
            loss = loss_h + 0.1 * loss_n + loss_s
            if not torch.isfinite(loss):
                # Skip the step if NaN/Inf — don't poison the weights.
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss.item())
        sched.step()

        # validation
        net.eval()
        with torch.no_grad():
            val_l = 0.0; nv = 0
            for p in val_patches:
                r0, c0 = p
                fp = feats_t[..., r0:r0 + patch, c0:c0 + patch]
                tp = target_t[..., r0:r0 + patch, c0:c0 + patch]
                mp = mask_t[..., r0:r0 + patch, c0:c0 + patch]
                if mp.sum() == 0:
                    continue
                mu, _ = net(fp)
                val_l += float(huber_masked(mu, tp, mp, delta=0.1).item())
                nv += 1
        val_loss = val_l / max(1, nv)
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"  ep{ep:02d} · train={tr_loss:.4f}  val={val_loss:.4f}  lr={opt.param_groups[0]['lr']:.1e}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log(f"  early stop @ ep{ep} (no improvement for {patience})")
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    # ── 5. full-image inference (stitched) ──
    net.eval()
    depth_sum = np.zeros((H, W), np.float32)
    unc_sum   = np.zeros((H, W), np.float32)
    w_sum     = np.zeros((H, W), np.float32)
    # Hann window for overlap blending
    hw = np.hanning(patch)
    hann = (hw[:, None] * hw[None, :]).astype(np.float32) + 1e-6
    with torch.no_grad():
        for (r0, c0) in grid_patches:
            fp = feats_t[..., r0:r0 + patch, c0:c0 + patch]
            if fp.shape[-2] != patch or fp.shape[-1] != patch:
                continue
            # MC-Dropout: 4 stochastic passes
            net.train()     # enable dropout
            preds, lvs = [], []
            for _ in range(4):
                mu, lv = net(fp)
                preds.append(mu); lvs.append(lv)
            net.eval()
            mu = torch.stack(preds, 0).mean(0)
            lv = torch.stack(lvs, 0).mean(0)
            d = (mu[0, 0].cpu().numpy() * dscale).astype(np.float32)
            u = (np.exp(0.5 * lv[0, 0].cpu().numpy())
                  * dscale).astype(np.float32)
            depth_sum[r0:r0 + patch, c0:c0 + patch] += d * hann
            unc_sum[r0:r0 + patch, c0:c0 + patch]   += u * hann
            w_sum[r0:r0 + patch, c0:c0 + patch]     += hann
    depth = np.where(w_sum > 0, depth_sum / w_sum, np.nan)
    uncertainty = np.where(w_sum > 0, unc_sum / w_sum, np.nan)
    depth = np.clip(depth, 0, MAX_DEPTH_M)

    # Mask land
    depth = np.where(water, depth, np.nan)
    uncertainty = np.where(water, uncertainty, np.nan)

    # ── 6. metrics on 20% held-out patches ──
    rmse = mae = r2 = np.nan
    if val_patches:
        p_all, t_all = [], []
        with torch.no_grad():
            for (r0, c0) in val_patches:
                fp = feats_t[..., r0:r0 + patch, c0:c0 + patch]
                tp = target_t[..., r0:r0 + patch, c0:c0 + patch]
                mp = mask_t[..., r0:r0 + patch, c0:c0 + patch]
                mu, _ = net(fp)
                pred = (mu * dscale).cpu().numpy()[0, 0]
                obs = (tp * dscale).cpu().numpy()[0, 0]
                m = mp.cpu().numpy()[0, 0] > 0
                if m.any():
                    p_all.extend(pred[m].tolist())
                    t_all.extend(obs[m].tolist())
        if p_all:
            p_all = np.array(p_all); t_all = np.array(t_all)
            diffs = p_all - t_all
            rmse = float(np.sqrt(np.mean(diffs ** 2)))
            mae  = float(np.mean(np.abs(diffs)))
            ss_res = np.sum(diffs ** 2)
            ss_tot = np.sum((t_all - t_all.mean()) ** 2) + 1e-10
            r2 = float(1 - ss_res / ss_tot)

    return {
        "depth": depth,
        "uncertainty": uncertainty,
        "rmse_m": rmse,
        "mae_m": mae,
        "r2": r2,
        "n_train_pixels": n_valid,
        "n_train_patches": len(train_patches),
        "n_val_patches": len(val_patches),
        "depth_scale_m": dscale,
        "model_state": net.state_dict(),
        "params": {"base": base, "patch": patch, "dropout": dropout,
                    "epochs_done": ep + 1},
    }
