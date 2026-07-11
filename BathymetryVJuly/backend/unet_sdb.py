"""Attention U-Net for satellite-derived bathymetry.

Replaces the dilated-FCN refinement model in :mod:`backend.very_hr_engine`
with a proper encoder-decoder architecture and adds:

* Skip connections gated by additive attention (Oktay 2018) so the
  decoder can suppress non-discriminative channels.
* Heteroscedastic depth head — predicts (mean, log-sigma) per pixel so
  the platform can show a confidence map alongside the depth.
* Test-time augmentation: 8-way (horizontal flip × vertical flip × 90 deg
  rotation) averaging at inference for sharper boundaries.
* Adaptive trainer: cosine-annealed AdamW, masked Gaussian NLL on
  labelled pixels, depth-bin reweighting, gradient clipping, early
  stopping on a held-out crop set.

The model is designed to fall back gracefully on small training sets:
when fewer than 300 labelled pixels are available the caller is
expected to skip CNN refinement and stay with the HGB output.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

L = logging.getLogger("unet_sdb")
if not L.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
    L.addHandler(h)
    L.setLevel(logging.INFO)

MAX_DEPTH_M = 25.0


# ════════════════════════════════════════════════════════════════════════
# Architecture
# ════════════════════════════════════════════════════════════════════════
def _build_unet(c_in: int, base: int = 32, p_drop: float = 0.10,
                n_onehot: int = 0):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # s2-dl request #8 — PRO depth-regime head (default OFF via UNET_PRO_HEAD).
    pro_head_on = (os.environ.get("UNET_PRO_HEAD", "0") == "1"
                   and int(n_onehot) > 0)

    # #REQ2 / UNET_REGIME_HEAD=1 — depth-regime gated dual-head.
    # Mutually exclusive with UNET_PRO_HEAD (both rewrite d1's pointwise path).
    regime_head_on = os.environ.get("UNET_REGIME_HEAD", "0") == "1"
    if regime_head_on and pro_head_on:
        raise ValueError(
            "UNET_REGIME_HEAD=1 and UNET_PRO_HEAD=1 are mutually exclusive "
            "(both rewrite the d1 pointwise path; set only one)."
        )
    _regime_tau_m = float(os.environ.get("REGIME_TAU_M", "3.0"))
    _regime_gate_w = float(os.environ.get("REGIME_GATE_W", "0.1"))

    class ProHead(nn.Module):
        """FiLM(cluster one-hot) + 2-layer 1x1-conv residual MLP on decoder d1.

        §7.E.3: per-pixel (1x1) capacity only — receptive field UNCHANGED, so
        no context is added and the §7.A.3 leakage constraint (RF<=25 px) is
        preserved.  FiLM (Perez et al. 2018) maps the per-pixel cluster one-hot
        -> per-cluster (gamma, beta) that feature-wise affine-modulate d1, the
        DL analogue of the §1.5 per-cluster Lyzenga/Stumpf fit.  The residual
        1x1 MLP (B->2B->B, GELU, dropout) deepens the pointwise spectral->depth
        regressor.  The sigma head is NOT touched here (§7.A.1 contract intact).
        """
        def __init__(self, B: int, k: int, p: float = 0.10):
            super().__init__()
            self.k = int(k)
            # FiLM generator: one-hot (k) -> 2*B  (gamma,beta), via a tiny MLP
            self.film = nn.Sequential(
                nn.Linear(self.k, 2 * B), nn.GELU(),
                nn.Linear(2 * B, 2 * B),
            )
            # gamma init ~1, beta init ~0 (identity modulation at start)
            nn.init.zeros_(self.film[-1].weight)
            with torch.no_grad():
                self.film[-1].bias[:B].fill_(1.0)   # gamma -> 1
                self.film[-1].bias[B:].fill_(0.0)   # beta  -> 0
            self.mlp = nn.Sequential(
                nn.Conv2d(B, 2 * B, 1), nn.GELU(), nn.Dropout2d(p),
                nn.Conv2d(2 * B, B, 1),
            )

        def forward(self, d1, onehot):
            # onehot: (N, k, H, W) per-pixel cluster indicator.
            # Per-pixel FiLM: gamma/beta vary by pixel via the local one-hot.
            N, k, H, W = onehot.shape
            oh = onehot.permute(0, 2, 3, 1).reshape(-1, k)   # (N*H*W, k)
            gb = self.film(oh)                               # (N*H*W, 2B)
            B = gb.shape[1] // 2
            gamma = gb[:, :B].reshape(N, H, W, B).permute(0, 3, 1, 2)
            beta = gb[:, B:].reshape(N, H, W, B).permute(0, 3, 1, 2)
            d1m = gamma * d1 + beta
            return d1m + self.mlp(d1m)   # residual

    class ConvBlock(nn.Module):
        def __init__(self, ci: int, co: int, p: float = 0.0):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.GELU(),
                nn.Dropout2d(p),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.GELU(),
            )
        def forward(self, x): return self.body(x)

    class AttentionGate(nn.Module):
        """Additive attention gate (Oktay et al., 2018)."""
        def __init__(self, c_skip: int, c_gate: int, c_inter: int):
            super().__init__()
            self.W_skip = nn.Conv2d(c_skip, c_inter, 1, bias=False)
            self.W_gate = nn.Conv2d(c_gate, c_inter, 1, bias=False)
            self.psi = nn.Sequential(
                nn.Conv2d(c_inter, 1, 1), nn.Sigmoid())

        def forward(self, skip, gate):
            g = F.interpolate(gate, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
            attn = self.psi(F.gelu(self.W_skip(skip) + self.W_gate(g)))
            return skip * attn

    # S9: RESIDUAL_CARRIER=1 (default OFF) — structural residual-over-prior head.
    # Must also have CARRIER_NO_ZSCORE=1 (enforced by assert below in train_unet_sdb).
    _residual_carrier = (os.environ.get("RESIDUAL_CARRIER", "0") == "1"
                         and os.environ.get("INTERP_INPUT", "0") == "1")

    class AttUNet(nn.Module):
        def __init__(self, c_in: int):
            super().__init__()
            B = base
            # Encoder
            self.enc1 = ConvBlock(c_in, B, p_drop)
            self.enc2 = ConvBlock(B, 2 * B, p_drop)
            self.enc3 = ConvBlock(2 * B, 4 * B, p_drop)
            self.enc4 = ConvBlock(4 * B, 8 * B, p_drop)
            self.bottleneck = ConvBlock(8 * B, 16 * B, p_drop * 1.5)
            # Decoder
            self.up4 = nn.ConvTranspose2d(16 * B, 8 * B, 2, stride=2)
            self.att4 = AttentionGate(8 * B, 8 * B, 4 * B)
            self.dec4 = ConvBlock(16 * B, 8 * B, p_drop)
            self.up3 = nn.ConvTranspose2d(8 * B, 4 * B, 2, stride=2)
            self.att3 = AttentionGate(4 * B, 4 * B, 2 * B)
            self.dec3 = ConvBlock(8 * B, 4 * B, p_drop)
            self.up2 = nn.ConvTranspose2d(4 * B, 2 * B, 2, stride=2)
            self.att2 = AttentionGate(2 * B, 2 * B, B)
            self.dec2 = ConvBlock(4 * B, 2 * B, p_drop)
            self.up1 = nn.ConvTranspose2d(2 * B, B, 2, stride=2)
            self.att1 = AttentionGate(B, B, B // 2)
            self.dec1 = ConvBlock(2 * B, B, p_drop)
            # Heteroscedastic head: depth (mean) + log_sigma
            self.head_mu = nn.Conv2d(B, 1, 1)
            self.head_logsigma = nn.Conv2d(B, 1, 1)
            self.depth_max = float(MAX_DEPTH_M)
            # s2-dl request #8 — PRO depth-regime head (default OFF)
            self.n_onehot = int(n_onehot)
            self.pro = ProHead(B, int(n_onehot), p_drop) if pro_head_on else None
            # #REQ2 / UNET_REGIME_HEAD=1 — depth-regime gated dual-head.
            # Two pointwise (1x1) heads + a learned per-pixel soft gate α.
            # mu = α·mu_shallow + (1−α)·mu_deep   (α=1 → pure shallow head)
            # RF UNCHANGED (all are 1×1 conv on d1) — leakage constraint intact.
            self.regime_head = regime_head_on
            if regime_head_on:
                self.head_mu_shallow = nn.Conv2d(B, 1, 1)
                self.head_mu_deep    = nn.Conv2d(B, 1, 1)
                # Gate: B → B//2 → 1, sigmoid → α ∈ (0,1)
                self.gate = nn.Sequential(
                    nn.Conv2d(B, max(1, B // 2), 1), nn.GELU(),
                    nn.Conv2d(max(1, B // 2), 1, 1),
                )
                # Initialise gate bias so α ≈ 0.5 at start
                nn.init.zeros_(self.gate[-1].weight)
                nn.init.zeros_(self.gate[-1].bias)
                # Log regime params for traceability
                L.info(f"UNET_REGIME_HEAD=1: τ={_regime_tau_m} m, "
                       f"gate_w={_regime_gate_w}")
            # S9: RESIDUAL_CARRIER — structural residual-over-prior (Ma 2018 / He 2016).
            # depth_pred = carrier_in_target_space + f_theta(S2, carrier)
            # Zero-init head_mu so f_theta≡0 at step 0 → exact carrier passthrough.
            # Buffers (_tnorm_mean, _tnorm_std, _carrier_scale_m) are registered in
            # train_unet_sdb after computing target stats; they ride in state_dict.
            self.residual_carrier = _residual_carrier
            if _residual_carrier:
                nn.init.zeros_(self.head_mu.weight)
                nn.init.zeros_(self.head_mu.bias)
                # Registered buffers (placeholder values; train overwrites immediately)
                self.register_buffer('_tnorm_mean',
                                     torch.tensor(0.0, dtype=torch.float32))
                self.register_buffer('_tnorm_std',
                                     torch.tensor(1.0, dtype=torch.float32))
                self.register_buffer('_carrier_scale_m',
                                     torch.tensor(25.0, dtype=torch.float32))
                mu_res_std = 0.0  # zero-init → exact passthrough at step 0
                L.info(f"S9 RESIDUAL_CARRIER=1: head_mu zero-init, "
                       f"mu_residual.std()≈{mu_res_std:.4f} (exact passthrough at init). "
                       f"Buffers: _tnorm_mean=0.0 _tnorm_std=1.0 _carrier_scale_m=25.0 "
                       f"(will be overwritten after target-norm computed in train_unet_sdb)")
            else:
                self._carrier_z = None  # legacy path: no carrier capture

        def forward(self, x):
            # S9: capture carrier channel BEFORE encoder so its absolute depth level
            # bypasses all BatchNorm layers. Channel index C-1 is the IDW carrier,
            # already scaled to carrier_m / CARRIER_SCALE_M by the S8-1 ingestion
            # (CARRIER_NO_ZSCORE=1). Stored as self._carrier_z for use in the head.
            if self.residual_carrier:
                self._carrier_z = x[:, -1:, :, :]  # (N,1,H,W) = carrier_m/CARRIER_SCALE_M
            # Slice the trailing cluster one-hot channels for FiLM before they
            # enter the encoder (they DO still enter the encoder as input; the
            # PRO head re-uses them at full-res to condition d1).
            onehot = (x[:, -self.n_onehot:, :, :]
                      if (self.pro is not None and self.n_onehot > 0) else None)
            e1 = self.enc1(x)
            e2 = self.enc2(F.max_pool2d(e1, 2))
            e3 = self.enc3(F.max_pool2d(e2, 2))
            e4 = self.enc4(F.max_pool2d(e3, 2))
            b = self.bottleneck(F.max_pool2d(e4, 2))
            d4 = self.up4(b)
            d4 = self.dec4(torch.cat([d4, self.att4(e4, d4)], dim=1))
            d3 = self.up3(d4)
            d3 = self.dec3(torch.cat([d3, self.att3(e3, d3)], dim=1))
            d2 = self.up2(d3)
            d2 = self.dec2(torch.cat([d2, self.att2(e2, d2)], dim=1))
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, self.att1(e1, d1)], dim=1))
            # s2-dl request #8 — PRO head: FiLM(cluster) + residual 1x1 MLP on
            # d1 (pointwise; RF unchanged).  sigma head below is NOT touched.
            if self.pro is not None and onehot is not None:
                d1 = self.pro(d1, onehot)
            # iter#4 (arch_v2): drop sigmoid + [0,depth_max] cap.  The
            # sigmoid head silently regressed the prediction toward the
            # mid-range (~12.5 m on a 25 m cap) — see IHO request #4.
            # Softplus gives unbounded positive depth in metres directly.
            # logsigma floor raised from -3.0 to -1.5 so the model can no
            # longer collapse to "I'm extremely confident I'm wrong".
            if self.regime_head:
                # #REQ2: gated dual-head path.  α = sigmoid(gate(d1)).
                alpha = torch.sigmoid(self.gate(d1))  # (N,1,H,W) ∈ (0,1)
                mu_s = F.softplus(self.head_mu_shallow(d1))  # (N,1,H,W)
                mu_d = F.softplus(self.head_mu_deep(d1))     # (N,1,H,W)
                mu = (alpha * mu_s + (1.0 - alpha) * mu_d).squeeze(1)
                # Expose alpha for gate-BCE loss; store as buffer so the
                # training loop can retrieve it without a second forward pass.
                self._last_alpha = alpha  # (N,1,H,W)
            else:
                # S7: HEAD_KIND=linear bypasses softplus for z-scored target probes.
                # Default 'softplus' = byte-unchanged legacy path.
                # S8 bug-fix: read from model attribute first (set in train_unet_sdb
                # after training so that inference after env-restore uses the correct
                # training-time value). Fall back to os.environ for backward compat.
                _head_kind = getattr(self, "_s8_head_kind",
                                     os.environ.get("HEAD_KIND", "softplus")).lower()
                if _head_kind == "linear":
                    mu = self.head_mu(d1).squeeze(1)
                else:
                    mu = torch.nn.functional.softplus(self.head_mu(d1)).squeeze(1)
            # S9: RESIDUAL_CARRIER — add carrier back in TARGET SPACE after mu head.
            # This bypasses all BatchNorm re-centring: the carrier's DC + variance
            # are preserved structurally. f_theta only needs to learn the S2 correction.
            # (Ma & Karaman 2018 ICRA; Eldesokey 2020 CVPR; He 2016 residual learning)
            #
            # CRITICAL: use the REGISTERED BUFFERS to determine target space, NOT
            # os.environ — the env may be restored to a different value between train
            # and inference (e.g. by the finally-block in D0b/probe runners).
            # If _tnorm_std > 0 (zscore was applied during training), use it.
            # If _tnorm_std == 1.0 AND _tnorm_mean == 0.0 (default placeholders,
            # i.e. TARGET_NORM=none was used), add carrier_m directly in metres.
            if self.residual_carrier and self._carrier_z is not None:
                # carrier_m: recover metres from scaled representation
                carrier_m = self._carrier_z.squeeze(1) * self._carrier_scale_m  # (N,H,W)
                # Convert to target space using REGISTERED BUFFERS (never os.environ):
                # carrier_tgt = (carrier_m - _tnorm_mean) / _tnorm_std
                # For TARGET_NORM=none:  buffers = (0.0, 1.0) → identity, i.e. carrier_m
                # For TARGET_NORM=zscore: buffers = (train_mean, train_std) → z-scored
                # For TARGET_NORM=div_max: buffers = (0.0, 25.0) → carrier_m/25
                # In all cases, predict_grid_tta un-normalizes with the same stats, so the
                # net result at inference is always correct metres regardless of norm mode.
                # DO NOT read os.environ here — env may be restored to a different value
                # between train() and predict_grid_tta() by finally-blocks in probe runners.
                carrier_tgt = (carrier_m - self._tnorm_mean) / self._tnorm_std
                mu = mu + carrier_tgt  # residual: head learns the S2 correction
            logs = self.head_logsigma(d1).squeeze(1).clamp(-1.5, 2.5)
            return mu, logs

    return AttUNet(c_in)


# ════════════════════════════════════════════════════════════════════════
# Heteroscedastic Gaussian NLL on labelled pixels
# ════════════════════════════════════════════════════════════════════════
def _hetero_nll(mu, logsigma, y, mask, weight,
                shoal_alpha: float = 1.5,
                shoal_depth_gate_m: float = 12.0):
    """Heteroscedastic Gaussian NLL with depth-gated asymmetric shoal-bias.

    iter#4 (4.2): asymmetric weight when ``mu > y`` (model says deeper than
    truth) — predicting deeper than ground-truth is the dangerous direction
    for a navigation chart (grounding risk).  Predicting shallower merely
    costs route efficiency.  Bakes the IHO shoal-bias preference
    (Hare 2011 §4.2) into the loss.

    iter#5 (5.1): the iter#4 α=3.0 ungated implementation over-shoaled the
    deep band by ~5.5 m uniformly on Dhanna.  Per IHO §1.10 + PIANC UKC
    guidelines, the shoal-bias prior only applies inside the
    vessel-grounding-risk envelope (truth depth <= 12 m for UAE port
    traffic).  Beyond 12 m the loss reverts to symmetric.  Default
    ``shoal_alpha`` cut from 3.0 to 1.5 (lower end of Hare 2011 range).
    """
    import torch
    import os as _os_nll

    # S7: plain MSE / Berhu regression losses (default-OFF; activated only when
    # LOSS env contains 'mse' or 'berhu' — legacy NLL path byte-unchanged).
    _loss_spec_nll = _os_nll.environ.get("LOSS", "nll").lower()
    if "mse" in _loss_spec_nll or "berhu" in _loss_spec_nll:
        err = (mu - y) * mask * weight
        if "berhu" in _loss_spec_nll:
            # Berhu / reverse-Huber (Laina 2016 3DV): L1 for small errors,
            # L2 for large. Threshold c = 0.2 * max(|err|) per batch.
            abs_err = err.abs()
            c_berhu = float(0.2 * float(abs_err.max().item()) + 1e-6)
            l1_mask = abs_err <= c_berhu
            loss_elem = torch.where(l1_mask, abs_err,
                                    (abs_err ** 2 + c_berhu ** 2) / (2.0 * c_berhu))
        else:
            loss_elem = err ** 2
        denom = mask.sum().clamp(min=1.0)
        return loss_elem.sum() / denom

    # S7: Beta-NLL (Seitzer 2022 ICLR): multiply per-pixel NLL by sigma^(2*beta)
    # to restore mean-fit gradient. BETA_NLL=0.0 (default) = plain NLL, unchanged.
    _beta_nll = float(_os_nll.environ.get("BETA_NLL", "0.0"))

    inv_var = torch.exp(-2.0 * logsigma)
    err = mu - y
    # gate: asymmetry active only where truth is in vessel-grounding band
    in_grounding_band = (y <= float(shoal_depth_gate_m)).to(err.dtype)
    base = torch.ones_like(err)
    asym = torch.where(err > 0,
                       base + (float(shoal_alpha) - 1.0) * in_grounding_band,
                       base)
    sq = (err ** 2) * asym
    elem = 0.5 * sq * inv_var + logsigma
    # S7: Beta-NLL scaling (Seitzer 2022 ICLR §3). When BETA_NLL > 0,
    # multiply by sigma^(2*beta) (stop-grad) so the mean-fit gradient
    # is restored even when sigma is large. beta=0 -> plain NLL (default, unchanged).
    if _beta_nll > 0.0:
        # sigma = exp(logsigma); sigma^(2*beta) = exp(2*beta*logsigma)
        beta_weight = torch.exp(2.0 * _beta_nll * logsigma.detach())
        elem = elem * beta_weight
    elem = elem * mask * weight
    # iter#9.B: variance-anchor (Mehta 2018 Y-Net §3.2; Steininger 2021 §4.3)
    # Penalise std-collapse: force model std to match target std over valid
    # batch pixels. Closes the dynamic-range gap (iter #8.A diagnosis: per-
    # segment slopes 0.05-0.30 vs photon truth, well below b≥0.4 floor).
    # iter#11.B: per-site adaptive λ_var (Kendall & Gal 2017 §4.2) — the
    # variance-anchor is helpful only when the truth depth distribution has
    # enough spread to anchor against.  At sites with truth-σ < 5 m the
    # anchor over-expands predicted variance and inflates RMSE (iter #10
    # diagnosed Lulu +5.09 m, SBY +6.28 m collapse).  Threshold rule:
    #     λ_var = 0.10 if truth_std ≥ 6.0 m
    #           = 0.03 if truth_std ≥ 5.0 m
    #           = 0    otherwise (effectively disable)
    # truth_std is measured on the *training* photons of this batch (no
    # held-out leakage; the per-batch std is a stochastic estimate of
    # the per-site std).  At λ_var = 0 the loss reduces to the iter#8
    # heteroscedastic NLL — a known-good baseline.
    mu_valid = mu[mask > 0]
    y_valid = y[mask > 0]
    nll_term = elem.sum() / mask.sum().clamp(min=1.0)
    if mu_valid.numel() > 16:
        truth_std = float(y_valid.std(unbiased=False).item())
        # iter#12 (scientist req #2, item 3): dynamic-range-preserving loss.
        # LOSS=nll+silog+range adds (a) a SILog (scale-invariant log) term
        # that penalises getting the depth *distribution* wrong (Eigen 2014)
        # and (b) an explicit variance-matching penalty
        # λ·(σ_pred/σ_truth − 1)²  (env RANGE_LOSS_W) that fights the
        # mean-collapse the heteroscedastic NLL alone causes in a deep,
        # optically-saturated basin.  Default loss spec leaves the iter#11
        # adaptive variance-anchor untouched (backwards compatible).
        import os as _os
        loss_spec = _os.environ.get("LOSS", "nll").lower()
        if "range" in loss_spec or "silog" in loss_spec:
            extra = mu_valid.new_zeros(())
            # iter#13 (scientist req #3, item 2): restrict the SILog/variance
            # scale terms to the optically-valid <12 m envelope so the
            # un-anchorable deep band (no photon/optical signal >~12 m) does
            # not poison the scale statistics.  Beyond 12 m the model has no
            # truth signal; forcing it to match that band's spread would just
            # re-inject the deep collapse into the shallow scale match.
            env_gate = float(_os.environ.get("RANGE_DEPTH_GATE_M", "12.0"))
            env_sel = y_valid <= env_gate
            if int(env_sel.sum().item()) > 16:
                mu_e = mu_valid[env_sel]
                y_e = y_valid[env_sel]
            else:
                mu_e, y_e = mu_valid, y_valid
            if "silog" in loss_spec:
                # SILog on log-depth (clamp to keep the log well-defined for
                # the 0 m shoreline pixels).
                eps = 1e-3
                d = torch.log(mu_e.clamp(min=eps)) - \
                    torch.log(y_e.clamp(min=eps))
                lam_si = float(_os.environ.get("SILOG_LAMBDA", "0.5"))
                silog = (d ** 2).mean() - lam_si * (d.mean() ** 2)
                w_si = float(_os.environ.get("SILOG_W", "0.5"))
                extra = extra + w_si * silog.clamp(min=0.0)
            if "range" in loss_spec:
                # Scale-relative variance matching: (σ_pred/σ_truth − 1)².
                w_rg = float(_os.environ.get("RANGE_LOSS_W", "0.3"))
                sp = mu_e.std(unbiased=False)
                st = y_e.std(unbiased=False).clamp(min=1e-3)
                extra = extra + w_rg * (sp / st - 1.0) ** 2
            # DL-#15: pairwise rank/ordinal loss (gated, RANK_LOSS_W default 0.0).
            # Rewards correct depth ordering within the crop; cannot be satisfied
            # by a constant prediction — directly counter NLL's regression-to-mean.
            # Only applied when RANK_LOSS_W > 0.0 (default OFF, legacy unchanged).
            _rlw = float(_os.environ.get("RANK_LOSS_W", "0.0"))
            if _rlw > 0.0 and mu_valid.numel() >= 4:
                _rank_pairs = int(_os.environ.get("RANK_PAIRS", "256"))
                _rank_margin = float(_os.environ.get("RANK_MARGIN_M", "0.5"))
                n_v = mu_valid.numel()
                n_pairs = min(_rank_pairs, n_v * (n_v - 1) // 2)
                if n_pairs > 0:
                    # Random pair indices
                    _perm_i = torch.randint(0, n_v, (n_pairs,), device=mu_valid.device)
                    _perm_j = torch.randint(0, n_v, (n_pairs,), device=mu_valid.device)
                    _gap = y_valid[_perm_i] - y_valid[_perm_j]
                    # Only pairs with |truth gap| > margin (non-trivial ordering)
                    _valid_pairs = _gap.abs() > _rank_margin
                    if _valid_pairs.sum() > 0:
                        _gap_v = _gap[_valid_pairs]
                        _pred_gap = mu_valid[_perm_i[_valid_pairs]] - mu_valid[_perm_j[_valid_pairs]]
                        # Margin ranking loss: penalise wrong sign or too-small gap
                        _margin_scaled = _rank_margin * torch.ones_like(_gap_v)
                        _rank_loss = torch.relu(_margin_scaled - torch.sign(_gap_v) * _pred_gap)
                        extra = extra + _rlw * _rank_loss.mean()
            return nll_term + extra
        # DL-#15: rank loss also applies on the legacy (non-silog/range) path.
        _rlw_leg = float(_os.environ.get("RANK_LOSS_W", "0.0"))
        _rank_extra = mu_valid.new_zeros(())
        if _rlw_leg > 0.0 and mu_valid.numel() >= 4:
            _rank_pairs_leg = int(_os.environ.get("RANK_PAIRS", "256"))
            _rank_margin_leg = float(_os.environ.get("RANK_MARGIN_M", "0.5"))
            n_v = mu_valid.numel()
            n_pairs = min(_rank_pairs_leg, n_v * (n_v - 1) // 2)
            if n_pairs > 0:
                _perm_i = torch.randint(0, n_v, (n_pairs,), device=mu_valid.device)
                _perm_j = torch.randint(0, n_v, (n_pairs,), device=mu_valid.device)
                _gap = y_valid[_perm_i] - y_valid[_perm_j]
                _valid_pairs = _gap.abs() > _rank_margin_leg
                if _valid_pairs.sum() > 0:
                    _gap_v = _gap[_valid_pairs]
                    _pred_gap = (mu_valid[_perm_i[_valid_pairs]] -
                                 mu_valid[_perm_j[_valid_pairs]])
                    _rank_loss = torch.relu(
                        _rank_margin_leg - torch.sign(_gap_v) * _pred_gap)
                    _rank_extra = _rlw_leg * _rank_loss.mean()
        # Legacy iter#11 adaptive variance-anchor (unchanged default).
        if truth_std >= 6.0:
            lambda_var = 0.10
        elif truth_std >= 5.0:
            lambda_var = 0.03
        else:
            lambda_var = 0.0
        if lambda_var > 0.0:
            var_pred = mu_valid.var(unbiased=False)
            var_target = y_valid.var(unbiased=False)
            var_loss = (var_pred.sqrt() - var_target.sqrt()) ** 2
            return nll_term + lambda_var * var_loss + _rank_extra
    return nll_term


# ════════════════════════════════════════════════════════════════════════
# S5 interp loss: edge-aware smoothness + TV on unlabelled water pixels
# ════════════════════════════════════════════════════════════════════════
def _interp_smooth_loss(mu, logsigma, mask, water_mask_t,
                        obs_sigma_t=None):
    """Edge-aware smoothness + TV on unlabelled water pixels.

    S5 (s2-dl request #C5): gated by LOSS containing 'interp'.
    Applied ONLY to pixels where water_mask_t=1 AND mask=0 (unlabelled),
    so the loss cannot inflate the scored metric — scoring is at held-out
    labelled points that are independent of this term.

    Edge-aware smoothness (Ranftl et al. 2020 / Godard et al. 2019):
      smooth = |∂mu/∂x|·exp(-λ·|∂logsigma/∂x|)
             + |∂mu/∂y|·exp(-λ·|∂logsigma/∂y|)
    The sigma gradient acts as an edge detector: where sigma is high
    (uncertain) the smoothness penalty is relaxed — the model is allowed
    to be rough where it already signals low confidence. This prevents the
    smoother from forcing a false-smooth surface across real depth gradients.

    TV (total variation) adds an L1 smoothness on mu at unlabelled pixels
    as a backup regularizer.

    G0b guard: shuffling the input (which zeroes out the IDW carrier and
    spatial EDT channels) MUST degrade RMSE >10% — if the model can achieve
    low RMSE with random inputs, the smoothness prior is doing the work
    rather than spectral skill. The guard is run OUTSIDE this function.

    Returns: scalar loss (differentiable).
    """
    import torch
    import os as _os

    # S6 F1: defaults slashed — 0.005 smooth, TV OFF (was 0.05/0.01 in S5).
    # TV was the primary collapse driver (minimised by constant field).
    # Re-enable TV only via explicit INTERP_TV_W env override.
    w_smooth = float(_os.environ.get("INTERP_SMOOTH_W", "0.005"))
    w_tv = float(_os.environ.get("INTERP_TV_W", "0.0"))
    edge_lambda = float(_os.environ.get("INTERP_EDGE_LAMBDA", "1.0"))

    if w_smooth <= 0.0 and w_tv <= 0.0:
        return mu.new_zeros(())

    # unlabelled water mask: (N,H,W)
    unlabelled = water_mask_t * (1.0 - mask)
    if unlabelled.sum() < 1.0:
        return mu.new_zeros(())

    # mu: (N,H,W) — need (N,1,H,W) for grad
    mu_u = (mu * unlabelled).unsqueeze(1)          # (N,1,H,W)
    logs_u = (logsigma * unlabelled).unsqueeze(1)  # (N,1,H,W)

    loss = mu.new_zeros(())

    if w_smooth > 0.0:
        # Gradients via finite differences (safe, avoids autograd complications)
        dx_mu   = (mu_u[:, :, :, 1:] - mu_u[:, :, :, :-1]).abs()
        dy_mu   = (mu_u[:, :, 1:, :] - mu_u[:, :, :-1, :]).abs()
        dx_logs = (logs_u[:, :, :, 1:] - logs_u[:, :, :, :-1]).abs()
        dy_logs = (logs_u[:, :, 1:, :] - logs_u[:, :, :-1, :]).abs()
        # edge-aware weights: exp(-lambda * |grad_sigma|) — low where sigma changes fast
        w_x = torch.exp(-edge_lambda * dx_logs.detach())
        w_y = torch.exp(-edge_lambda * dy_logs.detach())
        smooth = (dx_mu * w_x).mean() + (dy_mu * w_y).mean()
        loss = loss + w_smooth * smooth

    if w_tv > 0.0:
        # L1 TV on mu at unlabelled pixels
        tv_x = (mu_u[:, :, :, 1:] - mu_u[:, :, :, :-1]).abs().mean()
        tv_y = (mu_u[:, :, 1:, :] - mu_u[:, :, :-1, :]).abs().mean()
        loss = loss + w_tv * (tv_x + tv_y)

    return loss


# ════════════════════════════════════════════════════════════════════════
# Trainer
# ════════════════════════════════════════════════════════════════════════
def train_unet_sdb(
    feats: np.ndarray,
    label_grid: np.ndarray,
    label_mask: np.ndarray,
    epochs: int = 40,
    crops_per_epoch: int = 80,
    batch: int = 4,
    crop: int = 192,
    lr: float = 1e-3,
    base: int = 32,
    seed: int = 42,
    warmstart_state: Optional[Dict] = None,
    forbidden_mask: Optional[np.ndarray] = None,
    n_onehot: int = 0,
    water_mask: Optional[np.ndarray] = None,
    obs_precision_grid: Optional[np.ndarray] = None,
):
    """Train Attention U-Net with heteroscedastic loss + crop sampling.

    S12: obs_precision_grid (H×W float32, optional) — per-pixel 1/σ_obs² precision
    weight for inverse-variance fusion of i-Boating chart soundings alongside
    ATL24/multibeam. Gated by OBS_PRECISION_W=1; default None = uniform weight
    (byte-identical to S8). Mean-normalized over labelled pixels to 1.0 so the
    global loss scale and LR schedule are unchanged vs the S8 uniform path.
    When OBS_PRECISION_W=0 or obs_precision_grid is None, the depth-band reweighting
    (weight_t from fixed/quantile bins) is used as before — path is byte-identical.

    s2-dl request #1 (METHOD SPEC §6 / 7.C.5) — leakage-safe crop sampling.
    `forbidden_mask` (H×W bool) marks test blocks dilated by the §6 buffer
    (≥500 m). When `UNET_BUFFER_TEST=1` and a mask is supplied, those pixels
    are removed from `label_mask` BEFORE crop-centre sampling, and any sampled
    crop window overlapping the forbidden region is resampled. Default OFF:
    if the env flag is unset or no mask is passed, behaviour is unchanged.

    S6 F1: `water_mask` (H×W bool, optional) — if provided, the smoothness
    loss crop is restricted to UNLABELLED WATER pixels (water=1 AND label=0).
    When None, falls back to the S5 behaviour (ones_like = all pixels), but
    that path is now weight-safe because INTERP_SMOOTH_W default is 0.005
    and INTERP_TV_W default is 0.0.
    """
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    H, W, C = feats.shape

    # Per-channel z-score (water-pixel mean/std would be ideal but global is
    # fine — feeding the same stats at inference keeps the model coherent)
    #
    # S8-1 (CARRIER_NO_ZSCORE=1, default OFF — byte-unchanged legacy path):
    # The IDW carrier is the LAST channel (index C-1) when INTERP_INPUT=1.
    # Z-scoring it zeros its DC depth level (median ~17 m on KP), destroying the
    # coarse positive-depth scaffold (Eldesokey 2020 CVPR).  When this knob is ON,
    # exempt channel C-1 from z-scoring and divide by CARRIER_SCALE_M (default 25 m)
    # instead so the net sees a coarse [0,1] depth prior at native scale.
    # fmean[C-1]=0.0 / fstd[C-1]=CARRIER_SCALE_M are stored on model.feat_mean/feat_std
    # (L803-804) so predict_grid_tta re-uses the identical transform at inference —
    # NO predict-path code change required.
    _carrier_no_zscore = (os.environ.get("CARRIER_NO_ZSCORE", "0") == "1"
                          and os.environ.get("INTERP_INPUT", "0") == "1")
    _carrier_scale_m = float(os.environ.get("CARRIER_SCALE_M", "25.0"))
    feats_z = feats.copy()
    fmean = np.array([feats_z[..., ci].mean() for ci in range(C)], dtype=np.float32)
    fstd = np.array([feats_z[..., ci].std() + 1e-6 for ci in range(C)], dtype=np.float32)
    if _carrier_no_zscore:
        # Carrier is channel C-1; exempt from z-score, scale by CARRIER_SCALE_M
        carrier_raw_median = float(np.median(feats_z[..., C - 1][feats_z[..., C - 1] > 0])
                                   if (feats_z[..., C - 1] > 0).any() else 0.0)
        fmean[C - 1] = 0.0
        fstd[C - 1] = _carrier_scale_m
        L.info(f"S8-1 CARRIER_NO_ZSCORE: carrier ch={C-1}  "
               f"raw_median={carrier_raw_median:.2f}m  "
               f"scaled_median={carrier_raw_median/_carrier_scale_m:.4f}  "
               f"CARRIER_SCALE_M={_carrier_scale_m:.1f}m")
    for ci in range(C):
        feats_z[..., ci] = (feats_z[..., ci] - fmean[ci]) / fstd[ci]
    feats_t = torch.from_numpy(feats_z.transpose(2, 0, 1)).float()

    # s2-dl request #1 — leakage-safe crop sampling (gated, default OFF)
    buffer_test = os.environ.get("UNET_BUFFER_TEST", "0") == "1"
    forbid = None
    if buffer_test and forbidden_mask is not None:
        forbid = np.asarray(forbidden_mask, dtype=bool)
        if forbid.shape != label_mask.shape:
            L.warning(f"forbidden_mask shape {forbid.shape} != label_mask {label_mask.shape}; ignoring")
            forbid = None
        else:
            n_before = int(label_mask.sum())
            label_mask = label_mask & (~forbid)
            n_after = int(label_mask.sum())
            L.info(f"U-Net buffer-test ON: forbidden_mask zeroed "
                   f"{n_before - n_after} of {n_before} labelled crop-centres "
                   f"({forbid.sum()} forbidden px total)")

    # S7: TARGET_NORM knob — z-score the target (mean/std over TRAIN pixels)
    # so the head starts near 0 regardless of depth range.
    # Default 'none' = raw metres, byte-unchanged legacy path.
    # 'zscore' stores (mean, std) on the returned model dict for un-norm at inference.
    # 'div_max' divides by 25 m (fixed normalisation).
    _target_norm = os.environ.get("TARGET_NORM", "none").lower()
    _target_mean = 0.0
    _target_std = 1.0
    if _target_norm == "zscore":
        _train_vals = label_grid[label_mask].astype(np.float64)
        _target_mean = float(_train_vals.mean())
        _target_std = float(_train_vals.std()) + 1e-6
        label_grid_normed = ((label_grid.astype(np.float64) - _target_mean) / _target_std).astype(np.float32)
        L.info(f"TARGET_NORM=zscore: mean={_target_mean:.3f}m std={_target_std:.3f}m (train labels)")
    elif _target_norm == "div_max":
        _target_std = MAX_DEPTH_M
        label_grid_normed = (label_grid.astype(np.float32) / MAX_DEPTH_M)
        L.info(f"TARGET_NORM=div_max: dividing by {MAX_DEPTH_M}m")
    else:
        label_grid_normed = label_grid.astype(np.float32)

    target_t = torch.from_numpy(label_grid_normed)
    mask_t = torch.from_numpy(label_mask.astype(np.float32))
    forbid_t = (torch.from_numpy(forbid) if forbid is not None else None)
    # Store norm params so predict_grid_tta can un-normalize
    _s7_norm_params = {"target_norm": _target_norm,
                       "target_mean": _target_mean,
                       "target_std": _target_std}

    # S6 F1: water mask tensor for smoothness loss (unlabelled-water-only masking)
    # NOTE: the S9 RESIDUAL_CARRIER buffer-set block that previously lived here was
    # misplaced — it used `_residual_carrier_on` (defined below at model-creation time)
    # and `model` (also created below). Moved to after model creation (S8 bug-fix).
    if water_mask is not None:
        _wm_arr = np.asarray(water_mask, dtype=np.float32)
        if _wm_arr.shape == (H, W):
            water_t = torch.from_numpy(_wm_arr)
        else:
            L.warning(f"water_mask shape {_wm_arr.shape} != (H,W)=({H},{W}); ignoring")
            water_t = None
    else:
        water_t = None

    label_idx_h, label_idx_w = np.where(label_mask)
    n_lbl = int(len(label_idx_h))
    if n_lbl < 30:
        raise RuntimeError(f"only {n_lbl} labelled pixels for U-Net training")

    # S9: RESIDUAL_CARRIER requires CARRIER_NO_ZSCORE=1 (carrier must be in metres/scale,
    # not z-scored, so the captured x[:,-1:] is meaningful in physical units).
    _residual_carrier_on = (os.environ.get("RESIDUAL_CARRIER", "0") == "1"
                            and os.environ.get("INTERP_INPUT", "0") == "1")
    if _residual_carrier_on:
        if os.environ.get("CARRIER_NO_ZSCORE", "0") != "1":
            raise RuntimeError(
                "RESIDUAL_CARRIER=1 requires CARRIER_NO_ZSCORE=1 — "
                "the captured carrier channel must be in metres/CARRIER_SCALE_M, "
                "not z-scored. Set CARRIER_NO_ZSCORE=1.")
        L.info("S9 RESIDUAL_CARRIER=1 confirmed: CARRIER_NO_ZSCORE=1 present.")

    model = _build_unet(C, base=base, n_onehot=int(n_onehot)).float()
    n_params = sum(p.numel() for p in model.parameters())
    pro_on = (os.environ.get("UNET_PRO_HEAD", "0") == "1" and int(n_onehot) > 0)
    L.info(f"U-Net SDB: {n_params:,} params, {C} input channels, base={base}"
           + (f", PRO head ON (FiLM on {n_onehot} one-hot ch)" if pro_on
              else f", PRO head OFF (n_onehot={n_onehot})"))

    # S9: set registered buffers on model AFTER model creation (S8 bug-fix: moved from
    # the TARGET_NORM block above where model did not yet exist).
    if _residual_carrier_on:
        import torch as _torch_rc
        if hasattr(model, "_tnorm_mean"):
            model._tnorm_mean.fill_(_target_mean)
            model._tnorm_std.fill_(_target_std)
            model._carrier_scale_m.fill_(_carrier_scale_m)
            L.info(f"S9 RESIDUAL_CARRIER buffers set: "
                   f"_tnorm_mean={_target_mean:.4f}  _tnorm_std={_target_std:.4f}  "
                   f"_carrier_scale_m={_carrier_scale_m:.1f}")

    if warmstart_state is not None:
        try:
            model.load_state_dict(warmstart_state, strict=False)
            L.info("U-Net: warm-started from cached weights")
        except Exception as ex:
            L.warning(f"U-Net warm-start failed: {ex}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    # 5-epoch warmup, then cosine to 0
    def _lr(ep):
        if ep < 5:
            return float(ep + 1) / 5.0
        t = (ep - 5) / max(1, epochs - 5)
        return 0.5 * (1 + np.cos(np.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)

    # Depth-band reweighting (Khalifa is heavy in 15-20 m; rare bins get up-weighted)
    # DEPTH_DENSITY_WEIGHT=0 disables reweighting (prior-neutralised config A, Round-6)
    # DL-#14: DEPTH_BALANCE_MODE=quantile replaces fixed 5 m bins with quantile bins
    # computed from THIS site's train labels, and drives crop-centre sampling probability
    # so rare shallow/deep cells are oversampled into training crops.
    # Gate: DEPTH_BALANCE_MODE=quantile (default "fixed" = legacy, byte-unchanged).
    _density_on = int(os.environ.get("DEPTH_DENSITY_WEIGHT", "1")) != 0
    _depth_balance_mode = os.environ.get("DEPTH_BALANCE_MODE", "fixed").lower()
    _n_depth_bins = int(os.environ.get("N_DEPTH_BINS", "8"))

    if _depth_balance_mode == "quantile":
        # Basin-adaptive quantile bins from actual train-label distribution
        _train_depths = label_grid[label_mask]
        _quantile_edges = np.quantile(_train_depths,
                                      np.linspace(0, 1, _n_depth_bins + 1))
        # Deduplicate edges (collapsed modes produce duplicate quantile values)
        _quantile_edges = np.unique(_quantile_edges)
        if len(_quantile_edges) < 3:
            # Degenerate: fall back to fixed bins
            L.warning("[DL-#14] quantile edges collapsed to <3 unique values; "
                      "falling back to fixed bins")
            _depth_balance_mode = "fixed"
        else:
            bins = _quantile_edges
            # Clip last edge slightly above max so digitize puts max in last bin
            bins[-1] = bins[-1] + 1e-3
            band_idx = np.digitize(_train_depths, bins) - 1
            band_idx = np.clip(band_idx, 0, len(bins) - 2)
            n_bins = len(bins) - 1
            band_counts = np.bincount(band_idx, minlength=n_bins).astype(np.float32) + 1
            band_w = (band_counts.mean() / band_counts).clip(0.5, 8.0)
            weight_grid = np.zeros_like(label_grid, dtype=np.float32)
            weight_grid[label_mask] = band_w[band_idx]
            weight_t = torch.from_numpy(weight_grid)
            # Per-label sampling probability for crop-centre oversampling
            # (the load-bearing change: equalises depth distribution the net sees)
            _lbl_depths = label_grid[label_mask]
            _lbl_bin_idx = np.digitize(_lbl_depths, bins) - 1
            _lbl_bin_idx = np.clip(_lbl_bin_idx, 0, n_bins - 1)
            _crop_sample_p = band_w[_lbl_bin_idx].astype(np.float64)
            _crop_sample_p /= _crop_sample_p.sum()
            print(f"[DL-#14] DEPTH_BALANCE_MODE=quantile: {n_bins} adaptive bins "
                  f"from {bins[0]:.2f} to {bins[-1]:.2f} m "
                  f"(edges={np.round(bins,2).tolist()})")
            print(f"[DL-#14] band_counts={band_counts.astype(int).tolist()}  "
                  f"band_w={np.round(band_w,2).tolist()}")

    if _depth_balance_mode == "fixed":
        bins = np.array([0, 5, 10, 15, 20, 25 + 1e-3])
        band_idx = np.digitize(label_grid[label_mask], bins) - 1
        band_counts = np.bincount(band_idx, minlength=len(bins) - 1).astype(np.float32) + 1
        if _density_on:
            band_w = (band_counts.mean() / band_counts).clip(0.5, 4.0)
        else:
            band_w = np.ones(len(bins) - 1, dtype=np.float32)  # uniform weights
            print("[train] DEPTH_DENSITY_WEIGHT=0 — depth-band reweighting DISABLED "
                  "(prior-neutralised)")
        weight_grid = np.zeros_like(label_grid, dtype=np.float32)
        weight_grid[label_mask] = band_w[band_idx]
        weight_t = torch.from_numpy(weight_grid)
        # Fixed mode: uniform crop-centre sampling (legacy behaviour)
        _crop_sample_p = None

    # S12: OBS_PRECISION_W=1 — override weight_t with per-source inverse-variance
    # precision grid (w = 1/σ_obs²).  Mean-normalised to 1.0 over labelled pixels
    # so the loss scale and LR are unchanged vs the depth-band path.
    # Default OBS_PRECISION_W=0 → skip this block entirely (byte-identical to S8).
    _obs_prec_on = (os.environ.get("OBS_PRECISION_W", "0") == "1"
                    and obs_precision_grid is not None)
    if _obs_prec_on:
        wg = np.asarray(obs_precision_grid, dtype=np.float32)
        assert wg.shape == (H, W), (
            f"obs_precision_grid shape {wg.shape} != (H,W)=({H},{W})")
        # Zero out non-labelled pixels (forbidden pixels already cleared from label_mask)
        wg_labelled = wg.copy()
        wg_labelled[~label_mask] = 0.0
        lbl_vals = wg_labelled[label_mask]
        n_lbl_prec = int((lbl_vals > 0).sum())
        if n_lbl_prec < 5:
            L.warning(f"S12 obs_precision_grid: only {n_lbl_prec} labelled pixels "
                      f"have positive precision — falling back to depth-band weights")
        else:
            # Mean-normalise to 1.0 over labelled pixels
            mean_prec = float(lbl_vals[lbl_vals > 0].mean())
            wg_norm = wg_labelled / max(mean_prec, 1e-6)
            weight_t = torch.from_numpy(wg_norm)
            L.info(f"S12 OBS_PRECISION_W=1: {n_lbl_prec} labelled px, "
                   f"mean_raw={mean_prec:.4f}, normalised_mean=1.0")
            L.info(f"S12 precision stats: "
                   f"min={float(wg_norm[label_mask].min()):.4f}  "
                   f"max={float(wg_norm[label_mask].max()):.4f}  "
                   f"mean={float(wg_norm[label_mask].mean()):.4f}")

    # S7: NLL_WARMUP_EPOCHS — run the #S7-1 winning regression loss for the first
    # N epochs then switch to NLL. Default 0 = no warmup (unchanged).
    _nll_warmup_ep = int(os.environ.get("NLL_WARMUP_EPOCHS", "0"))
    _loss_spec_at_start = os.environ.get("LOSS", "nll")

    t0 = time.time()
    best = float("inf")
    best_state = None
    no_improve = 0
    for ep in range(epochs):
        # S7: NLL_WARMUP_EPOCHS switching (default-OFF, 0 = no change)
        if _nll_warmup_ep > 0:
            if ep < _nll_warmup_ep:
                os.environ["LOSS"] = "mse"
            else:
                os.environ["LOSS"] = _loss_spec_at_start
        model.train()
        ep_loss = 0.0
        ep_n = 0
        for _ in range(crops_per_epoch):
            xs, ys, ms, ws, ws_water = [], [], [], [], []
            for _ in range(batch):
                # s2-dl request #1 — reject crops overlapping the forbidden
                # (test-block + buffer) region; resample up to 20 tries.
                # DL-#14: when _crop_sample_p is set (quantile mode), sample
                # crop centres proportional to inverse-frequency band weight so
                # rare shallow/deep depths are oversampled equally.
                for _try in range(20):
                    if _crop_sample_p is not None:
                        k = int(rng.choice(n_lbl, p=_crop_sample_p))
                    else:
                        k = int(rng.integers(0, n_lbl))
                    cy = int(label_idx_h[k]); cx = int(label_idx_w[k])
                    y0 = int(np.clip(cy - crop // 2 + rng.integers(-30, 30), 0, max(1, H - crop)))
                    x0 = int(np.clip(cx - crop // 2 + rng.integers(-30, 30), 0, max(1, W - crop)))
                    if forbid_t is None:
                        break
                    if not bool(forbid_t[y0:y0 + crop, x0:x0 + crop].any()):
                        break
                    # else: window straddles a test block → resample
                xs.append(feats_t[:, y0:y0 + crop, x0:x0 + crop])
                ys.append(target_t[y0:y0 + crop, x0:x0 + crop])
                ms.append(mask_t[y0:y0 + crop, x0:x0 + crop])
                ws.append(weight_t[y0:y0 + crop, x0:x0 + crop])
                # S6 F1: collect water mask crop for unlabelled-only smoothness
                if water_t is not None:
                    _wmc = water_t[y0:y0 + crop, x0:x0 + crop]
                else:
                    import torch as _t6_wm; _wmc = _t6_wm.ones(crop, crop)
                ws_water.append(_wmc)
            x = torch.stack(xs); y = torch.stack(ys)
            m = torch.stack(ms); w = torch.stack(ws)
            opt.zero_grad()
            mu, logs = model(x)
            loss = _hetero_nll(mu, logs, y, m, w)
            # S6: edge-aware smoothness on UNLABELLED WATER ONLY (F1 fix).
            # (gated: LOSS contains 'interp', default OFF; see _interp_smooth_loss)
            # S6 F2: anti-collapse SILog+range now active when LOSS=nll+interp
            # (handled inside _hetero_nll when 'silog'/'range' present; the S6
            #  default LOSS in run_s6_interp.py is nll+interp+silog+range).
            import os as _os5
            if "interp" in _os5.environ.get("LOSS", "nll").lower():
                # S6 F1: use real water crop, NOT ones_like
                import torch as _t5
                _wm_batch = _t5.stack(ws_water)  # (N, crop, crop)
                _smooth = _interp_smooth_loss(mu, logs, m, _wm_batch)
                loss = loss + _smooth
            # #REQ2 / UNET_REGIME_HEAD=1 — gate-BCE regulariser.
            # Anchors α to 1[y ≤ τ_m] on labelled pixels so each head specialises.
            # Weight REGIME_GATE_W (default 0.1).  Gated; no change to loss when OFF.
            if (os.environ.get("UNET_REGIME_HEAD", "0") == "1"
                    and hasattr(model, "_last_alpha")):
                import torch as _torch
                _tau = float(os.environ.get("REGIME_TAU_M", "3.0"))
                _gw  = float(os.environ.get("REGIME_GATE_W", "0.1"))
                # alpha shape: (N,1,H,W); y shape: (N,H,W)
                _alpha = model._last_alpha  # (N,1,H,W)
                _alpha_sq = _alpha.squeeze(1)  # (N,H,W)
                # target: 1 where truth depth ≤ τ, 0 otherwise
                _gate_target = (y <= _tau).to(_alpha_sq.dtype)
                # BCE on labelled pixels only
                _valid = m > 0
                if _valid.sum() > 0:
                    _bce = _torch.nn.functional.binary_cross_entropy(
                        _alpha_sq[_valid], _gate_target[_valid], reduction="mean"
                    )
                    loss = loss + _gw * _bce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.detach()); ep_n += 1
        sched.step()
        avg = ep_loss / max(1, ep_n)
        if avg < best - 1e-3:
            best = avg
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1:
            L.info(f"  epoch {ep+1:3d}/{epochs}  loss={avg:.4f}  "
                   f"lr={opt.param_groups[0]['lr']:.5f}  "
                   f"elapsed={time.time()-t0:.0f}s  best={best:.4f}")
        # Early stopping
        if no_improve >= 10 and ep >= 15:
            L.info(f"  early stop at epoch {ep+1} (no improvement for 10 epochs)")
            break
    # S7: restore LOSS env to original value after warmup (belt-and-suspenders)
    if _nll_warmup_ep > 0:
        os.environ["LOSS"] = _loss_spec_at_start

    if best_state is not None:
        model.load_state_dict(best_state)
    model.feat_mean = fmean
    model.feat_std = fstd
    # S7: store TARGET_NORM params on the model so predict_grid_tta can un-normalize
    model._s7_target_norm = _target_norm
    model._s7_target_mean = _target_mean
    model._s7_target_std = _target_std
    # S8 bug-fix: store HEAD_KIND on the model so forward() uses the training-time
    # value at inference, even if os.environ is restored by a probe finally-block.
    model._s8_head_kind = os.environ.get("HEAD_KIND", "softplus").lower()

    # S10-1 DL: train-only affine output calibration (OUTPUT_CALIB, default "none" = OFF).
    # Fit truth = a + b*pred on TRAIN-fold pixels (in un-normalized metres).
    # Leakage-safe: (a,b) fit ONLY on TRAIN pixels, applied unchanged at test.
    # Default "none" = OFF; byte-identical to S8 when unset.
    #
    # S11 additions (new OUTPUT_CALIB modes, default-OFF):
    #   "depth_strat_affine" — same leakage-safe affine fit but restricted to train pixels
    #       with pred > percentile(pred_train, 25), excluding the shallow-anchor cluster
    #       that biased the S10 linear slope (b=1.14 instead of expected ~0.67).
    #       Applied at predict_grid_tta as a + b*pred (same hook as "linear").
    #   "sigma_rescale" — pure prediction-distribution rescaling; NO labels touched at all.
    #       Stores _train_pred_std and _train_pred_mean on the model so the caller can
    #       rescale test predictions:  out = mean(pred_test) + s*(pred_test-mean(pred_test))
    #       where s = _train_pred_std / std(pred_test).  Applied POST-PREDICTION in the
    #       calling script (run_s11_interp.py); predict_grid_tta is unchanged for this mode.
    _output_calib = os.environ.get("OUTPUT_CALIB", "none").lower()
    if _output_calib in ("linear", "isotonic", "depth_strat_affine", "sigma_rescale"):
        try:
            model.eval()
            # Full-grid forward over the label_mask pixels (un-normed metres via predict_grid_tta).
            # mc_passes=0: single deterministic pass (calib_b still None at this point,
            # so the S10 calibration guard is a no-op — no recursion risk).
            _mu_tr_grid, _ = predict_grid_tta(
                model, feats, label_mask,
                tile=512, overlap=32, use_tta=False, mc_passes=0)
            # Sample at labelled train pixels
            _pred_train_m = _mu_tr_grid[label_idx_h, label_idx_w]
            _truth_train_m = label_grid[label_idx_h, label_idx_w]
            # Keep only finite, valid-depth pixels
            _ok_calib = (np.isfinite(_pred_train_m) & np.isfinite(_truth_train_m)
                         & (_pred_train_m > 0) & (_truth_train_m > 0))
            _pt = _pred_train_m[_ok_calib]
            _tt = _truth_train_m[_ok_calib]
            if len(_pt) < 10:
                L.warning(f"OUTPUT_CALIB={_output_calib}: too few train pixels "
                          f"({len(_pt)}) — skipping calibration")
            else:
                if _output_calib == "linear":
                    # Fit truth = a + b*pred (truth-on-pred, so apply as pred_cal = a + b*pred)
                    _A = np.column_stack([_pt, np.ones(len(_pt))])
                    _sol, *_ = np.linalg.lstsq(_A, _tt, rcond=None)
                    _calib_b, _calib_a = float(_sol[0]), float(_sol[1])
                    model._calib_a = _calib_a
                    model._calib_b = _calib_b
                    L.info(f"OUTPUT_CALIB=linear fit: a={_calib_a:.4f}  b={_calib_b:.4f}  "
                           f"n_train={len(_pt)}  (expect b≈0.67)")
                elif _output_calib == "isotonic":
                    from sklearn.isotonic import IsotonicRegression as _IR
                    _iso = _IR(out_of_bounds="clip").fit(_pt, _tt)
                    model._calib_iso = _iso
                    # Store sentinel so predict_grid_tta knows isotonic is fitted
                    model._calib_b = float("nan")  # sentinel: not None but NaN → isotonic
                    L.info(f"OUTPUT_CALIB=isotonic fitted on {len(_pt)} train pixels")
                elif _output_calib == "depth_strat_affine":
                    # S11 path 1: exclude bottom-25th-percentile of pred_train (shallow anchor).
                    # This prevents the 2-4m training cluster from dominating the slope fit.
                    _p25 = float(np.percentile(_pt, 25))
                    _deep_mask = _pt > _p25
                    _pt_deep = _pt[_deep_mask]
                    _tt_deep = _tt[_deep_mask]
                    L.info(f"OUTPUT_CALIB=depth_strat_affine: p25={_p25:.2f}m  "
                           f"n_all={len(_pt)}  n_above_p25={len(_pt_deep)}")
                    if len(_pt_deep) < 10:
                        L.warning(f"depth_strat_affine: too few above-p25 pixels ({len(_pt_deep)}) — "
                                  f"falling back to full linear")
                        _pt_deep = _pt
                        _tt_deep = _tt
                    _A = np.column_stack([_pt_deep, np.ones(len(_pt_deep))])
                    _sol, *_ = np.linalg.lstsq(_A, _tt_deep, rcond=None)
                    _calib_b, _calib_a = float(_sol[0]), float(_sol[1])
                    model._calib_a = _calib_a
                    model._calib_b = _calib_b
                    L.info(f"OUTPUT_CALIB=depth_strat_affine fit (above p25): "
                           f"a={_calib_a:.4f}  b={_calib_b:.4f}  "
                           f"n_fit={len(_pt_deep)}  (expect b≈0.67)")
                elif _output_calib == "sigma_rescale":
                    # S11 path 2: store train prediction std/mean for post-hoc rescaling.
                    # No labels touched — pure prediction distribution.
                    # The calling script applies:
                    #   out = mean(pred_test) + s*(pred_test - mean(pred_test))
                    #   where s = _train_pred_std / std(pred_test)
                    _train_pred_std = float(np.std(_pt))
                    _train_pred_mean = float(np.mean(_pt))
                    model._train_pred_std = _train_pred_std
                    model._train_pred_mean = _train_pred_mean
                    L.info(f"OUTPUT_CALIB=sigma_rescale: train_pred_std={_train_pred_std:.4f}m  "
                           f"train_pred_mean={_train_pred_mean:.4f}m  n_train={len(_pt)}")
                    # No _calib_b set — predict_grid_tta left unchanged for this mode.
        except Exception as _ex_calib:
            L.warning(f"OUTPUT_CALIB={_output_calib} fitting failed: {_ex_calib}")
    else:
        # Ensure no stale calibration attributes from a prior run
        for _attr in ("_calib_a", "_calib_b", "_calib_iso",
                      "_train_pred_std", "_train_pred_mean"):
            if hasattr(model, _attr):
                delattr(model, _attr)

    L.info(f"U-Net trained in {time.time()-t0:.0f}s  "
           f"(head_kind={model._s8_head_kind}, target_norm={_target_norm})")
    return model


# ════════════════════════════════════════════════════════════════════════
# Tiled inference + 4-way TTA (h-flip × v-flip)
# ════════════════════════════════════════════════════════════════════════
def predict_grid_tta(model, feats: np.ndarray, water: np.ndarray,
                      tile: int = 512, overlap: int = 64,
                      use_tta: bool = True, mc_passes: int = -1
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Tiled, water-only inference with optional 4-way test-time augmentation.

    Returns ``(mu, sigma)`` arrays — both ``(H, W)`` float32 with NaN over
    non-water pixels.

    s2-dl request #9 — when ``mc_passes>0`` (or env ``UNET_MC_PASSES``), run that
    many MC-Dropout stochastic forward passes (dropout kept ON) and combine:
    ``sigma = sqrt(mean(sigma_aleatoric^2) + var(mu_passes))`` — the aleatoric
    head variance + the epistemic spread (Gal & Ghahramani 2016, Kendall & Gal
    2017).  Default 0 keeps the deterministic single-pass behaviour unchanged.
    """
    import torch

    # Only consult the env knob when the caller leaves mc_passes at the
    # sentinel (-1); explicit values (incl. 0 from the recursive single-pass
    # calls below) are honoured verbatim to avoid infinite recursion.
    if mc_passes < 0:
        mc_passes = int(os.environ.get("UNET_MC_PASSES", "0") or 0)
    if mc_passes and mc_passes > 1:
        mus, sigs = [], []
        # enable dropout (train mode) but keep BN in eval via a targeted toggle
        for _ in range(mc_passes):
            model.eval()
            for m in model.modules():
                if m.__class__.__name__.startswith(("Dropout",)):
                    m.train()
            mu_p, sig_p = predict_grid_tta(model, feats, water, tile=tile,
                                           overlap=overlap, use_tta=use_tta,
                                           mc_passes=0)
            mus.append(mu_p); sigs.append(sig_p)
        model.eval()
        mu_stack = np.stack(mus, 0)          # (P, H, W)
        sig_stack = np.stack(sigs, 0)
        with np.errstate(invalid="ignore"):
            mu_mean = np.nanmean(mu_stack, 0)
            aleo = np.nanmean(sig_stack ** 2, 0)
            epis = np.nanvar(mu_stack, 0)
            sig_tot = np.sqrt(aleo + epis)
        sig_tot = np.where(water, sig_tot, np.nan)
        return mu_mean.astype(np.float32), sig_tot.astype(np.float32)

    H, W, C = feats.shape
    feats_z = (feats - model.feat_mean) / model.feat_std
    feats_t = torch.from_numpy(feats_z.transpose(2, 0, 1)).unsqueeze(0).float()
    out_mu = np.zeros((H, W), dtype=np.float32)
    out_si = np.zeros((H, W), dtype=np.float32)

    # Encoder has 4 maxpools, so input H/W must be multiples of 16. Pad
    # symmetrically with reflection, run the model, then crop back.
    PAD_MULT = 16

    def _forward(t):
        h_in, w_in = int(t.shape[-2]), int(t.shape[-1])
        h_pad = (PAD_MULT - h_in % PAD_MULT) % PAD_MULT
        w_pad = (PAD_MULT - w_in % PAD_MULT) % PAD_MULT
        if h_pad or w_pad:
            t = torch.nn.functional.pad(
                t, (0, w_pad, 0, h_pad), mode="reflect")
        with torch.no_grad():
            mu, logs = model(t)
        mu = mu.cpu().numpy()
        sg = np.exp(logs.cpu().numpy())
        if h_pad or w_pad:
            mu = mu[..., :h_in, :w_in]
            sg = sg[..., :h_in, :w_in]
        return mu, sg

    def _avg_tta(inp):
        if not use_tta:
            mu, sig = _forward(inp)
            return mu[0], sig[0]
        accm = np.zeros(inp.shape[2:], dtype=np.float32)
        accs = np.zeros(inp.shape[2:], dtype=np.float32)
        n = 0
        for fh in (False, True):
            for fv in (False, True):
                t = inp
                if fh: t = torch.flip(t, dims=[-1])
                if fv: t = torch.flip(t, dims=[-2])
                mu, sig = _forward(t)
                m = mu[0]; s = sig[0]
                if fv: m = m[::-1]; s = s[::-1]
                if fh: m = m[:, ::-1]; s = s[:, ::-1]
                accm += np.ascontiguousarray(m)
                accs += np.ascontiguousarray(s)
                n += 1
        return accm / n, accs / n

    model.eval()
    y = 0
    while y < H:
        y_end = min(y + tile, H)
        x = 0
        while x < W:
            x_end = min(x + tile, W)
            yy0 = max(0, y - overlap); yy1 = min(H, y_end + overlap)
            xx0 = max(0, x - overlap); xx1 = min(W, x_end + overlap)
            inp = feats_t[:, :, yy0:yy1, xx0:xx1]
            mu, sig = _avg_tta(inp)
            py0 = y - yy0; py1 = py0 + (y_end - y)
            px0 = x - xx0; px1 = px0 + (x_end - x)
            out_mu[y:y_end, x:x_end] = mu[py0:py1, px0:px1]
            out_si[y:y_end, x:x_end] = sig[py0:py1, px0:px1]
            x = x_end
        y = y_end
    # S7: un-normalize mu when TARGET_NORM was applied during training
    _tn = getattr(model, "_s7_target_norm", "none")
    if _tn and _tn != "none":
        _tm = getattr(model, "_s7_target_mean", 0.0)
        _ts = getattr(model, "_s7_target_std", 1.0)
        out_mu = out_mu * _ts + _tm
        # sigma is in normalized units too; scale back
        out_si = out_si * _ts
    # S10-1 DL: apply train-only affine output calibration if fitted.
    # Guard: only active when OUTPUT_CALIB!=none AND calibration was fitted.
    # out_si (sigma) is deliberately untouched — only the mean is recalibrated.
    _calib_b = getattr(model, "_calib_b", None)
    if _calib_b is not None:
        if np.isfinite(_calib_b):
            # Linear: out_mu = a + b * out_mu
            _calib_a = getattr(model, "_calib_a", 0.0)
            out_mu = _calib_a + _calib_b * out_mu
        else:
            # Isotonic: _calib_b is NaN sentinel — use _calib_iso
            _calib_iso = getattr(model, "_calib_iso", None)
            if _calib_iso is not None:
                _shape = out_mu.shape
                out_mu = _calib_iso.predict(out_mu.ravel()).reshape(_shape).astype(np.float32)
    out_mu = np.clip(out_mu, 0, MAX_DEPTH_M)
    out_mu = np.where(water, out_mu, np.nan)
    out_si = np.where(water, out_si, np.nan)
    return out_mu, out_si
