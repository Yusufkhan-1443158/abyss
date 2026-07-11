# DL Pro v2 — turbidity-aware bathymetry default model

22-feature MC-Dropout MLP (hidden 192, p_drop 0.25), 5-seed ensemble, trained on pooled multi-date Khalifa Port pixel-medians. IHO calibration α=-0.165 m, β=1.005 (R²=0.896). Hold-out RMSE = 2.23 m, IHO 1a = 72.2 %, IHO 2 = 85.6 %.

## Probability-of-accuracy parameters

σ_total² = σ_MC² + (k_T · turbidity_g_m3)² + (k_d · max(0, depth - kd_limit))²
with **k_T = 0.012**, **k_d = 0.04**, **kd_limit = 12.0 m**.
