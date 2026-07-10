# DL Pro v3 — multi-region UAE bathymetry default model

23-feature MC-Dropout MLP, 3-seed ensemble, trained on a pooled
set of 29871 pixel-medians from 5 UAE regions:

- **khalifa_port**: 11,218 medians (2/3 dates)
- **KP_inner**: 5,640 medians (3/3 dates)
- **saadiyat_lulu**: 9,902 medians (2/3 dates)
- **abu_al_abyad**: 2,901 medians (3/3 dates)
- **ad_mainland**: 210 medians (3/3 dates)

## Hold-out, IHO-calibrated

- RMSE = **2.84 m**, MAE = 1.28 m, bias = +0.03 m
- IHO 1A = **60.8 %**,   1B = 60.8 %,   2A = 78.5 %,   2B = 83.6 %
- α = +0.384 m, β = 0.979, R²_cal = 0.879
