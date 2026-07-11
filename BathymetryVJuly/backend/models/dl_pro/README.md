# DL Pro — bathymetry default model

MC-Dropout MLP with 5x5 neighbourhood features, trained on pooled
multi-date Khalifa Port (Sentinel-2 L2A 20 m) with 10% in-situ pixel-
medians.  Saved here as a deployable bundle:

- `model.pt`  — `torch.save({'state_dict':…, 'in_dim':…, 'hidden':128, 'p_drop':0.20})`
- `meta.json` — feature normalisation, IHO calibration α/β, hold-out metrics, S-44 compliance.

## Loading

```python
import torch, json, numpy as np
from backend.dl_pro_engine import predict_dl_pro
depth, sigma, info = predict_dl_pro(s2_dict, ref_pts=optional_refs)
```

If `ref_pts` is given (CSV/ATL03/GEBCO-derived), the engine
re-fits the IHO calibration line on those pixels (closed form)
and optionally fine-tunes the MLP head — see `dl_pro_engine.py`.

## IHO S-44 (6th ed.) hold-out compliance — calibrated

| Order   | a (m) | b      | % within TVU |
|---------|-------|--------|--------------|
| Special | 0.25  | 0.0075 | 43.8% |
| 1a      | 0.50  | 0.013  | 66.1% |
| 1b      | 0.50  | 0.013  | 66.1% |
| 2       | 1.00  | 0.023  | 81.9% |
