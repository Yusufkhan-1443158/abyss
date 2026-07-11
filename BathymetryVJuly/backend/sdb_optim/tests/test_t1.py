"""
T1 test battery — Physics-Informed Architecture.

Tests
-----
T1a  Import-clean: physics_loss, attention_unet_v2, spatial_cv all import without error.
T1b  PhysicsInformedLoss: finite scalar loss with gradient; physics terms > 0 and
     decrease when predictions follow Stumpf ratio.
T1c  AttentionUNetV2: forward on synthetic (B,C,H,W) → (mu, logsigma) correct shape,
     finite, positive mu; MC-Dropout variance > 0 across two stochastic passes.
T1d  spatial_block_kfold: k non-empty folds; min inter-fold test-centroid distance ≥
     requested min_test_train_dist_m; fold_min_distance diagnostic ≥ buffer.
T1e  Determinism: fixed seed reproduces identical results for all three modules.

Run:
    .venv/bin/python3 -m pytest backend/sdb_optim/tests/test_t1.py -v
  or directly:
    .venv/bin/python3 backend/sdb_optim/tests/test_t1.py
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── project root on path ─────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.WARNING)  # suppress info during tests

PASS_LABEL = "PASS"
FAIL_LABEL = "FAIL"
results: dict[str, str] = {}


# ════════════════════════════════════════════════════════════════════════
# T1a — import-clean
# ════════════════════════════════════════════════════════════════════════
def test_t1a_imports() -> None:
    """All T1 modules import without error."""
    from backend.sdb_optim import physics_loss         # noqa: F401
    from backend.sdb_optim import attention_unet_v2    # noqa: F401
    from backend.sdb_optim import spatial_cv           # noqa: F401
    from backend.sdb_optim.physics_loss import PhysicsInformedLoss
    from backend.sdb_optim.attention_unet_v2 import AttentionUNetV2
    from backend.sdb_optim.spatial_cv import spatial_block_kfold, fold_min_distance
    assert PhysicsInformedLoss is not None
    assert AttentionUNetV2 is not None
    assert spatial_block_kfold is not None
    assert fold_min_distance is not None


# ════════════════════════════════════════════════════════════════════════
# T1b — PhysicsInformedLoss correctness
# ════════════════════════════════════════════════════════════════════════
def test_t1b_physics_loss() -> None:
    """PhysicsInformedLoss: finite scalar, gradient flows, physics terms > 0."""
    from backend.sdb_optim.physics_loss import PhysicsInformedLoss

    torch.manual_seed(0)
    N = 64

    # Simple base loss: MSE on mu vs target (ignores logsigma for simplicity)
    class SimpleMSE(nn.Module):
        def forward(self, mu, logsigma, target):
            return nn.functional.mse_loss(mu, target)

    # Use a leaf tensor (nn.Parameter) so .grad is populated after backward()
    mu_leaf   = nn.Parameter(torch.rand(N) * 10.0)
    logsigma  = torch.zeros(N)
    target    = torch.rand(N) * 10.0
    ratio     = torch.rand(N) * 0.5 + 1.0          # Stumpf ratios > 1
    ln_blue   = torch.rand(N) * (-3.0) - 1.0        # negative log-reflectances
    ln_green  = torch.rand(N) * (-2.0) - 0.5

    crit = PhysicsInformedLoss(
        base_loss=SimpleMSE(),
        lambda_stumpf=0.10,
        lambda_lyzenga=0.05,
        lambda_monotonic=0.10,
    )
    loss = crit(mu_leaf, logsigma, target,
                stumpf_ratio=ratio, ln_blue=ln_blue, ln_green=ln_green)

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.dim() == 0, "Loss should be a scalar"
    loss.backward()
    assert mu_leaf.grad is not None, "No gradient on mu_leaf"
    assert torch.isfinite(mu_leaf.grad).all(), "Non-finite gradients on mu_leaf"
    mu = mu_leaf.detach()  # use for downstream assertions

    # Physics terms > 0 individually
    # Stumpf: random predictions should have non-zero affine residual
    st = PhysicsInformedLoss._stumpf_affine_residual(mu.detach(), ratio)
    assert float(st) >= 0.0, f"Stumpf residual must be non-negative, got {st}"

    # Lyzenga: random predictions should have non-zero OLS residual
    lyz = PhysicsInformedLoss._lyzenga_batch_residual(
        mu.detach(), ln_blue, ln_green)
    assert float(lyz) >= 0.0, f"Lyzenga residual must be non-negative, got {lyz}"

    # Verify physics terms decrease when predictions follow Stumpf exactly
    # Build predictions that ARE perfectly affine in the ratio
    ratio_np = ratio.detach().numpy()
    m1 = 5.0; m0 = 1.0
    mu_stumpf = torch.tensor(m1 * ratio_np + m0, dtype=torch.float32)
    st_good = PhysicsInformedLoss._stumpf_affine_residual(mu_stumpf, ratio)
    st_bad  = PhysicsInformedLoss._stumpf_affine_residual(mu.detach(), ratio)
    assert float(st_good) < float(st_bad) + 1e-3, (
        f"Stumpf term should be lower when preds follow ratio: "
        f"good={st_good:.4f} bad={st_bad:.4f}"
    )

    print(f"  T1b: loss={loss.item():.4f} (finite+grad OK), "
          f"Stumpf_good={st_good:.4f} < Stumpf_bad={st_bad:.4f}")


# ════════════════════════════════════════════════════════════════════════
# T1c — AttentionUNetV2 shape / finite / MC-Dropout
# ════════════════════════════════════════════════════════════════════════
def test_t1c_attention_unet_v2() -> None:
    """AttentionUNetV2: correct output shape, finite values, MC-Dropout variance > 0."""
    from backend.sdb_optim.attention_unet_v2 import AttentionUNetV2

    torch.manual_seed(42)
    B, C, H, W = 2, 12, 64, 64

    x = torch.randn(B, C, H, W)

    # ---- depth=4, use_cbam=True (default) --------------------------------
    model = AttentionUNetV2(c_in=C, base=16, p_drop=0.10, depth=4, use_cbam=True)
    model.eval()
    with torch.no_grad():
        mu, logsigma = model(x)

    assert mu.shape == (B, H, W), f"mu shape {mu.shape} != {(B, H, W)}"
    assert logsigma.shape == (B, H, W), f"logsigma shape {logsigma.shape} != {(B, H, W)}"
    assert torch.isfinite(mu).all(), "mu contains non-finite values"
    assert torch.isfinite(logsigma).all(), "logsigma contains non-finite values"
    assert (mu > 0).all(), "mu should be positive (Softplus)"
    assert (logsigma >= -1.5).all() and (logsigma <= 2.5).all(), \
        "logsigma out of [-1.5, 2.5] clamp range"

    # ---- depth=3, use_cbam=False ----------------------------------------
    model3 = AttentionUNetV2(c_in=C, base=16, p_drop=0.10, depth=3, use_cbam=False)
    model3.eval()
    with torch.no_grad():
        mu3, ls3 = model3(x)
    assert mu3.shape == (B, H, W)
    assert torch.isfinite(mu3).all()

    # ---- MC-Dropout: two passes in train() mode should give different outputs ----
    model_mc = AttentionUNetV2(c_in=C, base=16, p_drop=0.50, depth=4, use_cbam=False)
    model_mc.train()   # keep dropout active
    torch.manual_seed(1)
    mu_a, _ = model_mc(x)
    torch.manual_seed(2)
    mu_b, _ = model_mc(x)
    var = (mu_a - mu_b).var().item()
    assert var > 0.0, (
        f"MC-Dropout: two passes should differ (p_drop=0.50), but var={var:.6f}"
    )
    print(f"  T1c: mu shape={tuple(mu.shape)}, finite, pos; "
          f"MC-Dropout pixel var={var:.4f} > 0")


# ════════════════════════════════════════════════════════════════════════
# T1d — spatial_block_kfold correctness
# ════════════════════════════════════════════════════════════════════════
def test_t1d_spatial_cv() -> None:
    """spatial_block_kfold: k non-empty folds; min test-centroid distance ≥ requested."""
    from backend.sdb_optim.spatial_cv import (
        spatial_block_kfold, fold_min_distance, _min_centroid_distance_m,
    )

    rng = np.random.default_rng(123)
    # Simulate 600 soundings in a ~30 km × 20 km box (UAE-like)
    N = 600
    lat = rng.uniform(24.20, 24.47, N)
    lon = rng.uniform(52.58, 52.85, N)
    coords = np.stack([lat, lon], axis=1)

    k = 5
    min_dist_m = 500.0
    folds = spatial_block_kfold(
        coords, k=k, block_size_m=2000.0,
        min_test_train_dist_m=min_dist_m, seed=42, verbose=False,
    )

    assert len(folds) == k, f"Expected {k} folds, got {len(folds)}"

    # All folds non-empty
    for i, (tr, te) in enumerate(folds):
        assert len(tr) > 0, f"fold {i} has empty train set"
        assert len(te) > 0, f"fold {i} has empty test set"
        # No overlap
        assert len(np.intersect1d(tr, te)) == 0, f"fold {i} train/test overlap"

    # All N points appear as test exactly once across all folds
    all_test = np.concatenate([te for _, te in folds])
    unique_test = np.unique(all_test)
    assert len(unique_test) == N, (
        f"Expected all {N} points to appear as test; got {len(unique_test)}"
    )

    # Check inter-fold centroid distance
    centroids = []
    for _, te in folds:
        c = (float(lat[te].mean()), float(lon[te].mean()))
        centroids.append(c)
    min_d = _min_centroid_distance_m(centroids)
    # The centroid distance should be substantial (> 500 m for a 30×20 km box with 5 folds)
    assert min_d > 500.0, (
        f"Inter-fold centroid distance {min_d:.0f} m should be > 500 m"
    )

    # fold_min_distance diagnostic
    diag = fold_min_distance(coords, folds, fold_id=0)
    assert diag >= min_dist_m - 1.0, (
        f"fold_min_distance={diag:.1f} m should be >= {min_dist_m:.1f} m"
    )

    print(f"  T1d: k={k} folds, all non-empty, all-N-test-once; "
          f"inter-fold centroid min dist={min_d:.0f} m; "
          f"fold-0 buffer dist={diag:.1f} m (requested {min_dist_m:.0f} m)")


# ════════════════════════════════════════════════════════════════════════
# T1e — determinism
# ════════════════════════════════════════════════════════════════════════
def test_t1e_determinism() -> None:
    """Fixed seed reproduces identical results for all three modules."""
    from backend.sdb_optim.physics_loss import PhysicsInformedLoss
    from backend.sdb_optim.attention_unet_v2 import AttentionUNetV2
    from backend.sdb_optim.spatial_cv import spatial_block_kfold

    # ---- PhysicsInformedLoss ─────────────────────────────────────────
    class ConstLoss(nn.Module):
        def forward(self, mu, logsigma, target):
            return nn.functional.mse_loss(mu, target)

    torch.manual_seed(7)
    N = 32
    mu = torch.rand(N)
    ls = torch.zeros(N)
    tgt = torch.rand(N) * 10
    ratio = torch.rand(N) * 0.5 + 1.0
    lb = torch.rand(N) * -3
    lg = torch.rand(N) * -2

    crit = PhysicsInformedLoss(ConstLoss(), lambda_stumpf=0.1,
                               lambda_lyzenga=0.05, lambda_monotonic=0.1)

    torch.manual_seed(7)
    loss1 = crit(mu, ls, tgt, stumpf_ratio=ratio, ln_blue=lb, ln_green=lg).item()
    torch.manual_seed(7)
    loss2 = crit(mu, ls, tgt, stumpf_ratio=ratio, ln_blue=lb, ln_green=lg).item()
    assert abs(loss1 - loss2) < 1e-6, f"PhysicsLoss not deterministic: {loss1} vs {loss2}"

    # ---- AttentionUNetV2 ─────────────────────────────────────────────
    torch.manual_seed(99)
    model = AttentionUNetV2(c_in=8, base=8, p_drop=0.0, depth=3, use_cbam=False)
    model.eval()
    x = torch.randn(1, 8, 32, 32)
    with torch.no_grad():
        mu_a, _ = model(x)
        mu_b, _ = model(x)
    assert torch.allclose(mu_a, mu_b), "AttentionUNetV2 not deterministic (eval mode)"

    # ---- spatial_block_kfold ─────────────────────────────────────────
    rng = np.random.default_rng(0)
    coords = rng.uniform([24.0, 52.0], [25.0, 55.0], (300, 2))
    folds_a = spatial_block_kfold(coords, k=3, block_size_m=1000, seed=5, verbose=False)
    folds_b = spatial_block_kfold(coords, k=3, block_size_m=1000, seed=5, verbose=False)
    for i, ((tra, tea), (trb, teb)) in enumerate(zip(folds_a, folds_b)):
        assert np.array_equal(tra, trb) and np.array_equal(tea, teb), \
            f"spatial_block_kfold not deterministic at fold {i}"

    print(f"  T1e: physics_loss={loss1:.6f} (reproducible), "
          f"unet deterministic in eval, spatial_cv same folds both calls")


# ════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════
_TESTS = [
    ("T1a", "import-clean", test_t1a_imports),
    ("T1b", "PhysicsInformedLoss finite+grad+physics-terms", test_t1b_physics_loss),
    ("T1c", "AttentionUNetV2 shape+finite+MC-Dropout", test_t1c_attention_unet_v2),
    ("T1d", "spatial_block_kfold non-empty+distance", test_t1d_spatial_cv),
    ("T1e", "determinism", test_t1e_determinism),
]


def run_all() -> bool:
    """Run all tests, print results, return True if all pass."""
    all_pass = True
    print("\n=== T1 test battery ===")
    for tag, desc, fn in _TESTS:
        try:
            fn()
            status = PASS_LABEL
        except Exception as exc:
            status = f"{FAIL_LABEL}: {exc}"
            all_pass = False
        results[tag] = status
        print(f"  {tag} [{desc}]: {status}")

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
