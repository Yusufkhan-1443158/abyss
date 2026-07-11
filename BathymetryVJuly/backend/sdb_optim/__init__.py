"""
sdb_optim — drop-in SDB optimization subpackage.

Four independent module groups. Imports are lazy/guarded so a partially-built
package never breaks `import backend.sdb_optim`.

  T1 physics-informed architecture : physics_loss, attention_unet_v2, spatial_cv
  T2 feature engineering / fusion   : feature_engineering, tidal_correction
  T3 validation & calibration suite : hybrid_losses, stratified_metrics, uncertainty
  T4 HPC / IO / structured logging  : cog_chunker, parallel_infer, structured_logging,
                                      subsystem_interface

  integration facade                : integration  (composes T1-T4 against the live
                                      pipeline primitives by composition, not mutation)
"""
__all__ = [
    "physics_loss", "attention_unet_v2", "spatial_cv",
    "feature_engineering", "tidal_correction",
    "hybrid_losses", "stratified_metrics", "uncertainty",
    "cog_chunker", "parallel_infer", "structured_logging", "subsystem_interface",
    "integration",
]
__version__ = "0.1.0-dev"
