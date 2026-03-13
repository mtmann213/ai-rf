# Project Opal Vanguard: Progress Chronology

... (existing entries) ...

## [2026-03-13] Milestone 3: The "Numerical Stability" Pivot
- **The Challenge:** Encountered `nan` loss during the first epoch of full-scale ResNet training on the 3080 Ti.
- **Hurdle 1 (Normalization):** Identified that raw I/Q samples were causing exploding gradients. Implemented `L2 normalization` in `data_loader.py`.
- **Hurdle 2 (Stability):** Lowered the initial learning rate to `1e-4` to ensure smoother convergence.
- **Hurdle 3 (Checkpoints):** Fixed a `ModelCheckpoint` warning by transitioning from mid-epoch saves to end-of-epoch saves, ensuring `val_loss` is always available.

---
**Current Phase:** Phase 3 - ResNet Evolution
**Status:** Numerical stability fixes applied. Ready for stable training.
