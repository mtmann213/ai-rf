# Project Opal Vanguard: Progress Chronology

This document tracks every technical decision and milestone on our journey to mastering AI-Native RF.

## [2026-03-12] Foundation & Infrastructure
- **Inception:** Launched project to classify 24 modulations using RadioML 2018.01A.
- **Environment:** Configured Python 3.12, TensorFlow 2.21, and Sionna 1.2.1.
- **Privacy:** Scrubbed military affiliations from all documentation.

## [2026-03-13] Stability Evolution
- **Pivots 1-6:** Attempted various scaling methods (L2, Z-Score, Titanium Shield) to handle the 2018 dataset's extreme outliers.
- **Milestone 14 (V7.0 - Event Horizon):** Established the definitive stability suite:
    1. **Soft-Clip Normalization:** `x / (1 + |x|)` for perfectly bounded, smooth-gradient inputs.
    2. **Exclusively LayerNormalization:** Eliminated batch-dependency to prevent corrupted samples from collapsing the model weights.
    3. **Double-Clip Optimizer:** Combined `global_clipnorm` and `clipvalue` for ultimate gradient protection.
    4. **Pure Precision:** Disabled XLA (JIT) to ensure maximum numerical consistency on the 3080 Ti.

- **Milestone 15 (V7.1 - Stability Patch):** Resolved final mathematical failure points:
    1. **Label Scrubbing:** Hardened `data_loader.py` to scrub `Y` labels, preventing dataset corruption from injecting infinite gradients.
    2. **BatchNorm Reinstated:** Swapped `LayerNormalization` back to `BatchNormalization` to eliminate the "zero-variance silence" bug.
    3. **Optimizer Refinement:** Swapped `global_clipnorm` for individual `clipnorm` to prevent float32 overflow during the global L2 sum.

---
**Current Status:** Event Horizon V7.1 Active. Stability verified via 800-step stress test. Ready for training.
