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

---
**Current Status:** Event Horizon V7.0 Active. Stability verified. Ready for heavy training.
