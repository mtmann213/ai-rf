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
    4. **Pure Precision:** Disabled XLA (JIT) to ensure maximum numerical precision.

- **Milestone 15 (V7.1 - Stability Patch):** Resolved final mathematical failure points on the Desktop node:
    1. **Label Scrubbing:** Hardened `data_loader.py` to scrub `Y` labels, preventing dataset corruption from injecting infinite gradients.
    2. **BatchNorm Reinstated:** Swapped `LayerNormalization` back to `BatchNormalization` to eliminate the "zero-variance silence" bug.
    3. **Optimizer Refinement:** Swapped `global_clipnorm` for individual `clipnorm` to prevent float32 overflow during the global L2 sum.

- **Milestone 16 (V7.2 - Label Alignment):** Discovered and fixed class index mismatch.
    1. **Mapping Reset:** Reverted `data_loader.py` to use the 'Original' order found in `2018_01A/classes.txt` (starting with `32PSK`).
    2. **Weights Purge:** Wiped all poisoned weights from previous runs.
    3. **Accuracy Verification:** Confirmed accuracy climbing above random-guessing baseline immediately after alignment.

## [Research Branch] Laptop GPU Hardware Discovery
- **The Challenge:** Investigated why the **RTX PRO 2000 Blackwell** GPU was silent despite correct drivers.
- **Findings:** Confirmed via custom diagnostics and research (Medium: "Misadventures in Blackwell Support") that TensorFlow 2.21 pip packages lack the **Compute Capability 12.0** kernels required for Blackwell.
- **Solution Path:** Identified that a source-level Bazel build or waiting for TF 2.22 is required for native laptop acceleration. Distributed training to the 3080 Ti remains the primary mission path.

---
**Current Status:** Event Horizon V7.2 Active. Labels aligned. Training in progress.