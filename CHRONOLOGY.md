# Project Opal Vanguard: Progress Chronology (V1.0 - V9.4)

This document is the official technical diary of the Opal Vanguard Neural Receiver project. It tracks every architectural pivot, mathematical breakthrough, and operational milestone on our journey to mastering AI-Native RF.

---

## Phase 1: Foundation & Project Inception (V1.0 - V3.0)

### [2026-03-12] Milestone 1: The First Breath (V1.0)
- **Mission Defined:** Launched Project Opal Vanguard to build a Neural Receiver for 24-modulation classification using RadioML 2018.01A and NVIDIA Sionna.
- **Architectural Pivot:** Refactored from Keras subclassing to **Functional API** to fix `NotImplementedError` during model saving.
- **Execution:** Successfully completed 10-epoch training on synthetic AWGN data; established end-to-end pipeline with 4.17% baseline accuracy.

### [2026-03-12] Milestone 2: RadioML Integration (V2.0)
- **Data Factory:** Built `data_loader.py` using `h5py` for large-scale streaming.
- **Execution:** Completed 20-epoch training on local synthetic data (`opal_vanguard_v2.h5`).

### [2026-03-13] Milestone 3: ResNet Upgrade & Desktop Compute (V3.0)
- **ResNet Implementation:** Upgraded the architecture to a **Residual Network** for deeper feature extraction and better gradient flow.
- **Distributed Strategy:** Implemented Docker/Docker-Compose to move training from the laptop to the **RTX 3080 Ti Desktop**.

---

## Phase 2: The Stability Battles (V4.0 - V7.0)

### [2026-03-13] Milestone 4: Numerical Stability (V4.0)
- **Hurdle:** Encountered `nan` loss during full-scale training on the 3080 Ti.
- **Solution:** Implemented **L2 Normalization** and lowered the learning rate to `1e-4`.
- **The "Turbo" Fix:** Refactored the generator to read **contiguous chunks** (4096 samples) instead of random seeks, reducing training time by 90%.

### [2026-03-13] Milestone 8: The Absolute Shield (V7.0)
- **Hurdle:** Z-Score squaring caused overflow on extreme outliers.
- **Solution:** Implemented **Soft-Clip Normalization** (`x / (1 + |x|)`), mapping the infinite number line to a smooth (-1, 1) range without squaring. This remains the project's definitive stability fix.

---

## Phase 3: The Event Horizon & Breakthrough (V7.1 - V7.6.1)

### [2026-03-14] Milestone 18: Data Integrity Breakthrough (V7.6.1)
- **The Discovery:** Identified that the RadioML 2018.01A `Y` (Labels) dataset was internally corrupted with memory garbage.
- **The Mathematical Solution:** Implemented **Mathematical Label Reconstruction** in `data_loader.py`. By ignoring the HDF5 labels and reconstructing them from the sample index, we bypassed the corruption entirely.
- **Result:** Accuracy jumped from 2% to **34.01%** in a single epoch.

---

## Phase 4: Universal Intelligence (V8.0 - V8.5)

### [2026-03-15] Milestone 20: The Deep Intelligence Marathon (V8.3)
- **Architecture:** 1D-ResNet (Pure CNN / "Eyes").
- **Dataset:** 500,000 Synthetic samples (TorchSig).
- **Result:** Achieved **15.5% accuracy** on the expanded 57-class vocabulary.

### [2026-03-16] Milestone 21: Bridging the Generalization Gap (V8.4)
- **Dataset:** `VDF_SPECTER_GOLDEN.h5` (Real USRP hardware snapshots).
- **Strategy:** Super-Hybrid Training (50% real hardware / 50% simulation).
- **Result:** Rapid breakthrough, jumping from 2.5% to **47.6% hardware accuracy**.

### [2026-03-16] Milestone 22: The High-Intensity Refinement (V8.5)
- **Result:** Accuracy plateaued at **57.1%**.
- **Takeaway:** Reached the mathematical limit of a purely visual (CNN) learner.

---

## Phase 5: Temporal & Multi-Modal Intelligence (V9.0 - V9.4)

### [2026-03-17] Milestone 24: The 70% Breakthrough (V9.1)
- **Strategy:** Specialist Refinement (Weighted Generator oversampling QAM/Analog).
- **Architecture:** **CNN-LSTM Ensemble (CLDNN)**.
- **Result:** Achieved **76.5% Training / 72.7% Validation Accuracy**.
- **Insight:** Successfully broke the 70% ceiling using the LSTM "Ears."

### [2026-03-17] Milestone 26: The "Triple-Source" Regression (V9.3)
- **Problem:** Attempted to mix 50% hardware, 25% clean gold sim, and 25% physics baseline from scratch.
- **Result:** Severe regression; accuracy collapsed to **51%** due to gradient clash.

### [2026-03-18] Milestone 27: Surgical Weight Transfusion & Recovery (V9.4)
- **Innovation:** Built `src/weight_transfusion.py` to surgically migrate 16 critical layers from the 72.7% model into the new Sovereign architecture.
- **Strategy:** Warm-started the Sovereign Eye with proven V9.1 knowledge. 
- **Breakthrough:** Recovered from the 51% regression and successfully resumed the push for 80%.

### [2026-03-18] Milestone 28: The Stabilized Data Factory
- **Solution:** Hardened `src/mega_generator_v9.py` with **Stability Hooks** (HDF5 flushing, manual GC, and cooldown sleeps) to survive high-intensity dual-runs.
- **Status:** Launched the 1.14 Million sample overnight marathon.

---
**Status:** V9.4 Active. Refinement Marathon in progress.
**Tech Lead:** Mike Mann
