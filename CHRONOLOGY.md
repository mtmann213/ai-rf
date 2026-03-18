# Project Opal Vanguard: Progress Chronology (V1.0 - V8.5)

This document is the official technical diary of the Opal Vanguard Neural Receiver project. It tracks every architectural pivot, mathematical breakthrough, and operational milestone on our journey to mastering AI-Native RF.

---

## Phase 1: Foundation & Project Inception (V1.0 - V3.0)

### [2026-03-12] Milestone 1: The First Breath (V1.0)
- **Mission Defined:** Launched Project Opal Vanguard to build a Neural Receiver for 24-modulation classification using RadioML 2018.01A and NVIDIA Sionna.
- **Tech Stack:** Standardized on Python 3.12, TensorFlow 2.21, and Sionna 1.2.1.
- **Architectural Pivot:** Refactored from Keras subclassing to **Functional API** to fix `NotImplementedError` during model saving.
- **Execution:** Successfully completed 10-epoch training on synthetic AWGN data; established end-to-end pipeline with 4.17% baseline accuracy.

### [2026-03-12] Milestone 2: RadioML Integration (V2.0)
- **Data Factory:** Built `data_loader.py` using `h5py` for large-scale streaming.
- **The "Bad Gateway" Pivot:** Manufactured a local 7,200-sample synthetic RadioML dataset (`GOLD_XYZ_OSC.0001_1024.hdf5`) when official servers failed.
- **Execution:** Completed 20-epoch training on local synthetic data (`opal_vanguard_v2.h5`).

### [2026-03-13] Milestone 3: ResNet Upgrade & Desktop Compute (V3.0)
- **ResNet Implementation:** Upgraded the architecture to a **Residual Network** for deeper feature extraction and better gradient flow.
- **Distributed Strategy:** Implemented Docker/Docker-Compose to move training from the laptop to the **RTX 3080 Ti Desktop**.
- **WSL2 Integration:** Authored `DOCKER_GUIDE.md` for Windows/WSL2 deployment.

---

## Phase 2: The Stability Battles (V4.0 - V7.0)

### [2026-03-13] Milestone 4: Numerical Stability (V4.0)
- **Hurdle:** Encountered `nan` loss during full-scale training on the 3080 Ti.
- **Solution:** Implemented **L2 Normalization** and lowered the learning rate to `1e-4`.
- **The "Turbo" Fix:** Refactored the generator to read **contiguous chunks** (4096 samples) instead of random seeks, reducing training time by 90%.

### [2026-03-13] Milestone 5: Extreme Robustness (V5.0)
- **Hurdle:** Persistent outliers in the 2018 dataset still caused weight collapse.
- **Solution:** Signal pre-scaling by maximum absolute value before L2 calculation, making the math "overflow-proof."

### [2026-03-13] Milestone 6: Fortress Stability (V6.0)
- **Solution:** Switched from L2 normalization to **Standard Z-Score scaling** (Mean 0, Std 1).
- **Architecture Shield:** Added a `BatchNormalization` layer as the very first operation in the ResNet to filter hardware-level outliers.
- **Learning Rate:** Dropped to `2e-5` for safe convergence.

### [2026-03-13] Milestone 7: The "Nuclear-Grade" Logits Fix (V6.5)
- **Hurdle:** Softmax explosion diagnosed as `-log(0)` in `categorical_crossentropy`.
- **Solution:** Stripped `softmax` from the final layer to output raw **Logits**; used `from_logits=True` in the loss function for C++ backend stability.
- **Initialization:** Applied `he_normal` kernel initialization to align weights with ReLU activations.

### [2026-03-13] Milestone 8: The Absolute Shield (V7.0)
- **Hurdle:** Z-Score squaring caused overflow on extreme outliers.
- **Solution:** Implemented **Soft-Clip Normalization** (`x / (1 + |x|)`), mapping the infinite number line to a smooth (-1, 1) range without squaring. This remains the project's definitive stability fix.

---

## Phase 3: The Event Horizon & Breakthrough (V7.1 - V7.6.1)

### [2026-03-14] Milestone 15: The Zero-Variance Bug (V7.1)
- **Bug:** Discovered that `LayerNormalization` caused "silence" failure on low-energy samples.
- **Fix:** Reinstated `BatchNormalization` for statistical consistency across batches.

### [2026-03-14] Milestone 16: Label Alignment (V7.2)
- **Correction:** Found class index mismatch; reverted `data_loader.py` to the original RadioML index order (starting with `32PSK`).

### [2026-03-14] Milestone 17: Phase Physics (V7.4)
- **Discovery:** `LayerNormalization(axis=-1)` was normalizing I and Q against each other, destroying phase information.
- **Fix:** Shifted normalization to **Time-Axis** (`axis=1`) to preserve the physical relationship between I and Q components.

### [2026-03-14] Milestone 18: Data Integrity Breakthrough (V7.6.1)
- **The Discovery:** Identified that the RadioML 2018.01A `Y` (Labels) dataset was internally corrupted with memory garbage.
- **The Mathematical Solution:** Implemented **Mathematical Label Reconstruction** in `data_loader.py`. By ignoring the HDF5 labels and reconstructing them from the sample index, we bypassed the corruption entirely.
- **Result:** Accuracy jumped from 2% to **34.01%** in a single epoch.

---

## Phase 4: Universal Intelligence (V8.0 - V8.4)

### [2026-03-15] Milestone 19: Universal Expansion (V8.0)
- **Vocabulary:** Expanded the model from 24 classes to **57 distinct modulations** using the TorchSig V2 framework.
- **Unified Node:** Built a combined PyTorch/TensorFlow Docker environment.

### [2026-03-15] Milestone 20: The Deep Intelligence Marathon (V8.3)
- **Architecture:** 1D-ResNet (Pure CNN / "Eyes").
- **Dataset:** 500,000 Synthetic samples (TorchSig).
- **Goal:** Establish a baseline "Radio Professor" brain for 57 modulation classes.
- **Result:** Achieved **15.5% accuracy** on the expanded 57-class vocabulary.
- **Takeaway:** The model mastered the "ideal" math of radio but remained "blind" to real-world hardware impairments.

### [2026-03-16] Milestone 21: Bridging the Generalization Gap (V8.4)
- **Architecture:** Resumed V8.3 ResNet.
- **Dataset:** `VDF_SPECTER_GOLDEN.h5` (Real USRP hardware snapshots).
- **Strategy:** Super-Hybrid Training (50% real hardware / 50% simulation).
- **Result:** Rapid breakthrough, jumping from 2.5% to **47.6% hardware accuracy**.
- **Takeaway:** Proven that the "Radio Professor" foundation could rapidly adapt to specific hardware accents (DC offset, phase noise).

### [2026-03-16] Milestone 22: The High-Intensity Refinement (V8.5)
- **Task:** Precision fine-tuning with a lower learning rate (`0.00002`).
- **Result:** Accuracy plateaued at **57.1%**.
- **Diagnostic Failure Analysis:** Top confusion points identified: 4ASK vs 8ASK and 32PSK vs 8PSK.
- **Takeaway:** Reached the mathematical limit of a purely visual (CNN) learner. The "Eyes" could see the constellation cloud but couldn't "hear" the rhythm of symbol transitions.

---

## Current Phase: Phase 5 - Temporal Intelligence (V9.0+)

### [2026-03-17] Milestone 24: The 70% Breakthrough (V9.1)
- **Strategy:** Specialist Refinement (Weighted Generator oversampling QAM/Analog).
- **Result:** Achieved **76.5% Training / 72.7% Validation Accuracy**.
- **Insight:** Successfully broke the 70% ceiling. Diagnostic analysis showed the LSTM fixed ASK and PSK confusion, but high-density QAM remains the final bottleneck.

---

## Current Phase: Phase 6 - Multi-Modal Intelligence (V9.2+)

### [2026-03-17] Milestone 26: The "Triple-Source" Regression (V9.3)
- **Problem:** Attempted to mix 50% hardware, 25% clean gold sim, and 25% physics baseline from scratch.
- **Result:** Severe regression; accuracy collapsed to **51%** as gradients from diverse noise sources clashed.
- **Takeaway:** Multi-modal architectures are too complex to start from scratch with high-entropy data mixes.

### [2026-03-18] Milestone 27: Surgical Weight Transfusion (V9.4)
- **Innovation:** Built `src/weight_transfusion.py` to surgically migrate 16 critical layers from the 72.7% Specialist model into the new Sovereign architecture.
- **Strategy:** Warm-started the Sovereign Eye with proven V9.1 knowledge. Reverted to stable Dual-Source (Hardware + Simulation) strategy.
- **Goal:** Focus the multi-modal brain exclusively on learning Polar (Amplitude/Phase) features to solve QAM density.
