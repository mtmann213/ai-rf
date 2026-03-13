# Opal Vanguard: AI-Native RF Fundamentals

## 1. The Paradigm Shift: Deep PHY
Traditional RF systems rely on modular, hand-engineered blocks. **AI-Native RF** replaces these with Neural Receivers that learn end-to-end mapping from raw I/Q samples.

### Core Advantages:
*   **Neural Resilience:** CNNs recognize patterns in high-interference or non-linear environments where traditional matched filters fail.
*   **Hardware Agnostic:** Models adapt to hardware impairments (DC offset, phase noise) through training.
*   **Real-time Adaptability:** On-the-fly fine-tuning to specific channel conditions.

## 2. Methodology: Neural Modulation Classification (NMC)

### The RadioML 2018.01A Challenge
We target 24 modulation types (Analog & Digital) across varying SNR levels (-20dB to +30dB).

### Architecture: Residual Networks (ResNet)
To process temporal I/Q data at scale, we use a **1D-ResNet**:
*   **Skip Connections:** Allow the model to be deeper without the "Vanishing Gradient" problem.
*   **Global Average Pooling:** Reduces parameters and prevents overfitting.
*   **Softmax Output:** Probability distribution across all 24 classes.

## 3. Engineering Pivots for Stability

### Indestructible Normalization
RF data often contains extreme outliers or corrupted `NaN` values. Our pipeline uses:
1.  **NaN Scrubbing:** Replaces corrupted samples with 0.
2.  **Outlier Clipping:** Hard-limits signal spikes to prevent math overflows.
3.  **L2 Scaling:** Ensures every I/Q vector has unit energy before entering the model.

### Turbocharged I/O
Streaming 21GB datasets on WSL2 creates a disk bottleneck. Our **Chunked Generator** reads large contiguous blocks (4096 samples) to maximize disk throughput and keep the GPU saturated.

---
**Status:** Phase 3 - ResNet Evolution
**Tech Lead:** Mike Mann
