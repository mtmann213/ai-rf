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
1.  **Label Scrubbing:** Hardens the dataset against internal corruption by forcing labels to strict [0, 1] bounds.
2.  **Soft-Clip Normalization:** Uses `x / (1 + |x|)` to map the entire infinite number line to a smooth (-1, 1) range, providing perfect gradient stability.
3.  **Zero-Variance Protection:** Uses `BatchNormalization` to maintain statistical consistency even during periods of signal silence.

### Turbocharged I/O
Streaming 21GB datasets on WSL2 creates a disk bottleneck. Our **Chunked Generator** reads large contiguous blocks (4096 samples) to maximize disk throughput and keep the GPU saturated.

---
**Status:** Phase 3 - ResNet Evolution
**Tech Lead:** Mike Mann
