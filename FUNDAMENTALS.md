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
3.  **Zero-Variance Protection:** Uses **Time-Axis LayerNormalization** to maintain statistical consistency even during periods of signal silence, while strictly preserving the phase/amplitude relationship between I and Q channels.

### Turbocharged I/O
Streaming 21GB datasets on WSL2 creates a disk bottleneck. Our **Chunked Generator** reads large contiguous blocks (4096 samples) to maximize disk throughput and keep the GPU saturated.

## 4. Post-Training Deployment & Operations

Once the neural architecture is fully trained on the RadioML dataset, we transition from building an AI to commanding a piece of cognitive radio infrastructure. The resulting weights (`.h5` / `.keras` files) represent a "Neural Digital-to-Analog Brain"—a universal classifier capable of instantly determining the probability distribution of 24 different modulation types from a raw 1024-sample chunk of I/Q radio data, regardless of noise distortion.

Deployment proceeds in three distinct stages:

### A. "Waterfall" Validation (Immediate Post-Training)
Before field deployment, the model undergoes strict simulated limits testing (`benchmark_snr.py`):
*   **SNR Stress Testing:** The model is fed signals at specific Signal-to-Noise Ratios, ranging from perfectly clear (+30dB) down to pure static (-20dB).
*   **Performance Analytics:** We generate an "Accuracy vs. SNR" waterfall curve to identify the exact noise floor where the AI's classification vision begins to degrade. Confusion matrices are mapped to see where the model struggles to differentiate closely related modulations (e.g., 64-QAM vs. 128-QAM).

### B. Simulated Adversarial Testing (Phase 4)
The baseline training utilizes Additive White Gaussian Noise (AWGN). For real-world robustness, we inject brutal physical phenomena into the testing data using the `Sionna` framework:
*   **Multipath Fading:** Simulates signals bouncing off urban infrastructure before hitting the receiver array.
*   **Doppler Shift:** Simulates transmitter mobility at extreme velocities (e.g., UAVs, jets).

### C. Live Over-The-Air Intercept (Phase 5)
The ultimate hardware integration (`usrp_vanguard.py`). We bridge the Neural Receiver to physical Software Defined Radios (USRP B205-mini):
*   **The Execution:** The SDR streams live, unadulterated electromagnetic waves directly into the ResNet inference engine in real-time.
*   **The Outcome:** The AI acts as a live Signals Intelligence (SIGINT) node, actively identifying and classifying ambient signals (e.g., WiFi, Bluetooth, FM, digital telemetry) as they propagate through the local environment.

---
**Status:** Phase 3 - ResNet Evolution
**Tech Lead:** Mike Mann
