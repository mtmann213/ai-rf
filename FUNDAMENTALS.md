# Opal Vanguard: AI-Native RF Fundamentals & Methodology

## 1. The Paradigm Shift: Why AI-Native RF?

Traditional Radio Frequency (RF) systems rely on modular, hand-engineered blocks (Synchronization, Equalization, Demodulation, Decoding). **AI-Native RF** (or "Deep PHY") replaces or augments these blocks with neural networks that can learn the end-to-end mapping from raw I/Q samples to bits.

### Core Advantages:
- **Neural Resilience:** CNNs can learn to recognize signal patterns in high-interference or non-linear environments that traditional matched filters might struggle with.
- **Hardware Agnostic:** A single model can potentially adapt to different front-end hardware impairments (DC offset, phase noise) through training.
- **Real-time Adaptability:** Models can be fine-tuned on the fly to specific channel conditions.

## 2. NVIDIA Sionna: The Differentiable PHY

Sionna is a Python-based library for link-level simulations built on top of TensorFlow. Its "Differentiable PHY" means the entire communications chain (Source → Mapper → Channel → Receiver) is a single, differentiable computational graph.

- **Differentiable Gradient Flow:** This allows training the transmitter and receiver jointly (Autoencoders).
- **GPU Acceleration:** Built for CUDA, enabling massive parallel simulation of complex channels (Rayleigh, Rician, Cluster-based models).

## 3. Methodology: Neural Modulation Classification (NMC)

### The RadioML 2018.01A Challenge
We are targeting 24 modulation types. This dataset includes:
- **Analog:** AM-DSB, AM-SSB, WB-FM.
- **Digital:** BPSK, QPSK, 8PSK, 16QAM, 64QAM, 128QAM, 256QAM, PAM4, GFSK, CPFSK, etc.

### Model Architecture: Conv1D CNN
To process temporal I/Q data, we use 1D Convolutional layers.
- **Input:** `[Batch, 1024, 2]` (1024 samples, I and Q components).
- **Feature Extraction:** Stacked `Conv1D` layers with `'same'` padding to preserve sequence length, followed by `MaxPooling1D` to reduce dimensionality while capturing translation-invariant features.
- **Classification:** A `Flatten` layer leading to `Dense` (Fully Connected) layers, ending with a `Softmax` activation over 24 classes.

### SNR Benchmarking (The "Waterfall" Curve)
We evaluate the model's accuracy across a range of SNR levels (-20dB to +30dB).
- **Low SNR (-20 to -5dB):** Noise dominates; the model learns to identify coarse spectral shapes.
- **High SNR (+10 to +30dB):** Signal is clean; the model distinguishes between subtle constellation differences (e.g., 64QAM vs 128QAM).

## 4. Operational Plan: Phased Implementation

### Phase 1: Standalone Sionna Baseline (Current)
- Implement a 24-modulation Sionna generator.
- Train the `OpalVanguardModel` on synthetic AWGN samples.
- Benchmark accuracy and generate a baseline Confusion Matrix.

### Phase 2: RadioML 2018.01A Integration (Future)
- Incorporate the 31.7GB RadioML 2018.01A dataset for diverse, real-world impairments.
- Hybrid Training: Mix Sionna's clean signals with RadioML's noisy/impaired samples.

### Phase 3: Complex Channel Evolution (Future)
- Introduce **Rayleigh Fading** (multipath) and **Doppler Shifts**.
- Transition the model from a simple classifier to a **Neural Receiver** that performs joint Demapping and Equalization.

### Phase 4: Opal-Vanguard Integration (Target)
- Port the trained models into the broader Opal-Vanguard SDR ecosystem.
- Implement real-time classification on live SDR hardware (e.g., USRP or HackRF).
