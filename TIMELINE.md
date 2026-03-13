# Project Opal Vanguard: Implementation Timeline

This document outlines the evolutionary path from a synthetic Sionna baseline to a fully realized AI-Native Neural Receiver for real-world RF environments.

## Phase 1: Foundation & Benchmarking (Month 1)
**Goal:** Establish a verified baseline for 24-modulation classification in AWGN.
- [x] **Milestone 1.1:** Initial Sionna Link-Level Simulation (AWGN).
- [x] **Milestone 1.2:** 24-Modulation Conv1D "OpalVanguardModel" Architecture.
- [ ] **Milestone 1.3:** `benchmark_snr.py` implementation for automated Confusion Matrix generation.
- [ ] **Milestone 1.4:** Establish "Waterfall" Accuracy Curves (-20dB to +30dB).

## Phase 2: Dataset Integration & Real-World Impairments (Month 2)
**Goal:** Transition from pure synthetic data to the RadioML 2018.01A dataset.
- [ ] **Milestone 2.1:** Implement RadioML 2018.01A HDF5 Data Loader.
- [ ] **Milestone 2.2:** Hybrid Training: Augmenting RadioML samples with Sionna's differentiable noise.
- [ ] **Milestone 2.3:** Implement Hardware Impairment Simulation (DC Offset, Phase Noise, IQ Imbalance).

## Phase 3: Complex Channel Evolution (Month 3-4)
**Goal:** Train the model to survive non-line-of-sight (NLOS) and high-mobility environments.
- [ ] **Milestone 3.1:** Integrate Sionna Rayleigh & Rician Fading Channels.
- [ ] **Milestone 3.2:** Doppler Shift Resilience Training (High-speed mobility simulation).
- [ ] **Milestone 3.3:** Multi-path Interference Mitigation via Neural Feature Extraction.

## Phase 4: Neural Receiver & Joint Optimization (Month 5)
**Goal:** Evolve from a simple classifier to an end-to-end signal recovery system.
- [ ] **Milestone 4.1:** Neural Demapper Implementation (Soft-bit estimation/LLRs).
- [ ] **Milestone 4.2:** Joint Source-Channel Coding (JSCC) experiments using Autoencoders.
- [ ] **Milestone 4.3:** Quantization-Aware Training (QAT) for edge deployment (INT8/FP16).

## Phase 5: Hardware Integration & Deployment (Month 6+)
**Goal:** Real-time inference on SDR hardware.
- [ ] **Milestone 5.1:** Export model to ONNX/TensorRT for NVIDIA Blackwell optimization.
- [ ] **Milestone 5.2:** Integration with GNU Radio or custom Python SDR wrappers (SoapySDR).
- [ ] **Milestone 5.3:** Live Over-the-Air (OTA) Validation using USRP/HackRF hardware.

---
**Lead SDR Architect:** Mike Mann
**Project Status:** Phase 1 - In Progress
