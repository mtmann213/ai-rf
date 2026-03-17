# Opal Vanguard: Strategic Roadmap & Operational Timeline

**Objective:** Transition the "Opal Vanguard" from a laboratory ResNet to a full-stack **Cognitive Radio & Signals Intelligence (SIGINT) System.**

---

## I. The "Data-First" Mission (Immediate Priority)
Before advancing the architecture, we must maximize the quality and diversity of the model's "nutrients."

### 1. External Dataset Acquisition
*   **DeepSig RadioML 2016.10B:** A larger, cleaner version of the 2016 dataset to provide more diverse baseline samples.
*   **TorchSig High-Fidelity Augmentation:** Generate additional 1M+ samples with extreme CFO (Carrier Frequency Offset) and multipath distortion.
*   **Public SIGINT Repositories:** Search for and integrate open-source USRP captures from the SDR community.

### 2. Custom Laptop Data Factory (Mike's SDR Tasks)
*   **The "Dirty" Hardware Library:** Focus on capturing signals that the model currently fails on (4ASK, 8ASK, 64QAM).
*   **Gain-Waterfall Series:** Capture the same modulation at 5dB increments from the floor to saturation.
*   **Environmental Fingerprinting:** Record signals in different physical locations (Lab, Outdoor, Near EMI sources) to teach the model to ignore local interference.

---

## II. Implementation Timeline

### Phase 1-3: Foundation & ResNet Evolution (COMPLETED)
*   [x] Standardized V8.0 ResNet with 57-class vocabulary.
*   [x] Achieved 57.1% Hardware Accuracy on B205-mini data.

### Phase 4: Hardware Acclimation & V9.0 Transition (IN PROGRESS)
*   [x] Identified 1D-ResNet visual plateau.
*   [x] **V9.0 Ignition:** Launched CNN-LSTM Ensemble ("Eyes + Ears") to capture temporal signal rhythm.
*   [ ] **Validation:** Break the 60% accuracy mark.

---

## III. Bleeding-Edge Potential (The Future Vision)

### Phase 10: Neural Demodulation & Synchronization
*   **Blind Demodulation:** Train models to output raw bitstreams (LLRs) directly from distorted I/Q data, bypassing traditional math.
*   **Neural Synchronization:** AI-driven frame detection and clock recovery.

### Phase 11: RF Foundation Models (Self-Supervised Learning)
*   **The "RF-GPT" Concept:** Train a Masked Autoencoder on millions of unlabeled raw captures.
*   **Zero-Shot Detection:** Use the foundation model to identify unknown or adversarial waveforms without specific training data.

### Phase 12: Edge Deployment ("Specter-in-a-Box")
*   **Optimization:** Use TensorRT and INT8 Quantization to shrink the V9.0 brain.
*   **Deployment:** Field testing on NVIDIA Jetson / Mobile platforms.

### Phase 13: Cognitive Electronic Warfare (EW)
*   **Smart Jamming:** Use Reinforcement Learning to learn how to disrupt adversarial links with minimal power.
*   **LPI/LPD:** AI-driven frequency hopping and modulation morphing to evade detection.

---
**Status:** V9.0 Temporal Ignition Active.
**Tech Lead:** Mike Mann
**Data Nutrients:** Prioritizing USRP gain-sweeps and external dataset integration.
