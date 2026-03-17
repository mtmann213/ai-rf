# Opal Vanguard: Strategic Roadmap & Operational Timeline

**Objective:** Transition the "Opal Vanguard" from a laboratory ResNet to a full-stack **Cognitive Radio & Signals Intelligence (SIGINT) System.**

---

## I. Project Chronology & Implementation Timeline

### Phase 1: Foundation & Baseline (COMPLETED)
*   [x] **Sionna Integration:** Link-level simulation (AWGN).
*   [x] **Functional API:** Fully serializable model architecture.
*   [x] **Benchmarking:** "Waterfall" accuracy curves established.

### Phase 2: Data Engineering (COMPLETED)
*   [x] **HDF5 Streaming:** Memory-efficient loading for 21GB datasets.
*   [x] **Stability Engine:** Overflow-proof normalization and NaN scrubbing.
*   [x] **Turbocharged I/O:** Chunked loading for WSL2 disk optimization.

### Phase 3: Heavy Compute & ResNet (COMPLETED)
*   [x] **ResNet Upgrade:** Advanced Residual Network implementation (57-class vocabulary).
*   [x] **Distributed Compute:** Docker/WSL2 setup for RTX 3080 Ti.
*   [x] **Data Integrity Breakthrough (V7.6.1):** Overcame HDF5 label corruption via Mathematical Reconstruction.
*   [x] **The "Radio Professor" Milestone:** Achieved **15.5% accuracy** on 500,000 synthetic samples (V8.3).

### Phase 4: Hardware Acclimation (IN PROGRESS - V8.5)
*   [x] **The Generalization Gap:** Identified 2.5% accuracy drop on USRP hardware data.
*   [x] **Super-Hybrid Training:** Implemented 50/50 mix of hardware and simulation.
*   [x] **Hardware Breakthrough:** Jumped from 2.5% to **57.1% validation accuracy** (V8.5).
*   [ ] **Refinement Marathon:** Complete 100 epochs on 3080 Ti with 0.00002 learning rate.
*   [ ] **Failure Analysis:** Generate Confusion Matrix to identify hardware-specific confusion points.

---

## II. Current Strategic Focus: V9.0 - Temporal Intelligence

As the V8.5 ResNet reaches its accuracy plateau, we must transition to the **CLDNN (CNN-LSTM-DNN)** architecture identified by **DeepSig/RadioML research** as the definitive "Intelligence" milestone.

### The CNN-LSTM Ensemble ("Eyes + Ears")
*   **The "Eyes" (CNN/ResNet):** Focus on the 2D geometric structure of the I/Q constellation.
*   **The "Ears" (LSTM):** Focus on the 1D temporal rhythm and symbol-to-symbol transitions that the ResNet is currently "blind" to.
*   **The Goal:** Break the 60% hardware accuracy mark by capturing both spatial and temporal signal signatures.

---

## III. Future Mission Phases (V10.0 - V12.0)

### Phase 10: Demodulation & Synchronization
**Goal:** Extract the raw bitstream (1s and 0s) from the classified signal.
*   **Neural Synchronization:** Implement "Synchronization Heads" to detect frame headers and preambles.
*   **Adaptive Demodulation:** Use Phase 1 classification to dynamically select the correct demodulation algorithm (e.g., QPSK, 16QAM).

### Phase 11: Framing & Bitstream Analysis
**Goal:** Understand the structure and protocol of the data.
*   **De-Interleaving/Descrambling:** Reverse channel coding or data whitening.
*   **Protocol ID:** Determine if the bitstream follows a known standard (WiFi, Bluetooth, LTE, AIS).

### Phase 12: Content Extraction (The "Grand Vision")
**Goal:** Translate raw bits into actionable meaning.
*   **Payload Decoding:** Extract the actual message content (Audio, Text, Telemetry).
*   **Intelligence Layer:** Analyze data for intent, metadata, and origin.

---

## IV. The "Specter's Edge" Operational Roadmap
To harden the model for field deployment, we must capture a "Dirty" real-world dataset on the laptop SDRs:
1.  **Spatial Diversity:** NLOS (Non-Line of Sight) and Multipath (Echo) environments.
2.  **Dynamic Range:** Capture signals from "barely audible" to "near saturation" (Gain Sweeps).
3.  **Electronic Warfare:** Integrate jamming gauntlets (CW, Swept, Intermittent) to teach adversarial resilience.

---
**Status:** Phase 4 Refinement Active (57.1% Accuracy).
**Tech Lead:** Mike Mann
**Research Reference:** DeepSig/RadioML (O'Shea et al.) - ResNet and CLDNN Benchmarks.
