# Mission Plan: Vanguard Data Factory (VDF) - Strategic Roadmap

**Objective:** Transition the "Opal Vanguard" from a simple Signal Classifier to a full-stack **Cognitive Radio & Signals Intelligence (SIGINT) System.**

---

## Phase 1: The Signal Classifier (Current Milestone: V8.5)
**Goal:** Detect and identify the modulation type of a signal within the noise.

*   **Current State:** ~57-59% accuracy using 1D-ResNet (Conv1D) on a 57-class vocabulary.
*   **The "Anti-Stagnation" Plan (V9.0):**
    *   **Ensemble Learning:** Integrate an **LSTM (Long Short-Term Memory)** or **1D-Transformer** layer alongside the ResNet. This allows the AI to "hear" the temporal patterns (timing/rhythm) of the signal, not just the visual spectral shapes.
    *   **Synthetic-Real Hybrid:** Continuously mix the 500k TorchSig samples with active USRP hardware captures.
    *   **Environmental Hardening:** Inject real-world noise floors, multipath fading, and interference into the training set.

## Phase 2: Demodulation & Synchronization (V10.0)
**Goal:** Extract the raw bitstream (1s and 0s) from the classified signal.

*   **Neural Synchronization:** Implement a "Synchronization Head" in the model to detect frame headers, preambles, and seed sequences.
*   **Adaptive Demodulation:** Use the Phase 1 classification to dynamically select and parameterize the correct demodulation algorithm (e.g., QPSK, 16QAM, GMSK).
*   **Neural Demodulators:** Experiment with models that output Soft-bits (LLRs) directly from raw I/Q data.

## Phase 3: Framing & Bitstream Analysis (V11.0)
**Goal:** Understand the structure and protocol of the data.

*   **De-Interleaving & Descrambling:** Reverse any channel coding or data whitening used by the transmitter.
*   **Protocol Identification:** Determine if the bitstream follows a known standard (WiFi, Bluetooth, LTE, AIS) or a custom framing structure.
*   **Packet Reconstruction:** Reassemble the bitstream into logical data packets.

## Phase 4: Content Extraction & Intelligence (V12.0 - The "Grand Vision")
**Goal:** Translate the raw bits into actionable meaning.

*   **Payload Decoding:** Extract the actual message content (Audio, Text, Telemetry, Video).
*   **Intelligence Layer:** Analyze the data for intent, metadata, and origin (Who is talking? Where are they? What is the mission?).
*   **Cognitive Loop:** Feedback decoded information to the receiver to improve future tracking and classification of that specific signal source.

---

## 5. Development Roadmap: The "Data Factory" Branch

### Immediate Tasks (Evening Strategy):
1.  **V8.5 Failure Analysis:** Once the current refinement run plateaus, generate a **Confusion Matrix** for the `VDF_SPECTER_GOLDEN` dataset.
2.  **V9.0 Architecture Sketching:** Prepare the Python code for the **CNN-LSTM Hybrid** ("Eyes + Ears" model).
3.  **Hardware Capture (Mike):** Focus on "Gain Sweeps" and "Multipath Diversity" captures with the USRP to provide more training "nutrients."

---
**Status:** Phase 1 Refinement Active (57.1% Accuracy).
**Tech Lead:** Mike Mann
