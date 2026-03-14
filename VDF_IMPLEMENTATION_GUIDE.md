# VDF Implementation Guide: Project Opal Vanguard
**Technical Specifications for hardware-based dataset generation.**

## 1. Data Schema
*   **Shape:** `(N, 1024, 2)`
*   **Structure:** Flat datasets (`X`, `Y`, `Z`) for O(1) random access during training.
*   **Metadata:** Use HDF5 Attributes for per-sample "Sigma Hardening" details.

## 2. Signal Generation
*   **Bridge:** Pre-compute Sionna waveforms on the GPU node.
*   **Format:** Export as `.npy` files for the USRP sequencer to consume.
*   **Diversity:** Ensure every modulation is recorded across all frequency bands (433/915/2.4) and symbol rates (SPS 4/8/16).

## 3. Synchronization
*   **Pilot Tone:** 10ms CW burst at the start of every capture session.
*   **Receiver:** Use a Cross-Correlation trigger to align the window start-point. This eliminates "Label Drift."

## 4. Adversary Control
*   **Coordination:** Master Sequencer must trigger USRP #3 via ZMQ/TCP message ports.
*   **Tagging:** Only mark snapshots as "Jammed" if the Adversary USRP confirms the TX-Start ack.

## 5. Acclimation Prep
*   **Pilot Run:** First capture should be <100MB (2 classes) to verify the HDF5-to-ResNet pipeline.
*   **Normalization:** The `VDF_Receiver` must apply the `x/(1+|x|)` Soft-Clip before saving to match the training baseline.
