# Mission Plan: Vanguard Data Factory (VDF)
**Objective:** Capture 500,000+ synchronized I/Q snapshots using the Hardware Trinity (3x USRP B205-mini) to create a high-fidelity, environment-aware training dataset for the `ai-rf` ResNet.

## 1. Modulation Requirements
We will target the "Trinity Suite," which includes all modulations from the 2018.01A dataset plus two "Vanguard-Specific" classes for real-world robustness.

*   **Digital Suite:** BPSK, QPSK, 8PSK, 16PSK, 32PSK, GMSK, OQPSK.
*   **Amplitude Suite:** OOK, 4ASK, 8ASK.
*   **Quadrature Suite:** 16QAM, 32QAM, 64QAM, 128QAM, 256QAM.
*   **Analog Suite:** AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC, FM.
*   **Vanguard Specials:** 
    *   **Class 25: "Pure Noise"** (TX disabled, capturing floor noise).
    *   **Class 26: "Active Jamming"** (Capturing the output of your `mike-bpsk-jammer` logic).

## 2. Signal Construction Strategy
To make the AI robust, the signals must be transmitted in two distinct modes:

### A. Random Symbol Mode (70% of Dataset)
*   **Purpose:** Teaches the AI the "Physics" of the modulation (the constellation shape).
*   **Method:** Generate purely random bitstreams and map them to symbols.
*   **Benefit:** Prevents the AI from "cheating" by learning a specific recurring data pattern.

### B. Framed Packet Mode (30% of Dataset)
*   **Purpose:** Teaches the AI the "Structure" of real radio comms.
*   **Method:** Include standard preambles, sync words, and header structures.
*   **Benefit:** Helps the AI identify signals even when the payload is silent or zeroed out.

## 3. Environment & Interference Matrix
Using the 3rd USRP as an **Adversary Node** is critical.

*   **SNR Control:** Vary the TX gain from 0dB to 60dB to simulate distance.
*   **The Adversary Node (USRP #3):**
    *   **Adjacent Channel Interference:** Transmit high-power BPSK signals 1MHz away.
    *   **In-Band Jamming:** Intermittently transmit noise bursts on top of the target.
*   **Multipath Simulation:** Vary RX placement to capture real-world echoes.

## 4. Technical Requirements & Dataset Scale
*   **Sample Rate:** 1 Msps (Standard) or 5 Msps (High-Res).
*   **Snapshots per Class:** 20,000 snapshots.
*   **Snapshot Shape:** `(1024, 2)` (I and Q channels).
*   **Total Dataset Size:** ~25GB (HDF5 format).
*   **Synchronization:** Use TCP-based triggers or a shared clock between TX/RX to ensure zero-error label assignment.

## 5. Development Roadmap (The "Data Factory" Branch)

1.  **The Sequencer (`src/vdf_sequencer.py`):** Automates the loop of switching modulations and gain levels.
2.  **The Labeler (`src/vdf_labeller.py`):** Implements a hard-coded tagging engine to ensure labels are never corrupted.
3.  **The Sionna Bridge:** Pre-calculates waveforms to ensure mathematical purity before hardware impairments are added by the radio.

---
### **Transfer Learning Integration**
Once the VDF dataset is generated, the `ai-rf` project will perform **Transfer Learning**:
1.  Load `best_resnet_v7.keras`.
2.  Freeze early convolutional layers.
3.  Fine-tune on the physical USRP data to adapt the "Simulated Brain" to the "Hardware Realities."
