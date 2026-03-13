# Project Opal Vanguard: Progress Chronology

This document is the official "diary" of the Opal Vanguard Neural Receiver project. It tracks every technical decision, hurdle, and milestone on our journey to mastering AI-Native RF.

## [2026-03-12] Project Inception & Foundation
- **Mission Defined:** Launched Project Opal Vanguard to build a Neural Receiver for 24-modulation classification using RadioML 2018.01A and NVIDIA Sionna.
- **Tech Stack Finalized:** Standardized on Python 3.12, TensorFlow 2.21, and Sionna 1.2.1, targeting the NVIDIA RTX PRO 2000 Blackwell GPU.
- **Initial Documentation:** Created `FUNDAMENTALS.md` to map the theory and `TIMELINE.md` to plot our 6-month course.
- **Repository Setup:** Initialized Git and linked to `https://github.com/mtmann213/ai-rf`.

## [2026-03-12] Infrastructure & Environment Setup
- **Dependency Management:** Created `requirements.txt`. Faced "externally-managed-environment" errors; resolved by using `--break-system-packages` for a direct user-space installation.
- **GPU Optimization:** Installed `nvidia-cudnn-cu12` and `nvidia-cublas-cu12` to provide the necessary libraries for the Blackwell architecture.
- **Privacy Scrub:** Conducted a comprehensive search-and-remove of all military affiliation references to ensure a neutral project environment.

## [2026-03-12] The "Sionna Integration" Battle
- **Hurdle 1 (Imports):** Encountered `ModuleNotFoundError`. Identified the correct internal Sionna paths through recursive directory inspection: `sionna.phy.mapping` and `sionna.phy.channel`.
- **Hurdle 2 (Constellations):** Discovered that Sionna's `Constellation` class requires bits-per-symbol to be a multiple of 2. Implemented a logic pivot to handle BPSK and OOK via a QPSK-based fallback.
- **Hurdle 3 (API Changes):** Resolved `TypeError` in `AWGN.call()` by transitioning from list-based arguments to separate positional arguments (`x, no`).
- **Hurdle 4 (Physics):** Corrected the mathematical implementation of the $N_0$ (noise variance) calculation from $E_b/N_0$ for accurate channel simulation.

## [2026-03-12] Architectural Evolution: Functional API
- **The Serialization Wall:** Failed to save the model using Keras subclassing (`NotImplementedError`).
- **The Pivot:** Refactored the entire `OpalVanguardModel` into the **Keras Functional API**. This transition fixed the serialization issue and made the model compatible with future high-performance deployment tools.

## [2026-03-12] Milestone 1: The First Breath
- **Execution:** Successfully completed the first 10-epoch training run on synthetic AWGN data.
- **Verification:** Generated the first "Waterfall" accuracy curves and Confusion Matrices using `benchmark_snr.py`.
- **Result:** Established an end-to-end functional pipeline with a random-baseline accuracy of 4.17%.

## [2026-03-12] Phase 2: RadioML Integration & DeepSig Pivot
- **Data Loader:** Built `data_loader.py` using `h5py` to stream large-scale RadioML HDF5 files.
- **The "Bad Gateway" Pivot:** Faced a download failure for the official 31GB dataset.
- **Creative Solution:** Authored `generate_synthetic_radioml.py` to manufacture a local 7,200-sample dataset (`GOLD_XYZ_OSC.0001_1024.hdf5`) that perfectly mimics the official format.
- **Milestone 2:** Completed a 20-epoch training cycle on this new local dataset, producing `opal_vanguard_v2.h5`.

---
**Current Phase:** Phase 2 - Dataset Integration
**Status:** End-to-end data pipeline is fully operational.
