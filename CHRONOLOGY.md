# Project Opal Vanguard: Progress Chronology

This document is the official "diary" of the Opal Vanguard Neural Receiver project. It tracks every technical decision, hurdle, and milestone on our journey to mastering AI-Native RF.

## [2026-03-12] Project Inception & Foundation
- **Mission Defined:** Launched Project Opal Vanguard to build a Neural Receiver for 24-modulation classification using RadioML 2018.01A and NVIDIA Sionna.
- **Tech Stack Finalized:** Standardized on Python 3.12, TensorFlow 2.21, and Sionna 1.2.1, targeting the NVIDIA RTX PRO 2000 Blackwell GPU (Laptop) and RTX 3080 Ti (Desktop).
- **Initial Documentation:** Created `FUNDAMENTALS.md` (Theory) and `TIMELINE.md` (Roadmap).
- **Repository Setup:** Initialized Git and linked to `https://github.com/mtmann213/ai-rf`.

## [2026-03-12] Infrastructure & Environment Setup
- **Dependency Management:** Created `requirements.txt`. Resolved "externally-managed-environment" errors using `--break-system-packages`.
- **GPU Optimization (Laptop):** Installed `nvidia-cudnn-cu12` and `nvidia-cublas-cu12` for the Blackwell architecture.
- **Privacy Scrub:** Removed all military affiliation references to ensure a neutral project environment.

## [2026-03-12] The "Sionna Integration" Battle
- **Import Fixes:** Identified correct Sionna paths: `sionna.phy.mapping` and `sionna.phy.channel`.
- **Constellation Pivot:** Bypassed Sionna's "multiple of 2" bit requirement for BPSK/OOK via QPSK fallbacks.
- **API Realignment:** Fixed `AWGN.call()` positional argument signature.
- **Physics Correction:** Implemented proper $N_0$ calculation from $E_b/N_0$.

## [2026-03-12] Architectural Evolution
- **Functional API Pivot:** Refactored from Keras subclassing to Functional API to fix `NotImplementedError` during model saving.
- **Milestone 1:** Completed 10-epoch training on synthetic data; verified pipeline with `benchmark_snr.py`.

## [2026-03-12] Phase 2: RadioML Integration & Data Factory
- **Memory Optimization:** Built `data_loader.py` using `h5py` for large-scale streaming.
- **The "Bad Gateway" Pivot:** Manufactured a local 7,200-sample synthetic RadioML dataset (`GOLD_XYZ_OSC.0001_1024.hdf5`) when official servers failed.
- **Milestone 2:** Completed 20-epoch training on local synthetic data (`opal_vanguard_v2.h5`).

## [2026-03-13] Phase 3: ResNet Upgrade & Desktop Distributed Compute
- **ResNet Implementation:** Upgraded the architecture to a Residual Network for deeper feature extraction.
- **Distributed Strategy:** Implemented **Docker/Docker-Compose** to move heavy training from the laptop to the **RTX 3080 Ti Desktop**.
- **WSL2 Integration:** Authored `DOCKER_GUIDE.md` for Windows/WSL2 deployment.

## [2026-03-13] Milestone 4: Numerical Stability & Performance Tuning
- **Hurdle (NaN Loss):** Encountered numerical instability on the 3080 Ti.
- **Stability Fix:** Implemented **Indestructible Normalization** in `data_loader.py` (scouring NaNs, clipping outliers, and L2 scaling).
- **Hurdle (Disk I/O):** Identified a massive WSL2 disk bottleneck causing 20+ hour training estimates.
- **The "Turbo" Fix:** Refactored the generator to read **contiguous chunks** (4096 samples) instead of random seeks, reducing estimated training time by an order of magnitude.
- **Observability:** Added "Heartbeat Dots" and `sys.stdout.flush()` for live progress tracking in Docker.

---
**Current Phase:** Phase 3 - ResNet Evolution (Stable Compute)
**Status:** 3080 Ti Node Active. Turbo-streaming engaged.
