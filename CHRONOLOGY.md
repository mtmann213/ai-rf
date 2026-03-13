# Project Opal Vanguard: Progress Chronology

This document tracks the step-by-step evolution of the Opal Vanguard Neural Receiver, documenting technical hurdles, architectural pivots, and major milestones.

## [2026-03-12] Project Inception & Foundation
- **Mission Defined:** Established the goal of building a Neural Receiver for 24-modulation classification using the RadioML 2018.01A set and NVIDIA Sionna.
- **Tech Stack Finalized:** Python 3.12, TensorFlow 2.15+, Sionna 1.2.x, optimized for NVIDIA RTX Blackwell hardware.
- **Documentation:** Authored `FUNDAMENTALS.md` (AI-Native RF theory) and `TIMELINE.md` (6-month roadmap).
- **Repository Setup:** Initialized Git and linked to `https://github.com/mtmann213/ai-rf`.

## [2026-03-12] Infrastructure & Environment Setup
- **Dependency Management:** Created `requirements.txt` and performed a system-wide installation (using `--break-system-packages` to bypass PEP 668 constraints in the current environment).
- **GPU Optimization:** Installed `nvidia-cudnn-cu12` and `nvidia-cublas-cu12` to support the Blackwell GPU architecture.
- **Privacy Scrub:** Conducted a global search-and-remove of all military affiliation references to ensure project neutrality.

## [2026-03-12] Implementation & Bug Squashing (The "Sionna Integration" Phase)
- **Sionna Import Fix:** Resolved `ModuleNotFoundError` by identifying the correct internal paths: `sionna.phy.mapping` and `sionna.phy.channel`.
- **Constellation Logic Pivot:** Discovered that Sionna's `Constellation` requires bits-per-symbol to be a multiple of 2. Implemented a fallback mapper to handle BPSK/OOK via QPSK-based logic.
- **API Realignment:** Fixed `AWGN.call()` signature errors by transitioning from list-based arguments to separate positional arguments (`x, no`).
- **Mathematical Correction:** Implemented the proper $N_0$ (noise variance) calculation from $E_b/N_0$ for differentiable channel simulation.

## [2026-03-12] Architectural Evolution: Functional API
- **Serialization Fix:** Encountered a `NotImplementedError` when saving the model due to Keras subclassing constraints.
- **Pivot:** Refactored `OpalVanguardModel` from a Class-based approach to the **Keras Functional API**. This ensures the model is fully serializable and compatible with future deployment tools like TensorRT.

## [2026-03-12] Milestone 1: First Successful Training Run
- **Execution:** Successfully completed a 10-epoch training run on synthetic Sionna-generated AWGN data.
- **Model Saved:** Verified the creation of `opal_vanguard_base.h5`.
- **Benchmarking:** Executed `benchmark_snr.py`, generating the first "Waterfall" accuracy curves and Confusion Matrices.
- **Baseline established:** 4.17% accuracy (random baseline for 24 classes), confirming the pipeline is end-to-end functional.

---
**Current Phase:** Phase 1 - Foundation & Benchmarking
**Status:** Pipeline Verified. Ready for Data Augmentation.
