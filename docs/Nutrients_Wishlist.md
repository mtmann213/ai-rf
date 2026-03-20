# Opal Vanguard: Data & Model Nutrients Wishlist

This document tracks high-value external intelligence sources (datasets and pre-trained weights) identified for future Knowledge Transfusion and model expansion.

---

## I. Pre-trained Foundation Models (Weights)

### 1. SpectrumFM (The 100-Modulation Expert)
- **Source:** Hugging Face (`Singingkettle/SpectrumFM`)
- **Type:** Foundation Model / Masked Autoencoder.
- **Goal:** Transfuse the "Universal Language of Radio" into the Sovereign brain.

### 2. Zenodo RAN (Residual Aggregated Network)
- **Source:** [DOI: 10.5281/zenodo.10963509](https://doi.org/10.5281/zenodo.10963509)
- **Contents:** Official code and weights for Shuo Chang's high-performance AMC papers.
- **Goal:** Benchmarking against our Sovereign Eye architecture.

---

## II. External Datasets (Nutrients)

### 1. CSRD2025 (ChangShuo Radio Data)
- **Source:** [Singingkettle/ChangShuoRadioData](https://github.com/Singingkettle/ChangShuoRadioData)
- **Scale:** 200 TB (Synthetic with site-specific Ray Tracing).
- **Diversity:** 100 modulation types.
- **Strategy:** Use the generation framework to create "Surgical Chunks" (100GB) for local training.

### 2. DeepSig RadioML 2016.10B
- **Source:** DeepSig.io
- **Role:** The "Clean Baseline" for high-SNR mathematical grounding.

### 3. HisarMod2019.1
- **Source:** MATLAB / Research Community.
- **Role:** High-fidelity multipath fading models (Rayleigh/Rician) for field-hardening.

---

## III. Tactical Integration Path
1.  **Benchmarking:** Run the Zenodo weights through `src/generate_all_confusion_matrices.py`.
2.  **Transfusion:** Use `src/weight_transfusion.py` to migrate features from SpectrumFM into our CLDNN ensemble.
3.  **Synthesis:** Use the CSRD framework to generate 10k samples per class for our top 100 target modulations.

---
**Status:** Research Phase Active.
**Lead Scout:** Gemini CLI
