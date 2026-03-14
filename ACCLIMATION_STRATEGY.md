# Strategy: Neural Model Acclimation (NMA)
**Objective:** Transition the "Simulated Brain" (trained on RadioML 2018.01A) to the "Physical World" (VDF Dataset) without triggering Catastrophic Forgetting.

## 1. The Multi-Stage Transfer Logic
We will not retrain the model from scratch. Instead, we will use a tiered approach to introduce the physical hardware characteristics.

### Stage 1: The Zero-Shot Baseline
*   **Action:** Run the existing model on the new VDF Pilot data *without any training*.
*   **Goal:** Establish a baseline. If the model gets 20% on the USRP data immediately, we know the "General Physics" it learned are valid.

### Stage 2: Feature-Locked Fine-Tuning
*   **Action:** Freeze the first 75% of the ResNet layers (the convolutional feature extractors).
*   **Logic:** These layers have already mastered "how a wave looks." We only want to update the "Classification Head" to recognize the specific DC offset and IQ imbalance of your B205-minis.
*   **Training:** 5-10 epochs at a very low learning rate (`0.00001`).

### Stage 3: Full-Stack Thaw (Final Polish)
*   **Action:** Unfreeze all layers and train on a **Mixed Dataset**.
*   **The Mix:** 80% VDF Data + 20% RadioML 2018.01A Data.
*   **Purpose:** The 20% "Old" data acts as an anchor. It prevents the model from "forgetting" generic signal patterns while it masters the physical ones.

## 2. Technical Concerns & Safeguards

### A. Preventing Catastrophic Forgetting
If validation accuracy on the *original* 2018 dataset drops by more than 5%, we must stop and increase the "Anchor" percentage of the old data in the training mix.

### B. Gain-Normalization Alignment
The VDF dataset must use the exact same **Soft-Clip** scaling (`x / (1 + |x|)`) as the 2018 dataset. If the scaling is different, the model will be "blinded" by the change in input magnitude.

### C. Learning Rate Safety
Never use a high learning rate during acclimation. We are "nudging" a mature brain, not teaching a baby. High learning rates will "smear" the carefully learned convolutional filters.

## 3. Evaluation Metrics
We will track two waterfalls simultaneously:
1.  **Original Waterfall:** Accuracy vs. SNR on simulated data.
2.  **Physical Waterfall:** Accuracy vs. TX Gain on USRP data.
*Success is defined as maintaining the Original Waterfall while improving the Physical Waterfall.*
