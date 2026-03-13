# Project Opal Vanguard: Progress Chronology

... (existing entries) ...

## [2026-03-13] Milestone 7: The "Nuclear-Grade" Logits Fix
- **Hurdle (Softmax Explosion):** The ResNet experienced a catastrophic negative loss collapse (`-1.46e18`) even with gentle learning rates and Z-score scaling. Diagnosed as a `-log(0)` explosion in the `categorical_crossentropy` function caused by the final Softmax layer.
- **Solution 1 (Logits Loss):** Stripped the `softmax` activation from the final layer of `resnet_opal_vanguard.py` to output raw, unbounded "logits." Updated `train_resnet.py` to use `CategoricalCrossentropy(from_logits=True)`, allowing TensorFlow's stable C++ backend to handle the math safely.
- **Solution 2 (He Normal):** Applied `kernel_initializer='he_normal'` across all Conv1D and Dense layers to ensure the model's initial weights are mathematically scaled for ReLU activations, preventing large initial gradients.

---
**Current Phase:** Phase 3 - ResNet Evolution (Stable Compute)
**Status:** Nuclear-grade numerical stability achieved. Ready for true training.
