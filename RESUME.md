# Opal Vanguard: Mission Resume & Handoff (V7.6.1)

**Current Baseline:** V7.6.1 Event Horizon (Pure Intelligence)
**Distributed Node Status:** 
*   **Desktop (Compute):** 3080 Ti Active | Primary Training Node (34% accuracy reached).
*   **Laptop (Development):** Blackwell Research | Logic & Docs Node (CPU Training only).

---

## 1. Primary Mission Sequence (Desktop/3080 Ti)
To pick up exactly where we left off with the most stable code:

```bash
# 1. HARD RESET (Force match GitHub)
git fetch origin main
git reset --hard origin/main

# 2. SCORCH WIPE (Clear all mathematical poison)
sudo rm -f *.keras *.csv *.h5 step_log_v7.csv

# 3. IGNITE
sudo docker compose up --build -d

# 4. VERIFY
sudo docker logs -f opal-vanguard-receiver
# Check for: "[V7.6] Event Horizon Engine Active."
```

## 2. Active File Map
*   `data_loader.py`: **V7.6.1 Engine.** Bypasses corrupted HDF5 labels with **Mathematical Label Reconstruction**.
*   `resnet_opal_vanguard.py`: **V7.4 Vessel.** 1D-ResNet with **Time-Axis LayerNormalization** (preserving phase physics).
*   `train_resnet.py`: **V7.6 Pilot.** High-energy optimizer (LR 0.0005) + **Softmax Anchor** + Real-Time Step Logging.

## 3. Hardware Constraint Note
Laptop GPU (RTX PRO 2000) is currently in a "Software Hold" status. It requires TensorFlow 2.22 for native Blackwell support. Do not attempt heavy training on the laptop until then.

---
**Tech Lead:** Mike Mann
**Diary Reference:** See `CHRONOLOGY.md` for the full technical evolution from V1.0 to V7.2.