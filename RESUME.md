# Opal Vanguard: Mission Resume & Handoff (V7.2)

**Current Baseline:** V7.2 Event Horizon (Label Alignment)
**Distributed Node Status:** 
*   **Desktop (Compute):** 3080 Ti Active | Primary Training Node.
*   **Laptop (Development):** Blackwell Research | Logic & Docs Node (CPU Training only).

---

## 1. Primary Mission Sequence (Desktop/3080 Ti)
To pick up exactly where we left off with the most stable code:

```bash
# 1. HARD RESET (Force match GitHub)
git fetch origin main
git reset --hard origin/main

# 2. SCORCH WIPE (Clear all mathematical poison)
sudo rm -f *.keras *.csv *.h5

# 3. IGNITE
sudo docker compose up --build -d

# 4. VERIFY
sudo docker logs -f opal-vanguard-receiver
# Check for: "[V7.0] Event Horizon Engine Engaged."
```

## 2. Active File Map
*   `data_loader.py`: **V7.2 Engine.** Uses Soft-Clip scaling + Label Scrubbing + Correct 'Original' Class Mapping.
*   `resnet_opal_vanguard.py`: **V7.1 Vessel.** 1D-ResNet with `BatchNormalization` for zero-variance protection.
*   `train_resnet.py`: **V7.1 Pilot.** Double-clip optimizer (`clipnorm` + `clipvalue`) with XLA disabled.

## 3. Hardware Constraint Note
Laptop GPU (RTX PRO 2000) is currently in a "Software Hold" status. It requires TensorFlow 2.22 for native Blackwell support. Do not attempt heavy training on the laptop until then.

---
**Tech Lead:** Mike Mann
**Diary Reference:** See `CHRONOLOGY.md` for the full technical evolution from V1.0 to V7.2.