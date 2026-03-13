# Opal Vanguard: Mission Resume & Handoff

**Current Baseline:** V7.1 Event Horizon (Stability Patch)
**Status:** Architecture finalized for absolute stability. Ready for heavy training on RTX 3080 Ti.

---

## 1. Restart Sequence (Desktop/3080 Ti)
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
*   `data_loader.py`: **V7.1 Engine.** Uses Soft-Clip `x/(1+|x|)` scaling + Label Scrubbing for absolute stability.
*   `resnet_opal_vanguard.py`: **V7.1 Vessel.** 1D-ResNet with `BatchNormalization` for zero-variance protection.
*   `train_resnet.py`: **V7.1 Pilot.** Double-clip optimizer (`clipnorm` + `clipvalue`) with XLA disabled.

## 3. Mission Milestones
*   **Next Goal:** Complete 50 Epochs on the 2018.01A dataset.
*   **Success Indicator:** Loss starts at ~3.17 and smoothly decreases; accuracy climbs beyond 5%.

---
**Tech Lead:** Mike Mann
**Diary Reference:** See `CHRONOLOGY.md` for the technical history of our stability pivots.
