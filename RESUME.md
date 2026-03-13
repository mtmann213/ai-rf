# Opal Vanguard: Mission Resume & Handoff (V7.1)

**Current Engine:** V7.1 Stability Patch (BatchNorm + Label Scrubbing)
**Distributed Node Status:** 
*   **Desktop (Compute):** 3080 Ti Active | Primary Training Node.
*   **Laptop (Development):** Blackwell Research | Logic & Docs Node (CPU Training only).

---

## 1. Primary Mission Sequence (Desktop/3080 Ti)
To resume the heavy training:
```bash
# 1. HARD RESET
git fetch origin main
git reset --hard origin/main

# 2. CLEAR PREVIOUS LOGS (If needed)
sudo rm -f training_log_v7.csv best_resnet_v7.keras

# 3. IGNITE
sudo docker compose up --build -d

# 4. MONITOR
sudo docker logs -f opal-vanguard-receiver
# Check for: "[V7.1] Event Horizon Engine Active."
```

## 2. Technical Stack Verification
*   `data_loader.py`: **V7.1.** Includes Tanh-Squashing and Label Scrubbing.
*   `resnet_opal_vanguard.py`: **V7.1.** Re-implemented BatchNorm for numerical resilience.
*   `train_resnet.py`: **V7.1.** Individual `clipnorm` optimization for float32 safety.

## 3. Hardware Constraint Note
Laptop GPU (RTX PRO 2000) is currently in a "Software Hold" status. It requires TensorFlow 2.22 for native Blackwell support. Do not attempt heavy training on the laptop until then.

---
**Tech Lead:** Mike Mann
**Diary Reference:** See `CHRONOLOGY.md` for the full technical evolution from V1.0 to V7.1.
