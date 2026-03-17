# Opal Vanguard: Mission Resume & Handoff (V9.2)

**Current Baseline:** V9.2 "Sovereign Eye" (Multi-Modal IQ + Polar)
**Distributed Node Status:** 
*   **Desktop (Compute):** 3080 Ti Active | Primary Training Node (Multi-Modal Phase)
    *   **Milestone:** Achieved **72.7% hardware accuracy** (V9.1 Specialist).
    *   **Task:** Training the dual-stream "Sovereign Eye" model to solve QAM density.
    *   **Strategy:** On-the-fly Polar Coordinate calculation (Amplitude/Phase) + I/Q.
*   **Laptop (Development):** Blackwell Research | Logic & Docs Node (CPU Research only).

---

## 1. Primary Mission Sequence (Desktop/3080 Ti)
To monitor the Sovereign Eye marathon:

```bash
# 1. MONITOR PROGRESS
docker logs -f $(docker ps -lq)

# 2. CHECK MULTI-MODAL LOGS
cat reports/v9_sovereign_log.csv | tail -n 20

# 3. VERIFY WEIGHTS
ls -lh models/v9_sovereign_checkpoint.keras
```

## 2. Active File Map (V9.2 Architecture)
*   **`src/train_v9_sovereign.py`**: **V9.2 Pilot.** Dual-input generator (IQ + Polar). Performs real-time Amplitude/Phase conversion.
*   **`src/resnet_lstm_polar_v9.py`**: **V9.2 "Sovereign" Brain.** Dual-branch 1D-ResNet fused into a Bi-LSTM ensemble.
*   **`src/train_v9_specialist.py`**: **V9.1 Legacy.** The weighted engine that broke the 70% barrier.
*   **`data/VDF_SPECTER_GOLDEN.h5`**: **Primary Nutrients.** 5GB USRP B205-mini hardware snapshots.

## 3. Operational Critical Path
1.  **Sovereign Convergence:** Allow V9.2 to run through its 50-epoch marathon to see if multi-modal features break the 75% barrier.
2.  **SDR Specialist Capture:** Prepare for a laptop capture session focused exclusively on high-gain 32QAM and 64QAM.
3.  **Phase 2 Demo:** Begin drafting the neural demodulator "heads" for the classified signal streams.

---
**Tech Lead:** Mike Mann
**Diary Reference:** See `CHRONOLOGY.md` for the full technical evolution from V1.0 to V9.2.
