# Opal Vanguard: Session Resume & Handoff

**Last Updated:** 2026-03-12
**Current Milestone:** Phase 2 Complete / Phase 3 Initiated
**Status:** Pipeline verified on synthetic data; awaiting 31GB RadioML 2018.01A dataset.

---

## 1. Project Context
*   **Mission:** Neural Receiver for 24-modulation classification.
*   **Key Architectures:** 
    *   `OpalVanguard`: Initial 3-layer Conv1D CNN (Functional API).
    *   `OpalVanguard_ResNet`: Advanced Residual Network with skip connections (Current Lead).
*   **Hardware:** 
    *   Primary: Laptop (NVIDIA RTX PRO 2000 Blackwell).
    *   Target: Desktop (NVIDIA RTX 3080 Ti via WSL2/Docker).

## 2. Active File Map
| File | Role |
| :--- | :--- |
| `data_loader.py` | Memory-efficient HDF5 streaming for RadioML datasets. |
| `resnet_opal_vanguard.py` | Definition of the ResNet architecture. |
| `train_resnet.py` | Main training script with EarlyStopping and Checkpointing. |
| `generate_synthetic_radioml.py` | Local data factory (Mimics 31GB dataset structure). |
| `benchmark_snr.py` | Evaluation tool for accuracy curves and Confusion Matrices. |
| `Dockerfile` / `docker-compose.yml` | Containerization for Windows/WSL2 portability. |

## 3. Immediate Next Steps
1.  **Dataset Acquisition:** Wait for `GOLD_XYZ_OSC.0001_1024.hdf5` (31GB) to finish downloading.
2.  **Move Dataset:** Place the downloaded `.hdf5` file in the root directory.
3.  **Launch Heavy Training:**
    *   *Local:* `python3 train_resnet.py`
    *   *Docker (Desktop):* `docker-compose up --build`

## 4. Technical Debt / Open Issues
*   **GPU Detection:** TensorFlow 2.21 is currently defaulting to CPU on the Blackwell Laptop due to CUDA library pathing. Use the `LD_LIBRARY_PATH` export found in `CHRONOLOGY.md` to troubleshoot.
*   **Baseline:** Current accuracy is ~8% on 7,200 synthetic samples. Expect >90% once trained on the full 31GB set.

## 5. Resume Commands
```bash
# Verify the environment
pip install -r requirements.txt --break-system-packages

# Run the local data generator if starting fresh
python3 generate_synthetic_radioml.py

# Train the advanced ResNet model
python3 train_resnet.py

# Generate performance visualizations
python3 benchmark_snr.py
```

## 6. Troubleshooting: GPU Capability Error
If you see `could not select device driver "nvidia" with capabilities: [[gpu]]`, run these on the desktop's WSL2 terminal:
1. **Add Repo:** `curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list`
2. **Install:** `sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit`
3. **Configure:** `sudo nvidia-ctk runtime configure --runtime=docker`
4. **Restart:** Restart Docker Desktop on Windows.
