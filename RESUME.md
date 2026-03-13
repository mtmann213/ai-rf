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
## 4. Launch Heavy Training
*   **Recommended (Desktop/Windows):** Install **Docker Desktop**, enable WSL2 integration, and run:
    ```bash
    sudo docker compose up --build -d
    ```
*   **Local (Laptop):** `python3 train_resnet.py` (CPU only until Blackwell drivers mature).


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

## 7. Native Docker GPU Fix (WSL2)
If Docker Desktop is NOT installed and you are using native Linux Docker, run this to enable the GPU:
1. **Config:** 
```bash
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
```
2. **Restart:** `sudo service docker restart`
3. **Run:** `sudo docker compose up --build -d`
