# Opal Vanguard: Mission Resume & Handoff

**Status:** Phase 3 IN PROGRESS | Awaiting 50-epoch completion on RTX 3080 Ti.

---

## 1. Active File Map
*   `data_loader.py`: **The Engine.** Handles HDF5 streaming, normalization, and chunking.
*   `train_resnet.py`: **The Pilot.** Main training script with resumability.
*   `resnet_opal_vanguard.py`: **The Vessel.** Defines the 1D-ResNet architecture.
*   `DOCKER_GUIDE.md`: **The Manual.** How to run on the 3080 Ti desktop.

## 2. Desktop (RTX 3080 Ti) Commands
Use these commands on your Windows Desktop (WSL2) to restart the heavy training:
```bash
# Update the code
git pull origin main

# Start the container (Builds if necessary)
sudo docker compose up --build -d

# Watch the progress (Look for 'Heartbeat Dots')
sudo docker logs -f opal-vanguard-receiver
```

## 3. Laptop (RTX Blackwell) Commands
Use these for development or small-scale testing:
```bash
# Fix library paths for local Python
./run_vanguard.sh
```

## 4. Troubleshooting: GPU Fix (WSL2)
If Docker doesn't see the 3080 Ti, run this in your Desktop terminal:
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
sudo service docker restart
```

---
**Tech Lead:** Mike Mann
**Diary Reference:** See `CHRONOLOGY.md` for the technical history.
