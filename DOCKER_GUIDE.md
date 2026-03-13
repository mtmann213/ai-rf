# Opal Vanguard: Desktop (3080 Ti) Deployment Guide

This guide ensures your RTX 3080 Ti desktop is correctly configured as the primary compute node for the Neural Receiver.

## Step 1: Hardware & Drivers
1.  Install the latest **NVIDIA Studio or Game Ready Drivers** on Windows.
2.  Install **Docker Desktop** and check "Use the WSL 2 based engine" in Settings.
3.  Open Docker Settings -> Resources -> WSL Integration and enable **Ubuntu**.

## Step 2: Native WSL File Placement
For a 21GB dataset, disk speed is critical. **Do not use the Windows C: drive.**
1.  Open your Ubuntu terminal.
2.  Clone the project: `git clone https://github.com/mtmann213/ai-rf.git`
3.  Move the dataset to: `\\wsl.localhost\Ubuntu\home\<user>\ai-rf\2018_01A\` using Windows File Explorer.

## Step 3: Ignition
Run these commands in your Ubuntu terminal:
```bash
# 1. Start the engine
sudo docker compose up --build -d

# 2. Monitor the heartbeats
sudo docker logs -f opal-vanguard-receiver
```

## Step 4: Success Indicators
*   **GPU Detected:** You see `Created device /device:GPU:0 ... NVIDIA GeForce RTX 3080 Ti`.
*   **Data Flowing:** You see dots (`..........`) appearing quickly.
*   **Learning:** You see `Epoch 1/50` followed by a progress bar and a numerical `loss` value (not `nan`).

---
**Troubleshooting:** 
If Docker says `could not select device driver "nvidia"`, run the "Nuclear Fix" found in `RESUME.md`.
