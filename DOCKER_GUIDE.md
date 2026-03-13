# Opal Vanguard: Docker Desktop GPU Guide (Windows + 3080 Ti)

Follow this guide to transition the Neural Receiver training from your laptop to your high-performance desktop compute node.

## Phase 1: Windows Environment Setup
1.  **NVIDIA Drivers:** Ensure you have the latest **Game Ready** or **Studio** drivers installed on Windows. (Version 530+ recommended).
2.  **WSL2:** Open PowerShell as Administrator and run:
    ```powershell
    wsl --install
    ```
    *Note: If already installed, ensure it is updated via `wsl --update`.*
3.  **Docker Desktop:**
    *   Download and install from [docker.com](https://www.docker.com/products/docker-desktop/).
    *   **Crucial Setting:** During installation, ensure **"Use the WSL 2 based engine"** is checked.
    *   **WSL Integration:** Open Docker Desktop Settings -> Resources -> WSL Integration. Toggle **ON** for your specific Ubuntu distribution.

## Phase 2: Project Deployment (Inside WSL2)
Open your Ubuntu terminal on the desktop and execute:

1.  **Clone the Mission:**
    ```bash
    git clone https://github.com/mtmann213/ai-rf.git
    cd ai-rf
    ```
2.  **Move the Dataset (Performance Warning):**
    *   **DO NOT** run the training from a Windows path (e.g., `/mnt/c/...`). It will be 5x slower.
    *   Copy the 21GB dataset into your native Linux home folder:
    ```bash
    # Create the folder
    mkdir -p 2018_01A
    # Use Windows File Explorer to move the file to:
    # \\wsl.localhost\Ubuntu\home\<your_user>\ai-rf\2018_01A\
    ```

## Phase 3: Launching the Mission
In your desktop's WSL2 terminal, run:

1.  **Build and Launch:**
    ```bash
    docker compose up --build -d
    ```
2.  **Monitor GPU Detection:**
    ```bash
    docker logs -f opal-vanguard-receiver
    ```
    *Look for:* `Num GPUs Available: 1` and the start of **Epoch 1/50**.

## Phase 4: Syncing Progress
Once training completes (or if you stop it):
1.  The model (`best_resnet.keras`) and logs (`training_log.csv`) will be saved in your folder automatically via Docker Volumes.
2.  Commit and push them back to GitHub to resume on your laptop:
    ```bash
    git add best_resnet.keras training_log.csv
    git commit -m "Opal Vanguard: Heavy training complete on 3080 Ti"
    git push origin main
    ```

---
**Lead SDR Architect:** Mike Mann
**Mission Status:** Ready for High-Performance Compute.
