#!/bin/bash

# Exit on any error
set -e

# Assume you are already in the right conda environment: (braiv-lipsync)

# Install ffmpeg
conda install -y -c conda-forge ffmpeg

# Install Python dependencies
pip install -r latentsync-requirements.txt

# Install OpenCV runtime dependency for GUI support (for cv2.imshow)
sudo apt-get update
sudo apt-get install -y libgl1

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Download model checkpoints using huggingface-cli
huggingface-cli download ByteDance/LatentSync-1.5 whisper/tiny.pt --local-dir checkpoints --local-dir-use-symlinks False
huggingface-cli download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir checkpoints --local-dir-use-symlinks False