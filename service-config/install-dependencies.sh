#!/bin/bash

set -e  # Exit on error

echo "🚀 Starting setup for braiv-lipsync on Ubuntu 22.04 GCP VM..."

# Update package list and upgrade system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies (SoX, FFmpeg, NVIDIA tooling)
echo "🔧 Installing system dependencies..."
sudo apt install -y --no-install-recommends \
    build-essential yasm cmake pkg-config git nasm \
    libass-dev libfreetype6-dev libmp3lame-dev libopus-dev libvorbis-dev \
    libvpx-dev libx264-dev libx265-dev libnuma-dev libunistring-dev \
    libfdk-aac-dev libssl-dev python3-pip python3-setuptools \
    sox libsox-fmt-all ffmpeg \
    nvidia-utils-550 nvidia-cuda-toolkit

# Ensure NVML user-space libraries match the kernel module
echo "🔍 Checking NVIDIA kernel driver version..."
KERNEL_NVIDIA_VERSION=$(modinfo nvidia | grep '^version:' | awk '{print $2}')

echo "🧠 Checking existing NVML library version..."
USER_LIB_VERSION=$(strings /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 2>/dev/null | grep -Eo '550\.[0-9]+(\.[0-9]+)?' | head -n1)

if [ "$KERNEL_NVIDIA_VERSION" == "$USER_LIB_VERSION" ]; then
    echo "✅ NVML libraries already match the kernel driver version ($KERNEL_NVIDIA_VERSION). Skipping NVIDIA installer."
else
    echo "🌐 Downloading NVIDIA .run installer for $KERNEL_NVIDIA_VERSION..."
    wget -q "https://us.download.nvidia.com/XFree86/Linux-x86_64/${KERNEL_NVIDIA_VERSION}/NVIDIA-Linux-x86_64-${KERNEL_NVIDIA_VERSION}.run" -O "nvidia-${KERNEL_NVIDIA_VERSION}.run"
    chmod +x "nvidia-${KERNEL_NVIDIA_VERSION}.run"

    echo "📦 Installing matching NVIDIA user-space libraries (no kernel module)..."
    sudo ./nvidia-${KERNEL_NVIDIA_VERSION}.run \
        --silent \
        --accept-license \
        --no-kernel-module \
        --no-drm \
        --no-opengl-files

    echo "🧹 Cleaning up NVIDIA installer..."
    rm "nvidia-${KERNEL_NVIDIA_VERSION}.run"
fi

# Confirm NVML is working
echo "✅ Verifying NVIDIA stack with nvidia-smi..."
if ! nvidia-smi; then
    echo "❌ NVML initialization failed — possible driver mismatch or GPU issue."
    exit 1
fi

# Verify FFmpeg NVENC support
echo "🎥 Verifying FFmpeg NVENC support..."
ffmpeg -encoders | grep nvenc || echo "❌ NVENC not found!"
ffmpeg -hwaccels || echo "❌ No hardware acceleration found!"

# Install Miniconda
CONDA_DIR="/opt/miniconda"
CONDA_BIN="$CONDA_DIR/bin/conda"

echo "🐍 Installing Miniconda..."
if [ ! -d "$CONDA_DIR" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    sudo bash miniconda.sh -b -p "$CONDA_DIR"
    rm miniconda.sh
    echo "✅ Miniconda installed at $CONDA_DIR"
else
    echo "✅ Miniconda already present. Skipping installation."
fi

# Ensure correct ownership
sudo chown -R "$USER:$USER" "$CONDA_DIR"

# Initialize Conda if not already initialized
if ! grep -q 'conda initialize' ~/.bashrc; then
    echo "🔧 Initializing Conda for persistent shell use..."
    "$CONDA_BIN" init bash
else
    echo "🔁 Conda already initialized in bash profile. Skipping."
fi

# Activate Conda in this session
export PATH="$CONDA_DIR/bin:$PATH"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda init

# Verify Conda works
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not available in this session."
    exit 1
else
    echo "✅ Conda is available and working!"
fi

# Create or update the Conda environment
ENV_NAME="braiv-lipsync"
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "📦 Creating Conda environment: $ENV_NAME..."
    conda create -y -n "$ENV_NAME" python=3.10
else
    echo "✅ Conda environment $ENV_NAME already exists. Skipping creation."
fi

# Activate environment
echo "🔄 Activating $ENV_NAME environment..."
conda activate "$ENV_NAME"

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install CUDA runtime and cuDNN
echo "📦 Installing CUDA runtime and cuDNN..."
conda install -y conda-forge::cuda-runtime=12.8.0 conda-forge::cudnn=9.7.1.26

# Install TensorRT
echo "📦 Installing TensorRT..."
pip install tensorrt==10.8.0.43 --extra-index-url https://pypi.nvidia.com

echo "🎉 Setup complete! Use 'conda activate $ENV_NAME' to start working."

# Optional: Reboot to ensure NVIDIA stack is fully active
echo "🔁 Rebooting system to finalize driver and kernel changes..."
sudo reboot
