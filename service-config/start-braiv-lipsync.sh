#!/bin/bash

# Define system-wide paths
CONDA_PATH="/opt/miniconda"
CONDA_ENV_NAME="braiv-pipeline"
PROJECT_DIR="/opt/braiv-pipeline"
MODEL_DIR="/var/lib/braiv/models"

echo "🚀 Starting script..."

# Ensure an environment is provided
if [ -z "$1" ]; then
    echo "❌ Error: No environment specified. Usage: $0 {prod|preview|dev}"
    exit 1
fi

ENV="$1"

# Set the corresponding environment setup file
case "$ENV" in
    prod)    ENV_FILE="setenv-prod.sh" ;;
    preview) ENV_FILE="setenv-preview.sh" ;;
    dev)     ENV_FILE="setenv-dev.sh" ;;
    *)
        echo "❌ Error: Invalid environment '$ENV'. Use {prod|preview|dev}"
        exit 1
        ;;
esac

echo "🚀 Setting up environment: $ENV (Using $ENV_FILE)"

# Ensure the environment file exists before sourcing it
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: Environment file '$ENV_FILE' not found!"
    exit 1
fi

# Initialize Conda (if necessary)
echo "✅ Conda initialized..."
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Activate the Conda environment
conda activate "$CONDA_ENV_NAME"
echo "✅ Conda environment activated..."

# Export the Python path from the Conda environment
export PATH="$CONDA_PATH/envs/$CONDA_ENV_NAME/bin:$PATH"

# Set environment variables for model storage
export MODEL_SHARED_DIR="$MODEL_DIR"
export PYANNOTE_CACHE="$MODEL_DIR"
export HF_HOME="$MODEL_DIR"
export TORCH_HOME="$MODEL_DIR"
echo "✅ Model locations set to $MODEL_DIR"

# Change to the project directory
cd "$PROJECT_DIR" || { echo "❌ Failed to change directory to $PROJECT_DIR"; exit 1; }
echo "✅ Changed to project directory..."

# Source the selected environment setup script
if ! source "$ENV_FILE"; then
    echo "❌ Error: Failed to source environment setup file '$ENV_FILE'"
    exit 1
fi
echo "✅ Environment setup script '$ENV_FILE' sourced successfully..."

# Run the Python script using the full path
"$CONDA_PATH/envs/$CONDA_ENV_NAME/bin/python" app/main.py
echo "✅ Python script executed..."
