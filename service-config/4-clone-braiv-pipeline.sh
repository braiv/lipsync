#!/bin/bash

set -e  # Exit immediately if a command fails

# Ensure environment parameter is provided
if [ -z "$1" ]; then
    echo "❌ Error: Missing environment parameter."
    echo "Usage: $0 [prod|preview|dev]"
    exit 1
fi

ENV=$1
REPO_URL="git@github.com:braiv/braiv-pipeline.git"
CLONE_DIR="/opt/braiv-pipeline"
MODEL_DIR="/var/lib/braiv/models"

# Determine branch based on environment
case "$ENV" in
    prod) BRANCH="main" ;;
    preview) BRANCH="preview" ;;
    dev) BRANCH="develop" ;;
    *)
        echo "❌ Error: Invalid environment. Use 'prod', 'preview', or 'dev'."
        exit 1
        ;;
esac

echo "🚀 Cloning Braiv Pipeline ($BRANCH branch) into $CLONE_DIR..."

# Check if directory exists but is NOT a git repository
if [ -d "$CLONE_DIR" ] && [ ! -d "$CLONE_DIR/.git" ]; then
    echo "⚠️ $CLONE_DIR exists but is not a Git repository. Removing and re-cloning..."
    sudo rm -rf "$CLONE_DIR"
fi

# If repository doesn't exist, clone it
if [ ! -d "$CLONE_DIR" ]; then
    echo "⬇️ Cloning repository..."
    sudo mkdir -p "$CLONE_DIR"
    sudo chown $USER:$USER "$CLONE_DIR"
    git clone --branch "$BRANCH" "$REPO_URL" "$CLONE_DIR"
else
    echo "🔄 Repository already exists. Pulling latest changes..."
    cd "$CLONE_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
fi

echo "Set /opt/braiv-pipeline as safe directory in git"
sudo git config --global --add safe.directory /opt/braiv-pipeline

echo "✅ Successfully updated $CLONE_DIR to branch $BRANCH."


# Ensure model directory exists
sudo mkdir -p "$MODEL_DIR"

# Ensure chown runs only if USER is set and directory ownership actually needs changing
if [ -n "$USER" ] && [ -d "$MODEL_DIR" ]; then
    CURRENT_OWNER=$(stat -c "%U:%G" "$MODEL_DIR")
    EXPECTED_OWNER="$USER:$USER"

    if [ "$CURRENT_OWNER" != "$EXPECTED_OWNER" ]; then
        if sudo chown "$USER:$USER" "$MODEL_DIR"; then
            echo "✅ Ownership of $MODEL_DIR changed to $USER:$USER"
        else
            echo "⚠️ Failed to change ownership of $MODEL_DIR"
        fi
    else
        echo "✅ Ownership of $MODEL_DIR is already correct ($CURRENT_OWNER)"
    fi
else
    echo "⚠️ Skipping chown for $MODEL_DIR: USER is not set or directory does not exist"
fi
