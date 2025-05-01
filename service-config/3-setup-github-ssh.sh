#!/bin/bash

set -e  # Exit on error

# Define paths (move SSH key to ~/.ssh/)
SSH_DIR="$HOME/.ssh"
SSH_KEY_PATH="$SSH_DIR/github_service_key"
SSH_CONFIG_PATH="$SSH_DIR/config"

echo "🚀 Setting up SSH key for GitHub authentication..."

# Ensure the SSH directory exists and has correct permissions
mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

# Generate SSH key if it does not exist
if [ ! -f "$SSH_KEY_PATH" ]; then
    echo "🔑 Generating new SSH key..."
    ssh-keygen -t rsa -b 4096 -f "$SSH_KEY_PATH" -N ""
    chmod 600 "$SSH_KEY_PATH"
    chmod 644 "${SSH_KEY_PATH}.pub"
else
    echo "✅ SSH key already exists: $SSH_KEY_PATH"
fi

# Ensure SSH config is set up correctly
if [ ! -f "$SSH_CONFIG_PATH" ]; then
    echo "📝 Creating SSH config file..."
    echo -e "Host github.com\n    IdentityFile $SSH_KEY_PATH\n    StrictHostKeyChecking no\n    User git" > "$SSH_CONFIG_PATH"
    chmod 600 "$SSH_CONFIG_PATH"
else
    echo "✅ SSH configuration for GitHub already exists."
fi

# Ensure correct permissions
echo "🔧 Fixing permissions..."
chmod 600 "$SSH_KEY_PATH"

# Ensure SSH agent is running and add the key
echo "🔄 Restarting SSH agent..."
eval "$(ssh-agent -s)"

# Add the key without sudo (since it's now user-owned)
ssh-add "$SSH_KEY_PATH" || echo "⚠️ Failed to add SSH key to agent. You may need to run 'ssh-add $SSH_KEY_PATH' manually."

# Display public key for GitHub setup
echo ""
echo "📢 Follow these steps to add the SSH key as a **Deploy Key** in GitHub:"
echo "-------------------------------------------------------------"
echo "1️⃣ Copy the following public key:"
echo "-------------------------------------------------------------"
cat "${SSH_KEY_PATH}.pub"
echo "-------------------------------------------------------------"
echo "2️⃣ Go to your repository on GitHub: https://github.com/braiv/braiv-pipeline"
echo "3️⃣ Click **Settings** in the repository menu."
echo "4️⃣ In the left sidebar, click **Deploy Keys**."
echo "5️⃣ Click **Add deploy key**."
echo "6️⃣ In the **Title** field, provide a meaningful name (e.g., 'GCP VM Deploy Key')."
echo "7️⃣ In the **Key** field, paste the copied public key."
echo "8️⃣ **(Optional)** Check **Allow write access** if this key needs to push changes."
echo "9️⃣ Click **Add key**."
echo ""
echo "🚀 Once this step is completed the deploy key will be linked to the repository and will allow automated access from your server!"
echo ""


echo "🔍 Testing GitHub SSH authentication..."
SSH_TEST_OUTPUT=$(ssh -o StrictHostKeyChecking=no -o BatchMode=yes -i "$SSH_KEY_PATH" -T git@github.com 2>&1 || true)

if [[ "$SSH_TEST_OUTPUT" == *"successfully authenticated"* ]]; then
    echo "✅ SSH connection to GitHub is working!"
else
    echo "❌ SSH connection to GitHub failed! Make sure the key is added."
    echo "🔍 Debug Output:"
    echo "$SSH_TEST_OUTPUT"
fi

echo "✅ SSH key setup complete!"
