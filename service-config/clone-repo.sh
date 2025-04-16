# Clone the repo and go into it
REPO_DIR="$HOME/braiv-lipsync"
if [ ! -d "$REPO_DIR" ]; then
    echo "📥 Cloning Braiv Lipsync repo..."
    git clone https://github.com/braiv/lipsync.git "$REPO_DIR"
else
    echo "🔁 Repo already exists at $REPO_DIR. Pulling latest changes..."
    cd "$REPO_DIR"
    git pull
fi
cd "$REPO_DIR"

# Optionally install Python requirements
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
fi
