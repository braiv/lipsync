#!/bin/bash

set -e  # Exit immediately if any command fails

# Ensure an environment is provided
if [ -z "$1" ]; then
    echo "❌ Error: No environment specified. Usage: $0 {prod|preview|dev}"
    exit 1
fi

ENV="$1"
REAL_USER="${SUDO_USER:-$USER}"

echo "🚀 Setting up systemd services for Braiv Lipsync (Environment: $ENV)..."

# Define service files
UPDATE_SERVICE="/etc/systemd/system/update-braiv-lipsync.service"
MAIN_SERVICE="/etc/systemd/system/braiv-lipsync.service"

# Ensure the script is run as root
if [ "$(id -u)" -ne 0 ]; then
    echo "❌ Error: This script must be run as root. Use sudo."
    exit 1
fi

# Set up logging directory
echo "🔧 Setting up logging directory..."
touch /var/log/braiv-lipsync.log
chown "$REAL_USER:$REAL_USER" /var/log/braiv-lipsync.log

# Create or update the update service (runs on startup to pull latest code)
echo "🔧 Creating or updating $UPDATE_SERVICE..."
cat <<EOF > "$UPDATE_SERVICE"
[Unit]
Description=Update Braiv Lipsync on Startup
After=network.target
Before=braiv-lipsync.service

[Service]
Type=oneshot
ExecStart=/bin/bash /home/cody_braiv_co/braiv-lipsync/service-config/update-braiv-lipsync-new.sh
WorkingDirectory=/home/cody_braiv_co/braiv-lipsync
User=$REAL_USER
Group=$REAL_USER
RemainAfterExit=yes
SuccessExitStatus=0 1

[Install]
WantedBy=multi-user.target
EOF

# Create or update the main lipsync service
echo "🔧 Creating or updating $MAIN_SERVICE..."
cat <<EOF > "$MAIN_SERVICE"
[Unit]
Description=Braiv Lipsync Main Service
After=update-braiv-lipsync.service
Requires=update-braiv-lipsync.service

[Service]
ExecStart=/bin/bash /home/cody_braiv_co/braiv-lipsync/service-config/start-braiv-lipsync-new.sh dev
WorkingDirectory=/home/cody_braiv_co/braiv-lipsync
User=$REAL_USER
Group=$REAL_USER
Restart=always
RestartSec=10
Environment="PATH=/opt/miniconda/bin:/opt/miniconda/envs/braiv-lipsync/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd to apply changes
echo "🔄 Reloading systemd daemon..."
systemctl daemon-reload

# Enable update service (runs on startup))
echo "✅ Enabling update-braiv-lipsync.service..."
systemctl enable update-braiv-lipsync.service

# Start update service and check for errors
# Start update service and check for errors
echo "🚀 Running update-braiv-lipsync.service..."
systemctl start update-braiv-lipsync.service
if ! systemctl is-active --quiet update-braiv-lipsync.service; then
    echo "❌ Error: update-braiv-lipsync.service failed! Blocking main service."
    systemctl status update-braiv-lipsync.service --no-pager
    exit 1
fi

# Enable and run main service (restarts on failure)
echo "✅ Enabling and starting braiv-lipsync.service..."
systemctl enable --now braiv-lipsync.service
if ! systemctl is-active --quiet braiv-lipsync.service; then
    echo "❌ Error: braiv-lipsync.service failed!"
    systemctl status braiv-lipsync.service --no-pager
    exit 1
fi

# Show final statuses of both services
echo "📢 Checking service statuses..."
systemctl status update-braiv-lipsync.service --no-pager
systemctl status braiv-lipsync.service --no-pager

echo "🎉 Setup complete! Braiv Lipsync services are now active and will persist across reboots."