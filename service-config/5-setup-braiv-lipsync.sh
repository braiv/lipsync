#!/bin/bash

set -e  # Exit immediately if any command fails

# Ensure an environment is provided
if [ -z "$1" ]; then
    echo "❌ Error: No environment specified. Usage: $0 {prod|preview|dev}"
    exit 1
fi

ENV="$1"
REAL_USER="${SUDO_USER:-$USER}"

echo "🚀 Setting up systemd services for Braiv Pipeline (Environment: $ENV)..."

# Define service files
UPDATE_SERVICE="/etc/systemd/system/update-braiv-pipeline.service"
MAIN_SERVICE="/etc/systemd/system/braiv-pipeline.service"

# Ensure script is run as root
if [ "$(id -u)" -ne 0 ]; then
    echo "❌ Error: This script must be run as root. Use sudo."
    exit 1
fi

# Set up logging directory
echo "🔧 Setting up logging directory..."
touch /var/log/braiv-service.log
chown "$REAL_USER:$REAL_USER" /var/log/braiv-service.log

# Create or update the update service
echo "🔧 Creating or updating $UPDATE_SERVICE..."
cat <<EOF > "$UPDATE_SERVICE"
[Unit]
Description=Update Braiv Pipeline on Startup
After=network.target
Before=braiv-pipeline.service

[Service]
Type=oneshot
ExecStart=/bin/bash /opt/braiv-pipeline/service-config/update-braiv-pipeline.sh
WorkingDirectory=/opt/braiv-pipeline
User=$REAL_USER
Group=$REAL_USER
RemainAfterExit=yes
SuccessExitStatus=0 1

[Install]
WantedBy=multi-user.target
EOF

# Create or update the main service
echo "🔧 Creating or updating $MAIN_SERVICE..."
cat <<EOF > "$MAIN_SERVICE"
[Unit]
Description=Braiv Pipeline Service
After=update-braiv-pipeline.service
Requires=update-braiv-pipeline.service

[Service]
ExecStart=/bin/bash /opt/braiv-pipeline/service-config/start-braiv-pipeline.sh $ENV
WorkingDirectory=/opt/braiv-pipeline
User=$REAL_USER
Group=$REAL_USER
Restart=always
RestartSec=10
Environment="PATH=/opt/miniconda/bin:/opt/miniconda/envs/braiv-pipeline/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd to apply changes
echo "🔄 Reloading systemd daemon..."
systemctl daemon-reload

# Enable update service (runs on startup)
echo "✅ Enabling update-braiv-pipeline.service..."
systemctl enable update-braiv-pipeline.service

# Start update service and check for errors
echo "🚀 Running update-braiv-pipeline.service..."
systemctl start update-braiv-pipeline.service
if ! systemctl is-active --quiet update-braiv-pipeline.service; then
    echo "❌ Error: update-braiv-pipeline.service failed! Blocking main service."
    systemctl status update-braiv-pipeline.service --no-pager
    exit 1
fi

# Enable & start main service (restarts on failure)
echo "✅ Enabling and starting braiv-pipeline.service..."
systemctl enable --now braiv-pipeline.service
if ! systemctl is-active --quiet braiv-pipeline.service; then
    echo "❌ Error: braiv-pipeline.service failed!"
    systemctl status braiv-pipeline.service --no-pager
    exit 1
fi

# Show status of both services
echo "📢 Checking service statuses..."
systemctl status update-braiv-pipeline.service --no-pager
systemctl status braiv-pipeline.service --no-pager

echo "🎉 Setup complete! Services are now active and will persist across reboots."
