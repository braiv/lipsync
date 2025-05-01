#!/bin/bash

set -e  # Exit on any error

echo "🚀 Setting up Google Cloud Ops Agent for systemd logs..."

# Ensure script is run as root
if [ "$(id -u)" -ne 0 ]; then
    echo "❌ Error: This script must be run as root. Use sudo."
    exit 1
fi

# Install the Google Cloud Ops Agent if not installed
if ! command -v google_cloud_ops_agent_engine &> /dev/null; then
    echo "🔄 Installing Google Cloud Ops Agent..."
    curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
    sudo bash add-google-cloud-ops-agent-repo.sh --also-install
    sudo systemctl restart google-cloud-ops-agent
    echo "✅ Ops Agent installed and restarted."
else
    echo "✅ Ops Agent is already installed."
fi

# Define the config file path
CONFIG_PATH="/etc/google-cloud-ops-agent/config.yaml"

# Fix YAML syntax and apply proper indentation
echo "🔧 Configuring Ops Agent logging for systemd services..."
cat <<EOF | sudo tee "$CONFIG_PATH" > /dev/null
logging:
  receivers:
    systemd_receiver:
      type: systemd_journald

  processors:
    filter_systemd:
      type: exclude_logs
      match_any:
        - 'NOT jsonPayload._SYSTEMD_UNIT = "braiv-pipeline.service" AND NOT jsonPayload._SYSTEMD_UNIT = "update-braiv-pipeline.service"'

  service:
    pipelines:
      systemd_pipeline:
        receivers:
          - systemd_receiver
        processors:
          - filter_systemd

EOF

echo "✅ Configuration updated: $CONFIG_PATH"

# Restart the Ops Agent to apply changes
echo "🔄 Restarting Google Cloud Ops Agent..."
sudo systemctl restart google-cloud-ops-agent

# Check the status
echo "📢 Checking Ops Agent status..."
sudo systemctl status google-cloud-ops-agent --no-pager

echo "🎉 Setup complete! Logs from 'braiv-pipeline.service' and 'update-braiv-pipeline.service' are now forwarded to Google Cloud Logging."
