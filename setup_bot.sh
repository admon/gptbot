#!/bin/bash

# Create virtual environment and install dependencies
python3 -m venv /opt/larkbot/venv
source /opt/larkbot/venv/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Verify installations
pip list | grep -E "flask|larksuite-oapi|litellm|cryptography|python-dotenv|gunicorn"

# Create log directory
sudo mkdir -p /var/log/larkbot
sudo chown ubuntu:ubuntu /var/log/larkbot

# Create a systemd service file
sudo tee /etc/systemd/system/larkbot.service << EOF
[Unit]
Description=Lark Bot Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/larkbot
Environment=PYTHONPATH=/opt/larkbot
Environment=PYTHONWARNINGS=ignore::UserWarning:pydantic
EnvironmentFile=/opt/larkbot/.env
ExecStart=/opt/larkbot/venv/bin/gunicorn \
    --workers 4 \
    --bind 0.0.0.0:5000 \
    --log-level=debug \
    --access-logfile=/var/log/larkbot/access.log \
    --error-logfile=/var/log/larkbot/error.log \
    bot:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start the service
sudo systemctl daemon-reload
sudo systemctl start larkbot
sudo systemctl enable larkbot 