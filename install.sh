
# install.sh - Installation script for Raspberry Pi 5
#!/bin/bash
set -e

echo "Installing Enterprise Background Removal Service..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    redis-server \
    nginx \
    git \
    wget \
    curl

# Install Docker (optional)
if [ "$1" == "--with-docker" ]; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    sudo usermod -aG docker $USER
    sudo pip3 install docker-compose
fi

# Create application directory
sudo mkdir -p /opt/background-removal
sudo chown $USER:$USER /opt/background-removal
cd /opt/background-removal

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install Hailo SDK (placeholder - actual installation would vary)
echo "Note: Please install Hailo SDK separately according to Hailo documentation"
echo "Visit: https://hailo.ai/developer-zone/"

# Create directories
mkdir -p outputs temp ftp logs
chmod 755 outputs temp ftp

# Set up systemd service
sudo cp background-removal.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable background-removal

# Configure nginx
sudo cp nginx.conf /etc/nginx/sites-available/background-removal
sudo ln -sf /etc/nginx/sites-available/background-removal /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx

# Start Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server

echo "Installation complete!"
echo "To start the service: sudo systemctl start background-removal"
echo "To check status: sudo systemctl status background-removal"
echo "To view logs: sudo journalctl -u background-removal -f"
