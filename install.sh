#!/bin/bash
# install.sh - Installation script for Enterprise Background Removal Service
# Fixed version for Raspberry Pi 5

set -e  # Exit on any error

echo "Installing Enterprise Background Removal Service..."

# Check if running as root for system operations
if [[ $EUID -eq 0 ]]; then
   echo "This script should NOT be run as root for most operations."
   echo "Please run as regular user (pi), we'll use sudo when needed."
   echo "Usage: ./install.sh [--with-docker]"
   echo "If you want to run as root anyway, use: ./install.sh --force-root"
   if [[ "$1" != "--force-root" ]]; then
       exit 1
   fi
fi

# Check if we're on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo "Warning: This script is optimized for Raspberry Pi 5"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    libopencv-dev \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    git \
    wget \
    curl \
    htop \
    tree

# Redis and Nginx are already installed based on your output
echo "Redis and Nginx are already installed."

# Install Docker if requested
if [[ "$1" == "--with-docker" ]]; then
    echo "Installing Docker..."
    if ! command -v docker &> /dev/null; then
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        sudo usermod -aG docker $USER
        echo "Docker installed. You may need to log out and back in."
    else
        echo "Docker already installed."
    fi
    
    # Install Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        sudo pip3 install docker-compose
    else
        echo "Docker Compose already installed."
    fi
fi

# Create application directory
echo "Creating application directory..."
sudo mkdir -p /opt/background-removal
sudo chown $USER:$USER /opt/background-removal
cd /opt/background-removal

# Create directory structure
echo "Creating directory structure..."
mkdir -p {data/{temp,outputs,watch_input,backup},ftp/{input,output,processed},logs,models,config,ssl,tests,scripts,docs}
chmod 755 data/temp data/outputs ftp/input ftp/output

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3.11 -m venv venv

# Activate virtual environment and install dependencies
echo "Installing Python dependencies..."
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install core dependencies
pip install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    aiofiles==23.2.1 \
    aioredis==2.0.1 \
    Pillow==10.0.1 \
    opencv-python==4.8.1.78 \
    numpy==1.24.3 \
    pydantic==2.5.0 \
    python-dotenv==1.0.0 \
    pyftpdlib==1.5.7 \
    watchdog==3.0.0 \
    psutil==5.9.6 \
    prometheus-client==0.19.0 \
    aiohttp==3.9.0 \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1

# Create default configuration files
echo "Creating configuration files..."

# Create .env file
cat > .env << EOF
# Core Settings
HAILO_ENABLED=false
MAX_RESOLUTION=4096x4096
TARGET_PROCESSING_TIME=60
MAX_RETRIES=3
QUALITY_THRESHOLD=0.8

# Directories
TEMP_DIR=/opt/background-removal/data/temp
OUTPUT_DIR=/opt/background-removal/data/outputs
CLEANUP_AGE_HOURS=24

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# FTP Server Configuration
FTP_ENABLED=true
FTP_PORT=21
FTP_USER=bgremoval
FTP_PASSWORD=secure_password_change_this
FTP_ROOT=/opt/background-removal/ftp

# Logging
LOG_LEVEL=INFO
LOG_FILE=/opt/background-removal/logs/app.log

# Monitoring
PROMETHEUS_ENABLED=false
PROMETHEUS_PORT=9090
EOF

# Create basic hailo config
cat > config/hailo_config.yaml << EOF
# Hailo-8L Configuration
model_path: "./models/hailo_model.hef"
device_id: 0
batch_size: 1
input_format: "RGB"
output_format: "RGBA"
optimization_level: 2
EOF

# Create logging config
cat > config/logging_config.yaml << EOF
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
    stream: ext://sys.stdout
  file:
    class: logging.FileHandler
    level: INFO
    formatter: default
    filename: /opt/background-removal/logs/app.log
loggers:
  '':
    level: INFO
    handlers: [console, file]
EOF

# Create systemd service file
echo "Creating systemd service..."
sudo tee /etc/systemd/system/background-removal.service > /dev/null << EOF
[Unit]
Description=Enterprise Background Removal Service
After=network.target redis.service

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=/opt/background-removal
Environment=PATH=/opt/background-removal/venv/bin
Environment=PYTHONPATH=/opt/background-removal
ExecStart=/opt/background-removal/venv/bin/python main.py
ExecReload=/bin/kill -HUP \$MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create nginx configuration
echo "Configuring Nginx..."
sudo tee /etc/nginx/sites-available/background-removal > /dev/null << 'EOF'
server {
    listen 80;
    server_name _;
    
    # Increase client max body size for large images
    client_max_body_size 50M;
    
    # API endpoints
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeouts for long processing
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:8000/api/v1/health;
        access_log off;
    }
    
    # Static file serving for results (optional)
    location /results/ {
        alias /opt/background-removal/data/outputs/;
        expires 1h;
    }
    
    # Basic status page
    location / {
        return 200 'Background Removal Service - Status: Running';
        add_header Content-Type text/plain;
    }
}
EOF

# Enable nginx site
sudo ln -sf /etc/nginx/sites-available/background-removal /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx configuration
if ! sudo nginx -t; then
    echo "Nginx configuration test failed!"
    exit 1
fi

# Reload systemd and enable services
echo "Enabling services..."
sudo systemctl daemon-reload
sudo systemctl enable background-removal
sudo systemctl enable redis-server
sudo systemctl enable nginx

# Start Redis if not running
echo "Starting Redis..."
sudo systemctl start redis-server

# Restart nginx with new config
echo "Restarting Nginx..."
sudo systemctl restart nginx

# Create startup script
echo "Creating startup script..."
cat > scripts/start_service.sh << 'EOF'
#!/bin/bash
echo "Starting Background Removal Service..."

# Check if Redis is running
if ! systemctl is-active --quiet redis-server; then
    echo "Starting Redis..."
    sudo systemctl start redis-server
fi

# Check if Nginx is running
if ! systemctl is-active --quiet nginx; then
    echo "Starting Nginx..."
    sudo systemctl start nginx
fi

# Start main service
echo "Starting main service..."
sudo systemctl start background-removal

# Check status
echo "Service status:"
sudo systemctl status background-removal --no-pager -l
EOF

chmod +x scripts/start_service.sh

# Create basic monitoring script
cat > scripts/monitor_system.py << 'EOF'
#!/usr/bin/env python3
import psutil
import time
import json

def get_system_stats():
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'temperature': get_cpu_temp(),
        'timestamp': time.time()
    }

def get_cpu_temp():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = int(f.read().strip()) / 1000.0
            return temp
    except:
        return 0.0

if __name__ == "__main__":
    while True:
        stats = get_system_stats()
        print(json.dumps(stats, indent=2))
        time.sleep(5)
EOF

chmod +x scripts/monitor_system.py

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Install Hailo SDK (follow Hailo documentation)"
echo "2. Update the password in .env file"
echo "3. Start the service: sudo systemctl start background-removal"
echo "4. Check status: sudo systemctl status background-removal"
echo ""
echo "🔧 Quick commands:"
echo "• Start service: ./scripts/start_service.sh"
echo "• Monitor system: python3 scripts/monitor_system.py"
echo "• View logs: sudo journalctl -u background-removal -f"
echo "• Test API: curl http://localhost/health"
echo ""
echo "🌐 Access points:"
echo "• API: http://$(hostname -I | cut -d' ' -f1)/api/v1/"
echo "• Health: http://$(hostname -I | cut -d' ' -f1)/health"
echo "• FTP: ftp://$(hostname -I | cut -d' ' -f1):21"
echo ""
echo "⚠️  Important: Change the FTP password in .env before production use!"
EOF