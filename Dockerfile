# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Hailo SDK (this would be the actual Hailo installation)
# RUN wget https://hailo.ai/downloads/hailo-sdk-latest.deb && \
#     dpkg -i hailo-sdk-latest.deb || apt-get install -f -y

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/outputs /tmp/bg_removal

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "main.py"]
