# Custom Dockerfile for MotionGPT with OSMesa support for slow mode rendering
FROM python:3.10-slim

# Install system dependencies including OSMesa for headless OpenGL rendering
RUN apt-get update && apt-get install -y \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]


