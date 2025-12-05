# Use a slim Python image
FROM python:3.10-slim

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system packages required for transformers/torch
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (without torch)
COPY requirements.txt /app/requirements.txt

# Install torch CPU wheel from official PyTorch index, then the rest
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.2.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /app/requirements.txt

# Create a non-root user and switch to it
RUN useradd -m -s /bin/bash ahmed
ENV HOME=/home/ahmed
USER ahmed

# Set working directory to user's app folder
WORKDIR $HOME/app

# Copy application files with correct ownership
COPY --chown=ahmed:ahmed . $HOME/app

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
