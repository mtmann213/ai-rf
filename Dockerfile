# Use NVIDIA's official TensorFlow GPU image as the base
FROM tensorflow/tensorflow:2.15.0-gpu

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libhdf5-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Robust Pip Installation (Nuclear Option)
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (minus things in .dockerignore)
COPY . .

# Default command
CMD ["python3", "train_resnet.py"]
