# Use the TensorFlow 1.7.0 RC1 GPU image (Python 3)
FROM tensorflow/tensorflow:1.7.0-rc1-devel-gpu-py3

LABEL maintainer="you@example.com"
LABEL description="Docker image for Deep Learning with TensorFlow 1.7.0 and OpenSlide"

# Set environment variable to avoid some interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libopenjp2-7-dev \
    python3-opencv \
    python3-pip \
    python3-setuptools \
    python3-numpy \
    python3-scipy \
    python3-matplotlib && \
    rm -rf /var/lib/apt/lists/*
# Install OpenSlide and its Python bindings
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openslide-tools \
    openslide-python && \
    rm -rf /var/lib/apt/lists/*
# Install additional Python packages

COPY . ./app
WORKDIR /app
RUN pip3 install --no-cache-dir \
    -r requirements.txt && \
    rm -rf /root/.cache/pip
# Set the default command to run when starting the container
CMD ["python3", "segmentation_school.py"]