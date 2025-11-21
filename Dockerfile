# This Dockerfile is used to generate the docker image dsarchive/histomicstk
# This docker image includes the HistomicsTK python package along with its
# dependencies.
#
# All plugins of HistomicsTK should derive from this docker image

# Base: TensorFlow 2.15 (Python 3.10) GPU with matching CUDA/cuDNN
FROM tensorflow/tensorflow:2.15.0-gpu
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL maintainer="Anish Tatke <anish.tatke@ufl.edu>"


RUN echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! STARTING THE BUILD !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TF_CPP_MIN_LOG_LEVEL=2

# Remove any stale CUDA repo lists if present in this base (harmless if absent)
RUN rm -f /etc/apt/sources.list.d/cuda*.list || true

RUN apt-get update; \
    # apt-get install -y --no-install-recommends software-properties-common; \
    # add-apt-repository -y ppa:deadsnakes/ppa; \
    apt-get autoremove; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# System dependencies (lxml / openslide / pyvips / opencv-headless and basic build tools)
RUN apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get --yes --no-install-recommends -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" dist-upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git curl cmake ca-certificates wget unzip \
        build-essential pkg-config \
        libxml2-dev libxslt1-dev \
        openslide-tools libopenslide0 \
        libvips \
        ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
        libjpeg-turbo8-dev zlib1g-dev libpng-dev libopenjp2-7-dev libtiff5 libtiff-dev \
        libfreetype6-dev libwebp-dev libopenexr-dev \
        memcached; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHECKPOINT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

# IFTA-specific paths
ENV build_path=/build \
    ifta_path=/ifta
RUN mkdir -p "${ifta_path}"
WORKDIR "${ifta_path}"

# Copy source
COPY . "${ifta_path}/"

ENV PYTHON_BIN=/usr/bin/python \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install pip
RUN set -eux; \
    (${PYTHON_BIN} -m ensurepip --upgrade || true); \
    if ! ${PYTHON_BIN} -m pip --version >/dev/null 2>&1; then \
        curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py; \
        ${PYTHON_BIN} /tmp/get-pip.py; \
        rm -f /tmp/get-pip.py; \
    fi;

# Install IFTA package (setup.py drives dependencies)
RUN ${PYTHON_BIN} -m pip install --no-cache-dir --upgrade pip setuptools wheel setuptools-scm && \
    ${PYTHON_BIN} -m pip install --no-cache-dir /ifta && \
    # optional: verify imports
    ${PYTHON_BIN} -c "from matplotlib import pyplot as plt" && \
    ${PYTHON_BIN} -m pip list --format=freeze > /image-requirements.txt && \
    rm -rf /root/.cache/pip/*

# Sanity checks
RUN ${PYTHON_BIN} -c "import sys; print(sys.executable); print(sys.version_info)" && \
    ${PYTHON_BIN} -c "import numpy as np; print(np.__version__)" && \
    ${PYTHON_BIN} -c "import tensorflow as tf; print(tf.__version__)"

# Verify installation
RUN ${PYTHON_BIN} --version && ${PYTHON_BIN} -m pip --version && ${PYTHON_BIN} -m pip freeze | tail -n +1 | head -n 50

# Define entrypoint through which all CLIs can be run
WORKDIR "${ifta_path}/ifta/cli"
LABEL entry_path="${ifta_path}/ifta/cli"

# Test CLI discovery (optional; keep if you want to fail fast during build)
RUN ${PYTHON_BIN} -m slicer_cli_web.cli_list_entrypoint --list_cli && \
    ${PYTHON_BIN} -m slicer_cli_web.cli_list_entrypoint IFTASegmentation --help

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]