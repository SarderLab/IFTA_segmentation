FROM nvidia/cuda:12.4.0-base-ubuntu22.04

LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL maintainer="Sayat Mimar - Sarder Lab. <sayat.mimar@ufl.edu>"

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV DEBIAN_FRONTEND=noninteractive
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

ENV BUILD_PATH=/build
ENV IFTA_PATH=/ifta

# Update and install dependencies for building Python 3.6 and other tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    wget \
    curl \
    git \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    liblzma-dev \
    python2.7-dev \
    python-tk \
    python3-pyqt5 \
    ca-certificates \
    libcurl4-openssl-dev \
    libexpat1-dev \
    unzip \
    libhdf5-dev \
    libpython3-dev \
    python3-tk \
    cmake \
    autoconf \
    automake \
    libtool \
    pkg-config \
    libmemcached-dev && \
    rm -rf /var/lib/apt/lists/*

# Download, build, and install Python 3.6.15 from source
RUN cd /usr/src && \
    wget https://www.python.org/ftp/python/3.6.15/Python-3.6.15.tgz && \
    tar xzf Python-3.6.15.tgz && \
    cd Python-3.6.15 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd / && rm -rf /usr/src/Python-3.6.15*

# Set python3.6 as default python and install pip for Python 3.6
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.6 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.6 1 && \
    curl https://bootstrap.pypa.io/pip/3.6/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    ln -sf /usr/local/bin/pip3 /usr/bin/pip

# Create workspace and copy code
RUN mkdir -p $BUILD_PATH $IFTA_PATH
COPY . $IFTA_PATH
WORKDIR $IFTA_PATH

# Upgrade pip and install required Python packages (TF 1.7.0, tensorboard, others)
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir tensorboard==1.7.0 tensorflow==1.7.0 tensorflow-gpu==1.7.0 && \
    pip install --no-cache-dir . && \
    rm -rf /root/.cache/pip/*

# Set CLI workdir and test entrypoints
WORKDIR $IFTA_PATH/ifta/cli
LABEL entry_path=$IFTA_PATH/ifta/cli

RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli && \
    python -m slicer_cli_web.cli_list_entrypoint IFTASegmentation --help

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
