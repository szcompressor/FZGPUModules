# Base image: NVIDIA CUDA Toolkit (Development environment)
# Using 12.6 to match the development version used for FZGPUModules
# Supports local development, CI/CD testing, and distribution/deployment.
# FZGPUModules is pre-built and installed to /usr/local so users can link
# against it without building the library themselves.
FROM nvcr.io/nvidia/cuda:12.6.0-devel-ubuntu24.04

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools and dependencies
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    cmake \
    ninja-build \
    git \
    g++ \
    libasan8 \
    libubsan1 \
    curl \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# copy over test data
COPY data/ /data/

# Pre-build and install FZGPUModules to /usr/local.
# After this step: headers at /usr/local/include/fzgmod/, libraries at /usr/local/lib/,
# and CMake package config at /usr/local/lib/cmake/FZGPUModules/ (find_package ready).
COPY . /tmp/fzgpumodules-src
RUN cd /tmp/fzgpumodules-src && \
    cmake --preset release && \
    cmake --build build/release -j$(nproc) && \
    cmake --install build/release --prefix /usr/local && \
    rm -rf /tmp/fzgpumodules-src

# Set up a working directory for user code
WORKDIR /workspace

# By default, open a bash shell
CMD ["/bin/bash"]
