# Base image: NVIDIA CUDA Toolkit (Development environment)
# Using 12.6 to match the development version used for FZGPUModules
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
    && rm -rf /var/lib/apt/lists/*

# Set up a working directory
WORKDIR /workspace

# By default, open a bash shell
CMD ["/bin/bash"]
