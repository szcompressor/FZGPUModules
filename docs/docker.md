# Docker Setup

This document describes how to use Docker for FZGPUModules development, testing, and deployment.

## Overview

The provided Dockerfile creates a single image that supports all three use cases:
- **Local development**: Full environment for building and developing FZGPUModules
- **CI/CD testing**: Automated testing and validation in containerized environments
- **Distribution/deployment**: Packaged environment with the library pre-installed

The image is based on NVIDIA CUDA 12.6.0 with Ubuntu 24.04 and includes:
- CMake and Ninja build tools
- C++ compiler (g++)
- CUDA development libraries
- Python 3 and pip
- Sanitizer libraries (libasan8, libubsan1) for debugging

**FZGPUModules is pre-built and installed into the image** during the Docker build. After pulling the image, users can immediately write and compile code against the library — no library build step required.

Installed locations:
- Headers: `/usr/local/include/fzgmod/`
- Libraries: `/usr/local/lib/`
- CMake package config: `/usr/local/lib/cmake/FZGPUModules/` (`find_package` ready)

## Building the Docker Image

```bash
docker build -t fzgpumodules:latest .
```

Or with a specific version tag:

```bash
docker build -t fzgpumodules:1.0.0 .
```

The build takes a few minutes — it compiles FZGPUModules during the image build so subsequent runs are instant.

## Using the Pre-Installed Library

### Quick Start

Mount a directory containing your code and compile directly:

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  fzgpumodules:latest \
  nvcc my_app.cu -o my_app -lfzgmod -L/usr/local/lib
```

### With CMake (Recommended)

Use `find_package(FZGPUModules REQUIRED)` in your `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(MyApp CUDA CXX)

find_package(FZGPUModules REQUIRED)

add_executable(my_app my_app.cu)
target_link_libraries(my_app PRIVATE FZGMOD::fzgmod)
```

Then build inside the container:

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  fzgpumodules:latest \
  bash -c "cmake -B build -S . && cmake --build build"
```

### Interactive Shell

Start an interactive session to develop incrementally:

```bash
docker run --rm -it --gpus all \
  -v $(pwd):/workspace \
  fzgpumodules:latest
```

## Local Development (Building FZGPUModules Itself)

To build and modify FZGPUModules source, mount the repo and build inside the container:

```bash
docker run --rm -it --gpus all \
  -v /path/to/FZGPUModules:/workspace/src \
  fzgpumodules:latest \
  bash -c "cd /workspace/src && \
           cmake --preset release -DBUILD_EXAMPLES=ON && \
           cmake --build build/release -j$(nproc)"
```

The pre-installed library in the image is unaffected — your source build stays in the mounted directory.

## CI/CD Testing

### Running the Test Suite

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace/src \
  fzgpumodules:latest \
  bash -c "cd /workspace/src && \
           cmake --preset release -DBUILD_TESTING=ON && \
           cmake --build build/release -j$(nproc) && \
           ctest --test-dir build/release --output-on-failure"
```

### Full Build with All Targets

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace/src \
  fzgpumodules:latest \
  bash -c "cd /workspace/src && \
           cmake --preset release \
             -DBUILD_EXAMPLES=ON \
             -DBUILD_TESTING=ON \
             -DBUILD_PROFILING=ON && \
           cmake --build build/release -j$(nproc)"
```

## GPU Support

All commands above use `--gpus all` to enable GPU access. This requires:
- NVIDIA Docker runtime installed on the host
- NVIDIA driver compatible with CUDA 12.6

To verify GPU access in the container:

```bash
docker run --rm --gpus all fzgpumodules:latest nvidia-smi
```

On some systems you may need `--runtime=nvidia` instead of `--gpus all`.

## Development Notes

### Sanitizers

The image includes AddressSanitizer (libasan8) and UndefinedBehaviorSanitizer (libubsan1). To build with sanitizers:

```bash
docker run --rm -it --gpus all \
  -v $(pwd):/workspace/src \
  fzgpumodules:latest \
  bash -c "cd /workspace/src && \
           cmake --preset release \
             -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined' && \
           cmake --build build/release -j$(nproc)"
```

### Python Integration

Python 3 and pip are available for profiling scripts or future bindings:

```bash
docker run --rm -it fzgpumodules:latest python3 --version
```

## Troubleshooting

### GPU Not Detected

If `nvidia-smi` fails inside the container:
- Verify the NVIDIA container toolkit is installed on the host
- Check that your driver version supports CUDA 12.6 (`nvidia-smi` on the host shows the max supported CUDA version)
- Try `--runtime=nvidia` if `--gpus all` is not recognized

### find_package Cannot Find FZGPUModules

The CMake package config is at `/usr/local/lib/cmake/FZGPUModules/`. If `find_package` fails, set the path explicitly:

```bash
cmake -B build -S . -DFZGPUModules_DIR=/usr/local/lib/cmake/FZGPUModules
```

### Build Failures in CI

If the image build fails during the `cmake --install` step:
- Ensure `third_party/tomlplusplus/` is present (it must be vendored, not just a submodule reference)
- Verify `third_party/googletest/` is initialized if running with `-DBUILD_TESTING=ON` during the image build (it is excluded from the default release build)

## See Also

- [Building FZGPUModules](building.md) — General build instructions
- [Architecture Guide](architecture.md) — Project structure and design
