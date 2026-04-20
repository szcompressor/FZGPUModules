# profiling/CMakeLists.txt
# Profiling harnesses intended to be run under nsys / compute-sanitizer.
# Enabled with: cmake -DBUILD_PROFILING=ON ..
#
# Build:
#   cmake --build <builddir> --target fzgmod-profile
#   cmake --build <builddir> --target fzgmod-profile-repeat
#
# Run under nsys:
#   nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
#        -o my_profile ./build/bin/fzgmod-profile
#   nsys-ui my_profile.nsys-rep

if(NOT BUILD_PROFILING)
  return()
endif()

message(STATUS "Profiling mode enabled — NVTX3/nsys instrumentation active")

# Retain frame pointers so nsys CPU-sampling stack walks are complete.
target_compile_options(fzgmod_compile_settings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
    $<$<COMPILE_LANGUAGE:C>:-fno-omit-frame-pointer>
)
# Note: -lineinfo for CUDA device code is already set in fzgmod_compile_settings
# above for all non-Compute-sanitizer builds.

# ── Locate nvtx3/nvtx3.hpp ────────────────────────────────────────────────
# Ships with Nsight Systems but the path differs across machines
# (standalone install vs. bundled inside CUDA toolkit).
find_path(NVTX3_INCLUDE_DIR
    NAMES nvtx3/nvtx3.hpp
    HINTS
        # Cluster: nsight-systems bundled inside the CUDA toolkit tree
        "${CUDAToolkit_ROOT_DIR}/nsight-systems-2024.4.2/target-linux-x64/nvtx/include"
        "${CUDAToolkit_ROOT_DIR}/nsight-systems-2024.5.1/target-linux-x64/nvtx/include"
        "${CUDAToolkit_ROOT_DIR}/nsight-systems-2025.1.3/target-linux-x64/nvtx/include"
        # Derived from nvcc location (covers any version glob)
        "${CUDAToolkit_BIN_DIR}/../nsight-systems-2024.4.2/target-linux-x64/nvtx/include"
        "${CUDAToolkit_BIN_DIR}/../nsight-systems-2024.5.1/target-linux-x64/nvtx/include"
        "${CUDAToolkit_BIN_DIR}/../nsight-systems-2025.1.3/target-linux-x64/nvtx/include"
        # WSL / workstation: standalone Nsight Systems install
        "/opt/nvidia/nsight-systems/2025.1.3/target-linux-x64/nvtx/include"
        "/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nvtx/include"
        "/opt/nvidia/nsight-systems/2024.4.2/target-linux-x64/nvtx/include"
        # Fallback: CUDA toolkit headers
        "${CUDAToolkit_ROOT_DIR}/targets/x86_64-linux/include"
    DOC "Path containing nvtx3/nvtx3.hpp"
)

if(NVTX3_INCLUDE_DIR)
    message(STATUS "Found NVTX3 headers: ${NVTX3_INCLUDE_DIR}")
    # Preprocessor guard consumed by #ifdef FZ_PROFILING_ENABLED in source files.
    # Enables NVTX annotations in fzgmod_pipeline (decompressFromFile) and
    # cudaProfilerStart/Stop in profiling executables.
    target_compile_definitions(fzgmod_compile_settings INTERFACE FZ_PROFILING_ENABLED)

    # Profiling can be enabled for examples and library code via
    # fzgmod_compile_settings, so attach NVTX includes there.
    target_include_directories(fzgmod_compile_settings INTERFACE
        $<BUILD_INTERFACE:${NVTX3_INCLUDE_DIR}>
    )
else()
    message(WARNING "nvtx3/nvtx3.hpp not found — building without profiling instrumentation.\n"
                    "Override with: -DNVTX3_INCLUDE_DIR=/path/to/nvtx/include")
endif()


# ── nsys / NVTX3 harness ──────────────────────────────────────────────────────
add_executable(fzgmod-profile basic_profiling_example.cpp)
target_link_libraries(fzgmod-profile PRIVATE fzgmod)
set_target_properties(fzgmod-profile PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
if(NVTX3_INCLUDE_DIR)
    target_include_directories(fzgmod-profile PRIVATE ${NVTX3_INCLUDE_DIR})
endif()

# ── Repeated-compression throughput benchmark ─────────────────────────────────
# Measures the benefit of reusing a PREALLOCATE pipeline vs. constructing a
# fresh MINIMAL pipeline for every compress() call.
add_executable(fzgmod-profile-repeat profile_repeat.cpp)
target_link_libraries(fzgmod-profile-repeat PRIVATE fzgmod)
set_target_properties(fzgmod-profile-repeat PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
if(NVTX3_INCLUDE_DIR)
    target_include_directories(fzgmod-profile-repeat PRIVATE ${NVTX3_INCLUDE_DIR})
endif()
