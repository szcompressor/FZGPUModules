# profiling/CMakeLists.txt
# Profiling harnesses intended to be run under nsys / compute-sanitizer.
# Enabled with: cmake -DBUILD_PROFILING=ON ..
#
# Build:
#   cmake --build <builddir> --target fzgmod-profile-cuszp2
#   cmake --build <builddir> --target fzgmod-profile-repeat
#
# Run under nsys:
#   nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
#        -o cuszp2 ./build/bin/profiling/fzgmod-profile-cuszp2
#   nsys-ui cuszp2.nsys-rep

if(NOT BUILD_PROFILING)
  return()
endif()

message(STATUS "Profiling mode enabled — NVTX3/nsys instrumentation active")

# Sample dataset shipped in-repo (data/CLDHGH.f32); profiling programs read
# this instead of a hardcoded absolute path so they work on any checkout.
target_compile_definitions(fzgmod_compile_settings INTERFACE
    FZ_DATA_DIR="${CMAKE_SOURCE_DIR}/data")

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
        # Note: CUDAToolkit_ROOT_DIR is an internal FindCUDAToolkit variable
        # that the module unsets before returning — use the public result
        # variable CUDAToolkit_INCLUDE_DIRS instead (nvtx3/ ships inside it).
        ${CUDAToolkit_INCLUDE_DIRS}
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


# ── cuSZp2 nsys / NVTX3 harness ──────────────────────────────────────────────
# Pipeline: Quantizer(linear,ABS) → Lorenzo(block=32) → AdaptiveBitpack(block=32,outlier)
# Matches examples/presets/cuszp2.toml.
# Profiles PREALLOCATE (cold + steady-state) and CUDA Graph (compress replay).
add_executable(fzgmod-profile-cuszp2 cuszp2_profile.cpp)
target_link_libraries(fzgmod-profile-cuszp2 PRIVATE fzgmod)
set_target_properties(fzgmod-profile-cuszp2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/profiling")
if(NVTX3_INCLUDE_DIR)
    target_include_directories(fzgmod-profile-cuszp2 PRIVATE ${NVTX3_INCLUDE_DIR})
endif()

# ── L2-blocked tiling prototype (compress locality probe) ────────────────────
add_executable(fzgmod-profile-l2tile l2tile_profile.cpp)
target_link_libraries(fzgmod-profile-l2tile PRIVATE fzgmod)
set_target_properties(fzgmod-profile-l2tile PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/profiling")
if(NVTX3_INCLUDE_DIR)
    target_include_directories(fzgmod-profile-l2tile PRIVATE ${NVTX3_INCLUDE_DIR})
endif()

# ── Fully-fused cuSZp2 go/no-go proof ────────────────────────────────────────
add_executable(fzgmod-profile-fuse fuse_proof.cu)
target_link_libraries(fzgmod-profile-fuse PRIVATE fzgmod)
target_include_directories(fzgmod-profile-fuse PRIVATE
    ${CMAKE_SOURCE_DIR}/modules ${CMAKE_SOURCE_DIR}/include)
set_target_properties(fzgmod-profile-fuse PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/profiling")
if(NVTX3_INCLUDE_DIR)
    target_include_directories(fzgmod-profile-fuse PRIVATE ${NVTX3_INCLUDE_DIR})
endif()

# ── cuSZp3 nsys / NVTX3 harness ──────────────────────────────────────────────
# Pipeline: Quantizer(linear,ABS) → TiledLorenzo(8x8) → AdaptiveBitpack(64,outlier)
# Matches examples/presets/cuszp3_outlier.toml.
# Profiles PREALLOCATE (cold + steady-state) and CUDA Graph (compress replay).
add_executable(fzgmod-profile-cuszp3 cuszp3_profile.cpp)
target_link_libraries(fzgmod-profile-cuszp3 PRIVATE fzgmod)
set_target_properties(fzgmod-profile-cuszp3 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/profiling")
if(NVTX3_INCLUDE_DIR)
    target_include_directories(fzgmod-profile-cuszp3 PRIVATE ${NVTX3_INCLUDE_DIR})
endif()

# ── cuSZ-Hi throughput-mode nsys / NVTX3 harness ─────────────────────────────
# Pipeline: GInterp → {Zigzag→Bitshuffle→RRE} / {Merge→Bitshuffle→RRE→RZE}
# Matches examples/presets/cusz_hi_tp.toml.
add_executable(fzgmod-profile-cuszhi-tp cuszhi_tp_profile.cpp)
target_link_libraries(fzgmod-profile-cuszhi-tp PRIVATE fzgmod)
set_target_properties(fzgmod-profile-cuszhi-tp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/profiling")
if(NVTX3_INCLUDE_DIR)
    target_include_directories(fzgmod-profile-cuszhi-tp PRIVATE ${NVTX3_INCLUDE_DIR})
endif()

# ── FZ-GPU nsys / NVTX3 harness ──────────────────────────────────────────────
# Pipeline: LorenzoQuant → BitplaneRZE. Matches examples/presets/fzgpu.toml.
add_executable(fzgmod-profile-fzgpu fzgpu_profile.cpp)
target_link_libraries(fzgmod-profile-fzgpu PRIVATE fzgmod)
set_target_properties(fzgmod-profile-fzgpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/profiling")
if(NVTX3_INCLUDE_DIR)
    target_include_directories(fzgmod-profile-fzgpu PRIVATE ${NVTX3_INCLUDE_DIR})
endif()

# ── PFPL nsys / NVTX3 harness ────────────────────────────────────────────────
# Pipeline: Quantizer → Difference → Bitshuffle → RZE. Matches examples/presets/pfpl.toml.
add_executable(fzgmod-profile-pfpl pfpl_profile.cpp)
target_link_libraries(fzgmod-profile-pfpl PRIVATE fzgmod)
set_target_properties(fzgmod-profile-pfpl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/profiling")
if(NVTX3_INCLUDE_DIR)
    target_include_directories(fzgmod-profile-pfpl PRIVATE ${NVTX3_INCLUDE_DIR})
endif()

# ── Speed preset nsys / NVTX3 harness ────────────────────────────────────────
# Pipeline: Quantizer → Bitshuffle → RZE. Matches examples/presets/speed.toml.
add_executable(fzgmod-profile-speed speed_profile.cpp)
target_link_libraries(fzgmod-profile-speed PRIVATE fzgmod)
set_target_properties(fzgmod-profile-speed PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/profiling")
if(NVTX3_INCLUDE_DIR)
    target_include_directories(fzgmod-profile-speed PRIVATE ${NVTX3_INCLUDE_DIR})
endif()

# ── Huffman codebook drift harness ───────────────────────────────────────────
# Pipeline: LorenzoQuant → Huffman, one pipeline instance, many successive
# compress() calls on *different* data (files, or --slabs of one file).
# Measures whether HuffmanBookSource::Adaptive's pinned codebook decays across a
# run — the case a single-field measurement structurally cannot show.
add_executable(fzgmod-profile-huffman-drift huffman_book_drift.cpp)
target_link_libraries(fzgmod-profile-huffman-drift PRIVATE fzgmod)
set_target_properties(fzgmod-profile-huffman-drift PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/profiling")
if(NVTX3_INCLUDE_DIR)
    target_include_directories(fzgmod-profile-huffman-drift PRIVATE ${NVTX3_INCLUDE_DIR})
endif()
