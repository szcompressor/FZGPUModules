cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(FZGPUModules LANGUAGES CUDA CXX C VERSION 1.0)

########################
### GENERAL SETTINGS ###
########################

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "prefer shared libraries" ON)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures to compile for")
endif()

#####################
### WSL DETECTION ###
#####################

execute_process(
  COMMAND systemd-detect-virt
  OUTPUT_VARIABLE VIRT_ENV
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
  RESULT_VARIABLE DETECT_RESULT
)

if(DETECT_RESULT EQUAL 0 AND VIRT_ENV STREQUAL "wsl")
  message(STATUS "Running on WSL")
  set(IS_WSL TRUE)
  link_directories(/usr/lib/wsl/lib)
else()
  set(IS_WSL FALSE)
endif()

############################
### BUILD CONFIGURATIONS ###
############################

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}'")
  set(CMAKE_BUILD_TYPE ${default_build_type} CACHE
      STRING "Choose the type to build." FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -t 0")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -t 0")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
endif()

option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
    enable_testing()
    add_subdirectory(tests)
endif()

option(BUILD_EXAMPLES "Build example executables" OFF)
option(BUILD_PROFILING "Build profiling target with NVTX3/nsys instrumentation" OFF)

################################
### SANITIZERS CONFIGURATION ###
################################

if(NOT USE_SANITIZER)
  set(USE_SANITIZER "" CACHE STRING "use a build sanitizer" FORCE)
endif()
set_property(CACHE USE_SANITIZER PROPERTY STRINGS "" "ASanUbsan" "TSan" "Compute")
option(COMPUTE_SANITIZER_DEVICE_DEBUG "Build CUDA with -G for Compute Sanitizer (slower, deeper debug)" OFF)

if (USE_SANITIZER STREQUAL "ASanUbsan")
  message(STATUS "Compiling with Address+Undefined Sanitizers")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address,undefined -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined -fno-omit-frame-pointer")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address,undefined")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address,undefined")
elseif (USE_SANITIZER STREQUAL "TSan")
  message(STATUS "Compiling with Thread Sanitizer")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -fno-omit-frame-pointer")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=thread")
elseif (USE_SANITIZER STREQUAL "Compute")
  message(STATUS "Configuring for Compute Sanitizer")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xcompiler -rdynamic")
  if(COMPUTE_SANITIZER_DEVICE_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
  endif()
endif()

#########################
### COMPILER SETTINGS ###
#########################

add_library(fzgmod_compile_settings INTERFACE)
target_compile_options(
  fzgmod_compile_settings 
  INTERFACE 
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>
)

target_compile_options(
  fzgmod_compile_settings
  INTERFACE
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-lineinfo>
)

target_compile_features(fzgmod_compile_settings INTERFACE cxx_std_17 cuda_std_17)
target_link_libraries(fzgmod_compile_settings INTERFACE CUDA::cudart)

###########################
### INCLUDE DIRECTORIES ###
###########################

target_include_directories(fzgmod_compile_settings INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fzgmod>
)

#################
### LIBRARIES ###
#################

#### Memory Management
add_library(fzgmod_mem
  src/mem/mempool.cpp
)
target_link_libraries(fzgmod_mem
  PUBLIC
  fzgmod_compile_settings
)

#### Encoders Module
add_library(fzgmod_encoders
  modules/encoders/bitpacking/bitpacking.cu
  modules/encoders/diff/diff.cu
  modules/encoders/RLE/rle.cu
  modules/transforms/zigzag/zigzag_stage.cu
  modules/transforms/negabinary/negabinary_stage.cu
  modules/transforms/bitshuffle/bitshuffle_stage.cu
  modules/transforms/rze/rze_stage.cu
)
target_link_libraries(fzgmod_encoders
  PUBLIC
  fzgmod_compile_settings
  CUDA::cudart
)
add_library(FZGMOD::encoders ALIAS fzgmod_encoders)

#### Predictors Module
add_library(fzgmod_predictors
  modules/predictors/lorenzo/lorenzo.cu
  modules/predictors/quantizer/quantizer.cu
)
target_link_libraries(fzgmod_predictors
  PUBLIC
  fzgmod_compile_settings
  CUDA::cudart
)
add_library(FZGMOD::predictors ALIAS fzgmod_predictors)

add_library(fzgmod_pipeline
  src/pipeline/compressor.cpp
  src/pipeline/compressor_exec.cpp
  src/pipeline/compressor_inverse_dag.cpp
  src/pipeline/compressor_io.cpp
  src/pipeline/dag.cpp
  src/pipeline/perf.cpp
  src/pipeline/stat.cpp
  src/stage/mock_stages.cu
)
target_link_libraries(fzgmod_pipeline
  PUBLIC
  fzgmod_compile_settings
  fzgmod_mem
  fzgmod_encoders
  fzgmod_predictors
  CUDA::cudart
)
add_library(FZGMOD::pipeline ALIAS fzgmod_pipeline)

#### Combined Library
add_library(fzgmod
  utils/cli/cli.cpp
)
target_link_libraries(fzgmod
  PUBLIC
  fzgmod_compile_settings
  fzgmod_mem
  fzgmod_encoders
  fzgmod_predictors
  fzgmod_pipeline
  CUDA::cudart
)
add_library(FZGMOD::fzgmod ALIAS fzgmod)

###################
### EXECUTABLES ###
###################

if(BUILD_EXAMPLES)
    add_executable(fzgmod-bin examples/main.cpp)
    target_link_libraries(fzgmod-bin PRIVATE fzgmod)
    set_target_properties(fzgmod-bin PROPERTIES
        OUTPUT_NAME fzgmod
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    # ── Standalone Lorenzo code-distribution analyser ─────────────────────────────
    add_executable(analyze_lorenzo examples/analyze_lorenzo.cpp)
    target_link_libraries(analyze_lorenzo PRIVATE fzgmod)
    set_target_properties(analyze_lorenzo PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    # ── Full PFPL pipeline demonstration ──────────────────────────────────────────
    add_executable(pfpl_example examples/pfpl_pipeline.cpp)
    target_link_libraries(pfpl_example PRIVATE fzgmod)
    set_target_properties(pfpl_example PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    # ── Stage isolation diagnostic ─────────────────────────────────────────────
    add_executable(test_quant_only examples/test_quant_only.cpp)
    target_link_libraries(test_quant_only PRIVATE fzgmod)
    set_target_properties(test_quant_only PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

######################
### PROFILING MODE ###
######################

if(BUILD_PROFILING)
    message(STATUS "Profiling mode enabled — NVTX3/nsys instrumentation active")

    # Preprocessor guard consumed by #ifdef FZ_PROFILING_ENABLED in source files.
    # Enables NVTX annotations in fzgmod_pipeline (decompressFromFile) and
    # cudaProfilerStart/Stop in the fzgmod-profile binary.
    target_compile_definitions(fzgmod_compile_settings INTERFACE FZ_PROFILING_ENABLED)

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
        # fzgmod_pipeline compiles compressor.cpp which uses nvtx3 behind
        # #ifdef FZ_PROFILING_ENABLED — it needs the headers when profiling is on.
        target_include_directories(fzgmod_pipeline PRIVATE ${NVTX3_INCLUDE_DIR})
    else()
        message(WARNING "nvtx3/nvtx3.hpp not found — profiling builds may not compile.\n"
                        "Override with: -DNVTX3_INCLUDE_DIR=/path/to/nvtx/include")
    endif()

    # ── Profiling executable ──────────────────────────────────────────────────
    # Build:  cmake --build build --target fzgmod-profile
    # Run:    nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
    #              -o cldhgh_profile ./build/bin/fzgmod-profile
    #         nsys-ui cldhgh_profile.nsys-rep
    add_executable(fzgmod-profile examples/basic_profiling_example.cpp)
    target_link_libraries(fzgmod-profile PRIVATE fzgmod)
    set_target_properties(fzgmod-profile PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    if(NVTX3_INCLUDE_DIR)
        target_include_directories(fzgmod-profile PRIVATE ${NVTX3_INCLUDE_DIR})
    endif()

    # ── Repeated-compression benchmark ───────────────────────────────────────
    # Measures the throughput benefit of reusing a PREALLOCATE pipeline vs.
    # constructing a fresh MINIMAL pipeline for every compress() call.
    #
    # Build:  cmake --build build --target fzgmod-profile-repeat
    # Run:    ./build/bin/fzgmod-profile-repeat
    #   or:   nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
    #              -o repeat_profile ./build/bin/fzgmod-profile-repeat
    #         nsys-ui repeat_profile.nsys-rep
    add_executable(fzgmod-profile-repeat examples/profile_repeat.cpp)
    target_link_libraries(fzgmod-profile-repeat PRIVATE fzgmod)
    set_target_properties(fzgmod-profile-repeat PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    if(NVTX3_INCLUDE_DIR)
        target_include_directories(fzgmod-profile-repeat PRIVATE ${NVTX3_INCLUDE_DIR})
    endif()

    # ── PFPL pipeline profiling example ──────────────────────────────────────
    # Build:  cmake --build build --target pfpl_example  (with -DBUILD_PROFILING=ON)
    # Run:    nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
    #              -o pfpl_profile ./build/bin/pfpl_example 1e-3
    #         nsys-ui pfpl_profile.nsys-rep
    if(NVTX3_INCLUDE_DIR)
        target_include_directories(pfpl_example PRIVATE ${NVTX3_INCLUDE_DIR})
    endif()
endif()

####################
### INSTALLATION ###
####################

install(TARGETS fzgmod_compile_settings EXPORT FZGPUModulesTargets)
install(TARGETS 
  fzgmod_mem
  fzgmod_encoders
  fzgmod_predictors
  fzgmod_pipeline
  fzgmod 
  EXPORT FZGPUModulesTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(BUILD_EXAMPLES)
  install(TARGETS fzgmod-bin
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()

install(
  EXPORT FZGPUModulesTargets
  NAMESPACE FZGMOD::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FZGPUModules
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FZGPUModulesConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/FZGPUModulesConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FZGPUModules
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/FZGPUModulesConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion
)

install(FILES 
  "${CMAKE_CURRENT_BINARY_DIR}/FZGPUModulesConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/FZGPUModulesConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FZGPUModules
)

# Export build-tree targets so find_package(FZGPUModules) works without installing
# by pointing FZGPUModules_DIR at the build directory.
export(
  EXPORT FZGPUModulesTargets
  NAMESPACE FZGMOD::
  FILE "${CMAKE_CURRENT_BINARY_DIR}/FZGPUModulesTargets.cmake"
)

install(DIRECTORY
  include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fzgmod
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hh"
)

# modules/ headers are included with relative paths (e.g. "encoders/diff/diff.h")
# so they must be installed into the same fzgmod include root
install(DIRECTORY
  modules/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fzgmod
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hh"
)

# src/ headers are also an include root at build time (e.g. "mem/mempool.h")
install(DIRECTORY
  src/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fzgmod
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hh"
)

message(STATUS "FZGPUModules configuration complete")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build profiling: ${BUILD_PROFILING}")
