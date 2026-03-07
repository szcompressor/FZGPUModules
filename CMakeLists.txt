cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(FZGPUModules LANGUAGES CUDA CXX C VERSION 1.0)

########################
### GENERAL SETTINGS ###
########################

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "prefer shared libraries" ON)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures to compile for")

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

################################
### SANITIZERS CONFIGURATION ###
################################

if(NOT USE_SANITIZER)
  set(USE_SANITIZER "" CACHE STRING "use a build sanitizer" FORCE)
endif()
set_property(CACHE USE_SANITIZER PROPERTY STRINGS "" "Address" "Thread" "Undefined" "Leak" "Compute")

if (USE_SANITIZER STREQUAL "Address")
    message(STATUS "Compiling with Address Sanitizer")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
elseif (USE_SANITIZER STREQUAL "Thread")
    message(STATUS "Compiling with Thread Sanitizer")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
elseif (USE_SANITIZER STREQUAL "Undefined")
    message(STATUS "Compiling with Undefined Behavior Sanitizer")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=undefined")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
elseif (USE_SANITIZER STREQUAL "Leak")
    message(STATUS "Compiling with Leak Sanitizer")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=leak")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
elseif (USE_SANITIZER STREQUAL "Compute")
    message(STATUS "Compiling with Compute Sanitizer")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -Xcompiler -rdynamic")
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

if(NOT USE_SANITIZER STREQUAL "Compute")
  target_compile_options(
    fzgmod_compile_settings
    INTERFACE
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-lineinfo>
  )
endif()

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
)
target_link_libraries(fzgmod_predictors
  PUBLIC
  fzgmod_compile_settings
  CUDA::cudart
)
add_library(FZGMOD::predictors ALIAS fzgmod_predictors)

add_library(fzgmod_pipeline
  src/pipeline/compressor.cpp
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

add_executable(fzgmod-bin utils/main.cpp)
target_link_libraries(fzgmod-bin PRIVATE fzgmod)
set_target_properties(fzgmod-bin PROPERTIES OUTPUT_NAME fzgmod)

# ── Profiling target (CLDHGH dataset, marked with cudaProfilerStart/Stop) ─────
# Build:  cmake --build build --target fzgmod-profile
# Run:    nsys profile --trace=cuda --capture-range=cudaProfilerApi \
#              -o cldhgh_profile ./build/fzgmod-profile
add_executable(fzgmod-profile utils/target.cpp)
target_link_libraries(fzgmod-profile PRIVATE fzgmod)

# Locate nvtx3/nvtx3.hpp — ships with Nsight Systems but the path differs
# across machines (standalone install vs. bundled inside CUDA toolkit).
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
    # fzgmod_pipeline uses nvtx3 internally (decompressFromFile compute range)
    target_include_directories(fzgmod_pipeline PRIVATE ${NVTX3_INCLUDE_DIR})
    target_include_directories(fzgmod-profile PRIVATE ${NVTX3_INCLUDE_DIR})
else()
    message(WARNING "nvtx3/nvtx3.hpp not found — fzgmod-profile may not compile.\n"
                    "Override with: -DNVTX3_INCLUDE_DIR=/path/to/nvtx/include")
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

install(TARGETS fzgmod-bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

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
