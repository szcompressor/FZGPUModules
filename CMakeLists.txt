cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(FZModules LANGUAGES CUDA CXX C VERSION 1.0)

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

add_library(fzmod_compile_settings INTERFACE)
target_compile_options(
  fzmod_compile_settings 
  INTERFACE 
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>
)

if(NOT USE_SANITIZER STREQUAL "Compute")
  target_compile_options(
    fzmod_compile_settings
    INTERFACE
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-lineinfo>
  )
endif()

target_compile_features(fzmod_compile_settings INTERFACE cxx_std_17 cuda_std_17)
target_link_libraries(fzmod_compile_settings INTERFACE CUDA::cudart)

###########################
### INCLUDE DIRECTORIES ###
###########################

target_include_directories(fzmod_compile_settings INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fzmod>
)

#################
### LIBRARIES ###
#################

#### Memory Management
add_library(fzmod_mem
  src/mem/mempool.cpp
)
target_link_libraries(fzmod_mem
  PUBLIC
  fzmod_compile_settings
)

#### Encoders Module
add_library(fzmod_encoders
  modules/encoders/bitpacking/bitpacking.cu
  modules/encoders/diff/diff.cu
  modules/encoders/RLE/rle.cu
)
target_link_libraries(fzmod_encoders
  PUBLIC
  fzmod_compile_settings
  CUDA::cudart
)
add_library(FZMOD::encoders ALIAS fzmod_encoders)

#### Predictors Module
add_library(fzmod_predictors
  modules/predictors/lorenzo/lorenzo.cu
)
target_link_libraries(fzmod_predictors
  PUBLIC
  fzmod_compile_settings
  CUDA::cudart
)
add_library(FZMOD::predictors ALIAS fzmod_predictors)

add_library(fzmod_pipeline
  src/pipeline/compressor.cpp
  src/pipeline/dag.cpp
  src/pipeline/perf.cpp
  src/pipeline/stat.cpp
  src/stage/mock_stages.cu
)
target_link_libraries(fzmod_pipeline
  PUBLIC
  fzmod_compile_settings
  fzmod_mem
  fzmod_encoders
  fzmod_predictors
  CUDA::cudart
)
add_library(FZMOD::pipeline ALIAS fzmod_pipeline)

#### Combined Library
add_library(fzmod
  utils/cli/cli.cpp
)
target_link_libraries(fzmod
  PUBLIC
  fzmod_compile_settings
  fzmod_mem
  fzmod_encoders
  fzmod_predictors
  fzmod_pipeline
  CUDA::cudart
)
add_library(FZMOD::fzmod ALIAS fzmod)

###################
### EXECUTABLES ###
###################

add_executable(fzmod-bin utils/main.cpp)
target_link_libraries(fzmod-bin PRIVATE fzmod)
set_target_properties(fzmod-bin PROPERTIES OUTPUT_NAME fzmod)

# ── Profiling target (CLDHGH dataset, marked with cudaProfilerStart/Stop) ─────
# Build:  cmake --build build --target fzmod-profile
# Run:    nsys profile --trace=cuda --capture-range=cudaProfilerApi \
#              -o cldhgh_profile ./build/fzmod-profile
add_executable(fzmod-profile utils/target.cpp)
target_link_libraries(fzmod-profile PRIVATE fzmod)

# TODO: remove or make more generic
# nvtx3.hpp ships with Nsight Systems — add its include path so target.cpp can use it
target_include_directories(fzmod-profile PRIVATE
    /opt/nvidia/nsight-systems/2025.1.3/target-linux-x64/nvtx/include
)

####################
### INSTALLATION ###
####################

install(TARGETS fzmod_compile_settings EXPORT FZModulesTargets)
install(TARGETS 
  fzmod_mem
  fzmod_encoders
  fzmod_predictors
  fzmod_pipeline
  fzmod 
  EXPORT FZModulesTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS fzmod-bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
  EXPORT FZModulesTargets
  NAMESPACE FZMOD::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FZModules
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FZModulesConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/FZModulesConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FZModules
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/FZModulesConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion
)

install(FILES 
  "${CMAKE_CURRENT_BINARY_DIR}/FZModulesConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/FZModulesConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FZModules
)

# Export build-tree targets so find_package(FZModules) works without installing
# by pointing FZModules_DIR at the build directory.
export(
  EXPORT FZModulesTargets
  NAMESPACE FZMOD::
  FILE "${CMAKE_CURRENT_BINARY_DIR}/FZModulesTargets.cmake"
)

install(DIRECTORY
  include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fzmod
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hh"
)

# modules/ headers are included with relative paths (e.g. "encoders/diff/diff.h")
# so they must be installed into the same fzmod include root
install(DIRECTORY
  modules/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fzmod
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hh"
)

# src/ headers are also an include root at build time (e.g. "mem/mempool.h")
install(DIRECTORY
  src/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fzmod
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hh"
)

message(STATUS "FZModules configuration complete")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
