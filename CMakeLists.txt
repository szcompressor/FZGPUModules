cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(FZGPUModules LANGUAGES CUDA CXX VERSION 2.0.0)

########################
### GENERAL SETTINGS ###
########################

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "prefer shared libraries" ON)
include(GNUInstallDirs)

# RPATH: installed binaries find the installed shared libraries without LD_LIBRARY_PATH.
# $ORIGIN-relative so the install tree is relocatable (works with Spack, Docker, etc.)
if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
else()
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)  # use build-tree paths during build
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) # append any extra link dirs automatically

# FZM file format uses little-endian byte order throughout — reject big-endian hosts.
include(TestBigEndian)
test_big_endian(FZGMOD_IS_BIG_ENDIAN)
if(FZGMOD_IS_BIG_ENDIAN)
  message(FATAL_ERROR
    "FZGPUModules requires a little-endian host.\n"
    "  The FZM file format stores all multi-byte integers in little-endian order.\n"
    "  Big-endian support is not implemented.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS "11.2")
  message(FATAL_ERROR
    "FZGPUModules requires CUDA Toolkit 11.2 or newer.\n"
    "  Reason: stream-ordered allocators (cudaMallocAsync / cudaMemPool_t) require 11.2+.\n"
    "  Detected: ${CUDAToolkit_VERSION}")
endif()

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

    # ── generate_golden_files target ─────────────────────────────────────────
    # Builds and runs the golden-file generator so that tests/golden/ contains
    # fresh reference files matching the current FZM format version.
    # EXCLUDE_FROM_ALL keeps cluster/head-node default builds GPU-free.
    # Usage: cmake --build <build-dir> --target generate_golden_files
    add_executable(generate_golden_files EXCLUDE_FROM_ALL
        tests/golden/regenerate.cpp
    )
    target_link_libraries(generate_golden_files PRIVATE fzgmod)
    target_include_directories(generate_golden_files PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    # After the binary is built, immediately run it to write the reference files.
    add_custom_command(TARGET generate_golden_files POST_BUILD
      COMMAND $<TARGET_FILE:generate_golden_files>
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/golden"
        COMMENT "Generating golden reference FZM files in tests/golden/"
        VERBATIM
    )
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

# ── Logging level ─────────────────────────────────────────────────────────────
# Controls which FZ_LOG() call sites are compiled into the binary.
# Calls below the threshold are compiled out entirely — arguments are never
# evaluated, giving zero overhead even for expensive log expressions.
#
#   0 = TRACE  — everything (use for deep debugging)
#   1 = DEBUG  — construction, allocation, data stats
#   2 = INFO   — high-level milestones only  ← default
#   3 = WARN   — warnings only
#  255 = SILENT — all logging compiled out (embedded / production)
#
# Override: cmake -DFZ_LOG_MIN_LEVEL=0 ..
set(FZ_LOG_MIN_LEVEL "2" CACHE STRING
    "Minimum log level compiled in: 0=TRACE 1=DEBUG 2=INFO 3=WARN 255=SILENT")
set_property(CACHE FZ_LOG_MIN_LEVEL PROPERTY STRINGS "0" "1" "2" "3" "255")
target_compile_definitions(fzgmod_compile_settings INTERFACE
    FZ_LOG_MIN_LEVEL=${FZ_LOG_MIN_LEVEL})

###########################
### INCLUDE DIRECTORIES ###
###########################

target_include_directories(fzgmod_compile_settings INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/tomlplusplus/include>
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
  modules/transforms/bitpack/bitpack_stage.cu
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
  modules/predictors/lorenzo/lorenzo_nd.cu
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
  src/pipeline/config.cpp
  src/pipeline/concat_kernel.cu
  src/pipeline/dag.cpp
  src/pipeline/dag_coloring.cpp
  src/pipeline/dag_query.cpp
  src/pipeline/perf.cpp
  src/pipeline/stat.cpp
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
  src/utils/cli/cli.cpp
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

# Shared library versioning — creates libfzgmod.so.2 → libfzgmod.so.2.0.0
# SOVERSION (ABI version) = major only; bump on any breaking API/ABI change.
# VERSION   (full version) = PROJECT_VERSION, updated every release.
foreach(_lib fzgmod fzgmod_mem fzgmod_encoders fzgmod_predictors fzgmod_pipeline)
  set_target_properties(${_lib} PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
  )
endforeach()

###################
### EXECUTABLES ###
###################

option(BUILD_CLI "Build command-line executable" ON)

if(BUILD_CLI)
  add_executable(fzgmod-cli src/utils/cli/main.cpp)
  target_link_libraries(fzgmod-cli PRIVATE fzgmod)
  set_target_properties(fzgmod-cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

##########################
### Examples/Profiling ###
##########################

add_subdirectory(examples)
add_subdirectory(profiling)

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


if(BUILD_CLI)
  install(TARGETS fzgmod-cli
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

#---------------------------------------------------------------------------
# Documentation (Doxygen)  — build with: cmake --build <builddir> --target docs
#---------------------------------------------------------------------------
find_package(Doxygen OPTIONAL_COMPONENTS dot)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen ${DOXYGEN_VERSION}"
        VERBATIM
    )
    message(STATUS "Doxygen ${DOXYGEN_VERSION} found — 'docs' target available")
    if(Doxygen_dot_FOUND)
        message(STATUS "  Graphviz dot found — class diagrams enabled")
    else()
        message(STATUS "  Graphviz dot NOT found — set HAVE_DOT=NO in Doxyfile or install graphviz")
    endif()
else()
    message(STATUS "Doxygen not found — 'docs' target not available (install doxygen to enable)")
endif()

message(STATUS "FZGPUModules configuration complete")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build profiling: ${BUILD_PROFILING}")
