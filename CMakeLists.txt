cmake_minimum_required(VERSION 3.24 FATAL_ERROR)


set(FZGMOD_BACKEND "CUDA" CACHE STRING "GPU backend (stable main supports CUDA)")
set_property(CACHE FZGMOD_BACKEND PROPERTY STRINGS CUDA)
if(NOT FZGMOD_BACKEND STREQUAL "CUDA")
  message(FATAL_ERROR "Stable main supports CUDA only; backend development is on the development branch.")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures to compile for")
endif()
project(FZGPUModules LANGUAGES CUDA CXX VERSION 2.0.1)

# General settings

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "prefer shared libraries" ON)
include(GNUInstallDirs)

# RPATH: installed binaries find the installed shared libraries without LD_LIBRARY_PATH.
if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
else()
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)  # use build-tree paths during build
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) # append any extra link dirs automatically

# FZM file format uses little-endian byte order
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

# CUDA toolchain
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS "11.2")
  message(FATAL_ERROR "FZGPUModules requires CUDA Toolkit 11.2 or newer for stream-ordered allocation.")
endif()
set(FZGMOD_GPU_RUNTIME CUDA::cudart)

# Build-time header generators (stage fingerprints, embedded NVRTC headers).
find_package(Python3 3.7 REQUIRED COMPONENTS Interpreter)

# Retain the module-local source hook; CUDA needs no retagging.
function(fzgmod_set_gpu_source_language target)
endfunction()

# WSL detection

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

# Build configurations

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

option(BUILD_TESTING "Build tests" OFF)


option(BUILD_EXAMPLES "Build example executables" OFF)

# Sanitizers

if(NOT USE_SANITIZER)
  set(USE_SANITIZER "" CACHE STRING "use a build sanitizer" FORCE)
endif()
set_property(CACHE USE_SANITIZER PROPERTY STRINGS "" "ASanUbsan" "TSan" "Compute")
option(COMPUTE_SANITIZER_DEVICE_DEBUG "Build CUDA with -G for Compute Sanitizer (slower, deeper debug)" OFF)

if (USE_SANITIZER STREQUAL "ASanUbsan")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    message(FATAL_ERROR
      "ASan/UBSan is not supported by nvc++ (NVHPC). "
      "Configure with CC=gcc CXX=g++ to use a GCC-based sanitizer build: "
      "  CC=gcc CXX=g++ cmake --preset asan")
  endif()
  message(STATUS "Compiling with Address+Undefined Sanitizers")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address,undefined -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined -fno-omit-frame-pointer")
  if(FZGMOD_BACKEND STREQUAL "CUDA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fsanitize=address -Xcompiler=-fsanitize=undefined -Xcompiler=-fno-omit-frame-pointer")
  endif()
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address,undefined")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address,undefined")
elseif (USE_SANITIZER STREQUAL "TSan")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    message(FATAL_ERROR
      "TSan is not supported by nvc++ (NVHPC). "
      "Configure with CC=gcc CXX=g++ to use a GCC-based sanitizer build.")
  endif()
  message(STATUS "Compiling with Thread Sanitizer")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -fno-omit-frame-pointer")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=thread")
elseif (USE_SANITIZER STREQUAL "Compute")
  if(NOT FZGMOD_BACKEND STREQUAL "CUDA")
    message(FATAL_ERROR
      "USE_SANITIZER=Compute requires the CUDA backend (compute-sanitizer is an "
      "NVIDIA tool).")
  endif()
  message(STATUS "Configuring for Compute Sanitizer")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xcompiler -rdynamic")
  if(COMPUTE_SANITIZER_DEVICE_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
  endif()
endif()

# Compiler settings

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
    # Suppress toml++ warnings about side-effects in __builtin_assume
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wno-assume-expression-discarded>
)

target_compile_features(fzgmod_compile_settings INTERFACE cxx_std_17)
target_compile_features(fzgmod_compile_settings INTERFACE cuda_std_17)

target_link_libraries(fzgmod_compile_settings INTERFACE ${FZGMOD_GPU_RUNTIME})

# Select the backend in the shared runtime headers.
target_compile_definitions(fzgmod_compile_settings INTERFACE
    FZGMOD_BACKEND_${FZGMOD_BACKEND})

# Logging: calls below the configured threshold are compiled out.
set(FZ_LOG_MIN_LEVEL "2" CACHE STRING
    "Minimum log level compiled in: 0=TRACE 1=DEBUG 2=INFO 3=WARN 255=SILENT")
set_property(CACHE FZ_LOG_MIN_LEVEL PROPERTY STRINGS "0" "1" "2" "3" "255")
target_compile_definitions(fzgmod_compile_settings INTERFACE
    FZ_LOG_MIN_LEVEL=${FZ_LOG_MIN_LEVEL})

# Include directories

target_include_directories(fzgmod_compile_settings INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/tomlplusplus/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fzgmod>
)

option(ENABLE_NVTX "Enable NVTX annotations in production library code" OFF)
if(ENABLE_NVTX)
  find_path(NVTX3_INCLUDE_DIR NAMES nvtx3/nvtx3.hpp
    HINTS ${CUDAToolkit_INCLUDE_DIRS} REQUIRED)
  target_compile_definitions(fzgmod_compile_settings INTERFACE
    $<BUILD_INTERFACE:FZ_PROFILING_ENABLED>)
  target_include_directories(fzgmod_compile_settings SYSTEM INTERFACE
    $<BUILD_INTERFACE:${NVTX3_INCLUDE_DIR}>)
  target_compile_options(fzgmod_compile_settings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-fno-omit-frame-pointer>)
endif()

# Libraries

# Memory management
add_library(fzgmod_mem
  src/mem/mempool.cpp
)
target_link_libraries(fzgmod_mem
  PUBLIC
  fzgmod_compile_settings
)

# Internal utilities
add_library(fzgmod_utils STATIC)
set_target_properties(fzgmod_utils PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(fzgmod_utils
  PUBLIC
    fzgmod_compile_settings
)
add_library(FZGMOD::utils ALIAS fzgmod_utils)

# Compression modules
add_library(fzgmod_modules modules/stage_registry.cpp)

target_link_libraries(fzgmod_modules
  PUBLIC
    fzgmod_compile_settings
    ${FZGMOD_GPU_RUNTIME}
  PRIVATE
    fzgmod_utils
)
add_subdirectory(modules)
fzgmod_set_gpu_source_language(fzgmod_utils)
fzgmod_set_gpu_source_language(fzgmod_modules)

add_library(FZGMOD::modules ALIAS fzgmod_modules)

# Refresh fingerprints every build without rewriting unchanged output.
# Details: docs/codebase_notes.md CN-BUILD-3.
set(FZGMOD_STAGE_FP_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/fz_stage_fingerprints.h")
add_custom_target(fzgmod_stage_fingerprints ALL
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_stage_fingerprints.py"
          --out "${FZGMOD_STAGE_FP_HEADER}"
  BYPRODUCTS "${FZGMOD_STAGE_FP_HEADER}"
  COMMENT "Generating per-stage source fingerprints"
  VERBATIM
)

add_library(fzgmod_pipeline
  src/pipeline/compressor.cpp
  src/pipeline/compressor_exec.cpp
  src/pipeline/compressor_inverse_dag.cpp
  src/pipeline/compressor_io.cpp
  src/pipeline/config.cpp
  src/pipeline/concat_kernel.cu
  src/pipeline/device_buffer.cpp
  src/pipeline/dag.cpp
  src/pipeline/dag_coloring.cpp
  src/pipeline/dag_query.cpp
  src/pipeline/fusion_planner.cpp
  src/pipeline/fusion_registry.cpp
  src/pipeline/perf.cpp
  src/pipeline/stat.cpp
)
target_link_libraries(fzgmod_pipeline
  PUBLIC
  fzgmod_compile_settings
  fzgmod_mem
  fzgmod_modules
  ${FZGMOD_GPU_RUNTIME}
)
fzgmod_set_gpu_source_language(fzgmod_pipeline)
add_dependencies(fzgmod_pipeline fzgmod_stage_fingerprints)
target_include_directories(fzgmod_pipeline PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
add_library(FZGMOD::pipeline ALIAS fzgmod_pipeline)

# Combined library
add_library(fzgmod
  src/utils/cli/cli.cpp
  src/utils/cli/report_json.cpp
)
target_link_libraries(fzgmod
  PUBLIC
  fzgmod_compile_settings
  fzgmod_mem
  fzgmod_modules
  fzgmod_pipeline
  ${FZGMOD_GPU_RUNTIME}
)

# Tool identity for the --report-json artifact.
execute_process(
  COMMAND git rev-parse --short HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE FZGMOD_GIT_SHA
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
target_compile_definitions(fzgmod PRIVATE FZGMOD_VERSION="${PROJECT_VERSION}")
if(FZGMOD_GIT_SHA)
  target_compile_definitions(fzgmod PRIVATE FZGMOD_GIT_SHA="${FZGMOD_GIT_SHA}")
endif()
add_library(FZGMOD::fzgmod ALIAS fzgmod)

# Use the major version for ABI compatibility and the full release for VERSION.
foreach(_lib fzgmod fzgmod_mem fzgmod_modules fzgmod_pipeline)
  set_target_properties(${_lib} PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
  )
endforeach()

# Executables

option(BUILD_CLI "Build command-line executable" ON)

if(BUILD_CLI)
  add_executable(fzgmod-cli src/utils/cli/main.cpp)
  target_link_libraries(fzgmod-cli PRIVATE fzgmod)
  set_target_properties(fzgmod-cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

# Tests inherit the configured compiler and sanitizer flags.

if(BUILD_TESTING)
    add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
    enable_testing()
    add_subdirectory(tests)

    # Run the golden-file generator only when explicitly requested.
    add_executable(generate_golden_files EXCLUDE_FROM_ALL
        tests/golden/regenerate.cpp
    )
    target_link_libraries(generate_golden_files PRIVATE fzgmod)
    target_include_directories(generate_golden_files PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    add_custom_command(TARGET generate_golden_files POST_BUILD
      COMMAND $<TARGET_FILE:generate_golden_files>
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/golden"
        COMMENT "Generating golden reference FZM files in tests/golden/"
        VERBATIM
    )
endif()

# Examples

add_subdirectory(examples)

# Installation

install(TARGETS fzgmod_compile_settings EXPORT FZGPUModulesTargets)
install(TARGETS
  fzgmod_mem
  fzgmod_utils
  fzgmod_modules
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
  COMPATIBILITY SameMajorVersion
)

install(FILES 
  "${CMAKE_CURRENT_BINARY_DIR}/FZGPUModulesConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/FZGPUModulesConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FZGPUModules
)

# Include project and third-party notices in installed packages.
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
  "${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY.md"
  DESTINATION ${CMAKE_INSTALL_DOCDIR}
)

# Export targets for find_package() consumers using the build directory.
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


install(DIRECTORY
  modules/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fzgmod
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hh"
)

install(DIRECTORY
  src/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fzgmod
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hh"
)

# Documentation (cmake --build <builddir> --target docs)
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

# Wrap up

message(STATUS "FZGPUModules configuration complete")
message(STATUS "  Backend: ${FZGMOD_BACKEND}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
