cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# GPU backend selection. Exactly one backend is compiled into any given build
# (a compile-time choice, not a runtime switch) — see include/backend/types.h
# for the corresponding C++-side type aliases. CUDA is supported, HIP is
# experimental, and SYCL is planned.
set(FZGMOD_BACKEND "CUDA" CACHE STRING "GPU backend to build against: CUDA, HIP, or SYCL")
set_property(CACHE FZGMOD_BACKEND PROPERTY STRINGS CUDA HIP SYCL)

# The GPU language belongs in project()'s language list, so the backend has to be
# resolved before project() runs. HIP is the exception: its language is enabled
# after project() instead, because CMAKE_HIP_ARCHITECTURES has to be set first
# (HIP's compiler probe otherwise tries to autodetect the installed GPU, which
# fails on a GPU-less login/build node).
if(FZGMOD_BACKEND STREQUAL "CUDA")
  set(FZGMOD_PROJECT_LANGUAGES CUDA CXX)
  # Must precede project(): CMake's own CUDA architecture autodetection runs
  # during project()'s enable_language(CUDA) and, on a GPU-less build host
  # (e.g. a Docker builder with no GPU passed through), falls back to the
  # oldest architecture it knows (sm_52) rather than our intended default —
  # sm_52 predates block-scoped atomics (atomicAdd_block/atomicOr_block,
  # compute capability 6.0+) used by the RARE/RAZE stages, so that fallback
  # fails to compile.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures to compile for")
  endif()
elseif(FZGMOD_BACKEND STREQUAL "HIP")
  set(FZGMOD_PROJECT_LANGUAGES CXX)
  message(WARNING
    "FZGMOD_BACKEND=HIP is experimental: backend coverage and install/consumer "
    "compatibility are still being completed, and some stages are unavailable.")
elseif(FZGMOD_BACKEND STREQUAL "SYCL")
  message(FATAL_ERROR "FZGMOD_BACKEND=SYCL is not implemented yet.")
else()
  message(FATAL_ERROR "Unknown FZGMOD_BACKEND '${FZGMOD_BACKEND}' — must be CUDA, HIP, or SYCL.")
endif()

project(FZGPUModules LANGUAGES ${FZGMOD_PROJECT_LANGUAGES} VERSION 2.0.0)

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

###############################
### GPU BACKEND TOOLCHAIN   ###
###############################
#
# FZGMOD_GPU_RUNTIME holds the imported target(s) carrying the GPU runtime's
# headers, defines, and link libraries for the selected backend. Everything
# downstream links this rather than naming CUDA::cudart directly.

if(FZGMOD_BACKEND STREQUAL "CUDA")
  find_package(CUDAToolkit REQUIRED)
  if(CUDAToolkit_VERSION VERSION_LESS "11.2")
    message(FATAL_ERROR
      "FZGPUModules requires CUDA Toolkit 11.2 or newer.\n"
      "  Reason: stream-ordered allocators (cudaMallocAsync / cudaMemPool_t) require 11.2+.\n"
      "  Detected: ${CUDAToolkit_VERSION}")
  endif()

  # CMAKE_CUDA_ARCHITECTURES default is set above, before project(), which is
  # where CUDA actually gets enabled (see the note there).
  enable_language(CUDA)

  set(FZGMOD_GPU_RUNTIME CUDA::cudart)

else() # HIP — SYCL and unknown values already rejected above

  # Must precede enable_language(HIP); see the note by the backend selection.
  # gfx908 = MI100 (CDNA1). Override for other targets, e.g.
  #   cmake --preset hip -DCMAKE_HIP_ARCHITECTURES=gfx90a  (MI200 / CDNA2)
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx908" CACHE STRING
        "AMD GPU architectures to compile for (gfx908=MI100, gfx90a=MI200, gfx942=MI300)")
  endif()

  enable_language(HIP)

  # Tells hip-config which arch to bake into its own imported targets, and
  # suppresses its "system GPU detection was unsuccessful" warning — autodetect
  # (amdgpu-arch) necessarily fails when configuring on a GPU-less build node.
  set(GPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "AMD GPU targets for hip-config" FORCE)

  # These are CONFIG-mode packages living under the ROCm install root, so
  # CMAKE_PREFIX_PATH (not just PATH) has to include it — e.g.
  #   export CMAKE_PREFIX_PATH=/opt/rocm:$CMAKE_PREFIX_PATH
  find_package(hip       CONFIG REQUIRED)
  find_package(hipcub    CONFIG REQUIRED)  # cub facade — mirrors cub's API
  find_package(rocthrust CONFIG REQUIRED)  # thrust facade, used by ADM's mappings

  # hip::host, not hip::device. hip::device exists for the "compile .cpp as HIP
  # via hipcc" workflow and carries `-x hip` as an INTERFACE compile option,
  # which CMake would then apply to every plain C++ source in the project
  # (g++: "language hip not recognized"). We use enable_language(HIP) instead,
  # so CMake already drives -x hip / --offload-arch for exactly the sources
  # tagged LANGUAGE HIP; hip::host supplies just the runtime headers,
  # __HIP_PLATFORM_AMD__, and libamdhip64.
  set(FZGMOD_GPU_RUNTIME hip::host)

  # hipCUB / rocPRIM / rocThrust are header-only, but their imported targets all
  # link roc::rocprim_hip → hip::device, so linking them would drag `-x hip` back
  # in. Take their include paths directly instead.
  set(FZGMOD_HIP_HEADER_ONLY_INCLUDES "")
  foreach(_hdr_tgt hip::hipcub roc::rocprim roc::rocthrust)
    if(TARGET ${_hdr_tgt})
      get_target_property(_inc ${_hdr_tgt} INTERFACE_INCLUDE_DIRECTORIES)
      if(_inc)
        list(APPEND FZGMOD_HIP_HEADER_ONLY_INCLUDES ${_inc})
      endif()
    endif()
  endforeach()
  list(REMOVE_DUPLICATES FZGMOD_HIP_HEADER_ONLY_INCLUDES)

endif()

# hipcc/amdclang++ compiles the existing .cu sources as-is (verified against
# ROCm 6.4 on gfx908), so the HIP build keeps the current file names rather than
# renaming ~38 files to .hip. CMake still maps the .cu extension to the CUDA
# language by default, though, so each .cu source is retagged LANGUAGE HIP
# explicitly. No-op under the CUDA backend. Defined here (before any
# add_subdirectory() call) so subdirectories such as tests/ can use it too.
function(fzgmod_set_gpu_source_language target)
  if(NOT FZGMOD_BACKEND STREQUAL "HIP")
    return()
  endif()
  get_target_property(_srcs ${target} SOURCES)
  foreach(_src IN LISTS _srcs)
    if(_src MATCHES "\\.cu$")
      set_source_files_properties(${_src}
        TARGET_DIRECTORY ${target}
        PROPERTIES LANGUAGE HIP)
    endif()
  endforeach()
endfunction()

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

option(BUILD_TESTING "Build tests" OFF)
if(BUILD_TESTING)
    add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
    enable_testing()
    add_subdirectory(tests)

    # generate_golden_files target 
    # Builds and runs the golden-file generator
    # EXCLUDE_FROM_ALL keeps cluster/head-node default builds GPU-free
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
  if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    message(FATAL_ERROR
      "ASan/UBSan is not supported by nvc++ (NVHPC). "
      "Configure with CC=gcc CXX=g++ to use a GCC-based sanitizer build: "
      "  CC=gcc CXX=g++ cmake --preset asan")
  endif()
  message(STATUS "Compiling with Address+Undefined Sanitizers")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address,undefined -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined -fno-omit-frame-pointer")
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
      "NVIDIA tool). Sanitizer parity for HIP is not implemented.")
  endif()
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
    # Suppress toml++ warnings about side-effects in __builtin_assume
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wno-assume-expression-discarded>
)

target_compile_features(fzgmod_compile_settings INTERFACE cxx_std_17)
if(FZGMOD_BACKEND STREQUAL "CUDA")
  target_compile_features(fzgmod_compile_settings INTERFACE cuda_std_17)
else()
  set(CMAKE_HIP_STANDARD 17)
  set(CMAKE_HIP_STANDARD_REQUIRED True)

  # HIP declares the runtime API [[nodiscard]]; CUDA does not. Call sites that
  # intentionally drop the status (destructors, teardown paths, and everything
  # wrapped in FZ_CUDA_CHECK_WARN) would otherwise emit hundreds of warnings
  # that are a header-convention difference, not a defect in this code.
  target_compile_options(fzgmod_compile_settings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX,HIP>:-Wno-unused-result>)

  # __shfl_*_sync/__ballot_sync/__any_sync are declared only behind this flag
  # (see include/backend/warp.h) — ROCm gates them because their emulated
  # convergence semantics differ from CUDA's native ones.
  target_compile_definitions(fzgmod_compile_settings INTERFACE
    HIP_ENABLE_WARP_SYNC_BUILTINS)
endif()

target_link_libraries(fzgmod_compile_settings INTERFACE ${FZGMOD_GPU_RUNTIME})

# Selects the backend branch in include/backend/{types,api,algorithms}.h.
# Those headers fall back to CUDA when nothing is defined, so this is what
# actually switches the C++ side over to HIP.
target_compile_definitions(fzgmod_compile_settings INTERFACE
    FZGMOD_BACKEND_${FZGMOD_BACKEND})

# ── Logging level
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
  # Quarantined experimental/reference compressors (not public modules): included
  # so their FZM factories stay linked for archive back-compat. See
  # experimental/reference_compressors/.
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/experimental>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/tomlplusplus/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fzgmod>
)

# hipCUB / rocPRIM / rocThrust headers (HIP backend only; empty under CUDA).
# SYSTEM so their warnings don't surface as ours.
if(FZGMOD_HIP_HEADER_ONLY_INCLUDES)
  target_include_directories(fzgmod_compile_settings SYSTEM INTERFACE
    ${FZGMOD_HIP_HEADER_ONLY_INCLUDES})
endif()

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

#### Internal Utilities (shared helpers not part of the public API)
add_library(fzgmod_utils STATIC
  modules/util/histogram/histogram.cu
)
set_target_properties(fzgmod_utils PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(fzgmod_utils
  PUBLIC
    fzgmod_compile_settings
)
fzgmod_set_gpu_source_language(fzgmod_utils)
add_library(FZGMOD::utils ALIAS fzgmod_utils)

#### Modules (predictors, transforms, quantizers, coders, shufflers, fused)
add_library(fzgmod_modules
  modules/stage_registry.cpp
  modules/fused/fused_block/fused_block.cu
  modules/fused/fused_block/nvrtc_warp_fusion.cu
  modules/fused/chunk_fusion/chunk_fusion.cu
  modules/fused/chunk_fusion/nvrtc_chunk_fusion.cpp
  modules/fused/common/nvrtc_jit.cpp
  modules/predictors/diff/diff.cu
  modules/predictors/lorenzo/lorenzo_stage.cu
  modules/predictors/tiled_lorenzo/tiled_lorenzo_stage.cu
  modules/transforms/zigzag/zigzag_stage.cu
  modules/transforms/log_transform/log_transform_stage.cu
  modules/transforms/cdf97/cdf97_stage.cu
  modules/coders/speck2d/speck2d_stage.cu
  modules/coders/cdf97_outlier_correct/cdf97_outlier_correct_stage.cu
  modules/transforms/negabinary/negabinary_stage.cu
  modules/quantizers/quantizer/quantizer.cu
  modules/coders/rle/rle.cu
  modules/coders/rze/rze_stage.cu
  modules/coders/rre/rre_stage.cu
  modules/coders/rare/rare_stage.cu
  modules/coders/raze/raze_stage.cu
  modules/coders/clog/clog_stage.cu
  modules/coders/hclog/hclog_stage.cu
  modules/structural/merge/merge_stage.cu
  modules/structural/roibin_split/roibin_split_stage.cu
  modules/coders/bitpack/bitpack_stage.cu
  modules/coders/adaptive_bitpack/adaptive_bitpack_kernels.cu
  modules/coders/adaptive_bitpack/adaptive_bitpack_stage.cu
  modules/shufflers/bitshuffle/bitshuffle_stage.cu
  modules/shufflers/tupl/tupl_stage.cu
  modules/fused/adaptive_lorenzo/adaptive_lorenzo_stage.cu
    modules/fused/lorenzo_quant/lorenzo_quant.cu
  modules/fused/lorenzo_quant/lorenzo_quant_nd.cu
  modules/fused/szx/szx_stage.cu
  # Quarantined: monolithic SZp reference compressor. Compiled ONLY so its
  # StageType::SZP FZM header factory stays linked (pre-existing archives must
  # still decode). Not a public module — see experimental/reference_compressors/szp.
  experimental/reference_compressors/szp/szp_stage.cu
  # Huffman (PHF CPU layer + GPU kernels + stage wrapper)
  modules/coders/huffman/phf/hf_buf.cc
  modules/coders/huffman/phf/hf_hl.cc
  modules/coders/huffman/phf/hf_bk.cc
  modules/coders/huffman/phf/hf_bk_impl1.cc
  modules/coders/huffman/phf/hf_bk_internal.cc
  modules/coders/huffman/phf/hf_canon.cc
  modules/coders/huffman/phf/hf_kernels.cu
  modules/coders/huffman/huffman_stage.cu
  modules/coders/ans/ans_stage.cu
  modules/coders/ans/dietgpu/GpuANS.cu
  modules/transforms/adm/adm_stage.cu
  modules/transforms/adm/mapping_uint16.cu
  modules/transforms/adm/mapping_uint32.cu
  # GInterp (cuSZ-Hi spline interpolation predictor + quantizer — fused)
  modules/fused/ginterp/ginterp_kernels.cu
  modules/fused/ginterp/ginterp_stage.cu
  # BitplaneRZE (FZ-GPU fused bitplane transpose + zero-group RZE)
  modules/fused/bitplane_rze/bitplane_rze_kernels.cu
  modules/fused/bitplane_rze/bitplane_rze_stage.cu
  modules/coders/gpulz/gpulz_stage.cu
)

target_link_libraries(fzgmod_modules
  PUBLIC
    fzgmod_compile_settings
    ${FZGMOD_GPU_RUNTIME}
  PRIVATE
    fzgmod_utils
)
target_include_directories(fzgmod_modules
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/coders/ans/dietgpu
)
fzgmod_set_gpu_source_language(fzgmod_modules)

# NVRTC chunk-fusion codegen (CUDA backend only): links NVRTC + the CUDA driver
# API, and needs the source-tree include roots so NVRTC can resolve the op
# headers at runtime (a prototype constraint — see CN-NVRTC-FUSE).
# CUDA::cuda_driver is PUBLIC (not PRIVATE): fzgmod_modules is a shared lib, so its
# driver-API symbols (cuModuleLoadData, cuLaunchKernel, ...) stay undefined in the
# .so and must be resolved on each consumer's link line. Propagating the driver
# (the stub libcuda in a GPU-less build container) lets fzgmod-cli / tests link;
# PRIVATE left them unresolved (undefined reference to cuModuleLoadData). NVRTC is
# a real library present at build time, so it resolves fine as PRIVATE.
if(NOT FZGMOD_BACKEND STREQUAL "HIP")
  target_link_libraries(fzgmod_modules PRIVATE CUDA::nvrtc PUBLIC CUDA::cuda_driver)
  # Only nvrtc_jit.cpp bakes in the source-tree include paths (it invokes NVRTC); the
  # chunk/warp codegen files just emit source strings and launch.
  set_source_files_properties(
    modules/fused/common/nvrtc_jit.cpp
    TARGET_DIRECTORY fzgmod_modules
    PROPERTIES COMPILE_DEFINITIONS
      "FZGMOD_NVRTC_INC_INCLUDE=\"${CMAKE_CURRENT_SOURCE_DIR}/include\";FZGMOD_NVRTC_INC_MODULES=\"${CMAKE_CURRENT_SOURCE_DIR}/modules\"")
endif()

# The vendored dietgpu ANS tree uses inline NVPTX lanemask assembly and a
# hardcoded 32-lane warp model with no HIP translation, so it is excluded from compilation
# entirely on HIP rather than ported. This is a build-level stopgap:
# ANSStage's symbols are simply absent from the HIP library. The intended
# runtime mechanism (Stage::isSupportedOnBackend() + a clear error from
# Pipeline::addStage<ANSStage>() instead of an undefined-symbol link failure)
# is not yet wired up -- TODO before Phase 1 is considered complete.
if(FZGMOD_BACKEND STREQUAL "HIP")
  set_source_files_properties(
    modules/coders/ans/ans_stage.cu
    modules/coders/ans/dietgpu/GpuANS.cu
    TARGET_DIRECTORY fzgmod_modules
    PROPERTIES HEADER_FILE_ONLY TRUE
  )
endif()

add_library(FZGMOD::modules ALIAS fzgmod_modules)

#### Generated per-stage source fingerprints
#
# Hashes each stage's sources plus their transitive local includes, so a consumer
# can tell whether a stage's code changed between two builds (config.cpp's
# kStageRegistry supplies the directories). Runs on every build rather than only at
# configure time: the whole point is to track source edits, and a configure-time-only
# hash would go stale the moment someone edits a kernel and rebuilds. The generator
# rewrites the header only when a fingerprint actually changes, so this does not
# retrigger compilation on every build.
set(FZGMOD_STAGE_FP_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/fz_stage_fingerprints.h")
add_custom_target(fzgmod_stage_fingerprints ALL
  COMMAND ${CMAKE_COMMAND} -E env python3
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

#### Combined Library
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

# Shared library versioning — creates libfzgmod.so.2 → libfzgmod.so.2.0.0
# SOVERSION (ABI version) = major only; bump on any breaking API/ABI change.
# VERSION   (full version) = PROJECT_VERSION, updated every release.
foreach(_lib fzgmod fzgmod_mem fzgmod_modules fzgmod_pipeline)
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

# Ship both the project's own license and the notices required by incorporated
# third-party code.  Keeping THIRD_PARTY.md only in the source tree is
# insufficient for binary/install-tree redistribution.
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
  "${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY.md"
  DESTINATION ${CMAKE_INSTALL_DOCDIR}
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

###############
### WRAP UP ###
###############

message(STATUS "FZGPUModules configuration complete")
message(STATUS "  Backend: ${FZGMOD_BACKEND}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
if(FZGMOD_BACKEND STREQUAL "CUDA")
  message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "  HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
endif()
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build profiling: ${BUILD_PROFILING}")
