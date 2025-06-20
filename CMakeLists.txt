cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(FZModules LANGUAGES CUDA CXX C VERSION 1.0)

########################
### GENERAL SETTINGS ###
########################

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "prefer shared libaries" ON)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(FZMOD_BACKEND "CUDA" CACHE STRING "Select the backend for the FZ Compression Modules library")

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES native)

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
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weffc++")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
  # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
  # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra")
  # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpedantic")
  # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror=all-warnings")
endif()

#############################
### TESTING CONFIGURATION ###
#############################

option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
if(ENABLE_COVERAGE)
  message(STATUS "Compiling with coverage flags")
  add_compile_options(--coverage -O0 -g)
  add_link_options(--coverage)
  add_subdirectory(third_party/googletest)
  add_subdirectory(tests)
endif()

################################
### SANITIZERS CONFIGURATION ###
################################

# Add sanitizers option
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
    # check if ASAN_OPTIONS=protect_shadow_gap=0
    if(DEFINED ENV{ASAN_OPTIONS})
        set(ASAN_OPTIONS "$ENV{ASAN_OPTIONS}:protect_shadow_gap=0")
    else()
        set(ASAN_OPTIONS "protect_shadow_gap=0")
    endif()
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
    # $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Wno-deprecated-declarations>
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
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tools/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tools/api/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/codec/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/stat/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/predictor/>
  # fzg codec module:
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/codec/fzg/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/codec/fzg/src/>
  # histogram module
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/stat/histogram/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/stat/histogram/src/>
  # huffman module
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/codec/huffman/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/codec/huffman/src/>
  # lorenzo module
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/predictor/lrz/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/predictor/lrz/src/>
  # spline module
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/predictor/spline/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/predictor/spline/src/>
  # extrema module
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/stat/extrema/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/stat/extrema/src/>
  # spvn module
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/stat/spvn/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/stat/spvn/src/>
  # misc
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

#################
### LIBRARIES ###
#################

#### FZ utils
add_library(fz_utils
  modules/stat/extrema/src/extrema.cu
  modules/stat/spvn/src/spvn.cu
  modules/stat/extrema/src/calc_err.cu
  modules/stat/extrema/src/max_err.cu
)
target_link_libraries(fz_utils
  PUBLIC
  fzmod_compile_settings
  CUDA::cudart
)
####
#### FZG Codec Module
add_library(fzmod_cu_fzg
  modules/codec/fzg/src/fzg_kernel.cu
  modules/codec/fzg/src/fzg_class.cc
)
target_link_libraries(fzmod_cu_fzg
  PUBLIC
  fzmod_compile_settings
)
add_library(FZMOD::CUDA::fzg ALIAS fzmod_cu_fzg)
add_library(FZMOD::fzg ALIAS fzmod_cu_fzg)
####
#### Histogram Module
add_library(fzmod_cu_histogram
  modules/stat/histogram/src/hist_generic_seq.cc
  modules/stat/histogram/src/hist_generic.cu
  modules/stat/histogram/src/histsp.seq.cc
  modules/stat/histogram/src/histsp.cu
)
target_link_libraries(fzmod_cu_histogram
  PUBLIC
  fzmod_compile_settings
  CUDA::cudart
)
add_library(FZMOD::CUDA::histogram ALIAS fzmod_cu_histogram)
add_library(FZMOD::histogram ALIAS fzmod_cu_histogram)
####
#### Huffman Module
add_library(fzmod_cu_huffman
  modules/codec/huffman/src/hf_est.cc
  modules/codec/huffman/src/hf_bk_impl1.seq.cc
  modules/codec/huffman/src/hf_bk_impl2.seq.cc
  modules/codec/huffman/src/hf_bk_internal.seq.cc
  modules/codec/huffman/src/hf_bk.seq.cc
  modules/codec/huffman/src/hf_canon.seq.cc
  modules/codec/huffman/src/hf_kernels.cu
  modules/codec/huffman/src/hf_ood.cc
  modules/codec/huffman/src/hf_hl.cc
  modules/codec/huffman/src/hf_buf.cc
  modules/codec/huffman/src/libphf.cc
)
target_link_libraries(fzmod_cu_huffman
  PUBLIC
  fzmod_compile_settings
  CUDA::cuda_driver
)
add_library(FZMOD::CUDA::huffman ALIAS fzmod_cu_huffman)
add_library(FZMOD::huffman ALIAS fzmod_cu_huffman)
####
#### Lorenzo Module
add_library(fzmod_cu_lrz
  modules/predictor/lrz/src/lrz_c.cu
  modules/predictor/lrz/src/lrz_x.cu
)
target_link_libraries(fzmod_cu_lrz
  PUBLIC
  fzmod_compile_settings
  CUDA::cudart  
)
add_library(FZMOD::CUDA::lrz ALIAS fzmod_cu_lrz)
add_library(FZMOD::lrz ALIAS fzmod_cu_lrz)
####
#### Spline Module
add_library(fzmod_cu_spline
  modules/predictor/spline/src/spline3.cu
)
target_link_libraries(fzmod_cu_spline
  PUBLIC
  fzmod_compile_settings
  CUDA::cudart
)
add_library(FZMOD::CUDA::spline ALIAS fzmod_cu_spline)
add_library(FZMOD::spline ALIAS fzmod_cu_spline)
####
#### Combine Modules
add_library(fzmod
  examples/demo.cc
)
target_link_libraries(fzmod
  PUBLIC
  fzmod_compile_settings
  fz_utils
  fzmod_cu_fzg
  fzmod_cu_histogram
  fzmod_cu_huffman
  fzmod_cu_lrz
  fzmod_cu_spline
  CUDA::cudart
)
add_library(FZMOD::CUDA ALIAS fzmod)
add_library(FZMOD::all ALIAS fzmod)
####

###################
### EXECUTABLES ###
###################

add_executable(fzmod-bin-example examples/demo.cc)
set_source_files_properties(examples/demo.cc PROPERTIES LANGUAGE CUDA)
target_link_libraries(fzmod-bin-example PRIVATE fzmod)
set_target_properties(fzmod-bin-example PROPERTIES OUTPUT_NAME fzmod_demo)

add_executable(fzmod-bin tools/cli/cli.cc)
set_source_files_properties(tools/cli/cli.cc PROPERTIES LANGUAGE CUDA)
target_link_libraries(fzmod-bin PRIVATE fzmod)
set_target_properties(fzmod-bin PROPERTIES OUTPUT_NAME fzmod)

#############################
### CUDASTF CONFIGURATION ###
#############################

option(BUILD_CUDASTF "Build the CUDASTF API" OFF)
if(BUILD_CUDASTF)
  # Add CPM.cmake for fetching CCCL
  include(cmake/CPM.cmake)

  # Add CCCL package for CUDASTF support
  CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY "nvidia/cccl"
    GIT_TAG "main"
    OPTIONS "CCCL_ENABLE_UNSTABLE ON"
  )

  # CUDASTF demo executable
  add_executable(cudastf_demo cudastf/cudastf_demo.cc)
  set_source_files_properties(cudastf/cudastf_demo.cc PROPERTIES LANGUAGE CUDA)
  target_link_libraries(cudastf_demo 
    PRIVATE 
    fzmod_compile_settings
    fz_utils
    fzmod_cu_huffman
    CUDA::cudart
    CUDA::cuda_driver
    CCCL::CCCL 
    CCCL::cudax
  )

  # testing example
  add_executable(testing_run cudastf/testing.cc)
  set_source_files_properties(cudastf/testing.cc PROPERTIES LANGUAGE CUDA)
  target_link_libraries(testing_run
    PRIVATE 
    fzmod_compile_settings
    fz_utils
    fzmod_cu_huffman
    CUDA::cudart
    CUDA::cuda_driver
    CCCL::CCCL 
    CCCL::cudax 
  )

  # install executable
  install(TARGETS cudastf_demo
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

  install(TARGETS testing_run
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()

####################
### INSTALLATION ###
####################

install(TARGETS fzmod_compile_settings EXPORT FZModulesTargets)
install(TARGETS 
  fz_utils
  fzmod_cu_fzg 
  fzmod_cu_histogram 
  fzmod_cu_huffman 
  fzmod_cu_lrz 
  fzmod_cu_spline 
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
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FZModules)

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

install(DIRECTORY
  tools/include/
  modules/codec/fzg/include/
  modules/stat/histogram/include/
  modules/codec/huffman/include/
  modules/predictor/lrz/include/
  modules/predictor/spline/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fzmod
)

# Print Build Details
include(cmake/probe.cmake)