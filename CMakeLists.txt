cmake_minimum_required(VERSION 3.18)
project(FZModules CUDA CXX)

enable_language(CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/modules)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# CUDA files
add_library(stages STATIC
  modules/predictors/lorenzo/lorenzo_c.cu
  modules/encoding/run_length_encoding/RLE.cu
  modules/encoding/bitpacking/bitpacking.cu
  modules/encoding/concatenation/concat.cu
  modules/encoding/difference/diff_encode.cu
)
# TODO: Add these back after updating to new Stage interface:
#  modules/encoding/outlier/outlier.cu

# C++ files  
add_library(pipeline STATIC
  src/pipeline/dag.cpp
  src/pipeline/compressor.cpp
  src/stage/stage.cpp
  src/mem/memory_pool.cpp
)
target_include_directories(pipeline PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Executables
add_executable(compress_test utils/main.cpp)
target_link_libraries(compress_test pipeline stages cudart)

add_executable(pipeline_example utils/pipeline_example.cpp)
target_link_libraries(pipeline_example pipeline stages cudart)

add_executable(outlier_test utils/outlier_test.cpp)
target_link_libraries(outlier_test pipeline stages cudart)
