target_sources(fzgmod_modules PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/ans_stage.cu
  ${CMAKE_CURRENT_LIST_DIR}/dietgpu/GpuANS.cu
)

target_include_directories(fzgmod_modules PRIVATE ${CMAKE_CURRENT_LIST_DIR}/dietgpu)

# Exclude CUDA-only ANS on HIP; addStage() rejects unsupported stages.
if(FZGMOD_BACKEND STREQUAL "HIP")
  set_source_files_properties(
    ${CMAKE_CURRENT_LIST_DIR}/ans_stage.cu
    ${CMAKE_CURRENT_LIST_DIR}/dietgpu/GpuANS.cu
    TARGET_DIRECTORY fzgmod_modules
    PROPERTIES HEADER_FILE_ONLY TRUE
  )
endif()
