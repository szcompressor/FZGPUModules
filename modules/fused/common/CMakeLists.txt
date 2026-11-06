target_sources(fzgmod_modules PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/nvrtc_jit.cpp
)

# Embed JIT headers and propagate driver linkage; see docs/codebase_notes.md CN-BUILD-2.
if(NOT FZGMOD_BACKEND STREQUAL "HIP")
  target_link_libraries(fzgmod_modules PRIVATE CUDA::nvrtc PUBLIC CUDA::cuda_driver)
  set(FZGMOD_NVRTC_HEADERS "${PROJECT_BINARY_DIR}/generated/fz_nvrtc_headers.h")
  # Unchanged output preserves its timestamp.
  add_custom_target(fzgmod_nvrtc_headers ALL
    COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/scripts/gen_nvrtc_headers.py"
            --out "${FZGMOD_NVRTC_HEADERS}"
    BYPRODUCTS "${FZGMOD_NVRTC_HEADERS}"
    COMMENT "Embedding NVRTC device headers"
    VERBATIM
  )
  add_dependencies(fzgmod_modules fzgmod_nvrtc_headers)
  target_include_directories(fzgmod_modules PRIVATE "${PROJECT_BINARY_DIR}/generated")
endif()

# Pipeline specialization JIT-compiles its fused kernels with NVRTC and launches them through the CUDA driver API, and its
# harness assumes 32-lane warps: none of it is built on HIP, where the registry offers no fused implementation
# (src/pipeline/fusion_registry.cpp) and every stage runs staged.
if(FZGMOD_BACKEND STREQUAL "HIP")
  set_source_files_properties(
    ${CMAKE_CURRENT_LIST_DIR}/../fused_block/fused_block.cu
    ${CMAKE_CURRENT_LIST_DIR}/../fused_block/nvrtc_warp_fusion.cu
    ${CMAKE_CURRENT_LIST_DIR}/../chunk_fusion/chunk_fusion.cu
    ${CMAKE_CURRENT_LIST_DIR}/../chunk_fusion/nvrtc_chunk_fusion.cpp
    ${CMAKE_CURRENT_LIST_DIR}/nvrtc_jit.cpp
    TARGET_DIRECTORY fzgmod_modules
    PROPERTIES HEADER_FILE_ONLY TRUE
  )
endif()
